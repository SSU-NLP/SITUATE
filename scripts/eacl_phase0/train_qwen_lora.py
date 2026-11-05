"""Joint Qwen LoRA and FactoredSlot training with document-level gradient caching."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import train_anchor_m4m5 as T
from run_decoder_multiseed import aligned_sentences
from build_anchor_dataset import collect_docs
from eval_all_models_alldocs import metrics


def dump(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


class TextCorpus:
    def __init__(self, source, split, docs):
        ix = np.load(source / f'{split}_index.npz', allow_pickle=True)
        self.doc_ids = [str(x) for x in ix['doc_ids']]
        self.so, self.to = ix['sent_offsets'], ix['trans_offsets']
        self.resets = ix['resets'] if 'resets' in ix else None
        self.states = ix['states'] if 'states' in ix else None
        self.gold = ix['gold'] if 'gold' in ix else None
        self.sentences = aligned_sentences(ix, docs)
        self.n_docs = len(self.doc_ids)


def split_authors(c, records):
    sys.path.insert(0, str(T.REPO / 'scripts/eacl_phase1'))
    from author_review_list import parse_person
    amap = {}
    for r in records:
        name, years, _ = parse_person(r['author'].split(';')[0])
        amap[r['doc_id']] = (name.split('(')[0].strip().rstrip(',').lower(), years)
    groups = {}
    for i, name in enumerate(c.doc_ids):
        groups.setdefault(amap.get(name, ('__unknown__', name)), []).append(i)
    keys = sorted(groups, key=lambda k: (str(k[0]), str(k[1])))
    val = []
    for j in np.random.default_rng(20260728).permutation(len(keys)):
        if len(val) >= max(1, int(c.n_docs * .05)):
            break
        val.extend(groups[keys[j]])
    vs = set(val)
    return [i for i in range(c.n_docs) if i not in vs], val


def pack(c, indices, device):
    order = sorted(indices, key=lambda i: -(c.so[i + 1] - c.so[i]))
    lengths = [int(c.so[i + 1] - c.so[i]) for i in order]
    b, t = len(order), max(lengths)
    sm = torch.zeros(b, t, device=device)
    tm = torch.zeros(b, t - 1, device=device)
    rs = torch.zeros(b, t - 1, 5, dtype=torch.long, device=device)
    st = torch.zeros(b, t, 3, dtype=torch.long, device=device)
    texts = []
    for j, (i, n) in enumerate(zip(order, lengths)):
        s0, s1 = map(int, c.so[i:i + 2]); t0, t1 = map(int, c.to[i:i + 2])
        texts.extend(c.sentences[s0:s1])
        sm[j, :n] = 1; tm[j, :n - 1] = 1
        if c.resets is not None:
            rs[j, :n - 1] = torch.as_tensor(c.resets[t0:t1].astype(np.int64), device=device)
            st[j, :n] = torch.as_tensor(c.states[s0:s1].astype(np.int64), device=device)
    return texts, lengths, (rs, st, sm, tm), order


def padded(flat, lengths):
    return torch.nn.utils.rnn.pad_sequence(flat.split(lengths), batch_first=True)


class Encoder:
    def __init__(self, model, tokenizer, microbatch=32, token_budget=4096):
        self.model, self.tokenizer = model, tokenizer
        self.microbatch, self.token_budget = microbatch, token_budget

    def batches(self, texts):
        tokens = self.tokenizer(texts, padding=False, truncation=False)['input_ids']
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        chunk = []
        for i in order:
            n = len(tokens[i])
            if n > self.model.config.max_position_embeddings:
                raise ValueError(f'Sentence exceeds model context: {n}')
            if chunk and (len(chunk) >= self.microbatch or (len(chunk) + 1) * n > self.token_budget):
                yield chunk, self.tokenizer.pad({'input_ids': [tokens[j] for j in chunk]}, return_tensors='pt')
                chunk = []
            chunk.append(i)
        if chunk:
            yield chunk, self.tokenizer.pad({'input_ids': [tokens[j] for j in chunk]}, return_tensors='pt')

    def encode(self, batch):
        batch = batch.to('cuda')
        assert batch.attention_mask[:, -1].all()
        output = self.model(**batch, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
        return F.normalize(output, dim=-1)

    @torch.no_grad()
    def cache(self, chunks, count):
        z = torch.empty(count, self.model.config.hidden_size, device='cuda')
        for indices, batch in chunks:
            z[indices] = self.encode(batch)
        return z

    def replay(self, chunks, gradients, cached, check=False):
        for indices, batch in chunks:
            z = self.encode(batch)
            if check:
                torch.testing.assert_close(z.detach(), cached[indices], rtol=2e-3, atol=2e-4)
            z.backward(gradients[indices])


def tune_encoder(encoder, texts, optimizer, out):
    """Choose throughput settings on training text without updating weights."""
    candidates = [(32, 4096, True), (128, 16384, True), (256, 32768, True),
                  (128, 8192, False), (256, 16384, False), (512, 32768, False)]
    rows = []
    rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
    def benchmark():
        chunks = list(encoder.batches(texts))
        cache = encoder.cache(chunks, len(texts))
        gradients = torch.full_like(cache, 1. / cache.numel())
        encoder.replay(chunks, gradients, cache)
    for microbatch, budget, checkpoint in candidates:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        encoder.microbatch, encoder.token_budget = microbatch, budget
        if checkpoint:
            encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            encoder.model.gradient_checkpointing_disable()
        row = {'microbatch': microbatch, 'token_budget': budget, 'activation_checkpoint': checkpoint}
        torch.cuda.synchronize(); begin = time.monotonic()
        try:
            benchmark()
            torch.cuda.synchronize()
            row.update(seconds=time.monotonic() - begin,
                       peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, status='ok')
        except torch.cuda.OutOfMemoryError:
            row.update(status='oom')
        optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        rows.append(row)
        dump(out / 'throughput_benchmark.json', rows)
        print('BENCHMARK ' + json.dumps(row), flush=True)
    valid = [r for r in rows if r['status'] == 'ok' and r['peak_gpu_gb'] < 55]
    best = min(valid, key=lambda r: r['seconds'])
    encoder.microbatch, encoder.token_budget = best['microbatch'], best['token_budget']
    if best['activation_checkpoint']:
        encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        encoder.model.gradient_checkpointing_disable()
    torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng)
    torch.cuda.reset_peak_memory_stats()
    print('SELECTED ' + json.dumps(best), flush=True)
    return best


def smoke_equivalence():
    """Verify replay gradients equal direct joint backpropagation in FP32."""
    torch.manual_seed(11)
    encoder = torch.nn.Sequential(torch.nn.Linear(7, 12), torch.nn.Tanh(), torch.nn.Linear(12, 16))
    head = T.FactoredSlot(z_dim=16, slot_dim=8, bottleneck=4, dropout=0.,
                          dynamics='per_slot', pred_space='obs')
    x = torch.randn(9, 7); lengths = [5, 4]
    rs = torch.randint(0, 2, (2, 4, 5)); st = torch.zeros(2, 5, 3, dtype=torch.long)
    sm = torch.tensor([[1., 1, 1, 1, 1], [1., 1, 1, 1, 0]])
    tm = sm[:, 1:]
    def loss(flat):
        z = padded(flat, lengths)
        return head.forward_batched(z, rs, st, sm, tm, obs_target=z[:, 1:].detach())['loss']
    direct = F.normalize(encoder(x), dim=-1)
    loss(direct).backward()
    params = list(encoder.parameters()) + list(head.parameters())
    expected = [p.grad.clone() for p in params]
    for p in params: p.grad = None
    with torch.no_grad(): cached = F.normalize(encoder(x), dim=-1)
    leaf = cached.detach().requires_grad_()
    loss(leaf).backward()
    for start in range(0, len(x), 3):
        F.normalize(encoder(x[start:start + 3]), dim=-1).backward(leaf.grad[start:start + 3])
    for p, target in zip(params, expected):
        torch.testing.assert_close(p.grad, target, rtol=3e-4, atol=3e-6)
    assert sum(p.grad.abs().sum().item() for p in encoder.parameters()) > 0
    print('PASS: gradient replay equals direct joint backpropagation', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path)
    ap.add_argument('--raw', type=Path)
    ap.add_argument('--dev-source', type=Path)
    ap.add_argument('--test-source', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--revision')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        smoke_equivalence(); return
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'manifest.json').exists(): raise FileExistsError(a.out)
    records = [json.loads(line) for line in a.raw.open()]
    docs = {r['doc_id']: r['sentences'] for r in records}
    assert len(docs) == len(records)
    c = TextCorpus(a.source, 'phase1__personkey_i2', docs)
    tr, va = split_authors(c, records)
    assert (len(tr), len(va)) == (10430, 1406)
    evaluation = {}
    for split, names, suffix, directory in [
        ('dev3', ['pieman', 'tunnel', 'monkey'], 'michelmann', a.dev_source),
        ('test3', ['run', 'gohitler', 'mymotherslife'], 'eventrecall_goldtopn', a.test_source)]:
        source = {}
        for name in names:
            ids, sentences, _ = collect_docs(directory / f'{name}_{suffix}_master.jsonl')
            source.update(zip(ids, sentences))
        evaluation[split] = TextCorpus(a.source, split, source)
    del records, docs
    model_id = 'Qwen/Qwen3-1.7B'
    revision = a.revision or model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16,
                                    attn_implementation='sdpa').to('cuda')
    base.config.use_cache = False
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pos = np.clip(c.resets.mean(0), 1e-3, 1 - 1e-3)
    pw = np.clip((1 - pos) / pos, 1, 30).tolist()
    head = T.FactoredSlot(z_dim=2048, slot_dim=128, bottleneck=8, dropout=.4,
                         dynamics='per_slot', pred_space='obs', pred_weight=1., pos_weight=pw).cuda()
    lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                             lora_dropout=0., bias='none', target_modules=['q_proj', 'v_proj'])
    model = get_peft_model(base, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # All encoder dropout is disabled so cached and replayed representations agree.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    encoder = Encoder(model, tokenizer)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    assert lora_params and all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.Adam([{'params': head.parameters(), 'lr': 3e-4},
                                  {'params': lora_params, 'lr': 2e-5}])
    manifest = {'model': model_id, 'revision': revision, 'seed': a.seed, 'epochs': 1,
                'train_docs': len(tr), 'val_docs': len(va), 'batch_docs': 48,
                'split_seed': 20260728, 'pred_space': 'obs', 'pos_weight': pw,
                'prediction_target': 'current normalized next embedding; stop-gradient',
                'head_lr': 3e-4, 'lora_lr': 2e-5, 'lora_rank': 16, 'lora_alpha': 32,
                'lora_targets': ['q_proj', 'v_proj'], 'lora_dropout': 0.,
                'lora_parameters': sum(p.numel() for p in lora_params),
                'head_parameters': sum(p.numel() for p in head.parameters()),
                'pooling': 'last valid token, FP32 L2 normalization; no chat template',
                'precision': 'bf16 backbone, fp32 head and adapters',
                'gradient_cache': True, 'threshold': 'online_nf c=1.0',
                'train_ids_sha256': hashlib.sha256('\n'.join(c.doc_ids[i] for i in tr).encode()).hexdigest()}
    dump(a.out / 'manifest.json', manifest)
    dump(a.out / 'split.json', {'train': [c.doc_ids[i] for i in tr], 'val': [c.doc_ids[i] for i in va]})
    print(json.dumps(manifest), flush=True)
    model.train(); head.eval()
    probe_texts = ['A person waited at home.', 'Later she went to the station.',
                   'The train arrived.', 'She boarded and found a seat.']
    probe_chunks = list(encoder.batches(probe_texts))
    probe_cache = encoder.cache(probe_chunks, 4)
    with model.disable_adapter():
        baseline = encoder.cache(probe_chunks, 4)
    torch.testing.assert_close(probe_cache, baseline, rtol=2e-3, atol=2e-4)
    probe_leaf = probe_cache.detach().requires_grad_()
    probe_z = probe_leaf.unsqueeze(0)
    probe_rs = torch.zeros(1, 3, 5, dtype=torch.long, device='cuda')
    probe_st = torch.zeros(1, 4, 3, dtype=torch.long, device='cuda')
    probe_out = head.forward_batched(probe_z, probe_rs, probe_st,
                    torch.ones(1, 4, device='cuda'), torch.ones(1, 3, device='cuda'),
                    obs_target=probe_z[:, 1:].detach())
    probe_out['loss'].backward()
    encoder.replay(probe_chunks, probe_leaf.grad, probe_cache, check=True)
    probe_grad = sum(float(p.grad.float().square().sum()) for p in lora_params if p.grad is not None) ** .5
    assert np.isfinite(probe_grad) and probe_grad > 0
    optimizer.zero_grad(set_to_none=True)
    dump(a.out / 'gpu_preflight.json', {'zero_adapter_matches_base': True,
          'gradient_replay_matches_cache': True, 'lora_grad_norm': probe_grad})
    print('PASS: real Qwen LoRA initialization, replay, gradient flow', flush=True)
    del probe_chunks, probe_cache, probe_leaf, probe_z, probe_out, baseline
    model.train(); head.train()
    order = list(tr); np.random.default_rng(a.seed).shuffle(order)
    batches = T.batches(order, c, 48)
    # The same training-only sentence sample is used for every throughput candidate.
    first_texts = [text for i in batches[0] for text in c.sentences[int(c.so[i]):int(c.so[i + 1])]]
    sample = [first_texts[i] for i in np.linspace(0, len(first_texts) - 1, 1024, dtype=int)]
    manifest['encoder_execution'] = tune_encoder(encoder, sample, optimizer, a.out)
    dump(a.out / 'manifest.json', manifest)
    del first_texts, sample
    start = time.monotonic(); total = 0.; seen = 0
    for step, indices in enumerate(batches, 1):
        texts, lengths, labels, _ = pack(c, indices, 'cuda')
        chunks = list(encoder.batches(texts))
        cached = encoder.cache(chunks, len(texts))
        leaf = cached.detach().requires_grad_()
        z = padded(leaf, lengths)
        optimizer.zero_grad(set_to_none=True)
        result = head.forward_batched(z, *labels, obs_target=z[:, 1:].detach())
        loss_value = float(result['loss'].detach())
        if not np.isfinite(loss_value): raise ValueError('Non-finite training loss')
        result['loss'].backward()
        gradients = leaf.grad.detach()
        del result, z, leaf
        encoder.replay(chunks, gradients, cached, check=step == 1)
        grad_norm = sum(float(p.grad.float().square().sum()) for p in lora_params if p.grad is not None) ** .5
        if not np.isfinite(grad_norm) or grad_norm == 0: raise ValueError('Invalid LoRA gradient')
        if step == 1:
            probe = next(p for p in lora_params if p.grad is not None and p.grad.abs().max() > 0)
            before = probe.detach().clone()
        optimizer.step()
        if step == 1:
            assert not torch.equal(probe, before), 'LoRA weights did not update'
            del before
            print('PASS: replay check, nonzero LoRA gradients, adapter update', flush=True)
        total += loss_value; seen += len(texts)
        elapsed = time.monotonic() - start
        progress = {'stage': 'training', 'epoch': 1, 'step': step, 'steps': len(batches),
                    'loss': loss_value, 'mean_loss': total / step, 'sentences': seen,
                    'elapsed_seconds': elapsed, 'lora_grad_norm': grad_norm,
                    'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9}
        dump(a.out / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
        if step == 1 or step % 10 == 0:
            from peft import get_peft_model_state_dict
            checkpoint = {'step': step, 'head': head.state_dict(),
                          'adapter': get_peft_model_state_dict(model), 'optimizer': optimizer.state_dict(),
                          'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                          'manifest': manifest, 'total_loss': total, 'sentences': seen}
            torch.save(checkpoint, a.out / 'training_latest.tmp')
            (a.out / 'training_latest.tmp').replace(a.out / 'training_latest.pt')
        del chunks, cached, gradients, labels
    # Save before validation so a scoring failure cannot discard the trained model.
    model.save_pretrained(a.out / 'adapter_ep001')
    tokenizer.save_pretrained(a.out / 'adapter_ep001')
    torch.save({'state': head.state_dict(), 'args': {'slot_dim': 128, 'bottleneck': 8,
                'dropout': .4, 'pred_space': 'obs', 'pred_weight': 1., 'head_feat': 'delta'},
                'mono': False, 'dynamics': 'per_slot', 'pos_weight': pw, 'manifest': manifest},
               a.out / 'M5_lora_ep001_seed1.pt')
    model.eval(); head.eval()
    val_losses = []; per_doc = {}
    with torch.no_grad():
        for stage, corpus, indices in [('validation', c, va)] + [
                (split, obj, list(range(obj.n_docs))) for split, obj in evaluation.items()]:
            for batch_indices in T.batches(indices, corpus, 48 if stage == 'validation' else 1):
                texts, lengths, labels, ordered = pack(corpus, batch_indices, 'cuda')
                chunks = list(encoder.batches(texts))
                z = padded(encoder.cache(chunks, len(texts)), lengths)
                output = head.forward_batched(z, *labels, obs_target=z[:, 1:].detach())
                if stage == 'validation': val_losses.append(float(output['loss']))
                else:
                    probs = torch.stack([output['change_logits'][axis].sigmoid() for axis in T.AXES], -1)
                    for b, (i, n) in enumerate(zip(ordered, lengths)):
                        t0, t1 = map(int, corpus.to[i:i + 2])
                        key = f'{stage}/{corpus.doc_ids[i]}'
                        scores = probs[b, :n - 1].sum(-1).cpu().numpy()
                        np.save(a.out / f'{stage}_{i}_axis_probs.npy', probs[b, :n - 1].cpu().numpy())
                        per_doc[key] = metrics(corpus.gold[t0:t1].astype(np.int8), scores, .066)
                dump(a.out / 'progress.json', {'stage': stage, 'last_document': corpus.doc_ids[ordered[-1]]})
    assert len(per_doc) == 6
    # AUC can be undefined in general; requested metrics must be finite on all six works.
    average = {k: float(np.mean([row[k] for row in per_doc.values()])) for k in ('score', 'bor', 'lift')}
    assert all(np.isfinite(x) for x in average.values())
    result = {'seed': a.seed, 'epoch': 1, 'n_narratives': 6, 'average': average,
              'per_narrative': per_doc, 'train_loss': total / len(batches),
              'val_loss': float(np.mean(val_losses)), 'elapsed_seconds': time.monotonic() - start}
    dump(a.out / 'metrics.json', result)
    dump(a.out / 'progress.json', {'stage': 'complete', **result})
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
