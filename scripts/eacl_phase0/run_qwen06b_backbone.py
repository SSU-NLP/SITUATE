"""Re-embed the frozen corpus, preserving its index, then train the M5 head."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import train_anchor_m4m5 as T
from build_anchor_dataset import collect_docs


def aligned_sentences(index, docs):
    ids = [str(x) for x in index['doc_ids']]
    offsets = index['sent_offsets']
    flat = []
    for i, doc in enumerate(ids):
        sentences = docs[doc]
        if len(sentences) != int(offsets[i + 1] - offsets[i]):
            raise ValueError(f'Sentence count mismatch: {doc}')
        if not all(isinstance(s, str) and s.strip() for s in sentences):
            raise ValueError(f'Invalid sentence: {doc}')
        flat.extend(sentences)
    assert len(flat) == int(offsets[-1])
    return flat


def parallel(a):
    world = torch.cuda.device_count()
    children = []
    logs = []
    try:
        for rank in range(world):
            shard_dir = a.persist / f'rank{rank}'
            shard_dir.mkdir(exist_ok=False)
            log = (T.DATA / f'worker{rank}.log').open('w', buffering=1)
            logs.append(log)
            cmd = [sys.executable, '-u', __file__, '--source', str(a.source),
                   '--raw', str(a.raw), '--dev-source', str(a.dev_source),
                   '--test-source', str(a.test_source),
                   '--persist', str(shard_dir), '--batch', str(a.batch),
                   '--rank', str(rank), '--world-size', str(world)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
            children.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        last_report = 0.0
        while any(p.poll() is None for p in children):
            if time.monotonic() - last_report >= 30:
                for rank in range(world):
                    lines = (T.DATA / f'worker{rank}.log').read_text(errors='replace').splitlines()
                    if lines:
                        print(f'[GPU {rank}] {lines[-1]}', flush=True)
                last_report = time.monotonic()
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Embedding worker failed; see rank*/worker.log')
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError('Embedding worker failed')
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()
        for rank, log in enumerate(logs):
            log.close()
            shutil.copy2(T.DATA / f'worker{rank}.log', a.persist / f'rank{rank}' / 'worker.log')
    manifests = [json.loads((a.persist / f'rank{r}' / f'embedding_manifest.rank{r}.json').read_text())
                 for r in range(world)]
    assert len({m['revision'] for m in manifests}) == 1
    for split in ('dev3', 'test3', 'phase1__personkey_i2'):
        ip = a.source / f'{split}_index.npz'
        with np.load(ip, allow_pickle=True) as ix:
            n = int(ix['sent_offsets'][-1])
        dest = T.DATA / f'{split}_emb.npy'
        if dest.exists() or (a.persist / dest.name).exists():
            raise FileExistsError(dest)
        mm = np.lib.format.open_memmap(dest, mode='w+', dtype=np.float16, shape=(n, 1024))
        for rank in range(world):
            shard = np.load(T.DATA / f'{split}.rank{rank}_emb.npy', mmap_mode='r')
            lo, hi = n * rank // world, n * (rank + 1) // world
            assert shard.shape == (hi - lo, 1024)
            for start in range(0, len(shard), 8192):
                mm[lo + start:lo + min(start + 8192, len(shard))] = shard[start:start + 8192]
        mm.flush()
        del mm
        shutil.copy2(ip, T.DATA / ip.name)
        shutil.copy2(ip, a.persist / ip.name)
        shutil.copy2(dest, a.persist / dest.name)
        print(f'Merged {split}: {n:,} sentences from {world} GPUs', flush=True)
    (a.persist / 'embedding_manifest.json').write_text(json.dumps(
        {'world_size': world, 'shards': manifests, 'merge_order': list(range(world))}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--dev-source', type=Path, required=True)
    ap.add_argument('--persist', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--rank', type=int, default=-1)
    ap.add_argument('--world-size', type=int, default=1)
    ap.add_argument('--train-only', action='store_true')
    ap.add_argument('--embed-only', action='store_true')
    ap.add_argument('--test-source', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    a.persist.mkdir(parents=True, exist_ok=True)
    T.DATA.mkdir(parents=True, exist_ok=True)
    if a.train_only:
        train(a)
        return
    if a.rank < 0 and torch.cuda.device_count() > 1 and not a.preflight_only:
        parallel(a)
        if not a.embed_only:
            train(a)
        return
    docs = {}
    with a.raw.open() as f:
        for line in f:
            r = json.loads(line)
            if r['doc_id'] in docs:
                raise ValueError('Duplicate document ID')
            docs[r['doc_id']] = r['sentences']
    devdocs = {}
    for name in ('pieman', 'tunnel', 'monkey'):
        ids, sentences, _ = collect_docs(a.dev_source / f'{name}_michelmann_master.jsonl')
        devdocs.update(zip(ids, sentences))
    testdocs = {}
    for name in ('run', 'gohitler', 'mymotherslife'):
        ids, sentences, _ = collect_docs(a.test_source / f'{name}_eventrecall_goldtopn_master.jsonl')
        testdocs.update(zip(ids, sentences))
    datasets = []
    for split, source_docs in [('dev3', devdocs), ('test3', testdocs), ('phase1__personkey_i2', docs)]:
        ip = a.source / f'{split}_index.npz'
        with np.load(ip, allow_pickle=True) as index:
            sentences = aligned_sentences(index, source_docs)
            count = len(index['doc_ids'])
        if a.rank >= 0:
            lo = len(sentences) * a.rank // a.world_size
            hi = len(sentences) * (a.rank + 1) // a.world_size
            sentences = sentences[lo:hi]
        datasets.append((split, sentences, ip))
        print(f'Validated {split}: {count} documents, {len(sentences):,} sentences', flush=True)
    if a.preflight_only:
        return
    del docs, devdocs
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer
    model_id = 'Qwen/Qwen3-Embedding-0.6B'
    revision = model_info(model_id).sha
    enc = SentenceTransformer(model_id, revision=revision, device='cuda',
                              model_kwargs={'torch_dtype': torch.bfloat16,
                                            'attn_implementation': 'sdpa'},
                              tokenizer_kwargs={'padding_side': 'left'})
    assert enc.get_sentence_embedding_dimension() == 1024
    enc.eval()
    metadata = {'model': model_id, 'revision': revision, 'dimension': 1024,
                'storage_dtype': 'float16', 'compute_dtype': 'bfloat16',
                'normalize_embeddings': True, 'prompt': '', 'seed': a.seed,
                'max_seq_length': enc.max_seq_length, 'datasets': {}}

    def encode(texts):
        try:
            vectors = enc.encode(texts, batch_size=len(texts), prompt='',
                                 normalize_embeddings=False, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(vectors).all() or not (norms > 0).all():
                raise ValueError('Non-finite or zero embedding')
            return (vectors / norms).astype(np.float16)
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(texts) // 2
            return np.concatenate((encode(texts[:mid]), encode(texts[mid:])))

    for split, sentences, ip in datasets:
        suffix = f'.rank{a.rank}' if a.rank >= 0 else ''
        dest = T.DATA / f'{split}{suffix}_emb.npy'
        saved = a.persist / dest.name
        if saved.exists() or dest.exists():
            raise FileExistsError(f'Refusing to overwrite {dest} or {saved}')
        mm = np.lib.format.open_memmap(dest.with_suffix('.partial.npy'), mode='w+',
                                      dtype=np.float16, shape=(len(sentences), 1024))
        start = time.monotonic()
        digest = hashlib.sha256()
        for base in range(0, len(sentences), 1024):
            chunk = sentences[base:base + 1024]
            lengths = [len(x) for x in enc.tokenizer(chunk, truncation=False)['input_ids']]
            if max(lengths) > enc.max_seq_length:
                raise ValueError(f'{split}: sentence exceeds model context; refusing truncation')
            order = sorted(range(len(chunk)), key=lambda i: lengths[i])
            for j in range(0, len(order), a.batch):
                idx = order[j:j + a.batch]
                vectors = encode([chunk[i] for i in idx])
                norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
                if not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=0.002):
                    raise ValueError(f'Invalid embedding norms: {norms.min()}..{norms.max()}')
                mm[base + np.asarray(idx)] = vectors
            for s in chunk:
                digest.update(json.dumps(s, ensure_ascii=False).encode() + b'\n')
            done = base + len(chunk)
            if base == 0 or done % 10240 == 0 or done == len(sentences):
                elapsed = time.monotonic() - start
                print(f'{split}: {done:,}/{len(sentences):,}, {done/elapsed:.1f} sentences/s, '
                      f'ETA {(len(sentences)-done)*elapsed/done/3600:.2f} h', flush=True)
        mm.flush()
        del mm
        dest.with_suffix('.partial.npy').rename(dest)
        if a.rank < 0:
            shutil.copy2(ip, T.DATA / ip.name)
        shutil.copy2(dest, saved)
        shutil.copy2(ip, a.persist / ip.name)
        metadata['datasets'][split] = {'sentences': len(sentences),
             'text_sha256': digest.hexdigest(), 'index_sha256': hashlib.sha256(ip.read_bytes()).hexdigest(),
             'seconds': time.monotonic() - start}
        manifest_name = f'embedding_manifest{suffix}.json'
        (a.persist / manifest_name).write_text(json.dumps(metadata, indent=2))
    del enc, datasets, sentences
    gc.collect()
    torch.cuda.empty_cache()
    if a.rank < 0 and not a.embed_only:
        train(a)


def train(a):
    # Keep the original split names inside an isolated work directory.
    cmd = [sys.executable, str(Path(__file__).with_name('train_anchor_m4m5.py')),
           '--split', 'phase1__personkey_i2', '--models', 'M5', '--seed', str(a.seed),
           '--epochs', '1', '--patience', '0', '--ckpt-every', '1',
           '--group-split', 'author',
           '--author-map', str(a.raw), '--ns-suffix', 'qwen06b', '--log-every', '1']
    subprocess.run(cmd, check=True)
    checkpoints = sorted(T.EXP.glob(f'M5*qwen06b_ep???_seed{a.seed}.pt'))
    assert len(checkpoints) == 1, f'Expected 1 epoch snapshot, got {len(checkpoints)}'
    for p in T.EXP.glob('*qwen06b*'):
        if p.is_file():
            shutil.copy2(p, a.persist / p.name)
    from eval_all_models_alldocs import m5_scores
    corpora = {'dev-3': T.Corpus('dev3'), 'test3': T.Corpus('test3')}
    rows = []
    for epoch, checkpoint in enumerate(checkpoints, start=1):
        assert checkpoint.name.endswith(f'_ep{epoch:03d}_seed{a.seed}.pt')
        scores = m5_scores(str(checkpoint), corpora, 'cuda', .066)
        assert len(scores) == 6, f'Expected six narratives, got {len(scores)}'
        averages = {}
        for metric in ('score', 'lift', 'bor'):
            values = [v[metric] for v in scores.values()]
            # An undefined value in one narrative makes the six-narrative mean undefined.
            averages[metric] = float(np.mean(values)) if np.isfinite(values).all() else None
        rows.append({'epoch': epoch, 'checkpoint': checkpoint.name,
                     'average': averages, 'per_narrative': scores})
        print(f'Epoch {epoch:02d}: {averages}', flush=True)
    result = {'n_narratives': 6, 'seed': a.seed, 'epochs': 1, 'early_stopping': False,
              'aggregation': 'unweighted narrative mean; all six required',
              'boundary_rule': 'online_nf c=1.0', 'epoch_metrics': rows}
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (a.persist / 'epoch_metrics.json').write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    table = ['| Epoch | SCORE | LIFT | BOR |', '| ---: | ---: | ---: | ---: |']
    csv = ['epoch,score,lift,bor']
    for row in rows:
        vals = [row['average'][k] for k in ('score', 'lift', 'bor')]
        formatted = ['—' if v is None else f'{v:.4f}' for v in vals]
        table.append('| ' + str(row['epoch']) + ' | ' + ' | '.join(formatted) + ' |')
        csv.append(','.join([str(row['epoch'])] + ['' if v is None else str(v) for v in vals]))
    (a.persist / 'epoch_metrics.md').write_text('\n'.join(table) + '\n')
    (a.persist / 'epoch_metrics.csv').write_text('\n'.join(csv) + '\n')
    print('\n'.join(table), flush=True)
    print('DONE: 1 epoch and six-narrative evaluation for each checkpoint', flush=True)



if __name__ == '__main__':
    main()
