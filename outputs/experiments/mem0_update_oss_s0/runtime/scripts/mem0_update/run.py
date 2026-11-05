#!/usr/bin/env python3
"""Reproducible data generation, Mem0 traces, training, and official LongMemEval QA."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['MEM0_TELEMETRY'] = 'false'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch

from hi_ontop.memory_update.common import API, Encoder, atomic_json, bounded_workers, clean_session, digest, turn_texts
from hi_ontop.memory_update.model import UpdatePolicy, labels


def config(args):
    return json.loads(Path(args.config).read_text())


def resources(args, encoder=True):
    cfg = config(args)
    args.output.mkdir(parents=True, exist_ok=True)
    api = API(args.env, args.output / 'api.sqlite', cfg['llm'], extra_body=cfg.get('llm_extra_body'))
    enc = Encoder(cfg['encoder'], cfg['encoder_revision'], args.output / 'embeddings.sqlite') if encoder else None
    return cfg, api, enc


def prepare_benchmark(args):
    import ijson
    cfg = config(args)
    root = args.output / 'data'
    path = root / cfg['benchmark']
    references, index = [], []
    with path.open('rb') as stream:
        for entry in ijson.items(stream, 'item'):
            qid = entry['question_id']
            if qid in index:
                raise ValueError('Duplicate question ID')
            index.append(qid)
            references.append({k: entry[k] for k in ('question_id', 'question_type', 'question', 'question_date', 'answer')})
            inputs = {k: entry[k] for k in ('question_id', 'question', 'question_date', 'haystack_dates', 'haystack_session_ids')}
            inputs['haystack_sessions'] = [[{k: t[k] for k in ('role', 'content')} for t in s] for s in entry['haystack_sessions']]
            atomic_json(root / 'questions' / f'{qid}.json', inputs)
    if len(index) != cfg['expected_questions']:
        raise ValueError('Incomplete benchmark')
    atomic_json(root / 'references.json', references)
    atomic_json(root / 'index.json', index)
    print(f'Prepared {len(index)} questions; removed evidence annotations from inference inputs', flush=True)


class Benchmark(Sequence):
    def __init__(self, root, index):
        self.root, self.index = root, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        return json.loads((self.root / 'questions' / f'{self.index[key]}.json').read_text())


def benchmark(args, histories=True):
    path = args.output / 'data' / config(args)['benchmark']
    root = path.parent
    if not (root / 'index.json').exists():
        raise RuntimeError('Run prepare_benchmark first')
    index = json.loads((root / 'index.json').read_text())
    if len(index) != config(args)['expected_questions'] or len(set(index)) != len(index):
        raise ValueError('Incomplete or duplicate benchmark questions')
    return Benchmark(root, index) if histories else json.loads((root / 'references.json').read_text())


def synth(args):
    cfg, api, _ = resources(args, encoder=False)
    # Independent synthetic dialogues: no benchmark examples, questions, labels or histories in the prompt.
    topics = ['career and commuting', 'family and housing', 'hobbies and clubs', 'travel and reservations',
              'education and courses', 'fitness and routines', 'pets and care', 'food and preferences',
              'volunteering and events', 'creative projects', 'shopping and subscriptions', 'friends and plans',
              'gardening and seasons', 'team sports', 'music and concerts', 'language learning']
    total = cfg['train_conversations'] + cfg['dev_conversations']
    count = min(args.limit or total - args.start_index, total - args.start_index)
    dest = args.output / 'conversations'
    dest.mkdir(exist_ok=True)

    def generate(index):
        path = dest / f'profile_{index:04d}.json'
        if path.exists():
            return
        rng = random.Random(1741 + index)
        selected = rng.sample(topics, 4)
        prompt = (
            f'Create a realistic English chat history for fictional person P{index:04d}. '
            f'Use these domains: {selected}. Diversity seed: {1741 + index}. '
            f'Return JSON {{"sessions": [{{"date": "YYYY-MM-DD", "turns": '
            '[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}]}}. '
            f'Exactly {cfg["sessions_per_conversation"]} sessions in strictly chronological order during 2023-2024, '
            'each containing 2-4 alternating user and assistant messages, 40-100 words total per session. '
            'Spread several ongoing stories across distant sessions with intervening unrelated subjects. '
            'Introduce people with different attributes, references such as she/there/that plan, and dates. '
            'Include new facts, changed persistent preferences or circumstances, cancelled plans, explicit '
            'requests to forget obsolete information, corrections, and repeated unchanged facts. Distinguish '
            'temporary plans from completed actions; include historical facts that remain true. Roughly a '
            'quarter of sessions should add no important personal facts. Do not provide memory labels, '
            'evaluation questions, or answers. Keep each person internally consistent, with changes explained.'
        )
        messages = [dict(role='user', content=prompt)]
        for attempt in range(3):
            response = api.chat(messages, 'synthesis', True, max_tokens=10000)
            obj = json.loads(response)
            try:
                sessions = obj['sessions']
                if len(sessions) < cfg['sessions_per_conversation']:
                    raise ValueError(f'Need at least {cfg["sessions_per_conversation"]} sessions; got {len(sessions)}')
                obj['generation_original_sessions'] = len(sessions)
                # Fixed chronological prefix, independent of content/operations/QA scores.
                sessions = sessions[:cfg['sessions_per_conversation']]
                obj['sessions'] = sessions
                dates = [s['date'] for s in sessions]
                if dates != sorted(dates) or len(set(dates)) != len(dates):
                    raise ValueError('Dates must be strictly increasing')
                for session in sessions:
                    if not 2 <= len(session['turns']) <= 4:
                        raise ValueError('Each session needs 2-4 turns')
                    if any(t['role'] not in ('user', 'assistant') or not t['content'] for t in session['turns']):
                        raise ValueError('Malformed turn')
                break
            except (ValueError, KeyError) as exc:
                messages += [dict(role='assistant', content=response), dict(role='user', content=
                    f'Correct this error and return the complete JSON: {exc}')]
        else:
            raise ValueError(f'Invalid generated dialogue {index} after three attempts')
        obj.update(id=f'profile_{index:04d}', split='train' if index < cfg['train_conversations'] else 'dev',
                   generator=cfg['llm'], generation_prompt_sha256=digest(prompt))
        atomic_json(path, obj)
        print(f'generated {index + 1}/{total}', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(generate, range(args.start_index, args.start_index + count)))
    atomic_json(args.output / 'synthesis_usage.json', api.records)


def audit(args):
    cfg = config(args)
    data = benchmark(args)
    heldout = {digest(' '.join(t['content'].lower().split())) for q in data
               for session in q['haystack_sessions'] for t in session if len(t['content']) > 80}
    training = list((args.output / 'conversations').glob('*.json'))
    ids, split_counts, overlaps = set(), Counter(), []
    for path in training:
        obj = json.loads(path.read_text())
        if obj['id'] in ids:
            raise ValueError('Duplicate conversation ID')
        ids.add(obj['id'])
        split_counts[obj['split']] += 1
        for s in obj['sessions']:
            for t in s['turns']:
                if len(t['content']) > 80 and digest(' '.join(t['content'].lower().split())) in heldout:
                    overlaps.append(obj['id'])
    report = dict(benchmark_questions=len(data), generated=split_counts, overlaps=overlaps,
                  benchmark_sha256=hashlib.sha256((args.output / 'data' / cfg['benchmark']).read_bytes()).hexdigest(),
                  conversation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in training},
                  config_sha256=digest(cfg), audit='exact normalized long-turn overlap; synthetic independent provenance')
    atomic_json(args.output / 'data_audit.json', report)
    if overlaps:
        raise ValueError('Training/evaluation overlap')
    print(json.dumps({k: v for k, v in report.items() if k != 'conversation_sha256'}, indent=2))


def collect(args):
    from hi_ontop.memory_update.adapter import DecisionMemory
    cfg, root_api, enc = resources(args)
    files = sorted((args.output / 'conversations').glob('*.json'))
    if args.limit:
        files = files[:args.limit]
    def one(path):
        api = root_api.fork()
        obj = json.loads(path.read_text())
        out = args.output / 'traces' / f'{obj["id"]}.json'
        if out.exists():
            old = json.loads(out.read_text())
            if old['config_sha256'] != digest(cfg) or old['source_sha256'] != digest(obj):
                raise ValueError('Stale training trace')
            return
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='mem0-trace-') as temp:
            memory = DecisionMemory(enc, api, temp)
            try:
                for i, session in enumerate(obj['sessions']):
                    memory.ingest(clean_session(session['turns'], session['date']), session['date'], f'{obj["id"]}:{i}')
                atomic_json(out, dict(id=obj['id'], split=obj['split'], traces=memory.traces,
                                      usage=api.records, source_sha256=digest(obj),
                                      config_sha256=digest(cfg), upstream_sha256=memory.upstream_sha256))
            finally:
                memory.close()
        print(f'collected {obj["id"]} in {time.perf_counter()-start:.1f}s', flush=True)
    bounded_workers(one, (p for i, p in enumerate(files) if i % args.shards == args.shard), args.workers)


def pack(args):
    cfg, _, enc = resources(args)
    audit_report = json.loads((args.output / 'data_audit.json').read_text())
    if audit_report['overlaps'] or audit_report['config_sha256'] != digest(cfg):
        raise ValueError('Missing current clean data audit')
    packed, distribution, counts = [], Counter(), Counter()
    for path in sorted((args.output / 'traces').glob('*.json')):
        obj = json.loads(path.read_text())
        if obj['config_sha256'] != digest(cfg):
            raise ValueError('Trace/config mismatch')
        current = json.loads((args.output / 'conversations' / f'{obj["id"]}.json').read_text())
        if digest(current) != obj['source_sha256']:
            raise ValueError('Conversation changed after trace collection')
        sessions = []
        for t in obj['traces']:
            program = labels(t['actions'], t['memories'])
            distribution.update(a['event'] for a in t['actions'])
            sessions.append(dict(turns=torch.from_numpy(enc.encode_history(turn_texts(t['messages']))),
                                 facts=torch.from_numpy(enc.encode(t['facts'])),
                                 memories=torch.from_numpy(enc.encode([m['text'] for m in t['memories']])),
                                 actions=program))
        packed.append(dict(id=obj['id'], split=obj['split'], sessions=sessions))
        counts[obj['split']] += 1
    if not args.allow_partial:
        if counts != {'train': cfg['train_conversations'], 'dev': cfg['dev_conversations']}:
            raise ValueError(f'Incomplete traces: {counts}')
        if any(distribution[k] < 30 for k in ('ADD', 'UPDATE', 'DELETE', 'NONE')):
            raise ValueError(f'Insufficient operation coverage: {distribution}')
    torch.save(dict(data=packed, config=cfg, config_sha256=digest(cfg), audit=audit_report), args.output / 'training.pt')
    atomic_json(args.output / 'training_summary.json', dict(conversations=counts, operations=distribution,
                                                          partial=bool(args.allow_partial)))
    print(counts, distribution, flush=True)


def train(args):
    cfg = config(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Each training JOB must expose exactly one GPU')
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    bundle = torch.load(args.output / 'training.pt', weights_only=False, map_location='cpu')
    if bundle['config_sha256'] != digest(cfg):
        raise ValueError('Training/config mismatch')
    data = bundle['data']
    training = [d for d in data if d['split'] == 'train']
    dev = [d for d in data if d['split'] == 'dev']
    if not training or not dev:
        raise ValueError('Both train and dev conversations are required')
    model = UpdatePolicy(cfg['embedding_dim'], cfg['width'], cfg['slot_dim'], stateful=args.arm == 'stateful').to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    dest = args.output / 'models' / f'{args.arm}_s{args.seed}'
    dest.mkdir(parents=True, exist_ok=True)
    best, stale, logs = float('inf'), 0, []
    for epoch in range(1, cfg['epochs'] + 1):
        start = time.perf_counter()
        model.train()
        rng = random.Random(args.seed * 1000 + epoch)
        rng.shuffle(training)
        losses = []
        for conversation in training:
            state, window = None, []
            optimizer.zero_grad(set_to_none=True)
            for i, session in enumerate(conversation['sessions']):
                state = model.observe(session['turns'].to(args.device), state)
                if len(session['facts']):
                    loss, _, _ = model.loss(session['facts'].to(args.device), session['memories'].to(args.device), state, session['actions'])
                    window.append(loss)
                    losses.append(float(loss.detach()))
                if (i + 1) % cfg['tbptt_sessions'] == 0 or i == len(conversation['sessions']) - 1:
                    if window:
                        torch.stack(window).mean().backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        window = []
                    if state is not None:
                        state = state.detach()
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for conversation in dev:
                state = None
                for session in conversation['sessions']:
                    state = model.observe(session['turns'].to(args.device), state)
                    if len(session['facts']):
                        loss, _, _ = model.loss(session['facts'].to(args.device), session['memories'].to(args.device), state, session['actions'])
                        dev_losses.append(float(loss))
        metric = float(np.mean(dev_losses))
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), dev_loss=metric, seconds=time.perf_counter()-start)
        if not np.isfinite(metric):
            raise ValueError('Nonfinite validation loss')
        logs.append(row)
        print(json.dumps(row), flush=True)
        if metric < best:
            best, stale = metric, 0
            torch.save(dict(spec=model.spec, state_dict=model.state_dict(), config_sha256=digest(cfg),
                            seed=args.seed, epoch=epoch, dev_loss=metric, training_sha256=hashlib.sha256((args.output / 'training.pt').read_bytes()).hexdigest()),
                       dest / 'best.pt')
        else:
            stale += 1
        atomic_json(dest / 'training.json', dict(arm=args.arm, seed=args.seed, epochs=logs,
                                               parameters=sum(p.numel() for p in model.parameters())))
        if stale >= cfg['patience']:
            break
    atomic_json(dest / 'complete.json', dict(best_dev_loss=best, config_sha256=digest(cfg)))


def evaluate(args):
    from hi_ontop.memory_update.adapter import DecisionMemory
    cfg, root_api, enc = resources(args)
    dataset = benchmark(args)
    policy = None
    model_hash = None
    if args.arm != 'llm':
        checkpoint_path = args.output / 'models' / f'{args.arm}_s{args.seed}' / 'best.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['config_sha256'] != digest(cfg):
            raise ValueError('Checkpoint/config mismatch')
        model_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        policy = UpdatePolicy(**checkpoint['spec'])
        policy.load_state_dict(checkpoint['state_dict'])
        policy.eval()
    name = args.arm + (f'_s{args.seed}' if policy else '') + ('_native' if args.native else '') + ('_reset' if args.reset_state else '')
    # Smoke examples are stored separately and can never be mistaken for full evaluation.
    dest = args.output / ('smoke' if args.limit else 'evaluation') / name
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    def one(index):
        entry = dataset[index]
        api = root_api.fork()
        out = dest / f'{entry["question_id"]}.json'
        if out.exists():
            old = json.loads(out.read_text())
            if old['config_sha256'] != digest(cfg) or old['model_sha256'] != model_hash:
                raise ValueError('Stale evaluation output')
            return
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='mem0-eval-') as temp:
            memory = DecisionMemory(enc, api, temp, policy=policy, native=args.native, reset_state=args.reset_state)
            try:
                for turns, date, sid in zip(entry['haystack_sessions'], entry['haystack_dates'], entry['haystack_session_ids'], strict=True):
                    memory.ingest(clean_session(turns, date), date, sid)
                retrieved = memory.search(entry['question'], user_id='subject', limit=cfg['retrieval_k'])['results']
                context = [{'memory': m['memory'], 'date': m.get('metadata', {}).get('session_date')} for m in retrieved]
                prompt = ('Answer the question using only the retrieved conversation memories. Use the question '
                          'date when interpreting time. Prefer the most recent applicable information when it '
                          'changed; distinguish historical facts from current facts. If the memories do not '
                          'contain enough information, explicitly say you cannot determine the answer. '
                          'Give a concise but complete answer.\n' + json.dumps(dict(memories=context,
                          question=entry['question'], question_date=entry['question_date']), ensure_ascii=False))
                hypothesis = api.chat([dict(role='user', content=prompt)], 'answer', max_tokens=1024)
                atomic_json(out, dict(question_id=entry['question_id'], hypothesis=hypothesis,
                                      traces=memory.traces, usage=api.records, retrieved=retrieved,
                                      wall_seconds=time.perf_counter()-start, config_sha256=digest(cfg),
                                      model_sha256=model_hash))
            finally:
                memory.close()
        print(f'evaluated {name} {index+1}/{len(dataset)} in {time.perf_counter()-start:.1f}s', flush=True)
    bounded_workers(one, (i for i in range(count) if i % args.shards == args.shard), args.workers)


def judge(args):
    cfg, api, _ = resources(args, encoder=False)
    path = args.output / 'vendor/LongMemEval/src/evaluation/evaluate_qa.py'
    spec = importlib.util.spec_from_file_location('official_longmemeval', path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    refs = {e['question_id']: e for e in benchmark(args, histories=False)}
    for directory in sorted((args.output / ('smoke' if args.limit else 'evaluation')).iterdir()):
        paths = sorted(directory.glob('*.json'))
        if not args.limit and {p.stem for p in paths} != set(refs):
            raise ValueError(f'Incomplete evaluation: {directory.name} {len(paths)}/500')
        for path in paths:
            obj = json.loads(path.read_text())
            dest = args.output / ('smoke_judgments' if args.limit else 'judgments') / directory.name / path.name
            if dest.exists():
                continue
            ref = refs[obj['question_id']]
            prompt = official.get_anscheck_prompt(ref['question_type'], ref['question'], ref['answer'],
                                                  obj['hypothesis'], abstention='_abs' in ref['question_id'])
            text = api.chat([dict(role='user', content=prompt)], 'judge', model=cfg['judge'], max_tokens=10).strip()
            atomic_json(dest, dict(question_id=obj['question_id'], hypothesis=obj['hypothesis'],
                                   question_type=ref['question_type'], abstention='_abs' in ref['question_id'],
                                   autoeval_label={'model': cfg['judge'], 'label': 'yes' in text.lower()},
                                   judge_response=text, source_sha256=digest(obj)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare_benchmark', 'synth', 'audit', 'collect', 'pack', 'train', 'evaluate', 'judge'])
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/mem0_update_v1.json')
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--env', type=Path, default=ROOT / 'EACL/.env')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arm', choices=['llm', 'stateless', 'stateful'], default='llm')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--reset-state', action='store_true')
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()
    if args.output is None:
        args.output = ROOT / 'outputs/experiments' / config(args)['experiment']
    if not 0 <= args.shard < args.shards:
        ap.error('invalid shard')
    if args.workers < 1:
        ap.error('workers must be positive')
    torch.set_num_threads(4)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
