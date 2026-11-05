#!/usr/bin/env python3
"""Uncached decision/observation latency on fixed held-out teacher inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from hi_ontop.memory_update.common import API, Encoder, atomic_json, turn_texts
from hi_ontop.memory_update.model import UpdatePolicy


def main(output, env):
    from mem0.configs.prompts import get_update_memory_messages
    cfg = json.loads((ROOT / 'configs/mem0_update_v1.json').read_text())
    torch.set_num_threads(4)
    api = API(env, output / 'api.sqlite', cfg['llm'], extra_body=cfg.get('llm_extra_body'))
    encoder = Encoder(cfg['encoder'], cfg['encoder_revision'], output / 'embeddings.sqlite')
    models = {}
    for arm in ('stateless', 'stateful'):
        ckpt = torch.load(output / 'models' / f'{arm}_s0' / 'best.pt', map_location='cpu', weights_only=False)
        model = UpdatePolicy(**ckpt['spec']).eval()
        model.load_state_dict(ckpt['state_dict'])
        models[arm] = model
    report_path = output / 'latency.json'
    if report_path.exists():
        raise RuntimeError('Latency measurement already exists; do not accidentally pay for a duplicate measurement')
    records = []
    for path in sorted((output / 'traces').glob('*.json')):
        dialogue = json.loads(path.read_text())
        if dialogue['split'] != 'dev':
            continue
        states = {a: None for a in models}
        for index, trace in enumerate(dialogue['traces']):
            chunks = encoder.history_chunks(turn_texts(trace['messages']))
            start = time.perf_counter()
            turn_z = torch.from_numpy(encoder.model.encode(chunks, batch_size=32,
                        normalize_embeddings=True, show_progress_bar=False))
            encoding_seconds = time.perf_counter() - start
            fact_z = torch.from_numpy(encoder.encode(trace['facts']))
            mem_z = torch.from_numpy(encoder.encode([m['text'] for m in trace['memories']]))
            row = dict(conversation=dialogue['id'], session=index, history_chunks=len(chunks),
                       observation_encoding_seconds=encoding_seconds)
            for arm, model in models.items():
                start = time.perf_counter()
                with torch.no_grad():
                    states[arm] = model.observe(turn_z, states[arm])
                observation = time.perf_counter() - start
                if trace['facts']:
                    times = []
                    for _ in range(6):
                        start = time.perf_counter()
                        model.decide(fact_z, mem_z, states[arm], max_actions=max(128, 2 * (len(fact_z) + len(mem_z)) + 1))
                        times.append(time.perf_counter() - start)
                    row[arm] = dict(decision_seconds=float(np.median(times[1:])),
                                    recurrence_seconds=observation,
                                    observation_inclusive_seconds=(encoding_seconds + observation if arm == 'stateful' else 0))
            if trace['facts']:
                prompt = get_update_memory_messages(trace['memories'], trace['facts'])
                api.chat([dict(role='user', content=prompt)], 'latency_decision', json_mode=True, use_cache=False)
                row['llm_decision_seconds'] = api.records[-1]['seconds']
                row['llm_usage'] = api.records[-1]['usage']
                records.append(row)
                print(f'latency sample {len(records)}/64', flush=True)
            if len(records) >= 64:
                break
        if len(records) >= 64:
            break
    summary = {}
    for arm in models:
        values = [r[arm]['decision_seconds'] + r[arm]['observation_inclusive_seconds'] for r in records]
        summary[arm] = dict(median_seconds=float(np.median(values)), p95_seconds=float(np.quantile(values, .95)))
    values = [r['llm_decision_seconds'] for r in records]
    summary['llm'] = dict(median_seconds=float(np.median(values)), p95_seconds=float(np.quantile(values, .95)))
    atomic_json(report_path, dict(records=records, summary=summary, device='cpu', threads=4,
                                 cpu=platform.processor(), torch=torch.__version__,
                                 scope='Held-out synthetic teacher inputs; excludes shared extraction/retrieval/rendering; stateful includes uncached history encoding'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/experiments/mem0_update_oss_s0')
    ap.add_argument('--env', type=Path, default=ROOT / 'EACL/.env')
    args = ap.parse_args()
    main(args.output, args.env)
