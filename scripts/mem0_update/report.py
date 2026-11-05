#!/usr/bin/env python3
"""Paired QA comparisons and disaggregated update costs; refuses incomplete runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import sqlite3

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hi_ontop.memory_update.common import atomic_json


def paired_interval(a, b):
    difference = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(1979)
    samples = difference[rng.integers(0, len(difference), size=(10000, len(difference)))].mean(1)
    return dict(difference=float(difference.mean()), ci95=[float(x) for x in np.quantile(samples, [.025, .975])])


def summarize(output):
    cfg = json.loads((ROOT / 'configs/mem0_update_v1.json').read_text())
    refs = json.loads((output / 'data/references.json').read_text())
    qids = [r['question_id'] for r in refs]
    expected = ['llm_native', 'llm'] + [f'{arm}_s{s}' for arm in ('stateless', 'stateful') for s in cfg['seeds']]
    summary, arrays = {}, {}
    for name in expected:
        paths = list((output / 'judgments' / name).glob('*.json'))
        data = {p.stem: json.loads(p.read_text()) for p in paths}
        if set(data) != set(qids):
            raise ValueError(f'{name} has {len(data)}/{len(qids)} judged questions; no full score written')
        labels = np.array([float(data[q]['autoeval_label']['label']) for q in qids])
        arrays[name] = labels
        types = defaultdict(list)
        for qid, value in zip(qids, labels):
            types[data[qid]['question_type']].append(value)
            if data[qid]['abstention']:
                types['abstention'].append(value)
        phase_usage = defaultdict(lambda: dict(input_tokens=0, output_tokens=0, calls=0, cache_hits=0,
                                              original_api_seconds=0.0, reported_uncached_usd=0.0,
                                              reported_incremental_usd=0.0))
        decision_times, observation_times, wall_seconds = [], [], 0.0
        operations = defaultdict(int)
        for qid in qids:
            run = json.loads((output / 'evaluation' / name / f'{qid}.json').read_text())
            wall_seconds += run['wall_seconds']
            for entry in run['usage']:
                metrics = phase_usage[entry['phase']]
                metrics['calls'] += 1
                metrics['cache_hits'] += int(entry['cache_hit'])
                metrics['input_tokens'] += entry['usage']['prompt_tokens']
                metrics['output_tokens'] += entry['usage']['completion_tokens']
                metrics['original_api_seconds'] += entry['seconds']
                metrics['reported_uncached_usd'] += entry['usage'].get('cost', 0) or 0
                if not entry['cache_hit']:
                    metrics['reported_incremental_usd'] += entry['usage'].get('cost', 0) or 0
            for trace in run['traces']:
                observation_times.append(trace['observation_seconds'])
                if trace['facts']:
                    decision_times.append(trace['decision_seconds'])
                for act in trace['actions']:
                    operations[act['event']] += 1
        for usage in phase_usage.values():
            usage['reference_usd'] = (usage['input_tokens'] * cfg['input_usd_per_million'] +
                                      usage['output_tokens'] * cfg['output_usd_per_million']) / 1e6
        summary[name] = dict(n=len(labels), accuracy=float(labels.mean()),
                             by_type={t: dict(n=len(v), accuracy=float(np.mean(v))) for t, v in types.items()},
                             phase_usage=dict(phase_usage), operations=dict(operations),
                             observed_decision_seconds={'median': float(np.median(decision_times)), 'p95': float(np.quantile(decision_times, .95))},
                             observation_seconds_total=float(sum(observation_times)), observed_wall_seconds=wall_seconds)
    comparisons = {}
    for arm in ('stateless', 'stateful'):
        mean = np.stack([arrays[f'{arm}_s{s}'] for s in cfg['seeds']]).mean(0)
        comparisons[f'{arm}_minus_llm'] = paired_interval(mean, arrays['llm'])
        comparisons[f'{arm}_minus_native_mem0'] = paired_interval(mean, arrays['llm_native'])
        comparisons[f'{arm}_seed_accuracy'] = [float(arrays[f'{arm}_s{s}'].mean()) for s in cfg['seeds']]
    a = np.stack([arrays[f'stateful_s{s}'] for s in cfg['seeds']]).mean(0)
    b = np.stack([arrays[f'stateless_s{s}'] for s in cfg['seeds']]).mean(0)
    comparisons['stateful_minus_stateless'] = paired_interval(a, b)
    for typ in ('knowledge-update', 'temporal-reasoning'):
        subset = np.array([r['question_type'] == typ for r in refs])
        comparisons[f'{typ}_stateful_minus_stateless'] = paired_interval(a[subset], b[subset])
    decision = dict(noninferior_to_llm=comparisons['stateful_minus_llm']['ci95'][0] > -cfg['qa_noninferiority_margin'],
                    noninferior_to_native_mem0=comparisons['stateful_minus_native_mem0']['ci95'][0] > -cfg['qa_noninferiority_margin'],
                    state_advantage=comparisons['stateful_minus_stateless']['ci95'][0] > 0)
    db = sqlite3.connect(output / 'api.sqlite')
    api_cost = sum(json.loads(row[0])['usage'].get('cost', 0) or 0 for row in db.execute('SELECT value FROM api_calls'))
    db.close()
    atomic_json(output / 'results.json', dict(arms=summary, comparisons=comparisons, decisions=decision,
                 api_reported_total_usd=api_cost,
                 caveats=['llm_native uses original Mem0 wording; llm uses the common renderer as a generation control.',
                          'Cached API latency is not a measured uncached end-to-end latency.',
                          'Reference USD is a token-based estimate, not a proxy invoice.',
                          'Synthetic training data and no semantic-axis supervision in this first version.']))
    rows = ['# Mem0 갱신 결정 실험 결과', '',
            'LongMemEval-S 전체 500문항. llm_native는 원본 생성 경로, llm은 공통 생성기를 쓰는 통제 기준군이다.', '',
            '| 조건 | 전체 QA | 지식 갱신 | 시간 추론 |', '| --- | ---: | ---: | ---: |']
    for name, val in summary.items():
        rows.append(f'| {name} | {val["accuracy"]:.3f} | {val["by_type"]["knowledge-update"]["accuracy"]:.3f} | {val["by_type"]["temporal-reasoning"]["accuracy"]:.3f} |')
    rows += ['', '상황추적기 − 작은 분류기: ' + json.dumps(comparisons['stateful_minus_stateless']),
             '', '전체 재현 설정, 비용 분해, seed별 결과와 판정은 results.json에 수록했다.',
             '캐시가 포함된 실행 시간은 실제 서비스의 비캐시 지연시간으로 해석하지 않는다.']
    (output / 'RESULTS.md').write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/experiments/mem0_update_oss_s0')
    summarize(ap.parse_args().output)
