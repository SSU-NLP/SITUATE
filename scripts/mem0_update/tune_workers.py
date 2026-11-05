#!/usr/bin/env python3
"""Measure API concurrency with invented examples, no benchmark or user content."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import threading
import time

import httpx
from dotenv import dotenv_values
from openai import OpenAI, APIStatusError

ROOT = Path(__file__).resolve().parents[2]


def quantile(values, fraction):
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]


def invented_prompt(index):
    memories = [dict(id=str(i), text=f'Fictional person P{i} lives in City{i}, works on project Garden{i}, '
                     f'and prefers tea blend T{i}. The project deadline is October {i+1}, 2024.') for i in range(20)]
    facts = [f'Fictional person P{i} moved to NewCity{i} and changed the deadline to November {i+1}, 2024.' for i in range(5)]
    return ('This is an invented load-test fixture, not a real conversation. '
            'Return JSON {"memory":[{"id":string,"event":"ADD|UPDATE|DELETE|NONE","text":string}]}. '
            'Include each of the 20 existing memories exactly once. Update only the changed entries; '
            'preserve the others with NONE. Never invent target IDs. Preserve compatible old details. '
            'Return only the JSON. Test case number: ' + str(index) + '\n' +
            json.dumps(dict(memories=memories, facts=facts)))


def run(output, levels, env_file, config_path, cost_limit, timeout):
    cfg = json.loads(config_path.read_text())
    env = dotenv_values(env_file)
    headers = {k.replace('_', '-'): env[k] for k in ('CF_ACCESS_CLIENT_ID', 'CF_ACCESS_CLIENT_SECRET') if env.get(k)}
    client = OpenAI(api_key=env['OPENAI_API_KEY'], base_url=env['OPENAI_BASE_URL'], default_headers=headers,
                    max_retries=0, timeout=timeout,
                    http_client=httpx.Client(limits=httpx.Limits(max_connections=max(levels),
                                                               max_keepalive_connections=max(levels))))
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    records_path = output / f'worker_probe_{stamp}.jsonl'
    stages, total_cost, plateau = [], 0.0, 0
    run_nonce = time.time_ns()
    for workers in levels:
        if total_cost >= cost_limit:
            reason = 'probe_cost_limit'
            break
        halt = threading.Event()

        def one(index):
            if halt.is_set():
                return dict(skipped=True)
            start = time.perf_counter()
            try:
                response = client.chat.completions.create(model=cfg['llm'],
                    messages=[dict(role='user', content=invented_prompt(f'{run_nonce}-{workers}-{index}'))],
                    temperature=0, max_tokens=4096, response_format={'type': 'json_object'},
                    extra_body=cfg['llm_extra_body'])
                usage = response.usage.model_dump()
                valid = False
                try:
                    valid = isinstance(json.loads(response.choices[0].message.content).get('memory'), list)
                except (ValueError, TypeError, AttributeError):
                    pass
                return dict(seconds=time.perf_counter()-start, status=200,
                            success=valid and response.choices[0].finish_reason == 'stop',
                            finish_reason=response.choices[0].finish_reason,
                            input_tokens=usage['prompt_tokens'], output_tokens=usage['completion_tokens'],
                            reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0),
                            cost=usage.get('cost', 0), provider=response.model_dump().get('provider'))
            except APIStatusError as exc:
                if exc.status_code == 429 or exc.status_code >= 500:
                    halt.set()  # Respect throttling; do not queue another wave after overload.
                return dict(seconds=time.perf_counter()-start, status=exc.status_code, success=False,
                            retry_after=exc.response.headers.get('retry-after'))
            except Exception as exc:
                halt.set()
                return dict(seconds=time.perf_counter()-start, success=False, error=type(exc).__name__)

        start = time.perf_counter()
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, i) for i in range(max(4, 2 * workers))]
            with records_path.open('a') as log:
                for future in as_completed(futures):
                    row = future.result()
                    if row.get('skipped'):
                        continue
                    rows.append(row)
                    log.write(json.dumps(dict(workers=workers, **row)) + '\n')
                    log.flush()
        elapsed = time.perf_counter() - start
        successes = [r for r in rows if r['success']]
        total_cost += sum(r.get('cost', 0) or 0 for r in rows)
        latencies = [r['seconds'] for r in successes]
        stage = dict(workers=workers, completed=len(rows), success=len(successes),
                     errors=len(rows)-len(successes), seconds=elapsed,
                     successful_requests_per_second=len(successes)/elapsed,
                     median_seconds=statistics.median(latencies) if latencies else None,
                     p95_seconds=quantile(latencies, .95), cumulative_cost_usd=total_cost)
        stages.append(stage)
        print(json.dumps(stage), flush=True)
        (output / 'worker_probe_progress.json').write_text(json.dumps(stages, indent=2))
        if stage['errors']:
            reason = 'first_error_or_overload'
            break
        if len(stages) >= 2:
            previous = stages[-2]
            gain = stage['successful_requests_per_second'] / previous['successful_requests_per_second']
            plateau = plateau + 1 if gain < 1.10 else 0
            if plateau >= 2:
                reason = 'throughput_plateau'
                break
    else:
        reason = 'tested_ceiling'
    safe = [s for s in stages if s['errors'] == 0]
    if not safe:
        raise RuntimeError('Even the baseline probe failed; no worker setting selected')
    best = max(safe, key=lambda s: s['successful_requests_per_second'])
    # Prefer the smallest worker count within 10% of the best successful throughput.
    chosen = next(s for s in safe if s['successful_requests_per_second'] >= .9 * best['successful_requests_per_second'])
    result = dict(workers=chosen['workers'], reason=reason, stages=stages, timeout_seconds=timeout,
                  cost_usd=total_cost, model=cfg['llm'], provider=cfg['llm_extra_body']['provider'],
                  fixture='invented 20-memory update; no benchmark/user content',
                  limitations='Short concurrency probe, not a sustained throughput guarantee', records=str(records_path))
    path = output / 'workers.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)
    print('SELECTED ' + json.dumps(dict(workers=result['workers'], reason=reason, cost_usd=total_cost)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/experiments/mem0_update_oss_s0')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/mem0_update_v1.json')
    ap.add_argument('--env', type=Path, default=ROOT / 'EACL/.env')
    ap.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument('--cost-limit', type=float, default=.5)
    ap.add_argument('--timeout', type=float, default=120)
    args = ap.parse_args()
    run(args.output, args.levels, args.env, args.config, args.cost_limit, args.timeout)
