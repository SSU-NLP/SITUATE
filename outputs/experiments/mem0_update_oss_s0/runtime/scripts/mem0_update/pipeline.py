#!/usr/bin/env python3
"""Durable, resumable experiment driver. GPU jobs are submitted only after preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hi_ontop.memory_update.common import atomic_json


def run(output, env):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logs = output / 'logs'
        logs.mkdir(exist_ok=True)
        phase = 'preflight'

        def status(state, **extra):
            atomic_json(output / 'status.json', dict(state=state, phase=phase, pid=os.getpid(),
                        updated_at=datetime.now(timezone.utc).isoformat(), **extra))

        def command(stage, *extra):
            return [sys.executable, str(ROOT / 'scripts/mem0_update/run.py'), stage,
                    '--output', str(output), '--env', str(env), *map(str, extra)]

        def execute(cmd, label):
            with (logs / f'{label}.log').open('a') as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONPATH': str(ROOT / 'src')})

        try:
            cfg = json.loads((ROOT / 'configs/mem0_update_v1.json').read_text())
            tuning = json.loads((output / 'workers.json').read_text())
            if tuning['model'] != cfg['llm'] or tuning['provider'] != cfg['llm_extra_body']['provider']:
                raise ValueError('Worker probe belongs to a different backend')
            workers = int(tuning['workers'])
            if workers < 1:
                raise ValueError('Invalid worker count')
            status('running', workers=workers)
            execute([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_memory_update.py')], 'preflight_tests')
            phase = 'calibrate_generation'
            status('running')
            execute(command('synth', '--limit', 4, '--workers', workers), 'synthesis_preflight')
            phase = 'calibrate_teacher_traces'
            status('running')
            execute(command('collect', '--limit', 4, '--workers', workers), 'collect_preflight')
            phase = 'generate_training_conversations'
            status('running')
            execute(command('synth', '--workers', min(workers, 16)), 'synthesis')
            phase = 'audit_data'
            status('running')
            execute(command('audit'), 'data_audit')
            phase = 'collect_teacher_traces'
            status('running')
            execute(command('collect', '--workers', workers), 'collect')
            phase = 'pack_training'
            status('running')
            execute(command('pack'), 'pack')
            phase = 'vessl_training'
            status('running')
            execute([sys.executable, str(ROOT / 'scripts/mem0_update/vessl_train.py'), '--output', str(output)], 'vessl')
            phase = 'uncached_latency'
            status('running')
            if not (output / 'latency.json').exists():
                execute([sys.executable, str(ROOT / 'scripts/mem0_update/latency.py'), '--output', str(output),
                         '--env', str(env)], 'latency')
            phase = 'official_qa'
            cfg = json.loads((ROOT / 'configs/mem0_update_v1.json').read_text())
            phase = 'official_qa_native_mem0'
            status('running')
            execute(command('evaluate', '--arm', 'llm', '--native', '--workers', workers), 'eval_llm_native')
            for arm in ('llm', 'stateless', 'stateful'):
                for seed in ([0] if arm == 'llm' else cfg['seeds']):
                    phase = f'official_qa_{arm}_s{seed}'
                    status('running')
                    execute(command('evaluate', '--arm', arm, '--seed', seed, '--workers', workers), f'eval_{arm}_s{seed}')
            phase = 'judge'
            status('running')
            execute(command('judge'), 'judge')
            phase = 'state_reset_diagnostic'
            status('running')
            execute(command('evaluate', '--arm', 'stateful', '--seed', 0, '--reset-state', '--limit', 32, '--workers', workers), 'state_reset')
            execute(command('judge', '--limit', 32), 'state_reset_judge')
            phase = 'report'
            status('running')
            execute([sys.executable, str(ROOT / 'scripts/mem0_update/report.py'), '--output', str(output)], 'report')
            status('complete')
        except BaseException as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def freeze(output):
    runtime = output / 'runtime'
    if runtime.exists():
        raise RuntimeError('Runtime already frozen; resume the existing snapshot')
    paths = [ROOT / 'src/hi_ontop/factored_slot.py', ROOT / 'configs/mem0_update_v1.json',
             ROOT / 'scripts/vessl/job_memory_update.sh', ROOT / 'tests/test_memory_update.py']
    paths += list((ROOT / 'src/hi_ontop/memory_update').glob('*.py'))
    paths += list((ROOT / 'scripts/mem0_update').glob('*.py'))
    manifest = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (runtime / 'src/hi_ontop/__init__.py').touch()
    atomic_json(runtime / 'manifest.json', manifest)
    return runtime


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/experiments/mem0_update_oss_s0')
    ap.add_argument('--env', type=Path, default=ROOT / 'EACL/.env')
    ap.add_argument('--freeze', action='store_true')
    args = ap.parse_args()
    if args.freeze:
        print(freeze(args.output))
    else:
        run(args.output, args.env)
