"""Resume failed movies and reconcile the final complete-case inventory."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_full_api import ROOT, write_json


def movie_is_running(movie_id):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if movie_id in args and any(arg.endswith('/run_tracking.py') for arg in args):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    lock = (run / 'recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT)
    os.environ.update(STAGE_USAGE_LEDGER=str(run / 'usage.jsonl'),
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    state = {'pid': os.getpid(), 'status': 'running', 'attempts': {}, 'recovered': []}
    if (run / 'recovery_progress.json').exists():
        previous = json.loads((run / 'recovery_progress.json').read_text())
        state['attempts'] = previous.get('attempts', {})
        state['recovered'] = previous.get('recovered', [])
    methods = ('rewrite', 'rewrite_axes')
    inputs = ROOT / 'data/stage_task1_inputs'
    movies = json.loads((inputs / 'manifest.json').read_text())['movies']
    def complete(mid):
        return all((run / 'evaluation' / mid / method / 'summary.json').exists() for method in methods)
    began = time.time()
    while time.time() - began < 43200:
        progress = json.loads((run / 'progress.json').read_text())
        for failure in progress['failed']:
            mid = failure['movie_id']
            if mid == 'unknown' or movie_is_running(mid):
                continue
            if complete(mid):
                if mid not in state['recovered']:
                    state['recovered'].append(mid)
                continue
            if state['attempts'].get(mid, 0) >= 2:
                continue
            state['attempts'][mid] = state['attempts'].get(mid, 0) + 1
            state['current_movie'] = mid
            write_json(run / 'recovery_progress.json', state)
            common = ['--evaluation', str(ROOT / 'benchmarks/STAGE_v0/evaluation'),
                      '--inputs', str(inputs), '--config', str(run / 'config.json'),
                      '--num-shards', '1', '--movie-id', mid]
            commands = [
                [sys.executable, 'scripts/stage/run_tracking.py', *common, '--axes', str(run / 'axes'),
                 '--output', str(run / 'predictions'), '--workers', '16'],
                [sys.executable, 'scripts/stage/run_evaluation.py', *common,
                 '--predictions', str(run / 'predictions'), '--output', str(run / 'evaluation')]]
            with (run / 'logs' / f'recovery_{mid}.log').open('a') as log:
                for command in commands:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        break
            if complete(mid):
                state['recovered'].append(mid)
            state['current_movie'] = None
            write_json(run / 'recovery_progress.json', state)
        if progress['status'] != 'running':
            missing = [m['movie_id'] for m in movies if not complete(m['movie_id'])]
            if missing and any(state['attempts'].get(mid, 0) < 2 for mid in missing):
                time.sleep(15)
                continue
            if not missing:
                result = subprocess.run([sys.executable, 'scripts/stage/summarize.py',
                    '--inputs', str(inputs), '--evaluation', str(run / 'evaluation'),
                    '--output', str(run / 'summary.json')])
                if result.returncode == 0:
                    progress.update(status='completed', completed=[m['movie_id'] for m in movies],
                                    failed=[], finished_at=time.time())
                    write_json(run / 'progress.json', progress)
                    state['status'] = 'completed'
                else:
                    state['status'] = 'aggregation_failed'
            else:
                state.update(status='needs_attention', missing=missing)
            write_json(run / 'recovery_progress.json', state)
            return
        time.sleep(15)
    state['status'] = 'watch_timeout'
    write_json(run / 'recovery_progress.json', state)


if __name__ == '__main__':
    main()
