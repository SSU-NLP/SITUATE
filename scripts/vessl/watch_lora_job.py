"""Collect the artifacts from a single submitted LoRA job when it terminates."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument('--job', required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--prefix', required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
cli = '/home/namchailin/.local/bin/vesslctl'
terminal = {'succeeded', 'failed', 'terminated', 'deleted', 'cancelled', 'canceled'}
while True:
    try:
        jobs = json.loads(subprocess.check_output([cli, 'job', 'list', '-o', 'json'], text=True, timeout=60))
        job = next(j for j in jobs if j['slug'] == a.job)
        state = job['workloadState']
        (a.out / 'job_status.json').write_text(json.dumps({'job': a.job, 'state': state, 'checked_at': time.time()}, indent=2))
        if state in terminal:
            logs = subprocess.check_output([cli, 'job', 'logs', a.job, '--limit', '1000'], text=True, timeout=60)
            (a.out / 'terminal.log').write_text(logs)
            subprocess.run([cli, 'volume', 'download', 'objvol-gsvyr0eu87wt', str(a.out / 'artifacts'),
                            '--remote-prefix', a.prefix, '--overwrite'], check=True, timeout=300)
            candidates = list((a.out / 'artifacts').rglob('metrics.json'))
            if state == 'succeeded':
                assert len(candidates) == 1, 'Missing final metrics'
                result = json.loads(candidates[0].read_text())
                assert result['seed'] == 1 and result['epoch'] == 1 and result['n_narratives'] == 6
                (a.out / 'metrics.json').write_text(json.dumps(result, indent=2))
                report = a.out / 'REPORT.md'
                with report.open('a') as f:
                    f.write('\n최종 결과: ' + json.dumps(result['average']) + '\n')
            break
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:300], flush=True)
    time.sleep(60)
