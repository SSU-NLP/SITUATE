"""Submit the one-GPU training job only after the embedding job succeeds."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--embedding-job', required=True)
p.add_argument('--state-dir', type=Path, required=True)
p.add_argument('--embedding-output', default='/shared/hiontop/qwen8b-20260925/embeddings')
a = p.parse_args()
a.state_dir.mkdir(parents=True, exist_ok=True)
lock = a.state_dir / 'chain.lock'
with lock.open('x') as f:
    f.write(a.embedding_job + '\n')
last = None
while True:
    try:
        r = subprocess.run(['vesslctl', 'job', 'list', '-o', 'json'],
                           capture_output=True, text=True, check=True, timeout=60)
        rows = json.loads(r.stdout)
        state = next(row['workloadState'] for row in rows if row['slug'] == a.embedding_job)
    except (subprocess.SubprocessError, ValueError, KeyError, StopIteration) as e:
        print(f'Status lookup failed: {type(e).__name__}', flush=True)
        time.sleep(30)
        continue
    if state != last:
        print(f'Embedding job {a.embedding_job}: {state}', flush=True)
        (a.state_dir / 'embedding_state.json').write_text(json.dumps({'job': a.embedding_job, 'state': state}))
        last = state
    if state == 'succeeded':
        break
    if state in ('failed', 'terminated', 'cancelled', 'canceled', 'deleted'):
        raise SystemExit('Embedding did not succeed; no training job submitted')
    time.sleep(30)
cmd = ['vesslctl', 'job', 'create', '-n', 'hiontop-qwen8b-train-s1-20260925',
       '-r', 'resourcespec-a100x1',
       '-i', 'quay.io/vessl-ai/torch@sha256:8bd418d49eeed841a607582b614c9d3ba86590b07d1be671816176c47253344a',
       '--object-volume', 'objvol-gsvyr0eu87wt:/shared',
       '-e', 'CODE=/shared/hiontop/qwen8b-20260925/code',
       '-e', 'OUT=/shared/hiontop/qwen8b-20260925/train-seed1',
       '-e', f'EMB={a.embedding_output}',
       '-e', 'STAGE=train',
       '--cmd', 'bash /shared/hiontop/qwen8b-20260925/code/scripts/vessl/job_qwen8b_backbone.sh']
# Do not retry an ambiguous submission response, to avoid duplicate GPU jobs.
(a.state_dir / 'training_submission_started.json').write_text(json.dumps({'command': cmd}))
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
(a.state_dir / 'training_submission.txt').write_text(r.stdout + r.stderr)
print(r.stdout, flush=True)
r.check_returncode()
m = re.search(r'job-[a-z0-9]+', r.stdout + r.stderr)
if not m:
    raise SystemExit('Submission accepted but slug could not be parsed; inspect submission log')
(a.state_dir / 'training_job.json').write_text(json.dumps({'slug': m.group(0)}))
print('Training job submitted; six-narrative scores will be saved to train-seed1.', flush=True)
