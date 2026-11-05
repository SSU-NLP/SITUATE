#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hi_ontop.memory_update.common import atomic_json

CLI = '/home/namchailin/.local/bin/vesslctl'
VOLUME = 'objvol-gsvyr0eu87wt'
IMAGE = 'quay.io/vessl-ai/torch@sha256:8bd418d49eeed841a607582b614c9d3ba86590b07d1be671816176c47253344a'
TERMINAL = {'succeeded', 'failed', 'terminated', 'deleted', 'cancelled', 'canceled'}


def call(args, timeout=180):
    return subprocess.run([CLI, *args], check=True, text=True, capture_output=True, timeout=timeout).stdout


def jobs():
    result = []
    for page in range(1, 101):
        batch = json.loads(call(['job', 'list', '-o', 'json', '--page', str(page), '--per-page', '100']))
        if not isinstance(batch, list):
            raise ValueError('Unexpected VESSL list response')
        result += batch
        if len(batch) < 100:
            return result
    raise RuntimeError('Job listing exceeded pagination bound; cannot establish safe capacity')


def capacity(current):
    active = [j for j in current if j['workloadState'].lower() not in TERMINAL]
    # Count reservations, including queued/starting jobs, and fail closed on unknown resources.
    counts = [int(j['resourceSpecAcceleratorCount']) for j in active]
    return len(active), sum(counts)


def stage(output):
    manifest = {}
    paths = [ROOT / 'src/hi_ontop/factored_slot.py', ROOT / 'configs/mem0_update_v1.json',
             ROOT / 'scripts/vessl/job_memory_update.sh', output / 'training.pt']
    paths += list((ROOT / 'src/hi_ontop/memory_update').glob('*.py'))
    paths += list((ROOT / 'scripts/mem0_update').glob('*.py'))
    for path in paths:
        key = 'training.pt' if path == output / 'training.pt' else str(path.relative_to(ROOT))
        manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    prefix = f'hiontop/mem0-update-v1/{revision}'
    with tempfile.TemporaryDirectory(prefix='mem0-upload-') as temporary:
        temp = Path(temporary)
        for source in paths:
            destination = temp / ('training.pt' if source == output / 'training.pt' else source.relative_to(ROOT))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (temp / 'src/hi_ontop/__init__.py').touch()
        atomic_json(temp / 'manifest.json', manifest)
        call(['volume', 'upload', VOLUME, str(temp), '--remote-prefix', prefix + '/code/'], timeout=600)
    return prefix, manifest


def run(output):
    cfg = json.loads((ROOT / 'configs/mem0_update_v1.json').read_text())
    with (output / 'vessl.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'vessl_state.json'
        if path.exists():
            state = json.loads(path.read_text())
        else:
            summary = json.loads((output / 'training_summary.json').read_text())
            if summary['partial']:
                raise ValueError('Refusing GPU training on a partial smoke dataset')
            prefix, manifest = stage(output)
            state = dict(prefix=prefix, manifest=manifest, tasks=[dict(arm=a, status='pending') for a in ('stateless', 'stateful')])
            atomic_json(path, state)
        while True:
            current = jobs()
            by_name = {j['name']: j for j in current}
            for task in state['tasks']:
                name = f'mem0-update-{task["arm"]}-{state["prefix"].split("/")[-1]}'
                existing = by_name.get(name)
                if existing:
                    task.update(slug=existing['slug'], status=existing['workloadState'].lower(), name=name)
                    if task['status'] in TERMINAL and task['status'] != 'succeeded':
                        atomic_json(path, state)
                        raise RuntimeError(f'Job failed: {name}; no automatic expensive rerun')
                elif task['status'] != 'pending':
                    raise RuntimeError(f'Previously submitted job missing: {name}')
            active_jobs, active_gpus = capacity(current)
            for task in state['tasks']:
                if task['status'] != 'pending' or active_jobs >= 2 or active_gpus >= 2:
                    continue
                prefix = state['prefix']
                name = f'mem0-update-{task["arm"]}-{prefix.split("/")[-1]}'
                # Persist intent before submission; the unique name recovers a lost response.
                task.update(name=name)
                atomic_json(path, state)
                response = call(['job', 'create', '-n', name, '-r', 'resourcespec-a100x1', '-i', IMAGE,
                                 '--object-volume', f'{VOLUME}:/shared', '-e', f'CODE=/shared/{prefix}/code',
                                 '-e', f'OUT=/shared/{prefix}/{task["arm"]}', '-e', f'ARM={task["arm"]}',
                                 '--cmd', f'bash /shared/{prefix}/code/scripts/vessl/job_memory_update.sh', '-o', 'json'])
                obj = json.loads(response)
                task.update(slug=obj['slug'], status=obj.get('workloadState', 'pending').lower())
                active_jobs += 1
                active_gpus += 1
                atomic_json(path, state)
                print(f'submitted {name}: {task["slug"]}', flush=True)
            atomic_json(path, state)
            for task in state['tasks']:
                if task['status'] == 'succeeded' and not task.get('collected'):
                    remote = state['prefix'] + '/' + task['arm']
                    target = output / 'remote' / task['arm']
                    target.mkdir(parents=True, exist_ok=True)
                    call(['volume', 'download', VOLUME, str(target), '--remote-prefix', remote + '/'], timeout=600)
                    for source in target.rglob('best.pt'):
                        destination = output / 'models' / source.parent.name
                        if destination.exists():
                            raise RuntimeError(f'Refusing to overwrite local checkpoint {destination}')
                        shutil.copytree(source.parent, destination)
                    for seed in cfg['seeds']:
                        assert (output / 'models' / f'{task["arm"]}_s{seed}' / 'complete.json').exists()
                    task['collected'] = True
                    atomic_json(path, state)
            if all(t.get('collected') for t in state['tasks']):
                return
            print(f'waiting: active jobs={active_jobs}, GPUs={active_gpus}', flush=True)
            time.sleep(45)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/experiments/mem0_update_oss_s0')
    run(ap.parse_args().output)
