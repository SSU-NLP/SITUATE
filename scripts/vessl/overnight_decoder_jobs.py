"""Run two decoder backbone experiments with at most four active VESSL jobs."""
import argparse
import csv
import fcntl
import io
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import time
import urllib.request
import urllib.error

CLI = '/home/namchailin/.local/bin/vesslctl'
VOLUME = 'objvol-gsvyr0eu87wt'
PREFIX = 'hiontop/decoder-e10-s4-20260926'
ROOT = '/shared/' + PREFIX
IMAGE = 'quay.io/vessl-ai/torch@sha256:8bd418d49eeed841a607582b614c9d3ba86590b07d1be671816176c47253344a'
TERMINAL = {'succeeded', 'failed', 'terminated', 'deleted', 'cancelled', 'canceled'}
METRICS = ('score', 'bor', 'lift')


def gemma_access_status():
    """Check the explicitly selected project credential without logging its value."""
    path = Path('/home/namchailin/Projects/Hi-OnTop/EACL/.env')
    token = None
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix('export ')
        key, separator, value = line.partition('=')
        if separator and key.strip() == 'HF_KEY':
            parts = shlex.split(value, comments=True)
            token = parts[0].strip() if parts else None
    if not token:
        return 'missing_project_token'
    request = urllib.request.Request(
        'https://huggingface.co/google/gemma-2-9b/resolve/main/config.json',
        headers={'Authorization': 'Bearer ' + token})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read())
            return 'authorized' if data.get('hidden_size') == 3584 else 'unexpected_config'
    except urllib.error.HTTPError as exc:
        return f'http_{exc.code}'
    except (OSError, ValueError):
        return 'lookup_unavailable'


def call(args, timeout=120):
    return subprocess.run([CLI] + args, capture_output=True, text=True,
                          timeout=timeout, check=True).stdout


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def initial_tasks():
    tasks = [{'key': 'gemma9b-embed', 'backbone': 'gemma9b', 'stage': 'embed'}]
    for backbone in ('qwen17b', 'gemma9b'):
        tasks += [{'key': f'{backbone}-s{seed}', 'backbone': backbone,
                   'stage': 'train', 'seed': seed} for seed in range(4)]
    for task in tasks:
        task.update(status='pending', attempt=0)
    return tasks


def task_command(task, tasks):
    key, attempt = task['key'], task['attempt']
    output = f'{ROOT}/{key}/attempt{attempt}'
    name = f'hiontop-dec-e10-{key}-20260926-a{attempt}'
    args = ['job', 'create', '-n', name, '-r',
            'resourcespec-a100x4' if task['stage'] == 'embed' else 'resourcespec-a100x1',
            '-i', IMAGE, '--object-volume', f'{VOLUME}:/shared',
            '-e', f'CODE={ROOT}/code', '-e', f'OUT={output}',
            '-e', f'STAGE={task["stage"]}', '-e', f'BACKBONE={task["backbone"]}',
            '-e', f'SEED={task.get("seed", 0)}',
            '--cmd', f'bash {ROOT}/code/scripts/vessl/job_decoder_multiseed.sh']
    if task['stage'] == 'embed':
        args += ['--cluster-volume', 'clustervol-gashapuedsj4:/legacy',
                 '--secret', 'HF_TOKEN=HIONTOP_HF_CHAILIN']
    else:
        embedding = ('/shared/hiontop/qwen17b-20260925/embeddings'
                     if task['backbone'] == 'qwen17b'
                     else next(t['output'] for t in tasks if t['key'] == 'gemma9b-embed'))
        args += ['-e', f'EMB={embedding}']
    task.update(name=name, output=output)
    return args


def eligible(task, tasks):
    if task['status'] != 'pending':
        return False
    if task['stage'] == 'train' and task['backbone'] == 'gemma9b':
        return next(t for t in tasks if t['key'] == 'gemma9b-embed')['status'] == 'succeeded'
    return True


def validate_metrics(result, task):
    assert result['seed'] == task['seed'] and result['backbone'] == task['backbone']
    assert result['epochs'] == 10 and result['n_narratives'] == 6
    rows = result['epoch_metrics']
    assert [r['epoch'] for r in rows] == list(range(1, 11))
    for row in rows:
        assert row['checkpoint'].endswith(f'_ep{row["epoch"]:03d}_seed{task["seed"]}.pt')
        assert len(row['per_narrative']) == 6
        for metric in METRICS:
            values = [v[metric] for v in row['per_narrative'].values()]
            value = row['average'][metric]
            if any(v is None for v in values):
                assert value is None
            else:
                assert math.isclose(value, statistics.mean(values), rel_tol=1e-9, abs_tol=1e-12)
    return result


def collect(task, directory):
    target = directory / 'results' / task['key'] / f'attempt{task["attempt"]}'
    target.mkdir(parents=True, exist_ok=True)
    call(['volume', 'download', VOLUME, str(target), '--remote-prefix',
          task['output'].removeprefix('/shared/'), '--exclude', '*.pt', '--overwrite'], timeout=300)
    paths = list(target.rglob('epoch_metrics.json'))
    assert len(paths) == 1, f'Expected one metrics file in {target}'
    result = validate_metrics(json.loads(paths[0].read_text()), task)
    task['metrics_path'] = str(paths[0])
    return result


def aggregate(tasks, directory):
    records = []
    for task in tasks:
        if 'metrics_path' not in task:
            continue
        data = validate_metrics(json.loads(Path(task['metrics_path']).read_text()), task)
        for row in data['epoch_metrics']:
            records.append(dict(backbone=task['backbone'], seed=task['seed'],
                                epoch=row['epoch'], **row['average']))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['backbone', 'seed', 'epoch', *METRICS])
    writer.writeheader()
    writer.writerows(records)
    (directory / 'per_seed_epoch_metrics.csv').write_text(stream.getvalue())
    summary = []
    for backbone in ('qwen17b', 'gemma9b'):
        for epoch in range(1, 11):
            group = [r for r in records if r['backbone'] == backbone and r['epoch'] == epoch]
            if len(group) != 4:
                continue
            row = {'backbone': backbone, 'epoch': epoch, 'n_seeds': 4}
            for metric in METRICS:
                vals = [g[metric] for g in group]
                row[metric + '_mean'] = statistics.mean(vals) if all(v is not None for v in vals) else None
                row[metric + '_std'] = statistics.stdev(vals) if all(v is not None for v in vals) else None
            summary.append(row)
    atomic_json(directory / 'four_seed_summary.json', summary)
    if summary:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
        (directory / 'four_seed_summary.csv').write_text(stream.getvalue())
    def fmt(value):
        return 'undefined' if value is None else f'{value:.4f}'
    lines = ['# Decoder backbone comparison', '',
             'Frozen backbones; seeds 0, 1, 2, 3; 10 epochs without early stopping.',
             'Each metric first averages all six narratives equally. Seed standard deviation uses ddof=1.',
             '', '| Backbone | Epoch | SCORE mean ± std | BOR mean ± std | LIFT mean ± std |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in summary:
        cells = [f'{fmt(row[m+"_mean"])} ± {fmt(row[m+"_std"])}' for m in METRICS]
        lines.append('| ' + ' | '.join([row['backbone'], str(row['epoch']), *cells]) + ' |')
    lines += ['', '## Per-seed results', '', '| Backbone | Seed | Epoch | SCORE | BOR | LIFT |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        lines.append('| ' + ' | '.join([row['backbone'], str(row['seed']), str(row['epoch']),
                                        *[fmt(row[m]) for m in METRICS]]) + ' |')
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.state_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'scheduler.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / 'run_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'tasks': initial_tasks()}
    tasks = state['tasks']
    def save():
        state['updated_at_unix'] = time.time()
        atomic_json(state_path, state)
    save()
    while True:
        prerequisite = tasks[0]
        if prerequisite['status'] == 'awaiting_access' and time.time() - prerequisite.get('access_checked_at', 0) >= 60:
            access = gemma_access_status()
            prerequisite['access_checked_at'] = time.time()
            if access != prerequisite.get('access_status'):
                print('Gemma project-token access:', access, flush=True)
                prerequisite['access_status'] = access
            if access == 'authorized':
                prerequisite['status'] = 'pending'
                print('Gemma access confirmed; resuming experiment queue', flush=True)
            save()
        try:
            jobs = json.loads(call(['job', 'list', '-o', 'json']))
        except (subprocess.SubprocessError, ValueError):
            print('Status lookup unavailable; no jobs submitted', flush=True)
            time.sleep(30)
            continue
        by_slug = {r['slug']: r for r in jobs}
        # Reconcile an interrupted submission by unique job name; never duplicate an uncertain request.
        for task in tasks:
            if task['status'] == 'submitting':
                matches = [r for r in jobs if r['name'] == task['name']]
                if len(matches) == 1:
                    task.update(slug=matches[0]['slug'], status='active')
                elif not matches and time.time() - task['submitted_at'] > 600:
                    task.update(status='blocked', error='Ambiguous submission; manual reconciliation required')
            if task['status'] == 'active' and task['slug'] in by_slug:
                observed = by_slug[task['slug']]['workloadState']
                if observed != task.get('observed'):
                    print(task['key'], task['slug'], observed, flush=True)
                    task['observed'] = observed
                if observed == 'succeeded':
                    task['status'] = 'succeeded'
                elif observed in TERMINAL:
                    try:
                        log = call(['job', 'logs', task['slug']])
                        (directory / f'{task["key"]}-a{task["attempt"]}.log').write_text(log)
                    except subprocess.SubprocessError:
                        log = ''
                    task.setdefault('history', []).append({'slug': task['slug'], 'state': observed, 'output': task['output']})
                    if observed == 'failed' and task['attempt'] < 1 and 'access preflight failed' not in log:
                        task.update(status='pending', attempt=task['attempt'] + 1)
                        task.pop('slug', None)
                        print('Retrying failed task once:', task['key'], flush=True)
                    else:
                        task.update(status='failed', error=observed)
            if task['status'] == 'succeeded' and task['stage'] == 'train' and 'metrics_path' not in task:
                try:
                    collect(task, directory)
                    aggregate(tasks, directory)
                    print('Collected and verified ten epoch results:', task['key'], flush=True)
                except (subprocess.SubprocessError, OSError, ValueError, AssertionError, KeyError) as exc:
                    print('Result collection needs retry:', task['key'], type(exc).__name__, flush=True)
        gemma = tasks[0]
        if gemma['status'] in ('failed', 'blocked'):
            for task in tasks:
                if task['backbone'] == 'gemma9b' and task['stage'] == 'train' and task['status'] == 'pending':
                    task.update(status='blocked', error='Gemma embedding prerequisite failed')
        save()
        # Count every nonterminal job in the account, including pending/provisioning jobs.
        # Reserve slots for submissions not yet visible in the job listing.
        active = sum(r['workloadState'] not in TERMINAL for r in jobs)
        active += sum(t['status'] == 'submitting' for t in tasks)
        active += sum(t['status'] == 'active' and t.get('slug') not in by_slug for t in tasks)
        for task in tasks:
            if active >= 4:
                break
            if eligible(task, tasks):
                command = task_command(task, tasks)
                task.update(status='submitting', submitted_at=time.time())
                save()
                try:
                    result = subprocess.run([CLI] + command, capture_output=True, text=True, timeout=120)
                    (directory / f'{task["key"]}-a{task["attempt"]}-submission.txt').write_text(result.stdout + result.stderr)
                    # Reconcile via job list on the next loop even when output is ambiguous.
                    print('Submission requested:', task['key'], 'returncode', result.returncode, flush=True)
                except subprocess.SubprocessError:
                    print('Submission response uncertain; reserving slot:', task['key'], flush=True)
                active += 1
        save()
        done = all(t['status'] in ('succeeded', 'failed', 'blocked') for t in tasks)
        collected = all(t['stage'] != 'train' or t['status'] != 'succeeded' or 'metrics_path' in t for t in tasks)
        if done and collected:
            aggregate(tasks, directory)
            state['complete'] = all(t['status'] == 'succeeded' for t in tasks)
            save()
            (directory / 'COMPLETION.txt').write_text('All experiments complete\n' if state['complete'] else 'Some experiments blocked or failed; inspect run_status.json\n')
            print('Scheduler finished; all successful:', state['complete'], flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
