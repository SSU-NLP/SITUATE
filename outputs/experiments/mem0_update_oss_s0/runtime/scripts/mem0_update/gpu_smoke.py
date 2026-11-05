#!/usr/bin/env python3
"""Validate the frozen VESSL image with actual trace tensors before full training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from hi_ontop.memory_update.model import UpdatePolicy


def main(data_path, out):
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    blob = data_path.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'data.pt'
        path.write_bytes(blob)
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        records = []
        for stateful in (False, True):
            torch.manual_seed(0)
            cfg = bundle['config']
            model = UpdatePolicy(cfg['embedding_dim'], cfg['width'], cfg['slot_dim'], stateful=stateful).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
            state = None
            losses = []
            for session in bundle['data'][0]['sessions']:
                state = model.observe(session['turns'].cuda(), state)
                if len(session['facts']):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _ = model.loss(session['facts'].cuda(), session['memories'].cuda(), state, session['actions'])
                    loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    optimizer.step()
                    losses.append(float(loss.detach()))
                state = state.detach() if state is not None else None
            assert losses and all(torch.isfinite(torch.tensor(losses)))
            checkpoint = Path(tmp) / 'checkpoint.pt'
            torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False))
            records.append(dict(stateful=stateful, steps=len(losses), first_loss=losses[0], last_loss=losses[-1]))
    result = dict(passed=True, torch=torch.__version__, python=sys.version,
                  gpu=torch.cuda.get_device_name(0), training_sha256=hashlib.sha256(blob).hexdigest(), checks=records)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    main(args.data, args.out)
