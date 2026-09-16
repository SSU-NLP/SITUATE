#!/usr/bin/env python3
r"""지적 2번 — 인과 축의 정체 진단: 대사 밀도 × 축 발화율 상관.

주장: `reset_causality` 가 재는 것은 `event_type` 라벨 변화(분류학적 범주 전환)이며,
EVENT_TYPE 에 dialogue/physical_action/state_description 이 나란히 들어 있어
**대사↔행동↔묘사 교대 탐지기**로 작동한다는 가설. 발화율 0.6950 이 그 징후.

검정: 편별 **대사 밀도**(`*_master.meta.jsonl` 의 `dialogue_density`, 코퍼스 빌드 시 기록)와
문서별 축 발화율의 상관. 인과 축에서만 상관이 크면 정체 확정.

읽기 전용. 입력 = `anchor_axis_audit.py` 산출 `per_doc_axis_rates.json`.

    .venv/bin/python scripts/eacl_phase0/causality_identity_diag.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/experiments/2026-07-31_anchor_axis_audit"
META = REPO / "data/eacl_phase1/all_master.meta.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=META)
    a = ap.parse_args()

    rates = json.loads((OUT / "per_doc_axis_rates.json").read_text())
    axes, doc_ids = rates["axes"], rates["doc_ids"]
    pos = {d: i for i, d in enumerate(doc_ids)}

    dens, nsent, keep = [], [], []
    with a.meta.open() as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["doc_id"])
            if i is None or "dialogue_density" not in r:
                continue
            keep.append(i)
            dens.append(float(r["dialogue_density"]))
            nsent.append(int(r.get("n_sent", 0)))
    keep = np.asarray(keep)
    dens = np.asarray(dens)
    nsent = np.asarray(nsent, float)
    print(f"[data] 매칭된 문서 {len(keep):,} / 앵커 {len(doc_ids):,} · "
          f"대사 밀도 평균 {dens.mean():.4f} (min {dens.min():.3f} max {dens.max():.3f})",
          flush=True)

    res = {"n_docs": int(len(keep)), "dialogue_density_mean": float(dens.mean()),
           "pearson_r": {}, "spearman_r": {}, "by_tercile": {}}

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    print("\n=== 대사 밀도 × 문서별 축 발화율 ===")
    print(f"  {'축':16s} {'Pearson':>8s} {'Spearman':>9s}   대사밀도 3분위 발화율 (하 / 중 / 상)")
    q1, q2 = np.quantile(dens, [1 / 3, 2 / 3])
    grp = np.digitize(dens, [q1, q2])
    for k, ax in enumerate(axes):
        v = np.asarray(rates["rates"][ax])[keep]
        pr, sr = float(np.corrcoef(dens, v)[0, 1]), spearman(dens, v)
        ter = [float(v[grp == g].mean()) for g in range(3)]
        res["pearson_r"][ax], res["spearman_r"][ax], res["by_tercile"][ax] = pr, sr, ter
        print(f"  {ax:16s} {pr:8.3f} {sr:9.3f}   {ter[0]:.4f} / {ter[1]:.4f} / {ter[2]:.4f}")

    # 통제: 문서 길이와의 상관 (길이 교란 배제용)
    print("\n=== 통제: 문서 길이(n_sent) × 축 발화율 ===")
    res["pearson_r_nsent"] = {}
    for k, ax in enumerate(axes):
        v = np.asarray(rates["rates"][ax])[keep]
        pr = float(np.corrcoef(nsent, v)[0, 1])
        res["pearson_r_nsent"][ax] = pr
        print(f"  {ax:16s} {pr:8.3f}")

    (OUT / "causality_identity.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'causality_identity.json'}")


if __name__ == "__main__":
    main()
