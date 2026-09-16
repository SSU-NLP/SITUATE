#!/usr/bin/env python3
"""figure_02 — M5 축별 전이 확률 분포가 서사↔대화에서 어떻게 달라지는가.

`dump_axis_probs.py` 산출 JSONL 을 읽어 **축 × 데이터셋** 격자로 gold=1 / gold=0 의
확률 분포를 겹쳐 그린다. 각 칸에 AUC 를 적어 "순위 분리력" 과 "분포 자체" 를 함께 본다.

디자인 기준 = `EACL/figures/좋은figure예시.jpg` (제목 없음, 격자 없음, 위쪽 spine 제거,
범례는 그림 아래 중앙·테두리 없음). PDF+PNG 300dpi, fonttype 42.

    env -C /tmp /root/Hi-OnTop/.venv/bin/python \\
        /root/Hi-OnTop/scripts/eacl_phase0/plot_axis_collapse.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from hi_ontop.factored_slot import AXES              # noqa: E402

SRC = REPO / "outputs/experiments/2026-08-04_axis_collapse"
FIG = REPO / "EACL/figures"

SETS = [
    ("dev3", "Narrative dev-3"),
    ("test3", "Narrative test-3"),
    ("dts_full_tiage", "TIAGE"),
    ("dts_full_dialseg711", "DialSeg711"),
    ("dts_full_superseg", "SuperDialseg"),
]
LABEL = {"protagonist": "Protagonist", "space": "Space", "time": "Time",
         "causality": "Causality", "intentionality": "Intentionality"}
C_POS, C_NEG = "#C44E52", "#4C72B0"      # gold=1 / gold=0


def load(name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = [json.loads(l) for l in (SRC / f"axis_probs_{name}.jsonl").read_text().splitlines()]
    g = np.asarray([r["gold"] for r in rows], bool)
    return {a: (np.asarray([r[f"p_{a}"] for r in rows])[g],
                np.asarray([r[f"p_{a}"] for r in rows])[~g]) for a in AXES}


def main() -> int:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    summ = json.loads((SRC / "axis_collapse_summary.json").read_text())["sets"]
    data = {n: load(n) for n, _ in SETS}

    fig, axes = plt.subplots(len(AXES), len(SETS), figsize=(10.5, 8.6),
                             sharex=True, constrained_layout=True)
    bins = np.linspace(0, 1, 41)
    for r, ax_name in enumerate(AXES):
        for c, (sname, stitle) in enumerate(SETS):
            ax = axes[r, c]
            pos, neg = data[sname][ax_name]
            # density=True — gold=1 이 훨씬 드물어 raw count 로는 비교가 안 된다
            hn, _, _ = ax.hist(neg, bins=bins, density=True, color=C_NEG, alpha=0.55, lw=0)
            hp, _, _ = ax.hist(pos, bins=bins, density=True, color=C_POS, alpha=0.55, lw=0)
            # 🔴 time 축처럼 확률이 0 근처에 몰리면 첫 bin 하나가 나머지를 전부 눌러 버린다.
            # 첫 bin 을 뺀 최댓값으로 상한을 잡아 꼬리가 보이게 한다(잘린 막대는 위로 넘어감).
            tail = float(max(hn[1:].max(initial=0.0), hp[1:].max(initial=0.0)))
            if tail > 0 and max(hn[0], hp[0]) > 3 * tail:
                ax.set_ylim(0, tail * 1.15)
            auc = summ[sname]["axes"][ax_name]["auc"]
            dmean = summ[sname]["axes"][ax_name]["delta"]
            # AUC < 0.5 = 역상관. 굵게 표시해 눈에 띄게 한다.
            ax.text(0.97, 0.93, f"AUC {auc:.3f}\n$\\Delta${dmean:+.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                    fontweight="bold" if auc < 0.5 else "normal",
                    color="#B22222" if auc < 0.5 else "0.25")
            ax.set_yticks([])
            if r == 0:
                ax.set_title(stitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(LABEL[ax_name], fontsize=9)
            if r == len(AXES) - 1:
                ax.set_xlabel("axis change probability", fontsize=7.5)
            ax.set_xlim(0, 1)

    h = [plt.Rectangle((0, 0), 1, 1, fc=C_NEG, alpha=0.55, lw=0),
         plt.Rectangle((0, 0), 1, 1, fc=C_POS, alpha=0.55, lw=0)]
    fig.legend(h, ["no boundary (gold $=0$)", "human boundary (gold $=1$)"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.028))

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG / f"figure_02_axis_collapse.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
