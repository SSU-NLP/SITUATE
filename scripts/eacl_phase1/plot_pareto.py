#!/usr/bin/env python3
"""Figure S — 코퍼스 구성 방법 비교. 논문용.

메시지 하나: **같은 저자 다양성에서 층화 방식이 구성 왜곡을 3분의 1로 줄인다.**
곡선 교차와 레이블 충돌을 피하려고 막대 2개로만 그린다.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/2026-08-02_author_cap"
R = json.loads((EXP / "pareto.json").read_text())
A = next(r for r in R if r["method"] == "plain" and r["N"] == 16)
B = next(r for r in R if r["method"] == "strat" and r["N"] == 16)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})

fig, a = plt.subplots(figsize=(6.2, 4.0))
labels = ["A. author cap only\n(current)", "B. stratified +\nwithin-class cap"]
vals = [A["shift"], B["shift"]]
bars = a.bar(labels, vals, width=0.5, color=[ORANGE, BLUE], zorder=3)
for r, v in zip(bars, vals):
    a.text(r.get_x() + r.get_width() / 2, v + 0.18, f"{v:.2f} pp",
           ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)

a.annotate("", xy=(1.42, B["shift"]), xytext=(1.42, A["shift"]),
           arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=4)
a.plot([0, 1.42], [A["shift"]] * 2, color=MUT, lw=0.9, ls=(0, (2, 2)), zorder=1)
a.plot([1, 1.42], [B["shift"]] * 2, color=MUT, lw=0.9, ls=(0, (2, 2)), zorder=1)
a.text(1.52, (A["shift"] + B["shift"]) / 2,
       f"−{A['shift'] - B['shift']:.1f} pp\n(to one\nthird)", va="center", ha="left",
       fontsize=10.5, color=INK, fontweight="bold")

a.set_ylabel("composition distortion (pp)\nmax class shift vs full pool")
a.set_title("Same author diversity, one third the distortion", loc="left", color=INK)
a.set_ylim(0, 8.4)
a.set_xlim(-0.55, 2.25)
a.grid(axis="y", color=GRID, lw=0.9, zorder=0)
a.set_axisbelow(True)
a.tick_params(axis="x", length=0)
a.text(0.5, -0.30,
       f"both matched at ≈1,000 effective authors  "
       f"(A: {A['eff']:.0f} from {A['pieces']:,} pieces · B: {B['eff']:.0f} from {B['pieces']:,})",
       transform=a.transAxes, ha="center", va="top", fontsize=9, color=MUT)

fig.tight_layout()
out = REPO / "outputs/figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"figure_S_corpus_construction.{ext}", dpi=300, bbox_inches="tight")
print(f"A {A['shift']:.2f}pp / B {B['shift']:.2f}pp  ->",
      out / "figure_S_corpus_construction.png")
