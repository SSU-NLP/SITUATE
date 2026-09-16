#!/usr/bin/env python3
"""Figure R — 작가당 상한 N. 논문용. CLAUDE.md 규약: outputs/figures/figure_R_*.{pdf,png}

N = 1..48 을 빠짐없이 계산(48점). 판단 기준 둘:
  (a) 저자 다양성 — 유효 저자 수(inverse Simpson, 1/Σpᵢ²). 임의 컷오프 없음
  (b) 전체 풀과의 구성 차이 — **크기 맞춘 무작위 표본의 귀무 분포와 함께** 제시

🔴 (b)의 회색 띠가 이 그림의 통계적 핵심이다. 상한 코퍼스는 전체 풀의 결정적 부분집합이므로
   "전체와 같은가"를 독립 2표본 검정으로 물을 수 없다. 대신 답할 수 있는 질문:
   **"같은 크기의 무작위 표본에서 저절로 생기는 변화보다 큰가?"**
   귀무 = 작가 단위 무작위 추출로 같은 편 수를 만든 400회 재표집.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/2026-08-02_author_cap"
R = json.loads((EXP / "fine_grid_v2.json").read_text())
NT = {r["cap"]: r for r in json.loads((EXP / "null_test.json").read_text())["rows"]}

N = np.array([r["cap"] for r in R])
E = np.array([r["eff_auth"] for r in R])
V = np.array([r["even"] * 100 for r in R])
L = np.array([r["locc_pp"] for r in R])
P = np.array([r["n"] for r in R])
NUL95 = np.array([NT[int(n)]["null_p95"] for n in N])
NULMED = np.array([NT[int(n)]["null_med"] for n in N])
FULL_EFF, FULL_AUTH = 235.0, 2344

FLOOR = 50.0                                    # 🔒 선언한 설계 요구
pi = max((i for i in range(len(N)) if V[i] >= FLOOR), key=lambda i: N[i])
PICK = int(N[pi])

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUT, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e0", "#d9d8d2"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": MUT,
    "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))


def pick_marker(a, y, label, dx=11, dy=9):
    a.axvline(PICK, color=MUT, lw=1, ls=(0, (3, 3)), zorder=2)
    a.plot([PICK], [y], "o", ms=8, mfc="white", mec=INK, mew=1.9, zorder=6)
    a.annotate(label, (PICK, y), textcoords="offset points", xytext=(dx, dy),
               fontsize=9.5, color=INK, fontweight="bold", zorder=7)


# ── (a) 저자 다양성 ────────────────────────────────────────────────────────
a = ax[0]
a.axhline(FULL_AUTH, color=MUT, lw=1, ls=":", zorder=1)
a.text(47.5, FULL_AUTH - 75, "all 2,344 authors ", va="top", ha="right",
       color=MUT, fontsize=9)
a.axhline(FULL_EFF, color=MUT, lw=1, ls=":", zorder=1)
a.text(47.5, FULL_EFF + 60, "no cap: 235 ", va="bottom", ha="right",
       color=MUT, fontsize=9)
a.axhline(FULL_AUTH * FLOOR / 100, color=ORANGE, lw=1.4, ls="--", alpha=0.9, zorder=3)
a.text(47.5, FULL_AUTH * FLOOR / 100 + 60, f"design requirement: ≥ {FLOOR:.0f}% ",
       va="bottom", ha="right", color=ORANGE, fontsize=9)
a.plot(N, E, lw=2.2, color=BLUE, zorder=4)
pick_marker(a, E[pi], f"$N$={PICK}")
a.set_xlabel("author cap $N$  (pieces retained per author)")
a.set_ylabel("effective authors  $1/\\sum_i p_i^2$")
a.set_title("(a) Author diversity", loc="left", color=INK)
a.set_xlim(0, 49)
a.set_ylim(0, 2500)
a.grid(axis="y", color=GRID, lw=0.8)
a.set_axisbelow(True)

# ── (b) 구성 차이 + 귀무 띠 ───────────────────────────────────────────────
a = ax[1]
a.fill_between(N, 0, NUL95, color=BAND, alpha=0.75, lw=0, zorder=1,
               label="size-matched random subsets\n(author-clustered, 95th pct)")
a.plot(N, NULMED, color=MUT, lw=1.1, ls=":", zorder=3, label="random subsets, median")
a.plot(N, L, lw=2.2, color=ORANGE, zorder=4, label="author-capped corpus")
pick_marker(a, L[pi], f"$N$={PICK}\n{P[pi]:,} pieces", dx=11, dy=8)
a.set_xlabel("author cap $N$  (pieces retained per author)")
a.set_ylabel("max category shift vs full pool (pp)")
a.set_title("(b) Deviation from the full pool", loc="left", color=INK)
a.set_xlim(0, 49)
a.set_ylim(0, 33)
a.grid(axis="y", color=GRID, lw=0.8)
a.set_axisbelow(True)
a.legend(loc="upper right", fontsize=8.5, handlelength=1.6, labelspacing=0.7,
         borderaxespad=0.4)

fig.tight_layout()
out = REPO / "outputs/figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"figure_R_author_cap_sensitivity.{ext}", dpi=300, bbox_inches="tight")
print(f"N={PICK}: {P[pi]:,}편 · 유효저자 {E[pi]:.0f} ({V[pi]:.1f}%) · "
      f"관측 {L[pi]:.2f}pp vs 귀무95% {NUL95[pi]:.2f}pp (초과 {L[pi]-NUL95[pi]:+.2f}pp)")
print("->", out / "figure_R_author_cap_sensitivity.png")
