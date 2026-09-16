#!/usr/bin/env python3
"""논문 figure 공통 서식 — `EACL/figures/좋은figure예시.jpg` 에서 추출.

CLAUDE.md figure 규약의 "디자인 기준"을 코드로 고정한 것. 새 figure 는 이 파일을 import 하거나
복사해서 쓴다. 색은 예시 이미지의 실제 픽셀값에서 뽑았다.

사용:
    import sys; sys.path.insert(0, "EACL/figures")
    from GoodFigureExample import apply, COLORS, color_axis, save

    apply()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ...
    save(fig, "figure_02_something")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# ── 팔레트 (좋은figure예시.jpg 픽셀값) ────────────────────────────────────────
COLORS = {
    "red": "#d1495b",        # 주 계열. 예시의 벽돌색(#c1392b)보다 채도를 올린 장미빛 — 사용자 선호
    "green": "#2a9d8f",      # 부 계열. 예시의 세이지(#5d9383)보다 맑은 청록 — 사용자 선호
    "lavender": "#b69fca",   # 3번째 계열 (예시의 modality separability 막대)
    "orange": "#e08214",     # 4번째 계열 — 예시 밖. red 와 색상은 이웃이나 명도차가 커 구분된다
    "blue": "#3f6fb5",       # 5번째 계열 — 예시 밖. green 이 청록이라 확실히 파란 쪽으로 뺐다
    "purple": "#8a63ad",     # 6번째 계열 — 예시 밖. lavender 를 진하게 한 색
    "ink": "#000000",        # 텍스트·왼쪽 축·x축
    "muted": "#7a7a7a",      # 보조 텍스트
    "hairline": "#d9d9d9",   # 기준선(예시의 0.50 점선)
}
SERIES = [COLORS["red"], COLORS["green"], COLORS["blue"]]   # 고정 순서

# ── rcParams ─────────────────────────────────────────────────────────────────
RC = {
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.3,
    "lines.linewidth": 2.4,
    "lines.solid_capstyle": "round",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "axes.grid": False,                 # 격자 없음
    "axes.spines.top": False,           # 위쪽 spine 없음
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,            # 범례 테두리 없음
    "pdf.fonttype": 42,                 # 폰트 임베드
    "ps.fonttype": 42,
}

FIGSIZE = (7.2, 5.4)     # 예시와 같은 비율
DPI = 300
OUTDIR = Path(__file__).resolve().parent


def apply() -> None:
    """rcParams 적용. 그림 그리기 전에 한 번 호출."""
    matplotlib.use("Agg")
    plt.rcParams.update(RC)


def color_axis(ax, color: str, side: str) -> None:
    """축 하나를 계열 색으로 칠한다 — spine·눈금·눈금라벨·축라벨.

    이중 y축에서 어느 곡선이 어느 축인지 색으로만 잇는 예시 그림의 방식.
    `side` 는 "left" 또는 "right".
    """
    other = "right" if side == "left" else "left"
    ax.spines["top"].set_visible(False)
    ax.spines[side].set_color(color)
    ax.spines[other].set_visible(False)
    ax.tick_params(axis="y", colors=color)
    if ax.yaxis.label is not None:
        ax.yaxis.label.set_color(color)


def pad_xlim(ax, lo: float, hi: float, frac: float = 0.05) -> None:
    """눈금이 축 끝 모서리에 붙지 않도록 데이터 범위보다 넓게 잡는다(예시 그림 방식)."""
    m = (hi - lo) * frac
    ax.set_xlim(lo - m, hi + m)


def legend_below(fig, handles=None, labels=None, y: float = 0.01, ncol: int = 1):
    """범례를 그림 아래 중앙에, 테두리 없이, 한 항목당 한 줄로."""
    kw = dict(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
              handlelength=1.9, handletextpad=0.9, labelspacing=0.75,
              borderaxespad=0.0, frameon=False)
    return fig.legend(handles, labels, **kw) if handles else fig.legend(**kw)


def save(fig, name: str) -> None:
    """CLAUDE.md 규약대로 EACL/figures/ 에 PDF+PNG 둘 다 저장.

    `name` 은 확장자 없는 `figure_<번호>_<제목>` 형식.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=DPI)
    print("->", OUTDIR / f"{name}.png")
