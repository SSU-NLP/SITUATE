#!/usr/bin/env python3
"""Figure 01 — 작가당 상한 N 에 따른 코퍼스 다양성. 논문용.

CLAUDE.md figure 규약: `EACL/figures/figure_<번호>_<제목>.{pdf,png}`.
서식은 `EACL/figures/GoodFigureExample.py`(예시 이미지에서 추출한 공통 스타일)를 쓴다.

■ 지표를 **역할별 3패널**로 나눈다 (사용자 지시 2026-08-02).
    (a) 결정 입력 — 상한을 올려 잃는 것(유효 저자 수)과 얻는 것(편수)
    (b) 결정 기준 — 둘에서 파생된 한계 교환비. 정점 N=19 가 채택 근거
    (c) 확인용   — 상한이 시대·장르 구성을 훼손하지 않는가

  이전 버전은 다섯 계열을 한 평면의 이중축에 얹었는데, 어느 곡선이 어느 축인지 알 수 없고
  축 이름("Count", "Effective number / rate")이 뭉뚱그려져 뜻이 안 통했으며 5색이 역할을
  전혀 나타내지 못했다. 패널을 나누면 패널마다 y축이 하나라 소속이 자명하고, 축 이름은 그
  패널이 재는 것만 적으면 되며, 색 구분은 패널 안에서만 필요하다.

■ 범례 대신 **곡선 옆 직접 라벨**을 쓴다. 패널당 계열이 1~2개라 범례가 불필요하고,
  눈이 범례와 곡선을 오갈 필요가 없다.

⚠️ (a) 는 로그 눈금이다. 유효 저자 수(553~2,144)와 편수(4,353~14,545)는 둘 다 '개수'라
   축 하나를 공유해도 되지만, 선형이면 저자 곡선이 바닥에 눌린다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "EACL/figures"))

import matplotlib.pyplot as plt  # noqa: E402
from GoodFigureExample import COLORS, apply, save  # noqa: E402

# 상한 키에 따라 곡선이 통째로 달라진다 — 문자열 키는 한 인물이 상한을 여러 번 받아
# 편중을 통제하지 못한다(2026-08-03). person 이 논문용 기본이고, string 은 대조용이다.
EXP = REPO / "outputs/experiments/2026-08-02_author_cap"
ap = argparse.ArgumentParser()
ap.add_argument("--author-key", choices=("person", "string"), default="person")
ap.add_argument("--chosen", type=int, default=None,
                help="채택 상한. 미지정이면 교환비 정점을 쓴다. 정점과 다르면 그림이 둘을 구분해 표시한다")
args = ap.parse_args()
SRC = EXP / ("table_final_dev3_personkey.json" if args.author_key == "person"
             else "table_final_dev3.json")
NAME = ("figure_01_author_cap_diversity" if args.author_key == "person"
        else "figure_01_author_cap_diversity_stringkey")
NMAX = 50

D = json.loads(SRC.read_text())
N = list(range(1, NMAX + 1))
# 채택 상한 = 한계 교환비 정점. **데이터에서 읽는다** — 예전엔 20 이 박혀 있어 키를 바꿔도
# 그림의 세로선이 따라오지 않았다. 최적점 주장이 아니라 설계 선택이다.
PEAK = D["peak"]
CHOSEN = args.chosen or PEAK   # 채택값 ≠ 정점일 수 있다 — 그때 정점을 'peak' 로 부르면 거짓말이 된다

apply()
plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 12})

fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.8), sharex=True,
                         gridspec_kw={"height_ratios": [1.15, 1.0, 0.95], "hspace": 0.34})
ax_in, ax_cr, ax_ck = axes


LABELPAD = 6          # 세 패널의 (눈금글자 ↔ 축이름) 간격을 같게. 축이름 x위치는 서로 달라도 된다
FADE = 0.62           # 같은 패널의 부차 계열 투명도


def label_at(ax, x, y, text, color, dx=6, dy=0, ha="left", va="center", alpha=1.0):
    """곡선 옆 직접 라벨 — 범례를 대신한다."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, alpha=alpha, ha=ha, va=va, fontsize=12)


# ── (a) 결정 입력 ────────────────────────────────────────────────────────────
ax_in.plot(N, D["pieces"][:NMAX], color=COLORS["red"])
ax_in.plot(N, D["auth"][:NMAX], color=COLORS["red"], alpha=FADE)
ax_in.set_yscale("log")
_lo = min(min(D["auth"][:NMAX]), min(D["pieces"][:NMAX]))
_hi = max(max(D["auth"][:NMAX]), max(D["pieces"][:NMAX]))
ax_in.set_ylim(_lo * 0.82, _hi * 1.8)
_tk = [t for t in (200, 500, 1000, 2000, 5000, 10000, 20000) if _lo * 0.82 <= t <= _hi * 1.8]
ax_in.set_yticks(_tk)
ax_in.set_yticklabels([f"{t:,}" for t in _tk])
ax_in.minorticks_off()
ax_in.set_ylabel("Stories and effective\nauthors", labelpad=LABELPAD)
label_at(ax_in, 30, D["pieces"][29], "number of stories", COLORS["red"], dx=0, dy=15, ha="center")
label_at(ax_in, 36, D["auth"][35], "effective number of authors", COLORS["red"], dx=0, dy=15, ha="center", alpha=FADE)

# ── (b) 결정 기준 ────────────────────────────────────────────────────────────
ax_cr.plot(N, D["rate"][:NMAX], color=COLORS["blue"], lw=2.8)
_r = D["rate"][:NMAX]
ax_cr.set_ylim(min(_r) - 0.9, max(_r) * 1.14)
ax_cr.set_yticks([t for t in range(0, 40, 4) if min(_r) - 0.9 <= t <= max(_r) * 1.14])
ax_cr.set_ylabel("Marginal exchange rate\n(stories / effective author)", labelpad=LABELPAD)
if CHOSEN == PEAK:
    label_at(ax_cr, CHOSEN, D["rate"][CHOSEN - 1], f"peak at $N={CHOSEN}$",
             COLORS["blue"], dx=10, dy=16, ha="left")
else:
    # 채택값과 정점을 함께 보인다. 평평한 구간에서 규모를 고려해 고른 것이므로 둘 다 사실이다.
    label_at(ax_cr, CHOSEN, D["rate"][CHOSEN - 1], f"selected $N={CHOSEN}$",
             COLORS["blue"], dx=9, dy=17, ha="left")
    ax_cr.plot([PEAK], [D["rate"][PEAK - 1]], "o", ms=5, mfc="white",
               mec=COLORS["blue"], mew=1.6, zorder=5)
    label_at(ax_cr, PEAK, D["rate"][PEAK - 1], f"peak $N={PEAK}$",
             COLORS["blue"], dx=-9, dy=15, ha="right", alpha=FADE)

# ── (c) 확인용 ───────────────────────────────────────────────────────────────
ax_ck.plot(N, D["era"][:NMAX], color=COLORS["green"], lw=2.1)
ax_ck.plot(N, D["lc"][:NMAX], color=COLORS["green"], lw=2.1, alpha=FADE)
_c = D["era"][:NMAX] + D["lc"][:NMAX]
ax_ck.set_ylim(0, max(_c) * 1.42)
ax_ck.set_yticks([t for t in range(0, 20, 2) if t <= max(_c) * 1.42])
ax_ck.set_ylabel("Effective number\nof categories", labelpad=LABELPAD)
label_at(ax_ck, 32, D["era"][31], "era bins", COLORS["green"], dx=0, dy=15, ha="center")
label_at(ax_ck, 32, D["lc"][31], "LC classifications", COLORS["green"], dx=0, dy=15, ha="center", alpha=FADE)

PANEL = [(ax_in, "(a) What the limit trades"),
         (ax_cr, "(b) Selection criterion"),
         (ax_ck, "(c) Unaffected dimensions")]
for ax, title in PANEL:
    ax.axvline(CHOSEN, color=COLORS["muted"], lw=1.0, ls=(0, (5, 4)), zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 13),
                textcoords="offset points", fontsize=12.5, color=COLORS["ink"])

ax_ck.set_xlim(-2, 53)                           # 1·50 이 축 끝 모서리에 붙지 않게
ax_ck.set_xticks([1, 10, 20, 30, 40, 50])
ax_ck.set_xlabel("Per-author limits $N$", labelpad=10)

fig.subplots_adjust(bottom=0.075, left=0.185, right=0.975, top=0.945)
save(fig, NAME)
rate = D["rate"][:NMAX]
print(f"N=1..{NMAX} · authors {D['auth'][0]:.0f}→{D['auth'][NMAX-1]:.0f} · "
      f"stories {D['pieces'][0]:,.0f}→{D['pieces'][NMAX-1]:,.0f} · "
      f"rate max N={rate.index(max(rate)) + 1} ({max(rate):.3f})")
