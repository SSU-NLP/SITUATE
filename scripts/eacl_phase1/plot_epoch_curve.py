#!/usr/bin/env python3
"""Figure 02 — 학습량(epoch)에 따른 분절 성능. 논문용.

(a) val 손실 — **매 epoch** (학습 로그에서 읽는다)
(b) 6편 평균 Score — 스냅숏이 있는 epoch 에서만 (10 단위)
(c) 6편 평균 Lift

⚠️ (a) 와 (b)(c) 의 x 해상도가 다르다. val 손실은 매 epoch 기록되지만 분절 성능은
   체크포인트가 있어야 계산된다(`--ckpt-every 10`). 1 epoch 해상도로 (b)(c) 를 그리려면
   `--ckpt-every 1` 로 재학습해야 한다(체크포인트 5.5MB × epoch × seed × 모델).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "EACL/figures"))

import matplotlib.pyplot as plt  # noqa: E402
from GoodFigureExample import COLORS, apply, save  # noqa: E402

CURVE = json.loads((REPO / "outputs/experiments/2026-08-04_epoch_curve/epoch_curve.json").read_text())
NAME = "figure_02_epoch_curve"
LOGS = sorted((REPO / "outputs/runs").glob("train_i2e100_seed?.log"))

# ── (a) val 손실 — 로그에서 매 epoch ────────────────────────────────────────
PAT = re.compile(r"^\[(M5|M4) seed(\d)\] ep\s*(\d+) train ([\d.]+) val ([\d.]+)")
loss: dict[str, dict[int, list[float]]] = {"M5": {}, "M4": {}}
for lg in LOGS:
    for line in lg.open(encoding="utf-8", errors="ignore"):
        m = PAT.match(line)
        if m:
            loss[m.group(1)].setdefault(int(m.group(3)), []).append(float(m.group(5)))

apply()
plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 12})
fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.4), sharex=True,
                         gridspec_kw={"height_ratios": [1.0, 1.0, 0.9], "hspace": 0.32})
ax_l, ax_s, ax_f = axes
LABELPAD, FADE = 6, 0.62
CLR = {"M5": COLORS["red"], "M4": COLORS["blue"]}


def label_at(ax, x, y, t, c, dx=6, dy=0, ha="left", alpha=1.0):
    ax.annotate(t, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=c, alpha=alpha, ha=ha, va="center", fontsize=12)


# ⚠️ M4 는 VESSL Job 에서 돌아 **로컬 로그에 epoch 기록이 없다**. 있는 것만 그린다.
for mdl in ("M5", "M4"):
    eps = sorted(loss[mdl])
    if not eps:
        print(f"[warn] {mdl}: 로컬 로그에 epoch 손실 없음 — (a) 에서 생략")
        continue
    ys = [sum(loss[mdl][e]) / len(loss[mdl][e]) for e in eps]
    ax_l.plot(eps, ys, color=CLR[mdl], lw=2.2)
    label_at(ax_l, eps[len(eps) // 2], ys[len(eps) // 2], mdl, CLR[mdl], dy=13, ha="center")
ax_l.set_ylabel("Validation loss", labelpad=LABELPAD)

# ── (b)(c) 스냅숏 지점 ──────────────────────────────────────────────────────
for ax, key, lab in ((ax_s, "score", "Segmentation Score\n(6 works)"),
                     (ax_f, "lift", "Lift (6 works)")):
    for mdl in ("M5", "M4"):
        pts = sorted((int(k), v) for k, v in CURVE[mdl].items() if k != "999")
        x = [e for e, _ in pts]
        y = [v[key] for _, v in pts]
        sd = [v[key + "_sd"] for _, v in pts]
        ax.errorbar(x, y, yerr=sd, color=CLR[mdl], lw=2.2, marker="o", ms=5,
                    capsize=3, elinewidth=1.2)
        fin = CURVE[mdl].get("999")
        if fin:
            ax.plot([x[-1] + 6], [fin[key]], marker="*", ms=11, color=CLR[mdl], ls="none")
    ax.set_ylabel(lab, labelpad=LABELPAD)

label_at(ax_s, 30, CURVE["M5"]["30"]["score"], "★ = early-stopped final",
         COLORS["muted"], dx=10, dy=-16, ha="left")

PANEL = [(ax_l, "(a) Validation loss (every epoch)"),
         (ax_s, "(b) Segmentation quality (snapshots)"),
         (ax_f, "(c) Lift (snapshots)")]
for ax, title in PANEL:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 12),
                textcoords="offset points", fontsize=12.5, color=COLORS["ink"])

ax_f.set_xlim(0, 42)
ax_f.set_xticks([1, 10, 20, 30, 40])
ax_f.set_xlabel("Training epoch", labelpad=10)
fig.subplots_adjust(bottom=0.08, left=0.175, right=0.975, top=0.95)
save(fig, NAME)
print(f"val 손실 해상도 = 1 epoch (M5 {len(loss['M5'])} 점 · M4 {len(loss['M4'])} 점) · 성능 해상도 = 10 epoch "
      f"({len([k for k in CURVE['M5'] if k != '999'])} 점)")
