#!/usr/bin/env python3
"""Phase-1 앵커 코퍼스의 k=1 라벨 변동 실측 (겹치는 7,965편, 추가 GPU 비용 0).

배경
----
`outputs/eacl_phase0/phase1_all_v31/reset_matrix.jsonl` (10,598편 · 2,777,561 전이)는
**k=1 단일 draw**다. vLLM은 단일 요청에서도 결정적이지 않으므로(D-B14 부수 정정: 동일 요청
8회에 고유 출력 4종), 이 앵커 라벨이 얼마나 흔들리는지가 미측정 상태였다
(2026-07-28_phase1_full_relabel/REPORT.md §5).

그런데 직전 세대 `phase1_full_v31/` (9,603편)이 남아 있고 **7,965편이 겹친다** = 이미 확보된
독립 2차 draw. 이 스크립트는 그 겹침으로 변동 폭을 잰다.

⚠️ 측정되는 것의 정확한 정체
---------------------------
두 세대는 **조건이 다르다**:
    phase1_full_v31 : TP=2, prefix caching OFF
    phase1_all_v31  : TP=1, prefix caching ON
따라서 관측되는 불일치 = (재실행 샘플링 지터) + (서빙 설정 변경 효과) 이며,
**k=1 변동의 상한(upper bound)**이다. 순수 k 지터는 이보다 작다.
1차 탐색 패스에는 이 상한으로 충분하다 — 상한이 작으면 순수 지터는 더 작기 때문이다.

⚠️ 이 측정은 게이트가 아니다 (사용자 결정 2026-07-28, ledger D-B16)
------------------------------------------------------------------
학습을 막지 않는다. 용도는 (1) 3단계 M4/M5 결과를 읽을 때의 해석 폭 제공,
(2) 확정 패스의 k 재설정 입력. 확정 k는 1만편 전체에 대해 나중에 따로 정한다.

판정 기준은 아래 PRE_DECLARED에 **측정 전에** 박아두었다 (사후 기준변경 금지 규율).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

# ── 동결 설정 (변경 금지) ────────────────────────────────────────────────────
FROZEN_AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]  # D-B10
T_STAR = 3  # count >= t* => boundary

GEN_A = REPO / "outputs/eacl_phase0/phase1_full_v31/reset_matrix.jsonl"   # TP=2, prefix off
GEN_B = REPO / "outputs/eacl_phase0/phase1_all_v31/reset_matrix.jsonl"    # TP=1, prefix on
OUT_DIR = REPO / "outputs/experiments/2026-07-28_anchor_k_stability"

# ── 사전 선언 판정 기준 (2026-07-28, 측정 전 확정 — 결과 보고 고치지 않는다) ──
PRE_DECLARED = """
[사전 선언 — 측정 전 확정, 2026-07-28]

(1) 앵커 노이즈 규모 해석 밴드 — headline = S_self (공식 Score로 잰 두 세대 간 자기일치도)
    S_self >= 0.90          : 작음.   k=1로 1차 패스 진행. 해석 폭 = (1 - S_self).
    0.75 <= S_self < 0.90   : 중간.   k=1 진행하되, M5-M4 ΔScore가 (1 - S_self)보다 작으면
                                      "가설 지지"라고 부르지 않는다.
    S_self < 0.75           : 큼.     1차 패스 결과는 방향성만. 확정 패스에서 다중 draw 필수.

(2) 무작위 vs 계통적 판정
    다음 둘을 모두 만족하면 "무작위(random)":
      (a) 5축 전부 McNemar 비대칭 |b-c|/(b+c) < 0.10
      (b) 문서별 불일치율의 과분산비(관측분산/이항분산) < 1.5
    하나라도 위반하면 "계통적(systematic)".
    함의: 무작위면 다중 draw 평균으로 지워지므로 k를 키우는 게 유효.
          계통적이면 k로 안 지워지므로 확정 패스에서는 k보다 "조건 통일"이 우선.

(3) 기존 노이즈 floor(D-B11 후속)와의 접속
    단일추출 ΔScore ~0.06 미만 해석불가 / 불안정 doc ~0.10 미만 불가.
    본 측정의 (1 - S_self)가 그 0.06과 어떤 관계인지 병기한다.
"""


def load_matrix(path: Path) -> dict[str, dict[int, int]]:
    """reset_matrix.jsonl -> {doc_id: {transition: 5bit mask}}.

    5축을 비트마스크로 접어 메모리를 줄인다 (bit i = FROZEN_AXES[i]).
    """
    docs: dict[str, dict[int, int]] = defaultdict(dict)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mask = 0
            for i, a in enumerate(FROZEN_AXES):
                if int(r[a]):
                    mask |= 1 << i
            docs[r["doc_id"]][int(r["transition"])] = mask
    return docs


def unpack(masks: np.ndarray) -> np.ndarray:
    """(N,) 비트마스크 -> (N, 5) 0/1 행렬."""
    return np.stack([(masks >> i) & 1 for i in range(len(FROZEN_AXES))], axis=1)


def end_turn_labels(n_sent: int, bnd_transitions: list[int]) -> list[int]:
    """끝-turn 규약: transition i 가 경계면 문장 i-1 에 1."""
    y = [0] * n_sent
    for i in bnd_transitions:
        y[i - 1] = 1
    return y


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(PRE_DECLARED, flush=True)
    print(f"[load] A = {GEN_A}", flush=True)
    A = load_matrix(GEN_A)
    print(f"[load]   {len(A):,} docs", flush=True)
    print(f"[load] B = {GEN_B}", flush=True)
    B = load_matrix(GEN_B)
    print(f"[load]   {len(B):,} docs", flush=True)

    shared_docs = sorted(set(A) & set(B))
    print(f"[join] 겹치는 문서 = {len(shared_docs):,}편", flush=True)

    # ── 전이 단위 조인 ────────────────────────────────────────────────────
    a_rows, b_rows, doc_index, pos_frac = [], [], [], []
    n_trans_mismatch = 0
    per_doc_meta = []  # (doc, n_shared_t, n_a_only, n_b_only)
    for di, doc in enumerate(shared_docs):
        ta, tb = A[doc], B[doc]
        shared_t = sorted(set(ta) & set(tb))
        if len(ta) != len(tb):
            n_trans_mismatch += 1
        per_doc_meta.append((doc, len(shared_t), len(ta), len(tb)))
        n = len(shared_t)
        if n == 0:
            continue
        for j, t in enumerate(shared_t):
            a_rows.append(ta[t])
            b_rows.append(tb[t])
            doc_index.append(di)
            pos_frac.append(j / max(1, n - 1) if n > 1 else 0.0)

    a_mask = np.asarray(a_rows, dtype=np.int8)
    b_mask = np.asarray(b_rows, dtype=np.int8)
    doc_index = np.asarray(doc_index, dtype=np.int32)
    pos_frac = np.asarray(pos_frac, dtype=np.float32)
    del a_rows, b_rows

    A5 = unpack(a_mask.astype(np.int16))
    B5 = unpack(b_mask.astype(np.int16))
    N = A5.shape[0]
    print(f"[join] 공통 전이 = {N:,}개 (전이 수 불일치 문서 {n_trans_mismatch:,}편)", flush=True)

    lines: list[str] = []
    w = lines.append

    # ── 1. 축별 일치율 + McNemar 비대칭 + kappa ──────────────────────────
    w("## 1. 축별 라벨 일치율 (전이 단위, N = %s)\n" % f"{N:,}")
    w("| 축 | A 발화율 | B 발화율 | 일치율 | b=A1B0 | c=A0B1 | McNemar 비대칭 |b−c|/(b+c) | Cohen κ |")
    w("|---|---|---|---|---|---|---|---|")
    axis_stats = {}
    for i, ax in enumerate(FROZEN_AXES):
        a, b_ = A5[:, i], B5[:, i]
        agree = float((a == b_).mean())
        b_cnt = int(((a == 1) & (b_ == 0)).sum())
        c_cnt = int(((a == 0) & (b_ == 1)).sum())
        asym = abs(b_cnt - c_cnt) / max(1, b_cnt + c_cnt)
        pa, pb = float(a.mean()), float(b_.mean())
        pe = pa * pb + (1 - pa) * (1 - pb)  # chance agreement
        kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
        axis_stats[ax] = dict(agree=agree, asym=asym, kappa=kappa, b=b_cnt, c=c_cnt)
        w(f"| {ax} | {pa:.4f} | {pb:.4f} | {agree:.4f} | {b_cnt:,} | {c_cnt:,} | "
          f"{asym:.3f} | {kappa:.3f} |")
    w("")
    w("> 발화율이 낮은 축(time 0.02)은 원일치율이 자동으로 높아진다 — **κ를 함께 읽어야 한다.**")
    w("")

    # ── 2. count / 경계 ──────────────────────────────────────────────────
    ca, cb = A5.sum(1), B5.sum(1)
    d_count = np.abs(ca - cb)
    bnd_a, bnd_b = (ca >= T_STAR), (cb >= T_STAR)
    inter = int((bnd_a & bnd_b).sum())
    union = int((bnd_a | bnd_b).sum())
    w(f"## 2. count(0~5) · 경계(count ≥ t\\*={T_STAR})\n")
    w("| 지표 | 값 |")
    w("|---|---|")
    w(f"| count 완전일치율 | {float((ca == cb).mean()):.4f} |")
    w(f"| 평균 \\|Δcount\\| | {float(d_count.mean()):.4f} |")
    w(f"| \\|Δcount\\| ≥ 2 비율 | {float((d_count >= 2).mean()):.4f} |")
    w(f"| A 경계율 | {float(bnd_a.mean()):.4f} |")
    w(f"| B 경계율 | {float(bnd_b.mean()):.4f} |")
    w(f"| 경계 일치율 (전이 단위) | {float((bnd_a == bnd_b).mean()):.4f} |")
    w(f"| **경계 Jaccard (pooled)** | **{inter / max(1, union):.4f}** |")
    w(f"| 경계 both=1 | {inter:,} |")
    w(f"| 경계 A만 | {int((bnd_a & ~bnd_b).sum()):,} |")
    w(f"| 경계 B만 | {int((~bnd_a & bnd_b).sum()):,} |")
    w("")

    # ── 3. S_self (공식 Score) ────────────────────────────────────────────
    print("[score] S_self 계산 중...", flush=True)
    ys_a, ys_b, per_doc_s, per_doc_jac = [], [], [], []
    for di, doc in enumerate(shared_docs):
        sel = doc_index == di
        if not sel.any():
            continue
        n_sent = int(sel.sum()) + 1
        idx = np.flatnonzero(sel)
        ba = [int(k) + 1 for k in range(len(idx)) if bnd_a[idx[k]]]
        bb = [int(k) + 1 for k in range(len(idx)) if bnd_b[idx[k]]]
        ya, yb = end_turn_labels(n_sent, ba), end_turn_labels(n_sent, bb)
        ys_a.append(ya)
        ys_b.append(yb)
        sa, sb = set(ba), set(bb)
        per_doc_jac.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)

    s_ab = score_dialogues(ys_a, ys_b)["score"]  # A를 ref, B를 hyp
    s_ba = score_dialogues(ys_b, ys_a)["score"]  # 대칭
    s_self = (s_ab + s_ba) / 2

    # 문서별 S_self (분포용)
    print("[score] 문서별 S_self 계산 중...", flush=True)
    for ya, yb in zip(ys_a, ys_b):
        per_doc_s.append((score_dialogues([ya], [yb])["score"]
                          + score_dialogues([yb], [ya])["score"]) / 2)
    per_doc_s = np.asarray(per_doc_s)
    per_doc_jac = np.asarray(per_doc_jac)

    w("## 3. S_self — 공식 Score로 잰 두 세대 자기일치도 (headline)\n")
    w("공식 채점기 `hi_ontop.dts_scoring.score_dialogues`, 끝-turn 규약, deployable(count ≥ t\\*).\n")
    w("| 지표 | 값 |")
    w("|---|---|")
    w(f"| S(B\\|A) | {s_ab:.4f} |")
    w(f"| S(A\\|B) | {s_ba:.4f} |")
    w(f"| **S_self (대칭 평균)** | **{s_self:.4f}** |")
    w(f"| **노이즈 폭 (1 − S_self)** | **{1 - s_self:.4f}** |")
    w(f"| 문서별 S_self 평균 | {per_doc_s.mean():.4f} |")
    w(f"| 문서별 S_self 중앙값 | {np.median(per_doc_s):.4f} |")
    w(f"| 문서별 S_self 10퍼센타일 | {np.percentile(per_doc_s, 10):.4f} |")
    w(f"| 문서별 S_self 90퍼센타일 | {np.percentile(per_doc_s, 90):.4f} |")
    w(f"| 문서별 경계 Jaccard 평균 | {per_doc_jac.mean():.4f} |")
    w("")

    # ── 4. 무작위 vs 계통적 ───────────────────────────────────────────────
    # (a) McNemar 비대칭
    max_asym = max(v["asym"] for v in axis_stats.values())
    asym_ok = max_asym < 0.10
    # (b) 문서별 불일치율 과분산
    dis = (ca != cb).astype(np.float64)
    doc_dis_rate, doc_n = [], []
    for di in range(len(shared_docs)):
        sel = doc_index == di
        if sel.sum() == 0:
            continue
        doc_dis_rate.append(float(dis[sel].mean()))
        doc_n.append(int(sel.sum()))
    doc_dis_rate = np.asarray(doc_dis_rate)
    doc_n = np.asarray(doc_n)
    p_bar = float(dis.mean())
    obs_var = float(doc_dis_rate.var())
    exp_var = float((p_bar * (1 - p_bar) / doc_n).mean())  # 이항 기대분산
    overdisp = obs_var / exp_var if exp_var > 0 else float("nan")
    od_ok = overdisp < 1.5
    verdict_rand = "무작위(random)" if (asym_ok and od_ok) else "계통적(systematic)"

    w("## 4. 무작위 vs 계통적 판정\n")
    w("| 검정 | 기준(사전선언) | 관측 | 판정 |")
    w("|---|---|---|---|")
    w(f"| (a) 최대 McNemar 비대칭 | < 0.10 | {max_asym:.3f} | {'✅ 통과' if asym_ok else '❌ 위반'} |")
    w(f"| (b) 문서별 불일치 과분산비 | < 1.5 | {overdisp:.2f} | {'✅ 통과' if od_ok else '❌ 위반'} |")
    w(f"\n**판정 = {verdict_rand}**\n")

    # 위치·길이 편중
    w("### 불일치의 위치·길이 편중\n")
    w("| 문서 내 상대위치 | count 불일치율 |")
    w("|---|---|")
    for lo in range(0, 10):
        sel = (pos_frac >= lo / 10) & (pos_frac < (lo + 1) / 10 + (1e-9 if lo == 9 else 0))
        if sel.sum():
            w(f"| {lo/10:.1f}–{(lo+1)/10:.1f} | {float(dis[sel].mean()):.4f} |")
    w("")
    q = np.percentile(doc_n, [25, 50, 75])
    w("| 문서 길이 사분위 | 문서수 | count 불일치율 |")
    w("|---|---|---|")
    bins = [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e18)]
    for lo, hi in bins:
        m = (doc_n > lo) & (doc_n <= hi)
        if m.sum():
            w(f"| {int(lo)}–{int(min(hi, doc_n.max()))} 전이 | {int(m.sum()):,} | "
              f"{float(doc_dis_rate[m].mean()):.4f} |")
    w("")

    # ── 5. 사전선언 밴드 대조 ─────────────────────────────────────────────
    if s_self >= 0.90:
        band = "작음 — k=1로 1차 패스 진행. 해석 폭 = %.4f" % (1 - s_self)
    elif s_self >= 0.75:
        band = ("중간 — k=1 진행하되, M5−M4 ΔScore가 %.4f 미만이면 '가설 지지'라 부르지 않는다."
                % (1 - s_self))
    else:
        band = "큼 — 1차 패스 결과는 방향성만. 확정 패스에서 다중 draw 필수."
    w("## 5. 사전선언 밴드 대조\n")
    w(f"- S_self = **{s_self:.4f}** → **{band}**")
    w(f"- 무작위/계통 = **{verdict_rand}**")
    w(f"- 기존 노이즈 floor(D-B11 후속) 0.06 대비: 본 측정 노이즈 폭 = {1 - s_self:.4f} "
      f"({'더 큼' if (1 - s_self) > 0.06 else '더 작음'})")
    w("")

    (OUT_DIR / "_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    np.savez_compressed(
        OUT_DIR / "per_doc.npz",
        doc=np.array(shared_docs, dtype=object),
        s_self=per_doc_s, jaccard=per_doc_jac, n_trans=doc_n, dis_rate=doc_dis_rate,
    )
    print("\n".join(lines))
    print(f"\n[done] -> {OUT_DIR}/_metrics.md", flush=True)


if __name__ == "__main__":
    main()
