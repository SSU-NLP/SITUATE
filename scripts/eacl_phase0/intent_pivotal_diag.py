#!/usr/bin/env python3
"""intentionality pivotal 진단 — 서술 통계 전용 (API 0, 판정 아님).

질문: 경계 판정이 count ≥ t*=3 이므로, intentionality 를 "좁히는"(부분집합) 개입이 판정을 바꾸는
지점은 **다른 4축 합이 정확히 2인 곳**뿐이다. 3 이상이면 꺼도 경계고, 1 이하면 켜도 경계가 아니다.
그 지점 수가 이 개입의 기대 이득 상한이다.

산출:
  ① lift 곡선 — 거짓 발현 제거율 f 에 따른 lift (직전 산출의 '오라클lift' 는 f=0.5 임의 가정이었음)
  ② other4 분포 · pivotal 건수 · pivotal∧axis=1 을 gold 로 교차(잃는 것 vs 얻는 것)
  ③ 오라클 상한 ΔScore — 거짓 발현을 **전부** 제거했을 때(도달 불가능한 천장)를 노이즈 바닥과 대조

전부 frozen 캐시 + 사람 gold. 새 추출·동결 config 변경 없음.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402

from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

FROZEN = ["protagonist_present", "space", "time", "causality", "intentionality"]
T_STAR = 3
DEV3 = ("pieman", "tunnel", "monkey")
SIGMA2 = 0.060  # two-run ΔScore √2σ (D-B11 후속 노이즈 floor)


def load(d):
    p = REPO / f"outputs/eacl_phase0/{d}_michelmann_s5_q36_35b_v2/reset_matrix.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def end_turn(n, bt):
    y = [0] * n
    for i in bt:
        y[i - 1] = 1
    return y


def score(rows, col_override=None):
    """per-doc deployable Score at t*=3.  col_override: {transition: intentionality 값}"""
    n = len(rows) + 1
    cnt = {}
    for r in rows:
        t = int(r["transition"])
        s = sum(int(r[a]) for a in FROZEN if a != "intentionality")
        iv = col_override[t] if col_override is not None else int(r["intentionality"])
        cnt[t] = s + iv
    gold = end_turn(n, [int(r["transition"]) for r in rows if int(r["boundary"])])
    pred = end_turn(n, [t for t in cnt if cnt[t] >= T_STAR])
    return score_dialogues([gold], [pred])["score"]


def main():
    docs = {d: load(d) for d in DEV3}
    allrows = [r for d in DEV3 for r in docs[d]]
    out = []
    P = out.append

    # ------------------------------------------------------------------ ①
    P("## ① lift 곡선 — 직전 '오라클lift' 표기 오류 정정\n")
    P("직전 산출의 `오라클lift`(pieman 2.281 / tunnel 4.686 / monkey 3.723)는 **거짓 발현을 정확히**")
    P("**절반 제거**한다는 임의 가정(`p0/2`)이었다. `p1` 불변·`p0` 절반이라 lift 가 기계적으로 정확히")
    P("2.000 배가 된 것이며 오라클이 아니다. **인용 금지.** 올바른 형태는 제거율 f 에 대한 곡선이다")
    P("(f=1.0 이면 P(reset|비경계)=0 이라 lift 는 발산 — 상한이 존재하지 않는다).\n")
    P("| doc | f=0 (현행) | f=0.25 | f=0.5 | f=0.75 | f=1.0 |")
    P("|---|---|---|---|---|---|")
    for d in DEV3 + ("POOLED",):
        rows = allrows if d == "POOLED" else docs[d]
        nb = [r for r in rows if int(r["boundary"])]
        nn = [r for r in rows if not int(r["boundary"])]
        p1 = sum(int(r["intentionality"]) for r in nb) / len(nb)
        fp = sum(int(r["intentionality"]) for r in nn)
        cells = []
        for f in (0, 0.25, 0.5, 0.75, 1.0):
            p0 = fp * (1 - f) / len(nn)
            cells.append("∞ (발산)" if p0 == 0 else f"{p1 / p0:.3f}")
        P(f"| {d} | " + " | ".join(cells) + " |")
    P("\n**이 곡선이 lift 를 채택 기준으로 못 쓰는 이유를 그대로 보여준다** — 부분집합 개입은 발현을")
    P("줄이므로 lift 가 자동으로 오르고, 극단에서 발산한다(halo Δ=+0.000 hollow PASS 와 같은 함정).\n")

    # ------------------------------------------------------------------ ②
    P("## ② pivotal 카운트 (t*=3, other4 = intentionality 제외 4축 합)\n")
    P("### 2.1 other4 분포\n")
    P("| doc | 0 | 1 | **2 (pivotal)** | 3 | 4 | n |")
    P("|---|---|---|---|---|---|---|")
    for d in DEV3 + ("POOLED",):
        rows = allrows if d == "POOLED" else docs[d]
        dist = [0] * 5
        for r in rows:
            dist[sum(int(r[a]) for a in FROZEN if a != "intentionality")] += 1
        P(f"| {d} | {dist[0]} | {dist[1]} | **{dist[2]}** | {dist[3]} | {dist[4]} | {len(rows)} |")

    P("\n### 2.2 pivotal 지점에서 intentionality 의 상태\n")
    P("좁히는(부분집합) 개입은 **끄는 것만** 가능하므로, 판정이 뒤집힐 수 있는 곳은")
    P("`pivotal ∧ intentionality=1` 뿐이다 (count 3 → 2 로 떨어짐).\n")
    P("| doc | pivotal | ∧intent=1 | 그중 **경계 O**<br>(끄면 **잃음**) | 그중 **경계 X**<br>(끄면 **얻음**) | 얻음:잃음 | ∧intent=0 |")
    P("|---|---|---|---|---|---|---|")
    for d in DEV3 + ("POOLED",):
        rows = allrows if d == "POOLED" else docs[d]
        piv = [r for r in rows if sum(int(r[a]) for a in FROZEN if a != "intentionality") == T_STAR - 1]
        on = [r for r in piv if int(r["intentionality"])]
        lose = sum(1 for r in on if int(r["boundary"]))
        gain = len(on) - lose
        ratio = f"{gain / lose:.1f} : 1" if lose else (f"{gain} : 0" if gain else "—")
        P(f"| {d} | {len(piv)} | {len(on)} | **{lose}** | **{gain}** | {ratio} | {len(piv) - len(on)} |")

    P("\n### 2.3 축별 원시 건수 — **순이득(얻음−잃음) 절대량** 기준 정렬\n")
    P("비율이 아니라 절대 건수가 실제 지렛대 크기다. 비율은 참고로만 병기한다.\n")
    P("| 축 | pivotal | ∧축=1 | 경계O (잃음) | 경계X (얻음) | **순이득** | (참고) 얻음:잃음 |")
    P("|---|---|---|---|---|---|---|")
    tab = []
    for ax in FROZEN:
        piv = [r for r in allrows if sum(int(r[a]) for a in FROZEN if a != ax) == T_STAR - 1]
        on = [r for r in piv if int(r[ax])]
        lose = sum(1 for r in on if int(r["boundary"]))
        gain = len(on) - lose
        tab.append((gain - lose, ax, len(piv), len(on), lose, gain))
    for net, ax, npiv, non, lose, gain in sorted(tab, reverse=True):
        ratio = f"{gain / lose:.1f} : 1" if lose else (f"{gain} : 0" if gain else "—")
        mark = " ←" if ax == "intentionality" else ""
        P(f"| {ax}{mark} | {npiv} | {non} | {lose} | {gain} | **{net:+d}** | {ratio} |")
    r_ratio = [ax for _, ax, _, _, l, g in sorted(tab, key=lambda x: -(x[5] / x[4] if x[4] else 99))]
    r_net = [ax for _, ax, _, _, _, _ in sorted(tab, reverse=True)]
    P(f"\n- 순이득 순위: `{r_net}`")
    P(f"- 비율 순위:   `{r_ratio}`")
    P(f"- **{'일치' if r_net == r_ratio else '불일치 — 비율 1위와 절대 이득 1위가 다르다'}**")

    # ---------------------------------------------------- ②b gold 분해 (별도 절)
    P("\n---\n")
    P("# [별도 절] gold 66개의 other4 구간별 분해 — `count ≥ t*` 규칙 자체의 진단\n")
    P("이 절은 intentionality 진단이 **아니다.** 어떤 축을 어떻게 고치든 넘을 수 없는, 작동점 규칙")
    P("자체의 구조적 한계를 잰다. (other4 = intentionality 를 뺀 나머지 4축 합)\n")
    P("| doc | gold | other4≤1<br>**도달 불가** | other4=2 (pivotal)<br>∧intent=1 / ∧intent=0 | other4≥3<br>**자동 검출** | 검산 |")
    P("|---|---|---|---|---|---|")
    agg = [0, 0, 0, 0, 0]  # unreach, piv_on, piv_off, auto, gold
    for d in DEV3 + ("POOLED",):
        rows = allrows if d == "POOLED" else docs[d]
        g = [r for r in rows if int(r["boundary"])]
        unreach = piv_on = piv_off = auto = 0
        for r in g:
            o4 = sum(int(r[a]) for a in FROZEN if a != "intentionality")
            if o4 <= 1:
                unreach += 1
            elif o4 == 2:
                piv_on += int(r["intentionality"])
                piv_off += 1 - int(r["intentionality"])
            else:
                auto += 1
        ok = "✅" if unreach + piv_on + piv_off + auto == len(g) else "❌"
        P(f"| {d} | {len(g)} | **{unreach}** ({unreach / len(g):.0%}) | {piv_on} / {piv_off} | "
          f"**{auto}** | {unreach}+{piv_on}+{piv_off}+{auto}={unreach + piv_on + piv_off + auto} {ok} |")
        if d == "POOLED":
            agg = [unreach, piv_on, piv_off, auto, len(g)]
    u, po, pf, au, tg = agg
    P(f"\n- **구조적 recall 천장:** gold {tg}개 중 **{u}개({u / tg:.1%})가 other4≤1** — 이 경계들은")
    P(f"  intentionality 를 어떻게 만져도(켜도 count≤2) 절대 검출되지 않는다.")
    P(f"- **이미 자동 검출:** {au}개({au / tg:.1%})는 other4≥3 이라 intentionality 와 무관하게 잡힌다.")
    P(f"- **intentionality 가 실제로 좌우하는 gold = pivotal 의 {po + pf}개({(po + pf) / tg:.1%})뿐이고,**")
    P(f"  그중 이미 켜져 있는 {po}개가 §2.2 의 '잃음' 이고 꺼져 있는 {pf}개는 넓히는 개입의 대상이다.")

    # ------------------------------------------------------------------ ③
    P("\n---\n")
    P("## ③ 오라클 상한 ΔScore (도달 불가능한 천장) vs 노이즈 바닥\n")
    P("거짓 발현을 **전부** 제거하고 참 발현은 **전부** 보존하는 완벽한 분류기를 가정한다.")
    P("실제 분류기는 이보다 반드시 나쁘다.\n")
    P("| doc | 현행 Score | 오라클 좁힘 Score | ΔScore (천장) | vs 노이즈 바닥 0.060 |")
    P("|---|---|---|---|---|")
    tot_cur = tot_or = 0.0
    for d in DEV3:
        rows = docs[d]
        cur = score(rows)
        orc = score(rows, {int(r["transition"]): (int(r["intentionality"]) if int(r["boundary"]) else 0)
                           for r in rows})
        tot_cur += cur
        tot_or += orc
        dd = orc - cur
        P(f"| {d} | {cur:.4f} | {orc:.4f} | **{dd:+.4f}** | "
          f"{'초과' if abs(dd) >= SIGMA2 else '**미만 = 해석 불가**'} |")
    dd = (tot_or - tot_cur) / 3
    P(f"| **평균** | {tot_cur / 3:.4f} | {tot_or / 3:.4f} | **{dd:+.4f}** | "
      f"{'초과' if abs(dd) >= SIGMA2 else '**미만 = 해석 불가**'} |")

    # ------------------------------------------------------------------ ④ MC
    import random
    import statistics as stx
    import noise_propagate_pieman as npp
    flip = npp.measure_flip_rates()
    P("\n---\n")
    P("## ④ pivotal 집합 안정성 — 축별 flip rate 주입 MC\n")
    P("`noise_propagate_pieman.py` 와 동일한 모델: 축별 독립 Bernoulli flip 을 주입하고 pivotal 집합을")
    P("다시 계산한다. **⚠️ flip rate 는 repcorpus `eismv2`(deepseek) run1↔run2 에서 측정된 값이고 본")
    P("분석은 `q36_35b v2` 캐시다 — 라벨러가 다르다.** 측정된 flip 데이터가 이것뿐이라 그대로 쓰되,")
    P("절대값이 아니라 **흔들림의 규모**를 읽는 용도로만 해석한다.\n")
    P("측정된 flip rate: " + " · ".join(f"`{a}` {flip[a]:.3f}" for a in FROZEN) + "\n")
    rng = random.Random(20260725)
    N = 2000
    npiv, non, nlose, ngain, nnet = [], [], [], [], []
    for _ in range(N):
        pv = onc = lo = 0
        for r in allrows:
            v = {a: int(r[a]) ^ (1 if rng.random() < flip[a] else 0) for a in FROZEN}
            o4 = sum(v[a] for a in FROZEN if a != "intentionality")
            if o4 == T_STAR - 1:
                pv += 1
                if v["intentionality"]:
                    onc += 1
                    lo += int(r["boundary"])
        npiv.append(pv); non.append(onc); nlose.append(lo); ngain.append(onc - lo)
        nnet.append(onc - 2 * lo)
    def q(v):
        s = sorted(v)
        return f"{stx.mean(v):.1f} ± {stx.stdev(v):.1f}  [{s[int(.025*len(s))]}, {s[int(.975*len(s))]}]"
    P("| 통계량 | 관측(단일 draw) | MC 평균 ± sd  [95% 구간] |")
    P("|---|---|---|")
    obs_piv = [r for r in allrows if sum(int(r[a]) for a in FROZEN if a != "intentionality") == T_STAR - 1]
    obs_on = [r for r in obs_piv if int(r["intentionality"])]
    obs_lose = sum(1 for r in obs_on if int(r["boundary"]))
    P(f"| pivotal 건수 | {len(obs_piv)} | {q(npiv)} |")
    P(f"| ∧intent=1 | {len(obs_on)} | {q(non)} |")
    P(f"| 경계O (잃음) | {obs_lose} | {q(nlose)} |")
    P(f"| 경계X (얻음) | {len(obs_on) - obs_lose} | {q(ngain)} |")
    rr = [g / l if l else float("inf") for g, l in zip(ngain, nlose)]
    fin = [x for x in rr if x != float("inf")]
    P(f"| 얻음:잃음 비 | {(len(obs_on) - obs_lose) / obs_lose:.1f} | "
      f"{stx.mean(fin):.1f} ± {stx.stdev(fin):.1f}  [{sorted(fin)[int(.025*len(fin))]:.1f}, "
      f"{sorted(fin)[int(.975*len(fin))]:.1f}]  (∞ {len(rr)-len(fin)}회) |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
