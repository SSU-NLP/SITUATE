# ④ S5 재설계 실험 — v2 baseline vs v3 (프롬프트 재설계) × reasoning on/off

> ②진단 통과분(A·H·B②·E) + B①(경보강)을 v3 프롬프트에 적용. Gate C(주판정)=deployable Score
> ΔScore≥−0.03(v2-ON 대비). 배포후보=v3-OFF(재설계+no-reasoning, 4.5×빠름). v3-ON=귀속용 arm.
> dev-3(pieman/tunnel/monkey, gold 有). reset/schema/protagonist/causality = v2와 동일(최소변경).

## 실험 setup

- **데이터:** dev-3 michelmann master (pieman 96·tunnel 504·monkey 408 문장; gold 18/18/30).
- **라벨러:** deepseek-v4-pro. v2=frozen p0s5-eism-v2(reasoning ON). v3=p0s5-eism-v3(A/H space,
  E time, B② goals, B① intent). arm: ON(reasoning) / OFF(no-reasoning).
- **Score:** `hi_ontop.dts_scoring.score_dialogues`, deployable 5축 t\*=3(=v2 동결 작동점).
- **Gate C:** ΔScore(cfg − v2ON) ≥ −0.03 per doc. 타깃 진단=gold-free regex(②진단 동급 스크리닝).

## Gate C — deployable Score (5축 t\*=3)

| doc | v2-ON (frozen baseline) | v3-ON (redesign, reasoning) | v3-OFF (redesign, no-reason) |
|---|---|---|---|
| pieman | 0.578 (g18/p16) | 0.342 (g18/p5) | 0.490 (g18/p16) |
| tunnel | 0.344 (g18/p45) | 0.332 (g18/p50) | 0.250 (g18/p68) |
| monkey | 0.253 (g30/p46) | 0.308 (g30/p25) | 0.351 (g30/p47) |

**ΔScore vs v2-ON + Gate C(≥−0.03):**

| doc | v3-ON Δ | v3-OFF Δ | v3-ON gate | v3-OFF gate |
|---|---|---|---|---|
| pieman | -0.236 | -0.088 | ❌ | ❌ |
| tunnel | -0.012 | -0.094 | ✅ | ❌ |
| monkey | +0.055 | +0.099 | ✅ | ✅ |

## 타깃 진단 (fix 실제 작동?)

### B②: 목표 수 · 상시(span≥0.6)목표 수 (v3 Stage-1 제한 → 줄어야)

| doc | v2-ON (frozen baseline) (n목표/상시/발화율) | v3-ON (redesign, reasoning) (n목표/상시/발화율) | v3-OFF (redesign, no-reason) (n목표/상시/발화율) |
|---|---|---|---|
| pieman | 9/2/27% | 7/1/25% | 6/3/38% |
| tunnel | 5/4/21% | 10/3/18% | 12/3/25% |
| monkey | 23/7/30% | 8/5/17% | 12/2/19% |

### H: space no-cue 문장 non-unspecified율 (v3 → 줄어야)

| doc | v2-ON (frozen baseline) | v3-ON (redesign, reasoning) | v3-OFF (redesign, no-reason) |
|---|---|---|---|
| pieman | 3.0% (n67) | 1.4% (n69) | 1.5% (n67) |
| tunnel | 5.3% (n394) | 1.4% (n367) | 2.8% (n362) |
| monkey | 10.8% (n249) | 2.0% (n247) | 1.9% (n259) |

### E: non-absent time 라벨 문중(mid) 비율 (v3 → jump 문중 줄어야)

| doc | config | sequence mid | jump mid | simult mid |
|---|---|---|---|---|
| pieman | v2-ON (frozen baseline) | 14/16 (88%) | 0/1 (0%) | 0/0 |
| pieman | v3-ON (redesign, reasoning) | 1/4 (25%) | 2/3 (67%) | 0/0 |
| pieman | v3-OFF (redesign, no-reason) | 2/4 (50%) | 0/1 (0%) | 0/0 |
| tunnel | v2-ON (frozen baseline) | 0/5 (0%) | 3/8 (38%) | 0/0 |
| tunnel | v3-ON (redesign, reasoning) | 1/3 (33%) | 3/6 (50%) | 0/0 |
| tunnel | v3-OFF (redesign, no-reason) | 2/7 (29%) | 0/1 (0%) | 0/0 |
| monkey | v2-ON (frozen baseline) | 6/11 (55%) | 3/11 (27%) | 1/1 (100%) |
| monkey | v3-ON (redesign, reasoning) | 7/16 (44%) | 4/5 (80%) | 0/0 |
| monkey | v3-OFF (redesign, no-reason) | 8/16 (50%) | 2/5 (40%) | 0/0 |

### A: 탈것/공중 목록누락 문장 수 (dev-3 항공서사 없음 → ~0, regression 확인)

| doc | v2-ON (frozen baseline) | v3-ON (redesign, reasoning) | v3-OFF (redesign, no-reason) |
|---|---|---|---|
| pieman | 0 | 0 | 0 |
| tunnel | 1 | 1 | 1 |
| monkey | 1 | 1 | 1 |

## 해석 / 판정

_(데이터 수집 후 서술)_

## 한계

- A: dev-3 에 항공/탈것 서사 없어 A fix 효과는 여기서 안 드러남(regression 확인만). A 실증은 repcorpus obs_77176.
- v3-OFF 는 프롬프트+no-reasoning 동시 변경 → v3-ON arm 으로 귀속 분리. Gate C 판정은 배포후보(v3-OFF)로.
- 타깃 진단 = gold-free regex 스크리닝(②진단 동급). 방향·규모용, 정밀도 아님.

