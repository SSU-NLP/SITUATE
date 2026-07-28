# ④-fine: 개별 fix 기여 (타깃 지표 primary, Score 안전장치)

> base = Qwen3.6-35B-A3B(no-think). 채택 = (타깃↑ 의도방향) ∧ (Score −0.09 이내). **Score 상승은 채택근거 아님**
> (σ0.06<0.09). 각 fix = 영향축만 재추출, v2 다른 축 재조합. 단일 라벨러 caveat.

## Score 안전장치 (dev-3 deployable, v2=q36_35b_v2)

| fix | pieman Δ | tunnel Δ | monkey Δ | 평균 Δ | −0.09 이내? |
|---|---|---|---|---|---|
| space-H | +0.004 | -0.028 | +0.021 | -0.001 | ✅ |
| time(E+conseq+first) | +0.000 | +0.000 | +0.012 | +0.004 | ✅ |
| space-A | -0.024 | -0.024 | +0.018 | -0.010 | ✅ |
| intent-② | +0.097 | -0.017 | +0.003 | +0.028 | ✅ |

## 타깃 지표 (주 판정) — v2 → fix

### space-H: no-cue 문장 non-unspecified율 (↓ 의도) — dev-3
- **전체: v2 5.9%(42/716) → fix 3.6%(26/716)**. 개선=✅
- per-doc (v2 nu → fix nu / n): pieman: 6→4/67 ; tunnel: 14→11/393 ; monkey: 22→11/256

### time (E+conseq+first) — repcorpus 12편 (dev-3은 non-absent 13개로 표본부족→repcorpus로 제대로)
| marker | v2 문중 | fix 문중 |
|---|---|---|
| sequence | 9/26 (35%) | 4/29 (14%) |
| **jump** | 4/6 (67%) | 5/10 (50%) |
| simultaneity | 1/2 (50%) | 0/1 (0%) |

- **전체 non-absent 문중율: v2 41.2%(14/34) → fix 22.5%(9/40)**. 개선=✅
- jump 문중(E 핵심타깃): v2 67% → fix 50%. 개선=✅
- So/And yet→sequence(conseq): v2 0 → fix 0. 개선=✅
- First→sequence 검출(first): v2 0/1 → fix 1/1.

### space-A: obs_77176 탈것/공중 목록누락 문장 수 (↓ 의도)
- v2 conveyance-miss **30** (loc 12개, vehicle 목록포함=True) →
  fix **4** (loc 8개, vehicle 목록포함=True). 개선=✅
- fix locations: ['cockpit', 'open_sky', 'airdrome', 'enemy_lines', 'german_headquarters', 'chateau', 'route_nationale', 'no_mans_land']

### intent-②: 상시목표(span≥0.6) 수 + 총 목표수 (↓ 의도)
- 상시목표: v2 **3** → fix **5**. 총 목표: v2 43 → fix 42. 개선=❌
- 문서별(n목표/상시): v2 [('pieman', 6, 2), ('tunnel', 8, 1), ('monkey', 29, 0)] / fix [('pieman', 5, 1), ('tunnel', 9, 1), ('monkey', 28, 3)]

## 채택 판정 (타깃개선 ∧ Score −0.09이내)

_(위 두 표 종합 — 서술)_

