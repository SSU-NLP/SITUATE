# ④ 구성요소별 attribution (v3 축별 fix 기여, reasoning-ON, API 0)

> v2-ON·v3-ON 축별 states 재조합. protagonist·causality 는 항상 v2(불변축). deployable Score(t\*=3).
> ⚠️ v2/v3 다른 run → 축간 nondeterminism 잔존(정밀 attribution 아닌 방향). H/A=space, E=time, B②/B①=intent.

## deployable Score (t\*=3) — 축별 fix on/off

| config | pieman | tunnel | monkey | 평균 |
|---|---|---|---|---|
| v2 (baseline) | 0.578 p16 | 0.344 p45 | 0.253 p46 | 0.392 |
| +H/A (space→v3) | 0.580 (+0.002) p15 | 0.342 (-0.002) p45 | 0.258 (+0.005) p47 | 0.393 (+0.002) |
| +E (time→v3) | 0.518 (-0.060) p13 | 0.342 (-0.001) p40 | 0.275 (+0.022) p44 | 0.378 (-0.013) |
| +B②/B① (intent→v3) | 0.462 (-0.117) p13 | 0.314 (-0.029) p52 | 0.275 (+0.022) p26 | 0.350 (-0.041) |
| full v3 (3축 모두→v3) | 0.408 (-0.170) p8 | 0.342 (-0.002) p45 | 0.308 (+0.055) p29 | 0.353 (-0.039) |
| v3 leave-out H (space→v2) | 0.454 (-0.124) p10 | 0.326 (-0.018) p48 | 0.289 (+0.036) p29 | 0.356 (-0.035) |
| v3 leave-out E (time→v2) | 0.462 (-0.117) p13 | 0.316 (-0.028) p49 | 0.286 (+0.033) p27 | 0.354 (-0.037) |
| v3 leave-out B② (int→v2) | 0.479 (-0.099) p10 | 0.348 (+0.005) p41 | 0.288 (+0.035) p44 | 0.372 (-0.020) |

## 읽는 법

- **+X 단독행** = v2에 그 축 fix 하나만 → 그 fix의 순기여(Δ).
- **leave-out 행** = full v3에서 그 축만 v2로 되돌림 → 그 fix를 뺀 효과(full 대비).
- pieman 붕괴(예측수 p) 원인 축 = +단독행에서 Score 급락 or 예측수 급변하는 축.

