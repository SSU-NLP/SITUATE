# M4/M5 slot-alignment probe — Phase-1 앵커 (seed 0)

> 지표 정의 = Phase 2 `probe_slot_alignment.py` 와 **동일**(비교 가능). 프로브는 앵커 코퍼스 안에서만 학습·평가 — dev-3·시험지 미사용(D-B16).

프로브 학습 150편 / 평가 150편 · 앵커 발화율 prota 0.309 space 0.105 time 0.022 causa 0.697 inten 0.113


## M5 — probe AUPRC [slot k → axis j] (대각=자기 축, 비대각=교차 누출)

| slot\axis | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot prota | **0.483** | 0.201 | 0.047 | 0.770 | 0.165 |
| slot space | 0.343 | **0.432** | 0.048 | 0.731 | 0.173 |
| slot time | 0.336 | 0.204 | **0.280** | 0.761 | 0.179 |
| slot causa | 0.367 | 0.228 | 0.061 | **0.822** | 0.171 |
| slot inten | 0.352 | 0.190 | 0.036 | 0.742 | **0.306** |

## M5 — EXCESS off-axis leakage (라벨 상관 천장 B 제거)

| slot\axis | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot prota | own 0.483 | +0.096 | +0.026 | +0.065 | +0.049 |
| slot space | +0.032 | own 0.432 | +0.024 | +0.033 | +0.044 |
| slot time | +0.027 | +0.096 | own 0.280 | +0.062 | +0.065 |
| slot causa | +0.050 | +0.122 | +0.037 | own 0.822 | +0.058 |
| slot inten | +0.040 | +0.070 | +0.012 | +0.045 | own 0.306 |

## M4 (monolithic) — slot-0 → 각 축 (얽힘)

| | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot 0 | 0.470 | 0.434 | 0.319 | 0.799 | 0.299 |

## 판정

- M5 own-axis (대각): prota 0.483 space 0.432 time 0.280 causa 0.822 inten 0.306
- **M5 평균 EXCESS 누출 = +0.0527** (≈0 = 슬롯이 축에 특화)
- **M4 평균 excess = +0.2050** (높을수록 단일 잠재가 전 축을 얽음)
- **factoring gain = M4 − M5 = +0.1523** (>0 = D1 라우팅이 monolithic 이 얽는 것을 분리)

> 참조: Phase 2(10편·오염 Tier2 7편 포함) = M5 +0.072 / M4 +0.114 / gain **+0.042**.

⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16).
