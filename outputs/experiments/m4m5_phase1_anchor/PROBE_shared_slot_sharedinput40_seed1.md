# M4/M5 slot-alignment probe — Phase-1 앵커 (seed 1)

> 지표 정의 = Phase 2 `probe_slot_alignment.py` 와 **동일**(비교 가능). 프로브는 앵커 코퍼스 안에서만 학습·평가 — dev-3·시험지 미사용(D-B16).

> 평가된 체크포인트: M6 `M6_shared_slot_sharedinput40_seed1.pt` · M4 `M4_per_slot_slot_seed1.pt`  (표의 'M5' 행은 --ns 가 가리키는 조건의 factored 모델이다 — 이름이 아니라 경로를 본다)

프로브 학습 200편 / 평가 200편 · 앵커 발화율 prota 0.323 space 0.106 time 0.022 causa 0.699 inten 0.113


## M5 — probe AUPRC [slot k → axis j] (대각=자기 축, 비대각=교차 누출)

| slot\axis | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot prota | **0.435** | 0.456 | 0.340 | 0.788 | 0.304 |
| slot space | 0.429 | **0.458** | 0.339 | 0.787 | 0.305 |
| slot time | 0.432 | 0.456 | **0.338** | 0.788 | 0.304 |
| slot causa | 0.426 | 0.460 | 0.339 | **0.786** | 0.302 |
| slot inten | 0.426 | 0.460 | 0.333 | 0.786 | **0.304** |

## M5 — EXCESS off-axis leakage (라벨 상관 천장 B 제거)

| slot\axis | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot prota | own 0.435 | +0.349 | +0.319 | +0.079 | +0.190 |
| slot space | +0.104 | own 0.458 | +0.314 | +0.090 | +0.176 |
| slot time | +0.110 | +0.348 | own 0.338 | +0.087 | +0.190 |
| slot causa | +0.092 | +0.356 | +0.314 | own 0.786 | +0.188 |
| slot inten | +0.101 | +0.339 | +0.310 | +0.086 | own 0.304 |

## M4 (monolithic) — slot-0 → 각 축 (얽힘)

| | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot 0 | 0.463 | 0.459 | 0.349 | 0.791 | 0.313 |

## 판정

- M5 own-axis (대각): prota 0.435 space 0.458 time 0.338 causa 0.786 inten 0.304
- **M5 평균 EXCESS 누출 = +0.2070** (≈0 = 슬롯이 축에 특화)
- **M4 평균 excess = +0.2117** (높을수록 단일 잠재가 전 축을 얽음)
- **factoring gain = M4 − M5 = +0.0047** (>0 = D1 라우팅이 monolithic 이 얽는 것을 분리)

> 참조: Phase 2(10편·오염 Tier2 7편 포함) = M5 +0.072 / M4 +0.114 / gain **+0.042**.

⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16).
