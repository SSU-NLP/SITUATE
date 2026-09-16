# M4/M5 slot-alignment probe — Phase-1 앵커 (seed 1)

> 지표 정의 = Phase 2 `probe_slot_alignment.py` 와 **동일**(비교 가능). 프로브는 앵커 코퍼스 안에서만 학습·평가 — dev-3·시험지 미사용(D-B16).

> 평가된 체크포인트: M5 `M5_per_slot_slot_seed1.pt` · M4 `M4_per_slot_slot_seed1.pt`  (표의 'M5' 행은 --ns 가 가리키는 조건의 factored 모델이다 — 이름이 아니라 경로를 본다)

프로브 학습 200편 / 평가 200편 · 앵커 발화율 prota 0.323 space 0.106 time 0.022 causa 0.699 inten 0.113


## M5 — probe AUPRC [slot k → axis j] (대각=자기 축, 비대각=교차 누출)

| slot\axis | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot prota | **0.488** | 0.174 | 0.038 | 0.766 | 0.169 |
| slot space | 0.359 | **0.455** | 0.041 | 0.727 | 0.173 |
| slot time | 0.353 | 0.196 | **0.308** | 0.752 | 0.168 |
| slot causa | 0.376 | 0.244 | 0.066 | **0.814** | 0.161 |
| slot inten | 0.348 | 0.198 | 0.030 | 0.721 | **0.316** |

## M5 — EXCESS off-axis leakage (라벨 상관 천장 B 제거)

| slot\axis | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot prota | own 0.488 | +0.067 | +0.017 | +0.057 | +0.054 |
| slot space | +0.034 | own 0.455 | +0.016 | +0.030 | +0.045 |
| slot time | +0.030 | +0.087 | own 0.308 | +0.050 | +0.054 |
| slot causa | +0.043 | +0.140 | +0.042 | own 0.814 | +0.047 |
| slot inten | +0.023 | +0.076 | +0.007 | +0.020 | own 0.316 |

## M4 (monolithic) — slot-0 → 각 축 (얽힘)

| | prota | space | time | causa | inten |
|---|---|---|---|---|---|
| slot 0 | 0.463 | 0.459 | 0.349 | 0.791 | 0.313 |

## 판정

- M5 own-axis (대각): prota 0.488 space 0.455 time 0.308 causa 0.814 inten 0.316
- **M5 평균 EXCESS 누출 = +0.0469** (≈0 = 슬롯이 축에 특화)
- **M4 평균 excess = +0.2117** (높을수록 단일 잠재가 전 축을 얽음)
- **factoring gain = M4 − M5 = +0.1648** (>0 = D1 라우팅이 monolithic 이 얽는 것을 분리)

> 참조: Phase 2(10편·오염 Tier2 7편 포함) = M5 +0.072 / M4 +0.114 / gain **+0.042**.

⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16).
