# M5 W3 × 임계 규칙 — dev-3 deployable Score

## 실험 setup

- **모델:** M5 **W3** = ALPHA 현행 × pos_weight=one (D-B44 채택본).
  가중치 `outputs/experiments/m4m5_phase1_anchor/M5_per_slot_slot_pwone_seed{0,1,2}.pt`
  (seed0 = `models/m5_default/M5.pt`). **재학습 없음 — 임계 규칙만 바꿨다.**
- **신호:** s = Σ_k P(reset_k), **S5 5축 전부**(축 제거 없음).
- **c = 1.0 고정** (D-B39 채택값, 출처 = DTS 공식 test). 여기서 재탐색하지 않는다.
- **지표:** deployable Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD), `dts_scoring.score_dialogues`.
  경계 개수를 방법에 주지 않는다(matched@gold 아님).
- **데이터:** dev-3 3편 · gold 경계 **66**. 봉인 시험지 미개봉.

## 결과 (seed 0/1/2 평균 ±SD)

| 임계 규칙 | Score | ±SD | 예측 경계 | 과분절 배수 | Δ vs 적응임계 |
|---|---|---|---|---|---|
| adaptive (online_nf) | **0.3963** | ±0.0055 | 146 | 2.21× | — |
| ewma span=10 | **0.3615** | ±0.0032 | 195 | 2.95× | -0.0348 |
| ewma span=20 | **0.3675** | ±0.0124 | 176 | 2.66× | -0.0288 |
| ewma span=30 | **0.3683** | ±0.0021 | 168 | 2.55× | -0.0280 |
| ewma span=50 | **0.3815** | ±0.0068 | 165 | 2.51× | -0.0149 |

## 읽는 법 — 🔴 절대값 인용 금지

`configs/m5_default.json` 의 caveat 을 그대로 승계한다: **dev-3 절대 Score 를 성능으로
인용하지 않는다. Δ·방향만.** dev-3 는 연습장 3편이고 pieman 에는 선택압이 9회 누적됐다.
여기서 EWMA 가 이겨도 그것은 *dev-3 에서 더 높다* 는 뜻이지 채택 근거가 아니다 —
현행 c=1.0·`online_nf` 의 출처는 **DTS 공식 test** 이고(D-B39), 그 지위를 dev-3 수치로
뒤집으면 selection-on-dev 가 된다.

**과분절 배수**를 같이 본다. 현행 규칙의 알려진 결함이 **약 2.2배 과분절**
(dev-3 145 예측 vs gold 66)이고 경계 개수 제어는 최대 미결 과제다(D-B34/B37/B38).
EWMA 가 Score 를 올리더라도 그것이 *개수를 맞춰서* 인지 *위치를 잘 잡아서* 인지는
이 표만으로 갈리지 않는다.

