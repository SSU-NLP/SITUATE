# §4.4 sticky-CRP + §4.5 우도비 스위칭 — M5_per_slot seed 0

> 정수 임계(t\*=3) 대체 실험. **재학습 없음** — 학습된 슬롯 상태·f_k 재사용.
> 캘리브레이션(σ_k²·onset)은 phase1 val 에서만. dev-3 gold 미사용(D-B16).
> α=1.0 · CRP β=1.0 · `scrp.sticky_crp_unnormed`(SEM 식1) 재사용.


## λ 민감도 — 사전이 inert 하지 않은지 (2026-05-22 재발 방지 게이트)

| λ | dev-3 pooled Score | 예측 경계 수 |
|---|---|---|
| 0 | 0.3248 | 281 |
| 1 | 0.3248 | 281 |
| 3 | 0.3248 | 281 |
| 10 | 0.3262 | 280 |
| 30 | 0.3273 | 278 |
| 100 | 0.3219 | 275 |

**게이트: ✅ 사전이 결정을 바꾼다 (inert 아님).**

## 문서별 (λ=30, 최고 Score 행)

| doc | Score | 예측 | gold |
|---|---|---|---|
| pieman | 0.3906 | 35 | 18 |
| tunnel | 0.3335 | 70 | 18 |
| monkey | 0.2579 | 173 | 30 |

## 미구현 (정직 기록)

- §4.5 schema reuse 항 — §4.3(스키마 조건부 dynamics) 미구현이라 의미 없음. continue-vs-new 2지만.
- §4.7 particle smoothing — 전진 local MAP(식 8)만.
- §4.2 슬롯별 이종 분포족 — 전 슬롯 등방 가우시안(식 6). σ_k² 만 축별.

⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16).

---

## §4.7 A1 적용 결과 — 우도가 노이즈에서 신호로 바뀌었다

**변경점.** onset 을 "경계 직후 슬롯 상태의 **단순 평균**"(naive numerator) 에서 §4.7 A1 의
**amortized onset prior q_onset^(k)** 로 교체 — 세그먼트 첫 문장들로부터 **분포**(대각 공분산,
SEM 식 2 형태)를 적합하고, §4.6 *"Auto-normalization via likelihood (not point error)"* 에 따라
**정규화 상수 `−½Σ_j log σ²_j` 를 포함**한다.

🔴 **직전 구현의 실제 버그가 이 상수였다.** "비율에서 소거된다"며 버렸는데, 그건 두 분포의 분산이
같을 때만 참이다. 넓은 분포는 대가를 치러야 하는데 그 항이 없으니 "중심점에 가깝다"가 공짜로 이겨
ll 이 상시 양수(+0.8~+2.3)로 굳었다.

| 지표 | D-B18 (naive centroid) | **A1 (분포 + 정규화)** |
|---|---|---|
| 우도비 ll 평균 | +1.81 | −35.2 |
| **우도비 ll 표준편차** | **0.25** | **125.3** |
| **gold 경계 vs 비경계 ll 차이** | (신호 없음) | **+44.2** (경계 +0.6 / 비경계 −43.6) |
| dev-3 pooled Score (최고) | 0.294 | **0.327** |
| λ 거동 | **절벽** 603 → 349 → 0 | **안정** 281 → 275 |

**판정: A1 은 의도대로 작동했다.** 우도가 판별 신호를 나르기 시작했고 절벽이 사라졌다.

## 남은 문제 — 이번엔 우도가 사전을 압도한다

| | 값 |
|---|---|
| 우도비 표준편차 | **125.3** |
| 사전항 변동폭 (λ=10, C_prev 1→50) | **1.7** |

sticky-CRP 가 다시 무력해졌다(λ 0→100 에서 경계 281→275). 그리고 **과분절**이다
(281개 vs gold 66개). Score 0.327 은 여전히 percentile 규칙 0.41 아래.

**원인 = §4.2 미구현.** 설계 §4.5 는 *"비율이므로 차원·스케일 불일치가 대체로 상쇄된다 —
normalization 은 **emergent**"* 라고 했는데, 지금은 5축 전부 **같은 형태의 대각 가우시안**으로
오지정돼 있어 그 emergent normalization 이 일어나지 않는다. §4.2 원문:
> Broad-variance slots (causal) are unsurprised by ordinary internal jumps; usually-narrow slots
> (place) are very surprised when they move. **This normalizes each slot by its own normal
> variability** — the reason we insist on a probabilistic (not raw-error-sum) formulation.

## 다음
2. ⬜ **§4.6 Axis2 slot-native 예측** — `L_pred` 를 슬롯 공간에 직접 걸어 f_k 를 실제로 학습.
   현재 f_k 는 우연 수준이라 continue 분포가 지나치게 넓고, 그것이 과분절의 직접 원인이다.
3. ⬜ **§4.2 슬롯별 이종 분포족** — 축마다 다른 분포족(protagonist = point process 등).
   우도 스케일이 사전을 압도하는 문제의 근본 해결.
