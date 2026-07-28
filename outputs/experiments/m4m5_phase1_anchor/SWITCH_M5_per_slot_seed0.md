# §4.4 sticky-CRP + §4.5 우도비 스위칭 — M5_per_slot seed 0

> 정수 임계(t\*=3) 대체 실험. **재학습 없음** — 학습된 슬롯 상태·f_k 재사용.
> 캘리브레이션(σ_k²·onset)은 phase1 val 에서만. dev-3 gold 미사용(D-B16).
> α=1.0 · CRP β=1.0 · `scrp.sticky_crp_unnormed`(SEM 식1) 재사용.


## λ 민감도 — 사전이 inert 하지 않은지 (2026-05-22 재발 방지 게이트)

| λ | dev-3 pooled Score | 예측 경계 수 |
|---|---|---|
| 0 | 0.2661 | 603 |
| 1 | 0.2940 | 349 |
| 3 | 0.2790 | 0 |
| 10 | 0.2790 | 0 |
| 30 | 0.2790 | 0 |
| 100 | 0.2790 | 0 |

**게이트: ✅ 사전이 결정을 바꾼다 (inert 아님).**

## 문서별 (λ=1, 최고 Score 행)

| doc | Score | 예측 | gold |
|---|---|---|---|
| pieman | 0.2843 | 95 | 18 |
| tunnel | 0.3115 | 254 | 18 |
| monkey | 0.2861 | 0 | 30 |

## 미구현 (정직 기록)

- §4.5 schema reuse 항 — §4.3(스키마 조건부 dynamics) 미구현이라 의미 없음. continue-vs-new 2지만.
- §4.7 particle smoothing — 전진 local MAP(식 8)만.
- §4.2 슬롯별 이종 분포족 — 전 슬롯 등방 가우시안(식 6). σ_k² 만 축별.

⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16).

---

## 진단 — 왜 절벽인가 (실측)

| 항목 | 값 |
|---|---|
| σ_k² (축별) | 1.135 / 1.033 / 1.056 / 1.101 / 1.185 |
| e_cont (continue 예측오차, 축별 평균) | 1.125 / 1.045 / 1.058 / 1.096 / 1.188 |
| **e_new (onset 까지 거리)** | **0.609 / 0.238 / 0.118 / 0.326 / 0.240** |
| **우도비 ll** | 평균 **+1.81** · 표준편차 **0.25** · 범위 [+0.8, +2.3] |
| 사전항 (C_prev = 1/5/20/50, λ=1) | −0.69 / −1.79 / −3.04 / −3.93 |

**① 우도비가 신호를 거의 나르지 않는다.** 표준편차 0.25 vs 사전항 변동폭 4.1 →
스위칭 결정의 대부분이 **"현재 에피소드가 얼마나 길었나"**로 정해지고 관측 기여는 미미하다.
그래서 λ 를 조금만 올려도 전부-아니면-전무로 갈린다.

**② onset 이 학습된 dynamics 보다 항상 잘 맞는다** (e_new 가 e_cont 의 1/2~1/9) → ll 이 늘 양수라
"무조건 switch" 로 기울어 있다. 슬롯 상태가 L2 정규화 단위벡터라 **중심점까지의 거리는 자동으로
작고**, 반면 f_k 의 예측오차는 1.1 ≈ 우연 수준(단위벡터 두 개면 ‖a−b‖²=2−2cos → cos≈0.48).

### 🔑 설계 문서가 이 실패를 예고했다
`factored-slot-eism-v0_1.md` §4.7:
> The switch **numerator** ("if we switched, how well would a *fresh* schema explain this?") is the
> expensive term — in principle an integral over the fresh-schema prior.
> **A naive uninformative numerator breaks slot normalization.**

본 구현의 onset = "경계 직후 슬롯 상태의 **단순 평균**" = 정확히 그 naive numerator 이며,
예고된 대로 슬롯 정규화가 깨졌다.

### 부수 발견 — f_k 가 슬롯 상태를 예측하지 못한다
`L_pred` 가 **768D 임베딩**을 지도했고 슬롯 상태를 직접 지도하지 않았다. f_k 는 readout 을 통한
간접 기울기만 받았으므로 슬롯 공간에서의 예측력이 우연 수준에 머문다.
⇒ §4.6 Axis2("slot-native 예측 공간 T_k")가 미구현인 것의 **정량적 대가**다.

## 다음 (이 진단이 지목하는 순서)
1. **§4.5 amortized onset** — 단순 평균 대신 **학습되는 episode-onset 분포**(새 주인공 등장·무대
   전환·시간 리셋을 담는). naive numerator 교체가 1순위.
2. **§4.6 Axis2 slot-native 예측** — `L_pred` 를 슬롯 공간에서 직접 걸어 f_k 를 실제로 학습시킨다.
3. **§4.2 슬롯별 이종 분포족 + surprise = −log p** — 등방 가우시안 + L2 정규화 조합이 거리 척도를
   무의미하게 만드는 문제의 근본 해결.

**⚠️ 단, sticky-CRP 자체는 살아 있다** — λ 민감도 게이트 통과(603→349→0). v4.1.x 의 inert
(prior-cancel 상쇄)와는 다른 상황이며, 문제는 사전이 아니라 **우도**다.
