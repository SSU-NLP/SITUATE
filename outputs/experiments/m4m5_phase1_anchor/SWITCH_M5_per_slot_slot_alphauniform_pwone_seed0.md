# §4.4 sticky-CRP + §4.5 우도비 스위칭 — M5_per_slot_slot_alphauniform_pwone seed 0

> 정수 임계(t\*=3) 대체 실험. **재학습 없음** — 학습된 슬롯 상태·f_k 재사용.
> 캘리브레이션(σ_k²·onset)은 phase1 val 에서만. dev-3 gold 미사용(D-B16).
> α=1.0 · CRP β=1.0 · `scrp.sticky_crp_unnormed`(SEM 식1) 재사용.


## λ 민감도 — 사전이 inert 하지 않은지 (2026-05-22 재발 방지 게이트)

| λ | dev-3 pooled Score | 예측 경계 수 |
|---|---|---|
| 0 | 0.4340 | 43 |
| 1 | 0.4340 | 43 |
| 3 | 0.4340 | 43 |
| 10 | 0.4340 | 43 |
| 30 | 0.4340 | 43 |
| 100 | 0.4340 | 43 |

**게이트: 🔴 INERT — λ 를 바꿔도 출력 불변. v4.1.x 재발.**

## 문서별 (λ=0, 최고 Score 행)

| doc | Score | 예측 | gold |
|---|---|---|---|
| pieman | 0.3750 | 6 | 18 |
| tunnel | 0.5295 | 18 | 18 |
| monkey | 0.3976 | 19 | 30 |

## 미구현 (정직 기록)

- §4.5 schema reuse 항 — §4.3(스키마 조건부 dynamics) 미구현이라 의미 없음. continue-vs-new 2지만.
- §4.7 particle smoothing — 전진 local MAP(식 8)만.
- §4.2 슬롯별 이종 분포족 — 전 슬롯 등방 가우시안(식 6). σ_k² 만 축별.

⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16).
