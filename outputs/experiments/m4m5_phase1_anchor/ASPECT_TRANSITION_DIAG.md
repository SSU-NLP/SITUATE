# 사전등록 aspect 전이 기반 확증적 리셋 진단 (D-B21) — tag `q36_35b_v2_r1`

> 🔴 이것을 "결정→행동→완료 검증"이라 부르지 않는다(D-B21 명칭 규정).
> 전이표·기준·중단조건은 **결과 보기 전** D-B21 에 고정됨. 사후 수정 없음.
> 규칙: `aspect_reset = (직전 == completive) AND (현재 ∈ {inceptive, ongoing})`, `event_type ≠ 직전` 항 제거.


## 사전등록 3조건 (3/3 충족해야 채택)

| # | 조건 | 기준 | 관측 | 판정 |
|---|---|---|---|---|
| ① | lift 95% bootstrap CI 하한 | > 1.0 | **0.492** (점추정 1.611, 상한 3.118) | ❌ |
| ② | 발화율 | [0.15, 0.4] | **0.059** | ❌ |
| ③ | ΔScore (5축 count≥3) | ≥ -0.01 | **-0.0694** (0.4170 → 0.3475) | ❌ |

**판정 = ❌ 실패 — D-B21 중단조건 발동**

## 대조 — 동결 규칙

| | 발화율 | lift |
|---|---|---|
| 동결 `event_type≠직전 OR aspect==inceptive` | 0.667 | 1.073 |
| **신규 aspect 전이** | **0.059** | **1.611** |
| *참조: D-B13 #5 반사실창* | *0.309* | *CI하한 0.758 (FAIL)* |

## 전이 분포 (서술통계)

| 전이 | 건수 | 비율 |
|---|---|---|
| ongoing → ongoing | 784 | 0.780 |
| ongoing → completive | 53 | 0.053 |
| completive → ongoing **(RESET)** | 52 | 0.052 |
| inceptive → ongoing | 45 | 0.045 |
| ongoing → inceptive | 44 | 0.044 |
| completive → completive | 10 | 0.010 |
| completive → inceptive **(RESET)** | 7 | 0.007 |
| inceptive → completive | 6 | 0.006 |
| inceptive → inceptive | 4 | 0.004 |

> gold 66/1005 (밀도 0.066). 부트스트랩 = 문서 내 층화, 10,000회, percentile 95% CI.

⚠️ 검정력: gold 66개뿐이라 낮다. CI 하한 실패는 "효과 없음"이 아니라 **채택 근거 불충분**이며, 검정력 부족을 이유로 임계값을 사후 완화하지 않는다(D-B21).

⚠️ 앵커 k=1 · 탐색적 — dev-3 = 연습장(D-B16).
