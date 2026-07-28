# DTS 벤치마크 5축 효과 — dev-3 와 동일 분석 (GEE + permutation null)

> 🔴 **탐색적.** DTS 는 D-M5 에서 **scope 밖**으로 선언된 데이터 — 성능 주장에 쓰지 않는다.
> 프로젝트 기존 주장(경계조건 ③): *tiage 의 '화제 전환'은 타깃인 '상황 리셋'과 구성개념이 다르다.* 축이 약한 것이 **예측된 결과**이며 여기서 통계적 근거를 붙인다.
> 새 라벨 0 · API 0 (동결 v3.1 기추출분 재사용). 대화(doc_id) = 클러스터.



> ---
> ## 🔴 DTS 열 사용 제한 (2026-07-28 발견)
> **여기 DTS 수치는 프로젝트 공식 DTS 수치와 비교할 수 없다.** 최소 3가지가 다르다:
> 1. **부분집합** — `dts_*_master.jsonl` 은 S5 추출용 샘플이다. tiage = **20 대화**(전체 test 100),
>    dialseg711 12 · superseg 15 · ami 2.
> 2. **길이 절단** — 대화당 **정확히 15 전이**로 잘려 있다(원본은 가변 길이).
> 3. **인코더 상이** — 공식 DTS 수치(`2026-05-20_v411_tiage_test` 등)는 **`multi-qa-mpnet`**,
>    본 파이프라인은 **`paraphrase-multilingual-mpnet`**.
>
> 실제 대조: 공식 M1-CosDrift TIAGE **0.4675**(registry §2) vs 본 표 **0.3431**.
> ⇒ **이 표의 DTS 열은 내부 진단용이며, 논문·외부 비교에 인용 금지.**
> 전체 test 재실행에는 `benchmarks/{tiage,dialseg711,superdialseg_data}` 원본이 필요하나
> 현재 컨테이너에 없다(.gitignore 7~9·94~95행, 미반입).
> ---

| 데이터 | 전이 | 경계 | 밀도 | 대화 |
|---|---|---|---|---|
| tiage | 288 | 49 | 0.1701 | 20 |
| dialseg711 | 310 | 47 | 0.1516 | 12 |
| superseg | 167 | 24 | 0.1437 | 15 |
| ami | 436 | 11 | 0.0252 | 2 |
| *참조: dev-3(서사)* | *1005* | *66* | *0.0657* | *3* |

## tiage

| 축 | 오즈비 exp(β) | 95% CI | permutation p |
|---|---|---|---|
| protagonist_present | **0.80** | [0.37, 1.71] | 0.4833 |
| space | **1.12** | [0.52, 2.41] | 0.8398 |
| time | **1.00** | [nan, nan] | 0.2539 |
| causality | **1.75** | [0.89, 3.44] | 0.2737 |
| intentionality | **0.58** | [0.23, 1.46] | 0.1384 |

**p < 0.05: **없음****

## dialseg711

| 축 | 오즈비 exp(β) | 95% CI | permutation p |
|---|---|---|---|
| protagonist_present | **0.79** | [0.41, 1.52] | 0.2903 |
| space | **9.61** | [5.32, 17.35] | 0.1304 |
| time | **1.00** | [1.00, 1.00] | 0.4247 |
| causality | **0.60** | [0.28, 1.27] | 0.1928 |
| intentionality | **56.93** | [17.69, 183.21] | 0.1276 |

**p < 0.05: **없음****

## superseg

| 축 | 오즈비 exp(β) | 95% CI | permutation p |
|---|---|---|---|
| protagonist_present | **0.61** | [0.18, 2.12] | 0.3451 |
| space | **1.00** | [nan, nan] | 0.0652 |
| time | **0.00** | [0.00, 0.00] | 0.1050 |
| causality | **0.79** | [0.18, 3.55] | 0.5025 |
| intentionality **\*** | **5.69** | [1.46, 22.23] | 0.0036 |

**p < 0.05: intentionality**

## ami ⚠️ 대화 2편 — 클러스터 부족, 참고용

| 축 | 오즈비 exp(β) | 95% CI | permutation p |
|---|---|---|---|
| protagonist_present **\*** | **nan** | [nan, nan] | 0.0002 |
| space **\*** | **nan** | [nan, nan] | 0.0002 |
| time **\*** | **nan** | [nan, nan] | 0.0002 |
| causality **\*** | **nan** | [nan, nan] | 0.0002 |
| intentionality **\*** | **nan** | [nan, nan] | 0.0002 |

**p < 0.05: protagonist_present, space, time, causality, intentionality**

## 참조 — dev-3(서사 픽션) 같은 분석

| 축 | 오즈비 | permutation p |
|---|---|---|
| protagonist_present **\*** | **3.34** | 0.0005 |
| space | 2.32 | 0.0606 |
| time | 2.95 | 0.2837 |
| causality | 1.44 | 0.1113 |
| intentionality | 1.42 | 0.3460 |

⚠️ DTS 는 scope 밖(D-M5). 이 표는 **구성개념 불일치 주장의 증거**로만 읽고 성능 비교로 읽지 않는다.
