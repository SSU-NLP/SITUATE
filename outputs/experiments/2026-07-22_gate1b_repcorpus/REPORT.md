# ①B 스케일 QC — 대표 코퍼스 REPORT (D-B11)

> 작성 2026-07-22. 사전등록 = 동 디렉토리 `PRE_REGISTRATION.md`(ledger D-B11로 동결). 러너 =
> `fetch_repcorpus.py`(획득·기계선정) · `run_repcorpus_extract.sh`/`run_repcorpus_run2subset.sh`(추출) ·
> `repcorpus_qc.py`(QC). 산출 `qc.json`·`manifest.json`·`selection_log.json`.
> **주지표 = 집단별(유명/무명) 기계지표 + 결정성. 성능(Score) 아님. gold 경계 없음(Phase-1도 무라벨).**

## 1. 실험 setup

- **목적:** frozen `p0s5-eism-v2`(deepseek-v4-pro)가 Phase-1(무명작 다수)에서 어떻게 작동할지 미리 보는
  대표 코퍼스 ①B QC. dev-3 파일럿(유명 3편·캐시only) 보완. **암기(memorization) 통제 = 유명 vs 무명 대조.**
- **데이터:** 12편(작가 12명 전원 다름), 각 ≤250문장(비용 cap). 유명 6 = **연구자 지정**(London·Bierce·
  Gilman·Doyle·Jewett·Twain, 19세기 문학, id 2429/375/1952/1661/74980/3189). 무명 6 = **Gutendex 대체
  (PG RDF 덤프)의 download_count 하위10% 기계선정**(작가 무지목; id 37511/28922/74162/25744/74140/77176,
  대부분 20세기 pulp SF·아동서). 1인칭 = 유명 2(Gilman·Doyle)+무명 2(37511·28922). 격리: Phase-1 학습
  재사용 금지, 시험지 미접촉, Tier2/Poe 배제. 총 2245문장.
- **추출:** frozen v2, temp0, batch20, deepseek. **run1 = 전 12편**(reset_matrix 2233 전이). **run2 =
  결정성용, 4편만**(fam_gilman·london·doyle + obs_28922, 850 전이) — 트림 사유 §5.
- **지표(집단별):** provenance(prompt_version·model 단일) · coverage(item_id 완결) · anchor base rate(축별
  reset율) · **결정성 = run1 vs run2 reset-column 재현율**(frozen reset 결정이 재현되나; 보조필드 아님) ·
  retry/504. gold 없어 **boundary lift는 n/a**. over-fire 임계 0.70 = dev-3와 동일(실행 전 고정).
- **baseline:** 없음(무결성·재현 절대검사). 판정규칙(D-B11): 유명만 좋으면 암기의존 → Phase-1 예측치=무명.

## 2. 결과

### 2.1 집단별 (base rate = anchor 양성률; reset-determ = frozen reset 재현율)

| 축 | 유명 base | 유명 determ | 무명 base | 무명 determ |
|---|---|---|---|---|
| protagonist_present | 0.372 | 0.794 | 0.268 | 0.942 |
| space | 0.203 | 0.819 | 0.088 | 1.000 |
| time | 0.038 | 0.973 | 0.036 | 0.981 |
| **causality** | **0.707 ⚠OVERFIRE** | **0.679** | 0.461 | **0.563** |
| intentionality | 0.151 | 0.855 | 0.108 | 0.864 |
| **경계(count≥t\*) 재현** | | **0.865** | | **0.981** |
| count(0–5) exact 재현 | | 0.442 | | 0.447 |

- provenance 두 집단 **전 행 `p0s5-eism-v2`·deepseek 단일** ✅ / coverage **완결** ✅ (혼합·gap 0).
- retry/504(run1) = 0(로그상 최종), tracebacks 0. **단 추출 중 origin 504로 job 2건 FAIL → 수동 복구**(§5).
- 결정성 run2∩ = 유명 747전이(3편)·무명 103전이(1편).

### 2.2 유명 vs 무명 headline (base rate 격차)
protagonist 0.372/0.268 · **space 0.203/0.088(2.3×)** · time 0.038/0.036 · **causality 0.707/0.461** ·
intentionality 0.151/0.108. **전 축 유명 > 무명.**

## 3. 해석

- **기계 건전성 = 두 집단 통과.** 처음 보는 무명 pulp·아동서까지 provenance·coverage clean =
  추출기가 다양한 문체·미열람 텍스트에서 **기계적으로 안 깨짐**(①B "스케일에서 작동하나"의 핵심 통과).
- **🔴 causality = 유명서 over-fire(0.707) FAIL + 최저 재현(0.679/0.563).** over-fire는 무명(0.461)선
  안 남 = **장르 의존**(문학 산문이 인과표지·event_type churn 多). 관계적 구성개념의 기저율 잡음(REPORT
  §5, dev-3와 동일) + temp0인데도 축 중 **재현 최저** = **이중 약점**. 새 결함 아님, 알고도 진행.
- **결정성 = 완벽 아님(deepseek reasoning 모델의 temp0 비결정성).** 축별 reset 재현 0.56~1.0, **경계
  결정 재현 유명 0.865·무명 0.981.** 즉 같은 문장이 두 run서 ~14%(유명)·2%(무명) 경계판정이 뒤집힘.
  ⇒ **anchor 노이즈 floor 존재.** 이는 M4~M7 학습신호에 잡음 → 사전등록 Phase-1의 **"다중샘플 다수결"
  anchor 필요성을 실증**(단일 추출 anchor는 ~14% 흔들림). count(0–5) exact 재현 0.44는 5축 오차 누적 탓
  (경계는 count≥3 임계라 개별 흔들림에 강건 → 0.87~0.98).
- **🔴 암기검정 = inconclusive (교란).** 전 축 유명>무명이나, **무명 기계선정이 20세기 pulp·유명이
  19세기 문학**이라 "암기 vs era/genre" 분리 불가(space 2.3× 격차가 암기인지 문학산문 배경밀도인지 미상).
  기계선정이 era를 못 맞춘 정직한 결과 — 손으로 맞추면 선택편향 재유입이라 안 함(트레이드오프).

## 4. 판정

- **기계 무결성(provenance·coverage) = PASS(양 집단).** 스케일 추출 파이프라인 건전.
- **frozen over-fire 기준(0.70) = FAIL(유명 causality 0.707)** — **기존 known 문제**(REPORT §5), 알고도
  진행. 무명은 pass. 기준 안 바꿈(D-M5 교훈).
- **결정성 = 측정만, 사전 임계 없어 pass/fail 미선언**(post-hoc 임계 금지). boundary 재현 0.87~0.98 =
  moderate, causality 최저. **anchor 노이즈로 기록** → Phase-1 다중샘플 다수결로 대응.
- **암기 = inconclusive**(교란). Phase-1 예측치를 "무명 숫자"로 확정 못 함(무명이 교란된 표본이라).
- **🔴 D-B9 폐기+재등록 게이트 = 미발동.** ①B 스케일서 **새/다른 축의 dead·over-fire·파싱붕괴 없음**
  (causality over-fire는 기존 known). 따라서 frozen v2 폐기 트리거 아님. **단 결정성(anchor 잡음)은 신규
  발견 = Phase-1 anchor 설계(다중샘플)로 처리할 사안**(재등록 아님).

### 4.1 결정성 처방 부적합 명시 (2026-07-23 추가 — 판정 불변)

결정성 실패의 원인은 **프롬프트 결함이 아니라 모델 속성**이다 — deepseek-v4-pro는 reasoning 모델이라
temp0에서도 run마다 내부 추론 경로가 달라진다. 따라서 사전등록의 "폐기 + 새 VERSION 재등록" 처방은
*프롬프트 결함*을 전제하므로 이 실패 유형에 **부적합**하다(프롬프트를 새로 써도 모델 속성은 안 바뀜).
이는 **처방의 부적합을 인정**하는 것이지 기준을 완화하는 것이 아니다. 사전등록이 이 실패 유형(모델
속성 기인)을 예상하지 못했음을 명시한다. **실제 완화책 = Phase-1 anchor 에 k=3 다중샘플**(별도 설계 변경).

검토 과정에서 결정성에 0.95 합격선을 소급 적용해 FAIL 로 기록하자는 제안이 있었으나, **결과를 본 뒤
임계를 신설하는 것은 post-hoc 기준 도입**(D-M5 · `no-postresult-criterion-change`)이므로 채택하지 않았다.
결정성은 §4 대로 "측정만, 사전 임계 없음"을 유지한다 — 이것이 "왜 임계가 없나"에 대한 답이다.

## 5. 한계 / 검증 미해결 (제거 금지)

1. **🔴 암기검정 교란(핵심):** 무명(20c pulp) vs 유명(19c 문학) era/genre 차이가 암기효과와 얽힘 →
   유명>무명을 암기로 못 돌림. 깨끗한 검정 = **era/genre 맞춘 유명/무명 쌍** 별도 프로브 필요(후속).
2. **유명 6 = 임의(연구자) 선정** → 약근거(§1.1 사전등록). 판정 무게는 무명에 두려 했으나 무명도 교란(1).
3. **run2(결정성) = 6편 트림 → 실제 4편(3유명+1무명)만 완료.** origin(api.ssunlp) 밤샘 불안정(504 반복·
   응답지연으로 run1 3h→3.5h+, run2 무명 250문장 2편 미완). 비용·불안정으로 완료분 최종화 = **사전등록
   "2회 full" 이탈**(정직 기록). 무명 결정성 = obs_28922 1편(103전이)뿐이라 무명 재현수치 표본 작음.
4. **gold 없어 boundary lift·Score 불가** — anchor 양성률·재현만. Phase-1도 무라벨이라 설계상 정합이나,
   "경계를 잘 잡나"는 이 QC로 답 못 함(별개).
5. **소표본:** 집단당 6편(결정성 무명 1편), ≤250문장 cap. 축·문서 단위 결론은 노이즈.
6. **cost/문장 미계측**(토큰 회계 없음). retry/504는 로그 proxy(최종 0이나 과정 504 다수).
7. **over-fire 0.70·무명 십분위 = 실행 전 고정이나 근거 약**(0.70 왜 그 값, decile 왜 10%). 스케일 재보정 시 결과 前 고정.

## 6. 다음 (관문② 후속)

- **암기 재검정(우선):** era/genre 맞춘 유명/무명 쌍 프로브(1) — 이번 negative(inconclusive)의 정공 해소.
- **anchor 잡음 대응:** Phase-1 코퍼스 추출 시 **다중샘플 다수결 + 신뢰도**(결정성 0.87~0.98 실증이 근거).
- **causality:** over-fire+최저재현 이중약점 재확인 → 4축(causality 제외) 대안은 REPORT §5 이미 문서화.
- run2 무명 결정성 보강(obs_25744·74140)은 origin 안정 시 저비용 추가 가능.

## 7. ①B 종료 선언 (2026-07-23)

human 층화표본(마지막 미완 항목) 완료로 ①B QC 8항목을 모두 판정, **①B 종료.**

| 항목 | 결과 | 근거 |
|---|---|---|
| 파싱률 | ✅ PASS | tracebacks 0 |
| schema 유효율 | ✅ PASS | 전 행 유효 |
| provenance | ✅ PASS | 전 행 `p0s5-eism-v2`·deepseek 단일 |
| coverage | ✅ PASS | item_id 완결(gap 0) |
| 축 base rate | ⚠ causality over-fire(유명 0.707) = **known FAIL-but-proceed** | §4, 기준 안 바꿈(D-M5) |
| 결정성 | **측정만, 사전임계 없음**(post-hoc 금지) | §4·§4.1: reasoning=모델속성 → 폐기+재등록 부적합, 완화책 k=3 |
| **human 표본** | ✅ **PASS** | `human_sample/RESULT.md`: 전 축 ✗(model-wrong) ≤ 20%(최대 space 10%) |
| 비용/문장 | 미계측(토큰 회계 없음) | §5.6 |

**human 표본 핵심(RESULT.md):** ① 전 5축 ✗≤20% PASS. ② **모델의 none/unspecified 는 대체로 정당(회피 아님)**
— 모델-중립 문장의 73~95%를 사람도 "근거 없음" 동의, **깨끗한 회피 0건**(dominant_actor 우려 부재). ③ 모델 오류모드
= 회피의 반대, **과잉귀속**(space context-leak 2건 = 유일한 model-wrong). ④ 비서사 4(F)·병합 1(G) 분리(결과 불변).

**🔴 폐기+재등록 게이트(D-B9) = 미발동.** ①B 스케일서 새/다른 dead·over-fire·파싱붕괴 없음, human 표본도 PASS.
frozen v2 폐기 트리거 아님. **결정성(anchor 잡음)은 Phase-1 다중샘플로, 설계발견은 다음 VERSION으로 처리**(재등록 아님).

**설계 발견 A~H (human 표본 산출, `human_sample/FINDINGS.md`):** space 어휘공백(A)·context-leak(H) 양방향 ·
intentionality 장면vs상시목표(②)·대사 화제/목적(①,가설) · time opens-with 비대칭(E,확정) · 코퍼스 위생
앞뒤잡물(F)·장면구분선 병합(G). **전부 다음 VERSION 재료** — reasoning-off pre-check 통과로 Gate C 재오픈 시
프롬프트 수정(A·B·E·H)을 fold, 전처리(F·G)는 Stage-0 독립. 진단으로 확증 후 반영(①은 가설, A/②는 스케일 빈도).

**⇒ ①B QC 종료.** 다음 = reasoning-off pre-check(별도 실험) → 결정성 판정 → (통과 시)Gate C 재검증 → Part 2.
