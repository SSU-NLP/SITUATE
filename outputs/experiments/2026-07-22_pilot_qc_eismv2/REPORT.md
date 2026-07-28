# ①B 비시험 파일럿 QC — frozen p0s5-eism-v2 (dev-3, zero-API 부분판)

> 작성 2026-07-22. 러너 `scripts/eacl_phase0/pilot_qc.py`. 산출 `qc.json`(동 디렉토리).
> 맥락: 관문① 동결(D-B9)이 사전등록 §4의 ①B(비시험 파일럿 QC)를 **선행하지 않고** 먼저 일어난
> **순서 역전(검증 전 동결)** 상태. 본 QC는 그 역전을 **부분적으로 사후 검사**한다 — dev-3 캐시에서
> API 없이 뽑히는 기계-무결성·축-liveness 하위집합만. **전체 ①B 아님**(§5 한계).
>
> **🔴 판정 = FAIL (causality 축 over-fire). 통과로 바꾸지 않는다.** QC를 돌리고 FAIL이 나온 뒤
> 채점 기준을 고쳐 통과시키는 것은 D-M5(시험 후 채점기준 변경 → 데이터 7개 무효)의 반복이다. 기준
> (OVERFIRE=FAIL, 임계 0.70은 **실행 전** 설정)은 그대로 두고, **떨어졌다고 적은 뒤 옆에 "알고도
> 진행하는 이유"**를 쓴다. "알고도 진행"은 방어 가능하고, "기준을 바꿔 통과"는 방어 불가능하다.
> (경위: 최초 이 REPORT는 OVERFIRE를 WARN으로 강등해 "PASS"라 적었으나 — FAIL 직후의 기준변경이라 —
> 사용자 지적으로 원복. 이 문단은 그 사고를 숨기지 않고 남긴다.)

## 1. 실험 setup

- **목적:** frozen 추출기(`p0s5-eism-v2`, deepseek-v4-pro)가 dev-3에서 **완전·정합·비퇴화** 출력을
  냈는지 캐시만으로 검사. 사전등록 §4 ①B 지표 중 캐시-계산 가능분.
- **데이터:** dev-3 eismv2 캐시 `outputs/eacl_phase0/{pieman,tunnel,monkey}_michelmann_s5_eismv2/`
  (n_items 96 / 504 / 408; gold_density 0.189 / 0.036 / 0.074). 로그 `round1_pieman.log`,
  `round1_tm_parallel.log`(tunnel·monkey 공유).
- **frozen 축(count):** `protagonist_present, space, time, causality, intentionality` @ t\*=3
  (= `eval_seg_score.FROZEN_AXES`, D-B10 — protagonist_perspective 아님).
- **판정 기준(실행 전 고정, 결과 보고 변경 안 함):** provenance(model·prompt_version 단일성)
  + coverage(item_id 0..n−1 gap/중복 0) + **no over-fire(어느 축도 base rate ≤ 0.70)**. 셋 다 만족해야
  PASS. 0.70 임계는 구 causality churn 실패모드를 겨냥해 **실행 전** 설정.
- **known 주석(pass 안 뒤집음):** 떨어진 조건이 동결 시점에 이미 문서화됐으면 그 사실을 옆에 적는다.
  기준을 완화하는 게 아니라, "불합격이지만 알던 문제라 알고도 진행"을 기록으로 남기는 것.
- **비교 baseline:** 없음(무결성 절대검사). 축 liveness는 REPORT §5 / D-M13④⑤ 대조.

## 2. 결과

**판정:**

| doc | ①B 판정 | 떨어진 조건 |
|---|---|---|
| pieman | **PASS** | — |
| tunnel | **FAIL** | causality over-fire (base 0.761 > 0.70, lift 0.95) |
| monkey | **PASS** | — |
| **전체** | **🔴 FAIL** | tunnel causality |

**하위 점검(참고 — 셋 다 clean, 그러나 위 over-fire로 전체 FAIL):**

| doc | provenance | coverage | gold_dens | dens@t\* | 로그(504 / no_valid / trace) |
|---|---|---|---|---|---|
| pieman | v2·deepseek 단일 ✅ | 96/96 전축 ✅ | 0.189 | 0.137 | 0 / 0 / 0 |
| tunnel | v2·deepseek 단일 ✅ | 504/504 전축 ✅ | 0.036 | 0.040 | 99* / 0 / 0 |
| monkey | v2·deepseek 단일 ✅ | 408/408 전축 ✅ | 0.074 | 0.076 | 99* / 0 / 0 |

\* 504는 tunnel·monkey 공유 로그의 합(중복). 전부 재시도 복구(valid 실패 0).

**축 liveness (base rate / boundary lift):**

| axis | pieman | tunnel | monkey | 판정 |
|---|---|---|---|---|
| protagonist_present | .368 / **2.85** | .133 / **3.65** | .236 / **1.80** | 전편 살아있음·진단적 |
| space | .074 / 3.21 | .062 / 5.18 | .022 / **0.00** | monkey 死(무대정지) |
| time | .179 / 2.33 | .026 / 2.25 | .057 / 0.57 | monkey 반진단 |
| causality | .516 / 1.24 | **.761 / 0.95** | .619 / 0.91 | over-fire·lift≈1 |
| intentionality | .147 / 2.38 | .089 / 1.25 | .233 / 1.82 | 살아있음 |

## 3. 해석

- **떨어진 것 = tunnel causality over-fire (base 0.761 > 0.70, lift 0.95).** 실행 전 설정한 over-fire
  기준을 tunnel causality가 넘김 → **전체 ①B = FAIL.** 이 축은 관계적 구성개념이라 문장마다 발화(기저율
  잡음)해 경계 진단가치가 없다(lift≈1). **동결 시점에 이미 문서화된 문제**(REPORT §5 — "causality 제거
  4축이 근소 최선"), 새로 발견된 결함 아님. frozen이 EISM 완결성·타 벤치(dialseg에선 살아있음) 때문에
  **알고도 유지한** 축 → **알고도 진행**(기준 완화 아님).
- **provenance·coverage 는 clean.** 전 행이 단일 `p0s5-eism-v2` + deepseek-v4-pro(혼합·stale 없음),
  item_id gap/중복 0, 로그 `no_valid_outputs=0`·`traceback=0`. 추출기 `while len(done)<n` 루프가
  schema-strict 출력을 100% 수렴시킴 확인. **단 이 하위점검 통과가 over-fire FAIL을 상쇄하지 않는다.**
- **504 재시도 정책이 실효.** deepseek origin이 부하 시 504를 다수 반환하나 6회·백오프(20~120s)가
  전량 복구 → 스케일에서 **지연·비용 상방 리스크**로만 남고 데이터 손실 아님.
- monkey **space lift 0.00·time 0.57**(over-fire 아니라 판정엔 무영향) = journalism(무대 정지)에서
  상황축 死 = scope 증거(D-M13④⑤). liveness 전반이 dev-3 frozen 분석과 동일 패턴 = QC가 회귀 안 만듦.

## 4. 판정

- **①B (dev-3 캐시 하위집합) = 🔴 FAIL** (tunnel causality over-fire). **PASS로 바꾸지 않음.**
- **그래도 진행하는 이유:** 떨어진 조건이 **동결 시점(Gate C·REPORT §5)에 이미 알려지고 문서화된**
  causality 축의 관계적-구성개념 한계다. 새 결함이 아니라 **알고도 동결에 넣은** 속성. 따라서 "알고도
  진행"으로 기록 — **채점 기준을 고쳐 통과시키는 것과 다르다.** ("알고도 진행"은 심사 방어 가능,
  "기준 변경 통과"는 불가능.)
- **폐기+재등록 게이트(D-B9)는 여전히 살아있음.** 본 QC는 ①B를 **완결하지 않는다**(§5). 스케일-전이
  실패(대표 코퍼스에서 **다른/새 축**의 dead·over-fire, 또는 비용·결정성·human 미검분)가 나오면 사전등록
  §4대로 **프롬프트 패치 아님 → frozen v2 폐기 + 새 VERSION 재등록.** (causality over-fire는 이미
  알던 것이라 그 자체로 재등록 트리거는 아님 — 새로 나타나는 문제가 트리거.)

## 5. 한계 / 검증 미해결 (제거 금지)

1. **dev-3 ≠ 대표 비시험 코퍼스.** 3편·서사픽션·이미 튜닝 소진(pieman 다중 selection, REPORT §4).
   ①B 사전등록이 요구한 "대표 비시험 코퍼스"는 Phase-1 코퍼스(미구축) 필요. **본 QC는 파일럿의 파일럿.**
2. **캐시-only = 과소검사.** 파싱-거부율·재시도율·**비용/문장**·**출력결정성(temp-0 재실행 일치)**은
   캐시에 없음 → 미검(로그 504/valid만 대리). 결정성은 재추출(API) 필요.
3. **human 층화표본 QC 미실시**(사전등록 §4 명시 항목).
4. **over-fire 임계 0.70 = 실행 전 설정(사후 아님).** 구 causality churn 실패모드를 겨냥해 QC 스크립트
   최초 작성 시 못박음 → 결과 보고 고르지 않았다. tunnel 0.761이 이를 넘겨 FAIL. 다만 0.70 자체의
   보정 근거(왜 0.6·0.8 아닌지)는 약하므로, 스케일에서 재보정 시 **그것도 결과 보기 전에** 고정해야 함.
5. 본 QC(pass/fail 무관) ≠ 성능 근거. Score·성능 주장에 인용 금지(무결성/liveness 진단 전용).

## 6. 다음 (관문② 스케일 ①B 완결 조건)

대표 코퍼스 확보 후: (a) 코퍼스 스케일 liveness(신규 dead/over-fire 축 = 진짜 게이트) (b) 재추출
결정성 + 비용/문장 계측 (c) human 층화표본. **(a)~(c) 실패 시 = frozen v2 폐기 + 재등록**(D-B9).
러너 `pilot_qc.py`는 `--doc/--dir/--log`로 코퍼스에 그대로 재실행.
