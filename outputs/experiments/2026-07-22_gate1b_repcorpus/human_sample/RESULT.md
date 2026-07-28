# ①B human 층화표본 — 채점 결과 (RESULT)

> 블라인드 예/아니오 100문항(축당 20). 사람이 문장+맥락(한국어)만 보고 라벨 → 모델과 대조.
> 합격선 = 축별 **model-wrong(✗) ≤ 20%** (SCORING_RUBRIC.md 동결). 질적 설계 발견은 **FINDINGS.md** 참조.
> 채점 = `human_sample_score.py --stage1` + 불일치 19건 문면 판정 + 비서사/병합 5건 분리.

## 판정 = **PASS (전 축)**

| 축 | n | 일치율 | 모델-중립 정당성 | ✗ model-wrong | △ ambiguous | ✗율(전체) | ✗율(제외) | 판정 |
|---|---|---|---|---|---|---|---|---|
| space | 20 | 70% | **12/15 (80%)** | 2 | 4 | 10.0% | 10.0% | ✅ |
| intentionality | 20 | 75% | **13/15 (87%)** | 0 | 5 | 0.0% | 0.0% | ✅ |
| time | 20 | 95% | **18/19 (95%)** | 0 | 1 | 0.0% | 0.0%(n=18) | ✅ |
| protagonist | 20 | 95% | 2/3 (67%,소표본) | 0 | 1 | 0.0% | 0.0%(n=19) | ✅ |
| causality | 20 | 70% | **11/15 (73%)** | 0 | 6 | 0.0% | 0.0%(n=18) | ✅ |

- **✗율(제외)** = 비서사 4(F)·병합 1(G) 제외 후. model-wrong 2건은 둘 다 space라 제외 전후 동일.

## 핵심 결과 — QC의 원질문에 대한 답

**"이 none/unspecified 는 진짜 근거 부재인가, 모델 회피인가?" → 대부분 정당(회피 아님).**
- 모델이 중립(none/unspecified/absent/ongoing)을 낸 문장 중 **사람도 "근거 없음"에 동의 = 73~95%.** 즉
  **모델의 중립은 대체로 정당**하다. **깨끗한 '회피'(근거 명백한데 모델이 none) 사례 0건** — 중립 불일치는
  전부 **어휘공백(비행기)·진짜 애매**였지 모델이 명백한 걸 회피한 게 아니다.
- **모델의 실제 오류 모드 = 회피가 아니라 과잉귀속(over-attribution).** 유일한 model-wrong 2건(space)이
  둘 다 **문장에 장소 단서가 없는데 특정 위치를 부여**(context-leak). dominant_actor 때 우려한 "회피"와 반대.

## 불일치 19건 판정 요약

- **✗ model-wrong = 2 (전부 space context-leak, 신규 발견 H):**
  - `fam_london_fire#214` "…cheeks and nose were freezing." → 문장에 장소단서 0인데 모델 **yukon_trail** 부여.
  - `fam_doyle_speckled#166` "…sent for medical aid from the village." → village는 언급뿐인데 모델 **julia_bedroom** 부여.
  - ⇒ **space 라벨이 target-only 를 소프트하게 위반(맥락 누출).** 회피의 반대 방향 오류. → FINDINGS H.
- **△ ambiguous = 17** (사람 "애매" 8 + 판정상 애매 9):
  - space: 비행기 무대 목록밖 3(A), 함교 함선정체 불명 1.
  - intentionality: ① 대사 화제/목적(collect_birds) 1, ② control_self=기질·enjoy_leave=회색 2, 무대밖 저격 1, 요리사≠먹이기 1.
  - time: **"First" 가 sequence 목록(then/next/after that)에 없음 1** → E 보강(목록 불완전).
  - causality: 아스펙트 애매(변화 유추·이산vs지속) 5 + 1.
- **human-wrong = 0.**

## 표본 오염 (F/G 실측, 5건 분리)

- **비서사 4(F):** #56(time, "Produced by…" boilerplate)·#60(time, "THEN CLAWED BACK…" 대문자 헤딩)·
  #63(protagonist, 출판사 광고)·#82(causality, 제목/byline).
  - **🔴 #60 실례:** 대문자 헤딩이 "THEN"으로 시작해 **모델이 time=sequence 로 오라벨** → 사람과 "false 일치".
    비서사가 축 라벨을 오염시킨 구체 증거(F+E 상호작용).
- **병합 1(G):** #90(causality, obs_25744#203) 대사+구분선+갈아입기가 한 문장.
- 이 5건은 추출 품질(✗율)이 아니라 **코퍼스 위생**을 재므로 ✗율에서 제외(결과 불변 — 어차피 PASS).

## 사람의 독립 검증 값

사람이 스스로 "애매"로 표시한 8건이 **내가 라벨링 중 지목한 설계 이슈와 독립적으로 일치**:
enjoy_leave(②), control_self(②), collect_birds(①), 저격 귀속, 비행기 목록밖(A), 눈덮임 아스펙트. →
**FINDINGS 의 이슈들이 사람 눈에도 실재**함을 교차검증(숫자로는 안 나오는, human 표본의 값).

## 해석 / 한계 (제거 금지)

- 축당 20, n=100(제외 후 95). 축 단위 결론은 소표본 노이즈. protagonist 모델-중립 n=3 소표본.
- 예/아니오(중립정당성) 모드라, "정확 라벨" 일치가 아니라 "중립 여부" 일치를 잼(정밀 라벨 검증 아님).
- ✗ 2건(space context-leak)은 target-only 위반이나, **맥락상 옳은 위치**라 boundary 에 해롭다곤 단정 못 함
  (별도 진단 필요). 회피 부재 = 긍정 결과.

## ①B 결론

human 표본 = **PASS(전 축 ✗≤20%)**, 모델 중립 대체로 정당(회피 아님), 오류모드=과잉귀속(space 2건).
→ ①B QC 마지막 항목(human 표본) 충족. **①B 종료 선언 가능**(REPORT.md §7 + ledger D-B11).
설계 발견(A~H)은 다음 VERSION 재료(FINDINGS.md).
