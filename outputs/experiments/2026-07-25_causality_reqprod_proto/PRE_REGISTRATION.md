# causality requires/produces 프로토타입 — 사전등록 (결과 보기 전 확정)

> 작성 2026-07-25. **결과 수치를 보기 전에** 통과조건·설계사례·변형을 고정한다. 확정 후 수정은 하단
> "수정 로그"에 append(삭제 금지). 근거 규약: `[[no-postresult-criterion-change]]`(D-M5 재현 회피),
> S3 교훈(자유서술 표현변이 → 닫힌 목록 = G3), dominant_actor 실패(목록선택 어려우면 빈답 도피 87% none).

## 0. 배경·문제

현 `reset_causality = (event_type≠직전) OR (aspect=inceptive)` 는 event_type-change(65%)로 **포화**.
event_type-change 는 장면경계가 아니라 **문장유형 교대**(대사↔묘사↔행동)라 소설의 정상 리듬을 경계로 오판.
Zacks/EISM 기준 = "이 절의 활동이 직전 활동에 의해 **야기되지 않을 때** 경계" = 종류가 아니라 **인과 연결**.
"톱질(physical_action)→탈출(movement)"은 종류가 달라도 인과 연속 → 현 규칙은 경계로 찍음(정확히 거꾸로).

## 1. 제안 메커니즘 (Indexter plan causal_link 방식)

Stage 1 (문서 전체 1회): `states` = 이야기에서 참/거짓이 바뀌는 원자 조건 목록
  (snake_case, 예: bars_intact, has_saw, inside_cell, phone_ringing...).
Stage 2 causality (문장별): 닫힌 목록에서 선택 —
  `requires` ⊆ states : 이 행동이 성립하려면 참이어야 할 조건(복수 가능, 없으면 [])
  `produces` ⊆ states : 이 행동이 참/거짓으로 만드는 조건(복수 가능, 없으면 [])
코드 판정: `produces[t−1] ∩ requires[t] ≠ ∅` → **인과 연결(경계 아님)**, 교집합 ∅ → 경계.
매칭 = **집합 교집합**(문자열/의미 비교 LLM 불필요).

## 2. 변형 (둘 다 시도 — 어느 쪽이 안정적인지가 핵심)

- **V-closed**: 위 닫힌 목록 방식.
- **V-free**: requires/produces 를 자유서술(짧은 구)로. 매칭 = 정규화 문자열/토큰 중첩.
  (S3 예측 = V-free 가 표현변이로 불안정할 것. 이걸 실측으로 확인.)

## 3. 설계 사례 (기대 라벨 = 결과 보기 전 고정)

18문장 통제 서사(`design_passage.jsonl`). 명확 케이스만 채점:
- **인과연쇄 transition → 기대=연속(link)**: 탈옥 사슬(examine→saw→break→push→drop), 사무실 사슬
  (ring→answer→warn→shout), 담벼락 사슬(sprint→scale→drop). 유형은 달라도 인과 연속.
- **장면전환 transition → 기대=경계(∅)**: 감방마당→사무실(5→6), 사무실→담벼락(9→10).
- 그 외(유형교대·주변묘사)는 관찰만, 채점 제외.

## 4. 🔴 통과 조건 (사전 선언 — 하나라도 실패 시 기각)

1. **빈 집합 비율 < 40%**: (requires 또는 produces 가 빈) 문장 비율. ≥40%면 dominant_actor식 도피 = 자체 실패.
2. **N회(≥5) 반복 경계집합 일치율 > 0.85**: run 간 경계 라벨 평균 일치(1 − mean pairwise Hamming, 또는 Jaccard).
3. **인과연쇄 → 연속 판정**: 설계 인과연쇄 transition 중 "연속" 비율 (기준 ≥ 0.8).
4. **장면전환 → 경계 판정**: 설계 장면전환 transition 중 "경계" 비율 (기준 = 2/2 맞아야, 소표본).

- 하나라도 실패 → **causal_marker(명시 인과)로 후퇴** 또는 **Phase X 이월**. requires/produces 채택 안 함.
- 통과 시에도 채택 아님 — **Gate C DTS Score**(dev 측정·test 봉인)가 최종 판정. 이 프로토는 "매칭이 신뢰성
  있게 되는가"의 사전 게이트일 뿐.
- 라벨러 = frozen Qwen3.6-35B-A3B no-thinking v2, 로컬 vLLM(:8001). temperature=0.

## 5. 한계 (제거 금지)

- 통제 서사 18문장 = 소표본. 실제 소설의 미묘한 인과는 더 어려움(프로토 통과≠실전 통과).
- Stage 1 목록 자체가 비결정(±30%) → 조건 2가 이걸 직접 잼. V-closed 라도 "어떤 state가 목록에 오르나"가
  run마다 달라 매칭이 흔들릴 수 있음.
- 채점 사례가 hand-crafted라 낙관 편향 가능 → 조건 통과해도 Gate C 전엔 채택 금지.

## 수정 로그
- (없음 — 초판)
