# causality requires/produces 프로토타입 — 결과 (RESULT)

> 사전등록 `PRE_REGISTRATION.md`(통과조건 결과 전 봉인). frozen Qwen3.6-35B-A3B no-thinking v2, temp=0,
> 통제 서사 13문장(`design_passage.jsonl`), N=5. 스크립트 `scripts/eacl_phase0/causality_reqprod_proto.py`,
> 원자료 `proto_result.json`.

## 판정 = **두 변형 모두 FAIL** (사전선언 조건 기준)

| 조건(사전선언) | V-closed(닫힌목록) | V-free(자유서술) |
|---|---|---|
| 빈집합 < 40% | **0.55 ✗** | 0.00 ✓ |
| N회 경계일치 > 0.85 | 0.93 ✓ | **0.80 ✗** |
| 인과연쇄 → 연속 ≥ 0.8 | **0.69 ✗** | **0.44 ✗** |
| 장면전환 → 경계 = 2/2 | 1.0 ✓ | 1.0 ✓ |
| **종합** | **FAIL** | **FAIL** |

## 실패 모드 = 사전 예상 그대로

- **V-closed = dominant_actor 재현(빈집합 55%)**: 닫힌 목록에서 requires/produces를 고르는 게 어려워
  모델이 절반 넘게 **빈 답으로 도피**([]). produces[t-1]이 비면 교집합이 항상 ∅ → **인과연쇄 안에서 가짜 경계**
  → 연속 판정 0.69로 미달. 목록 크기·빈집합이 run마다 흔들림(empty 0.31~0.69).
- **V-free = 매칭 붕괴(연속 0.44) + 불안정(0.80)**: 빈답은 없으나, "bars_cut"↔"slip through gap"처럼
  **표현이 갈려 토큰 교집합이 안 터짐** → 인과연쇄를 경계로 오판(연속 0.44). 경계집합도 run마다 5~11개로
  요동(S3 표현변이 예측 그대로).
- **공통 성공**: 장면전환(감방마당→사무실, 사무실→담벼락) 2/2 경계 = 둘 다 맞춤. 즉 **큰 단절은 잡는데
  "인과 연속을 유지"하는 게 안 됨** — 정확히 이 메커니즘의 존재 이유가 실패.

## 결론 / 다음

- **requires/produces 채택 안 함**(사전등록 조건 위반, D-M5 회피 규약대로 결과 보고 기준 안 바꿈).
  이론(인과-연결 기준)은 옳으나, 이 라벨러 규모에서 **매칭 가능한 requires/produces를 신뢰성 있게 못 뽑음**
  (closed=빈답 도피 / free=표현변이·불안정).
- **fallback(사전등록 명시) = causal_marker(명시 인과 접속만) 로 후퇴 또는 Phase X 이월.** causal_marker는
  enum이라 안정적이나 암시적 연쇄(톱질→탈출)를 못 잡는 한계 명시. 채택 판정은 **Gate C DTS Score**(별도 실험).
- 상위 문제(event_type-change 65% 포화)의 직접 처방은 여전히 **reset 규칙 ablation**(event_type-change 제거,
  time축 선례 tense-change 제거→Score↑와 동형)이며, 이건 Gate C Score로만 판정.

## 한계

- 통제 13문장 = 소표본·hand-crafted(낙관 편향). 그런데도 FAIL이라 신호는 강함(실전은 더 어려움).
- Stage-1 목록 비결정성(±30%)이 V-closed 불안정에 그대로 기여. 더 큰/추론 모델이면 빈집합이 줄 수는 있으나
  frozen 라벨러 범위 밖(별도 결정·비용).
