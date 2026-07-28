# Stage-0 junk 분리집계 — 입력정제(N) vs 실제 라벨오류(M−N)

> 목적: 정제본(`repcorpus_master_stage0.jsonl`) 재추출 후 새 human-verify 를 할 때, 개선분을
> **"라벨러가 좋아졌다"로 오인하지 않기** 위해, 옛 human-verify(dirty 코퍼스, 100문항)에서 나온
> 라벨 문제를 **입력 잡물 유래(N)** 와 **진짜 라벨오류(M−N)** 로 분리한다.
> 근거 규약: `[[no-postresult-criterion-change]]`, `[[dev-test-seal-before-tuning]]`.
> 방법 = 옛 100문항 표본의 각 (doc, item_id) 를 Stage-0 정제 인벤토리(`STAGE0_AUDIT.txt`)와
> **객관 교차대조**(수기 판정 아님).

## 요약

| 구분 | 수 | 성격 | Stage-0 처리 | 새 정제본에서 |
|---|---|---|---|---|
| **N (입력정제)** | **7 / 100** | 표본 문장 자체가 잡물(광고·PG크레딧·title-page·구분선융합) | DROP/SPLIT 로 제거·분할 | **구조적으로 사라짐** |
| **M−N (실제 라벨오류)** | **2 / 100** | 정상 서사 문장인데 모델이 틀림 (space 과잉귀속) | 무관(정상서사) | **잔존** — v3/라벨러 과제 |

- 옛 RESULT.md 는 오염을 **수기 5건**으로 셌으나, 객관 교차대조는 **7건**(구분선 2·boilerplate 5).
  차이 2건 = 라벨을 뒤집지 않아 수기에서 안 잡힌 것(space 구분선 `obs_74140#70`, 2번째 광고행 `obs_37511#42`).
  둘 다 **분명히 비서사**이므로 N 에 포함(더 엄밀).
- 옛 disagreements 19건 중 진짜 model-wrong 은 **2건뿐**(둘 다 space), 나머지 17 = ambiguous, 0 = human-wrong.

## N = 입력 잡물 유래 (7) — Stage-0 가 제거

객관 교차대조: 옛 표본 (doc, item_id) ∈ Stage-0 정제 대상.

| 축 | doc#item | Stage-0 연산 | 문장(발췌) |
|---|---|---|---|
| space | `obs_74140#70` | SPLIT(구분선) | "When things happen to people … `* * * * *` They summoned courage…" |
| space | `obs_37511#42` | DROP(서점광고) | "Also Manufacturers of every description of ACCOUNT BOOKS, MEMORANDUMS…" |
| time | `obs_28922#0` | DROP_HEAD(PG크레딧) | "Produced by Greg Weeks, … Online Distributed Proofreading Team…" |
| time | `fam_twain_frog#2` | DROP_HEAD(풍자 title-page) | "THEN CLAWED BACK INTO A CIVILIZED LANGUAGE ONCE MORE BY PATIENT…" |
| protagonist | `obs_37511#43` | DROP(서점광고) | "Their Assortment of SCHOOL AND MISCELLANEOUS BOOKS…" **(=옛 No.74)** |
| causality | `obs_77176#0` | DROP_HEAD(제목/바이라인) | "IN MID-AIR By R. [J. McSwiney]" |
| causality | `obs_25744#203` | SPLIT(구분선) | "…best for my own people." `* * * * *` "Returning to his assigned apartment…" |

- **`fam_twain_frog#2`(time)** = 옛 verify 의 대표 오염: 대문자 title-page 가 "THEN"으로 시작해
  모델이 **time=sequence** 로 오라벨 → 사람과 "false 일치". 비서사가 축 라벨을 오염시킨 구체 증거.
- **`obs_37511#43`(protagonist)** = 서점 광고문에 protagonist 억지 배정(옛 No.74). Stage-0 DROP.

## M−N = 진짜 라벨오류 (2) — Stage-0 무관, 잔존

둘 다 **space 과잉귀속(context-leak)**: 문장 자체엔 장소 단서가 없는데 모델이 이야기 전역 배경에서
위치를 역으로 채움 = target-only 소프트 위반. 회피(none 남발)의 **반대 방향** 오류.

| doc#item | 모델(Qwen v2) | 사람 | 문장 · 진단 |
|---|---|---|---|
| `fam_london_fire#214` | `yukon_trail` | no | "But it was surprising, the rapidity with which his cheeks and nose were freezing." — 장소어 0. 작품 전역 무대(유콘)를 채움. |
| `fam_doyle_speckled#166` | `julia_bedroom` | no | "…he poured brandy down her throat and **sent for medical aid from the village**…" — 유일 장소어 "village"는 의료 요청 출처일 뿐 무대 아님. 앞 죽음-장면에서 침실 추론. |

## 새 verify 에서 지킬 것 (오인 방지)

1. **새 정제본 verify 의 ✗율을 옛 raw 와 직접 비교 금지.** 옛 raw 는 잡물 7 을 포함/일부만 제외.
   비교 기준선 = 옛 **✗(제외) = space 2/20(10%), 그 외 0%**. 이래야 잡물 제거분과 라벨러 변화가 안 섞임.
2. **space 과잉귀속(M−N=2)이 새 verify 에서도 남으면** = Stage-0 로 안 고쳐지는 **진짜 라벨 이슈** 확정
   → v3 프롬프트가 "target sentence 에 명시 장소 없으면 unspecified" 제약을 강화해야 할 대상.
3. 새 표본(100)은 **정제본 pool 에서 재추출**(동일 seed 20260723) — 잡물 문장 0 확인됨. 즉 새 verify 는
   **깨끗한 입력 위 라벨 품질**만 잰다.

## 한계

- 옛/새 표본은 **동일 문장 집합이 아님**(인덱스 shift·라벨 변화로 pool 상이) → 문항별 old↔new 대조 불가.
  분리집계는 "옛 verify 가 드러낸 문제의 귀속"이며, 새 verify 는 독립 측정.
- N=7 은 옛 100 표본 한정. 코퍼스 전체 잡물은 40연산(Stage-0)이 제거(감사 `STAGE0_AUDIT.txt`).
