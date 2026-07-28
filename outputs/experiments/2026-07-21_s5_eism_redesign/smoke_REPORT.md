# 스모크 v2 결과 (관문①-c §3.0) — deepseek eismv2, 문서별-축

> 실행 2026-07-21. monkey/space + pieman/intentionality 재추출(신 프롬프트), `smoke_eval.py` 평가.
> 주지표 = A↔B 리셋 F1(vs fable5). 리셋 수·none-rate는 보조(사용자 정정).

## 결과

| doc/축 | A↔B F1 (주) | dsv4pro→신 | none/unspec | 리셋 수 | count≥3 혼합 |
|---|---|---|---|---|---|
| pieman/intentionality | **0.437→0.519 (+0.081) ✅** | | 0.760→**0.792**(↑) | 10/95 (fable5 17) | 0.105 = dsv4pro 0.105 |
| monkey/space | **0.467→0.480 (+0.013) △** | | 0.963→**0.855**(↓, fable5 0.824) | 9/407 | 0.025 vs dsv4pro 0.022 |

## 해석

### 1. 🔴 사용자 예측이 (부분적으로) 실증됨 — 그러나 방향은 반대
- **과잉발화 우려는 나타나지 않음.** intentionality none-rate가 오히려 **올라감**(0.760→0.792), 리셋 수 낮음(10, 9), 폭주(≥40) 없음. none-보존 문구가 **안전**(오히려 약간 보수적). → 우려한 "none 소멸·리셋 폭증"은 발생 안 함.
- **대신 "rate 내렸는데 F1 그대로" 패턴이 space에서 실증.** unspecified 0.963→0.855(수리처럼 보임)인데 F1 0.467→0.480(거의 평평). **rate만 봤으면 "고쳤다" 오판했을 것.** F1-주지표(사용자 주장)가 이를 잡음.

### 2. 정성 확인이 space F1-평평을 재해석 (결정적)
monkey space 불일치 46/408을 분류:
- **둘다 specified 다름 = 5개** (대부분 **granularity**: `wikimania_conference_london`↔`london_conference_facility`, `davids_home_wales`↔`david_slaters_home`). 의미상 같은 곳, 라벨 세분화 차이. (1개만 오류: →england 과일반화.)
- **fable5 specified인데 신=unspecified = 27개** (잔여 과침묵 — 여전히 못 잇는 자리).
- **fable5 unspecified인데 신=specified = 14개** — 샘플: `[17] "A full face portrait…" → sulawesi_rainforest`, `[22] "They were curious." → sulawesi_rainforest`, `[44] "The monkeys were taking selfies." → sulawesi_rainforest`. = **deepseek이 숲 장면 연속성을 fable5가 놓친 자리에서 올바르게 이음.**

⇒ **space F1이 평평한 건 "실패"가 아니라, deepseek이 fable5와 *다르게*(때로 더 낫게) 이었기 때문.** A↔B F1은 **fable5=정답 가정**인데, 신 라벨러가 fable5를 능가하면 F1이 낮게 나온다 → **A↔B-F1-vs-fable5 자체가 한계 있는 대리지표**(이번 스모크의 2차 발견).

### 3. count≥3 구조는 둘 다 회복 안 함 (dsv4pro 수준 유지)
단일 축만 고쳐선 boundary 구조가 안 움직임(예상 범위 — 혼합이라). 나머지 4축이 옛 dsv4pro라 당연.

## 판정

- **intentionality 문구 = 안전·소폭 개선** (F1 +0.081, 과잉발화 없음). ✅ 채택 가능.
- **space 문구 = 기전은 방향대로 작동**(연속 장면 올바르게 이음, granularity 차이 위주, 과침묵 27 잔여)이나 **A↔B-F1로는 판정 불가**(fable5 초과분을 F1이 못 구분).
- **핵심: 스모크가 "안전"은 확인(과잉발화·폭주 없음), "품질"은 A↔B-F1 한계로 미결.** → **진짜 판정 = Gate C vs 사람 gold**(전 5축 Round-1 추출 필요).
- deepseek 파국적 실패(옛 −0.25) 재현 아님 → **R1 아님.**

## 다음 (사용자 결정)
- (A) full Round-1(dev-3 5축 전부) 재추출 → **Gate C vs 사람**으로 실판정. (권고)
- (B) space 문구 1회 더 수정(granularity 억제·과침묵 27 겨냥) 후 재스모크.
- (C) 기타.
