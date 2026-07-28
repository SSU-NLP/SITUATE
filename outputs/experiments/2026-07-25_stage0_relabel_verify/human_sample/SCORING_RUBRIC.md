# ①B human 층화표본 — 채점 규칙 (시트와 동시 동결)

> seed=20260723 · 축당 20문장 × 5축 = 100문장.
> 생성 = `scripts/eacl_phase0/human_sample_build.py`. 결과 본 뒤 이 규칙 변경 금지
> (`no-postresult-criterion-change`).

## 🔴 합격선 (결과 보기 전 동결)
- **축별 model-wrong(✗) ≤ 20%.** (3단계에서 '모델이 명백히 틀림'으로 판정된 것만 카운트.)
- 검토 중 0.95 소급 임계 제안이 있었으나 post-hoc 기준이라 **불채택** — 이 spot-check 는 라벨
  타당성(특히 none/unspecified 정당성) 확인용이며 결정성 임계와 무관.

## 라벨링 방식 (2026-07-23, 사용자 선택)
- **1차 = 예/아니오 웹도구** `label_web.html`(생성 `human_sample_webpage.py --mode yesno`, Artifact 로 open).
  각 문장에 축별 질문 '이 문장에 [축]의 근거가 있나?' → 예(근거 있음)/아니오(근거 없음)/애매. 20지선다·규칙
  암기 없이 중립토큰(none/unspecified/absent/ongoing) **정당성만** 블라인드로 본다 — 사전 fallback(§합격선
  주석의 '중립값 판정만이라도')을 전 축으로 확장한 것. **합격선(≤20%)·표본·seed 불변**(기준변경 아님).
- 대안 = full 20지선다(`--mode full` 또는 sheet.csv 직접). 정밀하나 느림. 채점기는 두 방식 자동 판별.

## 블라인드 3단계 절차
1. **블라인드 라벨링:** 웹도구(`label_web.html`)에서 문장을 읽고 예/아니오/애매를 누른다(모델 답 미노출).
   끝나면 CSV 내보내기(`row_id,axis,human_label,notes`). full 방식이면 sheet.csv 의 human_label 열 채움.
2. **자동 대조:** `human_sample_score.py --stage1 --labels <내보낸csv>` → 축별 human↔model 일치율
   + `disagreements.jsonl` + `stage1_counts.json`. (yes/no: 아니오↔모델중립, 예↔모델 concrete 로 대조.)
3. **불일치 판정:** `disagreements.jsonl` 의 각 행에 `verdict` ∈ {model-wrong, human-wrong, ambiguous}
   + `error_tag`(예: pronoun-unresolved, mention-vs-stage, avoidance-none) 기입 →
   `human_sample_score.py --final` → 축별 ✗율·PASS/FAIL·오류유형 히스토그램 → RESULT.md.

## 축별 채점 대상 필드 · 중립토큰
| 축 | 라벨 필드 | 중립토큰 | choices |
|---|---|---|---|
| space | location | unspecified | 문서 locations + unspecified |
| intentionality | active_goal | none | 문서 goals + none |
| time | temporal_shift_marker | absent | absent/sequence/jump/simultaneity |
| protagonist | perspective | none | first/second/third/none |
| causality | aspect | ongoing | inceptive/ongoing/completive |

## 층화 확인 (생성 시)
| 축 | pool | 중립비율 full | 중립비율 표본 | 밀도 bin 분포 |
|---|---|---|---|---|
| space | 2234 | 0.74 | 0.75 | {'고': 2, '극저': 3, '저': 8, '중': 7} |
| intentionality | 2234 | 0.82 | 0.80 | {'고': 2, '극저': 3, '저': 7, '중': 8} |
| time | 2234 | 0.99 | 1.00 | {'고': 2, '극저': 3, '저': 7, '중': 8} |
| protagonist | 2234 | 0.12 | 0.10 | {'고': 2, '극저': 4, '저': 6, '중': 8} |
| causality | 2234 | 0.83 | 0.85 | {'고': 2, '극저': 3, '저': 7, '중': 8} |

> `is_neutral` 는 answer_key.jsonl 에만 있고 블라인드 sheet 엔 없음(anchoring 방지).
