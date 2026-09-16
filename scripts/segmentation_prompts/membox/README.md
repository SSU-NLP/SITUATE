# Membox 프롬프트 4종 — 논문 부록 전사(verbatim)

출처 = **Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents**
(Tao, Ma, Huang, Jiang — Tsinghua/Xinjiang, arXiv **2601.03785v3**, 2026-06-24)
전사 원본 = arXiv e-print LaTeX 소스 `acl_latex.tex` Appendix "Prompt Templates"
(Table 9~12 = `tab:information_extraction`, `tab:gg_construction_prompt`,
`tab:relation_pruning`, `tab:choose_prompt`).

🔴 **공식 코드 없음.** 2026-08-01 확인 — `TsinghuaC3I/Awesome-Memory-for-Agents`,
`xiaowu0162/awesome-long-memory` 둘 다 Membox 를 등재하지만 **paper 링크만 있고 code 배지가 없다.**
논문 본문·초록·comments 어디에도 저장소 URL 이 없고 저자 이메일만 있다.
⇒ **이 디렉토리의 프롬프트가 재현의 유일한 1차 자료다.**

| 파일 | 논문 이름 | 용도 | 플레이스홀더 |
|---|---|---|---|
| `msg_continuation.md` | PROMPT_MSG_CONTINUATION | 온라인 경계 판정 — Yes / No / Partially Shifted | `ref`, `curr` |
| `dialog_extract.md` | PROMPT_DIALOG_EXTRACT | 봉인된 박스에서 keywords·topic·events 추출 | `{text}` |
| `trace_event_filter.md` | PROMPT_TRACE_EVENT_FILTER | 새 이벤트를 기존 trace 에 붙일지 판정 | `{content_a}`, `{content_b}` |
| `trace_init.md` | PROMPT_TRACE_INIT | 이벤트 집합에서 새 trace 생성 | `{events}` |

## 전사 규칙 (지킨 것)

- **본문 문구는 한 글자도 바꾸지 않았다.** LaTeX 마크업(`\textbf{}`, `\\[0.5em]`, `\hspace{1em}`,
  `` ` `` 따옴표)만 제거하고 굵게/들여쓰기를 마크다운으로 옮겼다.
- JSON 블록(`verbatim` 환경)은 **줄바꿈·들여쓰기까지 그대로**다.
- 플레이스홀더는 논문 표기를 유지했다 — 원문이 `\texttt{ref}`(중괄호 없음)와
  `\{\texttt{text}\}`(중괄호 있음)를 **섞어 쓴다**. 그대로 뒀고, 코드에서 채울 때만 통일한다.
- 예시(Example 1/2)는 프롬프트의 일부이므로 **전부 포함**했다. 빼면 few-shot 조건이 달라진다.

## 논문이 명시한 실행 설정 (재현 시 고정)

| 항목 | 값 | 출처 |
|---|---|---|
| 슬라이딩 윈도 | **연속 2 메시지**(user 발화 1 + agent 응답 1), 현재 미봉인 박스의 최근분 | §Topic Loom |
| 봉인 조건 | 판정이 **partial shift 또는 discontinuous** 이면 봉인 후 새 박스 | 〃 |
| 최소 문맥 규칙 | 새 박스는 가능하면 **인접 응답 1개 이상**을 포함 | 〃 |
| 임베딩 | `text-embedding-3-small` | §Experiments |
| 검색 | 박스 표현(messages·topic·events·keywords) 유사도 → **박스 top-k=10** | 〃 |
| trace 확장 | 후보 이벤트에서 **event top-k=2** → 그 이벤트를 포함한 trace 를 문맥에 추가 | 〃 |
| 백본 | GPT-4o / GPT-4o-mini (메모리 구축·QA 동일 백본) | 〃 |
| 모드 | `Membox-Compact`(박스만) · `Membox-Trace`(trace 확장) | 〃 |

## 논문이 보고한 LoCoMo 수치 (재현 대조 목표)

| 백본 | 방법 | Multi-Hop | Temporal | Open Domain | Single Hop |
|---|---|---|---|---|---|
| GPT-4o-mini | Mem0 | 38.72 | 48.93 | 28.64 | 47.65 |
| GPT-4o-mini | **Membox-Compact** | 39.88 | 58.03 | 27.96 | 60.09 |
| GPT-4o-mini | **Membox-Trace** | 41.19 | 59.63 | 30.36 | 61.18 |
| GPT-4o | **Membox-Compact** | 48.35 | 65.06 | 30.61 | 61.69 |
| GPT-4o | **Membox-Trace** | 50.48 | 66.61 | 38.77 | 62.56 |

초록 기준 요약: Topic-Loom-only 가 Mem0/A-MEM 최적 검색깊이 대비 **+13.00 F1**(53.95 vs 40.95),
trace 확장 시 **55.28**, GPT-4o 에서 **59.71**.
baseline 은 저자들이 **직접 로컬 배포**해 검색깊이 k∈{5,10,20,30} 중 **최고점**을 보고했다.

⚠️ Adversarial 카테고리는 정답 라벨 부재로 **제외**됐다. 지표 = F1 + BLEU-1.

## 🔴 재현 시 주의 (미명시 항목 = 우리가 정해야 하는 것)

- **연속성 판정의 출력 파싱**이 미명시다. "Partially Shifted" 를 봉인으로 처리한다는 것은 본문에
  있으나, 모델이 형식을 벗어난 답을 냈을 때의 처리는 없다.
- **박스 표현의 임베딩 방법**(4개 필드를 이어붙이는지, 따로 임베딩해 합치는지) 미명시.
- **라벨 분포 ablation 이 없다.** 논문은 "대부분의 전환이 discontinuous 가 아니라 partial shift 로
  라벨링된다"고 서술하지만 **빈도표를 싣지 않았다.** 우리 재현에서는 이 분포를 반드시 찍는다 —
  세 라벨 중 둘을 같은 동작(봉인)에 사상하므로, 실질적으로 **이진 분류기**일 가능성이 있다.
- QA 프롬프트는 부록에 없다(LoCoMo/DialSim 원 논문 것을 쓴다고만 밝힘).
