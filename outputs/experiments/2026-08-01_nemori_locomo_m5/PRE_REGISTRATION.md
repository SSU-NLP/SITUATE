# 사전등록 — LoCoMo × Nemori 분절기 교체 (2026-08-01 KST, 본 실행 **전**)

> 결과를 보기 **전에** 동결한 것만 적는다. 실행 후에는 이 파일을 고치지 않고
> `REPORT.md` 에 결과와 "동결한 것에서 벗어난 부분"을 따로 적는다.

## 1. 무엇을 묻나

Nemori(`nemori-ai/nemori`, arXiv 2508.03341)의 **LLM 배치 분절기**를 우리 **M5-FactoredSlot
분절기**로 갈아끼우면 LoCoMo 다운스트림(사건 단위 retrieve → QA)이 어떻게 되나.

우리 기여 주장은 **"사건 단위 분절"** 하나다. retrieval·QA·메모리 구조는 Nemori 것을 그대로 쓴다
(프로젝트 규약: retrieval 알고리즘을 우리가 새로 만들지 않는다).

## 2. 팔 (arms) — 차이는 분절기 클래스 **하나뿐**

| 팔 | 분절기 | 그 외 |
|---|---|---|
| `baseline` | `nemori.llm.generators.segmenter.BatchSegmenter` (원본, gpt-4.1-mini, temp 0.2, JSON 모드, 80메시지 청크) | Nemori 발표 설정 그대로 |
| `m5` | `hi_ontop.nemori_segmenter.M5BatchSegmenter` (M5-FactoredSlot I2 + 적응임계) | ← 동일 |

원본 저장소 파일은 **한 줄도 수정하지 않는다.** 교체는 `nemori.core.memory_system.BatchSegmenter`
모듈 전역 이름 치환으로만 한다. `add.py`/`search.py`/`evals.py`/`generate_scores.py` 는
`runpy` 로 **원본 그대로** 실행한다.

## 3. 동결값 (실행 전 확정 — 실행 후 재탐색 금지)

### 3.1 M5 쪽
| 항목 | 값 | 출처 |
|---|---|---|
| 체크포인트 | `models/m5_default/M5.pt` (= I2, 2026-07-31 동결 디폴트) | `configs/m5_default.json` · 사용자 결정 |
| 신호 | `s = Σ_k P(reset_k)` (5축 전부) | 동결 |
| 임계 | `online_nf`, **c = 1.0** | D-B39. c 는 **DTS 공식 test 36,109 전이**가 골랐다. LoCoMo 에서 재탐색 **안 한다** |
| 단위 | **메시지(turn) 1개 = 1단위** | 프로젝트 기존 대화 규약(`tolerance_bands.load_dialogue_val`) |
| 인코더 입력 | **원문만**, 화자 접두사 없음 | 같은 규약 (`t["utterance"]` 원문) |
| 청킹 | 없음 (대화 전체를 한 번에) | M5 는 온라인이라 컨텍스트 한계가 없다 |
| topic 문자열 | `"conversation"` | Nemori **자신의** 분절-없음/실패 폴백 값(`memory_system.py:113`, `segmenter.py:87`) |
| device | CPU | 로컬 A100 4장은 vLLM 이 점유 중. 전체 분절 60초로 충분 |

🔴 **c·단위·입력형식을 LoCoMo 점수를 보고 바꾸지 않는다.** 바꿀 일이 생기면 그 사실을
REPORT 에 "결과를 본 뒤 바꿨다"고 적고 별도 사전등록을 새로 한다.

### 3.2 Nemori 쪽 (발표 설정, 손대지 않음)
`evaluation/locomo/config.json` 그대로: LLM `openai/gpt-4.1-mini` · 임베딩
`google/gemini-embedding-001` · `buffer_size_max 20` · `episode_min_messages 1` ·
`enable_semantic_memory true` · `enable_prediction_correction true` ·
`search_top_k_episodes 10` · `search_top_k_semantic 20` · 병합 on(threshold 0.85).
검색 기본값 `--search-method vector` · `--include-original-messages-top-k 2`.
심판 = `gpt-4.1-mini`, temperature 0.

### 3.3 데이터
`benchmarks/locomo/locomo10.json` — 10 대화 · 5,882 메시지 · 1,986 QA
(채점은 카테고리 5(446문항)를 제외한 **1,540문항**; 제외는 발표판 `evals.py` 의 규칙).

## 4. 환경 때문에 어쩔 수 없이 바꾼 것 (두 팔에 **동일** 적용 · 방법 아님)

| # | 무엇 | 왜 | 방법에 영향? |
|---|---|---|---|
| E1 | 모델 경로 접두사 `openrouter/` | 사내 게이트웨이(litellm) 라우팅 규칙 | 없음 — 같은 모델 |
| E2 | `openai` 클라이언트에 CF-Access 헤더 주입 | 게이트웨이가 Cloudflare Access 뒤에 있음 | 없음 — 인증 헤더 |
| E3 | 임베딩 `encoding_format="float"` 고정 | Google AI Studio 라우팅이 base64 거부 | 없음 — 전송 인코딩 |
| E4 | Qdrant REST 타임아웃 5s → 60s | 기본 5초에서 대화 하나가 통째로 실패(1대화 스모크 실측) | 없음 — 전송 계층 |
| E5 | **주입 프로토콜 `single-flush`** | 원본 기본값(`buffer_size_min=1` + 1건씩 push)은 버퍼 적재가 LLM 지연에 좌우돼 **run 마다 분절기 입력이 달라진다**. 두 팔에 같은 입력을 주기 위해 대화 전체를 한 번에 flush | 🔴 **에피소드 구성에 영향 있음.** 두 팔에 동일 적용하며, 발표 기본값 재현은 별도 팔로 확인 |

## 4-1. 부칙 1 — upstream 버그 2건 수정 (2026-08-01 17:44 KST, **점수 보기 전**)

본 실행을 띄운 뒤 약 10분 시점에 발견해 **두 팔 모두 중단하고 처음부터 다시 돌렸다.**
이 시점에 산출된 정확도 수치는 **하나도 없다**(add 단계 중이었고 search 도 안 갔다) —
따라서 "결과를 보고 조건을 바꾼 것"이 아니다.

| # | 버그 | 증상 | 조치 |
|---|---|---|---|
| B1 | `PgMessageBufferStore.get_unprocessed` 의 `ORDER BY timestamp ASC` 에 **동점 처리 없음** | LoCoMo 는 한 세션의 모든 턴이 타임스탬프 하나를 공유 → 세션 안 턴이 **임의 순서**로 반환. 실측 위치열 `[1, 8, 9, 46, 48, 2, 4, 6, 7, 39, ...]` | `ORDER BY timestamp ASC, id ASC` |
| B2 | `content` JSONB 를 asyncpg 가 **문자열 그대로** 반환 (코드 주석은 반대로 적혀 있음) | 모든 메시지 본문에 바깥 따옴표 한 겹 추가. 멀티모달 리스트는 문자열이 되어 `has_images()` 상시 False | 읽을 때 `json.loads` 복원 |

**왜 고치는 게 맞나:** ① 대화 순서가 뒤섞이면 *사건 분절*이라는 과제 자체가 성립하지 않는다 —
LLM 분절기도 M5 도 똑같이 무의미해진다. ② 고치면 **원본(baseline)이 더 강해지는** 방향이다.
프로젝트 규약(`baselines-stay-as-published`)의 "명백한 upstream 버그는 고치되 공시한다"에 해당한다.
③ 두 팔에 **동일하게** 적용된다.

구현: `scripts/nemori_locomo/nemori_env.py::patch_buffer_store` (원본 파일 미수정, 메서드 치환).
검증: 같은 타임스탬프 30턴 push→get 왕복에서 순서 보존·따옴표 없음 확인.

⚠️ 이 버그 때문에 §3 사전점검(dry-run) 그룹 수(958)와 파이프라인 실측(946, 대화별로 최대 14% 차이)이
달랐다. 재실행 후 수치는 dry-run 과 일치해야 한다 — REPORT 에서 확인한다.

## 5. 지표

**주지표 = LLM judge 정확도**(발표판 headline, `generate_scores.py` 의 `llm_score`).
보조 = BLEU-1 · token F1. 카테고리별(1 multi-hop / 2 temporal / 3 open-domain / 4 single-hop) 분해.

**비용 축 동시 보고 (필수)** — 정확도만으로 판단하지 않는다:
에피소드 수 · 그룹 크기 분포 · 주입 단계 LLM 호출/토큰 · **질문당 검색 컨텍스트 토큰 수**.
프로젝트 규약대로 *정확도 vs 컨텍스트 크기* Pareto 로 읽는다.

## 6. 판정 기준 (실행 전 확정)

- **1차 판정은 LLM judge 전체 평균의 Δ(m5 − baseline)** 로 한다.
- 표본은 1,540문항 · run 1회다. **seed 반복이 없다** — 따라서 Δ 의 부호만 보고
  "우위"를 주장하지 않는다. 부트스트랩(문항 복원추출) 95% CI 를 같이 낸다.
- CI 가 0 을 포함하면 **"구별되지 않는다"** 로 적는다. 사후에 부분집합(카테고리·대화)을
  골라 유리한 셀만 인용하지 않는다.
- LoCoMo 는 이번이 **최초 개봉**이다. 여기서 본 수치로 M5 의 임계·단위·입력형식을 되돌아가
  고치면 그 시점부터 LoCoMo 는 dev 가 된다 — 그렇게 되면 REPORT 에 명시한다.

## 7. 안 하는 것

- retrieval 알고리즘 변경·튜닝 (우리 기여가 아님)
- Nemori 프롬프트·하이퍼 튜닝 (baseline 은 발표된 그대로)
- M5 재학습·앵커 교체
- 로컬 GPU 사용 (vLLM 4개가 점유 중 — CPU 로 돈다)
