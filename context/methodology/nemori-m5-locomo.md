# Nemori × M5 — 분절기 교체 (LoCoMo 다운스트림)

> 상태: 2026-08-01 최초 작성. 관련 문서 = `factored-slot.md`(M5 본체) ·
> `outputs/experiments/2026-08-01_nemori_locomo_m5/{PRE_REGISTRATION.md,REPORT.md}`

## 1. 무엇인가

외부 메모리 시스템 **Nemori**(`nemori-ai/nemori`, arXiv 2508.03341)의 에피소드 경계 결정 모듈을
우리 **M5-FactoredSlot 분절기**로 교체한 구성. 프로젝트 정체성("LLM 대화의 episodic memory")의
다운스트림 증명 자리이며, retrieval 알고리즘은 우리가 만들지 않는다는 규약을 지키기 위해
**SOTA 메모리 시스템을 통째로 빌려 쓰고 분절만 우리 것으로 바꾼다**.

## 2. 교체 지점 (딱 하나)

```
MemorySystem._process(user_id)                     # nemori/core/memory_system.py:106-113
  └─ if enable_batch_segmentation and len(messages) >= batch_threshold:
        segmenter = BatchSegmenter(orchestrator=...)   # ← 이 이름만 바꿔 끼운다
        groups = await segmenter.segment(messages)
```

| | 원본 | 우리 |
|---|---|---|
| 구현 | `nemori.llm.generators.segmenter.BatchSegmenter` | `hi_ontop.nemori_segmenter.M5BatchSegmenter` |
| 신호 | LLM(gpt-4.1-mini, temp 0.2, JSON 모드) 프롬프트 판정 | `s = Σ_k P(reset_k)` (5축 리셋확률 합) |
| 임계 | 프롬프트 지시("의심스러우면 나눠라", 에피소드당 2–15 메시지) | 적응임계 `online_nf`, `s_t ≥ mean(s_<t) + 1.0·std(s_<t)` |
| 컨텍스트 | 80메시지 청크 (청크 간 정보 단절) | 없음 — 대화 전체를 온라인으로 훑는다 |
| 출력 | 비연속 인덱스 묶음 허용, **메시지 누락 가능** | 빈틈없는 연속 분할 |
| LLM 호출 | 청크당 1회 | **0회** |
| topic 문자열 | LLM 이 생성 | `"conversation"` (nemori 자신의 분절-없음 폴백 값) |

교체 방식은 **모듈 전역 이름 치환**(`install()`)이다. `external/nemori` 파일은 수정하지 않는다 —
저장소를 다시 clone 해도 우리 코드가 그대로 붙는다.

## 3. 입력 규약 (동결)

- **단위 = 메시지(turn) 1개.** 프로젝트 기존 대화 평가 규약과 동일
  (`scripts/eacl_phase0/tolerance_bands.py::load_dialogue_val` 는 SuperDialseg turn 을
  `t["utterance"]` 원문 그대로 넣는다).
- **화자 접두사 없음.** 같은 규약. `include_role=True` 변형은 존재하지만 **미탐색**이며,
  LoCoMo 점수를 보고 고르면 그 순간 LoCoMo 가 dev 가 된다.
- **문장 단위 변형**(`unit="sentence"`)도 구현돼 있다. 문장 경계를 그 문장을 담은 메시지로
  사영한다. 기본값 아님.
- device 기본 **CPU**. 5,882 메시지 전체 분절이 60초(대화당 4~10초).

## 4. 알려진 한계

- M5 는 **영어 서사 픽션 앵커**로 학습됐다(D-M8). 대화는 도메인 밖이다.
- 서사 dev-3 에서 약 2.2배 과분절한다. 단 **LoCoMo 에서는 재현되지 않았다** —
  사전점검(`scripts/nemori_locomo/segment_dryrun.py`)에서 5,882 메시지 → 958 그룹
  (경계율 0.161, 그룹 크기 median 4 / mean 6.1), 이는 원본 프롬프트가 지시하는
  "에피소드당 2–15 메시지"와 같은 대역이다.
- topic 문자열을 안 만들므로 에피소드 생성 프롬프트의 `Boundary detection reason` 자리가
  두 팔에서 다르다. **비대칭이며 우리 쪽이 불리한 방향**이다(힌트 없음). 공시 대상.

## 5. 인프라 (이 서버)

| 구성요소 | 값 |
|---|---|
| PostgreSQL | 16.14 · `127.0.0.1:5432` · db/user `nemori` · `pg_trgm` |
| Qdrant | 1.18.3 (musl 바이너리, `external/_infra/`) · `127.0.0.1:6333/6334` |
| LLM | 게이트웨이 `api.ssunlp.co.kr/v1`(litellm) → `openrouter/openai/gpt-4.1-mini` |
| 임베딩 | `openrouter/google/gemini-embedding-001` (3072차원) |
| 파이썬 | 저장소 `.venv` 에 `asyncpg` · `qdrant-client` · `rouge-score` · `bert-score` 추가 |

환경 호환 패치는 전부 `scripts/nemori_locomo/nemori_env.py` 한 곳에 모았다
(CF-Access 헤더 · 임베딩 `encoding_format="float"` · Qdrant REST 타임아웃 5s→60s).
셋 다 **모델·랭킹·점수에 영향이 없는 전송 계층**이며 두 팔에 동일하게 적용된다.

## 5-1. 🔴 upstream 버그 2건 (수정함 · 공시 대상)

`patch_buffer_store()` 가 `PgMessageBufferStore.get_unprocessed` 를 치환한다.

| # | 버그 | 왜 치명적인가 | 수정 |
|---|---|---|---|
| B1 | `ORDER BY timestamp ASC` 에 **동점 처리 없음** | LoCoMo 는 한 세션의 모든 턴이 `session_N_date_time` **하나**를 공유 → 세션 안 턴이 임의 순서로 반환된다. 실측 위치열 `[1, 8, 9, 46, 48, 2, 4, 6, 7, 39, ...]`. 대화 순서가 깨지면 *사건 분절* 과제 자체가 성립하지 않는다 | `ORDER BY timestamp ASC, id ASC` |
| B2 | JSONB `content` 를 asyncpg 가 문자열 그대로 반환 (주석은 반대로 적혀 있음) | 모든 본문에 바깥 따옴표 한 겹 추가 · 멀티모달 리스트가 문자열이 되어 `has_images()` 상시 False | 읽을 때 `json.loads` 복원 |

**baseline 조작 아님**: 둘 다 두 팔에 동일하게 작용하며, 고치면 원본 LLM 분절기 쪽이 **더 강해지는**
방향이다. 프로젝트 규약(`baselines-stay-as-published`)의 "명백한 upstream 버그는 고치되 공시한다".
발견 시점 = 첫 본 실행 10분 시점(정확도 수치 산출 **전**) → 두 팔 모두 폐기하고 재실행.

## 6. 코드 위치

| 파일 | 역할 |
|---|---|
| `src/hi_ontop/nemori_segmenter.py` | M5 분절기 어댑터 + `install()/uninstall()` |
| `scripts/nemori_locomo/nemori_env.py` | 환경 부트스트랩·호환 패치 |
| `scripts/nemori_locomo/run_arm.py` | 한 팔 end-to-end (add→search→evals→scores) |
| `scripts/nemori_locomo/segment_dryrun.py` | LLM 없이 분절 개수만 보는 사전점검 |
| `scripts/nemori_locomo/collect_stats.py` | 비용·구조 지표 수집 |
| `scripts/nemori_locomo/make_report.py` | 정확도·비용 표 + 짝지음 부트스트랩 CI |
| `tests/test_nemori_segmenter.py` | 드롭인 계약 검증 8건 (LLM·DB 불필요) |
