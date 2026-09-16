# Stage-0 정제본 재추출 + 인간검증 (REPORT)

> ①B 대표 코퍼스에 Stage-0 전처리(F boilerplate/front-matter strip · G 구분선 강제분할) 적용 후,
> 동결 라벨러(Qwen3.6-35B-A3B no-thinking v2)를 정제본 위에서 **재추출**하고, 정제본 라벨의 블라인드
> 인간검증 표본을 재생성한다. 목적 = 잡물 제거 후 라벨 품질을 깨끗한 입력에서 재측정하되, 개선분을
> **"라벨러 개선"으로 오인하지 않도록 junk 유래(N) / 실제 라벨오류(M−N)를 분리**한다.

## 1. Setup

- **코퍼스**: `outputs/eacl_phase0/repcorpus_master_stage0.jsonl` (12문서, 2222 transitions).
  Stage-0 = `scripts/eacl_phase0/stage0_clean.py`, 원본 `repcorpus_master.jsonl`(2233 tr) 보존.
  40연산: DROP_HEAD 12 · STRIP_PREFIX 7 · DROP 9 · STRIP_SUFFIX 2 · SPLIT 10(→ gold scene boundary=1).
  전량 감사 = `../2026-07-22_gate1b_repcorpus/STAGE0_AUDIT.txt`. 잔여 divider/boilerplate = 0(검증).
- **라벨러(동결)**: `Qwen/Qwen3.6-35B-A3B` fp8, 로컬 vLLM(:8001), `--no-reasoning --engine vllm
  --prompt-version v2`. 설정 동결·입력만 교체(동결 위반 아님, `[[s5-labeler-frozen-qwen-nothink]]`).
  드라이버 `run_repcorpus_stage0.sh`(race-safe: protagonist 선생성 → 4축 → full pass).
  산출 `outputs/eacl_phase0/repcorpus_s5_q36_35b_v2_stage0`. 60/60 [ok], FAIL 0, reset_matrix 2222행,
  12문서×5축 item 수 = 문장 수 일치(완결성 검증 통과).
- **인간검증 표본**: `human_sample/` — `human_sample_build.py`(seed 20260723, 축당 20 × 5축 = 100문항,
  중립비율·밀도bin 층화). 블라인드 sheet(모델 라벨 없음) + answer_key 분리. **새 표본 잡물 0 확인.**
- **양식(사용자 지정 = 저번과 동일)**: 예/아니오 웹도구 `label_web.html`(`human_sample_webpage.py --mode
  yesno`) + 한국어 번역 + 맥락 gloss. 두 변형:
  - **fresh 100** (`human_sample/`, 정제본 pool 재추출):
    https://claude.ai/code/artifact/d1721de7-b0af-4a3b-a8e1-c6b612cfc31b
  - **matched 93** (`human_sample_matched/`, 사용자 지정): 저번 100문항과 **똑같은 문장·똑같은 순서**로
    재정렬하되 잡물 7(=N) 스킵 → 93. 번역은 옛것 재사용, 라벨은 stage0 기준. 빌더 `build_matched_sheet.py`.
    Artifact: https://claude.ai/code/artifact/b8228109-d647-40a5-83c4-76180352b540 **← 라벨링 대상**

  🔴 **2026-08-04 재발행.** 최초 발행분(fresh `…/6d39ce78-…` · matched `…/12595f82-…`,
  둘 다 2026-07-25 발행)은 서버에서 **삭제돼 열리지 않는다**(`artifact not found`). 원본
  `label_web.html` 은 그대로 남아 있어 같은 파일을 다시 올린 것이며 내용은 동일하다.
  URL 은 발행 즉시 여기 적는다 — 이번 유실이 기록 부재 때문이었다.

  🔴 **matched 시트의 실제 문항 수는 97 이다**(93 아님). 문장 기준으로 최초 100문항과
  일치하는 것이 93, 나머지 4 는 Stage-0 정제로 문장이 바뀐 것이다. 머리말의
  "문항 1–100" 캡션도 생성기 고정문구라 실제 수와 다르다.

  🔴 **localStorage 키가 최초 시트와 같다**(`eacl_gate1b_human_v20260723v2`). 저장은
  `row_id` 단위이고 matched 시트는 최초 시트와 **row_id 34개를 공유**하므로, 같은 브라우저에서
  최초 시트를 라벨링한 사람이 matched 를 열면 그 34문항이 **이미 채워진 채로 뜬다** =
  블라인드 재검증이 깨진다. 다른 사람에게 돌리거나, 시크릿 창에서 열 것.
- **채점 규약(동결)**: `human_sample/SCORING_RUBRIC.md` — 축별 model-wrong(✗) ≤ 20%. 결과 후 변경 금지.

## 2. junk 분리집계 (완료) — `JUNK_SEPARATION.md`

옛 human-verify(dirty 코퍼스 100문항)의 라벨 문제를 Stage-0 인벤토리와 **객관 교차대조**:

| 구분 | 수 | Stage-0 | 새 정제본 |
|---|---|---|---|
| **N 입력정제** | **7/100** | 표본 문장 자체가 잡물(구분선 2·boilerplate 5) → DROP/SPLIT | 구조적 소멸 |
| **M−N 실제 라벨오류** | **2/100** | 정상서사, Stage-0 무관 | **잔존** (space 과잉귀속) |

- **M−N=2 (진짜 오류, 둘 다 space context-leak):** `fam_london_fire#214`("…cheeks and nose were
  freezing." → `yukon_trail`), `fam_doyle_speckled#166`("…sent for medical aid from the village…" →
  `julia_bedroom`). 문장에 장소단서 없는데 전역 배경을 역귀속 = target-only 소프트 위반. **회피의 반대
  방향.** → v3 프롬프트가 고쳐야 할 대상("명시 장소 없으면 unspecified").
- 옛 RESULT 수기 5건 < 객관 7건(라벨 안 뒤집은 space 구분선·2번째 광고행 2개 추가) = 더 엄밀.

## 3. 판정 / 다음 단계

- **완료**: Stage-0 완전판 정제 · 정제본 재추출(완결) · human_sample 재생성(잡물 0) · 예/아니오 웹도구
  재현 · junk 분리집계(N=7 / M−N=2).
- **대기(사용자 액션)**: Artifact 에서 100문항 블라인드 라벨 → CSV 내보내기 →
  `human_sample_score.py --stage1 --labels <csv>` 채점 → 불일치 판정 → 본 REPORT 에 **새 정제본 verify
  결과**(축별 ✗율 · PASS/FAIL) 추가.
- **오인 방지 규약**: 새 verify ✗율은 옛 **✗(제외) = space 2/20(10%), 그 외 0%** 와만 비교(잡물 제거분과
  라벨러 변화 분리). space 과잉귀속이 새 verify 에서도 남으면 v3 과제로 확정.

## 4. 한계 / 검증 미해결

- 옛/새 표본은 동일 문장집합 아님(인덱스 shift·라벨 변화) → 문항별 old↔new 대조 불가. 분리집계는 귀속,
  새 verify 는 독립 측정.
- `HEAD_SPEC`(문서 머리 front-matter)·STRIP 접두는 이 12문서 고정 코퍼스 전용 앵커 — 코퍼스 재획득 시 재검증.
- 초판 Stage-0(23연산) 문서-머리 front-matter 미스코프 → 완전판(40) 재작성(decision-log 2026-07-25 정정).
- 새 정제본 verify 점수는 **아직 없음**(사람 라벨링 대기). 본 REPORT §3 갱신 예정.
- **⚠️ Stage-1 목록 비결정성 (미해결·수용, 2026-07-25):** story_lists 추출이 temp=0인데도 run-to-run 변동
  (obs_74140 loc 17~44, ±30~50%). 원인=긴 prefill+MoE greedy 커널 reduction-order. 표준 vLLM flag(동시성=1·
  chunked-prefill off·prefix-cache off·enforce-eager) **전부 실패**, 진짜 fix(batch-invariant 커널)는 vLLM 0.25.1에
  없음 → **수용+문서화** 결정(infrastructure §4c). frozen stage0 run·human_sample·answer_key 은 **단일 draw** 기준.
  즉 gold 라벨의 재현성이 목록 축(space/intent)에서 제한적. 상세=`infrastructure.md §4c`, decision-log 2026-07-25.
- **구분선 gold 의 성격 (진단, 전체 데이터):** 구분선=10건(0.45%). **dev3(michelmann)=0**(음성 내러티브) →
  12편+dev3 전체 구분선 = repcorpus 10개가 전부. 축 발현율 대비(causality 67%·protagonist 66%·space 14%·intent
  10%·time 1.4%) 1~2자릿수 드묾 → 단일 축 아닌 **다축 장면경계**(구분선당 평균 2.4축).
- **⚠️ "구분선 = causality 새 상태 시작"은 데이터상 기각(inceptive 0/10):** `reset_causality =
  (event_type≠직전) OR (aspect=inceptive)`. 10개 post-divider 분해 = **event_type 변화 10/10 · inceptive 0/10**.
  즉 reset_causality "10/10 켜짐"은 전부 event_type-change(baseline 65%, 포화 노이즈)에서 왔고 **진짜 새-인과상태
  시작(inceptive)은 0/10**(baseline 7.7%보다도 낮음; aspect=completive 6·ongoing 4, causal_marker 전부 absent).
  ⇒ 구분선을 causality-inceptive 신호로 보면 **안 됨**. (앞선 "10/10 발현" 서술은 reset flag 미분해 오독 → 정정.)
