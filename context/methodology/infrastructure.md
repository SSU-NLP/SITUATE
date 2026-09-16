
# 공용 Infrastructure (cross-cutting design)

버전 간에 공유되는 *non-version-specific* 인프라/구현 설계. 사소한 cache 정책, locking, encoding 단위 같은 항목도 여기 누적.

각 항목은 다음 형식:
- 무엇 (정의)
- 어디 (코드 위치)
- 왜 (도입 동기)
- 행동 영향 (어느 method 가 받는가)
- 알려진 한계 / 변형 후보

---

## 1. `EncoderCache` — segmentation 임베딩 캐시

- **무엇**: `(sample_id, kind)` 키로 conversation 별 corpus 임베딩을 캐싱.
- **왜**:
  - 같은 대화의 turn 임베딩을 여러 번 인코딩하지 않도록. 캐시 없으면 재사용 시마다 전체 turn 을 재인코딩.
  - encoder 는 thread lock 으로 직렬화돼 있어 (`QueryEncoder.encode` 의 global Lock), worker 수를 늘려도 인코딩 비용은 그대로.
- **행동 영향**:
  - 의미 자체는 안 바꿈 — 동일한 임베딩을 한 번만 계산.
- **Locking 패턴**: per-key build lock 을 outer `_dict_lock` 안에 둠. concurrent worker 가 같은 sample 에 대해 중복 인코딩 안 하도록.
- **알려진 한계 / 변형 후보**:
  - 현재 sample 단위로 인메모리. conversation 간 공유 캐시 없음. 다른 sample 이 같은 텍스트 chunk 를 갖고 있어도 재인코딩.

---

## 3. Encoder lock (단일 thread 직렬화)

- **무엇**: `QueryEncoder.encode` 가 global `threading.Lock` 으로 인코딩 호출을 직렬화.
- **어디**: `src/hi_ontop/embedding.py` (간접 — `EncoderCache` docstring 에 기록).
- **왜**: 임베딩 모델 자체가 multi-thread 안전하지 않음 (HuggingFace tokenizer + torch model). 한 thread 만 동시에 forward.
- **행동 영향**:
  - `--workers N` 늘려도 *인코딩 단계* 는 직렬. LLM 호출만 병렬.
  - Hi-OnTop 의 segmentation 단계는 conversation 단위 직렬 → encoder 가 직렬이라도 큰 문제는 안 됨.
- **알려진 한계**:
  - 한 sample 의 인코딩 시간이 wall-clock 직선적으로 들어감.
  - GPU encoding 으로 가도 lock 은 그대로 (일관성 보장).

---

## 4. LLM `--no-thinking` (reasoning bypass)

- **무엇**: OpenAI-compatible `extra_body` 로 두 가지 키를 동시에 보냄:
  ```
  extra_body = {
      "chat_template_kwargs": {"enable_thinking": False},  # vLLM
      "reasoning": {"enabled": False},                     # Crts / OpenRouter
  }
  ```
- **어디**: LLM-baseline segmenter 의 chat 호출에서 같은 `extra_body` dict 를 LLM kwargs 에 주입.
- **왜**:
  - qwen / DeepSeek / Crts proxied 모델들은 reasoning mode 가 default ON 이면 답을 `message.reasoning` 또는 `<think>...</think>` 로 보냄.
  - LLM 어댑터는 `message.content` 만 추출 → reasoning 모델이면 `content=None` 이 와서 모든 응답이 빈 문자열.
  - 2026-05-07 사고: `--no-thinking` 빠진 채 LLM-baseline 비교를 돌려서 모든 run 의 `error_rate=1.00`. 결과 무의미.
- **행동 영향**:
  - 모든 LLM-baseline segmenter 의 LLM 호출.
  - hyperparameter 가 아니라 *환경 호환성 플래그*.
- **권장**: 외부 OpenAI-compatible endpoint (Crts / OpenRouter) 사용 시 *항상 켤 것*. 새 LLM-baseline script 작성 시 누락하지 않도록 기본 포함.
- **예외 — S5 추출기(`llm_s5_states.py`)는 default reasoning ON**: frozen `p0s5-eism-v2` 는 deepseek-v4-pro 를
  reasoning 켠 채 동결했다(그래서 temp0에서도 anchor 비결정 ~14%; D-B11 REPORT §4.1). 2026-07-23 추가된
  **`--no-reasoning` 플래그**(`extra_body={"reasoning":{"enabled":False}}` 를 stage1·stage2 create 에 주입)는
  "reasoning 끄면 결정성 잡히나" 실험 전용이며 **frozen config 아님**(default 미적용 = 기존 동작 불변).
  라벨러 교체=측정도구 변경이라 켜서 쓰면 Gate C 재검증 대상(EACL/s5-eism-redesign-guide.md §6.2).
  참고(2026-07-23 `/v1/models`): api.ssunlp 목록에 `deepseek-v4-pro` **미표시**(호출은 될 수 있음, 런타임 확인).
  non-reasoning fallback 후보 = `deepseek-v3.2` · `deepseek-chat-v3.1`(DeepSeek V3 계열, reasoning 없음).

---

## 4b. deepseek-v4-pro 서빙 특성 — 레이턴시·504·병렬·no-reasoning 비용 (2026-07-23 실측)

`api.ssunlp`(openrouter proxy)의 `openrouter/deepseek/deepseek-v4-pro` 대상 통제 프로브. **S5 추출·대량 LLM
라벨링 병렬화 튜닝의 근거**. (원자료: 이 대화 프로브; RESULT `outputs/runs/2026-07-23_noreason/`.)

- **긴 생성의 정체 = 인라인 reasoning(콜당 ~3500tok 고정), JSON 아님.** stage2 protagonist ×20 실측 out
  **4339tok/67s(reasoning ON)**. batch 5/10/20 모두 out ~3500~4300tok로 **거의 고정**(사고세금이 콜당 고정)
  → **batch 줄여도 출력·시간 안 줆 = 쪼개기 손해.** stage1(monkey 최장) out 4495tok/64s.
- **`--no-reasoning` = 출력 −80%, 시간 4.5배↓ (877tok/14.8s, 20/20 정상).** 이 배포판은 분리형 reasoning
  채널을 노출 안 함(`reasoning_content` 항상 None)이나, 무거운 구조화 라벨링에선 사고가 completion 토큰으로
  잡혀 폭증 → 끄면 통째 제거. **긴생성 낮추는 압도적 1순위 레버.** (단문 프롬프트는 사고 미유발이라 −16%뿐 =
  프롬프트 난이도 의존.) ⚠️ **라벨러 교체 = 측정도구 변경** → frozen 추출 금지, 새 버전은 Gate C 재검증 필요.
- **504 = origin gateway timeout(≠ client timeout).** 조건 = `큐대기 + 생성시간 > gateway 한도`. 드라이버:
  (1) 긴 생성(위 stage1·stage2 55~67s), (2) **공유 워커풀 큐 깊이**(랩 전체 부하), (3) 시간대. idle 시간대
  (KST 00:50) 소출력 conc8도 http504 0건 → 재현은 부하 의존. 추출기의 6회 재시도+20~120s 백오프 = transient
  504 드레인 대기 장치. client timeout(httpx 60~180s)과 층이 다름.
- **병렬 스윗스팟 = 4.** 처리량(고정 ~150tok 출력): conc1 37tok/s → conc4 **145tok/s(4.3x)** → conc6 141(평평)
  → conc8 90tok/s(wall 5→16s, **역효과·큐잉**). ⇒ CLAUDE.md `xargs -P4` 실측 검증. 긴출력이면 천장 더 낮음.
- **cold-start ~20s.** idle 후 첫 콜이 warm(1.8s) 대비 10배+. 병렬 워커 워밍업하면 초반 지연 감소.
- **입력 크기는 거의 무관**(~4k토큰 입력=2.6s). run-wall(nr1 23s vs nr2 253s, 동일 조건)은 백오프·부하에 지배
  → **레이턴시 측정에 run-wall 쓰지 말 것**(순차 통제 프로브로).

- **행동 영향**: (a) 대량 추출 병렬은 `-P4` 고정, batch는 크게 유지(작게 X). (b) 새 라벨러 버전에서 속도/504가
  문제면 **no-reasoning 채택 검토**(Gate C 재검증 조건). (c) 레이턴시 벤치는 순차·warmup 후.
- **한계**: 단일 엔드포인트·시간대 1회 측정. 시간대 효과·절대 처리량은 부하 의존이라 종단 데이터 필요. 방향(레버
  우선순위·병렬 천장·504 메커니즘)은 견고.

---

## 4c. **동결: S5 base labeler = Qwen3.6-35B-A3B (no-thinking, 자체호스팅 vLLM)** (2026-07-24 결정·실측)

프록시(deepseek) 504·부하 의존성 회피 위해 S5 추출기 base labeler 를 **자체호스팅 오픈모델로 동결**. 사용자
지정(2026-07-24). 대량 확장(1만편급)·재현성의 기준 라벨러.

- **모델/양자화**: `Qwen/Qwen3.6-35B-A3B`(총 35B / 활성 3B MoE, "30B급"이라 통칭하나 정식 35B-A3B), **fp8**.
  1×A100 80GB. served-name `qwen36-35b`.
- **기동(재현)**: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.6-35B-A3B
  --served-model-name qwen36-35b --quantization fp8 --max-model-len 16384 --gpu-memory-utilization 0.85
  --port 8001 --no-enable-log-requests`. vLLM 0.25.1(`/root/vllm_pilot/venv`), HF 캐시 공유(CLAUDE.md).
- **no-thinking 지정 = 요청 단위** `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`
  (`llm_s5_states.py --no-reasoning --engine vllm`). 이 모델 chat_template 은 `<think>` 태그 사용
  (enable_thinking=False → 빈 `<think></think>` 프리필). **frozen = no-thinking.**
- **thinking 을 쓰려면** 서버에 `--reasoning-parser qwen3` 추가 필수. **없으면 guided `json_schema` 가 첫 토큰부터
  JSON 강제 → 추론 억제**(23tok, reasoning 없음). 파서 있으면 추론 후 guided JSON(콜당 ~1000~1500tok). (CONFIG_NAME=`qwen3`.)

**동시성/KV 벽 (배치20 프롬프트, 실측 2026-07-24, KV풀 1,297,785 tok):**
- **스케줄러 상한 `--max-num-seqs` 기본 256** = running 하드캡(초과분 waiting 큐잉). 모드 무관 공통.
- **no-thinking KV 벽 ≈ 320 동시** (max_num_seqs 풀었을 때 running~320서 KV=100%+preemption 시작). 기본 256 하에선
  KV 겨우 ~80%(여유) — 스케줄러 256 이 먼저 묶음. **시퀀스 ~2.5k tok(입력 지배).**
- **thinking KV 벽 ≈ 256 동시** (256서 KV=100%+preemption). 추론이 시퀀스 ~1.25배 길게 만들어 KV 더 먹음.
- 부팅 로그 "Maximum concurrency 79.21x"는 **최대길이 16384 시퀀스 가정치**일 뿐, 실제 짧은 시퀀스라 훨씬 위.

**thinking 품질 판정 (100문장 human-verify, 격리조건):** 추론 순효과 **+4/100**(off격리86→think90), 근데 **축별
불균등**(intent·causality +3 / space −2·protagonist −1, n=20/축이라 ±2=노이즈). **비용 토큰 ~40배·콜당 ~46초.**
⇒ **thinking 미채택.** 가성비 없음(+4 위해 40배). (별건: 배치20 vs 1문장격리 = human 일치 +5 관측됐으나 run
nondeterminism(~25~30% flip)과 교락 → 통제 테스트 전엔 "배치가 해롭다" 채택 금지. [[no-postresult-criterion-change]].)

> 🔴 **정정 (2026-07-25):** 아래 §4b 의 "single-stream 완전결정" 및 "동시성을 낮추면 결정성 회복"은
> **반증됨.** 동시성=1·완전 동일 요청 순차 8회에서 고유 출력 4종(항목 불일치 최대 16/20), 동시성
> 1~64 곡선에서 라벨 일치 0.854/0.906/0.844/0.812/0.823/0.833/0.781 로 **절벽도 단조성도 없음.**
> ⇒ 동시성 하향은 재현성을 사지 못하고 처리량만 잃는다(1 = 0.29 콜/s vs 64 = 4.14 콜/s). 대량 추출
> **동시성 64 권장**(200은 포화·지연 2배). 완화책은 인프라가 아니라 **k=3 다중샘플**뿐.
> 원인 미확정(fp8 / prefix-caching / MoE 라우팅). 상세 = `context/06-decision-log.md` 2026-07-25.

- **처리량 실측 (2026-07-25, 배치20 stage-2, fp8, max-num-seqs 512):** 동시성 1 = 0.29 콜/s ·
  16 = 1.27 · 64 = **4.14** · 200 = 6.01(지연 중앙값 32s = 큐잉 포화). 문장 처리량 @64 ≈ 83/s.

- **행동 영향**: (a) 대량 추출은 **no-thinking + `--max-num-seqs 300`(운영점 ~280, 선점 여유)**. 요청 ~2초/turnover
  빨라 처리량 높음. (b) 더 필요하면 **GPU 추가(KV 확장)만** 유효(같은 GPU 물리상한 ~320). (c) thinking 은 소규모
  진단용만. (d) ⚠️ 라벨러/모드 교체 = 측정도구 변경 → frozen 추출 혼용 금지, 새 버전 Gate C 재검증(§4b 규칙 계승).
- **한계**: 단일 GPU·1회 램프. tok/s 절대 처리량은 미정밀(램프 창<요청지연). 동시성 벽·모드 비교 방향은 견고.

- **⚠️ Stage-1 비결정성 (2026-07-25 실측, 미해결·수용)**: story_lists(cast/locations/goals) 추출이 **temperature=0
  greedy인데도 run-to-run 크게 변동**. obs_74140 stage1 5회: locations 17~44 (±30~50%), 고유 목록 5/5. 짧은
  프롬프트는 완전 결정적(6/6)이나 **긴 prefill(255문장)+MoE(A3B) greedy의 커널 reduction-order 비결정성**이
  긴 JSON 생성의 수많은 near-tie를 뒤집어 증폭. **시도한 fix 전부 실패**: 동시성=1(무효), `--no-enable-chunked-prefill
  --no-enable-prefix-caching`(부분, 3/5 동일), `--enforce-eager`(무효). 진짜 fix=batch-invariant 커널은 vLLM 0.25.1에
  flag로 없음. **행동 영향**: closed-choice 목록(space location·intent goal)이 run마다 달라 Stage-2 라벨·재현성에
  영향. frozen stage0 run·human_sample 은 **단일 draw**임을 명시하고 사용(수용, decision-log 2026-07-25). 별개로
  **causality 67%·protagonist 66% reset 발현율=포화**(경계 변별력 낮음) — space 14%·intent 10%·time 1.4%가 희소·변별적.
- **🔧 운영 노트 (서버 재기동)**: vLLM 재시작 시 flashinfer 가 sampling 커널(top_k_mask_logits)을 **JIT 컴파일** →
  `ninja` 빌드도구 필요. venv 바이너리 `/root/vllm_pilot/venv/bin/ninja`(pkg 1.13.0)를 **PATH 에 넣어 기동**
  (`env PATH=/root/vllm_pilot/venv/bin:$PATH …`). 없으면 `FileNotFoundError: 'ninja'`로 engine init 실패. nvcc=CUDA 13.

---

## 5. Stage-0 코퍼스 전처리 — boilerplate strip(F) / 구분선 강제분할(G) (2026-07-25)

- **무엇**: ①B 대표 코퍼스 master 의 서사 외 잡물 제거·장면경계 복원. 연산(40) — **DROP_HEAD**(문서 머리
  front-matter 통삭제: PG 크레딧·저자 약력·풍자 title-page·섹션번호), **STRIP_PREFIX**(제목·바이라인(저자명)·
  판본이 서사 첫 문장에 융합된 접두 제거), **DROP**(중간/뒤 순수 boilerplate: 크레딧·`Transcriber's Note`·
  저작권 갱신 안내·서점 광고), **STRIP_SUFFIX**(`[Transcriber ...]` 접미), **SPLIT**(장면 구분선 `* * * * *`
  분할·글리프 제거 → 분할점 transition = **gold scene boundary=1**; 구분선 = 저자 지정 장면경계). 12문서 고정이라
  머리 front-matter 는 **문서별 명시 스펙(`HEAD_SPEC`)** — regex 추측보다 감사 가능. tally: DROP_HEAD 12 ·
  STRIP_PREFIX 7 · DROP 9 · STRIP_SUFFIX 2 · SPLIT 10.
- **어디**: `scripts/eacl_phase0/stage0_clean.py`. 입력 `outputs/eacl_phase0/repcorpus_master.jsonl`(원본 보존),
  산출 `repcorpus_master_stage0.jsonl`. 감사 = `outputs/experiments/2026-07-22_gate1b_repcorpus/STAGE0_AUDIT.txt`
  (전량 before/after). **동결 `split_sentences`(`build_narrative_pack`) 불변** — 문장화 *후처리*로만 적용해
  frozen splitter 재현성 유지.
- **왜**: `fetch_repcorpus.py`의 `strip_pg`는 PG 헤더/푸터만 제거 → 본문 내부 광고·구분선·크레딧이 추출 입력에
  잔존. human-verify 오류의 일부가 이 잡물 유래(광고문에 protagonist 억지 배정, 구분선이 두 장면을 한 transition
  에 융합)로 확정됨. 사전등록 시 예정한 Stage-0 prep 의 적용(결과 후 기준변경 아님, `[[no-postresult-criterion-change]]`).
- **행동 영향**: (a) 정제본은 문장/transition 인덱스가 원본과 다름(2233→2222 tr). ⚠️ 인덱스 shift 로 **기존 라벨러
  출력·human_sample 과 정렬 안 됨** → 정제본 측정에는 동결 라벨러 **재추출**(설정 동결·입력만 교체 = 동결 위반 아님,
  `repcorpus_s5_q36_35b_v2_stage0`) + human_sample 재생성 필요. (b) **바이라인 저자명이 Stage-1 cast/protagonist
  추출을 오염**시킬 수 있어(저자명이 등장인물化) 머리 front-matter 제거는 샘플 밖 라벨 품질에도 유효. (c) gold
  boundary=1 은 **평가용**만(10개, 구분선 유래); method 에 경계 개수 미제공(deployable 규약).
- **⚠️ 초판 미스코프 정정(2026-07-25)**: 초판(23연산)은 문서 머리 제목/바이라인/저자약력/풍자부제를 통째로 놓쳤다
  (블런트 ALLCAPS 회피를 과하게 적용 → Twain title-page "THEN CLAWED BACK…"을 정상서사로 오판, 옛 verify #60 오염
  재현). junk 분리집계 중 발견 → 전 12문서 머리 실사 후 완전판(40연산) 재작성. 교훈: front-matter 는 축 라벨(특히
  Stage-1 cast)을 오염시키므로 ALLCAPS 오탐 회피보다 **문서별 narrative-start 명시**가 안전.
- **한계**: `HEAD_SPEC`·STRIP 접두는 문서별 앵커 기반(이 12문서 고정 코퍼스 전용) — 코퍼스 재획득 시 재검증 필요.
  DROP 서점광고는 ALLCAPS 상품군 시그니처 = obs_37511 전용. 규칙별 오탐은 감사본(`STAGE0_AUDIT.txt`)에서 전수
  확인(잔여 divider/boilerplate = 0, 12문서 narrative-start 육안 확인). 라벨 품질 개선폭은 재추출 후 junk-유래 vs
  실제 라벨오류 분리집계로 측정.

---

## 8. 산출물 디렉토리 구조

top-level 구조:

```
outputs/                  # 모든 active/historical generated
├── experiments/             # segmentation 러너 산출 (sweep / ablation / comparison). self-contained.
│   └── <name>/<run_label>/{run.log, exit_code.txt, results/...}
├── runs/                    # 단독 ad-hoc 실행 데이터 (날짜별 누적)
│   └── _misc/               # smoke / scratch
├── reports/                 # 독립 분석 MD (committed)
└── design/                  # 설계 문서 (committed)

archive/                  # 의도적으로 폐기한 것만
└── README.md                # "왜 버렸는지"
```

**원칙**:
- 새 experiment = 분절 러너 → `outputs/experiments/<name>/` (self-contained)
- standalone debug = `outputs/runs/`
- 시간 흐른 experiment 데이터 = `outputs/runs/<date>/` 에 누적 (정리 X — 자료)
- `archive/` = *의도적 폐기* 만. "오래됐다" 가 아니라 "더 이상 쓰지 않는다고 결정함" 의 표시.

git 정책: `outputs/experiments/<name>/REPORT.md`, `outputs/reports/*.md`, `outputs/design/*` = committed. `outputs/runs/`, sweep 안의 jsonl 데이터 = gitignored.

---

## 9. SEM2 cold-start gating / dynamics / σ² / fresh-baseline 계열 (v3.3.5~8, 2026-05-17)

- **무엇**: idx374 segmentation 진단에서 파생된 4개 cross-cutting 메커니즘.
  - `f_is_trained` gating (v3.3.5): untrained topic(transition_count<min_transitions_for_pe) = fresh slot 과 동일 L0 → likelihood 동률, prior(λ) 결정. SEM2 복원.
  - persistence+replay dynamics (v3.3.6, `TopicV336`): untrained=직전임베딩(identity), per-topic 독립 EventRNN, topic history 전체 replay(n_epochs), `rnn_ready` ≠ `f_is_trained`.
  - map_variance σ² (v3.3.7): `σ²=(ν₀var₀+n·v)/(ν₀+n+2)`, n≥2 즉시. `pe_var_min_samples` gate 폐기.
  - fresh-baseline `pe_prior` (v3.3.8): L0 가 cos_threshold 아닌 pe_prior(chance PE)에서 도출. non-prev topic = f0-likelihood(SEM2 `k0≠k_prev`).
- **어디**: `src/hi_ontop/sem_core_v33{5,6,7,8}.py`, `src/hi_ontop/topic_v336.py`, `version` dispatch + HP(`min_transitions_for_pe`,`rnn_n_epochs`,`rnn_ready_min_transitions`,`rnn_max_history`,`seed`,`pe_var_df0`,`pe_var_window`,`pe_prior`), `tests/test_sem_core_v33{5,6,8}.py`.
- **왜**: methodology v3.3.5~8.md + decision-log 2026-05-17. 핵심 = v3.3.4 의 young-topic centroid 처벌(chicken-and-egg) → SEM2 충실 복원 연쇄.
- **행동 영향**: v3.3.5~8 전부. v3.3.8 default `pe_prior=1.0`(원칙값)은 idx374 mega-collapse — **작동값은 벤치마크 calibration 대상, N=1 production default 금지**.
- **한계**: v3.3.7 은 #14|15 경험적 반증(보존). v3.3.8 fresh-baseline 은 embedding-공간 의존 HP(SEM2 단일상수 불가). non-prev f0 는 generic-opener 취약.

## 10. 재현성 — segmenter seed (필수, 2026-05-17)

- **무엇**: EventRNN/per-topic 모델 random init 이 v3.3.4/5 까지 unseeded → 동일 입력 다른 분절(앞선 v3.3.4 REPORT 수치 불일치의 정체). v3.3.6+ 는 `seed`(per-topic `manual_seed(seed·100003+topic_id)`, RNG snapshot/restore) 로 결정적.
- **어디**: `TopicV336.__init__`, `HiOnTopSegmenterV33{6,7,8}(seed=...)`, `seed` param(default 0).
- **왜**: CLAUDE.md "모든 randomness seed 고정". 논문 재현성 필수.
- **행동 영향**: v3.3.6~8. v3.3.4/5 는 unseeded(결과 해석 시 명시). 실험 시 seed 보고 의무.
- **한계**: v3.3.4/5 소급 적용 안 함(별 버전).

## 15. Online graph baseline — `GraphSegWindowD` (2026-05-21)

- **무엇**: `src/hi_ontop/baselines/graphseg_window.py` 의 `GraphSegWindowD`
  class. GraphSeg (Glavaš et al. 2016) 의 sentence similarity (IC × GloVe +
  Hungarian) + Bron-Kerbosch maximal clique + sequential merge 3-phase 를
  *window=d 안에서만* 적용. `push() → list[int]` (lag-emission, 1-based),
  `flush()`, `state()`.
- **어디**: `src/hi_ontop/baselines/`, runner `methods/graphseg/online_window.py`
  (3-dataset, Def-DTS 번들, segeval 직접). 의존 = numpy/scipy/networkx +
  NLTK brown (IC table) + GloVe 6B.300d (`benchmarks/glove/`, gitignored,
  외부 download).
- **왜**: Hi-OnTop 의 online 비교표에 *graph-based unsupervised* 카테고리 보강.
  codex 2026-05-21 검증: 전역 graph 본질이 깨졌으므로 **AUXILIARY only**.
- **행동 영향**: 강한 `GraphSeg-online` 명명 금지 (paper 본문 = `GraphSeg-
  inspired bounded-window`, short `GraphSeg-window-d` 는 file/CLI 만). 원본
  GraphSeg paper 결과와 같은 표 등재 금지. TextTiling-streaming / GreedySeg-
  online-delay2 와 같은 latency 표에 *직접 비교 금지* — encoder/연산 카테고리
  다름. 결정성: numpy/scipy/networkx 모두 결정적 → seed 무관.
- **한계**: word frequency 원본 = Wikipedia, 본 구현은 NLTK brown corpus
  (실용 선택). Bron-Kerbosch worst-case 지수 — window_d=10 default, edge
  density 높으면 d=5 권장. boundary lag-emission 의 비가역성 = window 평가
  결과가 후속 window 에서 바뀌어도 *최초 발견 boundary 유지*.

## 14. Online BERT baseline — `GreedySegOnlineDelay2` (2026-05-21)

- **무엇**: `src/hi_ontop/baselines/greedyseg_delay2.py` 의 `GreedySegOnlineDelay2`
  class. SuperDialseg GreedySegmenter (Jiang et al. 2023) 의 BERT cosine score
  공식·HP·argmin greedy 선택을 *그대로 보존* 한 채 streaming 입력 + boundary
  emit lag (right context window_size=2). `push() → list[int]` (1-based,
  lag-emission), `flush() → list[int]`, `state() → dict`.
- **어디**: `src/hi_ontop/baselines/`, runner `methods/greedyseg/online_delay2.py`
  (3-dataset, Def-DTS 번들, segeval 직접). device helper `src/hi_ontop/baselines/
  _device.py` (resolve_device, enable_mps_fallback) + shared utils
  `src/hi_ontop/baselines/_seg_utils.py`.
- **왜**: Hi-OnTop 의 online 비교표에 *zero-shot BERT cosine* 카테고리 보강.
  codex 2026-05-20/21 검증 = "score 공식·greedy 선택 보존 → 강한 명명 OK,
  5행 핵심표 가능 (본 plan 유일)".
- **device-agnostic**: cuda/mps/cpu 한 코드. `--device {auto,cuda,mps,cpu}`,
  default `auto` (cuda → mps → cpu). MPS 사용 시 `PYTORCH_ENABLE_MPS_FALLBACK=1`
  반드시 transformers import 이전 설정 (runner 최상단). model.to(device) +
  tokenizer output device 이동 + model.eval() + torch.inference_mode().
- **행동 영향**: 본 baseline 은 TextTiling-streaming (encoder-free, ~0.01ms)
  과 같은 latency 표에 *직접 비교 금지* — encoder cost 차원이 다름. 별도 § 또는
  열 분리. 결정성 CPU > CUDA > MPS — 논문 reproducibility 시 동일 device
  반복 측정.
- **한계**: 원본 BERT pooling 방식 (CLS vs mean) 확인 못함 (superdialseg
  install 없음), 본 구현 default = CLS. *strict prefix-causal 아님* (delay=2).
  segment finalize 의 실제 lag = max_seg_round + window_size 발화까지 가능.

## 13. Streaming baseline segmenter — `StreamingTextTiling` (2026-05-20)

- **무엇**: `src/hi_ontop/baselines/texttiling_streaming.py` 의 `StreamingTextTiling`
  class. `push(utterance) → list[int]` (이번 호출에서 새로 확정된 boundary
  utterance index 1-based), `flush() → list[int]` (대화 종료시 잔여 gap 처리).
  block-cosine + Welford running mean/std threshold + one-sided causal depth +
  min_gap suppression. per-turn 실질 O(w).
- **어디**: `src/hi_ontop/baselines/`, 진입점 runner `methods/texttiling/online_streaming.py`
  (Def-DTS 번들 jsonl 3 dataset 로드 — tiage/dialseg711/superseg, segeval 직접 호출).
- **왜**: 기존 `run_texttiling_prefix.py` 는 매턴 nltk fresh 호출 = O(t)/turn
  → online baseline 의 "latency 비교값" 정책 위배. 진짜 streaming 버전을 별도
  method 명 (`TextTiling-online-streaming`) 으로 신설 (codex:rescue 2026-05-20,
  decision-log 참조).
- **행동 영향**: TextTiling 비교 시 *3종 구분 강제* — offline (NLTK 전체), prefix-
  recompute (NLTK 매턴 fresh), streaming (자체 구현). 같은 표에 섞지 말 것.
- **한계**: 원본 NLTK TextTiling 점수 재현 안 함 (causal running threshold ≠
  offline global threshold). Pk/WD/F1 = INDICATIVE; latency 가 비교값. class
  default w=10/k=6 (NLTK 호환), runner default w=5/k=3 (tiage 짧은 대화 대응).

---

## online reset 벽 = 인코더-독립 + 판별자 정보-한계 (2026-06-20)

- **무엇**: AMI online 화제분절의 oracle→deploy 격차(±2F1 0.71~0.77 vs 0.14)가 **표현(embedding) 한계가 아니라
  online reset 결정 한계**임을 확정. spike 시점 onset/outlier 판별자 AUC ≈ 0.49(랜덤), 0-lag·1-lag 모두.
- **어디**: 측정 `scripts/probe_encoder_wall.py <emb_subdir>`, 임베딩 생성 `scripts/gen_ami_emb_encoder.py`(로컬 ST)·
  `scripts/gen_ami_emb_api.py`(Crts API). 캐시 `outputs/runs/_misc/{ami_emb(minilm-int8),ami_emb_mpnet(all-mpnet-base-v2),
  ami_emb_te3large(text-embedding-3-large)}`. 채점 `src/hi_ontop/ami_scoring.py`.
- **왜**: 정보-한계 결론([[HANDOFF_01]] §2.6)이 MiniLM-int8 하나로만 측정됐어 인코더-의존 가능성 미확인 → 3체급 비교로 falsify.
- **행동 영향**: (1) **모든 method**(de-neut/V_rel/δ_eff deploy) — 더 강한 인코더로 deploy 천장 못 올림(oracle만 ↑).
  인코더 교체로 reset 벽 해결 시도 금지. (2) **판별자 AUC 측정 시 per-row 단위정규화 필수** — 2D 배열에
  `np.linalg.norm(e)`(Frobenius, 미팅별 스칼라) 쓰면 cross-meeting pooling 누수로 AUC 부풀려짐(옛 `ami_reset_transaction.load_ami::_nrm`
  버그 = "1-lag AUC 0.665" 아티팩트, 실제 ≈0.52). per-meeting-mean AUC 또는 per-row 정규화로 측정.
- **한계**: clean(gold-reset) oracle 0.69~0.77 은 유효(미팅-내). 다음 레버 = spike-gated LLM judge + label/state commit 분리
  (codex 2026-06-20, [[HANDOFF_01]] §"축2 재공략"). lag/학습 판별자는 0-lag·no-train 제약상 별도 relaxation 트랙.

---

## 작성 규칙

새 인프라/cross-cutting 설계 항목 추가 시:
1. 위 형식 (무엇/어디/왜/행동영향/한계) 으로 한 섹션 추가.
2. 어느 버전이 영향받는지 명시.
3. 실험에서 이 인프라가 *결과에 영향을 줬으면* 반드시 기록 (예: `--no-thinking` 누락으로 sweep 망친 사례).

---

## 16. LLM attention mass 추출 — pre-hook + 블록 causal softmax (2026-07-31)

- **무엇**: 문서 전체에 대해 토큰별 "받은 attention 질량"
  `a_t = Σ_{layer,head,query} p[q,t]` 를, `[L,H,T,T]` attention 행렬을 **만들지 않고** 얻는다.
  attention 모듈에 `register_forward_pre_hook(with_kwargs=True)` 를 걸어 모듈 입력
  (post-`input_layernorm` hidden)과 `position_embeddings=(cos,sin)` 을 캡처한 뒤, forward 종료 후
  모듈 자신의 `q_proj`/`k_proj`/`apply_rotary_pos_emb`/`repeat_kv` 로 post-RoPE Q,K 를 재구성한다.
- **어디**: `scripts/segment_attn_entity.py`
  — `attention_mass()` (드라이버), `_post_rope_qk()` (재구성), `_mass_rowsent()` / `_mass_full()`
  (두 정규화 경로). 검증 = `tests/test_attn_mass.py`.
  미러링 대상 = `transformers/models/qwen2/modeling_qwen2.py:150-206` (`Qwen2Attention.forward`).
- **왜**:
  - `output_attentions=True` 는 **쓸 수 없다**. T=6,637 에서 한 레이어 `[1,28,T,T]` fp16 = 2.47 GB,
    transformers 가 28 레이어 전부 보관 = **69 GB**. 게다가 sdpa 가 조용히 eager 로 폴백한다
    (`modeling_qwen2.py:184-189`).
  - EM-LLM 원본은 이 값을 내부에 이미 갖고 있으나(`context_manager.py:716` `exc_repr_score`)
    라이브러리 밖에서는 못 꺼낸다. `scripts/segment_manual_emllm.py:20-21` 의 "못 뽑아" 주석은
    **모델 밖 재구성이 가능하다는 점에서 정정 대상**이다.
- **★ 핵심 성질 — `rowsent` 정규화에서는 전체 causal prefix softmax 가 불필요하다.**
  타깃을 문장 i 의 토큰 집합 `core(i)` 로 제한하고 각 쿼리 행을 그 위에서 재정규화하면
  전역 분모 `Z_q` 가 **정확히 소거**된다:
  `p̃[q,t] = (e^{s_qt}/Z_q) / (Σ_{t'∈core(i)} e^{s_qt'}/Z_q) = e^{s_qt} / Σ_{t'∈core(i)} e^{s_qt'}`
  ⇒ `core(i)` 위의 restricted softmax 와 동일(인과 마스크는 그대로). 비용이 `O(Σ_i |Q(i)|·|core(i)|)`
  로 떨어져 dev-3 문서당 **7~9초**. `count`/`none` 정규화는 전역 분모가 필요해 블록 경로로 간다.
- **행동 영향**: M9-AttnEntityDrift 전용으로 도입. 다른 method 는 아직 쓰지 않는다.
  LM 은 `Qwen/Qwen2.5-7B`(base) — `segment_manual_emllm.py` 와 같아야 EM-LLM 대조가 like-for-like.
- **알려진 한계 / 변형 후보**:
  - `char_start` 는 **토큰의 첫 비공백 문자** offset 이어야 한다. Qwen BPE 토큰은 선행 공백을
    포함하므로(`" McGowan"`) 토큰 시작 offset 을 쓰면 spaCy 토큰 span 매칭이 대량 실패한다
    (실측 868/2,928/4,652 건). `assert_token_offsets()` 가 재발을 막는다.
  - 추출 자체는 건전하나 **그 위의 "상위 k = 엔티티" 해석은 기각됐다**(D-B58): 상위 토큰의
    40~50% 가 불용어·구두점이고 문장 첫 내용어에 위치 편향된다. 재사용 시 이 점을 전제할 것.
  - `scope=retrieved`(쿼리가 과거 전체 어디를 보는가) 미구현.
  - 캐시는 `[n_layer, n_head, T]` fp16 로 **전 레이어·전 헤드** 저장(문서당 1.6~7 MB) — 레이어대역·
    헤드집계 ablation 이 재추출 없이 가능하다.

---

## 17. Tier2 silver gold 생성 — B2/B3 절차 재현 설정 (2026-07-15 동결, 2026-08-01 기록)

- **무엇**: 봉인 시험지에 사람 경계가 없으므로, **B2(Michelmann)/B3(Panela) 절차로 LLM 은(silver)
  정답**을 만든다. 설정 = ① 입력을 `re.sub(r"\s+", " ")` 로 눌러 **문단 없는 한 덩어리**로 만들고
  ② **B2 zero-shot 프롬프트를 한 글자도 바꾸지 않고**("사건이 끝나고 새 사건이 시작될 때마다 새 줄을
  시작하며 이야기를 그대로 베껴 써라") ③ **20회 반복**(temp 1.0) ④ 각 run 이 삽입한 줄바꿈을
  **단어 오프셋 정렬**로 문장 전이에 사상 ⑤ **과반(⌈R/2⌉) run 이 찍은 전이만 경계**로 하는 다수결 합의.
- **어디**: `scripts/eacl_phase0/build_tier2_gold.py` — 정규화 `:143`, 프롬프트 `PROMPT_HEAD/TAIL`,
  합의 파싱 `:67-70`. 문장화는 동결 `split_sentences`(`build_pieman_pack`) 그대로.
  산출 `outputs/eacl_phase0/tier2/{sid}_gold.json`(run별 + 합의), `--apply` 시 master 의 `boundary` 채움.
  스펙 = `EACL/tier2-heldout-spec.md`, 요약 = `EACL/pipeline-overview-ko.md` §1.3.1.
- **왜**: ① **문단 제거는 재현 충실도이자 파싱 전제다.** B3 는 사람이 서식에서 경계를 읽지 못하게
  문단 없이 인쇄했는데, 우리 쪽에서는 **줄바꿈이 곧 출력 신호**라 입력에 줄바꿈이 남으면 모델이
  삽입한 경계와 원문 줄바꿈을 구별할 수 없다. ② 프롬프트 무수정 = B3 가 "B2 의 zero-shot 프롬프트를
  그대로 썼다"고 명시하므로 계보를 그대로 잇는다. ③ 20회 다수결 = B3 의 반복 설계 이식.
- **행동 영향**:
  - 🔴 **모델이 `gpt-4` → GPT-4.1 로 승계됐다.** 원 `gpt-4` 는 8k 컨텍스트라 전문 verbatim 복사가
    기계적으로 잘린다(실측: 4,623단어 문서 절단). **성능 목적의 사후 교체가 아니라 동결 모델의
    비실행성 교정**(D-M9 feasibility amendment). 인용은 *"B3 가 검증한 GPT-4 기반 **절차**를 채택,
    더 긴 자극을 무손실 처리하려 후속 GPT-4 계열 사용"* 까지만 — **"B3 가 검증한 바로 그 모델"이라
    쓰지 않는다.**
  - 🔴 **순환 방지**: 이 gold 위에서 **B2/B3 를 베이스라인으로 쓸 수 없다**(gold 생성기, D-M6-c).
    쓸 수 있는 비교군 = B1·M1·M2.
  - 교차 계보(D-M6-b): gold = GPT-4 계열(OpenAI) vs 추출기 M3 = 다른 벤더 — 같은 모델이
    gold 와 방법 양쪽에 들어가지 않게 한다.
- **한계**:
  - **B3 가 검증한 조건 밖이다.** B3 자극은 챕터당 **1,500단어 절단**인데 우리는 **절단하지 않는다**
    (최대 4,623단어, 3배 이상). B3 의 "LLM 이 인간보다 일관됨" 주장은 그 길이에서 검증된 것이라
    우리 조건으로 그대로 이월되지 않는다.
  - **경계 주체가 다르다** — B3 는 사람 20명, 우리는 GPT-4.1 20회. **회상 과제(구술)는 없다.**
  - **run 간 변이가 크다**: 고유 분절 7.9/20, 경계 수 최대 3배 변동. B3 의 일관성 주장과 긴장
    관계이며, 구현 차이인지 자극 의존인지 **미확인**. 합의 산출물 자체의 안정성도 별도 검증 대상.
