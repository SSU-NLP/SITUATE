# D-B47·D-B49 — S5 라벨러 v3.1 dev-3 비교 (후보: gpt-oss-120b · DeepSeek-V3.2-AWQ)

## 실험 setup

- **목적:** 로컬 호스팅 가능한 더 나은 S5 라벨러가 있는지. DeepSeek 계열은 671B MoE·FP8 ~670GB 라
  A100-80GB×4(320GB)에 안 올라가 탈락 → `openai/gpt-oss-120b`(117B MoE, MXFP4 65.2GB)가 후보.
- **데이터:** dev-3 = pieman / tunnel / monkey (`outputs/eacl_phase0/{doc}_michelmann_master.jsonl`).
  **held-out(Tier1 6편) 미개봉.** gold = `reset_matrix.jsonl` 의 `boundary`(Michelmann, author-filtered).
- **방법:** `llm_s5_states.py --prompt-version v3.1 --engine vllm`, temp 0, batch 20, seed 20260711,
  vLLM 0.25.1 (VESSL A100×1 job, 문서 1편 = job 1개, 5축 동시). 5축 전부
  (protagonist·space·time·causality·intentionality), 축 병렬 후 full 패스로 reset_matrix 조립.
- **⚠️ 추론 설정이 두 모델에서 대칭이 아니다.** Qwen = `--no-reasoning`
  (`chat_template_kwargs={'enable_thinking':False}`) 로 **완전 off**. gpt-oss 는 **off 스위치가 없어**
  `reasoning_effort='low'` + `--reasoning-parser openai_gptoss` 로 최저값만 줬다 — **추론이 켜져 있다.**
  ledger 근거상 이 과제에서 추론은 해롭다(v3: deepseek reason-ON 0.327 vs OFF 0.364).
  ⇒ **gpt-oss 에 불리한 교란**이며, 아래 열세를 '모델이 약함' 으로 읽으면 안 된다.
- **가중치:** `openai/gpt-oss-120b` MXFP4 65.2GB. A100(sm80)에서 `Mxfp4Backend=MARLIN` 으로
  **네이티브 적재**(bf16 디양자화 아님) — A100 1장에 올라간다. 적재 2분 35초(runai_streamer).
- **후보 2 (D-B49): `QuantTrio/DeepSeek-V3.2-AWQ` — 4-bit AWQ 361.9GB, a100x8(TP=8).**
  네이티브 FP8 원본은 689.5GB 라 640GB 에 안 들어가 **양자화본만 가능**하다.
  🔴 **원본이 아니다** — 열세가 나와도 *모델 능력* 인지 *양자화 손실* 인지 분리 안 됨.
  🟢 추론은 `chat_template_kwargs={'thinking':false}` 로 **완전 off** — Qwen 과 대칭이고
  기존 `dschat_v3`(프록시 no-think) 조건과도 일치한다. gpt-oss 의 비대칭 문제는 여기 없다.
  ⚠️ k=3 구조가 다르다: gpt-oss/Qwen 은 run 마다 새 서버, DeepSeek 은 **같은 인스턴스에서 3회**
  (362GB 를 한 번만 적재). 배치 비결정은 동일하게 잡히나 적재-시점 변동은 안 잡힌다.
- **지표:** deployable Score(t\*=3) = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD), `dts_scoring.score_dialogues`.
  gold 경계 개수를 방법에 주지 않는다(matched@gold 아님).
- **반복:** 각 모델 k=3 run(접미사 `_v31`, `_v31_r2`, `_v31_r3`). 배치 비결정이 노이즈원.

## 결과 — v3.1 (동결 프롬프트), 평균 ±SD over k run

| 모델 | k | pieman | tunnel | monkey | 평균 |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B-v3.1 (기준선) | 3 | 0.424 ±0.018 | 0.410 ±0.012 | 0.314 ±0.032 | **0.382** |
| Gemma-4-26B-A4B-v3.1 | 3 | 0.468 ±0.061 | 0.338 ±0.032 | 0.364 ±0.007 | **0.390** |
| Gemma-4-31B dense-v3.1 | 3 | 0.456 ±0.006 | 0.379 ±0.000 | 0.294 ±0.004 | **0.376** |
| gpt-oss-120b-v3.1 | 3 | 0.515 ±0.056 | 0.327 ±0.023 | 0.296 ±0.036 | **0.380** |
| MiniMax-M2.7-v3.1 (추론 OFF) | — | _pending_ | | | |
| MiniMax-M2.7-v3.1 (추론 ON) | — | _pending_ | | | |
| DeepSeek-V3.1-AWQ(4bit)-v3.1 | — | _pending_ | | | |

## 판정 — Gemma-4-26B-A4B-v3.1 (D-B51 §2 사전등록 기준)

- ΔScore(평균) = **+0.007** — 기준 **+0.03 이상** → **미달**
- 하락 없는 문서 = **2/3** (pieman, monkey) — 기준 **2편 이상** → 충족

**→ 노이즈 내 동등 또는 열위. 동결 유지, 라벨러 교체 없음.**
Qwen3.6-35B-A3B / v3.1 그대로 간다. Gemma-4-26B-A4B-v3.1 수치는 기록으로만 남는다.

> 🟢 **교란 없음** — 원본 가중치, 템플릿 무개조, `enable_thinking:False` 가 Qwen 과 동일 키. 이 판정만은 주석 없이 인용 가능하다.

## 판정 — gpt-oss-120b-v3.1 (D-B47 §2 사전등록 기준)

- ΔScore(평균) = **-0.003** — 기준 **+0.03 이상** → **미달**
- 하락 없는 문서 = **1/3** (pieman) — 기준 **2편 이상** → **미달**

**→ 노이즈 내 동등 또는 열위. 동결 유지, 라벨러 교체 없음.**
Qwen3.6-35B-A3B / v3.1 그대로 간다. gpt-oss-120b-v3.1 수치는 기록으로만 남는다.

> 교란 = **추론 off 불가**(reasoning_effort=low 가 최저). gpt-oss 에 불리 → 이 값은 하한.

> _D-B50 §3 판정 보류 — MiniMax-M2.7-v3.1 (추론 OFF) 아직 미완료._

> _D-B49 §1 판정 보류 — DeepSeek-V3.1-AWQ(4bit)-v3.1 아직 미완료._

## 해석

**평균은 동률인데 문서별로는 정반대다.** 문서별 Δ(gpt-oss − Qwen) = pieman +0.091, tunnel -0.082, monkey -0.017.
pieman 에서 gpt-oss 가 크게 앞서고 tunnel 에서 크게 뒤진다 — 평균이 상쇄해 0 이 된 것이지
'두 모델이 같은 라벨을 만든다'는 뜻이 아니다. 교차모델 표에서 이미 관측된
**문서별 지배 도메인 상이** 패턴(D-B14 era: tunnel=space·monkey=protag/intent)과 같은 종류다.
pieman 의 gpt-oss SD(±0.056)가 다른 셀의 2~3배라 이 우위 자체도 안정적이지 않다.

**Qwen v3.1 기준선(0.382)은 기존 v3 값(0.387)과 사실상 같다** — 프롬프트 v3→v3.1 전환이
dev-3 Score 를 흔들지 않았다는 뜻이고, 이번에 처음 만든 v3.1 dev-3 기준선의 sanity check 가 된다.

## 한계 / 검증 미해결

- 🔴 **추론 비대칭(위 setup 참조)이 최대 한계다.** gpt-oss 는 추론을 끄지 못한 채 겨뤘으므로
  이 결과는 **gpt-oss 의 하한**이다. '열위' 가 아니라 '이 조건에서 우위 없음' 까지만 말할 수 있다.
- dev-3 는 D-M5 로 소진된 연습장이다. 여기 수치는 **탐색적**이며 held-out 성능이 아니다.
- k=3 은 배치 비결정만 흡수한다. 문서 수 3편이라 doc-level bootstrap CI 는 불가.
- gpt-oss-120b 는 v3.1 프롬프트에 맞춰 튜닝된 적이 없다 — deepseek 기각(D-M13⑥)이
  '능력이 아니라 스펙 해석 차이'였던 것과 같은 교란이 남아 있을 수 있다.
- 🔴 **재실행 시 주의:** gpt-oss 는 `--quantization` 을 주지 않는다(체크포인트 config 가
  `gpt_oss_mxfp4` 를 선언한다). Qwen 처럼 `--quantization fp8` 을 붙이면 안 된다.

