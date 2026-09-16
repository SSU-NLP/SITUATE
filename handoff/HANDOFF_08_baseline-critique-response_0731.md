# HANDOFF 08 — 외부 비판 3건 대응 (베이스라인 공정성 · LLM 직접 질의 · 인간 기준선)

작성 2026-07-31 KST. ledger 항목 = **D-B57** (`EACL/phase0-decisions-and-runs.md` 최하단).

## 문제

외부 비판 3건.
1. *"M3-S5(Qwen 35B) 대 B1(GPT-2)은 이론 대결이 아니라 모델 세대 대결"*
2. *"Michelmann(LLM 직접 질의)에 답이 없으면 5축 분해의 필요성이 논증 안 됨"*
3. *"밀도만 맞추면 무작위도 경쟁력 있는 점수 → DTS Score 자체가 위협"*

**1·2는 실험이 이미 있었고 요약 문서에만 없었다.** 3은 신규 대응 필요.

---

## ✅ 한 것

### A. 개별 인간 주석자 기준선 (신규)
- 사전등록 `outputs/experiments/2026-07-31_human_annotator_baseline/PREREG.md`
- REPORT 동 디렉터리 · 데이터 `human_baseline.json` · 러너 `scripts/eacl_phase0/human_annotator_baseline.py`
- 원자료 `data/eacl_phase0/{pieman,tunnel}_michelmann/sourcedata/alldata*.mat`
- **결과**: pieman 0.4994±0.105(205명, BOR **1.02**) · tunnel 0.5422±0.132(10명, BOR 1.39) · **monkey 원자료 없음(영구 결측)**
- **핵심**: 인간은 BOR≈1.0 ⇒ 비판 3 기각. 우리 과분절(2.21배)은 지표 아티팩트가 아니라 실제 결함
- 이중 해리: M5가 pieman 인간 상위 24%(0.578) / tunnel 10명 전원보다 낮음(0.276)
- 🔴 작성자 사전 예상 실패(3연속). PREREG 문구 미수정

### B. B1 모델 강건성 — 라벨러와 동일 모델
- REPORT `outputs/experiments/2026-07-31_b1_model_scale/REPORT.md`
- 러너 `scripts/eacl_phase0/b1_model_scale.py`(신규) · 산출 `b1_{label}.json`
- 신호 `outputs/eacl_phase0/b1_bayes/qwen36_35b/{doc}_words.jsonl`
- **Qwen3.6-35B-A3B**: pieman 0.4062 · pooled **0.3238** · **AUC 0.517~0.529(우연)**
- **GPT-2-XL 재실행**: pooled 0.2836 · AUC 0.533~0.563 (pieman 0.3739 = 기록 0.374 **회귀 확인 통과**)
- **핵심**: 23배 키워도 AUC는 오히려 하락 ⇒ 비판 1 기각. Score 상승분은 순위력 아니라 BOR 완화

### C. B1 을 폐쇄 API 로 — 종전 "실행 불가" 부분 철회
- 러너 `scripts/eacl_phase0/b1_api_logprobs.py` · 절단검증 `scripts/eacl_phase0/b1_topk_fidelity.py`
- 캐시 `outputs/eacl_phase0/b1_bayes/api_logprobs/{doc}_{model}.json`
- 절단 충실도 `outputs/experiments/2026-07-31_b1_model_scale/topk_fidelity.json`:
  top-20 단어KL ρ **0.9866** · Score 0.4050 vs 전체 0.4062 ⇒ **top-20 근사 정당**(top-5는 부족)
- **GPT-4.1 ✅**: verbatim coverage 100.0% · pooled **0.2862** · AUC 0.536~0.594 (= GPT-2-XL 동률)
- **GPT-5/5-mini/5-nano 🔴 불가 확정**: OpenAI 원문 `"logprobs are not supported with reasoning models."` (`unsupported_parameter`). 프롬프트 우회 불가
- **GPT-4o 🔴 거부**: 전문 복사 요청에 `"I'm sorry, I can't do that."`(8토큰). ⚠️ **오프라인 B3도 4o로는 못 돌린다**는 뜻
- **gpt-oss-120b 🔴 실패**: API는 `top_logprobs` 0개 반환 / 로컬은 **CUDA illegal memory access 2회 연속**(mxfp4 커널 + bf16 전개 둘 다). A100 pre-Hopper 제약 — ledger DeepSeek 기각과 동종. 로그 `outputs/eacl_phase0/b1_bayes/gptoss120b/run.log`

### D. 문서 반영 (**부분 완료 — 아래 §마무리 참조**)
- `EACL/pipeline-overview-ko.md` §1.3.1(154~175행) — B1 강건성(A·B) · B2/B3 · 인간 개별 · **AUC 0.717/0.558 출처가 dev-3 아니라 Tier2 7편**임을 명시
- 같은 파일 §2.1.1 — *"모든 베이스라인이 같은 인코더"* → **"M1~M7"** 로 축소
- `EACL/phase0-decisions-and-runs.md` **D-B57** 신규 (어젯밤 LLM 6칸 매트릭스 포함)
- 🔴 **C(폐쇄 API B1)는 REPORT §4.5 에만 있고 pipeline-overview·ledger 에 아직 없다.**

---

## ⬜ 할 것 (우선순위 순)

1. **밀도 매칭 무작위 하한** — `outputs/experiments/2026-07-29_baseline_compare/floors.json` 에 `floor_none`/`floor_all` **뿐**이다. BOR=1(gold 개수)과 BOR=2.21(우리 개수) 두 지점에 무작위 하한 추가. 판정: M5가 BOR-매칭 무작위를 못 이기면 🔴 치명. **별도 사전등록 필요**
2. **C 결과를 REPORT §4.5 에 이미 적었으나 ledger D-B57 §5 에는 미반영** — GPT-4.1/GPT-5/4o/oss 표를 D-B57 에 옮길 것
3. **오프라인 B3 reps=10 재실행+저장** — `scripts/eacl_phase0/dev_b2b3_topline.py` 에 **저장 코드가 없다**. 07-30 GPT-4.1/GPT-5 오프라인 수치는 stdout 으로만 나가 소실. 인용 금지 상태
4. **stale 스크립트 2건** (S5 v3.1 축 개명 `protagonist_perspective`→`protagonist_present`):
   `scripts/eacl_phase0/b1_deployable.py:89` (`KeyError: 'space'`) · `scripts/eacl_phase0/dev_b2b3_topline.py` M3/M1/B1 열
5. gpt-oss-120b 로컬은 **A100에서 포기**. 재시도하려면 Hopper 이상 필요

---

## 🔒 마무리 규약 — **끝나면 반드시 `pipeline-overview-ko.md` 에 반영한다**

이 문서(`EACL/pipeline-overview-ko.md`)가 요약본이자 비판자가 실제로 읽는 문서다.
**이번 비판 3건이 맞은 지점이 전부 "실험은 있는데 이 문서에 없다"였다.** 같은 실수를 반복하지 말 것.

### 사용자 확정 제약 (2026-07-31)
- **절 신설은 가능하나 길게 쓰면 안 된다.** 기존 절 안에 **불릿 1~2개**로 넣는다.
- 수치를 먼저 놓고, 한계는 지우지 않는다(문서 서두 규약).

### 아직 안 적힌 것 — 반영 대상 (좌표: §1.3.1 불릿 목록, 현재 154~175행)
1. **폐쇄 API B1** (§C). 넣을 문장 골자:
   > B1 은 폐쇄 API 로도 돌릴 수 있다 — top-20 절단이 전체 어휘 KL 을 보존하고(단어 ρ 0.9866,
   > Score 0.4050 vs 0.4062) verbatim 복사로 위치를 맞춘다. **GPT-4.1 pooled 0.2862 ·
   > AUC 0.536~0.594 로 GPT-2-XL(0.2836)과 동률** — 프론티어 모델로도 천장이 안 오른다.
   > GPT-5 계열은 `logprobs are not supported with reasoning models` 로 **실행 자체가 불가**.
2. **GPT-4o 전문 복사 거부** — §1.3.1 이 아니라 **B2/B3 불릿(167~170행)** 에 붙일 것.
   *"오프라인 B3 를 4o 로는 못 돌린다"* 는 베이스라인 실행 가능성 사실이다.
3. **밀도 매칭 무작위 하한**(⬜1 완료 후) — 인간 개별 불릿(171~174행) 바로 뒤에.
   이 셋이 합쳐져야 *"BOR 때문에 눌린 것 아니냐"* 반론이 완결된다.

### cascade (CLAUDE.md 최우선 대상)
- `context/methodology/factored-slot.md` — 베이스라인 패널·지표 해석 기준선이 바뀌었으므로 최소 1줄
- `EACL/method-registry.md` §B1 — "폐쇄 API 실행 불가" 서술이 **부분 철회**됨(GPT-4.1 가능)
- `outputs/experiments/2026-07-29_baseline_compare/REPORT.md` — 인간 개별 행·무작위 하한 미반영

## 인프라 메모 (재현에 필수)

- **transformers 4.49 는 `qwen3_5_moe` 미지원.** 동결 venv 를 건드리지 않고 격리 경로에 5.14.1 설치 후 주입:
  `uv pip install --python /root/Hi-OnTop/.venv/bin/python --target <DIR> --upgrade transformers accelerate kernels`
  → 실행 시 `PYTHONPATH=<DIR>`. **venv 자체는 4.49 그대로 유지할 것**
  (이번 세션 DIR = `/tmp/.../scratchpad/tf_new` — **컨테이너 재생성 시 소실**, 재설치 필요)
- `b1_bayes_surprise.py` 에 플래그 2개 추가(기본값은 기존 동작 그대로, 조건 불변):
  `--device-map` · `--dequantize-mxfp4`
- 다중 문서 1회 적재: master jsonl 들을 `cat` 으로 합치면 `doc_id` 로 자동 그룹핑됨
- 🔴 `pkill -f` 금지(자폭 exit 144, 이번 세션에서 또 밟음). **PID 기반 kill 만**
- API 모델명은 **`openrouter/` 접두 필수** (`openai/gpt-4.1` 은 no healthy deployments 400)
- `.env` 로드 후 실행: `set -a; . ./.env; set +a`

## 알려진 한계 (인용 전 반드시 확인)

- **B1 의 c 를 pieman argmax 로 동결**하는 기존 절차 때문에 LM 마다 다른 c 를 받았다
  (GPT-2-XL 0.75 · Qwen35B 1.0 · GPT-4.1 0.0). **Score 차에 작동점이 섞여 있다** ⇒
  1차 근거는 작동점 무관인 **AUC**. 절차를 바꾸면 기존 행과 비교 불가가 되므로 그대로 뒀다.
- **B1 의 실패는 문장 접합 수준에 국한.** 단어 해상도 AUC(~0.70, Kumar 재현)는 이번에 재지 않았다.
  *"B1 은 아무것도 못 한다"* 로 인용 금지.
- **인간 개별 행은 해석 앵커 전용.** gold A 가 그 205명에서 유도됐다(1/205 순환).
  *"우리가 인간을 이겼다"* 로 쓰지 않는다. 양식도 다르다(실시간 청취 vs 텍스트).
- **복사 프롬프트가 B3 원문 문구가 아니다** (`b1_api_logprobs.py` 의 `PROMPT`).
  GPT-4o 거부가 프롬프트 탓인지 모델 정책 탓인지 **미분리**. 재시도하려면 B3 발표 문구로.
- **미시도**: `gpt-4.1-mini` 는 top-20 logprobs 지원 확인됨(사용자 지시로 생략). 필요하면 바로 가능.
- dev-3 = 3편. **봉인 시험지 미개봉.** monkey 인간 개별 행은 영구 결측.
