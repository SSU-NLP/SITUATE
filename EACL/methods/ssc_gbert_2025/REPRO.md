# BERT-SSC / Llama 파인튜닝 (Zehe et al. 2025, NAACL — 현 SotA) — 재현 불가

**판정: 🔴 주력 모델 재현 불가 · 🟡 프롬프팅 baseline 은 실행 준비 완료(예산 대기).**

---

## 1. 대조하려던 발표 수치

### Table 1 · Train-Full 학습 SSC (완화 F1, tolerance t=3)

| 모델 | STSS-Test-1 | STSS-Test-2 | OOD-Test | Test-Full |
|---|---:|---:|---:|---:|
| **GBERT-Large + Half Stride** | 0.68 | 0.66 | 0.69 | **0.68** |
| GBERT-Large | 0.59 | 0.61 | 0.60 | 0.60 |
| LLPro(fiction-gbert-large) + Half Stride | 0.62 | 0.57 | 0.66 | 0.62 |
| LLPro | 0.51 | 0.47 | 0.43 | 0.47 |
| GBERT-Base + Half Stride | 0.37 | 0.46 | 0.46 | 0.42 |
| GBERT-Base | 0.26 | 0.37 | 0.33 | 0.31 |

### Table 2 · 파인튜닝 없는 프롬프팅 (Test-Full)
llama3:8b 0.13 · llama3:70b 0.34 · llama3.1:405b 0.34 · gpt-4o-mini 0.14 · **gpt-4o 0.45**

## 2. 왜 막혔나

### 학습 코퍼스가 공개되지 않는다 🔴
> "Since many of the texts both in the original dataset as well as in our extension are
> copyrighted, we cannot publish the full dataset" — 논문 §5

공개된 것은 standoff 주석(경계 오프셋 + 문장 경계 + 원문 md5)뿐이다. Train-Full 32편 중
퍼블릭 도메인은 6편 남짓(Kleist·Storm·Droste-Hülshoff·Kafka·Ebner-Eschenbach 등)이고
나머지는 통속소설이라 합법적으로 구할 수 없다. 사용자 결정(2026-08-03)에 따라
**저자에게 코퍼스를 요청하지 않는다.**

### 체크포인트도 공개되지 않았다 🔴
HuggingFace `LSX-UniWue` 계정의 공개 모델 22개를 전수 확인했지만 장면분절 파인튜닝
체크포인트는 없다(LLäMmlein·ModernGBERT 등 사전학습 모델뿐). 레포에도 가중치가 없다.

⇒ **학습도 추론도 불가.** 6편으로 학습한 모델은 논문 모델이 아니므로 발표 수치와
대조하면 독자를 오도한다([[baselines-stay-as-published]]).

## 3. ✅ 재현된 것 — 이 논문의 **평가 지표**

주력 수치는 못 맞춰도, 논문의 지표 정의는 우리 손으로 정확히 복원했다.
`../common/verify_metric_port.py` 가 저자 `ssc/evaluation.py` 를 직접 import 해
**무작위 2,000 케이스에서 완화 F1·IoU 전부 비트 단위 일치**를 확인했다.

이게 중요한 이유: 처음 논문 설명대로 직접 짠 완화 F1 은 **틀렸다.** 저자 구현은 세 군데가
직관과 다르다 —
1. 창이 비대칭이다. `labels[i-t : i+t]` 라 실제 허용 구간은 **[i−t, i+t−1]**, ±t 가 아니다.
2. 위치가 아니라 **전이 타입까지** 같아야 매칭된다.
3. TP 와 FP 를 독립 집계한다. precision 의 분모 `TP+FP` 가 예측 개수와 일치하지 않는다.

이 셋을 모르고 채점하면 어떤 방법을 돌려도 "재현 실패"가 나온다.

## 4. 🟡 실행 준비만 된 것 — 프롬프팅 baseline

`prompting.py` — 저자 `prompting/classify.py` 이식. 문장마다 이진 질의를 던지고 문맥을
앞뒤 대칭으로 `512×0.8` 토큰까지 채운다. 프롬프트 문자열·재시도 규칙(10회)·`INPUT_PERCENTAGE`
모두 원본 그대로. langchain·wuenlp 의존을 걷어내 우리 데이터로 바로 돈다.

**현재 실행 불가 사유 = LLM 프록시 예산 소진**(429 ExceededBudget, 2026-08-03).

**실행해도 논문과 직접 대조는 불가**: Table 2 는 Test-Full(4+2+3편) 수치인데 우리는
**Test-2 2편**만 있다. Test-2 는 고전 문학이라 통속소설 중심의 Test-1 과 성격이 다르고,
같은 모델이 test set 별로 크게 갈린다는 것이 이 논문 Table 1 의 관찰이다. 따라서 산출은
**참고 수치**로만 쓴다.

## 5. 방법 요약 (RelatedWork 서술용)

문장 시퀀스를 겹치는 문맥 창으로 잘라 인코더에 넣고 각 문장이 장면 경계인지 분류한다.
논문의 핵심 기여 하나가 **Half Stride** — 이전 연구는 창을 붙여서 만들었는데(full stride),
그러면 창 경계에 놓인 문장이 문맥을 거의 못 받아 그 자리 오류가 몰린다. 창을 절반씩
겹쳐 만들고 문장마다 여러 예측을 합치면 그 오류가 사라진다(GBERT-Large 0.60 → 0.68).
인코더는 `deepset/gbert-large`(최고) · `deepset/gbert-base` · `lkonle/fiction-gbert-large`.
Llama 파인튜닝은 Unsloth QLoRA(`unsloth/llama-3-8b-Instruct-bnb-4bit`)에 CoT-List 템플릿을 쓴다.
