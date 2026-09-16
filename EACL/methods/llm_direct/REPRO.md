# B2-LLM-Direct (Michelmann et al. 2023/2025) — 절차 이식 완료, 재현 불가

**판정: 🔴 발표 수치 재현 불가(모델 폐기 + 예산 소진) · ✅ 절차는 충실히 이식.**

---

## 1. 🔴 가장 중요한 발견 — 우리가 "B2" 라고 부르던 것은 B2 가 아니었다

저장소는 `scripts/eacl_phase0/build_tier2_gold.py` 의 `PROMPT_HEAD/TAIL` 을 **B2 프롬프트**라
불러 왔고, `main_result.md` 의 "LLM 직접질의" 행이 그것으로 산출됐다. 원문과 문자열 대조한 결과:

| | 본문 | 꼬리 |
|---|---|---|
| 우리 | ...segmented into **large** events. ... whenever one event ends**,** and another begins. | ...segmented into **large event units**: |
| **B3 (Panela/EventRecall)** | ...segmented into **large** events. ... whenever one event ends and another begins. | ...segmented into **large event units**: |
| **B2 (Michelmann)** | ...segmented into **long** events. ... whenever one **long** event ends and another begins. | ...segmented into **long events**: |

⇒ 우리 꼬리 문장은 **B3 와 완전히 동일**하고, 본문은 B3 와 **쉼표 하나**만 다르다.
B2 와는 `large`/`long` 어휘부터 다르다. **즉 우리는 B3 프롬프트를 B2 라고 부르며 써 왔다.**

절차도 다르다:

| | 모델 | 입력 방식 |
|---|---|---|
| 우리 | gpt-4.1 (chat), max_tokens 32000 | 전문을 **한 번에** |
| B3 | gpt-4 / Llama-3-8B-Instruct, max_completion_tokens 4096 | 전문을 **한 번에** |
| **B2** | **text-davinci-002 (completion)**, 컨텍스트 2048 | **슬라이딩 창으로 여러 번** |

**조치**: 이 폴더(`llm_direct/`)에 B2 를, `../event_recall/` 에 B3 를 각각 원문대로 새로 두었다.
`main_result.md` 의 "LLM 직접질의" 행 명칭은 B3 계열로 정정 대상이다.

## 2. B2 의 핵심 절차 (`segment.py` 로 이식)

한 번에 넣지 않고 컨텍스트 창(2048 토큰)에 맞춰 자른 뒤:
1. 응답을 개행으로 잘라 사건 리스트를 만들고
2. **아직 안 끝났으면 마지막 사건을 버린다** (창 끝에서 잘린 사건이므로)
3. 그 버린 사건의 앞머리를 **원문에서 `rfind` 로 다시 찾아** 다음 창의 시작점으로 삼는다
4. 못 찾으면 25자 밀어 재시도, 그래도 못 찾으면 중단

이 "마지막 사건 폐기 + 재정렬"이 B2 의 정체성이다. 한 번에 넣는 방식과 경계 분포가 달라진다.

### ✅ 검증한 것 — `min_check_length`
3단계의 탐색 길이는 원문의 **최장 비중첩 반복 부분문자열 길이 + 1** 이다. 저자 구현은
O(n²) DP 라 수십만 자 원문에서 못 쓴다. 이분탐색 + 롤링해시 O(n log n) 으로 교체하고
`verify_lrs.py` 로 무작위 400 케이스 대조 → **전부 일치**.

🔴 이때 함정이 있었다. 저자 DP 의 `(j-i) > LCSRe[i-1][j-1]` 조건은 **겹치는 반복을 배제**한다.
처음 접미사 자동자로 (겹침 허용) 최장 반복을 구했더니 400 중 53건이 어긋났다
(예: `aaaa` → 저자 답은 `aa`, 겹침 허용은 `aaa`). 이 값이 1만 달라도 재정렬 위치가 바뀌어
분절이 통째로 달라진다.

## 3. 왜 발표 수치를 재현할 수 없나

### `text-davinci-002` 가 폐기됐다 🔴
OpenAI 가 서비스를 종료했다. 모델을 바꾸면 그건 재현이 아니라 변형이다
([[baselines-stay-as-published]]).

### 프록시 예산이 소진됐다 🔴
2026-08-03 기준 429 ExceededBudget. 어떤 모델로도 호출이 안 된다.

## 4. 우리 저장소에 남아 있던 관련 기록
ledger 에 "④ Michelmann 주장은 우리 조건에서 재현되지 않는다 — *LLM 이 개별 인간보다 합의에
더 가깝다*"가 있다. 그런데 위에서 밝혔듯 **그 실험은 B2 가 아니라 B3 프롬프트로 돌린 것**이다.
따라서 그 기록은 "Michelmann 미재현"이 아니라 **"B3 프롬프트 + gpt-4.1 조건에서 미재현"**으로
범위를 좁혀 다시 써야 한다.
