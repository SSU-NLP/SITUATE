# B3-EventRecall (Panela et al. 2025, Communications Psychology) — 이식·실행 완료, **대조 불가**

**판정: 🟡 이식 완료 + 실행 완료. 🔴 논문 수치와의 대조는 원리적으로 불가 — 종결.**

## 0. 왜 여기서 멈추는가 (2026-08-03 결정)

이 방법은 **비교할 논문 수치가 없다.** 논문의 세그멘테이션 관련 지표(Hamming distance)가
경계 희소성에 지배되어 우리 gold 에서는 예측이 **무경계 하한보다 나쁘게** 나오고
(§5 실측), gold 집계 방식도 다르다. ⇒ 모델·파싱을 논문에 더 맞춰도 **사는 것이 없다.**

한때 "논문 모델 GPT-4 로 다시 돌리자"고 제안했으나 사용자가 중단시켰다 —
*"어차피 똑같은 코드 아닌가"*. 맞다. 기존 실행(`dev_b3ref/`)이 이미 같은 B3 프롬프트로
같은 방법을 돌린 것이고, 남는 차이는 파싱 경로뿐인데 **대조 대상이 없으면 무의미**하다.
([[verify-comparison-target-before-repro]])

**실행된 것:**
| 경로 | 모델 | 파싱 | 산출 |
|---|---|---|---|
| 기존 | gpt-4.1 · temp 0 · 20회 | 우리 `run_to_transitions` | `outputs/eacl_phase0/dev_b3ref/` |
| 이번 | Llama-3.1-8B · greedy (로컬, API 0) | **저자 `parse_llm_output`** | `outputs/experiments/2026-08-03_eventrecall_llama_dev3/` |

**남은 미해결(추적하지 않음):** 두 실행의 파싱 경로가 달라 직접 비교가 안 된다.
기존 실행이 **원 응답을 저장하지 않아**(경계 인덱스만) 저자 파서로 재분석할 수 없기 때문이다.
같은 이유로 `main_result.md` 의 GPT-5 "경계 0개" 행도 판정 불가다(§6). **원 응답 보존이 규약이 되어야 한다.**

---

## 1. 범위
논문의 주 기여는 **회상(recall) 자동 채점**이고 분절은 그 전처리다. 우리가 쓰는 것은
분절뿐이므로 `module/segmentation.py` 의 분절 경로만 옮겼다. 임베딩 기반 회상 채점은 범위 밖이다.
RelatedWork 서술 시 **"세그멘테이션 파이프라인만 차용"**을 반드시 명시할 것.

## 2. 🔴 기존 구현의 오분류를 정정했다
우리 저장소가 "B2 프롬프트"라 부르던 문자열은 실제로 **이 논문(B3)의 프롬프트**였다.
꼬리 문장("...segmented into large event units:")이 B3 와 **완전히 동일**하고, 본문은
**쉼표 하나**만 다르다(`one event ends, and another begins` vs `one event ends and another begins`).
자세한 대조표는 `../llm_direct/REPRO.md` §1.

이 폴더는 **쉼표 없는 B3 원문 그대로**를 쓴다.

## 3. 이식한 절차 (`segment.py`)
1. `clean_text` — 큰따옴표(", ", ") 제거 · `...`→`.` · **하이픈→공백**. 원 코드 그대로.
2. onset + 전문 + offset 을 **한 번에** 호출 (슬라이딩 창 없음).
3. chat, temperature=0, `max_completion_tokens=4096`, frequency/presence penalty 0.
4. `finish_reason == 'stop'` 인 응답만 채택 — **잘린 출력은 버린다**.
5. `parse_llm_output` — 정규식 7단계로 개행을 `\n\n` 로 정규화하고 "Event N:" 류 머리표 제거.
   **순서까지 동일해야** 같은 분절이 나온다.
6. `split_text` — `\n\n` 단위로 사건 분리.

## 4. 왜 아직 못 돌렸나
논문 모델은 **GPT-4** 와 **Meta-Llama-3-8B-Instruct** 다. 2026-08-03 기준 우리 프록시는
예산 소진(429 ExceededBudget)이라 실행 불가. 예산이 복구되면 그대로 돈다.
로컬 vLLM(Qwen3.6-35B)은 논문이 지정한 모델이 아니므로 재현으로 쓸 수 없다.

## 5. "이미 재현한 것 아닌가?" — 아니다 (실측으로 확인)

저장소에는 이미 돌린 결과가 있다. `dev_b2b3_topline.py` 가 dev 3편에서 gpt-4.1 · 20회 반복으로
분절해 ledger 에 수치를 남겼다(B3 mean **0.474** vs M3-S5 0.459 · M1 0.367 · B1 0.284).
예측 원자료도 `outputs/eacl_phase0/dev_b3ref/{doc}_runs.json` 에 남아 있다.

**그러나 그건 "우리 무대에서 돌린 값"이지 "논문이 발표한 숫자와의 대조"가 아니다.** 셋이 어긋난다:

| | 우리가 돌린 것 | 논문 |
|---|---|---|
| 프롬프트 | B3 원문 + 쉼표 1개 | B3 원문 |
| 모델 | **gpt-4.1** | GPT-4 / Llama-3-8B-Instruct (B3) · **davinci-002**(B2, 폐기) |
| 지표 | 우리 DTS Score | Hamming distance 등 논문 자체 지표 |

### 저장된 예측을 논문 지표로 재채점해 봤다 (`rescore_existing_runs.py`, API 비용 0)

| 문서 | 전이 | gold 경계 | 예측 경계 | 우리 Hamming | **무경계 하한** | 판정 |
|---|---:|---:|---:|---:|---:|---|
| pieman | 95 | 18 | 3.5 | 0.163 | 0.189 | 하한을 0.026 상회 |
| tunnel | 503 | 18 | 9.7 | 0.044 | **0.036** | 🔴 무경계보다 나쁨 |
| monkey | 407 | 30 | 14.1 | 0.083 | **0.074** | 🔴 무경계보다 나쁨 |

Michelmann 발표값은 Pieman **0.223 / 0.191**(인간 주석 ~26.4 / 24.5개 기준)이다.
우리 0.163 이 더 낮지만 **"더 잘했다"고 말할 수 없다** — Hamming 은 경계가 희소하면
*아무것도 안 찍기*에 지배되는데, 두 조건의 무경계 하한이 다르다. 논문 gold 는 경계가
~26개(밀도 높음)라 하한이 훨씬 높고, 우리 gold 는 저자 필터링 합의 18개다.

⇒ **gold 집계 방식이 다르면 Hamming 은 비교 불가.** 저장된 예측이 있어도 대조가 성립하지 않는다.
필요한 것은 ① 논문과 같은 gold 집계(개별 행동런 주석) ② 논문과 같은 모델. 둘 다 지금 없다.

## 6. 🔴 reasoning 모델에 temperature 를 무조건 보내고, 실패를 "경계 없음"으로 삼킨다

러너 = `scripts/eacl_phase0/llm_baseline_binary.py` (`--criterion event` 로 B3 기준 사용,
`--label` 로 표 이름을 붙이므로 파일명 `B3event-*` 는 코드에 문자열로 없다).

### 코드 실측 (L145~165)

```python
r = client.chat.completions.create(
    model=args.model, max_tokens=args.max_tokens, temperature=0,   # ← 하드코딩, 항상 전송
    messages=[...], **kw)
...
except Exception as e:
    if use_re and attempt == 0:      # reasoning_effort 만 떼고 재시도
        ...
    time.sleep(2 * (attempt + 1))
_fail["v"] += 1
return "SAME"                        # 🔴 5회 실패 → "경계 없음" 으로 반환
```

두 가지가 겹친다:
1. **`temperature=0` 이 모델 종류와 무관하게 항상 전송된다.** 적응 분기는 `reasoning_effort`
   에만 있고 temperature 에는 없다. reasoning 모델이 400 을 던지면 그대로 예외로 떨어진다.
2. **모든 실패가 조용히 `SAME`(경계 아님)이 된다.** 즉 API 거부·빈 응답·절단이
   **"이 지점은 경계가 아니다"라는 모델 판단과 구분되지 않는다.**

### 코드가 이미 이 함정을 알고 있었다
`--max-tokens` 인자에 붙은 주석(2026-07-30):

> 🔴 GPT-5 계열 대응: 추론 토큰을 먼저 쓰는 모델은 max_tokens=4 면 답이 안 나오고
> 예외 처리로 전부 SAME 이 되어 **무경계 하한이 "모델 성능"으로 오독된다.**

기본값은 여전히 `max_tokens=4` 이고, 큰 값은 실행자가 매번 넘겨야 한다.

### 실제로 무엇이 일어났나 — 구분 불가
| 실행 | 결과 | 해석 |
|---|---|---|
| dev-3 · GPT-5 | Score 0.3914 · 경계 18 | **정상 동작.** ledger 에 "reps=10, temp 0, event 기준" 기록 → 이 경로에선 `temperature=0` 이 수용됐다 |
| 시험지(test3) · GPT-5 | **경계 0** | ledger·REPORT 어디에도 기록 없음 |

같은 스크립트·같은 모델이 dev 에서 동작했으므로 **temperature 400 이 원인이라고 단정할 수 없다.**
남은 후보는 `max_tokens=4` 로 돌린 추론토큰 소진(위 주석의 그 경로) 또는 산발적 API 실패다.
그런데 **둘 다 구분할 수 없다** — `_fail`·`_empty` 카운터가 `llm_results.json` 에 영속되지 않고,
응답 캐시(`cache_B3event-online-r10_gpt-5_*.json`)는 **0 entries** 로 비어 있다.

⇒ `main_result.md` 의 *"GPT-5 는 경계를 0개 찍어 성능이 아니라 퇴화 구성"* 이라는 판정은
**퇴화인지 비실행인지 확인할 근거가 없는 상태**에서 내려져 있다.

### Panela 재현 관점의 함의
Panela 의 핵심 결론이 **"temperature 0 이 인간과 가장 정렬되고 일관적"** 이다. reasoning 모델은
그 설정을 못 쓰므로(`reasoning_effort` 기본값이 `none` 인 5.1/5.2 계열은 예외) **"같은 프롬프트로
백본만 교체한 비교"가 성립하지 않는다.** 우리 기록의 모델 문자열은 `openrouter/openai/gpt-5` 로
**버전 접미사가 없어** 어느 쪽인지 판별되지 않는다.

### 조치 필요
1. `llm_baseline_binary.py` 에 **reasoning 모델 분기** 추가 — temperature 를 보내지 않고
   `reasoning_effort` 를 쓰며, 400 을 조용히 삼키지 않는다.
2. **실패를 `SAME` 으로 떨어뜨리지 말 것.** 최소한 `_fail`·`_empty` 카운터를
   `llm_results.json` 에 영속시켜 "0개 = 성능"인지 "0개 = 비실행"인지 사후 판별 가능하게 한다.
3. 응답 캐시를 실제로 남길 것(현재 비어 있다).
4. 모델 문자열을 **버전까지** 기록할 것.
5. `main_result.md` 의 GPT-5 행에 위 불확정성을 명시할 것.

## 6-b. 🔴 ledger 의 "B3 프롬프트 = B2 프롬프트" 주장은 사실이 아니다

ledger(L3116 부근)에 이렇게 적혀 있다:

> 🔒 **B3(Panela)의 프롬프트 = B2(Michelmann)의 프롬프트** — B3 논문이 명시.

원문 문자열을 직접 대조하면 **다르다**:

| | 본문 | 꼬리 |
|---|---|---|
| B2 (Michelmann) | segmented into **long** events … whenever one **long** event ends | segmented into **long events**: |
| B3 (Panela) | segmented into **large** events … whenever one event ends | segmented into **large event units**: |

B3 가 B2 를 **각색**한 것이지 그대로 쓴 것이 아니다. 이 오인이 "우리 B2 = 실제로는 B3"
혼동의 뿌리다(§5, `../llm_direct/REPRO.md` §1).

## 7. 대조할 발표 주장
논문의 세그멘테이션 관련 핵심 주장은 정확도 수치가 아니라 **"LLM 분절이 개별 인간보다
일관적이다"** (20회 반복·온도 스윕으로 검증)이다. 우리 ledger 에는 이 주장이 우리 재현의
run-level 변이와 긴장 관계라는 기록이 있고, "우리 구현이 B3 와 다른지 미확인"으로 남아 있었다.
§2 에서 밝혔듯 프롬프트는 사실상 B3 였으나 **모델(gpt-4.1)과 파싱 경로가 달랐다.**
예산 복구 후 이 폴더 구현으로 다시 재면 그 미확인 항목이 해소된다.
