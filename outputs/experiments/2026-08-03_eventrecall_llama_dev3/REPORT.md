# B3-EventRecall — 저자 파이프라인(Llama, 로컬) dev-3 실행 REPORT

**판정: 실행 성공 · 논문 수치 대조는 하지 않음(원리적 불가).**
그 대신 **저자 기본 설정의 실패 모드**를 실측했다 — 긴 문서에서 출력이 잘리는데
Llama 경로에는 **그것을 걸러내는 장치가 없다.**

방법론 판정의 canonical = `EACL/methods/event_recall/REPRO.md`.

---

## 1. 실험 setup

| 항목 | 값 |
|---|---|
| **목적** | 저자 모델(Llama)·저자 파싱으로 B3 분절 파이프라인이 端到端 도는지 확인 |
| 러너 | `EACL/methods/event_recall/run_llama_dev3.py` |
| 방법 | 저자 `module/segmentation.py::llama_segmentation` 이식 (`event_recall/segment.py`) |
| 모델 | **`meta-llama/Llama-3.1-8B-Instruct`** 🔴 논문은 3.0(`Meta-Llama-3-8B-Instruct`) — 다른 모델 |
| 디코딩 | **greedy** — 저자 코드가 `do_sample` 을 안 넘겨 HF 기본값(False)이 적용된다 |
| max_new_tokens | 4096 (저자 기본값) |
| 프롬프트 | 저자 `prompt_onset`/`prompt_offset` 문자열 그대로 (쉼표 없음) |
| 전처리 | 저자 `clean_text` (따옴표 제거 · `...`→`.` · **하이픈→공백**) |
| 파싱 | 저자 `parse_llm_output` 정규식 7단계 → `split_text`(`\n\n`) |
| 데이터 | dev-3 (pieman 96문장 / tunnel 504 / monkey 408), 사람 gold |
| 비용 | **API 0** · 로컬 GPU |

## 2. 결과

| 문서 | 문장 | gold 경계 | 사건 | 예측 경계 | **복사 충실도** |
|---|---:|---:|---:|---:|---:|
| pieman | 96 | 18 | 26 | 25 | 0.99 ✅ |
| tunnel | 504 | 18 | 10 | 9 | 0.87 |
| monkey | 408 | 30 | 41 | 39 | **0.24 🔴** |

복사 충실도 = 출력 단어수 / 원문 단어수. 이 방법은 **원문을 통째로 복사**하게 하므로
1.0 근처여야 정상이고, 낮으면 출력이 잘렸다는 뜻이다.

## 3. 🔴 핵심 발견 — 저자 기본값이 긴 문서를 조용히 망친다

**monkey 는 원문의 24% 만 복사하고 끊겼다.** 그런데도 사건 41개·경계 39개가 나왔다 —
즉 **문서 앞 1/4 에 39개 경계가 몰려 있고 나머지 3/4 는 경계가 하나도 없다.** 뒤쪽은
모델이 본 적조차 없다.

이게 조용히 지나가는 이유는 저자 코드의 비대칭이다:

| 경로 | 잘림 처리 |
|---|---|
| GPT (`gpt_segmentation`) | `finish_reason != 'stop'` 인 반복을 **버린다** |
| **Llama (`llama_segmentation`)** | **필터가 없다** — HF pipeline 에 `finish_reason` 개념이 없고, 잘린 출력이 그대로 채택된다 |

⇒ **Llama 경로로 이 방법을 긴 문서에 쓰면 잘린 결과가 정상 결과처럼 반환된다.**
`max_new_tokens=4096` 이라는 저자 기본값은 대략 3,000단어 이상 문서에서 이미 부족하다
(dev-3 기준 tunnel 3,432단어에서 0.87, monkey 5,060단어에서 0.24).

우리 러너가 `copy_fidelity` 를 함께 재도록 만든 것은 이 때문이다 — **저자 코드에는 없는 안전장치**다.

## 4. 디코딩 설정 — 내 이식 오류를 정정한 기록

처음 이식할 때 `do_sample=True` 를 넣어 temp 0.1 샘플링으로 돌렸다. **저자 코드는
`do_sample` 을 넘기지 않아 HF 기본값대로 greedy 다**(`module/segmentation.py:54-57`).
따라서 저자 코드의 `if temperature == 0: temperature = 0.1` 강제는 **실질적으로 무효**이고,
Llama 경로는 애초에 결정적이다.

| pieman 조건 | 예측 경계 (gold 18) |
|---|---:|
| 샘플링 temp 0.1 (내 오류) | 44 |
| **greedy (저자 실제 동작)** | **25** |
| 참고: gpt-4.1 temp 0 (기존 실행, 우리 파서) | 4 |

수정만으로 44 → 25 로 줄었다. 남은 25 vs 4 는 **모델 능력 차이**이며 고칠 대상이 아니다.

## 5. 왜 논문 수치와 대조하지 않는가

논문 지표(Hamming distance)가 경계 희소성에 지배되어 **우리 gold 에서는 무경계 예측이
더 좋게 나오는 구간이 있다** — tunnel 0.044 vs 무경계 하한 0.036, monkey 0.083 vs 0.074
(`EACL/methods/event_recall/rescore_existing_runs.py` 실측). 논문 gold 는 경계가 ~26개로
밀도가 다르므로 하한 자체가 달라 값을 나란히 놓을 수 없다.
⇒ **대조 불가**로 판정하고 여기서 종결한다([[verify-comparison-target-before-repro]]).

## 6. 한계 / 검증 미해결

- **모델이 논문과 다르다** (3.1 vs 3.0). "저자 설정 그대로"가 아니다.
- **기존 gpt-4.1 실행과 직접 비교 불가** — 그 실행은 우리 파서를 썼고 **원 응답을 저장하지
  않아** 저자 파서로 재분석할 수 없다. 같은 이유로 `main_result.md` 의 GPT-5 "경계 0개" 행도
  판정 불가다. **원 응답 보존을 규약으로 삼아야 한다.**
- `iters=1` 만 돌렸다. 저자 기본값이지만 논문의 반복·합의 절차는 재현하지 않았다.
- `max_new_tokens` 를 올린 조건은 돌리지 않았다 — §5 대로 대조 대상이 없어 정밀도를
  더 올릴 이유가 없다고 판단했다.
