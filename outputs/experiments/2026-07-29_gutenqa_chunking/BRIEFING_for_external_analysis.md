# GutenQA 검색 평가에서 M5 < LumberChunker — 외부 분석용 사실 브리핑

이 문서는 **사실만** 담는다. 원인 가설·해석·판정·권고는 넣지 않았다. 두 방법의 동작 원리, 실험에서
통제한 것, 측정된 수치, 그리고 두 방법 사이에 존재하는 확인된 차이점만 나열한다.

---

## 1. 과제와 지표

**데이터.** GutenQA (LumberChunker, EMNLP 2024 Findings). Project Gutenberg 공개도메인 서사 도서
**100편**, 도서당 QA **30쌍** = 3,000쌍. 동일 코퍼스가 여러 청크 입도로 함께 공개돼 있다:
`GutenQA`(LumberChunker) / `GutenQA_Paragraphs` / `GutenQA_Recursive` / `GutenQA_Semantic` /
`GutenQA_Propositions`.

**도서 규모 (실측).** 문장 수 min 826 · median 4,839 · max 35,074 · 합 682,799.
**100편 중 99편이 900문장을 초과**한다. 문단 총 248,307개(책당 2,483), 문장/문단 = 2.75.

**정답 판정.** 각 질문에는 `Chunk Must Contain` 이라는 **원문 부분문자열**이 붙어 있다. 검색된 청크가
그 부분문자열을 포함하면 relevant. 판정 코드는 LumberChunker 레포
`Code/Retrieval_Granularities_Eval.ipynb` 원문:

```python
def find_index_of_match(respostas, gold_label):     # relevance
    relevance = []; gold_label = gold_label.lower()
    for item in respostas:
        if gold_label in item.lower():
            relevance.append(1)
            relevance += (len(respostas) - len(relevance)) * [0]   # 첫 히트에서 끊음
            break
        else: relevance.append(0)
    return relevance

def compute_Recall(rel): return 1 if 1 in rel else 0               # hit-rate
def compute_DCG(rel):  return sum((2**rel[i-1]-1)/log2(i+1) for i in 1..len(rel))
```

- relevance 는 **one-hot**(첫 히트 이후 전부 0) ⇒ `DCG@k = 1/log₂(rank+1)`, top-k 밖이면 0.
- **비정규화 DCG**(nDCG 아님).
- `Recall@k` 는 재현율이 아니라 **hit-rate**(top-k 안에 하나라도 있으면 1).
- 유사도 = **생 내적**(`np.dot`), 코사인 아님.
- 논문이 쓴 인코더 = OpenAI `text-embedding-ada-002`. 논문 θ=550, 분절 LLM = Gemini 1.0-Pro.

---

## 2. LumberChunker 의 동작 원리

출처: `Code/LumberChunker-Segmentation.py` (레포 원문).

**최소 단위 = 문단.** `GutenQA_Paragraphs`(HTML 구조가 다양해 저자들이 **수동 추출**한 문단열)를 입력으로
받는다. 각 문단에 `ID {n}: ` 접두사를 붙인다.

**반복 절차 (순차·국소):**

1. 현재 위치 `chunk_number` 부터 문단을 하나씩 누적한다. 누적 토큰 추정치가 **550 을 넘을 때까지**.
   토큰 추정은 `round(1.2 × 단어수)` (1단어 ≈ 1.2토큰 가정).
2. 그 누적 그룹(≈550토큰 분량의 문단들)을 LLM 에 넣는다. 시스템 프롬프트 원문:

   ```
   You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

   Task: Find the first paragraph (not the first one) where the content clearly changes
   compared to the previous paragraphs.

   Output: Return the ID of the paragraph with the content shift as in the exemplified
   format: 'Answer: ID XXXX'.

   Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance
   between identifying content shifts and keeping groups manageable.
   ```

3. 모델 = Gemini(`gemini-pro`) 또는 `gpt-3.5-turbo-0125`, **temperature 0.1**. 안전필터는 전부
   `BLOCK_NONE`. 출력에서 `Answer: ID (\d+)` 를 정규식으로 파싱한다.
4. 파싱된 ID = **다음 청크의 시작**. 청크 = 문단 `[이전 ID, 이번 ID)`. `chunk_number` 를 그 ID 로 옮기고
   1 로 돌아간다. 정규식 실패 시 재시도, 콘텐츠 차단 시 `chunk_number += 1`.
5. `chunk_number ≥ len(문단)-5` 가 되면 종료하고 마지막 청크를 끝까지로 붙인다.

**따라서 LumberChunker 는:**

- **학습이 없다.** 임베딩을 쓰지 않는다. 청크마다 LLM 호출 1회가 필요하다.
- 결정은 **국소적**이다 — LLM 이 한 번에 보는 것은 약 550 토큰 분량의 문단들이고, 그 창 안에서
  전환점 하나를 고른다. 전역 상태·이력·재귀가 없다.
- LLM 이 **원문 텍스트 자체**를 읽는다. 판정 기준은 자연어 문장 *"content clearly changes"* 이며,
  고정된 형식적 타깃 정의가 없다(모델이 그 말을 어떻게 해석하든 그것이 기준이 된다).
- 프롬프트에 **길이 통제 지시가 포함**돼 있다 (*"Avoid very long groups… keeping groups manageable"*).
- θ=550 은 **후보 창의 크기**이고, 실제 청크 크기는 LLM 이 창 안에서 어디를 찍는지에 따라 결정된다.

**실측 산출:** 도서당 **369.2 청크**, 청크당 **327.3 단어** (논문 Table 9 = 334 토큰).

---

## 3. M5-FactoredSlot 의 동작 원리

출처: `src/hi_ontop/m5_default.py`, `src/hi_ontop/factored_slot.py`, `configs/m5_default.json`.
이론 근거 = **EISM (Event-Indexing Model**, Zwaan, Langston & Graesser 1995; Zwaan & Radvansky 1998):
독자는 **time · space · protagonist · causality · intentionality** 다섯 지표를 추적하며, 지표가 바뀌면
상황모델을 갱신한다. 경계 = **다지표 동시 리셋**.

**최소 단위 = 문장 전이.** 입력은 문장열이고, 점수는 **문장 사이마다** 하나씩 나온다.

**① 인코딩.** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, **동결**, 768차원,
L2 정규화. 학습되지 않는다(278M 파라미터 고정).

**② 축별 입력 경로 분리 (`d1=True`).** 축 k 마다 별도 extractor:
`Linear(768→32) → GELU → Dropout(0.4) → Linear(32→8)`.
**슬롯 k 는 자기 축의 8차원만 본다.** 총 입력폭 5×8 = 40.
(대조군 `d1=False` 는 공유 투영 한 벡터를 5슬롯에 브로드캐스트.)

**③ 타입 슬롯 5개 + EntNet 재귀.** 축 = `protagonist_present · space · time · causality · intentionality`,
`slot_dim = 128`. 메모리 h 는 학습된 key 로 초기화(Recurrent Entity Networks 방식). 문장마다:

```
g    = sigmoid(⟨u_t, h⟩ + ⟨u_t, key⟩)              # 슬롯별 게이트
cand = φ(A·h + B·key + C·u_t)
h    ← normalize(h + g · cand)
```

슬롯 간 대칭성은 **key + 축별 입력 경로**로 깨진다(Slot Attention 의 exchangeable 슬롯 문제 회피).

**④ 축별 dynamics (`dynamics="per_slot"`).** 슬롯마다 별도 forward model f_k 가 ŝ_{k,t+1} 을 예측하고,
readout g 가 예측 슬롯들로부터 다음 문장 임베딩 ẑ_{t+1} 을 만든다. 자기지도 손실
`L_pred = ‖ẑ_{t+1} − z_{t+1}‖²`. `pred_space = "slot"`.
⚠️ **경계에서의 상태 리셋(event switching)은 미구현**이다.

**⑤ 축별 이진 head.** 특징 `[s_{t-1}, s_t, Δ, s_t⊙s_{t-1}]`(`head_feat="delta"`, compare-to-previous)에서
축별 change_logit 을 낸다. 학습 파라미터 = 동결 인코더 위 **1.43M**.

**⑥ 신호.** `s_t = Σ_{k=1..5} sigmoid(change_logit_k) ∈ [0, 5]` — 5축 리셋확률의 합.

**⑦ 임계.** `fixed_quantile r = 0.0642` — 문서 내 상위 6.42% 전이를 경계로. 앵커 val 529편
5-fold OOF 로 선택(200 fold 중 192 승). gold 개수·미래 정보 미사용.

**⑧ 학습 신호 = anchor 라벨.** M3-S5 가 만든다: LLM(**Qwen3.6-35B-A3B**, fp8, no-thinking, temp 0,
프롬프트 `p0s5-eism-v3.1`, k=1 단일 draw)이 2단계 **닫힌 enum 추출**을 한다 —
Stage1 은 이야기 전체에서 cast/locations/goals 닫힌 목록을 뽑고, Stage2 는 문장마다 그 목록+고정 enum
안에서만 값을 고른다. **리셋 판정과 carry-forward 는 100% 코드**로 계산된다.

**⑨ 학습 코퍼스.** Project Gutenberg 영어 **단편** 10,598편 (train 10,069 / val 529),
전이 2,777,561개. 도서 길이 필터 **30–900 문장**, 챕터 다수 권(장편)은 통째 배제,
작가당 상한 16편. epochs 8 · batch 32 · lr 3e-4 · seed 0 ·
pos_weight `[2.279, 8.549, 30.0, 1.0, 7.772]`.
**앵커 라벨 자기일치도 S_self = 0.659**(라벨 자체 노이즈, 성능 상한 요인으로 기록됨).

**⑩ 추론 시 LLM 없음.** API 없음. 온라인/인과적 처리 가능.

**⑪ 기각된 구성요소 (기록).** SEM 계열 추론층 — sticky-CRP 스위칭 사전, 우도비 경계 판정, local MAP —
은 구현·검정 후 **기각**됐다. 현행 M5 는 **지도 분류기 + 고정분위수 임계**다.

---

## 4. 이 실험에서 통제한 것

| 항목 | 값 |
|---|---|
| 검색 인코더 | `multi-qa-mpnet-base-dot-v1` — **6개 방법 전부 동일 재임베딩**, 생 내적 |
| 베이스라인 | 공개 청크 파일 **그대로**, 재청킹 없음 |
| M5 | `configs/m5_default.json` 무변경, `models/m5_default/M5.pt` 무변경, 재학습 없음 |
| 절단 제약 | **문단 경계에서만** (베이스라인 최소 단위와 동일하게 맞춘 것) |
| 총 텍스트량 | 전 방법 약 12.1M 단어 (텍스트 손실 없음) |
| 지표 | LumberChunker 레포 코드 그대로 |

**M5 청크 크기 통제.** `segment()` 가 임계와 무관하게 항상 돌려주는 연속 점수 `scores` 만 받아
호출부에서 분위수를 걸었다. 두 규칙을 각각 산출했다:

- `rate R` — **문단 경계만 후보**로 두고 그중 점수 상위 R 비율을 절단.
- `frozen` — 전체 문장 전이에서 상위 6.42%(동결 config 값)를 경계로 낸 뒤 **최근접 문단 경계로 스냅**·중복 제거.

**베이스라인 최소 단위 (실측·확인):**

| 방법 | 최소 단위 | 문장 중간 절단 |
|---|---|---|
| Paragraph | 문단 | 없음 |
| Semantic | 임베딩 거리 브레이크 | 문장 단위 |
| LumberChunker | 문단 | 없음 |
| Recursive | LangChain 문자 예산 + 구분자 폴백 `["\n\n","\n"," ",""]` | **있음** |
| Propositions | LLM 재작성 (원문 정렬 없음) | 해당 없음 |
| M5 (본 실험) | 문단 (제약) | 없음 |

---

## 5. 측정 결과 (전체 100편, DCG@20 / Recall@20, %)

| 방법 | 단어/청크 | 청크/책 | DCG@20 | Recall@20 |
|---|---|---|---|---|
| LumberChunker (Gemini) | 327.3 | 369.2 | **56.22** | 74.57 |
| Recursive (Langchain) | 397.5 | 317.9 | **50.35** | 71.53 |
| m5__rate0.5 | 97.6 | 1242.0 | 45.19 | 64.33 |
| m5__rate0.3 | 162.6 | 745.6 | 44.97 | 64.03 |
| m5__rate0.2 | 243.8 | 497.4 | 43.35 | 63.10 |
| m5__frozen | 352.1 | 344.4 | 42.83 | 62.70 |
| Paragraph | 48.8 | 2483.1 | 42.62 | 58.80 |
| m5__rate0.15 | 324.9 | 373.3 | 41.62 | 61.83 |
| m5__rate0.1 | 486.7 | 249.2 | 39.68 | 60.97 |
| Semantic | 63.1 | 1910.6 | 39.44 | 56.63 |
| m5__rate0.05 | 969.6 | 125.1 | 37.37 | 62.17 |
| Propositions | 12.2 | 9292.0 | 1.76 | 3.07 |

**평균 크기 매칭 대조.** M5 324.9단어 = 41.62 vs LumberChunker 327.3단어 = 56.22 → **차이 −14.60**.
이 두 셀의 ceiling 은 둘 다 82.47%, 총 텍스트량 동일, 인코더 동일.

🔴 **"크기 매칭" 은 평균에 한정된다 — 분포는 전혀 다르다.** 같은 두 셀의 길이 통계:

| 셀 | 평균 | **p50** | p95 | max | **CV** | >800단어 | <50단어 |
|---|---|---|---|---|---|---|---|
| LumberChunker | 327 | **311** | 586 | 6,761 | **0.60** | 1.5% | 1.8% |
| m5__rate0.15 | 325 | **156** | 1,129 | **20,082** | **1.80** | 8.9% | 19.3% |

LumberChunker 는 평균≈중앙값(327/311)이고 M5 는 평균이 중앙값의 2.1배(325/156)다. ⇒ −14.60 은
**평균만 일치시킨 상태에서 측정된 값**이며, 중앙값·CV·꼬리는 통제되지 않았다. 아래 §5-보 참조.

**M5 크기 곡선.** 청크가 작아질수록 단조 상승, 97.6단어에서 45.19 로 포화:
969.6→37.37 · 486.7→39.68 · 324.9→41.62 · 243.8→43.35 · 162.6→44.97 · 97.6→**45.19**.
전 구간에서 Recursive(50.35) 미달.

**k 별 전개 (전체 100편):**

| 방법 | D@1 | D@2 | D@5 | D@10 | D@20 |
|---|---|---|---|---|---|
| LumberChunker | 40.83 | 47.90 | 52.77 | 54.82 | 56.22 |
| Recursive | 33.60 | 40.27 | 46.21 | 48.61 | 50.35 |
| m5__rate0.5 | 30.40 | 36.04 | 41.34 | 43.59 | 45.19 |
| m5__rate0.15 | 26.63 | 32.02 | 37.16 | 39.76 | 41.62 |
| Paragraph | 30.03 | 35.00 | 39.29 | 41.29 | 42.62 |
| Semantic | 26.77 | 31.52 | 35.45 | 37.80 | 39.44 |

**논문 발표 수치와의 대조 (인코더가 ada-002 → 절대값 비교 불가, 순위만):**

| 방법 | 논문 DCG@20 | 우리 | 논문 순위 | 우리 순위 |
|---|---|---|---|---|
| LumberChunker | 62.09 | 56.22 | 1 | 1 |
| Recursive | 54.72 | 50.35 | 2 | 2 |
| Paragraph | 49.00 | 42.62 | 3 | 3 |
| Semantic | 44.74 | 39.44 | 5 | 4 |
| Propositions | 46.19 | 1.76 | 4 | 5 |

### §5-보. 청크 길이 분포 (전 방법) 및 순위 실패 해부

| 방법 | 평균 | p5 | **p50** | p95 | max | **CV** | >800단어 | <50단어 |
|---|---|---|---|---|---|---|---|---|
| recursive | 397 | 239 | 415 | 450 | 5,698 | **0.30** | 0.6% | 0.2% |
| lumberchunker | 327 | 101 | 311 | 586 | 6,761 | **0.60** | 1.5% | 1.8% |
| m5__frozen | 352 | 29 | 201 | 1,154 | 13,336 | **1.37** | 10.2% | 11.4% |
| m5__rate0.15 | 325 | 19 | 156 | 1,129 | 20,082 | **1.80** | 8.9% | 19.3% |
| m5__rate0.3 | 163 | 12 | 83 | 549 | 18,310 | **1.77** | 2.3% | 33.4% |
| m5__rate0.5 | 98 | 8 | 52 | 325 | 11,713 | **1.66** | 0.7% | **48.7%** |
| paragraph | 49 | 3 | 24 | 178 | 5,698 | 1.65 | 0.1% | 72.0% |
| semantic | 63 | 3 | 22 | 239 | 6,760 | 2.81 | 0.8% | 73.7% |

LumberChunker 는 프롬프트에 길이 통제 지시(*"Avoid very long groups… keeping groups manageable"*)와
θ=550 창을 갖고, Recursive 는 문자 예산으로 상한이 강제된다. **M5 에는 길이 통제가 없다.**

**순위 실패 분리.** `compute_DCG` 가 one-hot 이므로 `rank = 2^(1/dcg)−1` 로 순위를 역산할 수 있고,
`gold_in_some_chunk` 로 ceiling 실패와 순위 실패를 분리할 수 있다. ceiling 실패를 제외한 결과:

| 방법 | 순위실패율 | 정답청크 단어수(실패/성공) | 자기분포 백분위(실패/성공) | 희석률(실패/성공) |
|---|---|---|---|---|
| lumberchunker | **9.6%** | 353 / 328 | 62% / 55% | 0.076 / 0.095 |
| recursive | 13.3% | 412 / 417 | 48% / 52% | 0.072 / 0.074 |
| m5__rate0.5 | 22.0% | 158 / 183 | 84% / 87% | 0.163 / 0.161 |
| m5__rate0.3 | 22.4% | 461 / 284 | 93% / 85% | 0.063 / 0.099 |
| m5__frozen | 24.0% | 1,008 / 513 | 93% / 80% | 0.029 / 0.055 |
| m5__rate0.15 | **25.0%** | **1,154 / 545** | **95% / 80%** | 0.024 / 0.051 |
| paragraph | 28.7% | 63 / 120 | 78% / 90% | 0.469 / 0.260 |
| semantic | 31.0% | 125 / 236 | 89% / 95% | 0.220 / 0.129 |

**같은 문항 직접 대조** (M5 실패 & LumberChunker 성공):

| M5 셀 | M5실패/LC성공 | M5성공/LC실패 | 정답청크 단어수 M5 vs LC | 희석률 M5 vs LC |
|---|---|---|---|---|
| rate0.15 | **487** | 105 | **1,293 vs 326** | 0.022 vs 0.097 |
| frozen | 453 | 97 | 1,108 vs 331 | 0.026 vs 0.096 |
| rate0.3 | 417 | 101 | 466 vs 330 | 0.054 vs 0.090 |
| rate0.5 | 413 | 106 | 135 vs 317 | 0.183 vs 0.088 |

`rate0.5` 는 크기 비대칭이 역전(실패 158 < 성공 183)하는데도 순위실패율 22.0% 로 LumberChunker
9.6% 의 2.3배다. `paragraph`·`semantic` 은 **반대 방향**(실패 청크가 성공 청크보다 작고 희석률 0.22~0.47).

**책 길이 상관** (Spearman ρ, 책별 DCG@20 vs 문장수): paragraph −0.418 · semantic −0.374 ·
m5__rate0.5 −0.339 · lumberchunker −0.326 · recursive −0.276 · m5__rate0.3 −0.227 ·
m5__rate0.15 −0.216 · m5__frozen −0.163 (p=0.10).
🔴 **M5 특이적 길이 효과는 관측되지 않는다**(전 방법 공통 음수, M5 가 가장 약함). 단 앵커 학습 길이
상한 900문장 이내 도서가 **100편 중 1편**(book 72, 826문장)이므로 **도메인 일치 조건에서의 성능은
미검정**이다 — "OOD 기각" 이 아니라 **검정 불가(검정력 부족)** 다.

**dev/test (class 층화, seed 20260711, dev 20 / test 80):** 전 방법 ±1.5 안에서 일치.
LumberChunker 57.40/55.93 · Recursive 50.62/50.28 · Paragraph 44.06/42.26 ·
m5__rate0.15 42.14/41.49 · m5__rate0.5 44.47/45.37.

**서사 픽션 85편 부분집합** (PG `Subjects` 픽션 표지 기준, 철학·에세이·운문 11편 + 자서전 4편 제외):
LumberChunker 56.39 · Recursive 50.52 · Paragraph 40.70 · Semantic 38.58. 전체 100편과 순서·크기 동일.

---

## 6. 측정된 벤치마크 속성

**(a) ceiling — gold 가 어떤 청크에도 통째로 없어 원천적으로 0점인 질문.**

| 방법 | 단어/청크 | ceiling(원본) | ceiling(따옴표·줄표 정규화 후) |
|---|---|---|---|
| Propositions | 12.2 | **7.03%** | 7.17% |
| Paragraph | 48.8 | 82.47% | 86.10% |
| Semantic | 63.1 | 82.10% | 86.10% |
| LumberChunker | 327.3 | 82.47% | 88.40% |
| Recursive | 397.5 | 82.47% | 88.40% |
| M5 전 변형 (97.6~969.6단어) | — | **82.47%** | — |

산문 5종은 청크 크기가 **20배 차이나도 ceiling 이 82.1~82.47%** 로 같다. 미발견 526건 중 원문에
그대로 존재하는 것은 **57건**, 따옴표(`’→'`)·줄표(`—→-`) 정규화 후 **178건** 회복,
**348건은 정규화해도 원문에서 찾을 수 없다.**

**(b) `Chunk ID` 컬럼의 정체.** `questions.parquet` 의 `Chunk ID` 가 지목한 청크에 gold 가 있는 비율은
**분모를 ceiling 으로 두면 2,473/2,474 = 99.96%** 다 ⇒ `Chunk ID` 는 **정의상 "gold 를 포함하는
LumberChunker 청크의 인덱스"** 다.
🔴 **따라서 이 수치는 질문이 어느 단위에서 저술됐는지에 대해 아무 정보도 주지 않는다**(동어반복).
이 문서 초판은 같은 값을 3,000 으로 나눈 82.4% 를 "ceiling 82.47% 와 일치 ⇒ LumberChunker 분절
위에서 저술" 의 증거로 제시했으나 **그 추론은 철회한다.**

논문에 기술된 것: 질문은 `gpt-3.5-turbo-0125` 가 "passage" 로부터 1만개 이상 생성한 뒤 사람이 도서당
30개로 필터링했다. 🔴 **그 passage 의 단위는 논문에 기술되어 있지 않다.** `Chunk Must Contain` 필드와
`Chunk ID` 컬럼 자체도 논문에 전혀 기술되지 않는다.
⇒ **생성 단위로 인한 편향은 확인된 것도, 배제된 것도 아니다.** 검정하려면 6개 방법 어느 경계와도
무관한 중립 단위에서 질문을 새로 생성해 재측정해야 한다.

**(c) Propositions 는 이 판정 기준으로 측정 불가.** 명제는 LLM 이 원문을 다시 쓴 문장이라 원문
부분문자열을 거의 포함하지 않는다. 명제를 개별로 펴든(929,196개) 레포처럼 청크 단위로 묶든
(36,917개) ceiling 은 **7% 로 동일**하다. 상한이 7%인 이상 논문의 46.19 는 이 기준으로 산출될 수 없다.
논문은 **명제에 대한 검색 단위와 gold 판정 방식을 명시하지 않는다**(확인). 논문 Appendix A 는 명제
분절이 서사 텍스트에서 대명사 지시가 모호해진다는 점을 지적하며, 명제가 *"for contexts with
fact-based information, like Wikipedia"* 에 더 맞다고 서술한다.

**(f) 논문에 없는 것 (확인).** 논문은 **intrinsic 분절 평가를 보고하지 않는다** — Pk·WindowDiff·
사람 경계 주석이 전혀 없다. 평가는 다운스트림 검색·QA 성능에 한정된다.
또한 **100편 도서 목록을 부록에 싣지 않는다.** Gandhi·Helen Keller·Benjamin Franklin·
Francisco Pinto Balsemão 자서전 4권은 GutenQA 100편이 아니라 **Appendix E.2 의 별개 데이터셋**이다.
GutenQA 100편 중 논픽션 서사는 Jacobs(ID 27) · Wagner(36) · Douglass(37) · Augustine(55) **4편**이다.

**(d) 절단 규칙 대조.** `m5__frozen`(352.1단어·344.4청크/책, DCG@20 42.83)이
`m5__rate0.15`(324.9단어·373.3청크/책, 41.62)보다 청크가 크고 개수가 적은데 점수가 높다.
`rate` 곡선을 352.1단어로 선형 보간하면 41.29 ⇒ 같은 크기 기준 차이 **+1.54**.
두 규칙의 차이는 `rate` 가 문단 경계 지점의 점수만 쓰고 **문단 내부 전이의 점수를 쓰지 않는** 반면,
`frozen` 은 전체 전이에서 고른 뒤 문단으로 스냅한다는 점이다.

**(e) 오염 도서 2편.** M5 앵커 학습 코퍼스와 GutenQA 의 교집합 = Dubliners(PG 2814, 앵커 9조각) ·
The King in Yellow(PG 8492, 15조각) **2편**(둘 다 단편집, 장편은 0편).
해당 2편(질문 60개) vs 나머지 98편(2,940개) DCG@20:
LumberChunker(**앵커 미사용**) 65.35 vs 56.04 (+9.31) · m5__frozen 56.62 vs 42.55 (+14.07) ·
m5__rate0.3 52.75 vs 44.81 (+7.94) · m5__rate0.5 47.96 vs 45.13 (+2.83).

---

## 7. 두 방법 사이의 확인된 차이 목록

| 항목 | LumberChunker | M5-FactoredSlot |
|---|---|---|
| 결정 단위 | 문단 | 문장 전이 (본 실험에서는 문단 경계로 제한) |
| 읽는 대상 | **원문 텍스트** (약 550토큰 창) | 동결 문장 임베딩 → **축별 8차원** extractor 출력 |
| 타깃 정의 | LLM 이 해석하는 자연어 *"content clearly changes"* | 5개 EISM 축 리셋 (다른 LLM 의 닫힌 enum 추출로 라벨링) |
| 학습 | 없음 | 단편 10,598편 · 전이 2.78M · 8 epoch |
| 라벨 품질 | 해당 없음 | 앵커 자기일치도 **S_self 0.659** |
| 문맥 범위 | 창 안 국소, 이력 없음 | EntNet 재귀 상태 (전역 누적), head 는 s_{t-1}·s_t 만 비교 |
| 길이 통제 | **프롬프트에 명시** (*"Avoid very long groups"*) + θ=550 창 | 문서 내 고정분위수 (스스로 개수를 억제하는 메커니즘 없음) |
| 추론 비용 | 청크당 LLM 호출 1회 | LLM 없음 |
| 학습 도메인 길이 | 해당 없음 | **30–900 문장** (장편 배제) |
| 평가 도메인 길이 | — | **median 4,839 문장 · max 35,074 · 99/100 편이 900 초과** |
| 이벤트 스위칭 | 해당 없음 | **미구현** (구현·검정 후 기각) |

---

## 8. 원자료 위치

```
outputs/experiments/2026-07-29_gutenqa_chunking/
  retrieval/*.json          질문별 DCG/Recall/ceiling 원자료 (12개 방법)
  SUMMARY_TABLES.md         부분집합 × split 전체 표
  ceiling_diag.json         ceiling 및 손실 원인 분해
  contamination.json        100편 PG ID 매핑 + 앵커 교집합
  book_subsets.json         부분집합·split 정의 (동결)
  m5_scores/book*.npz       M5 전이별 연속 점수 (재분석용)
  data/chunks/*.jsonl       12개 방법의 실제 청크 텍스트
  ref/                      LumberChunker 원본 분절·평가 코드
scripts/gutenqa/            00~04 재현 스크립트 (run_all.sh)
```

**재현:** `bash scripts/gutenqa/run_all.sh`
