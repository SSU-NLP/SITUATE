# PREREG — M9-AttnEntityDrift (LLM attention 상위 토큰=엔티티 거리 기반 분절)

**작성 2026-07-31 20:34 KST. 어떤 Score 도 계산하기 전에 작성했다. 이 문서는 `--stage primary`
실행 이후로는 수정하지 않는다.**

수정이 허용되는 유일한 경우는 §7 "결과 이전 수정(pre-result amendment)" — 1~4단계(Score 를 하나도
찍지 않는 단계)의 진단 결과 때문에 HP 를 바꿔야 할 때뿐이고, 그때도 기존 문장을 고치지 않고
타임스탬프 찍힌 절을 **append** 한다.

---

## 1. 가설

EM-LLM(Fountas et al. 2025)은 경계를 **토큰 surprisal** `−log P(x_t | x_<t)` 로 잡는다.
이를 **"LLM 이 집중하는 토큰(=엔티티) 사이의 거리"** 로 교체하면 서사 분절이 개선되는가?

배경 사실: EM-LLM 원본은 per-key attention mass 를 이미 계산한다
(`em_llm/attention/context_manager.py:716` `exc_repr_score`,
`em_llm/attention/dot_product_attention/torch_impl.py:26` `tmp.sum(dim=-2)`)
— 그러나 **검색(retrieval)과 `repr_topk` 에만 쓰고 경계 판정에는 전혀 쓰지 않는다.**
이 실험은 그 미사용 신호를 경계 판정으로 돌린다.

**주장하려는 것은 "attention 으로 고른 토큰이 중요하다" 이지 "어휘 응집도가 surprisal 보다 낫다"
가 아니다.** 후자는 이미 알려진 결과다. 따라서 판정은 §5 의 G2·G3 에 걸린다.

---

## 2. 데이터 — dev-3 (연습장)

🔴 **dev-3 는 연습장이다. 절대 Score 를 성능으로 인용하지 않는다. Δ 와 방향만 읽는다.**
🔒 **Tier1 봉인 시험지(Run / MyMothersLife / GoHitler)는 이 실험에서 열지 않는다.**

`baseline_compare.load_dataset('dev3')` 로 로드. 2026-07-31 확인:

| doc | 문장 | 전이 | gold | gold 밀도 | T(tok, Qwen2.5-7B) | tok/문장 중앙값 | ≤6tok 문장 |
|---|---|---|---|---|---|---|---|
| pieman | 96 | 95 | 18 | 0.1895 | 1,274 | 11 | 15.6% |
| tunnel | 504 | 503 | 18 | 0.0358 | 5,205 | 9 | 30.8% |
| monkey | 408 | 407 | 30 | 0.0737 | 6,637 | 14 | 10.8% |
| **합** | 1,008 | **1,005** | **66** | **0.0657** | 13,116 | — | — |

### 사전 확인된 기준선 (공식 채점기 `hi_ontop.dts_scoring.score_dialogues`, 2026-07-31 재현)

- **no-boundary floor = 0.27896** (Pk=WD=0.44208, F1=0)
- all-boundary = 0.19802
- 노이즈 바닥 = **0.06** (프로젝트 기존 판정치, 단일 추출 Score 차이 0.06 미만은 해석하지 않음)
- 기존 dev-3 동일 하니스: M5 0.4190 / M4 0.3861 / M1-CosDrift 0.3330 / TextTiling 0.3212 /
  CSM 0.2880 / RoBERTa 0.2799 / GraphSeg 0.2212 / GreedySeg 0.1988

⚠️ 사용 가능 대역은 **0.279 → 0.419** 로 좁다. floor 를 못 넘는 방법이 기존 baseline 9종 중 5종이다.

---

## 3. 방법 (실행 전 확정, 이 값들로 primary 를 돌린다)

### 3.1 신호

```
core(i) = 문장 i 토큰에서 sep("\n")·공백/구두점-only 제외;  비면 sep 만 제외로 폴백
a_t     = mean_layers mean_heads mean_{q∈Q(i)} p̃[q,t]
          p̃ = 각 쿼리 행을 core(i) 위에서 재정규화
E_i     = a 상위 k 토큰,  k = min(K, |pool|)

A  embed      e_i = l2(mean_{p∈E_i} l2(h^(r)_p));  d_i = 1 − e_{i−1}·e_i
A' embed_ctx  baseline_compare.preds_m1 수식(m=2, ρ=0.7, a=0.5) 에 e_i 를 입력
B  setdrift   d_i = 1 − |E_{i−1}∩E_i| / |E_{i−1}∪E_i|   (엔티티 키 = 단어병합·소문자·구두점strip)
C  recency    d_i = mean_{e∈E_i} min(i − last[e], CAP)/CAP,  신규 엔티티 → 1.0

d[0] = 0.0 (sentinel),  d[i] = dist(i−1, i)  → 길이 n, gold(길이 n−1)에 정확히 정렬
```

### 3.2 고정 HP (primary 전체에서 불변)

| 항목 | 값 |
|---|---|
| LM | `Qwen/Qwen2.5-7B` (base) — `segment_manual_emllm.py` 와 동일해야 like-for-like 성립 |
| query scope | `sentnext` (문장 i 의 q>t + 문장 i+1 전체) |
| 정규화 | `rowsent` (쿼리 행을 core(i) 위에서 재정규화) |
| layers | `18-27` (= round(0.625·28); EM-LLM `refine_from_layer=20/32` 비율) |
| head 집계 | `mean` |
| K (top-k) | **3**, `fixed` |
| repr layer | `hidden_states[18]`, 토큰별 L2 → 평균 → 다시 L2 |
| 엔티티 단위 | `word` (offset_mapping → 공백구분 단어 → 소문자 → 구두점 strip → dedup) |
| CAP | 20 |
| refine | `none` |

### 3.3 작동점 (셋 다 **사전 동결**, 이 실험에서 튜닝하지 않음)

| id | 규칙 | 동결 출처 | 역할 |
|---|---|---|---|
| **OP-1** | `adaptive_boundaries(d, c=1.0, mode="online_nf")` | `hi_ontop_v2.py:193`, D-B39 | **headline**, 진짜 deployable |
| OP-2 | `surprise_boundaries(d, gamma=1.1, win=0, warmup=3)` | `segment_manual_emllm.py:163`, EM-LLM 원본 기본값 | surprisal 과 like-for-like |
| OP-3 | 문서별 분위수 `d >= quantile(d, 1−0.0642)` | D-B30 앵커 동결 경계율 | M5/M4/M1 과 같은 표에 올리기 |

- `online_nf` 는 `warmup` 인자를 무시한다(`cnt<2` 하드코딩) — REPORT 에 명기한다.
- 🔴 **OP-3 caveat:** dev-3 gold 밀도 0.0657 vs ANCHOR_R 0.0642. 앵커 코퍼스에서 동결했으므로
  선택-누출은 아니지만 **사실상 dev-3 밀도에 정합**이다 = oracle-count 한 걸음 앞.
  headline 은 OP-1 이고, OP-3 는 등장하는 모든 자리에 "near-gold density" 라벨을 붙인다.

### 3.4 평가

- **평가 모드 = deployable.** 방법이 경계 개수를 스스로 정한다. matched@gold / top-N /
  density-matched 는 성능 주장에 쓰지 않는다.
- **주 지표 = Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)**, `score_dialogues` 만 호출(재구현 금지).
  AUC 는 보조이며 headline 에 쓰지 않는다.
- 끝-turn 규약, 마지막 turn label·pred = 0.

---

## 4. 사전등록 격자 (primary) — 판정은 **여기서만** 나온다

- signal ∈ {`embed`, `embed_ctx`, `setdrift`, `recency`} (4)
- selection ∈ {`attn-topk`, `attn-topk ∩ spacy`, **`ALL-core-tokens`(attention 없는 쌍둥이)**} (3)
- OP ∈ {OP-1, OP-2, OP-3} (3)

= **36 셀.** 이것이 사전등록된 family 다.

### 통제군 (같은 실행·같은 하니스)

1. **no-boundary floor** — 0.27896 을 정확히 재현해야 한다. 아니면 하니스가 틀린 것이고 실험 중단.
2. **random @ matched density** — 각 셀의 자기 예측 밀도로 20 seed.
3. **shuffled-rank** — 같은 k·같은 pool, 문장 내 순위만 무작위, 5 seed. "*어느* 토큰인지가 중요한가"를 직접 시험.
4. **EM-LLM surprisal** — `encode_doc` + `surprise_boundaries`, refine ∈ {none, modularity, conductance} × 3 OP.
   **dev-3 수치가 아직 존재하지 않으므로 이 실험에서 생산한다.**
5. **M1-CosDrift** — 3 OP 재계산 (`m4m5_phase1_anchor/data/dev3_emb.npy`).

### secondary (부록, 탐색적으로만 표기 — 판정에 쓰지 않음)

layers {all, 0-8, 9-17} · scope {next, sent, window:256, retrieved} · norm {count, none} ·
topk {5, frac0.25} · repr_layer {14, 28} · head-agg perhead-vote · entity_unit token ·
center_embed · CAP {10, 50} · refine {modularity, conductance} · drop-init 128.

---

## 5. 판정 기준 — **수치를 보기 전에 확정한다**

| gate | 양 | 통과 조건 |
|---|---|---|
| **G0 생존** | 셀의 Score | > **0.339** (= floor 0.27896 + 노이즈 0.06). 미달 → "no-boundary floor 도 못 넘음" 으로 종결 |
| **G1 교체 대상 대비** | Δ_surp = M9 − EM-LLM-surprisal (같은 OP·같은 LM·같은 스크립트) | OP-1 과 OP-2 **양쪽에서** > **+0.06**. 부호가 갈리면 "결론 없음(노이즈)" |
| **G2 attention 선택의 기여** | Δ_sel = `attn-topk` − `ALL-core-tokens` (같은 signal·OP) | \|Δ_sel\| < 0.06 이면 **G1 결과와 무관하게** headline 문장은 "attention 선택의 측정 가능한 기여 없음 — 이것은 어휘/임베딩 응집도 신호다" 여야 한다 |
| **G3 무작위 순위 대비** | Δ_rand = `attn-topk` − `shuffled-rank` (5 seed 평균) | > **+0.06** |

**논문이 되는 것은 G2·G3 이지 G1 이 아니다.** surprisal 은 이겼는데 자기 자신의 attention 없는
쌍둥이와 비긴다면, 그것은 "문장 수준 어휘 응집도가 surprisal 보다 낫다"는 기지 결과이지
이 실험이 주장하려는 것이 아니다.

### 다중비교 규칙
OP 당 4 signal × 3 selection = 12 비교. 한 셀이 우연히 +0.06 을 넘는 것은 드물지 않다.
따라서 **이긴 셀은 G2·G3 도 통과하고 OP-1/OP-2 에서 부호가 유지되어야 한다.**
사후 best-cell 체리피킹 금지. "가장 좋은 셀"만 뽑아 보고하지 않고 36 셀 전부를 표로 낸다.

### DEGENERATE 규칙
`std(d[1:]) < 1e-9` 또는 `frac_truncated > 0.30` 또는 `frac_ties > 0.5` 인 셀은
**Score 가 아니라 `DEGENERATE` + 진단치**로 보고한다. 좋은 Score 가 나와도 채택하지 않는다.

---

## 6. 사전 예상 실패 모드와 그것을 잡는 검사 (전부 Score 계산 **전**)

| # | 실패 | 검사 (단계) | 레드라인 |
|---|---|---|---|
| **F1** | top-k 가 내용이 아니라 **위치 슬롯** (문장 첫/마지막 토큰만 반복 선택) | 4단계, pieman 만, Score 없이 진단 덤프 | top-1 상대위치가 첫 또는 마지막 bin 에 >50% · 상위 20 문자열이 선택의 >40% · 불용어/구두점 >30% |
| **F2** | **동적 범위 없음** — tunnel 9토큰 문장 + K=3 이면 인접 문장이 엔티티를 하나도 공유 못 해 setdrift≡1.0 → std≈0 → `>=` 때문에 전 구간 경계 → 0.198 | 3단계, CPU only, GPU 쓰기 전. B/C 를 ALL-content-words 로 돌려 통계만 출력 | 어느 문서든 `P(Jaccard=0) > 0.7`. embed 는 4단계에서 `std(1−cos) < 0.01` |
| **F3** | **transition 정렬 off-by-one** — 3가지 인덱스 규약이 만난다 | 1단계 `--selftest`, GPU 0, ~5s | `d_oracle = [0]+gold` 를 실제 경로로 통과시켜 **Score = 1.0** 이어야 함. floor 0.27896 재현. self-test 미통과 시 `--stage primary` 진입 거부 |

---

## 7. 결과 이전 수정 (pre-result amendment) 규칙

1~4단계는 **Score 를 하나도 출력하지 않는다.** 이 단계의 진단 때문에 HP 를 바꿔야 하면
(예: tunnel 절단율이 레드라인을 넘어 K=3→5) 이 절 아래에 **타임스탬프 + 무엇을 왜 바꿨는지**를
append 한다. Score 를 보지 않았으므로 합법이다.

**`--stage primary` 를 한 번이라도 실행한 뒤에는 §3·§4·§5 를 수정하지 않는다.**
기준 미달이면 기준을 고치지 않고 미달로 기록하고, "알고도 진행" 한 경우 그 이유만 적는다.

### 수정 이력

#### A1 — 2026-07-31 21:1x KST · signal B 퇴화 → `setdrift_w` 추가 (원본 유지)
**계기.** 3단계 사전점검(ALL-content-words, attention 없음, **Score 미계산**)에서 §6 F2 레드라인이
세 문서 모두에서 걸렸다: `P(Jaccard=0)` = pieman 0.842 / tunnel 0.859 / monkey 0.735 (> 0.70).
고장 양상은 예상의 **거울상**이었다 — "전 구간 경계"가 아니라 d 가 1.0 에 뭉쳐 μ+σ 가 1 을 넘어
**OP-1 이 경계를 0개** 냈다(pieman·monkey). attention top-k=3 은 집합이 더 작아 반드시 더 나쁘다.

**조치.**
1. 사전등록된 `setdrift`(인접 1문장 Jaccard)는 **그대로 둔다.** 예상대로 DEGENERATE 로 보고한다.
2. 신호 `setdrift_w` 를 primary family 에 **추가**한다 — 고치는 지점은 *비교 대상 창*이지 신호의 취지가 아니다:
   `d_i = 1 − |E_i ∩ (∪_{j=i−W..i−1} E_j)| / |E_i|` (직전 W문장 블록 대비 신규 엔티티 비율, 인과적).
   분모를 `|E_i|` 로 두는 이유: 과거 블록은 계속 커지므로 Jaccard 는 창 크기에 편향된다.
3. **W = 6 하나로 고정.** 출처 = NLTK TextTiling 발표 기본 블록크기 `k=6`
   (`run_emllm_dts.py --sanity` 가 쓰는 값). 외부·발표값이므로 dev-3 누출이 없다.
   ⚠️ **W 를 쓸어보지 않는다.** 사전점검 수치로 W=6 은 P(d=1) 을 0.53/0.55/0.37 로 낮추지만
   `frac_ties_max` 는 pieman 0.526 / tunnel 0.549 로 여전히 §5 DEGENERATE 문턱(0.5)을 넘는다.
   **문턱을 통과할 때까지 W 를 찾지 않는다** — 그것은 진단이 통과할 때까지 튜닝하는 것이다.
   §5 의 DEGENERATE 규칙은 **완화하지 않는다.** 넘으면 넘은 대로 DEGENERATE 로 적는다.

**격자 영향.** signal 4종 → 5종. primary 36셀 → **45셀**, OP 당 비교 12 → **15**.
다중비교 규칙(§5)은 늘어난 family 에 그대로 적용한다.

#### A2 — 2026-07-31 21:1x KST · `attn-spacy` 는 DEGENERATE 예상, K 는 바꾸지 않는다
spaCy 내용어/NER 필터 후 문장당 후보 수가 pieman 4.86 / tunnel 3.57 / monkey 6.05 라
K=3 절단율이 **0.500 / 0.605 / 0.360** 으로 §5 문턱(0.30)을 전부 넘는다.
"후보 4개 중 3개 고르기"는 선택이 아니다.
**K 를 이 arm 에서만 낮추지 않는다** — selection arm 마다 K 가 다르면 G2 비교가 불공정해진다.
K=3 을 유지하고 `attn-spacy` 는 DEGENERATE 로 보고한다.
(참고: `attn-topk` 의 core 풀 절단율은 0.104 / 0.240 / 0.069 로 전부 문턱 미만 → K=3 유지 타당.)

#### A3 — 2026-07-31 21:1x KST · char offset 버그 수정 (HP 변경 아님)
Qwen BPE 토큰은 선행 공백을 포함한다(`" McGowan"`). `char_start` 를 토큰 시작 offset 으로 두면
공백 위치를 가리켜 spaCy 토큰 span(`t.idx` = 단어 시작)과의 매칭이 대량 실패했다
(실패 868 / 2,928 / 4,652 건 → 문장당 내용어가 0.6~0.8 개로 잘못 집계).
`char_start` 를 **토큰의 첫 비공백 문자** offset 으로 바꿔 매칭 실패 **0** 건이 되었다.
이는 사양 변경이 아니라 **버그 수정**이며, `assert_token_offsets` 가 재발을 막는다.
