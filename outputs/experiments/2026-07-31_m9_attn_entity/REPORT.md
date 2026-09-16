# M9-AttnEntityDrift — LLM attention 상위 토큰(=엔티티) 거리 기반 문장 분절 (dev-3)

작성 2026-07-31 21:01 KST · 사전등록 [`PREREG.md`](PREREG.md) (실행 전 작성, §7 수정이력만 append)
스크립트 `scripts/segment_attn_entity.py` · `scripts/eacl_phase0/run_attn_entity_dev3.py` ·
`scripts/eacl_phase0/spacy_token_tags.py` · `tests/test_attn_mass.py`
산출 `results_primary.json` · `results_shuffled.json` · `results_controls.json` ·
`results_secondary.json` · `gates.json` · `diagnostics.json` · `precheck.json` ·
**`preds/`** (224개 — 🔒 저장소 원칙상 **예측·거리신호가 원자료**, 집계 Score 는 파생물) ·
`cache/` (258 MB, 레이어·헤드별 attention mass 전량 보존 → 재추출 없이 재분석 가능)

> 🔴 **dev-3 는 연습장 3편이다. 아래 절대 Score 를 성능으로 인용하지 말 것 — Δ 와 방향만 읽는다.**
> 🔒 Tier1 봉인 시험지(Run / MyMothersLife / GoHitler)는 열지 않았다.

---

## 0. 요약 — 결과 먼저

**사전등록 45셀 중 G0(생존)을 통과한 셀은 0개다.** 어느 조합도 no-boundary floor + 노이즈
(0.2790 + 0.06 = **0.3390**)를 넘지 못했다.

| 물음 | 답 |
|---|---|
| attention 엔티티 거리가 EM-LLM surprisal 을 대체할 수 있나 | **아니다.** 45/45 셀 G0 탈락. 최고 0.3386 (문턱 0.3390 미달) |
| attention 선택이 기여하는가 (**G2**) | **기여 없음.** 30셀 중 28셀이 \|Δ_sel\| < 0.06. attn-topk 의 Δ_sel 중앙값 **−0.0205** (있다면 오히려 해롭다) |
| attention 순위가 무작위 순위보다 나은가 (**G3**) | **아니다.** 30/30 FAIL. Δ_rand 중앙값 **−0.0175** |
| surprisal 은 이겼나 (**G1**) | 일부 셀이 이기지만 **의미 없다** — surprisal 자체가 floor 아래다(0.2711 / 0.2699). OP-1·OP-2 동시 통과 arm 은 0개 |
| 기전은 무엇인가 | attention 상위 토큰이 **엔티티가 아니라 기능어·위치 슬롯**이다 (§2.3) |

**한 문장 결론.** "LLM 이 집중하는 토큰 = 엔티티" 라는 전제가 dev-3 서사에서 **경험적으로 성립하지
않는다.** Qwen2.5-7B 의 문장 단위 attention mass 상위 토큰은 절반이 불용어·구두점이고, 문장의 첫
내용어에 위치 편향돼 있다. 따라서 그 위에 어떤 거리 정의를 얹어도 엔티티 거리가 되지 않는다.

### 게이트 판정

| gate | 기준 (PREREG §5) | 결과 |
|---|---|---|
| **G0 생존** | Score > 0.3390 | ❌ **0 / 45** |
| **G1 surprisal 대비** | Δ_surp > +0.06, OP-1·OP-2 양쪽 | ❌ 0 / 15 arm |
| **G2 attention 선택 기여** | \|Δ_sel\| ≥ 0.06 | ❌ **기여없음 28 / FAIL 1 / PASS 1**(PASS 셀은 DEGENERATE, n_pred=6) |
| **G3 무작위 순위 대비** | Δ_rand > +0.06 | ❌ **0 / 30** |

**PREREG §5 의 강제 문구 발동:** G2 가 노이즈 안이므로, G1 결과와 무관하게 headline 은
*"attention 선택의 측정 가능한 기여 없음"* 이다. 그리고 이 경우엔 G0 조차 통과하지 못했다.

---

## 1. 실험 setup

### 1.1 목적
EM-LLM(Fountas et al. 2025)의 경계 신호인 토큰 surprisal `−log P(x_t|x_<t)` 를
**"LLM 이 집중하는 토큰(=엔티티) 사이의 거리"** 로 교체했을 때 서사 분절이 개선되는지.

전제가 된 사실: EM-LLM 원본은 per-key attention mass 를 **이미 계산한다**
(`em_llm/attention/context_manager.py:716` `exc_repr_score`,
`dot_product_attention/torch_impl.py:26` `tmp.sum(dim=-2)`) — 그런데 검색(retrieval)과
`repr_topk` 에만 쓰고 **경계 판정에는 전혀 쓰지 않는다.** 그 미사용 신호를 경계로 돌린 것이 M9 다.
(부수 정정: `scripts/segment_manual_emllm.py:20-21` 의 "attention repr_score 를 밖에서 못 뽑아"
라는 주석은 사실이 아니다. 원본 트리에 값이 존재하며, 본 실험은 그것을 밖에서 재구성해 썼다.)

### 1.2 방법

**추출.** attention 모듈에 `register_forward_pre_hook(with_kwargs=True)` 를 걸어 모듈 입력
(post-`input_layernorm` hidden)과 `position_embeddings=(cos,sin)` 을 캡처하고, forward 후 모듈 자신의
`q_proj`/`k_proj`/`apply_rotary_pos_emb`/`repeat_kv` 로 post-RoPE Q,K 를 재구성한다.
`output_attentions=True` 는 불가 — T=6,637 에서 한 레이어 `[1,28,T,T]` fp16 = 2.47 GB, 28 레이어
전부 보관 = 69 GB. 게다가 sdpa 가 조용히 eager 로 폴백한다.

**⚠️ 계획 대비 정정(비용에 유리한 방향).** `norm="rowsent"` 는 **전체 causal prefix softmax 가
불필요하다.** 전역 분모가 정확히 소거되기 때문이다:

```
p̃[q,t] = p[q,t] / Σ_{t'∈core(i)} p[q,t']
       = (e^{s_qt}/Z_q) / (Σ_{t'∈core(i)} e^{s_qt'}/Z_q)
       = e^{s_qt} / Σ_{t'∈core(i)} e^{s_qt'}          ← Z_q 소거
```
즉 core(i) 위의 restricted softmax 와 **동일**하다(인과 마스크는 그대로 적용). 문서 전체 T×T 를 만들
필요가 없어 문서당 추출이 **7~9초**로 끝났다. `count`/`none` 정규화는 여전히 전체 prefix 가 필요해
블록 경로(`_mass_full`)로 간다. 두 경로의 일치는 `tests/test_attn_mass.py` 가 강제한다(§1.7).

**신호** (`d[0]=0` 센티넬, `d[i] = dist(i−1, i)`, 길이 n → gold 길이 n−1 에 정렬):

| id | 정의 |
|---|---|
| `embed` | `e_i = l2(mean_{p∈E_i} l2(h^(18)_p))`; `d_i = 1 − e_{i−1}·e_i` |
| `embed_ctx` | 위 `e_i` 에 M1-CosDrift 수식(`baseline_compare.py:127-134`, m=2, ρ=0.7, a=0.5) 적용 |
| `setdrift` | `d_i = 1 − \|E_{i−1}∩E_i\| / \|E_{i−1}∪E_i\|` (사전등록 원안) |
| `setdrift_w` | `d_i = 1 − \|E_i ∩ (∪_{j=i−6..i−1}E_j)\| / \|E_i\|` (**PREREG 수정이력 A1 로 추가**, W=6 = NLTK TextTiling 발표 k) |
| `recency` | `d_i = mean_{e∈E_i} min(i − last[e], 20)/20`, 신규 → 1.0 |

**엔티티 = attention mass 상위 k 토큰**, `k=3` 고정, core(i) = 문장 i 토큰에서 sep·공백·구두점 제외.
엔티티 키는 **서브워드를 원문 단어로 병합** 후 소문자·구두점 strip·중복제거
(`" McG"+"owan"` 같은 BPE 분할이 Jaccard 를 죽이므로).

### 1.3 데이터

| doc | 문장 | 전이 | gold | gold 밀도 | T(tok) | core/문장 | spaCy 내용어/문장 |
|---|---|---|---|---|---|---|---|
| pieman | 96 | 95 | 18 | 0.1895 | 1,274 | 10.69 | 4.86 |
| tunnel | 504 | 503 | 18 | 0.0358 | 5,205 | 7.69 | 3.57 |
| monkey | 408 | 407 | 30 | 0.0737 | 6,637 | 13.30 | 6.05 |
| **합** | 1,008 | **1,005** | **66** | **0.0657** | 13,116 | — | — |

로딩 = `baseline_compare.load_dataset('dev3')`. spaCy 캐시는 **별도 venv**
(`.venv-eacl-extractors`, torch 없음)에서 만들었고 문장 sha1 대조로 로딩 일치를 강제했다.

### 1.4 HP

LM `Qwen/Qwen2.5-7B`(base, `segment_manual_emllm.py` 와 동일) · scope `sentnext` · norm `rowsent` ·
layers `18-27` (= round(0.625·28), EM-LLM `refine_from_layer=20/32` 비율) · head-agg `mean` ·
K=3 fixed · repr layer `hidden_states[18]`, 토큰별 L2 → 평균 → 재L2 · entity unit `word` ·
CAP=20 · refine `none` · bf16 · A100 1장.
**seed 없음** — 추출은 결정적이다. seed 는 random / shuffled 통제군에만 있다.

### 1.5 작동점 (셋 다 사전 동결, 이 실험에서 튜닝하지 않음)

| id | 규칙 | 출처 |
|---|---|---|
| **OP-1** | `adaptive_boundaries(d, c=1.0, mode="online_nf")` | `hi_ontop_v2.py:193`, D-B39 — **headline** |
| OP-2 | `μ + 1.1σ` online full-past | `segment_manual_emllm.py:163`, EM-LLM 원본 기본값 |
| OP-3 | 문서별 분위수 `1 − 0.0642` | D-B30 — 🔴 **near-gold density** (dev-3 gold 밀도 0.0657) |

두 가지를 REPORT 에 명기한다:
- `online_nf` 는 `warmup` 인자를 **무시한다**(`cnt<2` 하드코딩, `hi_ontop_v2.py:199`).
- `adaptive_boundaries` 는 `return out + [0]`(L221)로 **길이 n 의 턴 단위** 예측을 준다.
  `score_A` 가 자기 후행 0 을 또 붙이므로 전이 벡터로 쓰려면 후행 0 을 떼야 한다.
  이 off-by-one 은 self-test 가 첫 실행에서 잡았다(§1.7).

### 1.6 metric

`Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)`, `hi_ontop.dts_scoring.score_dialogues` **만** 호출.
평가 모드 = **deployable**(방법이 경계 개수를 스스로 결정). AUC 는 산출하지 않았다.

### 1.7 배선 검증 (전부 통과)

| 검사 | 방법 | 결과 |
|---|---|---|
| attention 재구성 비트정확성 | 3-layer 랜덤 Qwen2(CPU, 다운로드 없음) — pre-hook 재구성 확률 vs 표준 eager `attn_weights` | max\|Δ\| **< 1e-6** |
| `rowsent` 싼 경로 == 정의대로 | 전체 softmax 후 core(i) 재정규화한 값과 대조 | max\|Δ\| **< 1e-6** |
| 블록 분할 불변성 | `_mass_full` block ∈ {4, 7, 1024} | 상대 Δ **1.9e-07** (float32 비결합, 버그 아님) |
| **전이 정렬** | 오라클 `d=[0]+gold` → 임계 0.5 → `score_A` | **Score = 1.0** (정확히) |
| off-by-one 대조 | 각 OP 정렬 vs 1칸-shift | 전부 Δ > 0.2 |
| floor 재현 | no-boundary | **0.2789608866** (사전 확인값과 1e-12 이내) |
| 상수신호 퇴화가드 | `d ≡ 1` → 3 OP | 경계 0개 |
| spaCy 오프셋 정합 | 전 토큰 `sents[i][c:c+len(frag)] == frag` | 매칭 실패 **0** 건 |
| **하니스 교차검증** | M1-CosDrift OP-3 | **0.3330** = 기존 발표값 0.3330 **정확 일치** |

마지막 항목이 가장 중요하다 — 내 하니스가 `2026-07-29_baseline_compare` 와 동일 배선임을 뜻한다.

---

## 2. 결과

### 2.1 PRIMARY 45셀 — 상위 12 (전체는 `results_primary.json` / `gates.json`)

**G0 문턱 0.3390. 통과 0개.**

| # | signal | selection | OP | Score | n_pred | Δ_surp | Δ_sel | Δ_rand | DEGENERATE |
|---|---|---|---|---|---|---|---|---|---|
| 1 | setdrift | attn-spacy | OP-1 | 0.3386 | 22 | +0.068 | +0.049 | −0.001 | frac_truncated 0.60 |
| 2 | setdrift_w | attn-topk | OP-2 | 0.3237 | 56 | +0.054 | +0.039 | +0.042 | frac_ties_max 0.64 |
| 3 | setdrift_w | attn-spacy | OP-1 | 0.3203 | 7 | +0.049 | +0.020 | −0.004 | frac_truncated 0.60 |
| 4 | recency | attn-spacy | OP-1 | 0.3186 | 6 | +0.048 | +0.067 | −0.014 | frac_truncated 0.60 |
| 5 | setdrift_w | attn-topk | OP-1 | 0.3137 | 87 | +0.043 | +0.014 | −0.010 | frac_ties_max 0.64 |
| 6 | **embed** | **all-core** | **OP-1** | **0.3124** | 181 | +0.041 | — | +0.018 | — ← **DEGEN 아닌 최고** |
| 7 | embed_ctx | all-core | OP-1 | 0.3067 | 170 | +0.036 | — | +0.020 | — |
| 8 | setdrift | all-core | OP-2 | 0.3030 | 21 | +0.033 | — | +0.024 | frac_ties_max 0.69 |
| 9 | setdrift_w | all-core | OP-1 | 0.2999 | 74 | +0.029 | — | −0.024 | — |
| 10 | embed_ctx | all-core | OP-2 | 0.2942 | 155 | +0.024 | — | +0.018 | — |
| 11 | **recency** | **attn-topk** | **OP-1** | **0.2912** | 139 | +0.020 | +0.040 | −0.041 | — ← **DEGEN 아닌 attn-topk 최고** |
| 12 | setdrift | attn-topk | OP-1 | 0.2902 | 10 | +0.019 | +0.000 | −0.050 | frac_ties_max 0.88 |

읽는 법 세 가지:
- **DEGENERATE 아닌 최고가 `all-core`(=attention 없는 쌍둥이)이다.** attention 을 쓴 최고 비퇴화
  셀은 0.2912 로 floor(0.2790)보다 0.012 높을 뿐 — 노이즈(0.06) 한참 안쪽.
- 상위권 상당수가 `n_pred` 가 6~22 로 극단적으로 적다. gold 66 대비 경계를 거의 안 내면서 floor 를
  살짝 넘은 것이라 "잘 맞혔다"가 아니다.
- 상위 5셀 전부 DEGENERATE 다.

### 2.2 통제군

| key | Score | n_pred | 비고 |
|---|---|---|---|
| floor · no-boundary | **0.2790** | 0 | 사소한 하한 |
| floor · all-boundary | 0.1980 | 1,005 | |
| **EM-LLM surprisal** OP-1 | **0.2711** | 124 | 🆕 dev-3 수치, 이 실험에서 생산 |
| **EM-LLM surprisal** OP-2 (원본 γ=1.1) | **0.2699** | 116 | 🆕 |
| EM-LLM surprisal OP-3 | 0.2434 | 67 | 🆕 |
| **EM-LLM + modularity** (원본 파이프라인) | **0.2978** | 116 | 🆕 |
| EM-LLM + conductance | 0.2559 | 116 | 🆕 |
| M1-CosDrift OP-1 | 0.3611 | 144 | |
| M1-CosDrift OP-2 | 0.3598 | 121 | |
| M1-CosDrift OP-3 | 0.3330 | 67 | = 기존 발표값 (하니스 교차검증) |

**🆕 새 사실: EM-LLM surprisal 은 dev-3 서사에서 no-boundary floor 아래다.** raw surprisal
0.2711 / 0.2699 < 0.2790. 원본의 그래프 리파인먼트(modularity)를 붙여야 0.2978 로 floor 를 겨우
넘고, conductance 는 오히려 더 내려간다(0.2559). 이 수치는 저장소에 없던 것이다.

**random @ matched density (20 seed, 표본).** 셀 Score 가 같은 밀도의 무작위와 구분되지 않는다:

| 셀 | 셀 Score | random @ 같은 밀도 |
|---|---|---|
| embed\|attn-topk\|OP-1 | 0.2492 | 0.2473 ± 0.0253 |
| embed\|attn-topk\|OP-3 | 0.2686 | 0.2730 ± 0.0300 |
| embed_ctx\|attn-topk\|OP-3 | 0.2744 | 0.2730 ± 0.0300 |
| recency\|attn-topk\|OP-1 | 0.2912 | 0.2852 ± 0.0335 |
| recency\|attn-topk\|OP-2 | 0.2665 | 0.2875 ± 0.0192 |

**shuffled-rank (G3, 5 seed).** 같은 pool·같은 k 에서 순위만 무작위화. **30/30 FAIL**,
attn-topk 의 Δ_rand 중앙값 **−0.0175**, 범위 [−0.0497, +0.0416].
즉 **어느 토큰을 고르는지가 결과를 바꾸지 않는다.**

### 2.3 퇴화·기전 진단 (`diagnostics.json`) — 왜 안 됐는가

**F1 레드라인이 세 문서 모두에서 걸렸다** (scope=`sentnext`, K=3):

| doc | top-1 첫bin | top-1 끝bin | top20 점유 | 불용어·구두점 | 절단@K3 | 레드라인 |
|---|---|---|---|---|---|---|
| pieman | 0.448 | 0.146 | **0.509** | **0.502** | 0.104 | top20, 불용어 |
| tunnel | **0.595** | 0.153 | 0.357 | **0.403** | 0.240 | 위치, 불용어 |
| monkey | **0.539** | 0.132 | **0.432** | **0.488** | 0.069 | 위치, top20, 불용어 |

pieman 최빈 선택 12개: `and, i, the, that, a, dean, so, pie, he, she, 't, me`
→ 내용어는 `dean`, `pie` 둘뿐이다.

**query scope 를 바꿔도 사라지지 않는다** (pieman):

| scope | top-1 첫bin | top-1 끝bin | top20 점유 | 불용어·구두점 |
|---|---|---|---|---|
| `sentnext` (기본) | 0.448 | 0.146 | 0.509 | 0.502 |
| `next` | 0.083 | 0.500 | 0.410 | 0.393 |
| `sent` | **0.802** | 0.000 | 0.586 | 0.607 |

`sent` 는 top-1 이 문장 첫 토큰인 비율 0.802 — **순수 위치 artifact**. `next` 가 가장 낫지만 역시
레드라인 미달. **통과할 때까지 설정을 찾지 않았다**(그것은 진단이 통과할 때까지 튜닝하는 것이다).

**F2 사전점검** (ALL-content-words, GPU 이전, Score 미계산). 사전등록 `setdrift` 는 여기서 죽었다:

| doc | signal | P(d=1) | n_distinct | std | frac_ties_max | n_pred OP-1 |
|---|---|---|---|---|---|---|
| pieman | setdrift | **0.842** | 14 | 0.078 | 0.842 | 0 |
| pieman | setdrift_w6 | 0.526 | 15 | 0.281 | 0.526 | 0 |
| pieman | recency | 0.358 | 44 | 0.245 | 0.358 | 0 |
| tunnel | setdrift | **0.859** | 23 | 0.141 | 0.859 | 10 |
| tunnel | setdrift_w6 | 0.549 | 19 | 0.378 | 0.549 | 4 |
| monkey | setdrift | **0.735** | 43 | 0.107 | 0.735 | 0 |
| monkey | setdrift_w6 | 0.374 | 47 | 0.291 | 0.374 | 0 |

고장 양상은 예상의 **거울상**이었다 — "전 구간 경계"가 아니라 d 가 1.0 에 뭉쳐 μ+σ 가 1 을 넘어
**OP-1 이 경계를 0개** 냈다. `setdrift_w`(W=6) 추가로 P(d=1) 은 내려갔지만 `frac_ties_max` 는
pieman·tunnel 에서 여전히 0.5 초과 → DEGENERATE 판정 유지.

### 2.4 SECONDARY (부록, 탐색적 — 판정에 쓰지 않음)

최고 **0.3311** (`embed|attn|no-l2|OP-3`) — 역시 G0 문턱 미달. 어느 축도 그림을 바꾸지 않았다.

| 조건 | Score(최고 OP) | 비고 |
|---|---|---|
| `embed` no-l2 | 0.3311 | 최고. 단 OP-1 에선 0.2998 이고 n_pred=3 (사실상 무예측) |
| `setdrift` layers=all | 0.3242 | DEGENERATE |
| `setdrift` layers=0-8 | 0.3158 | DEGENERATE |
| `embed` center | 0.3099 | anisotropy 보정해도 변화 없음 |
| `recency` cap=10 | 0.3048 | DEGENERATE |
| `setdrift` scope=sent / unit=token / lemma | 0.2939 / 0.2902 / 0.2902 | 전부 DEGENERATE |
| **`setdrift` perhead-vote** | **0.2902** | EM-LLM 원본 `get_block_k` 의 헤드별 top-k. mean 과 사실상 동일 |
| `embed` repr=28 / 14 | 0.2899 / — | 표현 레이어 무관 |

`perhead-vote` 결과가 중요하다: "헤드 평균이 엔티티 추적 헤드를 뭉갰을 것"이라는 가설을 배제한다.
원본 방식대로 헤드별로 뽑아도 같다.

---

### 2.5 🔶 사후 추가 측정 — "엔티티 집합 거리" 틀 자체의 천장 (판정에 쓰지 않음)

**⚠️ 이 절은 primary 결과를 본 뒤에 추가로 잰 것이다. 사전등록 격자가 아니며 G0~G3 판정에
포함되지 않는다.** 묻는 것: *"엔티티 추출을 attention 이 아니라 **제대로 하는 추출기**로 바꾸면
회복되는가?"* → selection `all-spacy` = spaCy 내용어/NER **전부**(top-k 없음, attention 무관).
이것이 "완벽에 가까운 엔티티 추출기를 가정했을 때의 상한" 근사다.

| signal | OP | **all-spacy (천장)** | n_pred | all-core | attn-topk | vs floor | DEGENERATE |
|---|---|---|---|---|---|---|---|
| embed | OP-1 | 0.2953 | 178 | 0.3124 | 0.2492 | **+0.016** | — |
| embed | OP-2 | 0.2772 | 154 | 0.2852 | 0.2623 | −0.002 | — |
| embed | OP-3 | 0.2673 | 67 | 0.2981 | 0.2686 | −0.012 | — |
| embed_ctx | OP-1 | 0.2569 | 177 | 0.3067 | 0.2774 | −0.022 | — |
| setdrift | OP-1 | **0.3386** | **22** | 0.2898 | 0.2902 | **+0.060** | frac_ties 0.88 |
| setdrift_w | OP-1 | 0.3186 | **6** | 0.2999 | 0.3137 | +0.040 | frac_ties 0.69 |
| recency | OP-1 | 0.3186 | **6** | 0.2514 | 0.2912 | +0.040 | frac_ties 0.51 |

**결론: 천장 최고 = 0.3386 으로 G0 문턱(0.3390)조차 넘지 못한다.** 그마저 DEGENERATE 이고
n_pred=22 (gold 66) — 경계를 거의 안 내서 얻은 값이다. **DEGENERATE 아닌 최고는 0.2953**
(embed|OP-1), 무경계 하한 대비 **+0.016** 으로 노이즈(0.06) 한참 안쪽이다.

대조로, **엔티티를 전혀 안 쓰는 M1-CosDrift 가 OP-1 에서 0.3611** 이다 — 문장 임베딩 코사인
드리프트만으로 엔티티 천장을 **+0.066** (비퇴화 기준) 앞선다.

⇒ **실패 원인은 엔티티 추출기가 아니라 "엔티티 집합 거리" 라는 틀 자체다.** 추출기를 고치는
어떤 개선(노름 가중 attention, retrieval head 선별 등)도 이 천장 위로 올려주지 못한다.

부수 관찰: `embed` 는 `all-core`(0.3124) > `all-spacy`(0.2953) 로, **내용어로 제한하면 오히려
나빠진다.** 이는 D-B38 의 기존 판정(*"태거를 덧붙이면 이기고 대체하면 진다"*)과 같은 방향이다.

🔧 이 측정 과정에서 진단 버그 1건 수정: `frac_truncated` 가 top-k 를 쓰지 않는 `all-*` 셀에도
계산돼 가짜 DEGENERATE 가 붙었다. `rank != "all"` 조건을 넣어 고쳤고,
**primary 45셀의 Score·DEGEN 플래그는 전혀 바뀌지 않았음을 재실행으로 확인**했다.

---

## 3. 해석

**1) 실패 지점은 거리 정의가 아니라 엔티티 추출이다.** 세 거리 정의(임베딩 드리프트 / 집합 교체율 /
재등장 거리)를 다 구현했고 전부 실패했지만, 원인은 셋 다 공유하는 **앞단**에 있다. §2.3 이 보여주듯
"LLM 이 집중하는 토큰"의 절반이 불용어·구두점이고 top-1 은 문장 첫 내용어에 몰린다. 엔티티가 아닌
것들 사이의 거리를 아무리 정교하게 재도 엔티티 거리가 되지 않는다.

**2) 이것은 attention sink 처리 실패가 아니다.** `rowsent` 재정규화로 전역 sink 는 산술적으로
소거했고 sep·구두점은 후보에서 뺐다. 그럼에도 남는 것은 **문장 내부의 국소적 기능어 선호**다.
이는 알려진 성질(빈도 높은 토큰이 attention mass 를 많이 받음)의 문장 단위 발현이며, scope·layer·
head 집계 어느 축으로도 제거되지 않았다(§2.3, §2.4).

**3) G2·G3 가 판정의 핵심이라는 사전등록의 판단이 옳았다.** 만약 G1(surprisal 대비)만 봤다면
"여러 셀이 surprisal 을 +0.05 이상 이겼다"는 잘못된 긍정 결론이 나왔을 것이다. 실제로는
(a) surprisal 자체가 floor 아래라 이겨도 의미가 없고, (b) attention 없는 쌍둥이가 더 낫고
(Δ_sel 중앙 −0.021), (c) 무작위 순위와 구분되지 않는다(Δ_rand 중앙 −0.018).

**4) 노이즈 범위.** dev-3 노이즈 바닥 0.06 기준으로, attn-topk 계열 셀의 Δ_sel·Δ_rand 는 전부
그 안이다. 즉 "attention 이 약간 해롭다"고 주장할 수도 없다 — **구분되지 않는다**가 정확한 진술이다.

**5) 문서별 밀도 불일치 × 적응 임계.** gold 밀도가 pieman 0.190 / tunnel 0.036 으로 5배 차이나는데
OP-1 은 문서마다 같은 c=1.0 을 쓴다. 집합 신호처럼 값이 1.0 에 뭉치는 경우 μ+σ > 1 이 되어 경계를
0개 내고(=floor 와 동점), 반대로 OP-3 는 고정 6.42% 를 강제해 최대 917개를 낸다. 상위 셀들의
`n_pred` 가 6~22 와 540~917 로 양극화된 것이 이 상호작용의 직접 증거다.
**이는 M9 만의 문제가 아니라 이 저장소가 이미 안고 있는 "경계 개수 제어" 미해결 문제와 같은 것이다.**

**6) 부산물로 얻은 새 수치.** EM-LLM 을 dev-3 서사에서 공식 채점기로 잰 수치가 저장소에 없었는데,
이번에 생산했다: raw surprisal 0.2711 / 0.2699 (**floor 아래**), +modularity 0.2978, +conductance
0.2559. EM-LLM 의 서사 분절 성능 자체가 약하다는 점은 M9 실패와 별개로 기록할 가치가 있다.

---

## 4. 판정

| gate | 결과 | 판정 |
|---|---|---|
| G0 생존 | 0 / 45 | ❌ no-boundary floor + 노이즈를 못 넘음 |
| G1 surprisal 대비 | 0 / 15 arm (OP-1·OP-2 동시) | ❌ (게다가 surprisal 자체가 floor 아래라 의미 없음) |
| G2 attention 선택 기여 | 기여없음 28 / 30 | ❌ **측정 가능한 기여 없음** |
| G3 무작위 순위 대비 | 0 / 30 | ❌ |

**결론: M9-AttnEntityDrift 는 현재 사양으로 폐기한다.**
"raw attention mass 상위 토큰 = 엔티티" 라는 추출 전제가 성립하지 않는 것이 원인이고, 거리 정의를
바꾸거나 작동점을 바꿔서 회복되지 않는다(45 primary + 20 secondary 셀에서 확인).

### 다음 iteration 결정

- **폐기**: raw attention top-k 를 엔티티 추출기로 쓰는 노선. 재시도 금지.
- **폐기 (§2.5 사후 측정 근거)**: 엔티티를 **더 좋은 추출기**로 뽑아 같은 거리를 재는 노선 전체.
  spaCy 내용어/NER 전체를 써도 천장이 0.3386(퇴화) / 0.2953(비퇴화)로 G0 문턱 아래이고,
  엔티티를 전혀 안 쓰는 M1-CosDrift(0.3611)보다 낮다.
  ⇒ **노름 가중 attention `‖α·v‖`(Kobayashi 2020) · retrieval head 선별 같은 "추출기 개선" 계열은
  이 천장에 막힌다.** 단 §2.5 는 사후 측정이라 이 판정은 **탐색적**이며, 재시도하려면 별도 사전등록.
- **유일하게 남은 방향**: 엔티티라는 중간 단계를 **없애는** 것. 주제를 토큰 집합이 아니라
  *서로 attend 하는 문장들의 커뮤니티*로 정의하고(문장×문장 attention 그래프) EM-LLM 의
  `modularity` 분할을 그 위에 얹는 안. 통제군에서 EM-LLM+modularity(0.2978)가 surprisal 계열 중
  유일하게 무경계 하한을 넘었다는 것이 약한 방증이다. **미검증.**
- **미검증으로 남긴 것**: `scope=retrieved` (문장 i 의 쿼리가 **과거 전체** 어디를 보는가).
  사용자의 "llm이 집중하는 토큰들" 이라는 표현에 가장 가깝고 signal C 를 진짜 검색거리로 만들지만,
  이번 격자에 구현하지 않았다(§5). **다만 §2.3 의 기전(기능어 편향)은 scope 를 바꿔도 유지되었으므로
  기대값은 낮다.**
- **별건으로 기록**: EM-LLM dev-3 수치(§2.2)는 baseline 표에 넣을 값이다.

---

## 5. 한계 / 검증 미해결 (제거 금지)

1. **dev-3 3편뿐.** 연습장이고 pieman 은 이미 9건의 선택압을 받았다. 절대 Score 는 성능이 아니다.
   Tier1 봉인 시험지는 열지 않았다 — 이 음성 결과로 시험지를 열 이유는 없다.
2. **seed 없음.** 추출은 결정적이라 M9 셀에 seed 변동이 없다. random(20)·shuffled(5)만 seed 가 있다.
   따라서 "셀 간 0.02 차이"의 유의성을 말할 수 없고, 판정은 노이즈 0.06 기준으로만 했다.
3. **`scope=retrieved` 미구현.** 계획 단계에서 "부록에서 가장 유망"으로 표시했으나 이번에 돌리지
   않았다. 이 노선에 대한 판정은 내리지 않은 것이다.
4. **LM 1종.** Qwen2.5-7B base 만 썼다. 기능어 편향이 모델 특이적일 가능성을 배제하지 못한다.
   (다만 이는 널리 보고된 성질이라 모델 특이적일 가능성은 낮다고 본다 — 검증하지는 않았다.)
5. **`attn-spacy` arm 은 K=3 절단율 0.36~0.61 로 DEGENERATE.** K 를 이 arm 에서만 올리면 G2 비교가
   불공정해져 그대로 뒀다. 결과적으로 "attention ∩ 언어필터" 조합은 **공정하게 시험되지 못했다.**
   이것이 이 실험의 가장 큰 미결 지점이다.
6. **`setdrift_w` 의 W=6 은 단일값.** TextTiling 발표값이라는 외부 출처는 있으나 이 데이터에
   적합한 값인지는 검증하지 않았다(의도적 — 진단이 통과할 때까지 W 를 찾지 않기 위해).
7. **OP-3 는 near-gold density.** dev-3 gold 밀도 0.0657 vs ANCHOR_R 0.0642. 앵커에서 동결했으므로
   선택-누출은 아니지만 oracle-count 에 인접하다. headline 은 OP-1 이다.
8. **G2 의 교란.** `attn-topk`(k=3) vs `all-core`(전체)는 attention 선택 **과 개수**가 동시에 다르다.
   교란 없는 시험은 G3(`shuffled-rank`, 같은 k·같은 pool)이며 그것도 전부 FAIL 이다.
9. **codex 위임 미이행.** `CLAUDE.md` 는 설계 옵션 선택을 `codex:rescue` 에 위임하라고 요구하는데
   이 세션에 해당 에이전트가 없어 대체했다. §1.4 의 기본값들은 codex 검토를 받지 않았다.
