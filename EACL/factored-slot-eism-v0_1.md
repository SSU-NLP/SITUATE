# Online, LLM-free Narrative Episode Segmentation via a Factored-Slot Latent Situation Model

**Working proposal — v0.4 (v0.1 동결 설계 + v0.2 granularity 정정 + v0.3 명명 정정을 본문에 통합)**

> **변경 이력:** v0.1 설계 동결 → v0.2 타깃 = 사건 경계(B)로 정정(BREAKING) → v0.3 명명 정정("IDM"→CosDrift) → **v0.4 정정사항을 본문(§1, §8, §9)에 직접 반영.** 정정 전 원문이 필요하면 git 이력 참조. `[v0.2 반영]` 표기가 붙은 문장이 수정된 부분이다.

> **⚠️ GRANULARITY 정정 (v0.2, BREAKING).** 타깃 경계는 **사건 경계(event boundary, B)** — 이야기 *안*의 사건 단위 — 이지, "완결된 썰 하나(A)"가 아니다. (인사이드아웃 기억구슬 = "하키 인생"이 아니라 "그날 골 넣은 순간".) 이 정정으로 **이론 정박이 오히려 강화된다**: EST·EISM·SEM은 전부 *사건* 분절 이론이며 B가 그 본래 대상이다. Motivating example(mouthguard, IDM 마진 0.04)도 원래 *한 괴담 안의* 장면 전환이었다 — 처음부터 B였다. 아래 본문에서 "episode/orb"는 **사건(event)** 으로 읽는다.
>
> **파생 변경:** (1) 이어붙이기(concatenation)로 만든 경계는 A이므로 **타깃 데이터가 아니다** → 병합 코퍼스 전략은 *dynamics 자기지도 사전학습*에만 유효하고, *경계 평가*에는 쓸 수 없다. 경계 평가는 **사람이 붙인 사건 경계**(Pieman 등)로 한다. (2) B에서는 주인공이 불변이므로 **protagonist 축은 "정체성"이 아니라 "장면의 활성 인물 집합"**으로 조작화한다. (3) B에서는 **time(시간 불연속)·causality(행위 사슬 단절)가 주력 축**일 것으로 예측한다(A에서는 protagonist가 지배했음) — 이 대조 자체가 H2의 검정 대상.
>
> **⚠️ 방법론 명명 (v0.3).** canonical 소스 = `method-registry.md`. 요지만:
> - 이 문서·모든 결과표의 baseline "IDM"은 **CosDrift(M1)**로 읽는다. IDM(Inverse Dynamics Model)은 **M2(MonoLAM) 내부 모듈**만 가리킨다.
> - 구 "Single latent (pure SEM) / K=1 / IDM" 항목은 **M1-CosDrift / M2-MonoLAM(완료·실패) / M4-MonoAnchor(미실행, 핵심 통제군)** 셋으로 분해된다. **M5−M4 = 분리(factoring)의 인과적 기여.**
> - **M2 실패는 폐기 자산이 아니라 근거다** (Locatello 2019 불가능 정리의 실증 → §6 anchor 필수성). 단 M2엔 5축 압력이 없었으므로 M4의 실패를 함의하지 않는다.
> - §4의 factored-slot = **M5-FactoredSlot**. 변종 M6(공유 dyn)/M7(slot-attn). FLAM(2026)·BRICKS-WM 근거, monolithic은 H1 검정 자체가 불가 — 상세는 registry §3.

> Scope note. This document freezes the design agreed so far (problem → theory → generative model → feasibility → baselines → data/eval strategy). Items still under active design are marked **[OPEN]**. The immediate next work item is precisifying the weak-supervision **anchor loss** (§6).

---

## 1. Problem & motivation

**Task.** `[v0.2 반영]` Segment a *streaming* transcript of spoken narrative (oral "tales" / anecdotes) into **events** — 이야기 *안*의 사건 단위(사건 경계 B). The *Inside Out* memory-orb unit: one orb ≈ **one event within a story** ("그날 골 넣은 순간"), *not* one complete story arc. Boundary density ≈ 수 문장당 1개.

**Constraints.**
- **Online / streaming**: decide boundaries as sentences arrive (bounded lookahead allowed).
- **LLM-free at inference**: no LLM call in the live segmentation loop. (LLMs *are* allowed at training/labeling time — see §8. This train/inference split is a core enabler.)

**Why existing methods fail.** Text segmentation is dominated by lexical/semantic **cohesion** signals (TextTiling; C99; supervised neural, Koshorek 2018; Cross-Segment BERT, Lukasik 2020; LLM-based LumberChunker). Episode boundaries are **not** semantic-distance events: *within* an episode every sentence brings large new information (e.g., knife → light → mouthguard), so embedding drift is high **inside** episodes too.

**Motivating evidence (the "IDM failure").** On a real sample, an embedding-drift online segmenter (IDM) scored within-episode sentences at s≈0.75–0.77 while the one true episode boundary scored s≈0.81 — a margin of ~0.04. Threshold tuning cannot separate "next scene of the same story" from "a brand-new story" because the *quantity being measured* (scalar semantic change) is only weakly related to episode boundaries. This is a **principled** mismatch, not a tuning bug.

---

## 🎯 연구 질문과 기여의 위계 (CANONICAL — `method-registry.md`와 동기)

> **이 프로젝트는 "가장 정확한 분절기"를 만드는 것이 목적이 아니다.** 그러나 정확도가 무의미하다는 뜻도 아니다 — **주목표 2개 + 서브목표 1개**의 위계가 있다.

### RQ1 (주) — EISM의 계산적 검증
> **"인간이 사건을 나누는 방식(Event-Indexing Model의 5지표 좌표화 리셋)을 실제로 구현했을 때 인간의 사건 경계를 설명하는가? 그리고 언제 실패하는가?"**

- **GPT-4 프롬프팅으로는 원리적으로 답할 수 없는 질문이다** (블랙박스 → "어느 지표가 리셋됐는가"를 읽을 수 없음).
- **이 프레임에서 기존 "실패"가 전부 발견으로 전환된다:** Monkey/대화 패배 = **이론의 경계조건 발견**(상황모델이 정지한 텍스트에서 EISM 불성립). place가 판별자 = **"무대 리셋"이 작동 핵심 조건**. time 노이즈(시제 뒤집힘 103/119) = **"시제 변화 ≠ 시간지표 변화"라는 조작화 발견**. M2 실패 = **5지표를 강제하지 않으면 대규모 학습으로도 창발하지 않음**(Locatello 2019 실증).
- **⇒ M4(monolithic+anchor) vs M5(factored+anchor) 애블레이션이 이 논문의 심장이다.** EISM의 구조적 주장이 "지표들이 **분리되어** 추적된다"이므로, monolithic이 동등하면 EISM의 계산적 근거가 약해진다.
- **⇒ B1(Kumar) 대결 = EST(스칼라 서프라이즈) vs EISM(다축 리셋)의 계산적 대결.** 성능 경쟁이 아니라 **두 인지 이론의 비교**이며, **져도 과학적 결론**이 된다.

### RQ2 (주) — 실시간 배포 가능성
> **"그 메커니즘이 온라인·LLM-free로 실시간 구현 가능한가?"**

- B2/B3(LLM 직접 프롬프팅)는 **전체 텍스트가 필요 → 스트리밍 불가**, 매 추론마다 API. **애초에 인간 인지의 모델이 아니다** (사람은 실시간 분절함).
- **RQ1과 RQ2는 서로를 강화한다:** 인간이 실시간으로 분절하므로 **실시간 구현 자체가 인지적 타당성의 일부**다.
- > **한 줄 주장: "인간처럼 5지표를 추적하고, 인간처럼 실시간으로 분절한다."**

### S1 (서브) — 경쟁력 있는 정확도 — **되도록 추구, 시간 부족 시 후순위**
- **비목표가 아니다.** 정확도가 높을수록 RQ1·RQ2의 설득력이 올라간다. **시간이 허락하면 반드시 추구한다.**
- **M1-CosDrift는 반드시 이긴다** (dev 6/7 달성).
- **B1-BayesSurprise는 이겨야 한다** — 정확도 때문이 아니라 **H1(스칼라 vs 다축)의 직접 검정**이므로. 우선순위 높음.
- **B2/B3(GPT-4 직접)는 못 이겨도 논문이 성립한다** — 오프라인 **topline/oracle**로 보고하고, **"LLM 없이 그 상한에 얼마나 근접하는가"**를 보인다. 이기면 보너스.

---

## §7-확장 — 베이스라인 전체 (M 계열 + B 계열)

**전체 표·상세는 `method-registry.md`(canonical)를 본다.** 이 문서에서 필요한 핵심만:

- **M 계열 = 우리 방법의 계보** (M1 CosDrift → M2 MonoLAM(실패) → M3 S5-Extractor(조건부 Go, anchor 공급원) → M4~M7 예정) / **B 계열 = 외부 선행연구** (B1 BayesSurprise·B2 LLM-Direct·B3 EventRecall).
- **핵심 대조:** `M5 vs M4` = **표현 분리의 인과적 기여**(RQ1의 심장) · `M4 vs M2` = anchor의 순수 효과 · `M5 vs B1` = **EISM vs EST**.
- **⚠️ 순환 방지:** B2/B3로 gold를 생성한 데이터(Tier 2)에서는 B2/B3를 베이스라인으로 쓸 수 없다. B1·M1·M2만 비교.

## 2. Central thesis & hypotheses

**Thesis.** Narrative episode boundaries are marked by a **coordinated, simultaneous reset across the dimensions of the situation model** (protagonist / place / time / causality / goal) — *not* by the magnitude of semantic change. What separates within-episode progression from a boundary is **prediction failure under learned per-dimension dynamics**, jointly across dimensions.

- **H1 (core).** The number of simultaneously-resetting situation-model indices predicts human boundary strength better than scalar semantic drift / scalar surprise.
- **H2.** "Stage-reset" indices (protagonist + place + time) dominate causal/semantic indices for episode boundaries.
- **H3.** Scalar surprise (EST-style) responds to *within-episode* sub-boundaries but is inferior for *episode* boundaries; the multivariate coordinated-reset signal is what distinguishes them.

## 3. Theoretical grounding

- **Event Segmentation Theory (EST; Zacks et al. 2007).** Humans segment continuous experience into events; boundaries occur at transient spikes of **prediction error** under the current event model. IDM is a *scalar approximation* of this surprise — and its scalar nature is exactly why it conflates within-episode churn with boundaries.
- **Event-Indexing Model (EISM; Zwaan, Langston & Graesser 1995; Zwaan & Radvansky 1998).** Comprehenders monitor **five indices — time, space, protagonist, causality, intentionality** — and update the situation model when any index changes. This predicts IDM's failure: within an episode only the *causal* index churns while protagonist/space/time stay fixed; a scalar absorbs the causal churn. The boundary signal is *multi-index simultaneous reset*.
- **SEM / latent-cause inference (Franklin, Norman, Ranganath, Zacks & Gershman 2020; Gershman & Niv).** A probabilistic generative model of event **dynamics** over structured scenes, embedded in vector space, with a **nonparametric (sticky-CRP / DP) prior** over events; boundaries = a switch to a new latent cause when the current schema stops predicting. Online inference via forward local-MAP (SEM notes it cannot re-evaluate boundaries retrospectively; **particle smoothing** can — we use this). This is the computational instantiation of "run a dynamical simulation and segment by prediction failure."

**Orb ≙ latent cause.** One memory orb = one latent cause = one event schema. The name *Structured Event **Memory*** aligns with the orb intuition directly.

## 4. Method: factored-slot latent situation model

### 4.1 Typed slots (breaking slot exchangeability)
Vanilla Slot Attention (Locatello et al. 2020) produces **exchangeable** slots that bind competitively to *any* content — so "slot = protagonist" does not emerge by default. We break the symmetry with **typed, keyed slots**, following **Recurrent Entity Networks (EntNet; Henaff, Weston, Szlam, Bordes & LeCun 2017)**: a bank of gated memory cells, each tracking a concept/entity, updated on-the-fly in a **single pass**, with keys that can be **seeded to force tracking of a specified set** (interpretable). We generalize EntNet's slot type from *entity* to the **five EISM dimensions**.

### 4.2 Per-slot heterogeneous predictive distributions
EISM's premise is that the dimensions are *qualitatively different*, so forcing one distribution family across all slots is wrong. Predictive families by slot:
- **Protagonist — point process (not Gaussian).** Target = the active-entity set; predictive = per-tracked-entity Bernoulli(persist) + CRP-style novel-entity introduction rate λ. Surprise = −log P(observed persistence/exit pattern + #new). "Cast wiped + new cast" → high surprise automatically.
- **Place / Time — small MoG (or categorical).** Low-variance; 2–3 components suffice.
- **Causality / Goal — larger MoG or VQ-categorical.** Must be **multimodal** because within-episode causal content jumps discretely. Schema embedding ψₑ parameterizes mixture weights/means ("the causal-jump modes typical of *this* genre schema").

### 4.3 Per-slot dynamics conditioned on episode schema
Within episode e: sₜ^(k) = f_θ^(k)(s_{t-1}^(k); ψₑ) + noise. A well-trained f^(k) predicts within-episode progression (low error); at a boundary ψ changes and dynamics fail to transfer (high error). **This is the structural difference from IDM**: we measure *prediction failure under learned dynamics*, not raw |Δx|.

### 4.4 Sticky-CRP switching prior
P(eₜ | e_{t-1}) ∝ κ·1[eₜ=e_{t-1}] + CRP(reuse past cause **or** create new). Nonparametric ⇒ number of episodes not fixed a priori. **The earlier "minimum-segment-length / cooldown" heuristic is absorbed into the persistence bias κ as a principled prior**, not an ad-hoc gate.

### 4.5 Boundary = coordinated switch (likelihood-ratio product)
The switch posterior:

```
P(switch)/P(cont) ∝ [P(new)/κ] · Π_k  p(target_t^(k) | new schema) / p(target_t^(k) | current schema)
```

Each factor is a **per-slot likelihood ratio**. Because it's a *ratio*, dimensional/scale mismatches across heterogeneous families largely cancel — heterogeneous slot families are therefore compatible. Boundary = several slots' ratios spiking together, multiplied by the CRP prior. **No ad-hoc per-slot weights**; normalization is emergent.

### 4.6 Prediction-space design (make-or-break: avoiding IDM re-import)
Two axes:
- **Axis 1 — predict-then-compare** (vs compare-to-previous): compare the observation to what dynamics *predicted*, not to x_{t-1}. Smooth narrative progression → high |Δx| but low prediction error.
- **Axis 2 — slot-native target spaces T_k**: each slot predicts/compares only in its own dimension's space, so causal churn is *quarantined* to the causal slot and does not leak into protagonist/place/time slots.

Realization of T_k (a fork; all three are knobs of one model):
- **D1** — extractor-defined space (run cheap NER/coref/time-tagger at inference): safest against leakage, but "discovery" lives only in the dynamics.
- **D2** — learned projection π_k (no extractor at inference): most "discovery", highest leakage risk.
- **D3 [v1]** — **anchored latent**: slot predicts in a learned subspace, with an auxiliary head anchored to extractor signals during training. Natural fit with **token-level input + typed slot-attention**: each typed slot attends to its dimension's tokens (protagonist slot → entity mentions, etc.), guided by the anchor loss. = Slot-Attention (token attend) + EntNet (typed persistent slots) + anchoring, fused.

**Auto-normalization via likelihood (not point error).** Predict a *distribution* per slot; surprise = −log p. Broad-variance slots (causal) are unsurprised by ordinary internal jumps; usually-narrow slots (place) are very surprised when they move. This normalizes each slot by its own normal variability — the reason we insist on a probabilistic (not raw-error-sum) formulation.

### 4.7 New-schema predictive: amortized onset prior + schema reuse
The switch **numerator** ("if we switched, how well would a *fresh* schema explain this?") is the expensive term — in principle an integral over the fresh-schema prior. A naive uninformative numerator breaks slot normalization. Fix: a new episode is not "anything" — there is a learnable **episode-onset distribution** (new protagonist introduced, new stage set, time reset).
- **A1 [v1] — amortized onset prior q_onset^(k)**: learned from the *first sentences* of every segment in the concatenation corpus (free, self-supervised). Collapses the integral to one forward pass. Dovetails with the protagonist point process: its novel-entity rate λ *is* the protagonist onset prior.
- **A3 [v1] — schema reuse term**: CRP also allows *returning* to a past schema (digression → return should not spawn a new orb). Numerator = p(brand-new)·q_onset + Σ_{e'∈past} p(reuse e')·p(·|ψ_{e'}); reuse terms reuse already-computed ψ_{e'} (cheap). = "the same-colored orb lighting up again."
- **A2** — few-particle Monte-Carlo of the fresh-schema prior: heavier; ablation/upper-reference only.

### 4.8 Online inference: particle filtering + delayed commit
Per sentence: for each particle (S_{t-1}, e_{t-1}) evaluate {continue / reuse / new} ratios (all forward passes, no integral) → resample eₜ → update Sₜ. Boundary posterior = weighted fraction of particles with eₜ≠e_{t-1}. **Delayed commit = K′-sentence particle smoothing** (retrospective re-evaluation; the capability SEM's forward local-MAP lacks). Cost = P particles × K slots × per sentence, constant-time per step thanks to onset amortization; P and K′ are latency–accuracy knobs, independent of H1–H3.

## 5. Feasibility grounding (is the factored-slot idea real?)
- Slot Attention slots are **exchangeable** → typing requires breaking symmetry (established: EntNet seeded keys; SAVi/flow-conditioned slots; COSA grounded slots).
- **Disentanglement impossibility (Locatello et al. 2019).** Purely unsupervised disentanglement is theoretically impossible without inductive bias — so slots will **not** self-align to protagonist/place/time unsupervised. Anchors cannot be skipped.
- **Escape routes (proven).** Weak supervision from **pairs differing in a subset of factors** makes factors identifiable without knowing which/how many changed (Locatello et al. 2020, "Without Compromises"); **few labels** suffice (Disentangling with Few Labels 2019); **temporal structure** grants further identifiability (nonlinear-ICA lineage) — narrative is inherently temporal, so we get this bias for free.
- **Net.** The naive unsupervised version is impossible; the **typed-slot + weak-anchor + temporal** version sits inside the theoretically identifiable regime. Novelty = aligning typed slots to the *five EISM dimensions* + per-slot dynamics + coordinated-switch **for online LLM-free episode segmentation** (EntNet tracked only entities for QA; Slot Attention is vision; SEM/latent-cause is vision/perception — the bridge is empty).

## 6. Weak supervision & anchor strategy  **[OPEN — next work item, §see roadmap]**
Cheap silver signals (coref/NER, spatial/temporal expressions, LLM silver) move from *inputs* to **anchors** that bind slots to dimensions. **Mixed anchoring**: protagonist/place/time strongly anchored; causality/goal weakly anchored (LLM silver) or latent-only in v1. Exact target definitions, loss form (absolute decoding vs pairwise change-supervision), masking/noise-robustness, and bind-then-release curriculum are being precisified next.

## 7. Baselines as knobs (one codebase, controlled ablations)
| Baseline | Structure | Controls for |
|---|---|---|
| Single latent (pure SEM) | K=1 | Does factoring help at all? |
| Single entity slot (EntNet-orig) | protagonist only | Is one dimension enough? |
| K exchangeable slots (vanilla SA) | K slots, no anchor (λ_anchor=0) | Is the gain from **typing** or just capacity? |
| **Typed slots + anchor (ours)** | anchored to EISM dims | — |
| IDM | no dynamics; boundary = ‖Δx‖ | Do **learned dynamics** beat raw drift? |
| Homogeneous-Gaussian ours | all slots Gaussian | Is the gain from distribution-family heterogeneity? |

Knobs: K=1, λ_anchor=0, mask non-protagonist slots, remove f^(k), force Gaussian. Ablation = hypothesis test.

## 8. Data strategy (minimal hand-labeling)
- **Train-time LLMs are allowed; inference is LLM-free** (distillation / "narration-supervision" paradigm: supervise at train time, base inputs at test time).
- **(a)** `[v0.2 반영 — 역할 강등]` **Concatenation corpus = dynamics 자기지도(next-obs) 사전학습 전용.** 이어붙이기 경계는 A(썰 경계)라 **타깃(B)의 gold가 될 수 없다** — 원안의 "boundaries are free gold (Choi trick)"는 폐기. 독립 저작 서사(Gutenberg 등)를 **이어붙이지 않고 원문 그대로** 대량 수집해도 dynamics 학습 목적은 동일하게 달성된다. **경계 gold는 (c′) 참조.**
- **(b) LLM silver for index labels.** LLM annotates *which index reset* at each boundary (needed for H1–H3), with multi-sample majority vote + confidence reweighting (LLMaAA); prompt validated on a small subsample (Pangakis workflow).
- **(c)** `[v0.2 반영]` **경계 gold의 실제 공급원 (c′):** ① dev = 사람 주석 사건 경계 3편(Pieman/Monkey/Tunnel, Michelmann) — 이미 소진(ledger D-M5), ② **메인 held-out = Tier 2 LLM-silver**(`tier2-heldout-spec.md`: GPT-4 × B2 프롬프트 × 20회 합의, doc 10+ → bootstrap CI), ③ **최종 방어선 = Tier 3 한국어 썰 사람 주석**(수백 경계; continuous-monologue = 원안 (c)의 역할이 여기로 이동).
- **Scale targets.** `[v0.2 주석: 아래 규모는 dynamics 사전학습·anchor 라벨 기준으로 읽는다 — concat 경계는 gold 아님]` Student training: ~10k–50k concatenated docs → hundreds of thousands of boundary decisions (Wikisection-scale is the pragmatic sweet spot; Wiki-727K is oversized). Index-label silver: 1k–5k boundaries. Human gold: 300–500 double-annotated. **Note:** onset-prior training uses only segment *first sentences* → effective sample ≈ total / mean-segment-length; recompute the concatenation-scale lower bound accordingly.

## 9. Evaluation
- **Metrics.** P_k, WindowDiff (near-miss tolerant), segmentation similarity B; plus an **online latency–accuracy curve** (lookahead K′ / particles P vs accuracy).
- **Track A (comparability).** Standard benchmarks: Choi, Wiki-727K, Wikisection en_city/en_disease.
- **Track B (the actual claim).** `[v0.2 반영]` **사건 경계 데이터**: dev 3편(Michelmann) + Tier 2 held-out(1회 평가) + Tier 3 한국어 썰. (원안의 concatenation 코퍼스 평가는 A 경계라 제외.)
- **Cross-lingual generalization.** Korean oral tales via the team's own faster-whisper large-v3 / Qwen3-ASR pipeline; tests whether "stage reset" is language-invariant (raises the paper's strength).
- **English is fine / preferred** for the main experiments (data + comparability); Korean is the generalization section.

## 10. Risks & threats to validity
- Concatenation boundaries are cleaner than real flow → the final test **must** be continuous monologue; report the gap. (Also biases the **onset prior** — calibrate with a little real gold.)
- LLM silver noise → majority vote + reweighting + human validation; report agreement.
- Coref/time-tagger quality caps protagonist/place/time slots → error-propagation analysis mandatory.
- Prediction-space leakage (semantic-space prediction re-importing IDM) → mitigated by slot-native spaces / VQ discretization; **must be empirically shown** that dynamics-surprise ≠ raw drift.
- Slot/codebook collapse → commitment loss, component/entropy regularization; or EntNet-style fixed keyed init.
- Causal/goal indices lack clean taggers → v1 anchors protagonist/place/time strongly, causal/goal weak/latent.
- Schema-reuse candidate blow-up → active window / age limit.
- Particle degeneracy near switches → resampling + jitter.
- Identifiability holds only **up to anchor quality**; anchors cannot be skipped (impossibility theorem).

## 11. Roadmap / open decisions
1. **[NOW] Anchor loss (§6)** — target per slot; absolute-decoding vs pairwise change-supervision (the identifiability-granting form); masking + noise-robustness; bind-then-release curriculum; mixed strengths for causal/goal.
2. Learning recipe + concatenation-corpus construction (onset effective-sample lower bound; leakage/cheating controls).
3. Predictive-distribution specifics: MoG vs VQ-categorical; T_k geometry (entity-set vectorization & distance).
4. Full training objective assembly + collapse regularizers.
5. Precise related-work survey to finalize novelty statements.

## Key references (grounded)
- Hearst 1994 (TextTiling); Choi 2000; Koshorek et al. 2018 (Wiki-727K, supervised segmentation); Lukasik et al. 2020 (Cross-Segment BERT); Arnold et al. 2019 (Wikisection); LumberChunker (Duarte et al. 2024); Kargupta et al. 2025 (EpiMine, ACL).
- Zacks et al. 2007 (EST); Zwaan, Langston & Graesser 1995 / Zwaan & Radvansky 1998 (Event-Indexing Model); Franklin, Norman, Ranganath, Zacks & Gershman 2020 (SEM); Gershman & Niv 2010 (latent structure).
- Locatello et al. 2020 (Slot Attention); Locatello et al. 2019 (disentanglement impossibility, ICML); Locatello et al. 2020 (weakly-supervised disentanglement "Without Compromises"); Disentangling with Few Labels 2019; Henaff, Weston, Szlam, Bordes & LeCun 2017 (Recurrent Entity Networks); Kipf et al. 2021 (SAVi, conditional slots).
- Adams & MacKay 2007 (Bayesian Online Changepoint Detection); Mintz et al. 2009 (distant supervision); Pangakis et al. 2024 (LLM-as-annotator validation).

*(Citations to be verified against primary sources in roadmap item 5.)*
