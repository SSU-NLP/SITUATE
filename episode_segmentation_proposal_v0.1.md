# Online, LLM-free Narrative Episode Segmentation via a Factored-Slot Latent Situation Model

**Working proposal — v0.1 (frozen design state)**

> Scope note. This document freezes the design agreed so far (problem → theory → generative model → feasibility → baselines → data/eval strategy). Items still under active design are marked **[OPEN]**. The immediate next work item is precisifying the weak-supervision **anchor loss** (§6).

---

## 1. Problem & motivation

**Task.** Segment a *streaming* transcript of spoken narrative (oral "tales" / anecdotes) into **episodes** at the granularity of "one complete story" — the *Inside Out* memory-orb unit. One orb ≈ one self-contained narrative arc.

**Constraints.**
- **Online / streaming**: decide boundaries as sentences arrive (bounded lookahead allowed).
- **LLM-free at inference**: no LLM call in the live segmentation loop. (LLMs *are* allowed at training/labeling time — see §8. This train/inference split is a core enabler.)

**Why existing methods fail.** Text segmentation is dominated by lexical/semantic **cohesion** signals (TextTiling; C99; supervised neural, Koshorek 2018; Cross-Segment BERT, Lukasik 2020; LLM-based LumberChunker). Episode boundaries are **not** semantic-distance events: *within* an episode every sentence brings large new information (e.g., knife → light → mouthguard), so embedding drift is high **inside** episodes too.

**Motivating evidence (the "IDM failure").** On a real sample, an embedding-drift online segmenter (IDM) scored within-episode sentences at s≈0.75–0.77 while the one true episode boundary scored s≈0.81 — a margin of ~0.04. Threshold tuning cannot separate "next scene of the same story" from "a brand-new story" because the *quantity being measured* (scalar semantic change) is only weakly related to episode boundaries. This is a **principled** mismatch, not a tuning bug.

## 2. Central thesis & hypotheses

**Thesis.** Narrative episode boundaries are marked by a **coordinated, simultaneous reset across the dimensions of the situation model** (protagonist / place / time / causality / goal) — *not* by the magnitude of semantic change. What separates within-episode progression from a boundary is **prediction failure under learned per-dimension dynamics**, jointly across dimensions.

- **H1 (core).** The number of simultaneously-resetting situation-model indices predicts human boundary strength better than scalar semantic drift / scalar surprise.
- **H2.** "Stage-reset" indices (protagonist + place + time) dominate causal/semantic indices for episode boundaries.
- **H3.** Scalar surprise (EST-style) responds to *within-episode* sub-boundaries but is inferior for *episode* boundaries; the multivariate coordinated-reset signal is what distinguishes them.

## 3. Theoretical grounding

- **Event Segmentation Theory (EST; Zacks et al. 2007).** Humans segment continuous experience into events; boundaries occur at transient spikes of **prediction error** under the current event model. IDM is a *scalar approximation* of this surprise — and its scalar nature is exactly why it conflates within-episode churn with boundaries.
- **Event-Indexing Model (EISM; Zwaan, Langston & Graesser 1995; Zwaan & Radvansky 1998).** Comprehenders monitor **five indices — time, space, protagonist, causality, goal** — and update the situation model when any index changes. This predicts IDM's failure: within an episode only the *causal* index churns while protagonist/place/time stay fixed; a scalar absorbs the causal churn. The boundary signal is *multi-index simultaneous reset*.
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
- **(a) Distant supervision via genre-matched concatenation [primary].** Concatenate independently-authored short narratives (Reddit story subs; Gutenberg chapters, cf. Chapter Captor; anecdote/joke compilations; act-marked podcasts) → **boundaries are free gold** (the Choi trick). **Genre/speaker-matched** concatenation prevents style-cue cheating so the only reliable signal is EISM reset.
- **(b) LLM silver for index labels.** LLM annotates *which index reset* at each boundary (needed for H1–H3), with multi-sample majority vote + confidence reweighting (LLMaAA); prompt validated on a small subsample (Pangakis workflow).
- **(c) Small human gold (hundreds).** Only for validating silver + the final **continuous-monologue** held-out test set (real flow, the true difficulty).
- **Scale targets.** Student training: ~10k–50k concatenated docs → hundreds of thousands of boundary decisions (Wikisection-scale is the pragmatic sweet spot; Wiki-727K is oversized). Index-label silver: 1k–5k boundaries. Human gold: 300–500 double-annotated. **Note:** onset-prior training uses only segment *first sentences* → effective sample ≈ total / mean-segment-length; recompute the concatenation-scale lower bound accordingly.

## 9. Evaluation
- **Metrics.** P_k, WindowDiff (near-miss tolerant), segmentation similarity B; plus an **online latency–accuracy curve** (lookahead K′ / particles P vs accuracy).
- **Track A (comparability).** Standard benchmarks: Choi, Wiki-727K, Wikisection en_city/en_disease.
- **Track B (the actual claim).** Narrative concatenation corpus + continuous-monologue gold.
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
