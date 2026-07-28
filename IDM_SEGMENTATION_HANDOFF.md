# IDM 기반 대화 화제분절 — 전체 시도·데이터·수치 기록 (핸드오프)

> **성격**: 측정 사실·실험 설정·수치만 기록. 해석·판단·후속방향은 기재하지 않음.
> **작성일**: 2026-07-10 · **실험 디렉토리**: `outputs/experiments/2026-07-09_idm_lam_gonogo/` · **코드**: `scripts/idm_lam/`
> (개정 설계문서는 `IDM_SEGMENTATION_RECORD.md` 로 분리됨; 본 파일은 지금까지 실행된 결과의 사실 기록.)

---

## 0. 배경 / 판가름 기준 (설계문서 `gonogo_expreiment.md` 정의)

영상용 LAM(IDM+FDM+VQ병목)을 대화에 이식: 전이(z_t→z_{t+1})를 이산 코드로 유형화, 코드가 바뀌는 지점을 경계로 사용. 사전 정의 관문:
- **관문 1**: 코드가 "다음 문장 내용"이 아니라 "전이 성격"으로 갈리는가. 실패기준 = 코드가 주제내용으로 갈림 / 누출율 높음 / 코드북 붕괴.
- **관문 2**: 코드 기반 경계가 surprise·밀도랜덤보다 인간 경계에 가까운가(밀도통제 후). 실패기준 = A≈B 또는 A≈D.

---

## 1. 데이터 — 전부 명시

### 1.1 인코더 (모든 실험 공통)
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, **768D, L2정규화**, 다국어. (HANDOFF_06 δ_eff와 동일 인코더 → 관문2 B' 공정비교 성립.)

### 1.2 학습 데이터
**(A) 최초(오염) — superseg_train** (`benchmarks/Def-DTS/data/DTS_session_datasets/superseg_train.jsonl`): **6,948 대화 / 85,203 turn / 78,255 인접쌍** (경계라벨 22,202, 학습 미사용). 평가셋 superseg_test와 동일 출처/분포.

**(B) 수정 설계 — 대규모 무라벨 코퍼스** (`build_corpus.py`): **21,494,673 발화 / 2,729,366 대화 / 18,765,307 인접쌍 / 11 샤드(fp16)** (최초 78k의 240×). scratch `idm_corpus/` 저장, `--resume` 지원. 소스별 대화수(HF):

| 소스 | HF 이름 | 대화 수 | 파싱 |
|---|---|---|---|
| SODA | `allenai/soda`(train+val+test) | 1,486,336 | `dialogue`(발화 리스트) |
| UltraChat | `stingning/ultrachat`(80만 cap) | 800,000 | `data`(교대 발화) |
| UltraChat-200k | `HuggingFaceH4/ultrachat_200k`(train_sft+test_sft) | 230,975 | `messages[].content` |
| HH-RLHF | `Anthropic/hh-rlhf` | 160,785 | `chosen` `\n\n(Human\|Assistant):` 분리 |
| Empathetic | `Estwld/empathetic_dialogues_llm` | 19,531 | `conversations[].content` |
| PersonaChat | `AlekseyKorshuk/persona-chat` | 17,878 | `utterances[-1].history` |
| Synthetic-Persona | `google/Synthetic-Persona-Chat` | 8,938 | `Best Generated Conversation` 줄분리 |
| Blended-Skill-Talk | `blended_skill_talk` | 4,819 | `guided`+`free` 교대 |
| AMI-train | AMI 수동주석(§1.3) | 104 미팅 | speaker-turn 병합 |

### 1.3 평가 데이터 — 전부 held-out (수정 설계에서 IDM 미학습)
**DTS Data Hub** (`benchmarks/Def-DTS/data/DTS_session_datasets/`): `{"id","dialogue"}`, `[NEWLINE]` 구분, `[BOUNDARY]` 마커. 파싱: `user:`/`agent:` 접두 제거, 경계=[BOUNDARY] 직후 turn.

| 셋 | 파일 | 대화 | turn | 경계 | 성격 |
|---|---|---|---|---|---|
| **TIAGE** | `tiage_test.jsonl` | 100 | 1,564 | 315 | **인간** topic-shift, 잡담 ★주판정 |
| DialSeg711 | `dialseg711_test.jsonl` | 711 | 18,639 | 2,743 | splice(구조) |
| SuperDialseg | `superseg_test.jsonl` | 1,322 | 16,006 | 4,017 | 문서섹션(구조) |
| (보정) | `tiage_train.jsonl`,`superseg_train.jsonl` | — | — | — | A_typed/A_bigram 보정 전용 |
| (관문1 진단) | `superseg_validation.jsonl` | 1,322 | 16,412 | 4,263 | 최초설계 진단 |

**AMI**: `ami_public_manual_1.6.2.zip`(groups.inf.ed.ac.uk, 22MB) → `ami_topic_prep.py --merge` → `data/ami/topic/*.json`(`{turns:[{start,speaker,text}],bnd_top,topic_levels}`). **139 미팅, 49,074 turn, 903 top-topic 경계**. 분할(`data/ami/split.json`, index%4): **train 104(37,846 turn,670 경계)→코퍼스 / eval 35(11,228 turn,233 경계)→held-out**. gold=start-turn 규약.

**ICML 다이어리**: `icml_sentences.txt` 89 한국어 문장, 1문장=1turn, gold 없음(정성).

### 1.4 채점기
- **DTS 공식**(`src/hi_ontop/dts_scoring.py`): 대화별 평균, F1 binary(exact), Pk/WD nltk, **Score=0.5F1+0.25(1−Pk)+0.25(1−WD)**, 끝-turn 규약, 마지막 turn=0.
- **AMI 공식**(`src/hi_ontop/ami_scoring.py`): F1 **±2 tolerant**, start-turn(shift 0), 동일 Score식. (ORACLE 배치 시 F1=1.000 확인.)

### 1.5 아키텍처 (LAM=IDM+VQ+FDM, `lam.py`)
- IDM: `concat(z_t, z_{t+1}−z_t)` → MLP[1536→512→256→code_dim] → a_t.
- **VQ 병목**: `vector-quantize-pytorch`, K∈{8,16,32}, code_dim=32, commitment=0.25, EMA decay=0.99, kmeans_init, threshold_ema_dead_code=2. **정보용량=log2(K)비트**.
- FDM: `concat(z_t, code_vec)`(prepend) → MLP[800→512→768] → ẑ_{t+1}. 손실=MSE(dim합)+commit.
- **정보병목 기법 범위(사실)**: VQ만 사용. 표준 대안 **노이즈 주입(VIB)·sparsity 규제는 미시도**. **code_dim=32 고정**(K만 스윕, code_dim 축소 미시도).
- 학습: 최초 Adam lr3e-4 batch512 60ep seed0 / 코퍼스 lr3e-4 batch8192 10ep seed0.

---

## 2. 평가 프로토콜
- **density-matched**: 대화별 gold 경계수 N만큼 top-N(=oracle 개수). 위치 정확도만 비교.
- **deployable**: 개수까지 자가결정(적응임계), density-match·튜닝 없음.

---

## 3. 관문 1 결과 (누출)

### 3.1 최초(오염), superseg_val 15,090 전이 (`gate1_diagnostics.py`)
| K | 코드사용 | ppl | 코드기여 | NMI(code;내용) | NMI(code;경계) | 조건부(code;경계\|topic) | 조건부(δ;경계\|topic) | R²(code→내용) |
|---|---|---|---|---|---|---|---|---|
| 8 | 8/8 | 4.46 | 0.098 | 0.225 | 0.017 | 0.052 | 0.110 | 0.063 |
| 16 | 16/16 | 8.10 | 0.126 | 0.260 | 0.041 | 0.084 | 0.141 | 0.092 |
| 32 | 32/32 | 15.5 | 0.160 | 0.258 | 0.051 | 0.109 | 0.204 | 0.118 |
공정대조: 전 K에서 NMI(code;내용) 0.22~0.26 ≫ NMI(δ-bin;내용) 0.05~0.06. 정성: 코드가 주제도메인으로 갈림, 66% catch-all 코드 존재.

### 3.2 수정 설계, corpus_f1.0(18.57M쌍), held-out (`gate1_heldout.py`)
| set | NMI(code;내용) | NMI(δ;내용) | 조건부(code;경계) | 조건부(δ;경계) | R²(code→내용) |
|---|---|---|---|---|---|
| tiage | 0.076 | 0.030 | 0.084 | 0.139 | 0.041 |
| superseg | 0.292 | 0.049 | 0.084 | 0.141 | 0.081 |
| dialseg711 | 0.265 | 0.040 | 0.077 | 0.415 | 0.038 |
| ami | 0.319 | 0.029 | 0.106 | 0.225 | 0.131 |

**관문1 실패기준 충족** (코드가 내용으로 갈림/누출).

### 3.3 코드 분할 vs 실제 gold 화제 (`code_vs_gold.py`, corpus_f1.0)
| held-out | NMI(코드;gold화제) | NMI(코드섞음;gold) | NMI(임베딩 KMeans;gold) | homogeneity(코드) |
|---|---|---|---|---|
| tiage | 0.311 | 0.329 | 0.484 | 0.349 |
| superseg | 0.296 | 0.320 | 0.474 | 0.320 |
| dialseg711 | 0.200 | 0.209 | 0.522 | 0.153 |
코드 분할 ≈ 코드섞음(우연 수준). 원본 임베딩 KMeans는 gold화제 NMI 0.47~0.52.

### 3.4 코드가 갈리는 축 (`code_organizes_by.py`, AMI 우연보정, corpus_f1.0)
| 축 | superseg AMI | tiage AMI | (코드섞음) |
|---|---|---|---|
| 내용(topic-6 KMeans) | 0.202 | 0.055 | ~0.000 |
| 내용(topic-30) | 0.165 | 0.052 | 0.000 |
| 말투:길이 | 0.135 | 0.006 | 0.000 |
| 말투:화자 | 0.043 | 0.001 | 0.000 |
| 말투:질문여부(?) | 0.031 | 0.032 | 0.000 |

---

## 4. 학습설정 감사 (`train_audit.py`, 1.14M쌍)
| 설정 | val recon | 코드기여 | ppl | NMI(code;내용) | NMI(δ;내용) | TIAGE A_bigram F1 | TIAGE Bp F1 |
|---|---|---|---|---|---|---|---|
| FDM-only(코드X) | 0.623 | — | — | — | — | — | — |
| current(lr3e-4,10ep) | 0.597 | 0.031 | 4.4 | 0.183 | 0.023 | 0.256 | 0.370 |
| better(lr1e-4,cos,25ep) | 0.582 | 0.049 | 11.2 | 0.217 | 0.023 | 0.236 | 0.370 |
| cosloss(lr1e-4,cos,25ep) | 0.688 | 0.032 | 6.9 | 0.220 | 0.023 | 0.211 | 0.370 |

---

## 5. 관문 2 결과 (분절 성능)

### 5.1 최초(오염), K16, density-matched, DTS Score (`gate2_segmentation.py`)
| set | A_typed | A_bigram | Bp | C_lex | D_random | E_unif |
|---|---|---|---|---|---|---|
| tiage | 0.413 | 0.420 | **0.492** | 0.373 | 0.385 | 0.426 |
| dialseg711 | 0.432 | 0.364 | **0.576** | 0.480 | 0.334 | 0.372 |
| superseg | **0.494** | 0.472 | 0.426 | 0.435 | 0.400 | 0.441 |
유의성: tiage A_typed vs Bp dF1 −0.121 p=6.9e-4(K16)/−0.233 p=1e-9(K8)/−0.120 p=9e-5(K32); vs random p=0.55.

### 5.2 규모 ablation, held-out TIAGE. Bp F1=0.370, random F1=0.229
| 학습쌍 | val recon | A_typed F1 | Score | vs Bp(ΔF1,p) | vs random(ΔF1,p) |
|---|---|---|---|---|---|
| 1.78M | 0.5422 | 0.333 | 0.471 | −0.038, .16 | +0.103, .005 |
| 5.43M | 0.5343 | 0.314 | 0.453 | −0.056, .12 | +0.085, .015 |
| 18.57M | 0.5146 | 0.231 | 0.404 | −0.139, 6e-5 | +0.002, .96 |

### 5.3 corpus_f1.0, held-out, density-matched (`gate2_heldout.py`)
| set | A_typed | A_bigram | Bp | C_lex | D_random | E_unif |
|---|---|---|---|---|---|---|
| tiage | 0.404 | 0.399 | **0.492** | 0.373 | 0.414 | 0.426 |
| dialseg711 | 0.337 | 0.289 | **0.576** | 0.480 | 0.325 | 0.372 |
| superseg | 0.357 | 0.322 | **0.426** | 0.435 | 0.378 | 0.441 |
| ami | 0.360 | 0.334 | 0.322 | **0.386** | 0.328 | 0.287 |

**관문2 실패기준 충족** (A≈/<B; tiage A≈D). 관문1·2 실패기준 모두 충족; 규모·학습설정·도메인 전 조건 동일.

---

## 6. 추가 신호·무파라미터 세그멘터 (corpus_f1.0)

### 6.1 학습-dynamics 예측오차, density-matched (`test_sem_official.py`)
| set | Bp_cos | FDM_err | FDM_best |
|---|---|---|---|
| tiage | **0.492** | 0.428 | 0.457 |
| superseg | 0.426 | 0.494 | **0.499** |
| dialseg711 | 0.576 | 0.649 | **0.656** |
| ami | 0.322 | 0.349 | 0.324 |
FDM_best > cosine: superseg +0.073(p=2e-25), dialseg +0.100(p=1e-26); tiage −0.041(p=.06).

### 6.2 sticky SEM (`sticky_sem.py`, λ sweep): tiage 0.384 / ami 0.349 / superseg 0.382 / dialseg 0.332 (전부 Bp_cos 미만).

### 6.3 결합·정규화 (density-matched): fusion dialseg w0.25 **0.662**, superseg 0.499, tiage 0.495; 적응정규화 ami cosine/W40 0.399.

### 6.4 무파라미터 세그멘터
- 배포 μ+cσ c=1.0(`adaptive_segment.py`): tiage cosine 0.388/FDM_best 0.360; superseg 0.341/0.370; dialseg 0.552/0.616; ami 0.239/0.206.
- **ms_mean**(다중스케일 z평균, `multiscale_test.py`) density-matched: tiage 0.497, ami 0.387, superseg 0.466, dialseg 0.613.
- **otsu_pk**(peak값 Otsu, `threshold_sparse.py`).
- **ms_mean+otsu_pk**(배포, `capstone_segment.py`): tiage **0.439**, superseg **0.405**, dialseg **0.621**, ami **0.263**. naive(cos_raw+μ+1σ): 0.388/0.341/0.552/0.239.
  - 주의: `hi_ontop_v2`에 `_otsu_threshold` 기존 존재.

---

## 7. AMI 심층
### 7.1 ±2 tolerant + ORACLE (`ami_check.py`, density-matched): ORACLE 1.000 / C_lex 0.390 / FDM_best 0.324 / cosine 0.322 / random 0.310.
### 7.2 window aggregation (`block_test.py`): ami cos_raw 0.322→cos_w5 0.382; tiage cos_raw 0.492→cos_w2 0.504.
### 7.3 텍스트-only 신호 (`ami_textonly.py`, density-matched): proto_ewma.9 0.408 / asym12_3 0.402 / ms_mean 0.387 / texttiling 0.331. emb+lex 0.360, emb+lex+fdm 0.337.
### 7.4 비-텍스트(타임스탬프/화자) (`ami_structure.py`): gap 0.553, spk_new 0.450, emb+gap 0.545; 배포 gap+otsu_pk 0.509(n_pred 296 vs 233).

---

## 8. ICML 다이어리 (`icml_segment.py`, 정성)
ms_mean+otsu_pk 경계: `[1,6,18,21,27,29,34,37,39,42,59,61,69,71,79,88]`. HANDOFF_06 δ_eff(A): `[1,2,7,18,20,21,27,29,34,59,61,69,79]`. 차이: ms_mean에만 [6,37,39,42,71,88], δ_eff에만 [2,7,20,79]. gold 없음.

---

## 9. 실제 δ_eff(HiOnTopV2 c=1.0) 비교 (deployable, `compare_deltaeff.py`)
| held-out | δ_eff Score | ms_mean+otsu_pk | n_pred(δ_eff/ours) vs gold |
|---|---|---|---|
| tiage | 0.447 | 0.439 | 428/229 vs 315 |
| dialseg711 | 0.561 | **0.621** | 4294/2748 vs 2743 |
| superseg | 0.333 | **0.405** | 4500/2410 vs 4017 |

---

## 10. 핵심 수치 요약 (사실)
- 관문1: NMI(code;내용) > NMI(δ;내용) 전 도메인. R²(code→내용) K=8/16/32 = 0.063/0.092/0.118. 코드 분할 vs gold화제 ≈ 코드섞음(우연); 임베딩 KMeans는 0.47~0.52. 코드 축=내용(AMI superseg 0.20, tiage 0.055)+길이(0.135), 말투(질문/화자) 미미.
- 관문2 규모: TIAGE A_typed F1 1.78M→18.57M = 0.333→0.231, vs random p 0.005→0.96.
- 관문2 held-out: A_typed vs Bp — tiage 0.404/0.492, dialseg 0.337/0.576, superseg 0.357/0.426, ami 0.360/0.322.
- FDM_best vs cosine: superseg 0.499/0.426, dialseg 0.656/0.576, tiage 0.457/0.492.
- 배포 ms_mean+otsu_pk vs 실제 δ_eff: tiage 0.439/0.447, dialseg 0.621/0.561, superseg 0.405/0.333.
- AMI 텍스트-only 최고 0.408(proto_ewma); 비텍스트 gap 0.553/배포 0.509.
- 병목 범위(§1.5 최초 설계 시점): VQ만 사용. 이후 §12에서 연속 sparsity/noise 추가 실행.

---

## 12. 병목 재판정 — 연속(sparsity/noise) vs VQ (2026-07-10, 진행 중)

### 12.1 배경·설계 (개정 설계문서 `gonogo_expreiment.md` §2 반영)
"Learning Latent Action World Models In The Wild"(FAIR, arXiv 2601.05230): in-the-wild 데이터에서 VQ가 전이를 뭉개 실패, 연속+정규화(sparsity/noise)가 VQ를 앞선다는 주장. 이에 병목을 교체해 재판정. VQ NO-GO가 (c)VQ부적합 때문인지 검증.

### 12.2 아키텍처·설정 (`scripts/idm_lam/lam_bottleneck.py`, `run_bottleneck.py`)
- IDM: `concat(z_t, z_{t+1}−z_t)` → MLP[1536→512→256→out] → a_t. FDM: `concat(z_t, ā_t)` → MLP → ẑ_{t+1}.
- **병목 3종**:
  - **sparse**: ā_t = a_t(연속, latent_dim=64), 손실 += **λ·mean(|a_t|)** (L1). 정보량 손잡이 = λ.
  - **noise**: IDM이 (μ, logvar) 출력, ā_t = μ+σ·ε(학습)/μ(평가), 손실 += **β·KL** (VIB/β-VAE). 손잡이 = β.
  - **vq**(대조군): K=16, code_dim=32, commitment 0.25 (§1.5와 동일).
- 손실 = ‖ẑ_{t+1}−z_{t+1}‖²(dim합) + 병목항. Adam lr3e-4, batch8192, **latent_dim=64**.
- 학습데이터 = §1.2(B) 동일 대규모 코퍼스(18.77M쌍). frac로 부분 사용.
- **관문2 신호 A(연속)** = `1−cos(ā_t, ā_{t-1})` (인접 전이표현의 질적 변화; 예측오차 아님). density-matched top-N, DTS/AMI 공식 채점기.
- **관문1 지표**: R²(ā_t → z_{t+1}) 선형프로브(누출), R²(ā_t → z_{t+1}−z_t), AMI(clust(ā,16); topic30), active_dims(|값|>0.01·max 비율).
- Bp(surprise, 고정) = 1−cos(z_t,z_{t+1}): TIAGE 0.492(F1 0.370)/superseg 0.426(0.338)/dialseg711 0.576(0.487)/AMI 0.322(0.115). random: 0.399/0.406/0.346/0.310.

### 12.3 Gate-1 (누출) — frac 0.3, 8ep
| 병목·정규화 | recon | R²(ā→내용) | R²(ā→변화) | AMI(clust;topic) | active_dims |
|---|---|---|---|---|---|
| sparse λ=1 | 0.2115 | 0.464 | 0.409 | 0.372 | 0.94 |
| sparse λ=3 | 0.2154 | 0.348 | 0.331 | 0.361 | 0.93 |
| sparse λ=10 | 0.3010 | 0.169 | 0.182 | 0.208 | 0.52 |
| sparse λ=30 | 0.3811 | 0.074 | 0.084 | 0.174 | 0.26 |
| noise β=1 | 0.4323 | 0.282 | 0.318 | 0.184 | 0.28 |
| noise β=3 | 0.5556 | 0.057 | 0.085 | 0.079 | 0.81 |
| noise β=10 | 0.5556 | 0.006 | 0.010 | 0.136 | 0.94 |
| vq K=16 | 0.5324 | 0.060 | 0.058 | 0.119 | 0.37 |
| **noise β=10 (frac 1.0, seed0)** | 0.5422 | 0.005 | — | 0.148 | — |

### 12.4 Gate-2 (분절 성능, Score) — frac 0.3, 8ep, density-matched, held-out
표기: A (F1) · p(A−Bp) · p(A−rand). Bp/random 값은 §12.2.

| 병목·정규화 | TIAGE A (p vs Bp / rand) | superseg A (p/p) | dialseg711 A (p/p) | AMI A (p vs Bp / rand) |
|---|---|---|---|---|
| sparse λ=1 | 0.426 (6.2e-4 / 1.9e-1) | 0.419 (1.1e-6 / 1.1e-3) | 0.355 (3.6e-87 / 2.8e-5) | 0.400 (2.1e-3 / 1.9e-2) |
| sparse λ=3 | 0.403 (4.5e-4 / 7.3e-1) | 0.409 (2.2e-9 / 2.3e-2) | 0.377 (1.2e-76 / 6.4e-11) | 0.347 (6.1e-2 / 5.7e-1) |
| sparse λ=10 | 0.402 (1.0e-4 / 7.2e-1) | 0.415 (3.6e-7 / 1.6e-2) | 0.362 (3.7e-78 / 3.0e-5) | 0.367 (2.8e-2 / 1.4e-1) |
| sparse λ=30 | 0.398 (1.9e-5 / 8.3e-1) | 0.409 (8.2e-7 / 9.2e-3) | 0.367 (3.4e-78 / 1.4e-7) | 0.385 (5.3e-3 / 4.0e-2) |
| noise β=1 | 0.392 (1.0e-4 / 9.0e-1) | 0.434 (8.5e-3 / 8.0e-5) | 0.347 (5.9e-87 / 1.1e-1) | 0.354 (9.0e-2 / 5.0e-1) |
| noise β=3 | 0.383 (4.4e-5 / 8.2e-1) | 0.398 (3.6e-12 / 9.3e-1) | 0.325 (3.1e-90 / 8.5e-2) | 0.363 (6.5e-2 / 3.7e-1) |
| **noise β=10** | 0.442 (5.0e-2 / 1.3e-1) | 0.387 (3.4e-20 / 9.1e-2) | 0.313 (2.5e-93 / 2.3e-2) | 0.405 (1.2e-3 / 1.8e-3) |
| vq K=16 | 0.390 (3.1e-5 / 4.5e-1) | 0.415 (1.3e-5 / 2.1e-2) | 0.353 (2.7e-83 / 1.2e-1) | 0.348 (3.3e-2 / 1.7e-1) |

### 12.5 확정 실행 — noise β·seed·규모 (`confirm_noise.py`, 완료)
| config | recon | R²(ā→내용) | TIAGE A (p vs Bp/rand) | superseg A (p/p) | dialseg A (p/p) | AMI A (p vs Bp/rand) |
|---|---|---|---|---|---|---|
| noise β=10 frac0.3 seed0 (§12.4) | 0.5556 | 0.006 | 0.442 (5.0e-2 / 1.3e-1) | 0.387 (3.4e-20 / 9.1e-2) | 0.313 (2.5e-93 / 2.3e-2) | 0.405 (1.2e-3 / 1.8e-3) |
| **noise β=10 frac1.0 seed0** | 0.5422 | 0.005 | 0.387 (9.0e-5 / 7.9e-1) | 0.380 (1.1e-19 / 1.8e-1) | 0.339 (1.0e-86 / 3.0e-1) | 0.425 (3.0e-3 / 1.6e-2) |
| **noise β=10 frac1.0 seed1** | 0.5424 | 0.006 | 0.416 (1.5e-3 / 5.8e-1) | 0.404 (5.5e-10 / 1.5e-1) | 0.353 (2.5e-80 / 2.9e-3) | 0.386 (2.3e-2 / 1.3e-1) |
| noise β=5 frac0.3 seed0 | 0.5556 | 0.019 | 0.422 (6.2e-3 / 4.4e-1) | 0.394 (9.9e-20 / 1.9e-1) | 0.315 (8.9e-93 / 4.6e-2) | 0.386 (7.6e-3 / 1.1e-2) |
| noise β=20 frac0.3 seed0 | 0.5556 | 0.003 | 0.419 (1.8e-2 / 2.2e-1) | 0.405 (2.4e-11 / 3.6e-1) | 0.316 (6.9e-94 / 2.7e-2) | 0.403 (3.5e-3 / 8.6e-3) |

- (사실) TIAGE A: 5개 config 중 4개 p(A−Bp)<0.02 (Bp 0.492 미만). p=5.0e-2 는 β=10·frac0.3·seed0 1건뿐이며, 같은 β=10 다른 seed(frac1.0 seed1=0.416, p1.5e-3)·다른 β(5/20)에서 재현 안 됨.
- (사실) AMI A: 5개 config 전부 p(A−Bp)<0.03 이며 Bp(0.322) 상회 (0.386~0.425), p(A−rand)도 β=5(1.1e-2)·β=20(8.6e-3)에서 유의.
- (사실) superseg A(0.38~0.41) < Bp 0.426, p(A−rand)>0.09(랜덤과 무차별~근소). dialseg A(0.31~0.35) ≈ random 0.346, Bp 0.576 대비 p<1e-80.
- (사실) λ/β↑ → R²(ā→내용) 감소: sparse 0.464→0.074(λ 1→30), noise 0.282→0.003(β 1→20). noise β≥3·sparse λ=30 의 R²(0.003~0.074) ≈ vq(0.060).

---

## 11. 재현 코드 (`scripts/idm_lam/`)
`lam.py`(모델) · `build_corpus.py`(코퍼스,--resume) · `train_corpus.py`(--frac) · `dts_common.py`/`ami_eval.py`(로더) · `gate1_diagnostics.py`/`gate1_heldout.py` · `gate2_segmentation.py`/`gate2_significance.py`/`gate2_heldout.py` · `run_final.sh`(scale ablation) · `train_audit.py` · `test_sem_official.py`/`sticky_sem.py`/`fusion_test.py`/`adaptive_test.py`/`adaptive_segment.py`/`adaptive_robust.py`/`multiscale_test.py`/`capstone_segment.py`/`threshold_sparse.py` · `ami_check.py`/`block_test.py`/`ami_textonly.py`/`ami_structure.py` · `icml_segment.py`/`compare_deltaeff.py` · `code_vs_gold.py`/`code_organizes_by.py` · **(§12 병목재판정)** `lam_bottleneck.py`(sparse/noise/vq 모델)/`run_bottleneck.py`(--only,--lam,--beta,--frac)/`confirm_noise.py`(β·seed 확정).
산출물: 모델 `results/lam_*.pt`, 진단 `results/{gate1,gate2}_*.json`, 정성 `results/{gate1_samples,code_typing}_*.txt`, 코퍼스(scratch) `idm_corpus/`, AMI `data/ami/`.
