# IDM 기반 에피소드 분절기 Go/No-Go — 풀학습 검증 (수정 설계, 최종)

**실험일**: 2026-07-09 · **설계**: `gonogo_expreiment.md` · **판정: ❌ NO-GO (관문 1·2 모두 실패, 규모·학습방식 불변)**

> ⚠️ **설계 수정 이력**: 최초 시도는 superseg_train으로 학습→superseg_test로 평가(동일분포 오염) + 78k쌍(과소)이었다. 사용자 지적으로 **학습(대규모 무라벨)·평가(held-out) 완전분리 + 최대 규모 + 노이즈 구어(AMI) 포함**으로 재설계해 재실행했다. 본 REPORT는 수정 설계의 결과다. (구 결과는 §附.)

---

## 1. 실험 setup

**아키텍처 (LAM = IDM + VQ + FDM)** — `scripts/idm_lam/lam.py` (doc §2 스펙 그대로)
- 상태 z_t = frozen SBERT `paraphrase-multilingual-mpnet-base-v2` (768D, L2정규화; HANDOFF_06 δ_eff와 동일 인코더 → 관문2 공정비교 B' 성립).
- IDM g: `(z_t, z_{t+1}−z_t)`(차이입력) → MLP[1536→512→256→32].
- VQ 병목: K=16, code_dim=32, commitment 0.25, EMA 0.99, kmeans_init, dead-code 재활성. **정보용량=log2(16)=4비트**.
- FDM f: **concat**[z_t; code_vec](prepend) → MLP[800→512→768].
- 손실 = ‖ẑ_{t+1}−z_{t+1}‖² + commitment.

**학습 코퍼스 (대규모·다양·무라벨, 평가와 완전분리)** — `scripts/idm_lam/build_corpus.py`
- **21.49M 발화 / 2.73M 대화 / 18.77M 인접쌍** (원본 78k의 **240×**), 11 샤드.
- 소스: SODA 1.486M · ultrachat 800k · ultrachat_200k 231k · hh_rlhf 161k · empathetic 19.5k · personachat 17.9k · persona_synth 8.9k · blended_skill_talk 4.8k · **AMI-train 104 미팅**(노이즈 구어).
- 인프라: fp16 샤드 저장 + **--resume**(크래시 시 마지막 샤드 이후만 재임베딩, 검증됨).

**평가 (전부 held-out, IDM이 학습에서 못 봄)**
| 데이터셋 | n | 경계정답 | 채점기 |
|---|---|---|---|
| **tiage_test** | 100 대화 | **인간** topic-shift | DTS(끝-turn, -1) |
| dialseg711_test | 711 | splice(구조) | DTS |
| superseg_test | 1322 | 문서섹션(구조) | DTS |
| **ami_eval** | 35 미팅 | 인간 topic(구어) | AMI(start-turn, ±2 tol) |

**경계 결정 규칙 보정**: A_typed(코드별 경계율)·A_bigram(코드 bigram)은 **train split 전용**(superseg_train + tiage_train + AMI-train)으로 보정 → 표상은 무라벨 코퍼스, 결정규칙은 각 평가셋의 train split(정상 train/test 분리).

**비교군 (전부 density-matched, 대화별 예측경계수 = gold수)**: A_typed·A_bigram(우리) / **Bp = 1−cos(z_t,z_{t+1})**(surprise, 같은 임베딩; = HANDOFF_06 δ신호) / C_lex(TextTiling식) / **D_random**(밀도랜덤 ★통제) / E_unif(균등).

---

## 2. 결과

### 2.1 데이터 규모 ablation (frac 0.1/0.3/1.0 = 1.78M/5.43M/18.57M 쌍) — 핵심

**TIAGE(인간) A_typed** (Bp=surprise F1 0.370 고정, D_random F1 0.229):

| 학습쌍 | val recon | A_typed F1 | A_typed Score | ΔF1 vs **Bp** (p) | ΔF1 vs **random** (p) |
|---|---|---|---|---|---|
| 1.78M | 0.542 | 0.333 | 0.471 | −0.038 (p=.16, **무승부**) | +0.103 (p=.005, 이김) |
| 5.43M | 0.534 | 0.314 | 0.453 | −0.056 (p=.12, **무승부**) | +0.085 (p=.015, 이김) |
| **18.57M** | **0.515** | **0.231** | 0.404 | **−0.139 (p=6e-5, 짐)** | **+0.002 (p=.96, =랜덤)** |

→ **규모↑(recon 0.542→0.515 개선)에 따라 인간 경계 성능이 오히려 랜덤 수준으로 붕괴.** "데이터 부족" 가설 반박.

### 2.2 학습방식 audit (`scripts/idm_lam/train_audit.py`, 1.14M 쌍) — 학습 버그 아님

| 설정 | val recon | 코드기여 | codebook ppl | NMI(code;내용) | NMI(δ;내용) | TIAGE A_bigram | TIAGE Bp |
|---|---|---|---|---|---|---|---|
| **FDM-only(코드X)** | 0.623 | — | — | — | — | — | — |
| current(lr3e-4,10ep) | 0.597 | 0.031 | 4.4 | 0.183 | 0.023 | 0.256 | 0.370 |
| **better(lr1e-4,cos,25ep)** | 0.582 | 0.049 | 11.2 | **0.217↑** | 0.023 | 0.236↓ | 0.370 |
| cosloss(다른 목적함수) | 0.688 | 0.032 | 6.9 | 0.220 | 0.023 | 0.211 | 0.370 |

→ FDM-only가 이미 recon 0.623 (z_t가 z_{t+1}을 이만큼 예측). 코드는 겨우 ~0.04 개선. **더 잘 학습(better)할수록 내용누출↑·TIAGE↓**. 목적함수를 바꿔도 동일. 학습 품질/수렴 버그 아님.

### 2.3 관문 1 (누출) — held-out, 최대규모 모델(frac 1.0)

| set | NMI(code;내용) | NMI(δ;내용) | cond(code;경계) | cond(δ;경계) | R²(code→내용) |
|---|---|---|---|---|---|
| tiage | 0.076 | 0.030 | 0.084 | 0.139 | 0.041 |
| superseg | 0.292 | 0.049 | 0.084 | 0.141 | 0.081 |
| dialseg711 | 0.265 | 0.040 | 0.077 | 0.415 | 0.038 |
| **ami(구어)** | 0.319 | 0.029 | 0.106 | 0.225 | 0.131 |

→ 모든 held-out 도메인에서 코드가 surprise보다 내용을 **2~10× 더** 담고(누출), topic 통제 후 전이신호는 surprise 이하. AMI(노이즈 구어)에서 누출 최대.

### 2.4 관문 2 (분절 성능) — held-out, 최대규모 모델(frac 1.0), density-matched

| dataset | 지표 | A_typed | A_bigram | **Bp(surprise)** | C_lex | **D_random** | E_unif |
|---|---|---|---|---|---|---|---|
| **tiage(인간)** | Score | 0.404 | 0.399 | **0.492** | 0.373 | 0.414 | 0.426 |
| | F1 | 0.231 | 0.258 | **0.370** | 0.218 | 0.229 | 0.253 |
| dialseg711 | Score | 0.337 | 0.289 | **0.576** | 0.480 | 0.325 | 0.372 |
| superseg | Score | 0.357 | 0.322 | **0.426** | 0.435 | 0.378 | 0.441 |
| **ami(구어)** | Score | 0.360 | 0.334 | 0.322 | **0.386** | 0.328 | 0.287 |

유의성(A_typed): tiage ΔF1 vs Bp −0.139(p=6e-5), vs random +0.002(p=.96) · dialseg vs Bp −0.341(p<1e-90) · superseg vs Bp −0.106(p<1e-27) · ami vs Bp +0.029(p=.03) but vs random +0.003(p=.96).

→ **A는 surprise를 어디서도 유의하게 못 이김.** TIAGE(인간)에서 유의하게 지고 **랜덤과 무차별**. dialseg/superseg 유의하게 짐. AMI에서 Bp를 근소하게 넘지만 거기선 surprise 자체가 랜덤 수준(0.322 vs random 0.328)이고 A도 랜덤과 무차별(p=.96)이며 C_lex(0.386)에 짐 → 무의미한 승리.

---

## 3. 해석 (메커니즘)

1. **근본원인 = 대화에선 z_t가 z_{t+1}을 약하게만 결정.** FDM-only recon 0.623. 비디오 LAM이 성립하는 이유(정적 배경 → o_t가 o_{t+1} 대부분 예측 → 행동은 저차원 잔차만)가 대화엔 없다. 그래서 VQ의 4비트는 재구성에 가장 유용한 잔차=**내용 방향**을 담는다(누출).
2. **더 잘 학습·더 많은 데이터 = 더 심한 누출.** recon이 좋아질수록(0.62→0.52) 4비트를 내용 인코딩에 더 효율적으로 쓴다 → TIAGE 신호 0.333→0.231(랜덤). 학습방식·규모 audit이 같은 방향.
3. **surprise가 구조적으로 우월.** Bp(1−cos)는 도메인·규모 무관 F1 0.370(TIAGE). 코드는 내용을 담느라 이 단순 신호를 못 이긴다.
4. **AMI(노이즈 구어)에선 임베딩 신호 전반이 무력** — surprise·코드·랜덤 모두 0.32 근처, 어휘신호(C_lex 0.386)가 그나마 나음. 이 도메인은 임베딩-거리류 자체의 한계.

---

## 4. 판정 — ❌ NO-GO

| 관문 | 기준 | 결과 |
|---|---|---|
| 1 | 코드가 내용으로 갈림/누출↑ | **위반**: NMI(code;내용) 2~10×(vs surprise), 모든 held-out 도메인, 규모 불변 |
| 2 | A > B (surprise) | **위반**: TIAGE 유의하게 짐(p=6e-5), 어디서도 유의한 승리 없음 |
| 2 | A > D (밀도랜덤) | **위반**: 최대규모에서 TIAGE A ≈ random(p=.96) |

**세 NO-GO 조건 모두 성립.** 그리고 doc §9가 우려한 두 교란요인을 정면 통제:
- **데이터 규모**: 78k→18.6M(240×). 규모↑에 성능 **하락**(0.333→0.231) → 데이터 부족 아님.
- **학습 방식**: lr·schedule·epochs·목적함수 4변형. 더 잘 학습할수록 누출↑ → 학습 버그 아님.
- **도메인**: 인간(TIAGE)·구조(dialseg/superseg)·**노이즈 구어(AMI)** 4 held-out. 전부 실패.

→ **IDM 기반 대화 분절 방법론 종료. 가설2(메모리 실험)로 진행 안 함.** 기존 δ_eff/surprise가 IDM 코드보다 우월함을 재확인. NO-GO는 튜닝·규모·데이터·도메인 어느 것으로도 뒤집히지 않는 **구조적 결론**(doc §9 사전 고지가 데이터로 확증됨).

---

## 5. 한계 / 검증 미해결 (솔직 기록)

- **A 조작화**: shiftness(supervised, train-split 보정)·bigram(unsup) 2종. "코드 질적 전환"의 다른 조작화(코드공간 change-point/HMM)는 미시도. 단 최유리 supervised 조작화가 규모·학습 전 구간에서 실패했고 메커니즘(FDM-only 0.62, 누출)이 설명하므로 뒤집힐 가능성 낮음.
- **EM-LLM 토큰 surprise(B)**: held-out 대량 평가엔 미포함(B' 임베딩 surprise가 공정 비교이며 그것조차 A를 이김). HANDOFF_06 §7에 89문장 참조 존재.
- **AMI 학습기여**: AMI-train 104미팅(38k턴)은 코퍼스의 0.2%로 소량. 노이즈 구어를 더 넣어도 결론 불변 예상(규모·학습 audit이 "더 넣을수록 악화"를 시사).
- **인코더 1종**(mpnet). 다른 인코더에서 FDM-only 예측도가 크게 달라지면 여지 있으나, δ신호도 같은 인코더라 상대비교는 유효.
- **seed 1**(학습). 효과크기(TIAGE ΔF1 −0.139, p<1e-4)가 seed 노이즈를 크게 상회.

---

## 6. 산출물

- 코드: `scripts/idm_lam/{lam,build_corpus,train_corpus,gate1_heldout,gate2_heldout,gate2_segmentation,train_audit,dts_common,ami_eval,run_final.sh}`
- AMI: `data/ami/{annotations,topic/*.json,split.json}` (139미팅, train104/eval35)
- 모델: `results/lam_corpus_f{0.1,0.3,1.0}.pt` · 진단: `results/{gate1_heldout,gate2_heldout}_corpus_f*.json`
- 코퍼스(scratch, 재생성가능): `idm_corpus/emb_shard_*.npy`(11샤드), `meta.json`

### §7. 후속 — SEM식 재구성 (학습 forward-dynamics를 surprise로) — 2026-07-10

**동기**: 코드가 "전이유형"이 아니라 "화제내용"으로 갈렸으니(§附/code_typing), 코드를 novel 신호로 쓰는 대신 **SEM 사건모델처럼 학습 dynamics의 예측오차**로 경계를 잡자는 아이디어(사용자 제안). 우리 FDM = 학습형 forward dynamics.

**신호**(`scripts/idm_lam/test_sem_official.py`, corpus_f1.0, density-matched, 정식 채점기):
- `FDM_err` = ‖FDM(z_t, c_mean) − z_{t+1}‖² (학습 dynamics surprise)
- `FDM_best` = min_k ‖FDM(z_t, c_k) − z_{t+1}‖² (어느 event-code로도 설명 안 되면 경계 = SEM 사건전환)
- vs `Bp_cos` = 1−cos (기존 δ_eff = "무변화 dynamics" surprise)

| set | Bp_cos Score | FDM_best Score | ΔF1(p) |
|---|---|---|---|
| **tiage(인간)** | **0.492** | 0.457 | −0.041 (p=.06) cosine 우세 |
| superseg | 0.426 | **0.499** | +0.070 (p=2e-25) 학습 승 |
| dialseg711 | 0.576 | **0.656** | +0.100 (p=1e-26) 학습 승 |
| ami(구어) | 0.322 | 0.324 | ~random |

**해석**: 학습 forward-dynamics surprise가 **구조 경계(splice/섹션)에선 cosine-δ를 유의하게 이김**(SEM 충실: "학습 event가 설명 못하는 전이=경계"). 그러나 (1) **인간 경계(TIAGE)에선 cosine-δ가 여전히 우세**, (2) 이 신호는 **surprise 계열**이라 doc §1의 "surprise와 구별되는 신호" 목표는 포기하는 것 = Hi-OnTop 기존 δ_eff의 **개선판**이지 새 패러다임 아님, (3) AMI는 임베딩류 전반 무력.
→ **중간결론**: IDM 코드의 novel-transition 가설은 여전히 NO-GO. 단 부산물로 "학습 dynamics surprise > cosine-δ (구조 경계 한정)"라는 δ_eff 개선 후보를 확보. 인간 경계 개선은 미해결.

**후속2 — 완전한 sticky SEM 추론** (`scripts/idm_lam/sticky_sem.py`): K코드=event 어휘, 각 event dynamics=FDM(·,c_k), sticky sCRP로 active event 유지·전환. density-matched, λ∈{0,.5,1,2} sweep.

| set | Bp_cos Score | sticky SEM best | ΔF1(p) |
|---|---|---|---|
| **tiage(인간)** | **0.492** | 0.384 | −0.139 (p=1e-4) **유의하게 짐** |
| ami(구어) | 0.322 | 0.349 | +0.040 (p=.01) but 둘 다 ~random(.328) |
| superseg | 0.426 | 0.382 | −0.069 (p=1e-15) 짐 |
| dialseg711 | 0.576 | 0.332 | −0.337 (p<1e-90) 크게 짐 |

→ **stickiness는 인간 경계를 구제 못함(오히려 악화)**, 그리고 per-step FDM_best가 구조에서 얻은 이득도 순차 sticky 추론이 파괴(dialseg 0.656→0.33). 원인: sticky는 이진 전환-궤적 메커니즘이라 density-matched top-N **랭킹 신호로 부적합**; active event 진입 후 staying이 싸져 큰 구조 점프를 랭킹에서 덜 부각.

→ **SEM식 재구성 중간결론**: 인간 경계에선 cosine-δ 못 이김. 구조 경계에선 per-step FDM_best가 이김(surprise 계열).

**후속3 — 배포기준(무파라미터) 재평가** (`scripts/idm_lam/adaptive_segment.py`): ⚠ 앞의 density-matched(gold 개수 부여=oracle) + fusion/window 데이터셋별 sweep은 벤치마크 최적화였음. 프로젝트 철학(c=1.0 고정 calibration-free 적응임계 μ+c·σ, 튜닝 없음)대로 **개수 자가결정·density-match 제거·무튜닝** 재평가:

| set | cosine-δ Score | FDM_best Score | 승자 |
|---|---|---|---|
| TIAGE(인간) | **0.388** | 0.360 | cosine |
| superseg | 0.341 | **0.370** | FDM_best |
| dialseg711 | 0.552 | **0.616** | FDM_best |
| ami(구어) | 0.239 | 0.206 | (둘 다 과분절: n_pred 1302~1765 vs gold 233) |

→ **배포기준(무파라미터)에서도 FDM_best가 구조 경계에서 cosine-δ를 이김**(dialseg +0.064, superseg +0.029, 튜닝 0). 인간 경계는 cosine 근소 우세(표상 천장). AMI는 sparse-boundary라 적응임계 개수보정 실패(신호 아닌 임계 문제, 공통). density-matched 수치(TIAGE 0.49 등)는 oracle-개수 값이며 배포성능은 이 표.

**총괄(중간)**: IDM 코드 novel-transition 가설 = NO-GO. 부산물 = FDM_best(SEM식 학습dynamics surprise)가 구조 경계에서 δ_eff를 무파라미터로 개선.

**후속4 — 노이즈 진단 + 무파라미터 노이즈/희소 적응 세그멘터** (2026-07-10):
- **AMI(구어) 실패 원인 규명**: 코드 내용typing은 AMI에서도 최강(NMI 0.319). 실패는 "내용 못담아서"가 아니라 **turn단위 내용 churn(필러·백채널)이 화제경계 신호를 삼켜서**. 검증: ±2 tolerant + oracle개수 줘도 임베딩 cosine 0.322≈random 0.310; **어휘 C_lex 0.390만 유의**. 그러나 임베딩도 **window aggregation(w=5)으로 0.322→0.382**(어휘 0.390 근접) → 노이즈 문제이지 임베딩 무능 아님. (`ami_check.py`, `block_test.py`)
- **무파라미터 노이즈-적응 신호 `ms_mean`**: block-cosine을 스케일 w∈{1,2,3,5,8}에서 z-score 후 평균. 고정 방법 하나가 **깨끗한 데이터=작은window 최적, 노이즈=큰window 최적을 동시 근접**(TIAGE 0.497≈cos_w2 0.504; AMI 0.387≥cos_w5 0.382). (`multiscale_test.py`)
- **무파라미터 희소-적응 개수보정 `otsu_pk`**: local-max peak **값들에만** Otsu. 희소경계(AMI 2%)에 자가적응.
- **배포형 무파라미터 세그멘터 = ms_mean + otsu_pk** (튜닝·density-match·파라미터 0), naive(cos_raw+μ+1σ) 대비 **4셋 전부 개선**:

| set | naive Score | ms_mean+otsu_pk Score |
|---|---|---|
| tiage(인간) | 0.388 | **0.439** |
| superseg | 0.341 | **0.405** |
| dialseg711 | 0.552 | **0.621** |
| ami(구어) | 0.239 | **0.263** (F1 .208→.255) |

(`capstone_segment.py`, `threshold_sparse.py`) → 부품: (a) FDM_best=학습dynamics surprise, (b) ms_mean=무파라미터 노이즈적응 다중스케일 신호, (c) otsu_pk=희소적응 개수보정.

**후속5 — AMI 상한 규명 (텍스트-only·무LLM·무파라미터)** (`ami_textonly.py`, `ami_structure.py`, 2026-07-10):
- **텍스트-only 천장 = ~0.40** (oracle개수 줘도). 시도한 ~10 신호 전부 걸림:
  cosine 0.32 / lexical 0.39 / ms_mean 0.387 / asym(화제-so-far vs 유입) 0.402 / **proto_ewma(SEM식 화제 prototype 거리) 0.408** / texttiling 0.331 / FDM_best 0.324 / 멀티모달 결합 무익. 프로젝트 자체 AMI text SOTA(~0.37)도 여기.
- ORACLE=1.0이므로 경계는 국소화 가능하나, **노이즈 낀 짧은 구어의 거친 안건 경계(2%)를 국소 텍스트 거리/응집으로는 못 집음** = 정보 한계. 0.5로 가려면 LLM(의미)·타이밍/화자(구어구조)·지도학습 중 하나를 풀어야 함(셋 다 제약 위반).
- **참고(텍스트-only 아님, 범위 밖)**: AMI 턴의 `start`(타임스탬프) `gap`(침묵) 신호는 density-matched 0.553 / 배포 gap+otsu_pk 0.509 로 0.5 초과 — 정보가 **타이밍**에 있음을 보여줌(실시간 스트리밍이면 활용 가능하나 텍스트-only 방법에는 미포함).
- **결론**: 텍스트-only+무LLM+무파라미터 3제약 하 AMI ~0.40이 정직한 상한. **텍스트 대화(TIAGE/DTS/ICML)에선 ms_mean+otsu_pk가 유효**하고 AMI 구어가 구조적 한계 도메인.

**후속6 — 실제 δ_eff와의 정직한 정면 대결** (`compare_deltaeff.py`, 2026-07-10):
⚠ 앞의 "naive"는 strawman(raw cos+μ+1σ). 진짜 검증 = 프로젝트 실제 **HiOnTopV2**(ctx m=2·ρ=0.7·blend + 적응임계 μ+cσ, c=1.0)와 배포기준·무튜닝·동일인코더 비교. (주의: `hi_ontop_v2`에 이미 `_otsu_threshold` 존재 → otsu 임계는 부분 기존작업.)

| held-out | 실제 δ_eff Score | ours(ms_mean+otsu_pk) | n_pred(δ_eff/ours) vs gold |
|---|---|---|---|
| tiage(인간) | **0.447** | 0.439 | 428/229 vs 315 |
| dialseg711 | 0.561 | **0.621** | 4294/2748 vs 2743 |
| superseg | 0.333 | **0.405** | 4500/2410 vs 4017 |

→ **ours가 실제 δ_eff를 구조 경계에서 유의 개선**(dialseg +0.06, superseg +0.07), 주원인 = **개수보정(otsu_pk)**: δ_eff는 과분절, otsu_pk는 gold 근접. **인간(TIAGE)은 무승부**(0.439 vs 0.447). ICML 다이어리서 ms_mean이 δ_eff가 놓친 [37]"주제가 달라지는"·[39]포스터시작 포착과 일치.
→ **이 스레드 최종**: IDM novel-signal NO-GO이나, **ms_mean(노이즈적응 신호)+otsu_pk(개수보정)가 실제 δ_eff를 구조 경계서 개선**(무파라미터·텍스트-only). 인간 경계·구어(AMI)는 미해결(정보 한계).

### §附. 구 설계(오염) 결과 — 참고용
superseg_train 학습→superseg_test 평가(78k쌍): A가 superseg_test에서만 Bp를 이겼으나(Score 0.494 vs 0.426), 이는 in-domain 과적합 아티팩트였고 held-out 재설계에서 무승부로 소멸. 세부: `results/gate2_K16.json`, `gate2_significance.json`.
