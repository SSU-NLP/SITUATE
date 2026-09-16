# Method Registry — 명칭 확정 · 기존 시도 · 앞으로의 사다리

> **이 문서가 방법론 명칭의 canonical 소스다.** 다른 문서·코드·대화에서 방법을 지칭할 때 반드시 여기 등록된 이름을 쓴다.
> 배경: "IDM"이라는 약어가 두 개의 완전히 다른 것을 가리키며 장기간 혼란을 유발했다(아래 §0). 이 문서로 종결한다.
> **이 문서의 canonical 범위:** 명칭·전체 지도(M/B 계열)·논문 포지셔닝·리뷰어 Q&A·베이스라인 순환 방지 규칙. 실행 수치의 canonical 기록은 `phase0-decisions-and-runs.md`(ledger), M3 스펙은 `phase0-annotation-guide.md`. ~~Tier 2 절차는 `tier2-heldout-spec.md`~~ → **그 문서는 전체 폐기됨**(D-M13①); 현행 held-out = **Tier1 6편**(신규 *Born a Crime* 3편, 사람 20명 주석).

---

## §0. 용어 사고 (TERMINOLOGY INCIDENT) — 반드시 먼저 읽을 것

### ❌ "IDM"을 M1(코사인 드리프트)의 이름으로 쓰지 말 것

- **IDM = Inverse Dynamics Model.** 월드모델/LAM 문헌의 용어이며, **오직 M2(LAM) 안의 역동역학 모듈**만을 가리킨다.
- 그런데 EACL Phase 0의 코드·결과표(`add_idm_scores.py`, 결과표의 "IDM" 열)에서 **M1(delta_eff, 코사인 드리프트)을 "IDM"이라 불러왔다.** 이는 **오칭(misnomer)**이다.
- **M1은 역동역학 모델이 아니다.** 학습이 전무하고, 예측도 없고, 단지 **문장 임베딩 간 코사인 거리**로 의미 변화를 재는 규칙 기반 화제경계 탐지기다.

### 정정 규칙

| 잘못된 표기 | 올바른 표기 |
|---|---|
| "IDM" (Phase 0 결과표의 baseline 열) | **CosDrift (M1)** |
| "IDM_raw" | **CosDrift-raw** (a=1, 순수 1−cos(s_{t-1},s_t)) |
| "IDM_tuned" | **CosDrift-tuned** (δ*/HP 캘리브 적용) |
| "IDM" (M2 아키텍처 내부 모듈) | **IDM** ✅ (이것만 정당한 사용) |

⚠️ **기존 결과표·스크립트명(`add_idm_scores.py`)에 남아 있는 "IDM"은 전부 CosDrift를 의미한다.** 논문·신규 문서에서는 CosDrift로 표기하고, 레거시 파일명은 각주로 처리한다.

---

## §1. 전체 지도 — 기존 3개(완료) + 앞으로 4개

> **상태 갱신 2026-07-28.** 아래 M4~M7 의 "[예정]" 표기는 2026-07-15 이후 stale 이었다.
> 실제 진행: **M4/M5 는 Phase 2 slot-alignment de-risk 를 이미 통과**(`src/hi_ontop/factored_slot.py`,
> `outputs/experiments/m5_phase2_slotalign/{M4.pt,M5.pt,REPORT.md}`, typed-slot **VIABLE**,
> factoring gain +0.044±0.005 — 단 학습 10편 중 7편이 오염 Tier2 라는 caveat, D-M11 ①).
> 2026-07-28 현재 **청정 앵커 10,598편(D-B15)으로 본 학습 착수** = 1차 탐색 패스(D-B16),
> 작업 디렉토리 `outputs/experiments/m4m5_phase1_anchor/`. **M6/M7 은 여전히 미착수.**

```
[완료]
 M1  CosDrift        규칙(코사인)      학습X      → EMNLP 계열. scalar 한계.
 M2  MonoLAM         단일잠재+VQ       무라벨학습  → 실패(누출·분절 랜덤)
 M3  S5-Extractor    LLM enum 5축      학습X      → 조건부 Go (fiction 승)
                                                    ★ M4~M7의 anchor 라벨 공급원
                                                    현행 = v3.1 (D-B14), 라벨러 Qwen3.6-35B-A3B

[진행 중 — M3의 enum을 anchor 라벨로 학습]
 M4  MonoAnchor      단일잠재 + anchor            ← 통제군 (분리가 필요한가?)
                                                    Phase2 de-risk ✅ / 본 학습 🔄 (D-B16)
 M5  FactoredSlot    5부분공간 + 축별dyn + anchor ← ★메인
                                                    Phase2 de-risk ✅ / 본 학습 🔄 (D-B16)
[미착수]
 M6  FactoredShared  5부분공간 + 공유dyn + anchor ← FLAM식
 M7  SlotAttnFactored 5슬롯(slot-attention)+anchor ← M5 부족 시
```

**핵심 관계:**
- **M3는 방법이자 인프라다.** Phase 0의 측정기이면서, 동시에 **M4~M7의 학습 라벨(anchor)을 만드는 공장**이다. M3 없이는 M4~M7이 성립하지 않는다.
- **M2와 M4의 차이는 단 하나: anchor 유무.** M2는 "다음 임베딩 예측"만 시켰고 5축을 담으라는 압력이 없었다 → 잠재가 내용(content)으로 붕괴. M4는 같은 단일잠재 구조에 **M3의 5축 라벨을 anchor로 추가**한다.
- **M4와 M5의 차이도 단 하나: 표현이 하나냐 다섯이냐.** 이 둘의 대조가 **"분리(factoring)의 인과적 기여"**를 측정하는 핵심 애블레이션이다.

---

## §2. 완료된 시도 (3개)

### M1 — CosDrift (구 "delta_eff", 1차, **EMNLP 계열**)

**정체:** v4.1.x SEM2 full 머신(EventRNN·f0/restart·per-topic 분산·sticky-CRP α)이 default에서 **전부 inert**임이 감사(2026-05-22, 38,242 turn byte-identical)로 확인 → 그 **환원형**. 즉 이름은 SEM이었으나 실제 작동한 건 코사인 드리프트뿐이었다.

**알고리즘** (온라인, 과거만, O(m)/turn — **학습 없음, 예측 없음**):
```
c_{t-1}    = normalize( Σ_{i=1..min(m,t-1)} ρ^{i-1}·s_{t-i} )   # 최근 가중↑
delta_prev = 1 − cos(s_{t-1}, s_t)
delta_ctx  = 1 − cos(c_{t-1}, s_t)
delta_eff  = a·delta_prev + (1−a)·delta_ctx
boundary ⟺ delta_eff ≥ δ*
```
**HP (canonical, TIAGE-calibrated):** m=2, ρ=0.7, a=0.5, δ*=0.5594.
**성능(v4.1.x):** TIAGE 0.4675 / Dialseg711 0.5897 / SuperDialseg 0.4631.
**유일한 fit:** δ* 스칼라 하나. 시드 없음.
**EACL(M3) 비교에서의 사용:** `add_idm_scores.py`(레거시명)로 delta_eff 스트림만 추출(δ*=2.0으로 경계판정 미사용, 연속 drift 점수로만). **CosDrift-raw** = a=1 = 순수 1−cos(s_{t-1},s_t).
**판정:** scalar 한계. 사건 경계(B)에서 무력화됨(Pieman AUC 0.53~0.63).

### M2 — MonoLAM (2차) — **월드모델 LAM 아이디어 차용, monolithic**

**출처 아이디어:** Latent Action Model (LAM) — 월드모델 계열. **IDM(Inverse Dynamics Model)으로 잠재 전이를 추론하고, FDM(Forward Dynamics Model)으로 다음 관측을 예측**하는 구조. **표현이 단일 잠재(monolithic)**라는 게 핵심 특징이자 결과적 死因.

**아키텍처** (`scripts/idm_lam/lam.py`):
- **IDM**: concat(z_t, z_{t+1}−z_t) [1536] → MLP [1536→512→256→32] GELU → 연속 전이 a_t (code_dim=32)
- **VQ 병목**: K∈{8,16,32}, code_dim=32, commitment=0.25, EMA decay=0.99, kmeans_init(10), threshold_ema_dead_code=2
- **FDM**: concat(z_t, code_vec) [800] (prepend) → MLP [800→512→768] → ẑ_{t+1}
- **Loss** = ‖ẑ_{t+1}−z_{t+1}‖² + commitment
- **anchor 없음.** 5축 라벨도, 5축 출력 헤드도 없음. **순수 자기지도.**

**학습 데이터:** (A) superseg_train 78,255쌍(test와 동일출처 → 오염) → (B) 무라벨 대규모 `build_corpus.py` = **18,765,307 인접쌍** (SODA·UltraChat·HH-RLHF·PersonaChat·AMI-train 등).
**설정:** Adam lr 3e-4, batch 8192, 10ep, seed 0.

**관문 1 (누출) — 실패:**
- NMI(code; 내용) held-out: tiage 0.076 / superseg 0.292 / dialseg 0.265 / ami 0.319 → **코드가 "내용의 압축본"이 됨**
- 코드분할 vs gold화제 ≈ 코드섞음(우연 수준): tiage 0.311 vs 0.329
- 코드가 갈리는 축: **내용 + 길이 > 화자·질문**

**관문 2 (분절, density-matched Score) — 실패:**
- held-out A_typed vs Bp: tiage 0.404/0.492 · dialseg 0.337/0.576 · superseg 0.357/0.426 · ami 0.360/0.322
- **규모 ablation (결정적):** TIAGE 1.78M 0.333 → 5.43M 0.314 → **18.57M 0.231** — **데이터를 키울수록 오히려 악화.** 데이터 부족이 아니라 **구조가 틀림.**

**§12 병목 재판정** (연속 vs VQ, latent_dim=64): sparse(L1) R² λ=1 0.464→λ=30 0.074; noise(VIB) R² β=1 0.282→β=20 0.003. **연속 병목도 VQ와 대동소이 — 관문 미통과.** ⇒ **VQ 탓이 아니라 monolithic 구조 자체의 문제.**

**판정: NO-GO.** 그리고 이 실패는 **폐기 자산이 아니라 핵심 근거**다:
> **"anchor 없는 monolithic 자기지도 잠재는 대규모 학습으로도 원하는 요인을 담지 않는다"** — Locatello 2019 불가능 정리의 실증. **M4~M7 전부에 anchor가 필수인 이유.**
> 또한 FLAM(2026)이 정확히 이 monolithic LAM의 한계를 지적하고 factored로 해결했다 → **M5/M6의 직접 근거.**

### M3 — S5-Extractor (3차, 현재 EACL) — **LLM enum 5축, 학습 없음. M4~M7의 anchor 공급원.**

**설계** (2-stage, `scripts/eacl_phase0/llm_s5_states.py`) — 상세는 `phase0-annotation-guide.md`:
- **Stage1** (이야기당 1회, 경계어 금지): cast/locations/goals **닫힌 목록** 추출
- **Stage2** (문장 단독, 축별 별도 호출): Stage1 목록 + 고정 enum에서만 선택
- 모델: `openrouter/anthropic/claude-fable-5`, temp 0, batch 20, shuffle seed=20260711+axis. **LLM frozen — 파라미터 학습 없음.**
- **리셋 판정 100% 코드.** coordinated count = 5축 리셋합(0~5). **t\*=3** (dev argmax-Score 동결).

**결과 (메인 Score, matched@gold; M3 / CosDrift-raw / CosDrift-tuned):**

| 데이터 | 유형 | K | M3 | CosDrift-raw | CosDrift-tuned | gain |
|---|---|---|---|---|---|---|
| pieman (dev) | fiction | 18 | **0.580** | 0.533 | 0.568 | +0.046 |
| tunnel (test) | fiction | 18 | **0.467** | 0.289 | 0.267 | **+0.178** |
| monkey (test) | journalism | 30 | 0.226 | **0.274** | 0.275 | −0.048 |
| tiage (test) | dialogue | 46 | 0.396 | **0.529** | 0.538 | −0.133 |

**축별 진단** (⚠️ ~~판별자 = place~~ → **D-M13⑤로 철회**, 표본 3~6건 + 문서별 재현성 없음):

| 축 | Tunnel(승) | Monkey(패) | tiage(패) |
|---|---|---|---|
| place | +0.28 **alive** | 死 (16/407) | +0.07 |
| causality | +0.44 **최강** | 약 (284/407 상시발화) | −0.04 |
| perspective | 死 | 약 | −0.14 |
| time | 死 | 死 | +0.09 |
| goal | 死 | 死 | +0.07 |

**판정: 조건부 Go — scope = narrative fiction.**
- 서사 픽션에선 CosDrift를 이김(Tunnel +0.178). 저널리즘·대화에선 짐.
- **이 실패는 사전 예측된 실패다.** 우리 전제("사건 경계 = 상황모델 축 리셋")가 성립하지 않는 텍스트(무대가 안 움직이는 다큐·대화)에서는 검출할 것이 없다. Monkey/tiage에서 place가 死한 것이 그 실증.
- **사전등록 메인(free-threshold Score, frozen t\*=3)에서는 3/3 패.** 원인 = t\*가 dev(96문장)에서 동결되어 긴 test 문서(504/408문장)에서 과예측(49개) → **작동점(operating point) 문제**. matched@gold는 이를 통제한 **사후 진단**임을 명시할 것. (결과를 "공정 렌즈로는 이겼다"로 뒤집지 말 것.)

---

## §3. M4~M7 — 전부 M3의 enum을 anchor로 학습

> **진행 상태 (2026-07-28):** M4/M5 = Phase 2 de-risk 통과 후 **본 학습 진행 중**(D-B16 1차 탐색 패스,
> 앵커 = 청정 10,598편/D-B15). M6/M7 = 미착수. 아래 본문의 "예정" 서술은 M6/M7 에만 해당한다.
>
> **🔴 Tier2 의 이중 지위 (어느 문서에도 안 적혀 있던 사실, 여기 기록):** Tier2 12편은 **held-out
> 으로는 폐기**됐지만(D-M13①) 그중 **7편이 M5 Phase 2 학습·val 데이터로는 계속 쓰였다**(train 6 +
> val 1 = `the_lady_with_the_dog`, D-M11 ①). Phase 2 REPORT 가 약축(space/time/intentionality)의
> 병목을 "아키텍처가 아니라 silver-label 품질"로 지목한 근거가 바로 그 silver 데이터다.
> ⇒ Phase 2 수치는 **오염 caveat 부착 필수**이며, 2026-07-28 본 학습은 이를 청정 앵커로 대체한다.

**공통 전제:**
- 인코더 **frozen** (M1~M3와 동일: `paraphrase-multilingual-mpnet-base-v2`, 768D, L2정규화) → 공정 비교 성립
- 학습 신호 = **자기지도(next-state 예측)** + **anchor(M3의 5축 enum 라벨)**
- **파인튜닝 아님.** 인코더 위에 얹은 작은 모듈만 처음부터 학습.

**설계 3축 (이 셋을 섞지 말 것):**
| 축 | 선택지 |
|---|---|
| **(가) 표현** | monolithic(1개) vs factored(5개) |
| **(나) 분해 방법** | 명시적 부분공간 vs slot-attention |
| **(다) 학습 신호** | 자기지도만 vs 자기지도+anchor |

| # | 이름 | (가) 표현 | (나) 분해 | (다) 신호 | dynamics | 역할 |
|---|---|---|---|---|---|---|
| M2 | MonoLAM | monolithic | — | 자기지도만 | 1개 | **완료·실패** |
| **M4** | **MonoAnchor** | monolithic | — | 자기지도+**anchor** | 1개 | **통제군** |
| **M5** | **FactoredSlot** | **factored 5** | 명시적 부분공간 | 자기지도+anchor | **축별 5개** | **★메인** |
| **M6** | **FactoredShared** | factored 5 | 명시적 부분공간 | 자기지도+anchor | **공유 1개** | FLAM식 |
| **M7** | **SlotAttnFactored** | factored 5 | **slot-attention** | 자기지도+anchor | 축별 5개 | M5 부족 시 |

**학습 없는 계열 (M1/M3 과 같은 rule-based — 위 3축 표에 해당 없음).** M4~M7 의 앵커 학습 계열과
혼동하지 말 것.

| # | 이름 | 무엇 | 상태 |
|---|---|---|---|
| M8 | **DirectBoundary** | 동결 인코더 + 이진 경계 헤드 직행 (학습 있음, 축 없음) | ablation — `scripts/eacl_phase0/train_direct_boundary.py` |
| M9 | **AttnEntityDrift** | LLM attention mass 상위 k 토큰 = 엔티티, 그 거리로 경계 | ⛔ **폐기 (D-B58)** — 전제 불성립 |

- **M9 (AttnEntityDrift)** — *"EM-LLM 의 surprisal 대신, LLM 이 집중하는 토큰을 엔티티로 뽑아
  그 사이 거리로 경계를 잡을 수 있나?"* → **아니다.** 사전등록 45셀 중 무경계 하한+노이즈를 넘은
  셀 0개, attention 선택의 측정 가능한 기여 없음(Δ 중앙 −0.021), 무작위 순위 대비 우위 없음
  (Δ 중앙 −0.018). 기전 = attention 상위 토큰의 40~50% 가 불용어·구두점이고 문장 첫 내용어에
  위치 편향. 상세 = D-B58 · `outputs/experiments/2026-07-31_m9_attn_entity/REPORT.md`.
  부산물로 **EM-LLM 의 dev-3 수치**를 처음 생산했다(raw surprisal 0.2711/0.2699 = 무경계 하한
  0.2790 **아래**, +modularity 0.2978).

### 각 방법이 답하는 질문

- **M4 (MonoAnchor)** — *"분리가 정말 필요한가?"*
  M2에 anchor만 추가한 것. **M5와의 유일한 차이가 (가) 표현 구조**이므로, M5−M4가 곧 **분리의 인과적 기여**.
  - M4 ≈ M5 → "분리 불필요, anchor가 전부였다" (novelty 약화, 그러나 정직한 결과)
  - M4 < M5 → **H1의 강한 지지** (분리가 인과적으로 기여)
  - **예상 위험:** monolithic은 "어느 축이 리셋됐는지 읽을 수 없다" → **H1(동시 리셋) 검정 자체가 불가능**. 성능이 비슷해도 M5가 필요한 이유.

- **M5 (FactoredSlot) — 메인** — 5개 부분공간, 각자 자기 dynamics, 각자 anchor.
  근거: FLAM(2026)이 monolithic LAM의 실패를 factored로 해결. BRICKS-WM은 slot-attention 없이 **명시적 부분공간 분해**로 충분함을 보임(agent/background 분리). Phase 0 데이터가 분리의 필요를 시사 — **Monkey에서 causality 284/407(상시발화) vs place 16/407(침묵)** → monolithic이면 causality 노이즈가 place 신호를 삼킴.

- **M6 (FactoredShared)** — 5개 부분공간이 **하나의 dynamics를 공유**.
  FLAM은 엔티티가 **교환 가능**(로봇 3대는 같은 법칙)해서 공유가 유리했다. **우리 5축은 질적으로 다르다**(인물의 변화 법칙 ≠ 시간의 변화 법칙) → 공유가 안 맞을 가능성이 높다. 단 **파라미터 효율·데이터 효율** 이점이 있어 검증 가치 있음.

- **M7 (SlotAttnFactored)** — slot-attention으로 분해.
  **주의:** slot-attention의 경쟁적 attention은 **"어느 슬롯이 뭘 담당할지 모를 때"** 필요한 장치다. 우리는 anchor로 담당을 이미 지정하므로 **불필요할 수 있다**. M5(단순 부분공간)가 부족할 때만 — 예: 문장 내 토큰 수준에서 "어디가 인물이고 어디가 장소인지"를 모델이 찾아야 할 때.

### 실행 순서 (문헌으로 거른 결과)

1. **M5 먼저 구현** (메인)
2. **M4로 대조** (분리의 기여 측정) ← 이 둘이 최소 필수 쌍
3. M6 (dynamics 공유가 이득인지)
4. M7 (M5가 부족할 때만)

**문헌이 거른 것:** monolithic + anchor 없음(M2) = 실패 확인. factored가 다중 factor 동시작동 환경에서 우세(FLAM/Dyn-O/BRICKS-WM 일치). monolithic은 해석 불가 → H1 검정 불가.
**문헌이 못 거른 것 (실험 필요):** M4(anchor+monolithic)의 실제 성능. dynamics 공유 vs 축별. **그리고 위 논문들은 전부 비전/RL이다 — 담화 축(인과·목표 같은 추상 차원)이 객체처럼 깔끔히 갈릴지는 미검증.** 이것이 최대 리스크.

---

## §4. 공통 실험 설정 (M1~M7 전부)

- **인코더:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, 768D, L2정규화, 다국어. **M1~M7 + baseline 전부 동일 → 공정비교 성립.**
- **채점기:** `Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)` (F1 binary exact; Pk/WD nltk window=max(2,round(len/(sum+1)/2)); 끝-turn 규약, 마지막 turn=0). AMI는 F1 ±2 tolerant, start-turn.
- **평가셋:**
  - **사건 경계(B, 서사):** Pieman(dev) 96문/18경계 · Tunnel 504/18 · Monkey 408/30(goldA)·40(goldB) — Michelmann author-filtered gold로 통일.
  - **대화/화제 경계 (scope 밖 — 실패 예측 검증용):** tiage_test, superseg_test, dialseg711_test, AMI.
- **메인 지표 = Score. 보조 = AUC.** 보고 순서는 항상 Score 먼저.

---

## §5. 미해결 / 정직 표기

- ⚠️ **mouthguard 수치 미검증:** proposal §1의 구체수치(내부 s≈0.75–0.77, 경계 0.81, margin 0.04)의 정확한 산출 run을 문서에서 찾지 못함. M1(CosDrift/V_rel) 계열 초기 관측으로 추정되나 스크립트/config 미확인. **논문 인용 전 재산출 필요.**
- **t\* 작동점 결함:** 정수 임계값이 문서 길이에 무관하게 적용되어 긴 문서에서 과예측. Phase 0의 구조적 한계. **M5의 sticky-CRP 지속성 사전(κ)이 이를 원리적으로 해결할 것으로 예상** — 단 이는 검증 대상이지 보장이 아님.
- **레거시 파일명:** `add_idm_scores.py` 등의 "idm"은 **CosDrift**를 의미(§0). 리네이밍 여부는 별도 결정.

---

# B 계열 — 외부 베이스라인 (우리 방법 아님, 이겨야 할 상대)

> **M 계열(우리 방법)과 구분해서 관리한다.** M은 "우리 방법의 진화", B는 "선행연구·비교 대상". 논문에서 역할이 다르다.
> ⚠️ **B 계열의 존재로 인해 "M1-CosDrift를 이겼다"는 매우 낮은 기준이 되었다.** 아래 참조.

## B1-BayesSurprise (Kumar et al. 2023, Cognitive Science)

- **방식:** GPT-2로 다음 단어 예측분포 → **Bayesian surprise**(믿음 갱신량) 시계열 → 경계.
- **핵심 결과:** 사건 경계는 **Bayesian surprise의 일시적 증가와 연관되나, 단순 예측오차(surprisal)와는 연관되지 않았다.**
- **⚠️ 우리와 같은 데이터셋:** Pieman(Michelmann), Tunnel(Lositsky 2016). **우리 held-out에 대한 선행연구가 이미 존재한다.**
- **왜 반드시 이겨야 하나:**
  - **B1 = 스칼라 서프라이즈의 최강 버전.** 우리 H1("스칼라로 못 잡는 걸 다축이 잡는다")의 **직접 시험대**다.
  - M1-CosDrift(코사인 거리, 학습·예측 없음)만 이겨서는 심사자가 "선행연구를 무시했다"고 본다.
  - B1은 **우리 이론적 조상**(EST 계산모델)이다.
- **M5 설계 함의:** B1이 **surprisal은 실패하고 Bayesian surprise만 성공**함을 보였다 → **M5를 순수 예측오차로 축소하면 안 된다.** 우도비/믿음갱신 형태(스위칭 사후분포)를 유지해야 한다.
- **✅ 재구현·검증 완료 (2026-07-15).** GPT-2-XL, 원문 수식(KL(Pt‖Pt-1) 연속위치 다음단어분포, Reynolds transient drift=0.10) verbatim 재현. **word 해상도 fidelity check 통과** — Bayesian surprise AUC **0.70** vs surprisal 0.52 = Kumar 시그니처 재현(구현 검증). 스크립트: `scripts/eacl_phase0/b1_bayes_surprise.py`(+`b1_kumar_regression.py`,`b1_deployable.py`). REPORT: `outputs/experiments/b1_bayes_surprise/REPORT.md`. (`kenmcgarry/BayesSurprise`는 무관 레포·미사용.)
  - **deployable 결과(dev, 탐색적):** 문장-전이 해상도에선 near-chance(AUC 0.53~0.56) — 문장 junction들이 서프라이즈상 동질이라 스칼라가 사건경계를 못 가림. **M3-S5(0.60~0.76) > M1 > B1** 3/3. **H1 강지지.** (버그 아님 — word-res 0.70과 대비되는 진짜 성질. 리뷰어 "baseline 약화" 반박 방어됨.) ~~정식 held-out 숫자는 Tier 2가 공급~~ → **철회(D-M13①, Tier2 폐기).** 현행 held-out = Tier1 6편이며 **2026-07-28 현재 미개봉**(D-B16).
- **✅ Monkey 결과 확인(원문):** B1은 **Monkey·Pieman에서 강, Tunnel에서 약**(drift=.10·window=200에서만) — **우리 M3와 정반대 패턴**(M3=Tunnel 승·Monkey 패). ⇒ scope 약화 아니라 **crossover 강화**: B1(스칼라 담화-서프라이즈)은 저널리즘(Monkey) 화제전환을 잡고, M3(상황모델 리셋)는 서사 픽션(Tunnel) 사건경계를 잡음. Monkey 경계 = 담화 서프라이즈 사건이라는 positive evidence.

## B2-LLM-Direct (Michelmann et al. 2023/2025, Behavior Research Methods)

- **방식:** LLM(GPT-3)에 **zero-shot 프롬프팅으로 직접 경계 추출.**
- **핵심 결과:** GPT-3 경계가 사람 주석과 유의하게 상관하며, **개별 인간 주석자보다 오히려 합의(consensus)에 더 가까웠다.**
- **코드:** `github.com/s-michelmann/GPT_event_segmentation` ✅
- **관련:** `github.com/s-michelmann/moment-by-moment-tracking` = Michelmann 2021 분석 스크립트 (**Pieman 사람 경계의 원소스**).
- **용도 2가지:**
  1. **사람-gold 데이터(Pieman/Monkey/Tunnel)에서 베이스라인**
  2. ~~**Tier 2 held-out의 silver gold 생성기**~~ — **실행 근거 소멸**(D-M13①로 Tier2 폐기). split-half 안정성 사실만 유지.
- ⚠️ **(2)로 쓴 데이터에서는 (1)로 쓸 수 없다 (순환).** 반드시 분리할 것.

## B3-EventRecall (Panela et al. 2025, Communications Psychology) — **최신·최강**

- **방식:** **GPT-4 + LLaMA-3**, 온도 0/0.5/1, **각 20회 반복**(사람 참가자 수와 동수). 같은 세 이야기.
- **핵심 결과:** **LLM이 인간보다 더 일관되게 사건 경계를 식별한다.**
- **코드:** `github.com/ryanapanela/EventRecall` ✅
- **검증된 zero-shot 프롬프트** (~~Tier 2에 그대로 사용 가능~~ — Tier2 폐기로 용처 소멸, 참고용 보존):
  > "An event is an ongoing coherent situation. The following story needs to be copied and segmented into large events. Copy the following story word-for-word and start a new line whenever one event ends, and another begins. This is the story: …"
  > (이야기 삽입 후) "This is a word-for-word copy of the same story that is segmented into large event units:"

---

# 논문 포지셔닝 — 무엇을 주장하고 무엇을 주장하지 않는가

> **핵심: 우리는 "가장 정확한 분절기"를 만드는 게 아니라, "인간의 사건분절 방식(EISM 5지표)이 계산적으로 성립하는가"를 검증하고, 그것이 "실시간으로 배포 가능한가"를 보인다.**
> 정확도는 **버리는 게 아니라 서브목표**다 — 되도록 확보하되, 주장의 무게중심은 아니다.

## 주 기여 (Primary Contributions)

### C1. EISM의 계산적 검증 — "인간 방식이 구현했을 때 통하는가"

인지과학의 **Event-Indexing Model**(인물·장소·시간·인과·목표 5지표 감시)이 실제 계산 시스템으로 구현했을 때 인간 사건 경계를 설명하는가? **그리고 언제 실패하는가?**

**이 프레임에서 기존 "실패"들이 전부 발견으로 전환된다:**

| 관측 | 성능 논문이면 | **이론 검증 논문이면** |
|---|---|---|
| Tunnel 승 | 성능 우위 | ✅ **EISM이 서사에서 작동** |
| Monkey·tiage 패 | 실패 | ✅ **이론의 경계 조건 발견** — 상황모델이 리셋되지 않는 텍스트(다큐·대화 화제전환)에서는 EISM 기반 분절이 성립하지 않는다 |
| ~~place가 판별자~~ | ~~축 하나가 중요~~ | ⛔ **D-M13⑤ 철회** — 경계 place 발화 3~6건뿐, 문서별 재현성 없음. 관찰 가능 결론 = "여러 축 동시 리셋 집계"까지 |
| time 축이 노이즈였음 (시제 뒤집힘 103/119 가짜) | 버그 | ✅ **"시제 변화 ≠ 시간 지표 변화"** — EISM 조작화에 대한 발견 |
| **M2(anchor 없는 monolithic) 실패** | 모델 실패 | ✅ **5지표를 명시적으로 강제하지 않으면 대규모 학습으로도 발견되지 않는다** — EISM 구조가 데이터에서 저절로 창발하지 않는다는 실증 |

**B1(Kumar)의 역할도 이 프레임에서 재정의된다:** Kumar = **스칼라 서프라이즈(EST 계열)**. 우리 = **다축 리셋(EISM 계열)**.
⇒ **이건 성능 비교가 아니라 두 인지 이론의 계산적 대결이다.** "어느 이론이 인간 분절을 더 잘 설명하는가."
⇒ **져도 과학적 결론이 된다** ("EST가 EISM보다 설명력이 높다"). 하지만 반드시 **돌려야** 한다.

### C2. 실시간 배포 가능성 — LLM-free 온라인 분절

**B2/B3(GPT 직접 프롬프팅)가 원리적으로 못 하는 지점이다.**

| | B2/B3 (LLM 직접 프롬프팅) | **우리 (M5-FactoredSlot)** |
|---|---|---|
| 입력 | **전체 텍스트를 통째로** | **문장 스트리밍** |
| 추론 시 LLM | **매번 API 호출** | **없음** (증류 후 슬롯만) |
| 지연 | 이야기가 끝나야 분절 | **온라인**(K'문장 lookahead) |
| 해석 | 블랙박스 | **어느 축이 리셋됐는지 읽힘** |
| **인지적 타당성** | ❌ 사람은 오프라인 분절 안 함 | ✅ **인간처럼 실시간 분절** |

**C1과 C2는 서로를 강화한다:** 인간은 **실시간으로** 분절한다. 따라서 **실시간 구현 자체가 인지적 타당성의 일부**다. 오프라인 GPT-4 프롬프팅은 애초에 인간 인지의 모델이 아니다.

> **한 줄 주장: "인간처럼 5지표를 추적하고, 인간처럼 실시간으로 분절한다."**

## 서브 목표 (Secondary — 되도록 확보, 시간 부족 시 후순위)

### S1. 경쟁력 있는 정확도
- **M1-CosDrift는 반드시 이긴다** (이미 6/7 dev에서 달성).
- **B1-BayesSurprise는 이겨야 한다** — 정확도 때문이 아니라 **H1(스칼라 vs 다축)의 직접 검정**이므로. **우선순위 높음.**
- **B2/B3(GPT 직접)는 이기면 좋지만 필수 아님.** 정확도만으로 GPT-4와 겨루면 불리하다. **topline/oracle로 보고**하고, "**LLM 없이 그 상한에 얼마나 근접하는가**"를 보인다.
- 시간이 허락하면 SOTA를 노린다. 시간이 부족하면 C1·C2를 우선한다.

## 리뷰어 예상 질문과 답

**Q: "그냥 GPT-4에 프롬프팅하면 되는데 왜 5축을 세나?"**
A: (1) GPT-4는 전체 텍스트가 필요해 **실시간 불가**, 매 추론마다 API 호출. 우리는 **LLM 없이 문장 스트리밍**. (2) GPT-4는 **어느 지표가 리셋됐는지 말해주지 않는다** — 우리는 읽힌다. (3) 애초에 우리 질문은 "**인간 인지 이론이 계산적으로 성립하는가**"이며, GPT-4 프롬프팅은 그 질문에 답할 수 없다.

**Q: "정확도가 SOTA가 아닌데?"**
A: 주 기여는 이론 검증과 실시간 배포다. 정확도는 **B1(스칼라 서프라이즈)을 상회**함으로써 H1을 지지하는 수준에서 의미를 갖는다. B2/B3는 오프라인 topline으로 보고한다.

**Q: "Monkey/대화에서 실패했잖아?"**
A: **사전 예측된 실패**다(DTS에서 안 될 것이라 예측 → held-out에서 실증). 상황모델이 정지한 텍스트에서 EISM이 성립하지 않음을 보인 것은 **이론의 경계 조건에 대한 발견**이다. **단, Monkey는 주석 프로토콜 교란 가능성이 있어 4구간 진단이 선행되어야 한다** (ledger D-M5-d).

---

# 베이스라인 사용 규칙 (순환 방지)

| 데이터 | gold 출처 | 사용 가능 베이스라인 |
|---|---|---|
| Pieman / Monkey / Tunnel | **사람 주석** | B1, B2, B3, M1, M2 전부 ✅ |
| ~~Tier 2 (새 서사, LLM-silver gold)~~ ⛔ **폐기**(D-M13①) | — | 표는 역사 기록. **단 7편은 M5 Phase 2 학습에 소비됨**(§3 이중 지위 참조) |
| Tier 3 (한국어 썰) | 사람 주석 (신규) | 전부 ✅ |
