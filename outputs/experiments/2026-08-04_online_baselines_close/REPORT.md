# 온라인 베이스라인 미착수분 마무리 (2026-08-04 KST)

`EACL/UNATTENDED_2026-08-03_online_baselines.md` 의 **미착수 6건 + 등급 미확인 8행**을 닫은 기록.
결정 저널은 같은 파일의 D-10 ~ D-16 에 append 되어 있다.

**결론 먼저**
1. 표에 `?` 로 남아 있던 **등급 8행을 전부 실측으로 확정**했다. 이름이 실제와 다른 게 2건
   (`GreedySeg` 실측 d=7 · `GraphSeg` 실측 d=6) 나왔다.
2. **D-07 이 닫혔다** — surprise LM 을 Llama 로 바꿔도 EM-LLM 은 무경계 하한 아래다. 오히려 더 나쁘다.
3. **DialSTART 를 서사에 이식**했다(개조 3건 명시). dev3 0.3295 로 **3위** — 대칭 실험의
   "DTS 방법은 서사에서 부진하다" 가설은 **dev3 에서 성립하지 않는다**.
4. **축별 붕괴 분석 결과가 appendix 주장과 다르다** — space 는 붕괴하지 않고, 역상관은 time 에서 난다.
5. **논문의 EM-LLM 문단 주장이 실험 결론과 반대**여서 정정했다.
6. API 비용 **$0** (한 건도 호출하지 않았다).

---

## 1. 실험 setup

| 항목 | 값 |
|---|---|
| 평가셋 | dev3 = pieman·tunnel·monkey (전이 1,005 / gold 66) · test3 = run·gohitler·mymotherslife (전이 432 / gold 43) |
| 데이터 | `outputs/eacl_phase0/{doc}_michelmann_master.jsonl` (dev3) · `{doc}_eventrecall_goldtopn_master.jsonl` (test3) |
| 지표 | Pk · WindowDiff · F1 · **Score**(주지표) = `hi_ontop.dts_scoring.score_dialogues` |
| 모드 | **deployable** — 경계 수 자가결정, gold 개수 미사용 |
| 하한 | 무경계 dev3 0.2790 · test3 0.3092 / 밀도일치 Random dev3 0.2860 · test3 0.3008 |
| 표 러너 | `EACL/methods/common/table4.py` → `outputs/experiments/2026-08-03_narrative_online/table4_6docs.json` |
| GPU | 로컬 A100 80GB ×4 (단, 이번 작업 내내 4장 모두 `/root/emllm` LongBench 런이 점유) |

---

## 2. 결과

### 2.1 온라인 등급 확정 (P0)

러너 `EACL/methods/common/audit_lookahead.py` → `lookahead_audit.json`

**측정 규약.** push 기반 분절기는 구조적으로 미래를 못 본다(아직 push 되지 않았으니까).
그래서 기존 `verify_causality.py` 의 미래-교란 검사는 이들에게 자명하게 통과해 **증거가 되지
못한다.** 실제 쟁점은 지연이다 — `push(u_t)` 가 경계 `b` 를 방출하면 전이 `b−1` 의 판정에
`u_b..u_t` 를 썼다는 뜻이므로 `delay = t − b`. 6편 전체 최대값을 `d` 로 박았다.

| 행 | 문서/이름상 | **실측** | 방출 수 | 근거 |
|---|---|---|---:|---|
| `TextTiling-online-stream` | bounded (d 미상) | **bounded d=22** | 169 | 실측 |
| `GraphSeg` | ? (`window_d=10`) | **bounded d=6** | 399 | 실측 |
| `GreedySeg` | ? (클래스명 `GreedySegOnlineDelay2`) | **bounded d=7** | 299 | 실측 |
| `CSM` | ? | **bounded d=2** | 276 | 실측 |
| `RoBERTa` | ? | **zero** | — | `segment.py::predict_online` 의 `wu = utts[ws:t+1]` |
| `Membox-TopicLoom` | ? | **zero** | — | `preds_membox.run` 창 = 직전 2문장 + 현재 |
| `B3event`(EventRecall) | ? | **zero** | — | `build(sents, cur=j+1, past=0)` |
| `SeCom-binary*` | ? | **zero** | — | 같은 러너 |

🔴 **이름이 실제와 다른 2건.** `GreedySegOnlineDelay2` 는 delay=2 를 암시하지만 실측 **d=7**
(`_try_finalize_segment` 가 후보를 누적하다 한꺼번에 확정). `GraphSeg` 는 `window_d=10` 인데
실측 **d=6**. D-03(TextTiling 이 실은 bounded 였던 사건)과 같은 종류의 함정이다.

🔑 **`TextTiling-online-stream` 의 d=22 가 D-03 의 격차를 설명한다.** dev3 에서 zero 0.2647
→ bounded 0.3212 로 +0.0565 뛰는 이유가 "22문장 미래" 다. 표에 `bounded` 라고만 적으면
독자가 d=2 급을 상상한다 — **d 를 숫자로 적어야 한다.**

### 2.2 주 표 — 6편 · 4지표 · deployable

**dev3** (gold 66 · 무경계 하한 0.2790)

| 방법 | 등급 | Pk↓ | WD↓ | F1↑ | **Score↑** | 예측 |
|---|---|---:|---:|---:|---:|---:|
| Hi-OnTop | online(zero) | 0.3847 | 0.4994 | 0.2800 | **0.4190** | 67 |
| EventRecall | online(zero) | 0.3635 | 0.3861 | 0.1764 | 0.4008 | 21 |
| **DialSTART (DTS→NES)** | online(bounded d=2) | 0.4722 | 0.6286 | 0.2095 | **0.3295** | 172 |
| CSM | online(bounded d=2) | 0.5180 | 0.7067 | 0.1884 | 0.2880 | 194 |
| *Random (밀도일치)* | online(zero) | 0.4682 | 0.5412 | 0.0767 | *0.2860* | 67 |
| Embedding Delta | online(bounded d=25) | 0.4534 | 0.4855 | 0.0316 | 0.2811 | 78 |
| RoBERTa | online(zero) | 0.5002 | 0.6710 | 0.1455 | 0.2799 | 151 |
| *무경계 하한* | — | — | — | — | *0.2790* | 0 |
| Fixed interval (밀도일치) | online(zero) | 0.5571 | 0.5957 | 0.1276 | 0.2756 | 67 |
| EM-LLM | online(zero) | 0.4771 | 0.5935 | 0.0751 | 0.2699 | 116 |
| BayesSurprise | online(zero) | 0.5105 | 0.6865 | 0.1308 | 0.2661 | 185 |
| TextTiling | online(zero) | 0.5127 | 0.5899 | 0.0807 | 0.2647 | 99 |
| Membox-TopicLoom | online(zero) | 0.5404 | 0.8349 | 0.1555 | 0.2339 | 396 |
| TextTiling | offline(원본) | 0.5736 | 0.7226 | 0.1037 | 0.2278 | 160 |
| GraphSeg | online(bounded d=6) | 0.5500 | 0.8096 | 0.1222 | 0.2212 | 285 |
| GreedySeg | online(bounded d=7) | 0.5702 | 0.7858 | 0.0756 | 0.1988 | 206 |

**test3** (gold 43 · 무경계 하한 0.3092)

| 방법 | 등급 | Pk↓ | WD↓ | F1↑ | **Score↑** | 예측 |
|---|---|---:|---:|---:|---:|---:|
| EventRecall | online(zero) | 0.3006 | 0.3589 | 0.1317 | **0.4010** | 18 |
| RoBERTa | online(zero) | 0.3751 | 0.5031 | 0.1517 | 0.3563 | 61 |
| Embedding Delta | online(bounded d=25) | 0.3956 | 0.4517 | 0.1122 | 0.3443 | 40 |
| CSM | online(bounded d=2) | 0.5322 | 0.6483 | 0.2474 | 0.3286 | 82 |
| TextTiling | offline(원본) | 0.4527 | 0.5739 | 0.1434 | 0.3150 | 84 |
| BayesSurprise | online(zero) | 0.4599 | 0.5219 | 0.1186 | 0.3138 | 54 |
| TextTiling | online(zero) | 0.4161 | 0.4597 | 0.0580 | 0.3100 | 28 |
| **DialSTART (DTS→NES)** | online(bounded d=2) | 0.5285 | 0.6059 | 0.1869 | 0.3098 | 79 |
| *무경계 하한* | — | — | — | — | *0.3092* | 0 |
| *Random (밀도일치)* | online(zero) | 0.5307 | 0.6085 | 0.1712 | *0.3008* | 72 |
| Fixed interval (밀도일치) | online(zero) | 0.5573 | 0.6248 | 0.1536 | 0.2813 | 72 |
| EM-LLM | online(zero) | 0.5251 | 0.5728 | 0.1058 | 0.2784 | 56 |
| GraphSeg | online(bounded d=6) | 0.5927 | 0.7243 | 0.1714 | 0.2564 | 114 |
| GreedySeg | online(bounded d=7) | 0.5726 | 0.7006 | 0.1478 | 0.2556 | 93 |
| Membox-TopicLoom | online(zero) | 0.5983 | 0.8054 | 0.1230 | 0.2106 | 140 |

🔴 **Hi-OnTop 의 test3 행이 주 표에 없다** — `M5` 의 test3 덤프가 없고 `M5-adaptive` 만 있어
부록으로 빠져 있다. 논문 표 확정 전에 `M5` 를 test3 에 돌려 채워야 한다(미착수, 아래 §5).

### 2.3 D-07 닫힘 — EM-LLM surprise LM 재계산 (P1)

러너 `EACL/methods/emllm/compute_surprise.py`(신규, LM 인자화) → `run_sm.py --cache-tag`.
분절 함수는 `segment_manual_emllm.py::encode_doc` **그대로 재사용**, γ=1.1 · n_local=4096 고정,
토크나이저는 각 LM 자신의 것.

| surprise LM | 성격 | dev3 Score | test3 Score |
|---|---|---:|---:|
| `Qwen/Qwen2.5-7B` (기존) | base | 0.2699 | 0.2784 |
| `meta-llama/Llama-3.1-8B` | base | **0.2511** | **0.2425** |
| `meta-llama/Llama-3.1-8B-Instruct` | instruct | **0.2230** | **0.2371** |
| *무경계 하한* | — | *0.2790* | *0.3092* |

**세 LM 모두 두 무대에서 무경계 하한 아래.** Llama 계열이 오히려 Qwen 보다 낮다.
⇒ *"LongBench 에선 재현되는데 서사에선 하한 아래"* 가 **모델 차이 때문이라는 가설은 기각**된다.
EM-LLM 원 논문·우리 LongBench 재현이 바로 Llama-3.1-8B-Instruct 인데 **그 모델에서 가장 나쁘다**
는 점이 결정적이다.

**GPT-5 는 물리적으로 불가.** surprise 는 토큰별 `−log P` **와** `n_local=4096` KV 창의
hidden/attention 을 함께 요구한다(`encode_doc` 이 `repr_per_layer` 를 층별로 수집). 호스티드
API 는 hidden state 를 노출하지 않고 gpt-5 는 logprobs 도 주지 않는다.
**GPT-OSS-120B 는 가능**하나 로컬 캐시(`models--openai--gpt-oss-120b`)가 512바이트 스텁이라
~63GB 다운로드가 선행돼야 한다 — 미착수(§5).

### 2.4 DialSTART NES 이식 (P2) — D-09 의 개조를 구현

러너 `EACL/methods/dialstart/run_narrative_online.py` → `outputs/experiments/2026-08-04_dialstart_nes/`
가중치 = 저자 배포본 `model_711` (847MB, dialseg711 학습본). 점수 함수·토크나이즈·`window_size`·
가중치는 **저자 것 그대로**. 바뀐 건 readout 뿐이다.

| # | 원본 | 이식판 | 이유 |
|---|---|---|---|
| ① | top-k 고정 (`test.py:137` `argsort(...)[-pick_num:]`, 기본 4) | **running μ + c·σ** | 문서 길이 무관 고정 개수는 deployable 규약 위반. 서사는 전이 95~503 · gold 13~30 이라 4개는 심한 과소예측 |
| ② | depth 오른쪽 봉우리 = 문서 끝까지 스캔 | `depth_span=1` 전이까지만 | 원본 `depth_score_cal` 은 오프라인이다 |
| ③ | — | 위 둘의 결과로 총 지연 **d=2** | `window_size=2` 의 우측 1문장 + depth_span 1 |

**c 동결**: dev3 에서만 스윕(0.0~2.0, 8점) → **c\* = 1.0** (dev3 Score 0.3295 최대).
test3 는 그 값을 그대로 적용, 재선택하지 않았다.

| ds | Pk | WD | F1 | Score | 예측/gold |
|---|---:|---:|---:|---:|---:|
| dev3 | 0.4722 | 0.6286 | 0.2095 | **0.3295** | 172/66 |
| test3 | 0.5285 | 0.6059 | 0.1869 | 0.3098 | 79/43 |

🔴 **원 논문 수치와 대응시킬 대상이 없다.** 원 논문은 대화(dialseg711/doc2dial)에 top-k readout
으로 Pk/WD 를 보고한다. 이 행은 "대화용 SotA 를 서사에 zero-shot 으로 옮기면 어떻게 되는가" 라는
**대칭 실험의 한쪽**이며, 개조 3건을 반드시 각주로 밝혀야 한다.

### 2.5 DTS ↔ NES 대칭 실험 (P3)

appendix 주장 *"두 태스크는 구분된다"* 의 양방향 근거.

**DTS → NES** (대화용 방법을 서사 6편에)

| 방법 | 출신 | dev3 | test3 | vs 무경계 하한 |
|---|---|---:|---:|---|
| DialSTART (SotA, 비지도) | DTS | **0.3295** | 0.3098 | dev3 **+0.051** / test3 +0.001 |
| RoBERTa (지도, SuperDialseg 학습) | DTS | 0.2799 | 0.3563 | dev3 +0.001 / test3 **+0.047** |
| TextTiling-online (고전, 도메인 중립) | 공통 | 0.2647 | 0.3100 | dev3 −0.014 / test3 +0.001 |

🔴 **"DTS 방법은 서사에서 일률적으로 부진하다" 는 성립하지 않는다.** DialSTART 는 dev3 에서
3위(하한 +0.051), RoBERTa 는 test3 에서 2위(하한 +0.047)다. 각 방법이 **한쪽 무대에서만**
하한을 확실히 넘고 다른 쪽에서는 하한 수준으로 떨어지는 **불안정성**이 관측되는 것이지,
일률적 부진이 아니다. 대칭 실험의 주장을 이 사실에 맞춰 다시 써야 한다.

**NES → DTS**: `scripts/eacl_phase0/m4m5_dts_transfer.py` 산출
(`outputs/experiments/m4m5_phase1_anchor/M4M5_DTS_TRANSFER.md`) 재사용. 🔴 DTS 는 D-M5 로
scope-out 이므로 절대값을 인용하지 않는다.

🔴 **harness 혼용 금지.** 위 표는 전부 하네스 A(우리 팩 + `score_dialogues`)다. 하네스 B/C
수치(예: TextTiling tiage 0.3628)와 한 표에 섞으면 안 된다 — 같은 방법이 B→C 로 옮기면
0.3081 → 0.3628 로 움직인다.

### 2.6 축별 붕괴 분석 (P4) — 🔴 appendix 주장과 다르다

러너 `scripts/eacl_phase0/dump_axis_probs.py`(신규) · 그림 `plot_axis_collapse.py`
→ `EACL/figures/figure_02_axis_collapse.{pdf,png}` · 데이터
`outputs/experiments/2026-08-04_axis_collapse/axis_probs_{set}.jsonl` (전이 단위 5축 확률)

축별 확률은 **그동안 디스크에 한 번도 남은 적이 없었다** — `axis_probs()` 를 부르는 13개
호출부가 전부 `[n_trans,5]` 를 즉시 `.sum(1)` 로 붕괴시킨다.

AUC / Δmean (= mean p|gold=1 − mean p|gold=0):

| 축 | dev3 | test3 | TIAGE | DialSeg711 | SuperDialseg |
|---|---:|---:|---:|---:|---:|
| protagonist | **0.688** / +0.111 | 0.578 / +0.049 | 0.517 / +0.005 | 0.589 / +0.022 | 0.527 / +0.007 |
| space | 0.652 / +0.063 | 0.691 / +0.083 | 0.529 / −0.002 | **0.671 / +0.141** | 0.566 / +0.006 |
| time | 0.688 / +0.061 | 0.588 / +0.012 | 0.599 / +0.006 | **🔴 0.425 / −0.003** | 0.552 / +0.006 |
| causality | 0.623 / +0.055 | 0.509 / +0.005 | 0.512 / +0.003 | 0.535 / +0.010 | 0.543 / +0.013 |
| intentionality | 0.576 / +0.030 | 0.537 / +0.016 | 0.524 / +0.005 | **0.694 / +0.091** | 0.529 / +0.003 |

사용자 주장 *"protagonist 와 space 는 대화에서 정의가 붕괴한다"* 를 데이터가 **절반만** 지지한다:

1. ✅ **protagonist 는 붕괴한다.** dev3 Δ+0.111 → TIAGE +0.005 · SuperDialseg +0.007 (사실상 0).
2. 🔴 **space 는 붕괴하지 않는다.** DialSeg711 에서 AUC 0.671 · Δ**+0.141** 로 **dev3 보다 강하다.**
   DialSeg711 이 서로 다른 대화를 이어붙인 합성 데이터라 화제 전환이 곧 장면 전환처럼 보이기
   때문으로 읽힌다. 이 문장을 그대로 쓰면 반증당한다.
3. 🔴 **역상관(AUC<0.5)이 나는 유일한 축은 `time`** (DialSeg711 0.425). 주장에 없던 축이다.
4. 🔴 **TIAGE·SuperDialseg 에서는 5축이 전부 붕괴한다**(AUC 0.51~0.60, Δ≤+0.013).
   "특정 두 축의 문제" 가 아니라 **도메인 전체의 문제** — 이게 더 강하고 방어 가능한 주장이다.

**권고 문구**: *"TIAGE·SuperDialseg 에서는 다섯 축이 모두 붕괴하고, DialSeg711 에서만
space·intentionality 가 살아남는데 이는 그 코퍼스가 서로 다른 대화를 이어붙인 합성
데이터이기 때문이다. protagonist 는 세 대화 코퍼스 모두에서 붕괴한다."*

🔴 **기존 `2026-07-31_channel_observability` 의 축별 AUC 를 인용하지 말 것.** 그건 작은 `dts_*`
서브셋이라 protagonist 가 TIAGE 0.428 · SuperDialseg 0.394 로 **역상관처럼** 보였는데, 공식 test
전체(`dts_full_*`)에서는 0.517 · 0.527 로 **역상관이 아니라 평탄**이다. 표본이 작아 과장된
수치였다. 인용은 `dts_full_*` 판으로 통일한다.

### 2.7 논문 EM-LLM 문단 정정 (P6a)

`EACL/EACL_Paper.tex` 의 문단 제목이 *"Replacing the segmentation module with Hi-OnTop
**improves** downstream memory performance"* 였는데, 그 실험을 실제로 돌린 `/root/emllm` 의
결론은 정반대였다:

| 근거 | 값 |
|---|---|
| LongBench Δ | **+0.06, 95% CI [−0.32, +0.43]** (`DECISION_JOURNAL.md:1507`) |
| 저널 결론 | *"어떤 M5 변종도 EM-LLM_S 를 유의하게 넘지 못했다"* (`:1506`) |
| ∞-Bench 6태스크 평균 | M5 **64.34** vs surprisal **66.93** (kv_retrieval 86.60 vs 97.80) |
| NarrativeQA +1.79 | 95% CI [−0.12, +3.89] — 0 을 걸침 |
| D-L13 | 7개 M5 변종 중 최대치 선택은 post-hoc selection |

[[no-postresult-criterion-change]] 대로 **결과에 맞춰 문구를 고쳤다.** 새 주장 =
*"Hi-OnTop matches surprisal-based segmentation on downstream memory without computing
token-level LLM surprisal"* (동등성 + 계산 절감: 5.7M 파라미터 분절기 vs 8B 백본의 토큰별
surprisal). 두 caveat(post-hoc selection · ∞-Bench 역전)을 본문에 명시했다.
덤으로 `.tex` 안에 **Markdown 표가 그대로 붙어 있어 컴파일되지 않던 것**을
`table*`(LongBench 15태스크) + `table`(∞-Bench) 로 변환했다.

### 2.8 재현불가로 닫은 항목 (P5)

사용자 결정에 따라 실행하지 않고 사유를 기록한다. [[no-external-data-requests]] 대로 저자
접촉·라이선스 요청은 하지 않았다.

| 방법 | 사유 | 근거 파일 |
|---|---|---|
| Kurfalı SSC (STSS 1위) | 저자 체크포인트 404 — `Dockerfile:68` 의 gdown ID `1yayKtOT2pGD7YQpL-r9p3D-ErMt6rVeR` → HTTP 404. 대체 ckpt 없음, HF 에도 없음 | `EACL/methods/ssc_kurfali/REPRO.md` §2 |
| BERT-SSC / half-stride (2025 SotA) | 학습 코퍼스 = 독일어 Heftroman 32편, 저작권물·비배포. 공개도서는 6편뿐이라 논문 모델이 아님 | `EACL/methods/ssc_gbert_2025/REPRO.md` §2 |
| BERT-Context (2021b) | 동일 코퍼스 + ckpt 없음 | `EACL/methods/bert_sentence_pair/REPRO.md` §3 |
| TSE (Barth) | GermaNet 라이선스 | `EACL/methods/tse_barth/REPRO.md` |
| LLM-Direct (B2, Michelmann) | 원 논문 모델 `text-davinci-002` 폐기 — 대조할 수치가 없다 | `EACL/methods/llm_direct/segment.py` |

본문에서는 half-stride 의 논거(*"단측 문맥으로는 경계를 신뢰성 있게 판정할 수 없다"*)를
**인용만** 하고 수치는 싣지 않는다. 우리 앵커로 학습하는 대안은 [[baselines-stay-as-published]]
위반(=베이스라인이 아니라 우리 방법 변형)이므로 채택하지 않았다.

---

## 3. API 비용 정산

| 항목 | 판정 |
|---|---|
| EventRecall 재실행 | **불필요** — gpt-4.1 reps=10 덤프 6편이 이미 있다. 재실행 시 55.8M 입력 tok ≈ $112 |
| LLM-Direct (B2) | **스킵** (사용자 결정) |
| SSC-2025 prompting | **미채택** (사용자 결정) |
| **총 API 호출** | **0회 · $0** |

---

## 4. 해석

1. **하한이 여전히 대부분을 걸러낸다.** dev3 에서 무경계 하한(0.2790)을 확실히 넘는 것은
   Hi-OnTop · EventRecall · DialSTART 셋뿐이다. test3 에서는 EventRecall · RoBERTa ·
   Embedding Delta · CSM 넷. **두 무대 모두에서 넘는 방법은 EventRecall 하나다.**
2. **LLM 신호 계열(EM-LLM · BayesSurprise)이 독립적으로 하한 근처에 모인다.** D-11 로
   EM-LLM 이 LM 3종에서 모두 하한 아래임이 확인돼, 단일 구현 아티팩트가 아니라는 근거가 강해졌다.
3. **lookahead 는 공짜가 아니다.** TextTiling zero(0.2647) → bounded d=22(0.3212). 22문장을
   더 보면 +0.057 이다. 반대로 CSM(d=2) · DialSTART(d=2) 는 작은 지연으로 하한을 넘는다 —
   **"미래를 얼마나 보는가"가 성능을 좌우하므로 d 없는 bounded 표기는 무의미하다.**
4. **대칭 실험의 서사가 예상과 다르다.** DTS 방법이 서사에서 일률적으로 부진한 게 아니라,
   각각 한쪽 무대에서만 하한을 넘는 **불안정성**을 보인다. "두 태스크가 구분된다" 는 주장을
   이 형태로 다시 써야 한다.

---

## 5. 한계 / 미해결 (솔직 기록)

| # | 항목 | 상태 |
|---|---|---|
| 1 | **Hi-OnTop 의 test3 행 없음** | `M5` test3 덤프 부재 — `M5-adaptive` 로 대체돼 부록에 있다. 논문 표 확정 전 필수 |
| 2 | **P6(b) NES 베이스라인 → EM-LLM splitter** | 🔴 **미실행.** 로컬 A100 4장이 작업 내내 `/root/emllm/benchmark/pred.py --config config/m5_paper.yaml` 8프로세스(장당 ~60GB)로 점유돼 착수 불가. 그 런은 이전 세션 것이라 죽이지 않았다. 예상 비용 arm 당 6.4 GPU-h × 3 arm ≈ 20 GPU-h(로컬 4장 ~5h wall, $0) |
| 3 | **∞-Bench En.QA** | 🔴 미실행 — 사전등록(D-I16)만 돼 있다. 4조건 ≈ 150 GPU-h/조건, VESSL 환산 ≈ $930. **별도 승인 필요** |
| 4 | **GPT-OSS-120B surprise** | 미실행 — 로컬 캐시가 512바이트 스텁, ~63GB 다운로드 승인 필요 |
| 5 | **dev3 = 3편** | c\* 동결 표본이 3문서뿐이다. DialSTART 의 c 스윕에서 c=1.0~2.0 이 0.3254~0.3295 로 거의 평평해 선택이 불안정할 수 있다 |
| 6 | **DialSTART 개조 3건** | 원본과 다른 방법이다. 원 논문 수치와 대응 불가 — 각주 필수 |
| 7 | **Llama base vs Instruct** | Qwen2.5-7B 는 base, 대조로 Llama base·Instruct 를 둘 다 돌려 셋 다 하한 아래임을 확인했다. 다만 EM-LLM 의 similarity refinement 는 이 SM 변형에서 제외돼 있다(원 설정 그대로) |
| 8 | **축별 확률의 신뢰구간 없음** | AUC·Δmean 점추정만 있다. 부트스트랩 CI 는 미산출 |
| 9 | 🔴 **`.tex` 에 Markdown 표가 하나 더 남아 있다** | `EACL_Paper.tex:266-270`(코퍼스 통계, 권/편수/문장/저자)도 Markdown 그대로라 컴파일되지 않는다. 이번에 고친 EM-LLM 표와 **같은 종류의 결함**이지만 요청 범위 밖이라 손대지 않았다 |
| 10 | 🔴 **`figure_02` 가 본문에서 참조되지 않는다** | 파일은 만들었으나 `.tex` 에 `\includegraphics`/`\ref` 가 없다. §2.6 의 권고 문구와 함께 삽입해야 한다 |
| 11 | 🔴 **`en_core_web_lg` 가 어느 venv 에도 없었다** | `EmbeddingDelta-online-d25` 행은 2026-08-03 21:47 에 나왔는데(그때는 있었다는 뜻), 지금은 4개 venv 어디에도 없다 — 컨테이너 재생성으로 소실된 것으로 보인다. **그 행은 현재 재현 불가**이며 재설치가 필요하다(이번에 재설치함). 같은 이유로 소실될 수 있는 것: DialSTART 가중치 847MB·CSM 체크포인트 99GB (둘 다 gitignored 트리) |

## 6. 환경 함정 (재사용 시 주의 — 이번에 다시 밟은 것 포함)

1. `device_map="auto"` + `encode_doc` → GPU 간 텐서 혼합 크래시. **단일 GPU 필수** (D-05).
2. `baseline_compare` 는 import 만으로 **torch·sklearn·train_anchor_m4m5** 를 끌어온다.
   `.venv-dialstart` 에는 sklearn 이 없어 죽었다 → DialSTART 러너에 **경량 로더를 직접 뒀다**.
3. `nltk` 가 **cwd 에서의 import 를 차단**한다 — 중립 cwd 에서 실행할 것.
4. 🆕 `GraphSegWindowD.glove_path` 기본값이 **상대경로**(`benchmarks/glove/...`)라 중립 cwd 에서
   `FileNotFoundError` 가 난다. 절대경로로 넘겨야 한다.
5. 🆕 GPU 를 다른 작업과 공유하면 Llama-8B 가 `[T,V]` fp32 logits 사본(V=128256, T=4900 → 2.5GB)
   에서 OOM 난다. `encode_doc(..., nll_chunk=512)` 로 우회하며, 수치는 비트 단위로 동일함을
   pieman 에서 확인했다(`max|Δsurprise| = 0.000e+00`).
