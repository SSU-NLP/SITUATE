# EACL/methods — 외부 방법 재구현·재현 검증

이 폴더의 목적은 하나다. **RelatedWork 에 올릴 11개 외부 방법을 우리 손으로 구현하고,
각 논문이 발표한 수치를 재현할 수 있는지 판정한다.**

- `_upstream/` — 원저자 공개 코드 클론(수정 금지, 참조 전용)
- `common/` — 여러 방법이 공유하는 채점기·데이터 로더
- `<method>/` — 방법별 우리 구현 + 재현 검사 스크립트 + `REPRO.md`

규칙: [[baselines-stay-as-published]] — **발표된 구성 그대로** 돌린다. 우리 데이터로
재학습·하이퍼튜닝하지 않는다. 재현이 안 되면 안 된다고 적지, 기준을 바꾸지 않는다.

---

## 0. 결론 먼저 (2026-08-03 작업 완료 시점)

| # | 방법 | 폴더 | 착수 전 상태 | **최종 판정** |
|---|---|---|---|---|
| 1 | **STSS 공식 채점기** (Zehe+ 2021a) | `common/` | 없음 | ✅ **재현** — 공식 표와 정확 일치 |
| 2 | **2025 완화 F1 · IoU** (Zehe+ 2025) | `common/` | 없음 | ✅ **재현** — 저자 코드와 2,000케이스 비트 일치 |
| 3 | **BayesSurprise** (Kumar+ 2023) | `bayes_surprise/` | 각색본만·미대조 | ✅ **재현** — 핵심 대조 5/5, r 차이 ≤0.012 |
| 4 | **TextTiling** (Hearst 97 / Jiang+ 23) | `texttiling/` | **판정 없음** | ✅ **재현** — 3/3 셋, 2개는 소수 4자리 일치 |
| 4b | **DialSTART** (Gao+ 2023, DTS SotA) | `dialstart/` | **구현 없음** | ✅ **재현** — 2/2 셋 · 4/4 지표 소수 4자리 일치 |
| 5 | Embedding Delta (Schneider+ 2021) | `embedding_delta/` | 없음 | 🟡 **부분 재현** — 저자 제출의 절반에서 정체 |
| 6 | **CSM** (Jiang+ 2023) | `CSM/` | dialseg711 미해명 | 🟡 **2/3 재현** — 원인 특정: **저자 ckpt 부재** |
| 7 | LitSeg (Zhang+ 2026) | `litseg/` | 없음 | 🟡 **분절 구현 완료** · 실행 대기(예산) |
| 8 | Storyline Trees (Ou & Lapata 2026) | `storyline_trees/` | 없음 | 🟡 **분절 구현 완료** · 실행 대기(예산) |
| 9 | EventRecall (Panela+ 2025) | `event_recall/` | 데이터팩만 | 🟡 **이식 완료** · 실행 대기(예산) |
| 10 | LLM-Direct (Michelmann+ 2023/25) | `llm_direct/` | **오분류 상태** | 🟡 **이식 완료** · 🔴 모델 폐기로 재현 불가 |
| 11 | BERT-SSC / Llama (Zehe+ 2025, SotA) | `ssc_gbert_2025/` | 없음 | 🔴 **불가**(코퍼스+체크포인트) · 프롬프팅만 준비 |
| 12 | SSC (Kurfalı & Wirén 2021, 1위) | `ssc_kurfali/` | 없음 | 🔴 **불가** — 체크포인트 404 소실 |
| 13 | BERT sentence-pair (Zehe+ 2021b) | `bert_sentence_pair/` | 없음 | 🔴 **불가** — 코퍼스 저작권 |
| 14 | TSE 자질 (Barth & Dönicke 2021) | `tse_barth/` | 없음 | 🔴 **불가** — GermaNet 미배포 |
| 15 | LumberChunker (Diallo+ 2024) | — | 평가만 재현 | ⏸️ **현 상태 유지**(사용자 결정) |

### 🔑 하네스를 논문 조건(C)으로 맞추는 것이 결정적이다
`common/superdialseg_harness.py` 신설 — SuperDialseg 공식 데이터 + 저자 `SegmentationEvaluation`.
**같은 방법인데 하네스만 B→C 로 바꾸면 TextTiling tiage Score 가 0.3081 → 0.3628 로 움직인다**
(논문 0.363). 이전에 "논문과 정확 일치 아님" 이라며 판정을 미룬 건들의 상당수가 하네스 문제였다.
단 **CSM dialseg711 은 하네스로 설명되지 않았다**(Δ −0.0138) — 그건 체크포인트다.

**요약: 숫자 단위로 재현된 것 3건, 부분 재현 1건, 구현 완료·실행 대기 4건, 원천 차단 4건.**
착수 전 우리 저장소에 있던 것은 3개뿐이었고 그중 **논문 수치와 대조된 것은 0개**였다.

### 🔴 이번에 잡아낸 오류 3건 (전부 우리 쪽)
1. **"B2" 라고 부르던 프롬프트가 실은 B3(Panela)였다** — 꼬리 문장이 B3 와 완전 동일하고
   본문은 쉼표 하나만 다르다. B2(Michelmann)와는 `large`/`long` 어휘부터 다르다.
   `main_result.md` 의 "LLM 직접질의" 행 명칭이 정정 대상이다. → `llm_direct/REPRO.md` §1
2. **2025 완화 F1 을 논문 설명대로 짜면 틀린다** — 저자 구현은 창이 비대칭이고(±t 가 아니라
   [−t, +t−1]), 전이 타입까지 맞아야 하며, TP/FP 를 독립 집계한다. → `ssc_gbert_2025/REPRO.md` §3
3. **STSS 공식 표의 "Micro avg." 는 pooled micro 가 아니다** — 문서별 micro 의 평균이다.
   pooled 로 재현하면 Track 1 에서 0.012 가 나와 "재현 실패"라는 잘못된 결론이 난다. → §2

### ℹ️ "7·8 은 이미 돌리지 않았나?" — 돌렸지만 대조는 성립하지 않는다
`dev_b2b3_topline.py` 가 dev 3편에서 gpt-4.1 · 20회 반복으로 이미 분절했고 ledger 에 수치가 있다
(B3 mean 0.474). 예측 원자료도 남아 있다. 그런데 그건 **우리 무대·우리 지표(DTS Score)** 의 값이고
논문이 발표한 숫자와 맞춰 본 적이 없다.

저장된 예측을 논문 지표(Hamming distance)로 재채점해 봤다(`event_recall/rescore_existing_runs.py`,
API 비용 0). 결과: **대조 불가**다. Hamming 은 경계가 희소하면 *아무것도 안 찍기*에 지배되는데,
tunnel·monkey 에서는 우리 예측이 **무경계 하한보다 나쁘다**(0.044 vs 0.036 · 0.083 vs 0.074).
Michelmann 발표값(Pieman 0.223/0.191)은 경계가 ~26개인 gold 기준이고 우리 gold 는 18개라
하한 자체가 다르다. **gold 집계가 다르면 Hamming 은 비교할 수 없다.** 상세 = `event_recall/REPRO.md` §5.

### 🔴 지금 막고 있는 것 2가지
- **LLM 프록시 예산 소진**(429 ExceededBudget, Spend=980.84). 5·6·7·8·9 의 실행이 전부 여기 걸려 있다.
- **독일어 코퍼스 미확보**(사용자 결정으로 저자 요청 안 함). 9·10·11·12 가 여기 걸려 있다.

---

## 1. 🔴 최대 장애물 — German Scene Dataset 은 저작권 자료다

2·3·4·5·6 은 전부 같은 코퍼스(KONVENS 2021 STSS)를 쓴다. 그리고 그 코퍼스는 공개되지 않는다.

> "Since many of the texts both in the original dataset as well as in our extension are
> copyrighted, we cannot publish the full dataset" — Zehe et al. (2025) §5

공개된 것은 **standoff 주석**(경계의 문자 오프셋 + 문장 경계 + 원문 md5)뿐이다. 원문은
저자에게 개별 요청하거나 EAN 으로 전자책을 구매해 병합해야 한다.

### 그럼에도 확보한 것 (텍스트 없이도 되는 것들)

| 자산 | 출처 | 쓸모 |
|---|---|---|
| **공식 gold + Schneider 팀 공식 제출 예측** (Track1 4편·Track2 2편) | `_upstream/embedding-delta/training/stss_results/` | 텍스트 없이 **공식 표 재계산 가능** → 채점기 검증 |
| **STSS-Test-2 전문 2편** (*Effi Briest*, *Aus guter Familie*) — 퍼블릭 도메인 | `_upstream/scene-segmentation/data/full/stss_test_2/*.xmi.zip` | **md5 일치 확인 완료.** 어떤 방법이든 端到端 실행 가능 |
| 41편 전체 standoff 주석 | `_upstream/scene-segmentation/data/standoff/` | 원문을 구하면 즉시 gold 결합 가능 |

md5 검증 결과 (`common/` 로 확인):

```
Aus guter Familie : 453,780자 · b85c06d5…20c2 = standoff md5 ✅ · 장면 219 · 문장 5,025
Effi Briest       : 666,423자 · ff24e980…3431 = standoff md5 ✅ · 장면 227 · 문장 5,906
```

### 학습이 필요한 방법(3·4·6)이 막히는 이유
Train-Full 32편 중 퍼블릭 도메인은 6편 남짓(*Das Erdbeben in Chili*, *Der Schimmelreiter*,
*Die Judenbuche*, *Die Verwandlung*, *Krambambuli* 등)뿐이다. 나머지는 현대 통속소설(Heftroman)이라
합법적으로 구할 수 없다. 6편으로 학습한 모델은 **논문 모델이 아니다** — 발표 수치와 비교하면
독자를 오도한다([[baselines-stay-as-published]]).

### 추론만 되는 방법(2·5)
- **Embedding Delta**: 학습된 SVM(`code/svm.pickle`)이 레포에 동봉돼 있다 → Test-2 에서 그대로 추론.
- **Barth & Dönicke**: 학습된 RandomForest(`pickled_clfs/scene_tagger_RandomForestClassifier_T11_MSL3.pickle`)
  동봉. 단 GermaNet 리소스(`code/resources/germanet/`)가 `.gitignore` 로 빠져 있고 라이선스가 필요하다 — 파이프라인이 돌지 확인 필요.
- **Kurfalı(STSS 1위)**: Docker 가 Google Drive 에서 받도록 돼 있는데 **해당 파일이 404 로 소실**됐다
  (`https://drive.google.com/uc?id=1yayKtOT2pGD7YQpL-r9p3D-ErMt6rVeR`). 저자 문의 없이는 추론도 불가.

---

## 2. ✅ 재현 검사 ① — STSS 공식 채점기 (완료)

`embedding_delta/check_official_submission.py` · 채점기 = `common/stss_eval.py`

| | Track 1 P/R/F1 | Track 2 P/R/F1 |
|---|---|---|
| 논문 Table 1 (Schneider et al. 행) | 0.01 / 0.02 / 0.02 | 0.06 / 0.03 / 0.04 |
| **우리 재계산** | **0.01 / 0.02 / 0.02** ✅ | **0.06 / 0.03 / 0.04** ✅ |

### 이 과정에서 밝혀낸 것 (논문에 안 적힌 구현 세부)

1. **표제 "Micro avg." 는 트랙 pooled micro 가 아니라 "문서별 micro 의 평균"이다.**
   Track 1 에서 두 방식이 갈라진다: 문서별 평균 0.014/0.021/0.017(→발표값과 일치) vs
   pooled 0.011/0.012/0.012(→불일치). Track 2 는 둘 다 같은 값으로 반올림돼 구분되지 않는다.
   **이걸 모르고 pooled 로 재현하면 "재현 실패"라는 잘못된 결론이 난다.**
2. **경계는 위치뿐 아니라 전이 타입까지 맞아야 정답이다** (Scene-to-Scene / Scene-to-Nonscene /
   Nonscene-to-Scene). 위치만 맞고 타입이 틀리면 오답 — Track 2 에서 위치기준 17개 중 15개만 정답.
3. **공식 코드에 라벨 배정 순서 의존 버그가 있다.** `label_to_int` 가 defaultdict 이고
   pred → gold 순으로 채워지는데 채점은 `labels=[1,2,3]` 고정이다. 4번째 타입
   **Nonscene-to-Nonscene** 이 등장하면 그 경계는 점수에서 통째로 빠진다.
   실제 STSS Track 2 gold(*Effi Briest*)에 1개 있다. 우리 구현은 이 동작을 **그대로 복제**한다.

---

## 3. 각 논문이 발표한 수치 (재현 대조표의 기준값)

### STSS 2021 공식 순위 (Zehe et al. 2021a, Table 1 — exact micro F1)

| 시스템 | T1 P | T1 R | T1 F1 | T1 γ | T2 P | T2 R | T2 F1 | T2 γ |
|---|---|---|---|---|---|---|---|---|
| Kurfalı & Wirén | 0.29 | 0.51 | **0.37** | 0.19 | 0.14 | 0.22 | 0.17 | 0.31 |
| Gombert | 0.22 | 0.13 | 0.16 | 0.09 | 0.39 | 0.20 | **0.26** | 0.17 |
| Barth & Dönicke | 0.06 | 0.11 | 0.07 | 0.06 | 0.13 | 0.12 | 0.12 | 0.08 |
| Hatzel & Biemann | 0.02 | 0.03 | 0.02 | 0.12 | 0.08 | 0.17 | 0.11 | 0.13 |
| Schneider et al. (Embedding Delta) | 0.01 | 0.02 | 0.02 | 0.05 | 0.06 | 0.03 | 0.04 | 0.06 |

### 현 SotA (Zehe et al. 2025, NAACL — **완화 F1, tolerance t=3**)

Table 1 · Train-Full 학습 SSC 모델:

| 모델 | STSS-Test-1 | STSS-Test-2 | OOD-Test | Test-Full |
|---|---|---|---|---|
| GBERT-Large + Half Stride | 0.68 | 0.66 | 0.69 | **0.68** |
| GBERT-Large | 0.59 | 0.61 | 0.60 | 0.60 |
| LLPro(fiction-gbert-large) + Half Stride | 0.62 | 0.57 | 0.66 | 0.62 |
| LLPro | 0.51 | 0.47 | 0.43 | 0.47 |
| GBERT-Base + Half Stride | 0.37 | 0.46 | 0.46 | 0.42 |
| GBERT-Base | 0.26 | 0.37 | 0.33 | 0.31 |

Table 2 · 파인튜닝 없는 프롬프팅 (Test-Full): llama3:8b 0.13 · llama3:70b 0.34 ·
llama3.1:405b 0.34 · gpt-4o-mini 0.14 · gpt-4o 0.45

**🔴 지표가 다르다.** 2021 = exact micro F1, 2025 = 완화 F1(t=3). 같은 표에 나란히 놓으면 안 된다.

### LumberChunker (Diallo et al. 2024) — GutenQA DCG@20, 인코더 ada-002
LumberChunker 62.09 · Recursive 54.72 · Paragraph 49.00 · Propositions 46.19 · Semantic 44.74

### LitSeg (Zhang et al. 2026) — GutenQA 검색
LitSeg-Lite CR 0.938 / MRR 0.683 / H@1 0.600 / H@5 0.813 ·
LumberChunker CR 0.885 / MRR 0.633 / H@1 0.552 / H@5 0.761

---

## 4. 우리 저장소에 이미 있던 것들 — 착수 전 vs 지금

| 방법 | 기존 구현 | 착수 전 대조 상태 | 이번 작업 결과 |
|---|---|---|---|
| BayesSurprise | `scripts/eacl_phase0/b1_bayes_surprise.py` | "Kumar 시그니처 재현(AUC 0.705 vs 0.52)" — **정성 방향만**, 논문 수치와 대조 0 | ✅ 숫자 대조 수행 → **재현**. `bayes_surprise/REPRO.md` |
| LumberChunker | 저자 배포 청크로 평가만 | ada-002 재평가 DCG@20 **63.65 vs 논문 62.09** — 평가는 재현됨. 분절 방법 자체는 미실행 | ⏸️ 사용자 결정으로 현 상태 유지 |
| "B2" LLM-Direct | `scripts/eacl_phase0/build_tier2_gold.py` | "Michelmann 주장 미재현" | 🔴 **그건 B2 가 아니라 B3 프롬프트였다.** 기록의 범위를 좁혀 다시 써야 함 |
| EventRecall | `build_eventrecall_pack.py` (데이터팩만) | "우리 구현이 B3와 다른지 미확인" | 🟡 분절 파이프라인 이식 완료 → 예산 복구 시 해소 |
| SeCom-binary | `scripts/eacl_phase0/llm_baseline_binary.py` | — | ℹ️ 이건 B2 도 B3 도 아닌 **SeCom** 계열이다(혼동 주의) |

---

## 5. 폴더 지도

```
methods/
  README.md                  ← 이 문서 (인벤토리·판정·공통 발견)
  common/                    STSS 채점기 · Test-2 데이터 조립 · 지표 이식 · LLM 클라이언트
    stss_eval.py             공식 exact F1 + 2025 완화F1/IoU (저자 코드 이식)
    stss_data.py             XMI→공식 JSON, md5 검증, 2021/2025 두 gold 판본
    verify_metric_port.py    저자 지표 함수와 비트 단위 대조 (2,000 케이스)
    verify_grid.py           문장 격자 기준점 A/B/C 산출
  <method>/REPRO.md          방법별 판정·근거·재현용 설정 기록
  _upstream/                 원저자 코드 클론 (수정 금지)
```

검증 스크립트는 전부 단독 실행 가능하고, 실패 시 0 이 아닌 종료코드를 낸다.

---

## 6. NES 무대 추가분 (2026-08-03)

`preds_table.py` 로 dev-3/test3 에 편입한 신규 행. 생성 스크립트는 각 폴더에 있다.

| 행 | 무엇 | 생성 | REPORT |
|---|---|---|---|
| `FixedInterval` · `RandomRate` | **밀도 하한** — 문서별 경계 수를 M5 와 일치시킨 통제군 | `trivial/run_nes.py` | — |
| `EM-LLM-SM` | EM-LLM **SM 변형** (이동 윈도우 μ+γσ, γ=1.1·n_local=4096 고정, SM+ 제외) | `emllm/run_sm_dev3.py` | `outputs/experiments/2026-08-03_emllm_sm_dev3/REPORT.md` |

### 🔑 밀도 하한이 드러낸 것 (dev-3, 무경계 하한 0.2790)
| 방법 | Score |
|---|---:|
| M5 | 0.4190 |
| **RandomRate (밀도일치, 20회)** | **0.2763 ± 0.0302** |
| **FixedInterval (밀도일치)** | **0.2756** |
| RoBERTa | 0.2799 |
| GraphSeg / GreedySeg / Membox | 0.2212 / 0.1988 / 0.2339 |

**M5 − RandomRate = +0.143** = 위치 선택의 기여(밀도가 통제됐으므로 개수 덕이 아니다).
반대로 **GraphSeg·GreedySeg·Membox 는 밀도만 맞춘 무작위보다 못하고**, RoBERTa·CSM 은
무작위와 구분되지 않는다 ⇒ 서사 무대에서 **밀도 이상의 기여가 없다.**
🔴 무작위는 **20회 분포**로 보고한다. 단일 시드는 dev-3 에서 0.2291~0.3295 로 흩어져
어느 결론이든 만들 수 있다.

### 🔴 EM-LLM 행 명칭 정정
| 기존 | 실제 | 조치 |
|---|---|---|
| "EM-LLM surprisal" 0.2711 | **OP-1 = 고정 분위수**(적응 임계 아님) | ablation 으로 재배치 |
| OP-2 `μ+1.1σ` | 규칙은 맞지만 **full-past**(이동 윈도우 아님) | ablation |
| "EM-LLM + modularity" 0.2978 | **SM+**(그래프 정제, **미래 문맥 사용**) | 온라인 표에서 **제외** |
| **`EM-LLM-SM` 0.2699** | 원 규칙 + 이동 윈도우 + γ 고정 | **EM-LLM 대표 행** |

EM-LLM-SM 은 **무경계 하한 아래**다. `n_local` 스윕(256~4096)에서 **원 설정이 최고**이므로
이식 결함이 아니다 — 상세는 위 REPORT §4.

## 7. DTS 무대 추가분 (2026-08-03)

| 산출 | 내용 |
|---|---|
| `common/superdialseg_harness.py` | 하네스 C (공식 데이터 + 저자 채점기) · `evaluate` / `evaluate_online`(prefix-recompute) |
| `common/verify_causality.py` | 온라인 인과성 코드 검증 — 입력 감사 + 미래 교란 불변 + **대조군** |
| `common/streaming_adapters.py` | 기존 스트리밍 구현을 하네스 C 로 (현재 TextTiling) |
| `common/run_lookahead_curve.py` | offline → bounded → zero **lookahead 곡선** |

### 🔑 lookahead 곡선 (TextTiling, w=5·k=3)
```
dialseg711 :  offline 0.3816 ──(−0.011)──> bounded 0.3707 ──(−0.067)──> zero 0.3041
tiage      :  offline 0.3628 ──(−0.023)──> bounded 0.3398 ──(−0.052)──> zero 0.2874
```
**offline→bounded 는 거의 공짜, bounded→zero 가 6배 비싸다.** ⇒ 무제한 미래는 대부분
불필요하지만 **약간의 lookahead 는 본질적**이다. 단일 Δ 로는 이 구분이 안 되므로
**두 구간을 나눠 보고**해야 한다. 상세 = `outputs/experiments/2026-08-03_lookahead_curve/REPORT.md`.

🔴 **스트리밍 = "미래 안 씀"이 아니다.** `StreamingTextTiling` 은 k 블록을 기다리는
**bounded-lookahead** 이며 판정이 지연 확정된다.
