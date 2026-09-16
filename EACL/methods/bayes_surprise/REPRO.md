# B1-BayesSurprise (Kumar et al. 2023, Cognitive Science) — 재현 결과

**판정: ✅ 재현.** 논문의 핵심 주장(“경계는 Bayesian surprise 와 연관되고 surprisal 과는
무관하다”)이 **5개 창 크기 전부에서 재현**되고, 상관계수 절대값도 큰 창에서 0.012 이내로 맞는다.
이 방법군에서 **숫자 단위 대조가 성립한 유일한 사례**다.

---

## 1. 대조표 (held-out story 의 Pearson r · drift 0.05)

| 신호 | 창 20 | 창 50 | 창 100 | 창 150 | 창 200 |
|---|---:|---:|---:|---:|---:|
| **Bayesian surprise (KL)** — 우리 | +0.051 | +0.063 | **+0.108** | **+0.120** | **+0.122** |
| Bayesian surprise — 논문 | +0.07 | +0.07 | +0.10 | +0.11 | +0.11 |
| *차이* | *−0.019* | *−0.007* | *+0.008* | *+0.010* | *+0.012* |
| surprisal — 우리 | −0.046 | −0.070 | −0.049 | −0.021 | −0.014 |
| surprisal — 논문 | +0.01 | −0.06 | −0.04 | −0.03 | — |
| entropy — 우리 | −0.085 | −0.090 | −0.024 | −0.031 | −0.067 |
| entropy — 논문 | −0.01 | −0.05 | +0.01 | +0.01 | — |

### 핵심 주장 재현 — 5/5 ✅
모든 창에서 **KL 만 양수**이고 surprisal·entropy 를 앞선다.

    창  20: KL(+0.051) > surprisal(−0.046) · entropy(−0.085)  ✅
    창  50: KL(+0.063) > surprisal(−0.070) · entropy(−0.090)  ✅
    창 100: KL(+0.108) > surprisal(−0.049) · entropy(−0.024)  ✅
    창 150: KL(+0.120) > surprisal(−0.021) · entropy(−0.031)  ✅
    창 200: KL(+0.122) > surprisal(−0.014) · entropy(−0.067)  ✅

창이 커질수록 r 이 오르는 경향(0.05 → 0.12)도 논문(0.07 → 0.11)과 같다.

### 어긋난 곳 (숨기지 않고 기록)
- **창 20 의 surprisal**: 논문 +0.01, 우리 −0.046 (차이 −0.056). 둘 다 0 근처이고
  "surprisal 은 경계를 예측하지 못한다"는 결론은 같지만 부호가 갈린다.
- **entropy 전반**이 논문보다 0.03~0.07 더 음수다. 논문에서도 0 근처이므로 결론은 같다.
- 작은 창(20·50)에서 KL 이 논문보다 낮다. 큰 창에서는 오히려 우리가 약간 높다.

이 정도 어긋남의 유력한 원인은 **패딩**이다 — 아래 §3 참조.

## 2. 무엇을 재현했나 (파이프라인)

| 단계 | 스크립트 | 내용 |
|---|---|---|
| ① 신호 | `compute_kl.py` | 저자 배포 **GPT-2-XL logits**(50,257차원/단어)에서 `KL(P_i ‖ P_{i−1})` 계산. `softmax` → `scipy.stats.entropy` 와 동일 식 |
| ② 회귀 | `reproduce_kumar_regression.py` | Reynolds transient 정규화 → trailing-window 자질 → Lasso · leave-one-story-out · 내부 GridSearchCV → held-out Pearson r |

**모델을 다시 돌리지 않았다.** Zenodo 배포본(10.5281/zenodo.8404102)에 저자가 쓴 GPT-2-XL
(문맥 1024)의 단어별 logits 전체가 들어 있어, **저자와 같은 입력**에서 신호를 다시 만들었다.
따라서 이 재현이 검증하는 것은 **신호 정의 + 분석 파이프라인**이지 GPT-2 추론 자체가 아니다.

데이터: *Monkey in the Middle*(4,744단어) · *Pieman*(853단어) · *Tunnel Under the World*(3,086단어),
버튼 프레스 = `*_button_density.csv` 를 단어 onset 시각으로 샘플링 후 최댓값 정규화.

## 3. 논문·코드만으로는 확정되지 않아 우리가 정한 것

1. **패딩을 넣지 않았다.** 저자 `context_helper.py::pad_df` 는 앞뒤로 200개씩 첫/끝 값을
   덧댄다. 그런데 그 함수는 `create_analyses_df.py` 경로에만 있고, 회귀 스크립트는 datum 에서
   onset 을 다시 붙인다. 회귀 입력에 패딩이 들어갔는지 코드만으로 확정되지 않아 **없이** 돌렸다.
   작은 창에서 우리 r 이 낮은 것은 패딩 유무와 방향이 맞는다(패딩은 앞부분 표본을 살린다).
2. **onset 정렬은 행 순서로** 했다. surp_entr 의 `Unnamed: 0` 열은 다른 프레임의 인덱스라
   (pieman 최댓값 1281 vs datum 854행) 정렬에 쓸 수 없다. 저자도 `df['onsets']=datum['onset']`
   으로 행 순서 그대로 붙인다.
3. **논문 r = 세 이야기 held-out r 의 평균**으로 해석했다. 논문은 창별로 값 하나만 보고한다.

## 4. 우리 저장소의 기존 B1 구현과의 관계

`scripts/eacl_phase0/b1_bayes_surprise.py` 는 GPT-2-XL 을 **우리가 직접 돌려** KL 을 계산하고
전이별 readout(onset-max)을 붙인 **우리 파이프라인용 각색**이며, drift 도 0.10 을 쓴다.
ledger 에는 "Kumar 시그니처 재현(AUC 0.705 vs surprisal 0.52)"이라는 **정성 확인**만 있고
논문의 어떤 수치와도 대조된 적이 없었다. 이 문서가 그 대조를 처음 수행한 기록이다.

두 구현은 목적이 다르므로 공존한다:
- `b1_bayes_surprise.py` = 우리 평가무대(전이 단위 deployable 채점)에 올리기 위한 각색
- 이 폴더 = **논문 재현 전용**. 저자 입력·저자 공식·저자 회귀 그대로

## 5. 재현용 기록

| 항목 | 값 |
|---|---|
| 입력 | Zenodo 10.5281/zenodo.8404102 · `extract-embeddings-data.zip`(3.8GB) · `button_press_proportion.zip`(48MB) |
| 신호 | `signals/{monkey,pieman,tunnel}_signals.npz` (KL · cross-entropy · surprisal · entropy) |
| drift | 0.05 (논문 표 조건) |
| 창 | 20 · 50 · 100 · 150 · 200 |
| 회귀 | Lasso(max_iter=5000, tol=1e-2), alpha ∈ logspace(−16, −5, 10), PredefinedSplit(story) 이중 |
| 산출 | `results/kumar_regression_drift5.json` |
