# codex 판정 2 — 손실 스케일 통제 · 2×2 설계 · 희소축 붕괴 판정

> 2026-07-30 KST · `codex exec resume --last` (앞 스레드 이어서). 질의 = `CODEX_QUESTION_2.md`

---

변경 사항을 반영하면 결론은 다음입니다.

- 경계 신호는 `S5`로 고정하고 `S4`는 진단만 합니다.
- 균등 ALPHA는 `0.63 × 5`가 올바른 1차 통제입니다.
- 제안한 2×2는 타당합니다.
- W0 체크포인트는 provenance가 동일하면 재사용할 수 있고, W0 1-seed 재학습으로 하네스를 검증합니다.
- W2는 본 2×2에 넣지 말고, time 붕괴 시에만 실행하는 사전등록된 rescue 조건으로 둡니다.
- 필요한 신규 학습은 기본 10런입니다: W3/W1/W4 각 3 seed + W0 무결성 1 seed.

## 1. 합 보존 균등화 0.63이 맞는가?

네. 이번 비교의 목적이 “축 간 상대비중만 바꾸기”라면

\[
\alpha_k=\frac{3.15}{5}=0.63
\]

이 가장 깨끗한 통제입니다.

현재 supervised 부분은 대략

\[
L_{\rm sup}
=\sum_k\alpha_k
\left(L_{{\rm change},k}+\beta L_{{\rm state},k}\right)
\]

이므로 모든 축에서 state head가 존재하고 reduction 방식도 같다면, ALPHA 합 보존은 다음을 동시에 유지합니다.

- change 손실의 총 명목 스케일
- state 손실의 총 명목 스케일
- change와 state 사이의 `beta=0.25`
- supervised 손실과 `γL_collapse` 사이의 명목 비율

따라서 W3 대 W1에서 의도적으로 변하는 것은 ALPHA의 축별 상대배분뿐입니다.

### `ALPHA=1`, LR을 3.15/5로 낮추는 대안

동등하지 않습니다.

LR 조정은 모든 gradient update를 축소하지만, `γL_collapse`의 loss 내부 상대비중은 복원하지 못합니다.

```text
ALPHA=1, lr×0.63:
supervised : collapse = 5 : 0.05

ALPHA=0.63:
supervised : collapse = 3.15 : 0.05
```

Adam 계열에서는 전체 loss scaling과 LR scaling도 정확히 등가가 아닙니다. 모멘트 정규화, epsilon, gradient clipping, weight decay 때문에 차이가 납니다.

### gamma나 beta를 함께 재조정하는 대안

이번 실험에서는 권하지 않습니다.

- `beta`를 바꾸면 change 대 state supervision이라는 별도 요인이 추가됩니다.
- `gamma`를 바꾸면 collapse regularization이라는 별도 요인이 추가됩니다.
- 둘 다 원인 귀속을 흐립니다.

따라서 `beta=0.25`, `gamma=0.05`, LR을 모두 고정하고 ALPHA만 합 보존 균등화하십시오.

### 한 가지 한계

`ΣALPHA` 보존은 명목 스케일 통제이지 매 step의 실제 gradient norm까지 같게 만들지는 않습니다. 축마다 BCE 값과 gradient norm이 다르기 때문입니다. 이를 억지로 맞추는 GradNorm류 동적 방법은 또 다른 알고리즘이 됩니다.

대신 모든 셀에서 다음을 로그로 남기면 충분합니다.

- `L_change`와 `βL_state`
- `γL_collapse`
- 가능하면 공유 파라미터에 대한 세 항의 gradient norm
- gradient clipping 발생률
- epoch별 optimizer step 수

이는 선택 기준이 아니라 통제가 실제로 작동했는지 확인하는 진단입니다.

## 2. 2×2 설계와 W0 재사용

제안한 2×2는 정확합니다.

| | clipped pos_weight | `pos_weight=1` |
|---|---:|---:|
| 기존 ALPHA | W0 | W3 |
| 합보존 균등 ALPHA | W4 | W1 |

이 설계로 다음을 분리할 수 있습니다.

- W0 대 W3: 기존 ALPHA에서 pos_weight 효과
- W0 대 W4: weighted BCE에서 ALPHA 효과
- W3 대 W1: plain BCE에서 ALPHA 효과
- W4 대 W1: 균등 ALPHA에서 pos_weight 효과
- 네 셀 전체: ALPHA × pos_weight 교호작용

다만 `W0 vs W3 = 순수 pos_weight 효과`라고 말하려면 학습 코드·데이터 순서·seed·checkpoint 선택까지 동일해야 합니다.

### W0 기존 체크포인트 재사용 조건

다음이 모두 확인되면 가능합니다.

- 같은 anchor split과 feature 파일
- 같은 데이터 순서 및 seed 의미
- 같은 optimizer, scheduler, batch size, epoch
- 같은 초기화 경로
- 같은 checkpoint-selection 규칙
- 같은 코드에서 새 CLI 기본값이 옛 동작과 동일
- 평가 코드가 동일

오늘 확인한 dev-3 예측 재현은 평가 경로의 무결성은 보여주지만, 새 학습 하네스가 기존 학습을 재현한다는 증거는 아닙니다. 따라서 제안한 W0 1-seed 재학습이 필요합니다.

가능하면 기존 W0와 동일한 seed 하나를 사용해 다음을 비교하십시오.

- 학습곡선
- anchor diagnostic metrics
- dev-3 예측 개수와 경계 위치
- DTS Score Δ
- 가능하면 출력 logits의 상관 및 평균 절대차

결과가 사전 설정한 재현 허용범위 안이면 기존 W0 3개를 factorial cell로 재사용해도 됩니다. 벗어나면 W0도 3 seed 전부 새 하네스로 재학습해야 합니다.

완전 결정론 학습이 보장되는 환경이라면 state dict 또는 logits의 거의 정확한 일치까지 요구할 수 있습니다. 그렇지 않다면 기존 seed 간 변동폭을 재현 허용범위로 사용하십시오.

## 3. plain BCE에서 time이 학습될 수 있는가?

표본 수만 보면 충분히 가능합니다. 양성 약 6만 건은 희소하더라도 절대적으로 작은 수가 아닙니다. 다만 다음은 구분해야 합니다.

- time posterior가 대체로 낮다: 정상일 수 있음
- 0.5 threshold에서 recall이 낮다: 붕괴 증거가 아님
- 모든 입력에 거의 동일한 0.0215를 출력한다: prevalence-only collapse 가능성
- 입력에 따른 순위 정보도 없다: 실제 판별 붕괴

plain BCE의 정상적인 희소축 출력은 대부분 낮은 확률이어야 합니다. 따라서 mean probability가 낮아졌다는 이유로 붕괴라고 판정하면 안 됩니다.

### 앵커를 선택셋으로 쓰지 않는 판정법

앵커 holdout은 모델 선택에는 쓰지 않되, 학습 유효성 검사에는 사용할 수 있습니다. 이를 사전에 “invalid-run 판정”으로만 정의해야 합니다.

time 붕괴 진단은 다음을 기록하십시오.

- positive와 negative의 평균 logit 차이
- AUROC와 AUPRC
- prevalence-only predictor 대비 log loss 개선
- Brier skill
- logit 표준편차
- time head의 gradient norm
- 여러 seed에서의 재현성

중요한 기준은 `0.5`에서의 F1이 아니라 prevalence-only 기준보다 조건부 정보를 학습했는지입니다.

예를 들어 고정된 anchor diagnostic split에서

\[
\Delta LL =
LL(\text{constant prevalence})-LL(\text{model})
\]

이 양수이고, AUROC가 명백히 0.5보다 높으며 positive/negative logit 분리가 seed 전반에서 나타나면 time은 학습된 것입니다.

이것은 silver time 라벨을 학습했는지 확인할 뿐, 사람 기준 time 의미가 맞다는 증거는 아닙니다. 그 한계는 명시해야 합니다.

### W2가 필요한가?

W2 없이도 W1과 W3에서 time 붕괴 여부는 판정할 수 있습니다. 다만 “붕괴했을 때 exact odds가 구조적으로 구제하는가”는 W2를 돌려야만 압니다.

따라서 W2를 조건부 rescue로 사전등록하십시오.

```text
W2:
ALPHA = 0.63
pos_weight = exact odds, clip 없음
배포 출력 = analytic inverse correction 적용
3 seed
```

실행 조건은 결과를 본 뒤 임의로 정하면 안 됩니다. 예를 들면:

- W1의 3 seed 모두에서 time이 prevalence baseline을 유의미하게 이기지 못함
- 그리고 AUROC/positive-negative logit separation도 함께 붕괴

정확한 수치 허용치는 기존 W0 seed 변동 또는 고정 bootstrap CI로 결과 확인 전에 잠가야 합니다.

가능하면 더 깨끗한 방법은 dev-3 및 봉인셋과 분리된 소규모 사람 축라벨 진단셋을 만드는 것입니다. 그것이 없으면 true time-axis collapse와 silver-label imitation 실패를 구분할 수 없습니다.

## 4. S5 고정 후 W 셀 선택 게이트

S4는 선택 후보가 아니므로 다음만 수행합니다.

- 모든 W 셀의 운영 신호는 `S5`
- S4는 causality 영향 해석용 표에만 기록
- S4 결과가 좋아도 모델이나 집계 규칙을 S4로 변경하지 않음

### W 선택 원칙

W1을 이론적으로 사전 지정한 주안으로 두는 것이 좋습니다. 네 셀 중 dev-3 최고점을 사후 선택하면 작은 선택셋에 대한 다중 비교가 됩니다.

권고 게이트는 다음 순서입니다.

1. 학습 유효성  
   모든 축이 non-degenerate하고 seed 발산이 없어야 함.

2. plain BCE 목적 달성  
   W1이 weighted posterior가 아니라 직접적인 silver posterior를 출력하는지 anchor에서는 sanity check만 수행.

3. 서사 deployable 성능  
   고정된 동일 경계 규칙에서 W1−W0의 DTS Score Δ 방향과 경계 개수를 dev-3에서 비교.

4. 대화 비열등성  
   validation 앞 60대화에서 W1이 사전등록한 허용폭 이상 악화하지 않아야 함.

5. 통과 시 W1 채택  
   W3/W4는 ALPHA와 pos_weight의 주효과·교호작용 해석용으로 사용.

dev-3에서는 “보고하는 셋에서 학습설계를 골랐다”고 공시하고 절대 Score는 인용하지 않습니다.

비열등 허용폭은 현재 정보로 확정할 수 없습니다. 기존 3-seed 분산과 문서/대화 단위 bootstrap 분포로 정한 뒤 결과를 보기 전에 잠가야 합니다.

### W 간 비교 시 경계 임계를 다시 맞출 것인가?

1차 W 게이트에서는 다시 맞추지 않는 것이 좋습니다. 동일한 기존 deployable 규칙을 사용해야 loss 변경의 직접 영향을 볼 수 있습니다.

W1을 채택한 뒤에만 개수 제어 트랙에서 `c`, EWMA, 최소간격을 다시 선택하십시오. 각 W마다 개수 제어까지 별도로 최적화하면 loss 셀과 operating-point 탐색이 섞이고 dev-3 과적합 위험도 커집니다.

### causality가 S5의 50%를 넘으면 문제인가?

그 자체로는 실패가 아닙니다.

causality 실제 발화율이 높다면 기대 리셋 축수에서 큰 비중을 차지하는 것은 확률적으로 자연스럽습니다. “5축을 동일 비중으로 합한다”는 것은 각 축의 발생률을 동일하게 만든다는 뜻이 아닙니다.

문제 여부는 비중이 아니라 경계 정보입니다.

확인할 진단은 다음입니다.

- gold 경계와 비경계에서 causality 확률의 차이
- causality의 boundary-conditional lift
- causality 제거 전후 S5 순위 변화
- 전체 S5 분산 중 causality가 설명하는 비율
- causality가 임계 초과 시점을 독점하는 비율
- S4 대 S5의 Δ — 단, 판독만 하고 채택에는 사용하지 않음

causality가 높은 상수에 가깝다면 누적 `μ+cσ`가 평균 성분을 상당 부분 제거하므로 영향이 제한될 수 있습니다. 반대로 높은 확률이 국소적으로 요동하거나 다른 축을 압도해 꼬리를 결정하면 경계 순위를 훼손할 수 있습니다.

후자의 경우라도 이번 결정 아래서는 causality를 제거하거나 낮추지 않습니다. 결과를 “5축 제약의 비용”으로 보고하고, 해결은 축 가중치가 아니라 별도 개수 제어 트랙에서 다뤄야 합니다.

## 5. 최소 셀과 seed 수

### 권고 기본안

- W0: 기존 3 seed 재사용
- W0: 하네스 무결성 1 seed 신규
- W3: 3 seed 신규
- W1: 3 seed 신규
- W4: 3 seed 신규

즉:

```text
factorial 분석: 4셀 × 3 seed = 12개 체크포인트
신규 학습: 10런
```

이는 과하지 않습니다. 2×2의 주효과와 교호작용을 보려면 각 셀에 동일한 3 seeds가 최소한 필요합니다. 가능하면 동일 seed 세트를 모든 셀에 사용해 paired comparison으로 분석하십시오. seed가 단 3개이므로 p-value 중심보다는 다음을 보고하는 편이 낫습니다.

- seed별 paired Δ
- 3/3 방향 일치 여부
- 평균 Δ
- 문서/대화 단위 bootstrap CI
- ALPHA × pos_weight interaction contrast

교호작용 contrast는 예를 들면 다음입니다.

\[
I=(W1-W3)-(W4-W0)
\]

단, 각 항은 같은 seed끼리 짝지어 계산합니다.

### W0 무결성 재학습이 실패하면

W0 2개를 추가로 학습해 모든 셀을 같은 하네스의 3 seed로 맞춥니다.

```text
그 경우 신규 학습 총수 = 12런
```

### W2가 발동하면

W2 3 seed를 추가합니다.

```text
기본 신규 학습: 10
W0 재사용 실패: +2
time rescue W2: +3
최대: 15런
```

## 최종 실행안

1. ALPHA 균등값은 전축 `0.63`으로 고정합니다.
2. `beta`, `gamma`, LR, scheduler, clipping은 변경하지 않습니다.
3. W0/W3/W1/W4의 2×2를 동일한 세 seed로 비교합니다.
4. 기존 W0 3 seed는 조건부 재사용하고, 동일 seed 하나를 새 하네스로 재학습해 검증합니다.
5. W1을 사전 지정 주안으로 두고 W3/W4는 요인 해석용으로 둡니다.
6. time 붕괴는 anchor diagnostic split에서 invalid-run 기준으로만 판정합니다.
7. 사전등록된 붕괴 조건이 충족될 때만 W2 exact-odds 3 seed를 추가합니다.
8. 모든 운영 평가는 S5로 하고 S4는 판독용으로만 기록합니다.
9. W1 채택 여부를 고정한 다음 개수 제어 트랙을 시작합니다.

핵심적으로, `0.63`은 단순 편의값이 아니라 현재 구조에서 ALPHA 상대배분만 바꾸기 위한 올바른 실험 통제입니다.