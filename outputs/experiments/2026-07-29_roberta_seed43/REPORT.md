# Supervised RoBERTa (SuperDialseg) — offline, 논문 충실 재현

`methods/RoBERTa/offline/train.py` 산출. Coldog2333/SuperDialseg (EMNLP 2023) Table 3 `RoBERTa` (plain supervised) 재현.

## 1. 실험 setup

- **모델**: `RobertaForTokenClassification` (`roberta-base`, num_labels=2). 입력 `<s> u1 </s></s> … uN </s>`, 각 발화 뒤 첫 `</s>` 에서 token-classification (논문 Appendix A + repo `data_collator.py`).
- **데이터**: `benchmarks/superdialseg_data/`. 학습 `superseg/train` (6948 dial), model selection `superseg/validation` (1322 dial) val Score, 평가 tiage/dialseg711/superseg test.
- **학습 HP**: AdamW lr=1e-05, batch=8, weight_decay=0.001, grad_clip=1.0, epochs=20, early stopping(patience=10, best 보존), 슬라이딩 윈도우 |T|=20 stride 1, max_utt_len=25, max_seq_len=512, seed=43 (단일 run).
- **metric**: official SuperDialseg Pk/WD (window=평균 segment 길이/2) + binary F1, `Score=(2·F1+(1−Pk)+(1−WD))/4`.

## 2. 결과 — test split vs 논문 Table 3 (`RoBERTa`)

| 평가셋 | in-domain? | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | 논문 Score | ΔScore |
|---|---|---:|---:|---:|---:|---:|---:|
| superseg | ✔ | 0.1936 | 0.2041 | 0.8339 | **0.8175** | 0.798 | +0.0195 |
| tiage | ✘ zero-shot | 0.3847 | 0.4257 | 0.5236 | **0.5592** | 0.482 | +0.0772 |
| dialseg711 | ✘ zero-shot | 0.2654 | 0.3323 | 0.6443 | **0.6727** | 0.702 | -0.0293 |
| **mean-3** |  |  |  |  | **0.6831** | 0.661 | +0.0225 |

## 3. 학습 곡선 (val = superseg/validation Score)

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train loss | 0.266 | 0.196 | 0.153 | 0.120 | 0.094 | 0.072 | 0.056 | 0.046 | 0.038 | 0.032 | 0.027 | 0.023 |
| val Score | 0.820 | 0.825 | 0.821 | 0.811 | 0.807 | 0.822 | 0.813 | 0.811 | 0.818 | 0.823 | 0.819 | 0.802 |

best = epoch 2 → test 평가는 그 체크포인트 사용.

## 4. 판정

- mean-3 ΔScore = +0.0225. 논문 RoBERTa 행 근방이면 방법 충실 재현 성공.
- **Hi-OnTop 대비 비대칭**: 본 모델은 supervised·offline(윈도우 안 미래 발화 관측)·학습 필요. Hi-OnTop 는 무감독 online past-only. RoBERTa 우위는 supervised offline 상한 — online 무감독과 1:1 비교 아님.

## 5. 한계 / 검증 미해결

- **byte 재현 불가**: repo 에 학습 코드·체크포인트·seed 미공개 (`main.py` 평가 전용). 논문은 다회 평균. 단일 seed run.
- **추론 윈도우 집계 미명시**: 겹치는 stride-1 윈도우의 발화별 예측을 logit 평균으로 통합 (논문/repo 미명시 → 본 재현의 결정).
- **데이터 버전**: `superseg-v2` (개정판) — 논문 Table 2 의 dialogue 수와 상이.
- **plain RoBERTa 만**: `MVRoBERTa`(role/DA 입력 임베딩)·`RobertaMultiTask`(da/role 출력 헤드)는 미구현.
- model 체크포인트는 `outputs/experiments/<name>/model/` (gitignored).

