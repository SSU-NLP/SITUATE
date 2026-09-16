# CSM — Coherence Scoring Model (Jiang et al. 2023 SuperDialseg Table 3) — 재현 결과

**판정: 🟡 3셋 중 2셋 재현.** tiage 99.9% · superseg 110.1% ✅ / **dialseg711 최대 80.4% 🔴**.
dialseg711 미재현의 사유는 **저자 체크포인트 부재**로 특정됐다 — 이전 REPORT 의
"데이터·전처리 계보 차이" 추정은 **기각**됐다.

---

## 1. 대조표 (하네스 C · ckpt 175335)

| 데이터셋 | 우리 Score | 논문 | 논문 대비 | |
|---|---:|---:|---:|---|
| superseg | 0.5044 | 0.458 | 110.1% | ✅ |
| tiage | 0.5083 | 0.509 | 99.9% | ✅ |
| **dialseg711** | **0.4919** | **0.660** | **74.5%** | 🔴 |

전체 실행 기록·지표별 값 = `outputs/experiments/2026-08-03_csm_harness_diagnosis/REPORT.md`

## 2. 🔴 이전 추정을 기각했다

`outputs/experiments/2026-08-02_csm_offline_verify/REPORT.md` 는 이렇게 남겼다:

> dialseg711 은 최대 76.6% 로 재현되지 않는다. … 체크포인트가 아니라
> **데이터·전처리 계보 차이**로 보인다. 미해명.

**체크포인트를 고정하고 하네스만 B→C 로 바꿔** 검정했다(데이터·채점기를 저자 것으로 교체):

| 데이터셋 | 하네스 B | 하네스 C | Δ(C−B) |
|---|---:|---:|---:|
| superseg | 0.4647 | 0.5044 | **+0.0397** |
| tiage | 0.5353 | 0.5083 | −0.0270 |
| dialseg711 | 0.5057 | 0.4919 | **−0.0138** |

dialseg711 은 논문 조건으로 옮겨도 **오히려 조금 나빠졌다.** 하네스는 tiage·superseg 의 잔차를
설명하지만 dialseg711 격차는 설명하지 못한다. ⇒ **데이터·전처리 계보 가설 기각.**

## 3. 체크포인트 전수 — 11개 어느 것도 논문에 못 미친다

dialseg711 · 하네스 C · 에폭 말 체크포인트 11개:

| step | 25905 | **50810** | 75715 | 100620 | 108620 | 125525 | 150430 | 175335 | 200240 | 225145 | 248145 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Score | 0.5153 | **0.5304** | 0.4576 | 0.4335 | 0.4624 | 0.4589 | 0.4586 | 0.4919 | 0.4643 | 0.4599 | 0.4587 |
| 논문% | 78.1 | **80.4** | 69.3 | 65.7 | 70.1 | 69.5 | 69.5 | 74.5 | 70.3 | 69.7 | 69.5 |

최고가 **1~2 에폭 지점(80.4%)** 이고 이후 **떨어진다**. 논문 0.660 에 닿는 체크포인트가 없다.

## 4. 남은 원인 — 저자 체크포인트가 없다

저자 스크립트(`examples/reproduce/scripts/csm.sh`)는 `.cache/model_zoo/csm/csm-dailydial.pkl`
을 쓰는데 **배포처가 없다** (SuperDialseg README 는 데이터 Drive 만 링크한다).
5월에 있던 사본 `methods/CSM/cpt_277000.pth` 도 소실됐다.

우리가 가진 것은 DailyDialog 로 **자체 재학습한** 240개뿐이고, 그 재학습에는 알려진 결함이 있다
(ledger D-B56 §4):
- pseudo-sample 생성에 **시드가 없어** 표본 수가 실행마다 다르다(888,560 / 889,137 / 885,498)
- upstream 버그 3건을 고쳤다 — 특히 `validation()` 이 `model.train()` 을 복원하지 않아
  24,000 step 동안 dropout 이 꺼져 있던 것. 고친 결과 **released ckpt 를 만든 코드와 다른 코드**가 됐다

⇒ 사유는 **"우리 CSM 이 약해서"가 아니라 "저자 모델을 구할 수 없어서"** 다. 보고에 반드시 병기.

## 5. 왜 dialseg711 에서만 벌어지나 — **미규명**

같은 체크포인트가 tiage 99.9% · superseg 110.1% 를 낸다. 한 데이터셋에서만 20~30% 부족한 이유는
이 실험으로 답하지 못했다. dialseg711 은 경계가 조밀하고(논문 CSM 의 F1 0.610 이 세 셋 중 최고)
대화가 인위적으로 이어붙여진 데이터라 재학습 모델이 놓친 신호가 그 성격과 관련될 수 있다 —
**가설일 뿐 검정하지 않았다.**

## 6. 실행 조건

| 항목 | 값 |
|---|---|
| 러너 | `run_paper_repro.py` |
| 모델 | `super_dialseg.models.csm.CSMSegmenter` (무수정) |
| HP | backbone `bert-base-uncased` · `max_utterance_len=50` · `cut_rate(α)=1.0` |
| 데이터 | `benchmarks/superdialseg_data/*/segmentation_file_test.json` (공식) |
| 채점기 | 저자 `SegmentationEvaluation` |
| ckpt 로딩 | `methods/CSM/offline/whole_dialogue.py::_load_lxing532_ckpt` 재사용 — lxing532 `CoherenceNet` 키를 `CoherenceScoringModel` 로 remap (`bert.*`→`bert.bert.*`, `coherence_decoder`→`coherence_prediction_decoder`). NSP head 는 우리 ckpt 에 없지만 `forward` 가 `hidden_states[-1][:,0,:]` 만 쓰므로 **추론에 무관**(코드로 확인) |
| 환경 | `.venv-superdialseg` · GPU 1장 |

```bash
PYTHONPATH=benchmarks/superdialseg/src NLTK_DATA=/root/nltk_data TOKENIZERS_PARALLELISM=false \
  .venv-superdialseg/bin/python EACL/methods/CSM/run_paper_repro.py \
  --ckpt external/Dialogue-Topic-Segmenter/checkpoints/cpt_175335.pth
```

## 7. 검증 미해결

- dialseg711 특이성 미규명(§5).
- 240개 중 **에폭 말 11개만** 훑었다. test 에서 최고를 고르면 selection-on-test 이므로 하지 않는다.
- `csm-dailydial.pkl` 을 확보하면 이 판정은 뒤집힐 수 있다. 현재 경로 없음.
