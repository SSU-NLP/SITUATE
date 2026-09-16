# CSM dialseg711 미재현 — 원인 진단 REPORT

**판정: 하네스 가설 🔴 기각. 남은 원인은 체크포인트다.**
tiage·superseg 는 하네스 C 에서 논문값을 재현하지만(99.9% · 110.1%),
**dialseg711 만 우리 체크포인트 11개 전부에서 논문에 못 미친다**(최고 80.4%).

---

## 1. 실험 setup

| 항목 | 값 |
|---|---|
| **목적** | `2026-08-02_csm_offline_verify/REPORT.md` 가 "dialseg711 최대 76.6%, **데이터·전처리 계보 차이로 보인다, 미해명**" 으로 남긴 건의 원인 특정 |
| **가설(1순위)** | 격차의 상당 부분이 **하네스**다. TextTiling 을 하네스 C 로 재보니 같은 방법인데 tiage Score 가 0.3081→0.3628 로 움직였다(`2026-08-03_texttiling_paper_repro`) |
| **검정 설계** | **체크포인트를 고정**하고 하네스만 B→C 로 바꾼다. 차이가 나면 하네스, 안 나면 하네스가 아니다 |
| 러너 | `EACL/methods/CSM/run_paper_repro.py` |
| 모델 | `super_dialseg.models.csm.CSMSegmenter` (무수정 호출) |
| HP | backbone `bert-base-uncased` · `max_utterance_len=50` · `cut_rate(α)=1.0` (2026-08-02 실행과 동일) |
| 데이터 | **하네스 C** = `benchmarks/superdialseg_data/*/segmentation_file_test.json` (SuperDialseg 공식) |
| 채점기 | **하네스 C** = 저자 `SegmentationEvaluation` |
| 체크포인트 | 우리 DailyDialog 재학습본, 에폭 말 11개 (`external/Dialogue-Topic-Segmenter/checkpoints/`) |
| 비교 기준 | Jiang et al. (2023) Table 3 CSM: tiage **0.509** · dialseg711 **0.660** · superseg **0.458** |
| 환경 | `.venv-superdialseg` · GPU 1장 |

## 2. 결과 ① — 하네스 B vs C (체크포인트 고정 = cpt_175335)

| 데이터셋 | 하네스 B | **하네스 C** | Δ(C−B) | 논문 | C/논문 |
|---|---:|---:|---:|---:|---:|
| superseg | 0.4647 | **0.5044** | +0.0397 | 0.458 | **110.1%** |
| tiage | 0.5353 | **0.5083** | −0.0270 | 0.509 | **99.9%** |
| dialseg711 | 0.5057 | **0.4919** | **−0.0138** | 0.660 | **74.5%** |

**가설 기각.** dialseg711 은 하네스를 논문 조건으로 바꿔도 −0.0138 **더 나빠졌고**, 여전히 74.5% 다.
반면 tiage 는 99.9%, superseg 는 110.1% 로 붙는다. 즉 하네스는 tiage·superseg 의 잔차는 설명하지만
**dialseg711 격차는 설명하지 못한다.**

## 3. 결과 ② — 체크포인트 11개 전수 (dialseg711, 하네스 C)

| 체크포인트 (step) | Score | 논문 대비 |
|---|---:|---:|
| 25905 | 0.5153 | 78.1% |
| **50810** | **0.5304** | **80.4%** ← 최고 |
| 75715 | 0.4576 | 69.3% |
| 100620 | 0.4335 | 65.7% |
| 108620 (devbest) | 0.4624 | 70.1% |
| 125525 | 0.4589 | 69.5% |
| 150430 | 0.4586 | 69.5% |
| 175335 | 0.4919 | 74.5% |
| 200240 | 0.4643 | 70.3% |
| 225145 | 0.4599 | 69.7% |
| 248145 (최종) | 0.4587 | 69.5% |

**11개 중 하나도 논문 0.660 에 닿지 못한다.** 최고가 0.5304(80.4%)이고, 그것도 **1~2 에폭 지점**이다.
학습이 진행될수록 dialseg711 점수가 **떨어진다**(25905→50810 에서 정점 후 하락).

## 4. 해석

### 원인은 체크포인트로 좁혀졌다
사전에 선언한 판정 규칙 — *"Δ 가 작은데 논문과 벌어지면 남는 후보는 체크포인트"* — 이 발동했다.
- 하네스 ❌ (Δ −0.0138, 방향도 반대)
- 데이터 ❌ (하네스 C 는 저자 공식 배포본을 쓴다)
- 채점기 ❌ (저자 `SegmentationEvaluation` 그대로)
- **체크포인트 ✅ 남은 유일한 후보**

### 왜 체크포인트가 다른가 — 저자 것이 없다
저자 스크립트는 `.cache/model_zoo/csm/csm-dailydial.pkl` 을 쓰는데 **배포처가 없다**
(SuperDialseg README 는 데이터 Drive 만 링크한다). 5월에 있던 `methods/CSM/cpt_277000.pth` 도 소실됐다.
우리가 가진 것은 DailyDialog 로 **자체 재학습한** 240개뿐이고, 그 재학습에는 이미 알려진 결함이 있다
(ledger D-B56 §4): pseudo-sample 생성에 **시드가 없어** 표본 수가 실행마다 다르고(888,560 / 889,137 /
885,498), upstream 버그 3건을 고쳐 **released ckpt 를 만든 코드와 다른 코드**가 됐다.

### 왜 dialseg711 에서만 크게 벌어지나 — 미규명
tiage·superseg 는 같은 체크포인트로 99.9%·110.1% 가 나온다. 같은 모델이 한 데이터셋에서만
20~30% 부족한 이유는 이 실험으로 답하지 못했다. dialseg711 은 **경계가 조밀하고**(논문 CSM 의
F1 0.610 은 세 셋 중 최고) 대화가 인위적으로 이어붙여진 데이터라, 재학습 모델이 놓친 신호가
그 성격과 관련될 수 있다 — **가설일 뿐 검정하지 않았다.**

## 5. 판정 및 다음 단계

- 🔴 **CSM dialseg711 = 재현 불가.** 사유 = **저자 체크포인트 부재**(배포처 없음 + 사본 소실).
  이전 REPORT 의 "데이터·전처리 계보 차이로 보인다" 는 **틀린 추정**이었다 — 데이터·채점기를
  저자 것으로 맞춰도 격차가 남는다.
- ✅ **CSM tiage·superseg = 하네스 C 에서 재현**(99.9% · 110.1%).
- ⬜ 보고 시 **"CSM 은 3셋 중 2셋 재현, dialseg711 은 체크포인트 부재로 재현 불가"** 로 쓴다.
  "우리 CSM 이 약하다" 로 읽히지 않게 사유를 병기한다.
- ⬜ 저자 `csm-dailydial.pkl` 을 어디선가 확보하면 이 판정은 뒤집힐 수 있다. 현재는 경로 없음.

## 6. 한계 / 검증 미해결

- **dialseg711 에서만 벌어지는 이유 미규명** (§4 마지막). 가설만 있고 검정 안 함.
- **240개 중 11개(에폭 말)만** 훑었다. 나머지 229개에 더 좋은 것이 있을 수 있으나,
  test 에서 최고를 고르면 selection-on-test 다 — 하지 않는다.
- 우리 체크포인트는 lxing532 `CoherenceNet` 산이라 SuperDialseg `CoherenceScoringModel` 과
  **키 이름이 다르다**(`bert.*`→`bert.bert.*`, `coherence_decoder`→`coherence_prediction_decoder`).
  기존 검증된 remap(`methods/CSM/offline/whole_dialogue.py::_load_lxing532_ckpt`)을 재사용했다.
  NSP head 는 우리 ckpt 에 없지만 `CoherenceScoringModel.forward` 가 `hidden_states[-1][:,0,:]` 만
  쓰므로 **추론에 무관**하다(코드로 확인).
- 하네스 C 자체의 검증은 TextTiling 으로 이뤄졌다(3/3 재현). CSM 결과는 그 검증 위에 선다.
