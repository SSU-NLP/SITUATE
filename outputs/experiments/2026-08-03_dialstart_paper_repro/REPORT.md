# DialSTART 원본 재현 — REPORT

**판정: ✅ 완전 재현.** 2/2 데이터셋 · 4/4 지표가 **소수 4자리까지** 논문값과 일치.
방법론 판정의 canonical = `EACL/methods/dialstart/REPRO.md` (이 문서는 실행 기록).

---

## 1. 실험 setup

| 항목 | 값 |
|---|---|
| **목적** | DialSTART 는 저장소에 **구현이 아예 없었다**(서베이 표 인용값만). DTS SotA 를 대칭 실험에 쓰려면 원본 확보가 선행 |
| 논문 | Gao et al. (2023) SIGIR, *Unsupervised Dialogue Topic Segmentation with Topic-aware Utterance Representation*, arXiv:2305.02747 |
| 코드 | `EACL/methods/_upstream/DAMO-ConvAI/dial-start/` (sparse clone, **무수정**) |
| 명령 | `python test.py --model model_711 --dataset 711 --single_ckpt --ckpt ./model/model_711` (저자 `scripts/test.sh` 의 *"reproduce our paper result"* 줄 그대로) · doc 판 동일 |
| 모델 | 저자 배포 체크포인트 `model_711` · `model_doc` (각 847MB, Google Drive) |
| 데이터 | 저자 배포 `data/dialseg711` (711대화) · `data/doc2dial` (4,135대화) |
| 지표 | `segeval.pk` · `segeval.window_diff` (저자 `test.py` 가 직접 호출) |
| HP | `--window_size 2` (기본값) · `set_seed(3407)` (저자 하드코딩) |
| 환경 | `.venv-dialstart` (Python 3.11 · torch 2.13 · transformers **4.44.2**) · GPU 1장 |
| 벽시계 | DialSeg711 ≈ 10분 · Doc2Dial ≈ 50분 |

## 2. 결과

| 데이터셋 | 지표 | 우리 | 논문 Table 1 | 차이 |
|---|---|---:|---:|---:|
| DialSeg711 | Pk ↓ | **0.1786** | 0.1786 | 0.0000 |
| DialSeg711 | WD ↓ | **0.1980** | 0.1980 | 0.0000 |
| Doc2Dial | Pk ↓ | **0.3811** | 0.3811 | 0.0000 |
| Doc2Dial | WD ↓ | **0.4072** | 0.4072 | 0.0000 |

원시 출력: `0.17863404…` / `0.19802316…` / `0.38107097…` / `0.40719955…`

## 3. 해석

**네 지표 전부 자리 그대로 일치한다.** 저자 체크포인트 + 저자 코드 + 저자 데이터이므로
당연해야 하지만, 이번 감사에서 실제로 이 수준이 나온 것은 TextTiling 과 DialSTART 둘뿐이다:

| 방법 | 저자 코드 | 저자 데이터 | 저자 체크포인트 | 결과 |
|---|---|---|---|---|
| **DialSTART** | ✅ | ✅ | ✅ | **4/4 지표 정확 일치** |
| TextTiling | ✅ | ✅ | 학습 불필요 | 3/3 셋 재현 |
| CSM | ✅ | ✅ | 🔴 **부재** | 2/3 셋 (dialseg711 74.5%) |
| Embedding Delta | ✅ | 🔴 부재 | ✅ | 저자 제출의 절반 |
| Kurfalı SSC | ✅ | 🔴 부재 | 🔴 **404 소실** | 실행 불가 |

**패턴이 분명하다 — 재현 성패는 방법이 아니라 "세 자산이 다 살아 있는가"로 갈린다.**
DialSTART 가 셋 다 살아 있어서 완벽히 재현됐고, CSM 은 체크포인트 하나가 없어 한 셋에서 무너졌다.

## 4. 🔴 환경 재구성에서 손댄 것 (의존성 문제, 알고리즘 무관)

저자 `requirements.txt`(`torch==1.8.1 · tensorflow==2.6.2 · transformers==4.18.0`)는
Python 3.11 에서 설치되지 않는다(`tokenizers 0.12.1` Rust 빌드 실패).

1. **`keras.preprocessing.sequence` shim** — 저자 코드가 `pad_sequences` 를 거기서 import 한다.
   TF 2.6 시절 그 모듈은 독립 패키지 `keras_preprocessing.sequence` 의 **재노출**이었고 Keras 3 에서
   제거됐다. → `keras-preprocessing` 설치 + shim 으로 이어줌. **원본과 동일 구현.**
   TensorFlow 전체를 구버전으로 설치할 이유가 없다.
2. **transformers 4.44.2 고정** — 5.x 는 `modeling_bert` 에 `__all__` 을 도입해
   저자의 `from ... import *` 가 `Optional` 을 못 받고 `NameError` 로 죽는다(`model.py:170`).
   4.18 은 설치 불가라 그 사이 버전으로 맞췄다.

**두 조치가 무해했다는 증거 = 결과가 논문값과 소수 4자리까지 같다는 사실이다.**

## 5. 판정 및 다음 단계

- ✅ **DialSTART = 재현.** DTS 베이스라인 4종 중 마지막 미확보 항목이 닫혔다.
- ⬜ **온라인 변형 구현** — DTS 베이스라인 중 **유일하게 온라인 판이 없다**
  (TextTiling·CSM·RoBERTa·GraphSeg·GreedySeg 는 `methods/*/online/` 에 이미 존재).
  이 재현이 그 **오프라인 기준선**이 되므로 `Δ = online − offline` 을 같은 채점기로 잴 수 있다.
- 🔴 **자산 보존** — `_upstream/` 은 `.gitignore` 대상이라 **디스크에만 있다**(1.7GB).
  컨테이너가 재생성되면 Drive 에서 다시 받아야 하고, 그 링크는 언제든 죽을 수 있다
  (Kurfalı 전례). 장기 보존이 필요하면 별도 조치.

## 6. 한계 / 검증 미해결

- **저자 학습 과정은 재현하지 않았다.** 배포 체크포인트로 추론만 했다.
  `scripts/train.sh` 재현은 별건이며, 그것까지 해야 "학습부터 재현"이라 말할 수 있다.
- 지표는 **Pk·WD 뿐**이다 — 저자 `test.py` 가 그 둘만 계산한다. F1·Score 는 논문에 없다.
- `RelatedWork.md:66` 의 DialSTART SuperDialseg/TIAGE 수치는 **다른 논문의 재평가**이지
  이 논문의 값이 아니다. 같은 표에 놓지 말 것.
