# DialSTART (Gao et al. 2023, SIGIR — arXiv:2305.02747) — 재현 결과

**판정: ✅ 완전 재현. 2/2 데이터셋 · 4/4 지표가 소수 4자리까지 일치한다.**

논문: *Unsupervised Dialogue Topic Segmentation with Topic-aware Utterance Representation*
레포: `https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/dial-start`

---

## 1. 대조표 (논문 Table 1)

| 데이터셋 | 지표 | **우리** | 논문 | 차이 | |
|---|---|---:|---:|---:|---|
| **DialSeg711** (711대화) | Pk ↓ | **0.1786** | 0.1786 | 0.0000 | ✅ |
| | WD ↓ | **0.1980** | 0.1980 | 0.0000 | ✅ |
| **Doc2Dial** (4,135대화) | Pk ↓ | **0.3811** | 0.3811 | 0.0000 | ✅ |
| | WD ↓ | **0.4072** | 0.4072 | 0.0000 | ✅ |

원시 출력(반올림 전): DialSeg711 `pk 0.17863404…` `wd 0.19802316…` ·
Doc2Dial `pk 0.38107097…` `wd 0.40719955…`

**네 지표 전부 논문값과 자리 그대로 같다.** 저자 배포 체크포인트를 저자 코드로 저자 데이터에
돌렸으니 당연해야 하지만, 이번 감사에서 이게 실제로 된 건 TextTiling 과 이것뿐이다.
**이 레포의 재현성은 예외적으로 좋다.**

## 2. 왜 이게 가능했나 — 세 자산이 전부 살아 있다

착수 시점(2026-08-03) 확인:

| 자산 | 위치 | 상태 |
|---|---|---|
| 코드 | `_upstream/DAMO-ConvAI/dial-start/` (sparse clone) | ✅ |
| **학습된 모델** | Google Drive `1HhN_Gr3vgX6haFLCjIzainvBPVl40MNK` → `model/{model_711,model_doc}` (각 847MB) | ✅ |
| 데이터 | Google Drive `1Ttd83n2KMGErrioDDruuQonaTuHMFGMR` → `data/{dialseg711,doc2dial}` | ✅ |

🔴 **Drive 링크는 죽는다.** Kurfalı(STSS 1위) 체크포인트가 정확히 이 방식으로 404 가 됐다
(`../ssc_kurfali/REPRO.md`). 그래서 착수 즉시 전부 내려받아 저장소 안에 고정했다.
`_upstream/` 은 `.gitignore` 대상이므로 **디스크에만 있다** — 컨테이너가 날아가면 다시 받아야 한다.

## 3. 실행 조건

| 항목 | 값 |
|---|---|
| 명령 | `python test.py --model model_711 --dataset 711 --single_ckpt --ckpt ./model/model_711` (저자 `scripts/test.sh` 의 *"reproduce our paper result"* 줄 그대로) |
| 코드 수정 | **없음** |
| 데이터 | `data/dialseg711/` 711개 대화 (저자 배포본) |
| 지표 | `segeval.pk` · `segeval.window_diff` (저자 `test.py` 가 직접 호출) |
| 윈도우 | `--window_size 2` (기본값) |
| seed | `set_seed(3407)` (저자 하드코딩) |
| 환경 | `.venv-dialstart` · GPU 1장 |
| 벽시계 | DialSeg711 711대화 ≈ 10분 |

## 4. 🔴 환경 재구성에서 손댄 것 (전부 의존성 문제, 알고리즘 무관)

저자 `requirements.txt` 는 `torch==1.8.1 · tensorflow==2.6.2 · transformers==4.18.0` 인데
Python 3.11 에서 그대로 설치되지 않는다(`tokenizers 0.12.1` 이 Rust 빌드 실패).

1. **`keras.preprocessing.sequence` shim** — `data_preprocess.py`/`test.py` 가
   `from keras.preprocessing.sequence import pad_sequences` 를 한다. TF 2.6 시절 그 모듈은
   독립 패키지 `keras_preprocessing.sequence` 를 **그대로 재노출**한 것이었고 Keras 3 에서 제거됐다.
   → `keras-preprocessing` 을 설치하고 `keras/preprocessing/sequence.py` shim 으로 이어줬다.
   **원본과 동일한 구현**이 쓰인다. TensorFlow 전체를 구버전으로 설치할 이유가 없다.
2. **transformers 는 4.44.2 로 고정** — 최신(5.x)에서는 `model.py:170` 의
   `from transformers.models.bert.modeling_bert import *` 가 `Optional` 을 더 이상 노출하지 않아
   `NameError` 로 죽는다(5.x 가 `__all__` 을 도입했기 때문). 4.18 은 설치 불가라 그 사이 버전으로 맞췄다.

두 조치 모두 **모델 가중치·전처리·지표에 영향을 주지 않는다**. 결과가 논문값과 소수 4자리까지
일치한다는 사실이 이를 뒷받침한다.

## 5. 다음 단계

- ⬜ **온라인 변형 구현** — DTS 베이스라인 중 유일하게 온라인 판이 없다
  (TextTiling·CSM·RoBERTa·GraphSeg·GreedySeg 는 `methods/*/online/` 에 이미 있다).
  이 재현이 그 **오프라인 기준선**이 되므로 `Δ = online − offline` 을 같은 채점기로 잴 수 있다.

## 6. 🔴 서드파티 인용값과 혼동 금지

`RelatedWork.md:66` 에 DialSTART 의 SuperDialseg/TIAGE 수치
(Pk 0.373/0.482 · WD 0.412/0.528 · F1 0.627/0.378 · Score 0.617/0.437)가 인용돼 있는데,
그건 **다른 논문의 재평가**다. DialSTART 자신은 **DialSeg711 과 Doc2Dial 에서 Pk·WD 만** 보고한다.
이 문서의 대조는 **저자 논문 Table 1** 기준이다. 두 계열을 같은 표에 놓지 말 것.

또 `benchmarks/Def-DTS/src/autoseg.py:591` 의 `load_dialstart()` 는 존재하지 않는
`dialstart_reproduce/` 를 읽는 **서드파티 dead code** 다. 재사용 금지.
