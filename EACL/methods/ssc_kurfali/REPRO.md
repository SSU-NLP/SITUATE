# SSC (Kurfalı & Wirén 2021, STSS 1위) — 재현 불가

**판정: 🔴 재현 불가.** 학습도 추론도 막혔다. 우리 쪽 실수가 아니라 **자료가 소실·비공개**다.

---

## 1. 대조하려던 발표 수치 (Zehe et al. 2021a Table 1)

| | Track 1 P/R/F1 | Track 2 P/R/F1 |
|---|---|---|
| Kurfalı & Wirén | 0.29 / 0.51 / **0.37** | 0.14 / 0.22 / 0.17 |

두 트랙 통틀어 **STSS 2021 최고 성적**이다(Track 1 기준). 이 방법이 재현되지 않으면
독일어 장면분절 계열의 "이겨야 할 상대"를 우리 손으로 측정할 수 없다.

## 2. 왜 막혔나 (두 경로 모두 차단)

### 추론 경로 — 사전학습 모델이 소실됐다 🔴
저자 `Dockerfile` 은 학습된 AllenNLP 모델을 Google Drive 에서 받는다:

```dockerfile
RUN bash -c "... gdown --id 1yayKtOT2pGD7YQpL-r9p3D-ErMt6rVeR"   # → model.tar.gz
```

2026-08-03 확인 결과 **해당 파일이 404** 다.

```
https://drive.google.com/uc?id=1yayKtOT2pGD7YQpL-r9p3D-ErMt6rVeR  →  HTTP 404
```

레포 어디에도 대체 체크포인트가 없고, HuggingFace 에도 이 모델은 없다.

### 학습 경로 — 학습 코퍼스가 저작권 자료다 🔴
STSS Train 20편은 현대 통속소설(Heftroman)이라 공개되지 않는다. 우리에게 있는 것은
경계 오프셋 standoff 뿐이고 원문이 없다. 사용자 결정(2026-08-03)에 따라 **저자에게
코퍼스를 요청하지 않는다** → 학습으로 우회하지 않는다.

## 3. 그래도 확보한 것

- 저자 코드 전체(`_upstream/scene_segmentation/`) — Sequential Sentence Classification
  모델 정의·데이터리더·predictor·학습/추론 스크립트. 자료가 생기면 그대로 돌아간다.
- **저자의 공식 제출 예측은 없다.** embedding-delta 레포가 동봉한 공식 덤프
  (`stss_results/`)는 **Schneider 팀 자신의 예측**뿐이라 Kurfalı 예측은 그 안에 없다.
  즉 채점기 수준의 우회 재현도 불가능하다.

## 4. 방법 요약 (RelatedWork 서술용)

문장 분류를 **문서 순차 태깅**으로 푼다. SciBERT 계열 Sequential Sentence Classification
아키텍처(Cohan et al. 2019)를 장면 경계에 적용해, 문장 임베딩 시퀀스 위에서 각 문장이
장면 시작인지 분류한다. 저자들은 원 SSC 의 손실을 MSE 로 바꿔 경계의 희소성에 맞췄다.
Track 1 에서 재현율 0.51 로 다른 팀을 크게 앞섰다(다음이 0.13).

## 5. 판정 근거를 뒤집으려면

- Google Drive 사본이 어딘가(저자 개인 페이지·논문 부록·Zenodo)에 있는지 재확인
- 저자 문의로 체크포인트 확보 — **현재 사용자 결정으로 보류**
