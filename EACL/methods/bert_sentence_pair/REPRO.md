# BERT sentence-pair (Zehe et al. 2021b, "Detecting Scenes in Fiction") — 재현 불가

**판정: 🔴 재현 불가.** 학습 코퍼스가 저작권 자료이고 체크포인트도 없다.

---

## 1. 이 방법의 지위
장면 분절 태스크를 **처음 정의한 논문의 원 베이스라인**이다. 이후 STSS 2021 공유과제와
Zehe et al. (2025) SotA 논문이 모두 이 태스크 정의 위에 서 있다. RelatedWork 에서
"태스크의 출발점"으로 인용해야 하는 자리다.

## 2. 방법 요약
인접한 두 문장을 **문장쌍(sentence pair)** 으로 묶어 `deepset/german-bert` 에 넣고,
그 사이가 장면 경계인지 이진 분류한다. 구현은 FARM 프레임워크. 장면 정의는 시간·공간·
행위·인물 구성의 일관성이며, 경계는 그중 하나가 유의하게 끊길 때 둔다.

## 3. 왜 막혔나

### 학습 코퍼스 🔴
German Scene Dataset 은 저작권 자료다. 배포처(`professor-x.de/german-scene-dataset`)는
현재 `gitlab2.informatik.uni-wuerzburg.de/kallimachos/German-Scene-Dataset` 로 리다이렉트되며,
접근에는 저자 승인이 필요하다(EAN 으로 전자책을 구매해 병합하는 경로 안내).
사용자 결정(2026-08-03)에 따라 **요청하지 않는다.**

### 체크포인트 🔴
저자들이 파인튜닝 가중치를 배포한 기록이 없다. HuggingFace `deepset/german-bert` 는
**사전학습 모델**이지 장면분절 파인튜닝본이 아니다.

### 공식 제출 예측도 없다 🔴
이 논문은 STSS 2021 참가작이 아니라 **과제를 낸 쪽**이므로, embedding-delta 레포에
동봉된 공식 예측 덤프(참가 5팀 중 Schneider 팀 것만)에 포함되지 않는다.
채점기 수준의 우회 재현도 불가능하다.

## 4. 우리가 대신 확보한 것
- 이 논문이 정의한 **공식 채점기**는 재구현·검증 완료다(`../common/stss_eval.py`,
  `../embedding_delta/check_official_submission.py`). 공식 순위표의 Schneider 행을
  두 트랙 모두 정확히 재계산했다.
- 이 논문이 만든 데이터의 **Test-2 2편**은 퍼블릭 도메인이라 전문을 확보했고 md5 검증했다
  (`../common/stss_data.py`). 학습이 필요 없는 방법은 여기서 돌릴 수 있다.
