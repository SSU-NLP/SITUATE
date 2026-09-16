# Temporal/Spatial/Entity 자질 벡터 (Barth & Dönicke 2021) — 재현 불가

**판정: 🔴 재현 불가.** 학습된 분류기는 있는데 **자질을 만들 수가 없다.**

---

## 1. 대조하려던 발표 수치 (Zehe et al. 2021a Table 1)

| | Track 1 P/R/F1 | Track 2 P/R/F1 |
|---|---|---|
| Barth & Dönicke | 0.06 / 0.11 / 0.07 | 0.13 / 0.12 / **0.12** |

## 2. 처음 판정을 정정한다

조사 초기에 "학습된 RandomForest 가 동봉돼 있으니 **추론은 가능**"이라고 봤다. **틀렸다.**
모델 파일은 실제로 있다:

    _upstream/stss/code/pipeline/classifiers/pickled_clfs/
        scene_tagger_RandomForestClassifier_T11_MSL3.pickle    ← .gitignore 에 걸려 있는데도 커밋돼 있음

그러나 그 분류기에 **먹일 자질을 만들 수 없다.**

## 3. 막힌 지점

### GermaNet 이 없다 🔴
자질 추출이 GermaNet 동의어집합(synset)을 직접 쓴다:

    clf_scene_tagger.py:405   GERMANET.provide()
    clf_scene_tagger.py:421   features["germanet_verb_category:" + ... word_class]
    global_resources.py:67    data_path = GERMANET_PATH + "/GN_V150/GN_V150_XML"

그런데 `code/resources/germanet/` 는 `.gitignore` 로 빠져 있고 레포에 없다.
GermaNet 은 튀빙겐대 라이선스 자료라 신청·승인이 필요하다. 사용자 결정(2026-08-03)에 따라
**외부 자료 요청은 하지 않는다.**

### coreference 리소스도 없다 🔴
`settings.py` 의 `COREF_PATH = resources/coreference_resolution` 역시 배포본에 없다.
엔티티 축 자질이 coref 에 의존한다.

### 스택이 2020년산이다 🟡
`spacy==2.3.2` · `torch==1.6.0` · `flair==0.5.1` · `tensorflow` · `germanetpy` ·
HeidelTime(Java). 위 두 문제가 없더라도 Python 3.8 이하 환경을 따로 세워야 한다.
(레포에 동봉된 `code/resources/parsing` 2.5GB 는 UD 파서 모델이라 이건 있다.)

## 4. 방법 요약 (RelatedWork 서술용)

장면의 정의(시간·공간·행위·인물 일관성)를 **자질로 직접 조작화**한 유일한 참가작이다.
NLP 파이프라인으로 시제·시간표현(HeidelTime)·공간 명사·화자·coref·GermaNet 어휘범주를
문장/절 단위로 뽑아 특징 벡터를 만들고, RandomForest 로 전이 타입
(Scene-to-Scene / Scene-to-Nonscene / Nonscene-to-Scene)을 분류한다.
**우리 5축 분해(M3-S5)의 직접 선행 연구**라 RelatedWork 에서 위치가 중요하다.

## 5. 판정을 뒤집으려면
GermaNet GN_V150 라이선스 확보 + coref 리소스 확보 + Python 3.8 환경 구축. 셋 다 필요하다.
