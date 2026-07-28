# REPORT — SHARE frozen M3(S5) fitness probe (valid, 25 docs)

## 1. 실험 setup

- **목적.** "서술 없이 대사만으로 상황전환(씬 경계) 감지"가 되는가 — frozen M3(S5)를 SHARE 대사-only 스트림에 전이해 M1-CosDrift와 비교. EISM 5축이 서술문 채널을 잃으면 어떻게 되는지의 스트레스 테스트.
- **데이터.** SHARE(ACL 2025) valid, 화자쌍별 세션 시퀀스. 메타데이터(persona/temporary-event/label/session 마커) 전부 스트립, 대사 텍스트만 입력. gold 경계 = 세션(씬) 전환. 빌더 `scripts/eacl_phase0/build_share_pack.py` (seed=20260714 샘플 30, >=15턴). **실제 평가 = 5축 완전 완료 25 docs / 1406 전이 / 143 경계**(사용자 결정으로 25에서 종료).
- **방법.** M3-S5 = frozen(narrative-fiction용, 무변경): 2-stage 닫힌 enum, target-only 추출, 변화판정 100% 코드, place/goal carry-forward. coordinated count = 5축 리셋합(protagonist_perspective+place+time+causality+goal), deployable = count≥t\*(=3 동결). M1-CosDrift = CosDrift-raw(1−cos 인접), deployable = μ+cσ(c=1.0 동결).
- **지표.** 메인 = DTS Score(`dts_scoring.score_dialogues`), per-doc 평균 + **doc-level bootstrap CI(n=25, 10000 resample)**. 보조 = AUC, 축별 lift.
- **인코더.** paraphrase-multilingual-mpnet-base-v2, 768D, L2정규화(전 방법 공통).
- **스크립트.** 러너 `llm_s5_states.py`(claude-fable-5, temp 0), reset_matrix 재구성 `reconstruct_reset_matrix.py`(캐시→matrix, LLM 0), 평가 `share_probe_eval.py`.

## 2. 결과

### 사전 baseline (M1-CosDrift, 322 docs 전체 valid, LLM 무관)

- deployable Score 0.357, AUC 0.589, 경계-내부 margin +0.047(≈mouthguard 0.04). → 씬 전환은 **concat-easy 아님**(임베딩 큰 점프 아님). 층위1 우려 미현실화.

### 메인 — DTS Score (25 docs, per-doc mean ± std, bootstrap CI)

| Score | M3-S5 | M1-CosDrift | gap [95% CI] |
|---|---|---|---|
| deployable | 0.335 ± 0.102 | 0.311 ± 0.121 | +0.024 [−0.034, +0.082] |
| matched@gold (보조) | 0.357 ± 0.145 | 0.365 ± 0.138 | −0.007 [−0.093, +0.079] |

**두 렌즈 모두 95% CI가 0을 포함 → 유의차 없음(무승부).**

### 보조 — AUC / 축별 진단

- AUC: M3 0.568 ± 0.192 / M1 0.545 ± 0.177 (둘 다 near-chance).

| 축 | AUC | 경계율 | 내부율 |
|---|---|---|---|
| protagonist(perspective) | 0.498 | 0.411 | 0.414 |
| place | 0.527 | 0.104 | 0.049 |
| time | 0.507 | 0.020 | 0.006 |
| causality | 0.541 | 0.183 | 0.102 |
| goal | 0.529 | 0.343 | 0.286 |

## 3. 해석

- **부정 결과(무승부)이나 축별 해석이 갈린다 — place/time은 측정-가설 교란.**
- **🔴 place/time = 가설 미검정(측정 한계).** SHARE엔 장소·시간 ground-truth가 없다(스트립 메타데이터 = persona/temporary-event/shared-memory/mutual-event, 전부 사건·관계 요약이고 장소 無; 원 대본 씬 헤딩은 파싱 때 소실). 대사에 장소 명시 6.1%(94% unspecified)뿐이나 stage1은 이야기당 median 5개 장소를 찾음 → **LLM이 못 뽑는 게 아니라 개별 문장에 장소가 없음.** SHARE 세션=영화의 다른 씬이라 원 대본 기준 세션 경계에서 place/time이 실제 바뀔 개연성 높음(대사가 말 안 할 뿐). ⇒ **"안 바뀜"인지 "바뀌는데 대사에 안 보임"인지 구분 불가** → 가설에 무정보. (초안의 "place/time 가설 확증"은 철회.)
- **protagonist = 구조적 死** (화자쌍 2인 고정, 경계율≈내부율 0.41).
- **causality/goal = 상대적으로 정보값 있음** (대사에 표현되는 축인데도 순위질 약, AUC ≤0.54). 단 gold=세션경계가 진짜 사건전환인지 미검증.
- **설계 함의:** place가 死한 원인은 추출기 결함이 아니라 **상태가 텍스트에 없음**. P0-S1식 "변화 추론 추출"로 바꿔도 없는 신호를 만들 수 없고 만들면 halo 오염 → target-only + 코드판정 원칙을 지지.

## 4. 판정

- **SHARE null = 가설 반증 아님. "대사-only 모달리티에선 축 복원/gold 구성 불가"(측정·데이터 한계).** 모달리티 scope 관찰로만 인용, H1 검정 입력에서 제외.
- **진짜 구분하려면** 원 영화 대본의 씬 슬러그라인(INT/EXT+장소+시간)을 외부에서 확보해 세션 경계와 대조 필요 (SHARE 자체론 불가).
- 다음 선택(미결): (A) SHARE는 모달리티 scope 관찰로만 남기고 종결 / (B) 원 대본 확보해 place/time gold 구성 후 재검정 / (C) 대사용 M3 적응(설계 결정 → codex).

## 5. 한계 / 검증 미해결

1. **25 docs**(사용자 결정 조기 종료). n=25로 CI 성립, 24→25 doc 수치 불변으로 견고. 전체 322 docs 미실행.
2. **dev(valid)만.** test split 봉인 유지.
3. **frozen narrative-fiction 도구의 전이.** dialogue용 미적응 — (B) 미검토.
4. gold = 세션(씬) 경계 = 시나리오 구조 메타데이터(층위1). 각본가 저작 상황전환이나 LLM 가공 라벨과 별개로 세션 마커 자체는 script 구조 유래.
