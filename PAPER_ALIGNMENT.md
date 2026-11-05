# 논문·학습본·재학습 경로 대조

기준: 사용자가 제공한 SITUATE EACL 2027 PDF, 8페이지. 논문 p2, Appendix B의 식과 표를 기준으로 분리했습니다.

| 항목 | 제공 학습본 pretrained-i2 | paper 재학습 |
|---|---|---|
| 동결 인코더 | paraphrase-multilingual-mpnet-base-v2, 768 | 동일 |
| 축 | protagonist/space/time/causality/intentionality | 동일 |
| 축별 입력·상태 | 768→32→8, 상태128 | 동일 |
| 공유 게이트·갱신·정규화 | 원본 FactoredSlot 그대로 | 동일 |
| 변화 헤드 | 이전/현재/차이/곱→축별 sigmoid | 동일 |
| 의도 타깃 | carry 없음, none→goal만 | none carry, 유효 goal 간 변화 |
| 자기지도 예측 | 다음 관측 임베딩(obs) | 다음 잠재상태(slot) |
| 양성 가중치 | [2.2884473,8.4251316,30,1,7.4308626] | 비가중 BCE(모두1) |
| 예측 손실 축 집계 | 관측 제곱오차 합 | 원본 축 평균×5로 논문 축 합 구현 |
| 첫 실제 전이 | 억제 후 prefix 임계 | 논문 t≥2 조건대로 포함 |
| 임계 통계 | 현재 전이 포함 평균+1×모표준편차 | 동일 |
| 보조 절대상태 | perspective4 / time4 / causality10 | 동일, 논문의 protagonist 상세 라벨은 원본 구현을 따름 |

checkpoint 실제 인자를 읽어 위 차이를 확인했습니다. 원본 config의 pred_space·pos_weight 설명과 실제 체크포인트가 일치하지 않아 config 문구를 복사하지 않았습니다. 논문 μ_t/σ_t는 prefix 통계로 해석했으며 현재 전이 포함 여부 및 첫 전이 규칙은 명시했습니다. 기본 학습본 경계는 원본 online_nf와 동일하게 유지합니다.

인물 존재 변화, 공간 carry·유효 장소 변화, 시간 sequence/jump/simultaneity, 인과 event_type 변화 또는 inceptive는 공통입니다. gamma=.05, beta=.25, variance floor=.1, alpha=(1,.5,1,.4,.25), dropout=.4는 원본 구현 기본값을 유지합니다. 논문이 명시하지 않은 수치를 논문 확정값으로 주장하지 않습니다.

학습본 1,428,063 파라미터. 원본 전체 model state를 보존하기 위해 학습 시 사용하지 않는 관측 readout도 paper 구조에 남아 있습니다. 기본 checkpoint의 출처는 원본 저장소 기록이며 학습 데이터 전체를 이번 작업에서 재검증한 것은 아닙니다.

**완료한 검증:** 독립 설치, 실제 문장→동결 인코더→경계, 원본 확률·경계 비교, prefix 인과성, 두 라벨 규칙, 합성 데이터 1epoch 학습·재로딩, train/val 중복 차단.
**미실행:** 외부 Qwen 주석, 전체 코퍼스 재학습, 4seed 논문 수치 재현, LongBench/EM-LLM 재평가. 이 항목을 완료했다고 표시하지 않습니다.
