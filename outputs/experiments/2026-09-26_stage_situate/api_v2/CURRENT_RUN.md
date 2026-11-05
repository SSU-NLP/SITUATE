# 실행 상태와 설정

151편 일반판 전체, seed 1, rewrite / rewrite_axes 두 방법, RSU / ASU 두 조건.
학습된 Qwen3-1.7B LoRA epoch 1 체크포인트를 그대로 사용한다. 두 방법은 Qwen3-8B 상태 생성 절차와 출력 형식이 같고, rewrite_axes에만 SITUATE의 다섯 축 변화 점수를 보조 입력한다. 최초 체크포인트 예측은 공유한다. 요소별 추가·수정·삭제 방법은 이번 실행에서 제외한다.

상태 생성 API는 Qwen3-8B/Alibaba, 평가 API는 GPT-OSS-120B/AkashML로 공급자를 고정한다. 평가자는 low reasoning이며 실제 추론 토큰과 응답별 비용을 usage.jsonl에 남긴다. 논문 원래 평가자 및 디코딩 설정과 다르므로 논문 수치의 재현으로 해석하지 않는다. API 키는 EACL/.env에서 실행 시 읽고 원격 GPU job에 전달하지 않는다.

영어·중국어 각 1편에서 두 방법의 상태 생성 완료. 혼합 공급자로 수행한 최초 평가 연결 점검은 evaluation_routing_probe에 별도 보관하고 최종 집계에서 제외한다. 두 영화의 고정 공급자 평가가 모두 성공해야 나머지 149편을 실행한다. 실패 시 progress.json에 실패한 영화와 단계를 기록한다. 최종 집계는 전체 영화의 평가 파일이 갖춰진 경우에만 summary.json으로 생성한다.

GPU job job-0c3868y645mb: A100 80GB 1장, $1.48/시간. 기존 점수 66편 재사용, 나머지 축 점수만 계산하고 종료한다. 동시에 실행한 VESSL job은 1개다.

실측 기반 비용 추정: 관찰 추출 약 $4.42, 상태 생성 약 $3.71, 평가 약 $4.95. 병합·재시도·영화별 길이 차이와 GPU 비용을 포함한 전체 예상은 $15–20이며 지출 상한 보장은 아니다. 초기에 종료한 두 GPU job 약 $0.85도 포함한 범위다.

현재 실행 상태: progress.json. 영화별 로그: logs/. 응답별 API 사용량: usage.jsonl. 설정: config.json. 변경점: protocol.json. 전체 실행기: scripts/stage/run_full_api.py.
