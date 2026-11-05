# SITUATE

EACL 2027 논문 「SITUATE: Multidimensional Situation Tracking for Online Narrative Event Segmentation」의 5축 상황추적 서사 분절입니다. 인물·공간·시간·인과·의도를 학습하며 Hi-OnTop 코사인 분절기를 가져오지 않습니다.

## 설치 및 실행

```sh
uv sync --frozen --extra test --extra annotation
uv run situate segment examples/input.txt
uv run pytest -q
```

Python 3.11을 사용합니다. 입력 한 줄이 문장 하나입니다. 최초 실행 때 동결 MPNet 인코더가 다운로드됩니다. 기본 CPU 실행에는 API 키가 필요 없습니다. CUDA 환경은 --device cuda로 지정할 수 있으나 이번 검증 환경은 macOS CPU입니다.

```sh
uv run situate segment embeddings.npy --embeddings
uv run situate segment examples/input.txt --checkpoint runs/paper.pt
```

임베딩은 [문장수, 768]이며 L2 정규화합니다. `boundaries`는 새 구간 첫 문장 인덱스(0부터), `segments`는 [시작, 끝 제외], `axis_probs`는 전이별 5축 확률, `scores`는 그 합입니다. 결과 구간을 메모리 시스템에 전달할 수 있습니다. EM-LLM 자체와 LongBench 데이터는 포함하지 않습니다.

## 반드시 구분할 두 설정

- `pretrained-i2`: 원본 M5.pt를 그대로 보존한 실행 가능한 학습본. 의도는 carry 없이 none→goal만 변화로 취급하고 다음 임베딩을 예측하도록 학습되었습니다. 원본 온라인 임계의 첫 전이 억제도 유지합니다.
- `paper`: 논문 규칙을 적용한 재학습 경로. 의도 carry-forward, valid goal 변화, 다음 잠재상태 예측 손실을 사용합니다. **이 설정으로 새로 학습된 완전한 모델은 아직 없습니다.** 짧은 합성 학습으로 저장·재로딩만 검증했습니다.

기존 학습본을 논문 설정의 체크포인트로 표시하지 않습니다. 전체 차이는 PAPER_ALIGNMENT.md를 참조하세요.

## 주석 → 학습 데이터 → 학습

입력 JSON: `{"group":"author_or_source_id","sentences":["sentence 1","sentence 2",...]}`.

```sh
# 사용자가 준비한 Qwen3.6-35B-A3B FP8 호환 서버와 OPENAI_API_KEY 환경변수 필요
uv run situate annotate story.json states.json --model <served-model-id> --base-url <openai-compatible-url>
uv run situate prepare states.json data/train/story.npz --profile paper
uv run situate train data/train data/val runs/paper.pt --profile paper --seed 0
uv run situate evaluate evaluation.json
```

annotate는 전체 이야기에서 닫힌 어휘를 뽑고, 축마다 독립 순서로 섞어 20문장씩 v3.1 프롬프트에 전달합니다. temperature=0, enable_thinking=false를 사용합니다. 호출할 때만 외부 서버에 이야기가 전송되며 이번 검증에서는 외부 주석 API를 호출하지 않았습니다. 반환 항목 누락·중복·범위 오류는 실패 처리합니다. prepare는 원래 순서에서 결정론적 변화 타깃을 만들고 동결 인코더로 임베딩합니다. 이미 embeddings가 있으면 재인코딩하지 않습니다.

학습 입력 NPZ에는 embeddings[T,768], resets[T-1,5], states[T,3], group 문자열, profile 문자열이 필요합니다. states 열은 perspective(4), temporal_shift_marker(4), event_type(10)입니다. 디렉터리는 먼저 생성하세요. train/val의 동일 group 또는 동일 임베딩 문서는 차단합니다. 논문 데이터의 공식 분할을 별도로 제공해야 하며 테스트셋을 학습·선택에 사용하지 마세요.

기본 Adam 3e-4, 100 epochs, patience 3, batch 48입니다. 4개 seed는 각각 다른 출력 경로로 실행합니다. 기존 출력은 덮어쓰지 않습니다. 전체 11,836편 데이터와 논문 4-seed 재학습은 이 저장소에 포함되지 않습니다. validation 문서 평균 손실로 조기 종료합니다.

## 이력

이전 혼합 저장소 전체는 `archive/pre-separation-2026-11-05/main`에 있습니다. main의 모델 파일은 원본과 동일하며 별도 다운로드가 필요 없습니다. SOURCE_PROVENANCE.json으로 체크포인트와 논문 SHA-256을 대조할 수 있습니다.
