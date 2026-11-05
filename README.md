# SITUATE

> 논문 정합성 수정 중: 전체 원본 학습·평가 데이터 확보 전입니다. 논문 모델의 기본 체크포인트는 아직 없으며, 기존 I2 학습본을 자동으로 대신 사용하지 않습니다.

EACL 2027 논문 「SITUATE: Multidimensional Situation Tracking for Online Narrative Event Segmentation」의 5축 상황추적 서사 분절입니다. 인물·공간·시간·인과·의도를 학습하며 Hi-OnTop 코사인 분절기를 가져오지 않습니다.

## 설치 및 실행

```sh
uv sync --frozen --extra test --extra annotation
uv run situate segment examples/input.txt --checkpoint runs/paper.pt
uv run pytest -q
```

Python 3.11을 사용합니다. 입력 한 줄이 문장 하나입니다. 최초 실행 때 동결 MPNet 인코더가 다운로드됩니다. 기본 CPU 실행에는 API 키가 필요 없습니다. CUDA 환경은 --device cuda로 지정할 수 있으나 이번 검증 환경은 macOS CPU입니다.

```sh
uv run situate segment embeddings.npy --embeddings --checkpoint runs/paper.pt
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
uv run situate annotate story.json states.json --model <served-model-id> --base-url <openai-compatible-url> --tokenizer <matching-tokenizer-id>
uv run situate prepare states.json data/train/story.npz --profile paper
uv run situate train data/train data/val runs/paper.pt --profile paper --seed 0
uv run situate evaluate evaluation.json
```

annotate는 전체 이야기에서 닫힌 어휘를 뽑고, 축마다 독립 순서로 섞어 20문장씩 v3.1 프롬프트에 전달합니다. temperature=0, enable_thinking=false를 사용합니다. 호출할 때만 외부 서버에 이야기가 전송되며 이번 검증에서는 외부 주석 API를 호출하지 않았습니다. 반환 항목 누락·중복·범위 오류는 실패 처리합니다. prepare는 원래 순서에서 결정론적 변화 타깃을 만들고 동결 인코더로 임베딩합니다. 이미 embeddings가 있으면 재인코딩하지 않습니다.

학습 입력 NPZ에는 embeddings[T,768], resets[T-1,5], states[T,3], group 문자열, profile 문자열이 필요합니다. states 열은 기존과 동일한 perspective(4: first/second/third/none), temporal_shift_marker(4), event_type(10)이며 target_schema=perspective-v1을 명시합니다. 잘못 생성했던 protagonist_present-v1 보조 라벨은 호환되지 않습니다. 인물 변화 타깃은 계속 protagonist_present의 변화입니다. 디렉터리는 먼저 생성하세요. train/val의 동일 group 또는 동일 임베딩 문서는 차단합니다. 논문 데이터의 공식 분할을 별도로 제공해야 하며 테스트셋을 학습·선택에 사용하지 마세요.

기본 Adam 3e-4, 100 epochs, patience 3, batch 48입니다. 4개 seed는 각각 다른 출력 경로로 실행합니다. 기존 출력은 덮어쓰지 않습니다. 전체 11,836편 데이터와 논문 4-seed 재학습은 이 저장소에 포함되지 않습니다. validation 문서 평균 손실로 조기 종료합니다.

## 이력

이전 혼합 저장소 전체는 `archive/pre-separation-2026-11-05/main`에 있습니다. main의 모델 파일은 원본과 동일하며 별도 다운로드가 필요 없습니다. SOURCE_PROVENANCE.json으로 체크포인트와 논문 SHA-256을 대조할 수 있습니다.

## 논문 전체 재학습 실행 경로

```sh
uv run situate reproduce data/train data/val data/eval/manifest.json runs/paper-four-seed --device cuda
```

원본 10,430/1,406편 및 6편 평가 manifest가 없거나 타깃 스키마가 이전 버전이면 중단합니다. manifest는 id, embeddings(.npy 상대 경로), gold(문장 사이 T-1개 binary consensus 라벨)를 갖는 객체 6개의 JSON 배열입니다. 4개 seed 학습 후 각각 Score/BOR/Lift와 예측 개수에 맞춘 비교군을 기록합니다. 원문이 명시하지 않은 seed 번호 0~3과 손실 계수 등은 구현 선택으로 plan.json에 구분합니다. 이 명령은 LongBench 하류 실험을 포함하지 않습니다.

기존 학습본 확인은 `--checkpoint src/situate/checkpoints/pretrained_i2.pt`로 명시적으로만 가능합니다. 이는 논문 모델의 성능 검증을 대신하지 않습니다.
