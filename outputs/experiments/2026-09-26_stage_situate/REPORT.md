> 현재 실행은 [api_v2/CURRENT_RUN.md](api_v2/CURRENT_RUN.md)를 따른다. 비교는 rewrite/rewrite_axes 두 방법, seed 1이며 Qwen3-8B API와 GPT-OSS-120B 평가자를 사용한다. 아래의 4방법·GPU 생성·Qwen3-32B 평가 계획은 실행 중단된 초기 설계 기록이다.

# STAGE에서 SITUATE 축별 변화 예측기 검증

## 실험 설정

목적은 상태를 항목별로 보존하는 효과와, 기존 SITUATE가 학습한 다섯 축의 변화 신호를
추가하는 효과를 구분하는 것이다. EACL/EACL_Paper.tex의 분절 모델을 인물 상태 추적으로
확장하기 위한 실험이며, 기존 논문의 분절 성능과 이 실험의 상태 추적 성능은 다른 과제다.

STAGE v9의 Character Development Tracking 일반 조건을 사용한다. 공식 저장소 커밋은
`809ea665783879da77f4f30bf7f5be771525f514`이며, 151개 영화(영어 109, 중국어 42),
434개 인물 궤적, 2,925개 체크포인트다. 일반 조건만 평가하며 익명화 조건의 결과로
확대 해석하지 않는다. 영어 결과를 우선 확인하고 중국어와 전체 결과를 별도로 보고한다.

공식 배포본에는 대본이 없어 이전 공식 공개본
`roytian1992/STAGE_v0@d902abfa51bf71ac10f53313a7add8b6ad5b1092`에서 확보했다.
원문 바이트 또는 장면 ID 정규화·공개 제목 정규화·JSON 직렬화 조합 중 현재 정답이
요구하는 SHA-256과 정확히 일치하는 것만 채택했다. 151개 전부 일치한다.
`previous_release_reconstruction.json`에 파일별 변환과 해시를 기록했다.
`data/stage_task1_inputs`는 이 대본과 Task I 정답의 실행용 사본이다. 정답 내용은 보존하고
언어 필드만 `English/Chinese`에서 공식 실행기가 요구하는 `en/zh`로 정규화했다.

변화 예측기는 최근 완료된 Qwen LoRA 체크포인트를 그대로 동결해 사용한다.

- Backbone: Qwen/Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- LoRA: rank 16, alpha 32, q_proj/v_proj, dropout 0; seed 1, 1 epoch.
- 기존 학습: 10,430 train / 1,406 validation, head LR 3e-4, adapter LR 2e-5.
- Pooling: 문장별 마지막 유효 토큰, FP32 L2 정규화, chat template 미사용.
- 소프트웨어: 학습 때와 같은 Transformers 4.56.2, PEFT 0.17.1, tokenizers 0.22.2.
- 전체 head 2,289,503개 파라미터 중 변화 추론에 필요한 693,709개만 사용한다.
  학습 보조 state/dynamics/readout head와 분절 임계값은 호출하지 않는다.
- 문장 순서대로 recurrent state를 유지하고 영화가 바뀔 때만 초기화한다.
  장면 안의 문장 전이 확률 평균을 다섯 축의 점수로 사용한다. 첫 문장은 전이가 없으며
  근거 없이 0의 변화 확률을 만들지 않는다. 영어는 NLTK punkt, 중국어는 문장부호로 분리한다.
- 첫 checkpoint와 이전 상태가 빈 경우는 모든 방법이 같은 재작성 프롬프트를 사용한다.
  동일 프롬프트의 검증된 응답은 공유하므로 최초 상태 차이가 비교에 섞이지 않는다.

생성기는 Qwen3-8B, 채점기는 별도 가중치인 Qwen3-32B다. 두 revision은
`model_revisions.json`에 고정했다. 두 모델의 tokenizer.json SHA-256은 동일하다.
생성과 채점은 temperature 0, thinking off, bf16, context 24,000,
output reserve 3,072, safety reserve 1,024, chat overhead reserve 256이다.
모든 방법에 같은 3회 형식 재시도 상한을 적용하고 실패를 정상 예측으로 대체하지 않는다.
공식 채점기의 추가 fallback도 같은 독립 Qwen3-32B를 사용하며 해당 호출은 원본 로그에 남긴다.

## 비교 조건과 입력 통제

| 조건 | 상태 갱신 | 변화 신호 |
| --- | --- | --- |
| rewrite | 공식 프롬프트로 전체 재작성 | 없음 |
| edit | remove/replace/add를 적용하고 언급하지 않은 기존 항목은 그대로 유지 | 없음 |
| edit_axes | edit과 같은 연산 | 동결 SITUATE의 축별 점수 |
| edit_permuted_axes | edit과 같은 연산 | 축 대응만 한 칸 순환 이동한 점수 |

네 조건 모두 RSU(정답 이전 상태)와 ASU(자기 예측 이전 상태)를 평가한다. 공식 runner의
600-token scene-aligned chunk 추출·통합으로 얻은 인물별 구간 메모리를 모든 조건이 공유한다.
메모리 추출기에 정답 이전 상태를 주지 않는다. Actor에는 이전 상태의 자연어 claim만
제공하며 state_id, checkpoint_id, 평가용 정답과 미래 장면은 제공하지 않는다.
변화 점수도 해당 구간 메모리에 인용된 장면만 제공한다.

edit은 갱신 연산마다 구간 근거를 요구한다. 변화가 없는 항목을 자동 삭제하지 않으며,
최종 현재 상태는 공식 제한인 최대 8개, 구간 변화는 최대 4개다. 공간 부족을 이유로 임의로
뒤쪽 항목을 잘라내지 않는다. 모델이 명시적으로 선택한 삭제·교체만 적용한다.
다섯 축은 지식·관계·제약과 일대일 대응하지 않으므로 점수를 항목 갱신의 참고 정보로
제공한다. 확률 임계값으로 지식·관계를 강제 삭제하거나 저점수 축의 갱신을 금지하지 않는다.
따라서 이번 실험은 기존 축별 변화 신호의 전이 가능성을 평가하며, 학습된 항목별 hard gate의
효과를 검증했다고 주장할 수 없다.

## 평가와 집계

공식 `run_task1_state_update_evaluation.py`를 그대로 사용한다. Current-State Coverage,
Development Coverage 및 각각의 Contradiction Rate를 RSU/ASU별로 보고한다.
Unsupported/temporal-validity 등은 진단 지표다. 체크포인트를 독립 표본으로 합치지 않고
공식 인물 내 집계와 영화 내 집계 후 영화별 동등 가중 평균을 사용한다.
영화 단위 paired bootstrap 1,000회(seed 20260727)로 95% 신뢰구간을 계산한다.
정의되지 않는 지표는 0으로 바꾸지 않고 해당 지표에 사용된 영화 수를 함께 보고한다.

주요 대비는 edit−rewrite, edit_axes−edit, edit_axes−edit_permuted_axes다.
첫 대비만 개선되면 항목 유지 효과이며 SITUATE 신호의 효과로 해석하지 않는다.
Coverage와 함께 Contradiction Rate가 악화되는지도 확인한다. 결과를 보고 임계값이나
대상 영화를 고르지 않는다.

## 실행과 검증

VESSL 동시 job 최대 2개, GPU 합계 최대 4개. 각 job은 A100 SXM 80GB 두 개를 사용한다.
입력 manifest를 기준으로 두 shard를 나누고, 각 job 안에서 encoder → actor → judge를
순차 실행한다. 영어·중국어에서 대본 바이트 수가 가장 작은 영화 하나씩을 사전 선택해
각 shard의 첫 end-to-end 검사로 사용한다. 이 영화들도 최종 집계에 포함한다.
최초 생성 및 채점이 통과해야 전체 shard로 확장한다. GPU job의 코드·입력은 단일 tarball과
SHA-256으로 고정한다. 모호한 제출 응답은 자동 재제출하지 않는다.

확인한 사항:

- 공개 파일 2,318개 체크섬 및 일반·익명화 Task I paired asset 검증 통과.
- 일반 조건 대본 151개 모두 정답의 SHA-256과 일치.
- 일반 조건 151개 영화의 공식 입력 사전 검사 통과(LLM 호출 0회).
- 실제 체크포인트 추론과 추출 모듈의 최대 절대 오차 5.96e-8.
- 구간 분할/연결 추론 일치 및 미래 입력 변경에 대한 과거 출력 불변성 확인.
- 6개 단위 검사, 8개 영어·중국어 입력 계약 검사 통과.
- 임시/정답 필드 변조가 actor 프롬프트에 영향을 주지 않는 것을 검사.

재현 명령은 `scripts/stage/`에 있다. 핵심 파일은 `axis_predictor.py`, `extract_axes.py`,
`state_edits.py`, `run_tracking.py`, `run_evaluation.py`, `summarize.py`이며
VESSL 실행기는 `scripts/vessl/job_stage_tracking.sh`다.

## 결과 및 판정

아직 STAGE 성능 수치는 없다. CPU 검증 결과는 모델의 과제 성능이 아니다.
최종 결과를 확보하기 전에는 개선 여부, silent forgetting 감소, 논문 기여를 확정하지 않는다.

## 한계 및 검증 미해결

- 논문의 DeepSeek-V4-Pro 채점과 다른 채점기를 쓴다. 논문의 절대 수치와 직접 비교하지 않는다.
- 생성기와 채점기가 같은 모델 계열이므로 채점 편향과 사람과의 일치도는 추가 검증이 필요하다.
- 익명화 대본은 확보하지 않았다. 작품 정체성·사전학습 기억에 대한 통제는 이번 실험에 없다.
- 하나의 동결 seed/epoch 체크포인트를 사용하므로 학습 seed 변동성은 추정하지 않는다.
- 영어 소설로 학습한 문장 예측기를 대본, 특히 중국어에 전이한다. 문장 분리와 장면 평균은
  이 실험의 새로운 인터페이스이며 기존 여섯 분절 벤치마크와 동일한 분포가 아니다.
- 장면 평균과 고정 축 순열은 제한된 대조다. 효과가 관측돼도 축별 항목 대응을 학습한 모델이나
  변화 시점 자체를 통제한 실험을 대체하지 않는다.
- GPU 환경의 실제 생성·채점 동작과 최종 완주 여부는 아직 검증 중이다.

출처: [STAGE v9](https://arxiv.org/html/2601.08510v9),
[최신 공식 코드](https://github.com/roytian1992/STAGE_v0),
[이전 공식 공개본](https://github.com/zhuiwen-ai/STAGE).
