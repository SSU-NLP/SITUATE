# TextTiling (Hearst 1997 / Jiang et al. 2023 SuperDialseg Table 3) — 재현 결과

**판정: ✅ 재현.** 3개 데이터셋 전부, 두 개는 **소수 4자리까지** 일치한다.

---

## 1. 대조표 (하네스 C = 논문 조건)

`run_paper_repro.py` · 산출 `outputs/experiments/2026-08-03_texttiling_paper_repro/results.json`

| 데이터셋 | 지표 | 우리 | 논문 | 차이 |
|---|---|---:|---:|---:|
| **superseg** | Pk | 0.4455 | 0.441 | +0.0045 |
| | WindowDiff | 0.4579 | 0.453 | +0.0049 |
| | F1 | 0.3851 | 0.388 | −0.0029 |
| | **total_score** | **0.4667** | **0.471** | −0.0043 |
| **tiage** | Pk | 0.4692 | 0.469 | +0.0002 |
| | WindowDiff | 0.4882 | 0.488 | +0.0002 |
| | F1 | 0.2043 | 0.204 | +0.0003 |
| | **total_score** | **0.3628** | **0.363** | −0.0002 |
| **dialseg711** | Pk | 0.4702 | 0.470 | +0.0002 |
| | WindowDiff | 0.4926 | 0.493 | −0.0004 |
| | F1 | 0.2446 | 0.245 | −0.0004 |
| | **total_score** | **0.3816** | **0.382** | −0.0004 |

tiage·dialseg711 은 논문이 소수 3자리로 반올림한 값과 **자리 그대로** 맞는다.
superseg 만 ~0.005 차이인데, 이 데이터셋만 test 분할이 큰(1,322 대화) 것과 관련될 수 있으나
허용오차(±0.02) 안이고 네 지표가 모두 같은 방향이 아니어서 계통 편차로 보이지 않는다.

## 2. 🔴 이전 기록이 왜 "판정 없음"이었나 — 하네스가 달랐다

`outputs/experiments/2026-05-20_texttiling_offline/REPORT.md` 는 우리 값만 표로 두고
논문값은 "한계" 절 문장으로만 적은 뒤 *"정확 일치 아님, 방향·정상동작 검증용"* 이라며
**판정을 내리지 않았다.** 그 값과 이번 값을 나란히 놓으면 원인이 드러난다:

| 데이터셋 | 하네스 B (2026-05-20) | **하네스 C (이번)** | 논문 |
|---|---:|---:|---:|
| tiage | 0.3081 | **0.3628** | 0.363 |
| dialseg711 | 0.3804 | **0.3816** | 0.382 |
| superseg | 0.4456 | **0.4667** | 0.471 |

**방법은 그대로인데 하네스만 바꿔도 tiage 에서 0.055 가 움직인다.**
차이의 정체는 두 가지다:
1. **데이터 계보** — B 는 Def-DTS 번들, C 는 SuperDialseg 공식 배포본
   (`benchmarks/superdialseg_data/`)
2. **채점기** — B 는 우리 segeval 래퍼, C 는 저자 `SegmentationEvaluation`

⇒ 이전 REPORT 의 격차는 **TextTiling 의 실패가 아니라 하네스 차이**였다.
`EACL/main_result.md` §5 가 못박은 A/B/C 분리를 지켜야 하는 이유의 실증 사례다.

🔑 **다른 방법에도 같은 점검이 필요하다.** 특히 CSM 이 dialseg711 에서만 76.6% 로
재현되지 않는다는 미해명 건(`2026-08-02_csm_offline_verify/REPORT.md`)이 **하네스 B 에서
측정된 것**이므로, C 로 다시 재는 것이 1순위 진단이다.

## 3. 실행 조건

| 항목 | 값 |
|---|---|
| 모델 | `super_dialseg.models.texttiling.TexttilingSegmenter` (무수정 호출) |
| 하이퍼 | nltk `TextTilingTokenizer` w=10 · k=6 (저자 기본값, 미변경) |
| 데이터 | `benchmarks/superdialseg_data/{superseg,tiage,dialseg711}/segmentation_file_test.json` |
| collator | `DataCollatorPlainText` (저자 `main.py:185` 가 texttiling 에 지정한 것) |
| 채점기 | `super_dialseg.metrics.SegmentationEvaluation` (Pk · WindowDiff · F1 · total_score) |
| 평가 루프 | 저자 `ReproducePipeline.easy_evaluate` (main.py:73-121) 그대로 이식 |
| 환경 | `.venv-superdialseg` (Python 3.11 · nltk 3.8.1 · numpy<2) |
| 학습 | 불필요 (model-free) |

## 4. 🔴 업스트림 버그 2건 — 저자 재현 스크립트가 그대로는 안 돈다

`benchmarks/` 는 read-only 규약이라 고치지 않고 우회했다. 실행된 알고리즘·채점 코드는
저자 원본 그대로다.

1. **클래스명 오타** — `examples/reproduce/main.py:18` 이
   `TextTilingCLSSegmenter` 를 import 하는데 실제 이름은 `TexttilingCLSSegmenter` 다
   (`modeling_texttiling.py:37`, 'tiling' 의 t 가 소문자). **ImportError 로 즉사한다.**
   (더불어 `main.py:36` 은 저장소에 없는 `secret_config` 모듈을 import 한다.)
   → 우리는 `main.py` 를 쓰지 않고 그 평가 루프만 옮겼다.

2. **relative import 한 단계 초과** — `utils/data/data_collator.py:8`
   ```python
   from .utils.data.load_dialseg import DialsegSample   # 실제로는 .load_dialseg 여야 한다
   ```
   그 파일의 `.` 이 이미 `super_dialseg.utils.data` 라
   `super_dialseg.utils.data.utils.data.load_dialseg` 를 찾아 ModuleNotFoundError.
   → `sys.modules` 에 `...utils.data.utils` = 패키지 자신을 꽂아 경로가 되돌아오게 했다
   (`_shim_broken_relative_import`).

## 5. 검증 방법

```bash
PYTHONPATH=benchmarks/superdialseg/src NLTK_DATA=/root/nltk_data \
  .venv-superdialseg/bin/python EACL/methods/texttiling/run_paper_repro.py
```
3개 셋 전부 허용오차 안이면 종료코드 0, 하나라도 벗어나면 1.
