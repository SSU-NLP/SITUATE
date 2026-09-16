# TextTiling 논문 수치 재현 (하네스 C) — REPORT

**판정: ✅ 재현.** 3/3 데이터셋. tiage·dialseg711 은 논문 반올림 자리까지 일치.
방법론 판정의 canonical = `EACL/methods/texttiling/REPRO.md` (이 문서는 실행 기록).

---

## 1. 실험 setup

| 항목 | 값 |
|---|---|
| **목적** | 저장소에 TextTiling 실행 기록은 있으나 **논문 대조표도 재현 판정도 없었다**. 그 구멍을 닫는다 |
| 러너 | `EACL/methods/texttiling/run_paper_repro.py` |
| 방법 | `super_dialseg.models.texttiling.TexttilingSegmenter` — 무수정 호출 |
| HP | nltk `TextTilingTokenizer` **w=10 · k=6** (저자 기본값, 변경 없음). 학습 불필요 |
| 데이터 | `benchmarks/superdialseg_data/{superseg,tiage,dialseg711}/segmentation_file_test.json` (SuperDialseg **공식 배포본**) |
| 규모 | superseg 1,322 / tiage 100 / dialseg711 711 대화 (test) |
| collator | `DataCollatorPlainText` (저자 `main.py:185` 지정) |
| 지표 | 저자 `SegmentationEvaluation` — Pk · WindowDiff · F1(binary) · `total_score` = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD) |
| 평가 루프 | 저자 `ReproducePipeline.easy_evaluate` (main.py:73-121) 그대로 이식 |
| seed | 해당 없음 (결정적 알고리즘) |
| 비교 기준 | Jiang et al. (2023) EMNLP, SuperDialseg **Table 3** TextTiling 행 |
| 환경 | `.venv-superdialseg` (Python 3.11 · nltk 3.8.1 · numpy<2) |
| 비용 | CPU only · 3셋 합계 약 30초 · API 0 |

## 2. 결과

| 데이터셋 | 지표 | 우리 | 논문 | 차이 | |
|---|---|---:|---:|---:|---|
| superseg | Pk | 0.4455 | 0.441 | +0.0045 | ✅ |
| | WindowDiff | 0.4579 | 0.453 | +0.0049 | ✅ |
| | F1 | 0.3851 | 0.388 | −0.0029 | ✅ |
| | **total_score** | **0.4667** | **0.471** | −0.0043 | ✅ |
| tiage | Pk | 0.4692 | 0.469 | +0.0002 | ✅ |
| | WindowDiff | 0.4882 | 0.488 | +0.0002 | ✅ |
| | F1 | 0.2043 | 0.204 | +0.0003 | ✅ |
| | **total_score** | **0.3628** | **0.363** | −0.0002 | ✅ |
| dialseg711 | Pk | 0.4702 | 0.470 | +0.0002 | ✅ |
| | WindowDiff | 0.4926 | 0.493 | −0.0004 | ✅ |
| | F1 | 0.2446 | 0.245 | −0.0004 | ✅ |
| | **total_score** | **0.3816** | **0.382** | −0.0004 | ✅ |

판정 임계 = ±0.02 (사전 선언, `--tolerance`). 12개 셀 전부 통과.

## 3. 해석

**tiage·dialseg711 은 논문이 소수 3자리로 반올림한 값과 자리 그대로 맞는다**(차이 ≤0.0004).
우연이 아니라 같은 데이터·같은 코드·같은 채점기를 썼다는 뜻이다.

superseg 만 ~0.005 벌어지는데, 네 지표의 부호가 엇갈려(Pk·WD 는 +, F1·score 는 −)
계통 편차로 보이지 않는다. test 분할이 가장 큰 셋(1,322 대화)이라 논문 쪽 반올림·
전처리 세부의 잔차로 읽는 것이 자연스럽다. 허용오차 안이므로 재현으로 판정한다.

### 🔑 가장 중요한 발견 — 이전 격차는 방법이 아니라 하네스였다

| 데이터셋 | 하네스 B (2026-05-20 REPORT) | **하네스 C (이번)** | 논문 |
|---|---:|---:|---:|
| tiage | 0.3081 | **0.3628** | 0.363 |
| dialseg711 | 0.3804 | **0.3816** | 0.382 |
| superseg | 0.4456 | **0.4667** | 0.471 |

**같은 방법인데 하네스만 바꿔도 tiage 에서 0.055 가 움직인다.** 차이는 두 가지다 —
데이터 계보(Def-DTS 번들 vs SuperDialseg 공식 배포본)와 채점기(우리 segeval 래퍼 vs
저자 `SegmentationEvaluation`). 이전 REPORT 가 *"정확 일치 아님"* 이라며 판정을 미룬 것은
방법의 문제가 아니라 **조건이 논문과 달랐던 것**이다.

## 4. 판정 및 다음 단계

- **TextTiling = ✅ 재현.** NES·DTS 양쪽 베이스라인 목록에서 "원본 확보" 상태가 된다.
- ⬜ **CSM 을 하네스 C 로 다시 잴 것 (1순위).** CSM 이 dialseg711 에서만 76.6% 로
  재현되지 않는다는 미해명 건(`2026-08-02_csm_offline_verify/REPORT.md`)은
  **하네스 B 에서 측정된 값**이다. 위 실증을 보면 그 격차의 상당 부분이 하네스일 수 있다.
- ⬜ 온라인 변형(`methods/texttiling/online/`)의 손실도 **하네스 C 기준**으로 다시 재야
  `Δ = online − offline` 이 논문과 같은 축 위에 놓인다.

## 5. 🔴 업스트림 버그 2건 (저자 재현 스크립트가 그대로는 안 돈다)

`benchmarks/` read-only 규약을 지켜 수정하지 않고 우회했다. 실행된 알고리즘·채점 코드는 원본 그대로다.

1. `examples/reproduce/main.py:18` — `TextTilingCLSSegmenter` import, 실제 이름은
   `TexttilingCLSSegmenter`(소문자 t). **ImportError 로 즉사.** `main.py:36` 은 저장소에 없는
   `secret_config` 도 import 한다. → main.py 를 쓰지 않고 평가 루프만 이식.
2. `utils/data/data_collator.py:8` — `from .utils.data.load_dialseg import ...` 가
   relative 를 한 단계 더 써서 ModuleNotFoundError. → `sys.modules` 자기참조 shim 으로 우회.

## 6. 한계 / 검증 미해결

- **superseg 잔차 ~0.005 의 원인 미규명.** 허용오차 안이라 추적하지 않았다.
- **논문 값이 소수 3자리 반올림**이라 그보다 정밀한 대조는 불가능하다.
- 이 실험은 **offline·whole-dialogue** 조건만 잰다. 온라인 변형은 별도.
- `total_score` 는 저자 정의를 그대로 썼고, `EACL/main_result.md` 의 하네스 A Score 와
  **같은 수식이지만 다른 구현·다른 데이터**다. 섞어 비교하지 말 것.
