# S5 v2→v3 재설계 + reasoning-off 트래커 (①B 이후 ②~⑥)

> **문제 1개 = 이 파일 1개.** ①B QC(ledger D-B11 종료, human 표본 PASS)에서 나온 설계 발견 **A~H** + 결정성
> (reasoning-off / k=3)을 반영해 S5 추출기를 **다음 VERSION으로 개선(또는 v2 유지)** 하는 전 과정 추적.
> 시도·조건·결과를 논문용으로 누적. 시작 2026-07-23.
> 상위 계획: `/root/.claude/plans/whimsical-tinkering-seal.md`. 원본 기록은 각 canonical 파일(아래 산출물 위치) 참조.

## 문제 정의
- **배경:** `p0s5-eism-v2`(deepseek-v4-pro, reasoning ON) 동결(D-B9)·①B 종료(D-B11). human 표본 PASS(전축
  ✗≤20%, 모델 중립 73~95% 정당·회피 0)이나 **설계 발견 A~H** + **결정성 노이즈**(temp0에서도 ~14% 경계흔들림).
- **목표:** (a) 결정성 노이즈 완화(reasoning-off 가능성 or Phase-1 k=3 다중샘플), (b) 라벨 타당성 개선
  (A·B·E·H 프롬프트 + F·G 전처리) — **진단으로 확증한 것만** 반영, dev-3 검증 후 **새 VERSION 등록 or v2 유지** 결정.
- **성공 기준:** ④ 새 프롬프트가 dev-3 **Gate C ΔScore ≥ −0.03** 유지 + **타깃 진단서 실제 개선**(context-leak율↓,
  intentionality none-rate 정상화, time 비대칭 해소). 결정성 대책(reasoning-off vs k=3) 확정.

## 워크플로 + 현재 상태
| # | 단계 | 상태 |
|---|---|---|
| ① | ①B QC 종료 | ✅ 완료 (D-B11 종료, human PASS) |
| ② | 진단(diagnostics) — 어떤 fix가 진짜 필요한지, 캐시·예산~0 | ✅ 완료 (`2026-07-23_findings_diagnostics/REPORT.md`) |
| ③ | reasoning-off pre-check — 결정성만(pieman×space×2) | ✅ 완료 → **실패(k=3 유지)** (`outputs/runs/2026-07-23_noreason/RESULT.md`) |
| ④ | 재설계 실험 — 확정 fix 적용 새 프롬프트 → dev-3 누적 ablation(Gate C+진단) | ✅ 완료 → **v3.1 = H+E 동결**(ledger D-B14). 3/3 통과: H 타깃 7.1%→5.0% · E 타깃 41.2%→22.5% · 안전장치 ΔScore +0.0146 ≥ −0.09. 제외 = A(빈도 미충족)·B①(②진단 기각)·B②(타깃 악화) |
| ⑤ | ④ 통과 → 새 VERSION 등록 → M4~M7 재생성 (or v2 유지) | ✅ 완료 → **`p0s5-eism-v3.1` 등록 + Phase-1 앵커 전량 생성**(ledger D-B15): 10,598편·2,777,561전이, Qwen3.6-35B-A3B no-think, VESSL A100×20, $95 |
| ⑥ | Part 2 — manifest download_count 공변량 | ⬜ **미착수** |

> **이 handoff 는 ④⑤ 로 종결됐다.** 후속(M4~M7 학습)은 별도 트랙 —
> `outputs/experiments/m4m5_phase1_anchor/` + ledger **D-B16**(1차 탐색 패스 지위 사전 선언).
> 남은 것은 ⑥ 하나뿐이다.
>
> ⚠️ **③의 "k=3 유지" 는 실행되지 않았다.** Phase-1 앵커는 **k=1 단일 draw** 로 생성됐다(D-B15).
> 그 변동 폭은 D-B16 에서 잠정 실측했고(S_self 0.659 = 사전선언 기준 **FAIL**), 확정 k 는
> 확정 패스에서 다시 정한다.

> 분리 이유: findings **맹목 fold 금지**(실제 개선+Gate C 유지 측정) · reasoning-off(결정성)와 findings(라벨타당성)
> 는 다른 변수라 ablation 분리 · F·G(전처리)는 Stage-0라 Gate C 무관·독립.

## 설계 발견 A~H (원본 `outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/FINDINGS.md`)
| finding | 상태 | fix(다음 VERSION) | ②에서 확인 | ④ 포함 | 트랙 |
|---|---|---|---|---|---|
| **A** space 어휘공백(비행기 목록밖, under) | 확정(1 doc)·스케일 미확인 | Stage1 "physical setting"=행위자 점유 공간무대+vehicle/airspace 예시 | 무대 명백한데 unspecified 비율(문서·장르별) | 조건부(빈도 충분 시) | 프롬프트/Gate C |
| **H** space context-leak(문장근거0인데 위치부여, over) | 확정(채점 2/20) | Stage2 "문장 근거 없으면 반드시 unspecified, 맥락추론 금지" | 장소단서 없는 문장의 non-unspecified율 | ✅ | 프롬프트/Gate C |
| **B②** 장면목표 vs 상시/상황목표 혼합 | 확정(표본)·스케일 | Stage1 goals=이산·순차 commitment만, 상시/상황목표 reset 제외 | 목표 eventiveness(span/N·산발 re-entry) + leave-one-goal-out | ✅ | 프롬프트+reset/Gate C |
| **B①** 대사 화제/목적 예시 confound | **가설** | 예시를 대사내부 최소대조로 교체(discussed→none / 명령·계획→goal) | 화행별(명령/약속 vs 논의) 모델 P(goal≠none) | 확증 시 | 프롬프트/Gate C |
| **B③** carry 방향(goal-first) | **보류(안 고침)** | forward 유지(backward=SEM 온라인성 위반) | (민감도만) | ❌ | — |
| **E** time opens-with 비대칭 | 확정(캐시, jump 14/29 문중) | sequence/jump/simultaneity **모두 문두조건** + So/And yet 배제 + "First"등 순서표지 목록보강 | 완료(`time_marker_position_diag.py`) | ✅ | 프롬프트/Gate C |
| **F** 앞뒤 잡물 leakage(광고·boilerplate·제목) | 확정(스캔; 표본 4/100) | strip/slice 강화(PG구조표지+휴리스틱+LLM폴백) | 완료(스캔) | **Stage-0**(Gate C 무관) | 전처리 |
| **G** 장면 구분선 병합 | 확정(스캔 9/11; 표본 1) | split_sentences 구분선 강제분할+경계힌트 | 완료(스캔) | **Stage-0** | 전처리 |
| protagonist · causality | 표본發 새 결함 0 | 수정 없음(perspective 95%; causality 기존 known over-fire=별도 트랙) | — | ❌ | — |

## 산출물 위치 (각 단계)
- **②** 스크립트 `scripts/eacl_phase0/*_diag.py` → 결과 `outputs/experiments/2026-07-23_findings_diagnostics/REPORT.md` + FINDINGS status 갱신.
- **③** `llm_s5_states.py --no-reasoning` → `outputs/runs/2026-07-23_noreason/RESULT.md`.
- **④** 새 프롬프트 variant + ablation 러너 → `outputs/experiments/2026-07-23_s5_redesign/REPORT.md`.
- **⑤** `context/methodology/v<새버전>.md` + `infrastructure.md` + ledger `EACL/phase0-decisions-and-runs.md`(D-B## 등록·supersede).
- **⑥** `fetch_repcorpus.py`(download_count) + `manifest.json` 백필 + FINDINGS/ledger(무명전용 규칙 안 만듦 근거).
- **결정 ledger(전 단계)** = `EACL/phase0-decisions-and-runs.md` (append-only).

## 시도·조건·결과 로그 (누적 — ②부터)

### ② 진단 (2026-07-23 완료)
- **스크립트:** `scripts/eacl_phase0/findings_diagnostics.py` (A·H·B①·B②) + 기존 `time_marker_position_diag.py`(E).
- **데이터:** frozen `p0s5-eism-v2` run1 캐시(`outputs/eacl_phase0/repcorpus_s5_eismv2_run1`, 12편) + `repcorpus_master.jsonl`.
  gold-free, 예산 0(라벨 재호출 없음, regex 스크리닝). **판정 기준은 결과 전 사전 동결**(REPORT 각 절).
- **산출:** `outputs/experiments/2026-07-23_findings_diagnostics/REPORT.md`.
- **결과 (④ 포함 게이트):**
  | finding | 사전기준 | 결과 | ④ |
  |---|---|---|---|
  | A list-miss(under) | 국소유의+obs편중 | `obs_77176`(항공)=26 압도·국소, 타 문서 regex FP | **포함**(정합·저비용, genre국소) |
  | H context-leak(over) | no-cue non-unspec 상당 | 전체 8.8%, 최고 owlcreek 31.5%(다수 맥락상 정답, 순수성 문제 실재) | **포함 ✅** |
  | B① 대사화행 | gap≤10% & 수행<50% | **gap +16.8%**(수행 41.6% > 논의 24.8%) = 기준 위반 | **기각** → 예시보강은 선택 |
  | B② 상시목표 | span≥0.6 존재+reset영향 | 10/12 문서, leave-one-out reset 대폭↓(대량 기여) | **포함 ✅** |
  | E time 비대칭 | 별도 | jump 문중 48% vs sequence 24% | **포함 ✅** |
- **핵심:** B①만 스케일에서 **기각**(모델이 화행 무시 안 함; none-rate 주범은 state_description 19.9%). ④ intentionality
  재설계는 **B②(상시목표 제한)가 load-bearing**, ① 예시교체는 무해한 명료화로만. **repcorpus gold=0 → Score 영향은 ④ dev-3로 이월.**

### ③ reasoning-off pre-check (2026-07-23 완료 → 실패)
- **E0:** `deepseek-v4-pro` 는 모델목록에 없음(openrouter/* 라우팅). 토글 프로브에서 reasoning-ON/OFF 모두
  `reasoning_content=None`·completion_tokens 안 줄음 → **토글 관측상 no-op**(비-분리형 서빙).
- **E1-E2:** pieman×space×2씩(no-reasoning / reasoning-on 대조). per-item 일치: **no-reasoning 0.906 (flip 0.094)**
  vs reasoning-ON 0.958. 크로스조건 차이(0.042~0.104)=조건내 노이즈 수준.
- **E3 판정:** no-reasoning 0.906 < 목표 0.95 → **실패**. 비결정성=백엔드/샘플링 기인(toggle 불가) →
  **k=3 다중샘플 완화책 유지**. 전체 Gate C 재검증 미발동. 산출 `outputs/runs/2026-07-23_noreason/RESULT.md`.

### ④ 재설계 실험 (2026-07-23~24)
- **v3(전 fix 묶음) vs v2 + 7모델 cross-model 비교 완료.** 결과:
  - **H·E = Score 중립 + validity 개선**(H no-cue non-unspec 8.8%→~2%, E jump 문중 48%→~0). **B②는 강한
    라벨러(deepseek·Qwen3-Next)에서 Score 감점**(장면목표까지 잘라냄). "v3 fail" 은 deepseek-reasoning 아티팩트 —
    no-think 라벨러(GLM/Qwen)에선 v3 Score 중립. → **v3.1 = H+E 안전**(4모델 Δ∈[−0.001,+0.007]).
  - **오픈모델 자가호스팅 검증**(로컬 A100 vLLM): Qwen3-32B/GLM-4.5-Air/Qwen3-Next-80B/Qwen3.6-35B-A3B/-27B +
    deepseek-chat(프록시). v3 no-think Score 순위: deepseek-chat 0.393 ≳ **Qwen3.6-35B-A3B 0.387**(로컬 최고·79x·무료)
    ≳ Qwen3.6-27B 0.375 > deepseek-v4-pro-v3OFF 0.364 > GLM 0.352 ≈ Qwen3-32B 0.349 > Qwen3-Next 0.318.
    **문서별 지배 도메인 상이**(tunnel=space·monkey=protag/intent). 산출 `CROSSMODEL.md`·states `*_s5_{glm,qwen32b,
    qwennext,q36_35b,q36_27b,dschat}_{v2,v3}`.
  - **결정성**: vLLM temp0 = single-stream 완전결정 / 배치 시 flip 25~30%(continuous batching numerics). 인프라
    특성 → `context/methodology/infrastructure.md §4b`.

## 다음 액션 (④-fine, 2026-07-24 보정)
**개별 fix 기여 = 타깃 지표 primary**(Score 아님 — D-M5 재현 회피, σ0.06<0.09). 사전선언 채택규칙 =
(타깃지표 의도방향 개선) ∧ (Score −0.09 이내 안전장치). **4변종**(time묶음·space-A·space-H·intent-②;
intent-① 제외=진단 기각) on **Qwen3.6-35B-A3B(no-think 로컬)**. 상세 스펙 `plans/optimized-growing-crystal.md`,
결정 `context/06-decision-log.md`(2026-07-24). 산출 → `PERFIX_REPORT.md`. (F/G 전처리는 Stage-0, Phase-1 prep 별도.)
