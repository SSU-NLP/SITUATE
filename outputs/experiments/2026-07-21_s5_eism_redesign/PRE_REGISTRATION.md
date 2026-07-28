# 관문① S5 EISM 재설계 — 사전등록 (PRE-REGISTRATION)

> **작성일 2026-07-21. 이 문서는 어떤 dev-3 재추출·Gate C 수치도 보기 전에 확정한다.**
> 가드레일(계획 v3): 모든 탐색(프롬프트/축채택/onset)은 숫자를 보기 전에 공간·상한·채택규칙·동률처리를 문서화한다.
> 확정 후 수정 시 수정 사실·이유·일시를 이 문서 하단 "수정 로그"에 append(삭제 금지).
> 계획 원본: `/root/.claude/plans/shimmering-swimming-quilt.md`. 스펙: `EACL/s5-eism-redesign-guide.md`.

## 0. 목적·범위

- **목적:** S5-Extractor를 EISM 정합으로 재설계하고 **deepseek-v4-pro 단일 라벨러**로 dev-3(pieman/tunnel/monkey)에서 Gate C를 통과시켜 M4~M7 anchor 공급원을 **동결**한다.
- **라벨러(확정):** `openrouter/deepseek/deepseek-v4-pro` (dev-3 + 이후 코퍼스 스케일 공통). fable5 예산밖.
- **기술조건(확정):** `--stage1-max-tokens` 8000 이상(deepseek reasoning 토큰이 monkey 27,545자에서 2000이면 `content is None`).
- **평가:** DTS Score `0.5·F1+0.25(1−Pk)+0.25(1−WD)`, deployable(count≥t\*=3), mpnet frozen, `hi_ontop.dts_scoring.score_dialogues`. Score-first.

## 1. 재설계 코드 변경 (①-b) — 확정 스펙

`scripts/eacl_phase0/llm_s5_states.py`:
- **하드 리네임(축 키):** `place→space`, `goal→intentionality`. 필드명 `location`·`active_goal`은 **유지**(축 내부 필드).
- **Stage1:** `protagonist` 단일 id 라인 + 스키마 키 추가.
- **Stage2 프롬프트:** space·intentionality **null-명시화**(guide §3.3 3줄 / §3.5) + 공통 꼬리말(§4.2). "발화 안 하면 무조건 neutral" 문자지시 **삭제**.
- **실험 필드(additive 로깅, 동결 count 미포함, 별도 reset 함수):** `protagonist_present`(Stage1 주인공 1인 대비 boolean), `causal_marker`∈{absent,cause,consequence}.
- **VERSION:** `p0s5-story-list-v1` → `p0s5-eism-v2`. **구 캐시 보존(삭제 금지 — 사용자 지정 2026-07-21)**: fable5(`*_s5`)·dsv4pro(`*_s5_dsv4pro`) dev-3 캐시는 Gate C 기준선·재현·비교용으로 유지. 신규 deepseek 추출은 **별도 버전 out_dir `*_s5_eismv2`**로 보내 충돌·오염 방지. (rename으로 `space_states`/`intentionality_states`는 파일명이 달라 애초에 충돌 안 하나, `story_lists`/protagonist/time/causality는 재생성 필요 → 별도 dir로 격리.) 가이드 §2.2 "구 캐시 삭제"는 이 보존 방식으로 **대체**.
- **리셋 함수(코드, 불변 원칙):** `reset_space`(구 reset_place 로직), `reset_intentionality`(구 reset_goal 로직), reset_time=marker-only 유지, reset_causality 유지, reset_perspective 유지. 실험필드는 `reset_protagonist_present`·`reset_causal_marker` **별도 분리**.

### 1.1 리네임 완결성 체크리스트 (freeze 전 §8-6, scope = frozen S5 anchor 파이프라인)

**리네임 완료(✅, frozen 기계 파이프라인 — 반드시 신 키):**
- **[프로듀서]** `llm_s5_states.py` (AXES·AXIS_FIELDS·stage2_*·reset_*·main carry-forward·reset_matrix 컬럼).
- **[anchor 소비]** `build_slot_dataset.py`(AXES L23·상태언팩·SHARE 인덱스), `src/hi_ontop/factored_slot.py`(AXES·주석) → `probe_slot_alignment.py`는 `factored_slot.AXES` import라 자동 전파.
- **[평가·진단]** `eval_seg_score.py`(FROZEN_AXES), `deployable_3way.py`, `b1_deployable.py`, `axis_ablation.py`(AX+SHORT), `dev_b2b3_topline.py`, `adaptive_threshold_probe.py`, `axis_liveness.py`, `corpus_ablation_report.py`, `labeler_agreement.py`(AXES+필드분기+space 리터럴). 모두 문법검증 통과.
- **[남은 평가 소비자 — freeze 전 확인]** `tier2_eval.py`(M4/M5 평가 하네스 확장 시 신 키; 관문④에서 처리).

**명시적 EXEMPT (인간-주석 수집 서브시스템 — 자기완결 legacy, 미리네임):**
- `annotate_server.py`(SLOTS+프롬프트), `split_blinded_packs.py`/`merge_blinded_packs.py`, `build_{dts,pieman,ami,michelmann,narrative,tier2,annotation,share}_pack.py`의 `"place":None/"goal":None` **인간주석 placeholder 컬럼**, S4-era `llm_label_axes.py`/`llm_absolute_state_pipeline.py`/`llm_enum_states.py`/`carry_forward_slow.py`(import 안 됨, standalone).
- **근거:** 이들은 5축 어휘를 *인간 gold 수집*에 쓰는 matched 유닛(annotate_server↔pack 컬럼↔blinded split). frozen S5 기계 추출 파이프라인이 **읽지 않음**(master의 place/goal 컬럼은 vestigial, 파이프라인은 previous/current/boundary/transition/idm_score만 읽음). 부분 rename 시 짝이 깨짐. dev-3 human gold는 이미 수집 완료, Born-a-Crime gold는 EventRecall 20인 데이터 → 이 서브시스템 재사용 안 함. **재사용해 새 gold를 수집할 때만 일괄 rename.**

**검증(§1.4):** 구조적 grep/jq/find는 **anchor 파이프라인 파일에 한정**해 축-키 잔존 0 확인. EXEMPT 파일은 제외(단순 `grep place|goal` 아님 — 그건 location/active_goal/goals/일반단어에도 걸림).

## 2. goal divergence 진단 (①-a) — 사전등록된 측정

**목적:** Q5(intentionality 재정의 a/b/c) 결정 근거. **진단은 진단일 뿐, 재정의 동결은 dev-3 Gate C에서.**
측정(dev-3, deepseek 신 프롬프트 기준):
1. **식별가능성:** `active_goal`이 `none`이 아닌 문장 비율(place의 86% unspecified 유사 死변수 여부).
2. **라벨 발산:** 동일 문장에서 deepseek `active_goal` 라벨의 재현성(온도0, 재실행 일치율) + 사람 gold 대비 일치.
3. **취약성:** intentionality 축 제거 시 dev Score 낙폭(참조: −goal pieman 0.598→0.546, dialseg 0.634→0.477 = 신호 실재하나 취약).

**Q5 결정규칙(codex 위임, 진단 후):** (a) 현행 유지+null 명시화만 / (b) "진전시키는 goal"로 좁힘 / (c) Indexter frame 근사. **b/c는 진단이 지지할 때만.** 기본값 = (a)(가장 보수적, 신호 안 죽임).

**✅ Q5 확정 = (a)** (codex 2026-07-21, `scratchpad/codex_q5_out.md`). 근거: 진단은 "식별불가 가설"만 배제했고 b/c 우위 증거 없음. b는 "진전 여부"라는 새 해석차 유발·수단/준비 행위를 none으로 만듦; c는 상태공간을 "누구의 목표"로 파편화(target-only에서 소유자 판별 불안정). (a)=신호·carry 계약 보존하며 literal→continuity 직접 교정하는 **최소 변경**. 근거는 pieman +0.149가 **아니라** tunnel/monkey 일반화 취약성 + 3편 공통 literal 발산.
- **동결 문구**(llm_s5_states.py 반영 완료): `"the Stage 1 goal that best explains what the action depicted by THIS sentence is FOR. Infer its purpose from the meaning of THIS sentence; the goal's words need not appear. A goal that is only MENTIONED ... is NOT chosen. Use none only when THIS sentence gives no basis at all ..."` — 이전 `"pursues or advances"`(=b 판정) **삭제**.
- **reset_intentionality·carry_forward·Stage1 goals = 불변.** Q5=(a)는 base 추출에 포함 → **재추출 회차 추가 없음**(회차 규칙 §3.3): Round 1 base가 곧 최종 intentionality 정의.
- **잔여(codex):** intentionality 수정만으로 Gate C 통과 예단 금지. tunnel은 causality 0.455 무대응이라 실패 가능 — 그건 Q5 반증이 아니라 **별도 core causality 사안**(2회차 후보).

### 2.1 ✅ ①-a 사실상 완료 (예산 0, 기존 fable5+dsv4pro 산출; 2026-07-21)
세 요구가 모두 예산 0로 확보됨(`goal_divergence_REPORT_fable5.md` + `dsv4pro_axis_attribution.md`):
1. **식별가능성** ✅ none-rate fable5 0.594/0.506/0.730(死변수 아님 — space보다 자주 식별).
2. **라벨러 발산** ✅ = dsv4pro↔fable5 축별 리셋 A↔B F1 **intentionality 0.437/0.340/0.381 = 3편 전부 최악**(발산 최대 축).
3. **취약성** ✅ ΔScore +0.149/−0.037/−0.008(pieman-편중, 문서 의존; §해석 caveat 2·3).
**⇒ Q5(a/b/c) 결정을 지금 codex 위임 가능 — deepseek 재추출 대기 불필요.** Q5를 스모크(§3.0 pieman intentionality) 前에 확정해야 스모크가 최종 정의를 검증한다.
**caveat(REPORT 해석):** 축별 F1은 ΔScore를 예측 못 함(F1 최고 pieman이 Score 최악); Q5 판단에 pieman 개선을 근거로 쓰지 말 것(기준=tunnel/monkey ≥0); none-rate는 건강지표 아님(carry-forward가 메움).

## 3. Gate C 재검증 (①-c) — 사전등록된 프로토콜

### 3.0 SMOKE (Gate C 前 기전 feasibility — 5회 상한 미포함, Score-selection 아님)

**베팅:** null-명시화(무대/언급 구분)가 deepseek의 두 실패모드를 고친다. **전체 3×5 재추출 前 1~2편 space 축만으로 기전이 작동하는지 확인**(사용자 2026-07-21). 성능을 보고 고르는 게 아니라 기전 작동을 보는 feasibility → **사전등록 위반 아님**이며 **5회 상한에 미포함**(Score 판정이 아니므로).

- **🔴 스모크 doc는 문서마다 다른 축**(축별 귀속 `dsv4pro_axis_attribution.md`; codex "pieman space"는 오답 — pieman space=0.889 최청정이라 무의미). **붕괴한 축을 그 붕괴한 doc에서** 재추출(`*_s5_eismv2`):
  | doc | 재추출 축 | 검증 | before(dsv4pro) → 목표(fable5) |
  |---|---|---|---|
  | monkey | **space** | over-silence 수리(§3.3) | unspecified 0.963 → 0.824 |
  | pieman | **intentionality** | null-명시화 수리(§3.5, Q5 최종정의) | none-rate 0.760 → 0.594 |
  - 각 1문서×1축 = **3×5의 2/15**. deepseek 기본, max-tokens 8000.
- **합격 지표(정정 2026-07-21 — 리셋 수 무용·count≥3 비교대상 오류 수정):**
  | 지표 | 역할 | 판정 |
  |---|---|---|
  | **A↔B 리셋 F1** (deepseek-신 ↔ fable5) | **주 판정** | pieman intent **0.437** / monkey space **0.467** 대비 **상승**. "맞는 자리에 리셋 냈나" |
  | none/unspecified rate | 보조(방향) | dsv4pro→fable5 방향 이동. 단독 판정 금지(엉뚱한 값 집어도 rate는 내려감) |
  | 리셋 수 | 보조(폭주만) | **≥40이면 과잉발화 의심**([9,26] 밴드 폐기 — dsv4pro 15가 통과해버려 실패를 못 거름) |
  | count≥3 (신 축 + **dsv4pro 4축** 혼합) | 구조 | **dsv4pro 0.105 대비 상향**(혼합이라 fable5 0.158은 상한이지 목표 아님; gold 0.189) |
  - **🔴 주 지표 = A↔B 리셋 F1.** 리셋 수는 dsv4pro 15 ≈ fable5 17인데 F1 0.437(엉뚱한 자리) → **수로는 실패 못 거름**. rate도 대리지표. **맞는 자리(F1)가 올라야 실제 수리.**
- **불합격 처리:** 하락·구조회복 없으면 베팅 실패 → **프롬프트 문구 수정 재추출(회차 소모) 또는 R1**. 3×5 태우기 전 싸게 결정.
- **pieman 기준선 caveat:** pieman fable5 0.634 = 최고+t\* 선택문서라 가장 오염·구조적 최난 → Gate C 실패가 pieman 집중 시 라벨러 능력 아닌 기준선 문제 가능(§2.1 caveat).

### 3.0.1 스모크 v2 결과 + 문구 재수정 + Round 절차 (2026-07-21)

**스모크 v2 판정 = 미결 → intentionality 문구 재수정 후 Round 1로.** 상세 `smoke_REPORT.md`.
- **결정적 miss (사용자):** `count≥3` = 0.105 = dsv4pro 그대로. **Gate C 변수(경계=count≥3)가 안 움직임.** F1 +0.081은 경계를 안 만든다. pieman −0.253 원인이 이 붕괴(0.158→0.105)였는데 F1에 눈이 팔려 놓침 → "F1≠Score" 역방향 실증.
- **근본원인:** 제(Claude) 과잉발화 경고가 프롬프트를 **침묵으로 과교정** → none 0.760→0.792(dsv4pro보다↑), 리셋 10<fable5 17(recall 0.41, 정밀 0.70 = 찍은 자리는 정확하나 개수 부족). count≥3에서 한 축 덜 터지면 합이 3 미달 → 과소예측.
- **문구 재수정(적용 완료):** 억제문("Not every action serves a goal … do NOT attach a goal to every action") **제거**, 예시 유지. codex 원안과 이전 수정안의 **중간** — 목표 = 리셋 수 10→fable5 17 근처 회복.
- **space는 병목 아님 → (B) 기각:** monkey ΔScore −0.052 = 3편 중 최선인데 space F1 최악(0.467). space 더 고쳐도 Gate C 기여 미미. "deepseek가 fable5보다 낫다"는 정성 읽기는 **반증 불가·자기합리화 위험**이라 채택 안 함.

**🔒 Round 절차 (사전등록, 사용자 2026-07-21):** **각 Round = pieman 5축 먼저 재추출 → Gate C.**
- pieman = 가장 어렵고(기준선 0.634 최고 + t\* 선택 문서) 가장 쌈(96문장 = tunnel504+monkey407의 1/10). 실 Score를 1/10 비용에.
- **pieman이 Gate C(ΔScore≥−0.03) 실패 시 그 Round에서 tunnel/monkey로 확장하지 않음**(900문장 미소모, 다음 수 재고). 통과 시에만 tunnel/monkey 확장.
- **이는 적응적 중단이 아니라 계획된 절차**(사전등록). Score를 보므로 **Round로 카운트**. → **Round 1 = pieman 5축 full.**
- **예측(사용자):** pieman에서 실제 바뀐 축은 intentionality 하나(space 이미 0.889 청정, time/causality/protagonist core 불변) → 문구 재수정이 리셋 수를 실제 회복 못 하면 Round 1도 −0.2대 각오.

### 3.0.2 Round 1 잔여(−0.056) 원인규명 진단 (예산0, 회차 미포함, 2026-07-21)

**목적 = 원인규명(cause-finding), 재튜닝 아님. t\*=3 동결 불변(M4/M5 상속). pieman=dev(무제한).** Score를 보나 판정이 아니라 진범 축 특정용 → **회차 미포함.**

- **기각된 오진(사용자):** "느린 축(space 0.167·intent 0.278) 확산"은 **틀림** — space A↔B F1 0.889(deepseek≈fable5)면 fable5도 gold 경계에서 space 거의 안 터짐 = 5축 설계 성질이지 deepseek 결함/−0.056 원인 아님. fable5도 그 상태로 0.634.
- **진단 1:** 축별 gold-경계 발화율을 **deepseek vs fable5 나란히**. 차이 큰 칸 = 진범. 예측: space 유사, 범인은 causality(A↔B 0.730)·time(0.778) 등 중간 축의 조용한 손실.
- **진단 2 (baseline caveat 기전 직접 검사):** t\*=3은 fable5 pieman에서 argmax → 임계값이 fable5 리셋분포 특화 → 타 라벨러 구조적 불리 = caveat의 실제 기전. 두 라벨러의 **Score(t\*=1..5) 곡선** 비교. deepseek 최적도 t\*=3 → 홈어드밴티지 없음, −0.056=실력차 → **caveat 기각·(A)**. deepseek 최적이 2/4에서 확실히↑ → 임계값 fable5 특화 → **caveat 확인·(B) 근거**.
- **🔴 진단 후에도 t\* 변경 금지**(동결). 진단은 원인용.

**사전등록 충돌(정직 기록):** "pieman 실패→확장 안 함"(규칙1) vs "pieman 집중실패→기준선 문제 가능"(caveat)는 자기모순(집중 여부는 타 문서를 봐야 아는데 규칙1이 금지). **진단 2가 타 문서 없이 caveat 기전을 직접 검사해 교착 해소** → (A)/(B)를 규칙해석 아닌 **증거로** 결정. 충돌·해소를 ledger에 기록(리뷰어 대비).

### 3.0.3 tunnel/monkey 확장 결정 (pieman-first 규칙 초과 — 충돌 해소 기록, 2026-07-21)

**결정(사용자): pieman Gate C 실패에도 tunnel/monkey Round 1 실행.** pieman-first 규칙("실패 시 확장 안 함")을 넘으나, 근거는 원인규명 진단(§3.0.2)이 확보됨:
- 진단2가 t\* 홈어드밴티지를 기각 → 남은 caveat 기전 = **절대 기준선 높이**(pieman 0.634 최고·최오염). 이건 **tunnel(0.294)/monkey(0.359)를 열어야만** −0.041이 pieman 특유인지 답이 나옴. 진단만으로는 불가.
- 동시 목적: **protagonist_present 정식 채택 판정**(+0.015 on pieman, 바 +0.02 미달 → dev-3 완성 필요).
- **충돌 해소 논리:** 실패를 보고 예외를 찾은 게 아니라, 사전등록된 baseline caveat의 유일한 검증 경로가 타 문서라 그 경로를 밟음. ledger에 D-B9 시 이 충돌·해소를 명기(리뷰어 대비).
- 현재 pieman 최선 = 0.593(protagonist_present)/0.578(현행), fable5 0.634 대비 −0.041/−0.056. Q4 causal_marker 死(OR·flip 손해, 극성 정상). 실험 레버 소진.

### 3.1 순차 비교 순서 (효과 분리 — 한번에 다 바꾸지 않음)
1. **S1:** 새 기본 5축 프롬프트(rename + null-명시화, 실험필드 로깅만) → deepseek Gate C.
2. **S2:** intentionality 후보(Q5 a/b/c) 중 dev-3 최고 선택.
3. **S3:** `protagonist_present`·`causal_marker` **축별 독립** 채택 판정.

### 3.2 판정 기준 (숫자 확정)
- **Gate C 통과 = 사람 gold 대비 ΔScore ≥ −0.03, dev-3 3편 모두.** 5축 최종 Score로만(place 단독 통계 판정 금지).
- **🔴 기준선 오염 caveat:** Δ = deepseek − fable5, 그런데 fable5 값(pieman 0.598/tunnel/monkey; dialseg 0.634 등)은 **D-M5 selection-on-test로 dev 강등된 값**. ⇒ Gate C 통과 = "오염 기준선 **재현**"(replication)이지 **성능 근거 아님**. 논문에 성능으로 인용 금지. 기준선이 우연히 높으면 Gate C가 부당히 어려워 **R1 확률 상향**.
- **실험필드 축별 채택기준(확정):** 해당 축 안에서 (i) dev-3 평균 Score가 baseline(perspective 단독 / 기존 causality) 대비 **+0.02 이상** AND (ii) **3편 중 2편 이상에서 하락하지 않음**. 둘 다면 채택·동결 count 편입, 아니면 폐기·기록. 축지표(리셋 F1)가 아니라 **최종 Score 우선**.
- **2×2 조합 전수탐색 금지.** protagonist_present는 protagonist 축 안에서만, causal_marker는 causality 축 안에서만 겨룸.

### 3.3 프롬프트 탐색 예산 (오염 상한 — 확정)
- **회차 정의(사용자 2026-07-21 — 지금 못박음):** 1회차 = **dev-3에서 재추출+Score판정되는 동결 후보 프롬프트 1세트.** 진단·분석은 회차 아님.
  - **Round 1** = base eismv2 추출(5축 전부 + 실험필드는 additive라 같은 추출에 포함) on dev-3. **divergence 진단은 이 추출을 *읽는* 분석 → 회차 아님.** Gate C S1 Score, 실험필드(S3) 채택 판정도 이 추출 위 분석 → **회차 아님.**
  - **Q5 재정의가 intentionality *프롬프트를 바꿔* 재추출을 요구하면(=후보 b/c) = Round 2.** Q5=(a)(현행+null-명시화, base에 이미 포함)면 재추출 불요 → 추가 회차 없음.
  - **스모크(§3.0)는 회차 미포함.** 스모크 불합격으로 프롬프트 문구를 고쳐 **재추출**하면 그 재추출이 Round 1을 대체(회차 카운트는 재추출 세트 수).
- **최대 수정 회차 = 5회**(재추출 세트 기준). 각 회차 = null-명시화 문구·enum 경계 문구 조정만(축 정의·리셋 규칙 불변). 
- **회차별 변경 가능 = 프롬프트 문면 문구뿐.** 리셋 코드·t\*·enum 집합·축 키는 불변.
- **중단조건:** 5회 내 3편 모두 통과 → 즉시 동결. 5회 소진해도 미통과 → **R1 발동(아래)**.
- **최종 선택규칙:** 3편 모두 통과한 첫 variant. 복수면 **dev-3 평균 Score 최고**, 동률이면 **최소 회차** variant.
- **누적 시도수·전 회차 diff를 REPORT + ledger에 기록**(은닉 selection 금지).

### 3.4 R1 (blocking) — 실패 처리 확정
- **트리거:** 5회 소진 후에도 dev-3 3편 Gate C 미통과.
- **행동:** deepseek 진짜 기각 판정 → **즉시 사용자 에스컬레이션**(fable5 예산밖이라 체인 정지). Claude 자발적 다른 모델 도입 금지.
- **축소안(사용자 승인 시):** Gate C 통과한 축만 쓰는 **3축 M5′**를 별도 방법·축소 RQ로 **재등록**(조용한 fallback 금지).

## 4. 2단계 동결 (①-d) — 순환 방지 확정

- **Gate ①A = 후보 동결:** §3 통과한 스키마·프롬프트 확정.
- **Gate ①B = 비시험 파일럿 QC:** 대표 비시험 코퍼스 소량에서 파싱성공률·schema 유효율·retry율·비용/문장·출력 결정성·anchor 양성률·축 liveness + human QC 층화표본.
- **production freeze:** ①B 통과 후에만 `prompt_sha256` 확정.
- **freeze 후 Phase-1 GATE 실패 = 프롬프트 수정 금지, 실행 폐기 + 새 VERSION 재등록.**

## 5. 기록 (①-e)

- 산출: 이 디렉토리 `REPORT.md`(setup/결과/해석/판정/한계) + 회차별 프롬프트 diff + Score 표.
- ledger: **D-B9**(가칭) — 리네임 + VERSION + 모델교체 + 실험필드 승패 + Q5 결정을 한 D코드로 append.
- methodology: `context/methodology/`에 신 버전(p0s5-eism-v2) 파일.
- 시험지(Born a Crime) anchor 생성은 하되 **출력·성능 미열람**.

---

## 6. §3.1′ 통합 protagonist 축 후보 — dominant-actor turnover (사전등록, 2026-07-22)

**동기.** 다화자(AMI/LoCoMo) 전이 검토에서 발견: fiction 최적 `protagonist_present`(단일 주인공 boolean)의
관련 축(perspective)이 AMI 회의에서 gold발화율 0.000 = **완전 死**, 반면 charset(인물집합) 변화는 AMI에서 신호(+0.116).
⇒ 단일주인공↔다화자를 잇는 통합축 = **"지금 단위를 끌고 가는 *지배 행위자*의 교체(turnover)"** 가설.
perspective/present는 집합크기 1 특수화, 다화자는 floor집합 turnover — 하나의 정의로 축퇴.

**기존 charset(존재집합 변화)의 zero-budget 결과**(fiction dev-3, t\* pieman동결→적용):
perspective(현동결) 3편평균 0.377 < **charset 0.423** < **present 0.447**. charset은 이미 현동결 초과, present엔 −0.024.
charset은 "*존재* 집합"이라 과발화 잡음 → **목적설계 "*지배 행위자*(주절 행위주체) 단일 turnover"** 가 present를 따라잡을지 검증.

**변경(코드):** protagonist Stage2에 `dominant_actor`(단일 id, 이 문장 주된 행위를 이끄는 행위자; cast 1개 또는 none) 필드 추가.
`reset_dominant_turnover(a,b) = int(a.dominant_actor != b.dominant_actor and none 미포함)`. reset_matrix에 additive 컬럼 `dominant_turnover`.
**다른 4축(space/time/causality/intentionality)·리셋규칙·t\* 프로토콜·human gold 전부 불변**, protagonist 축만 재추출(deepseek, dev-3).

**측정.** count≥3 슬롯의 protagonist 자리에 `dominant_turnover` 대입 → BASE4와 합산, t\*=argmax Score(pieman)→tunnel/monkey 적용.
비교대상 = present(0.447)·charset(0.423)·perspective(0.377), 동일 하네스.

**🔒 채택바(숫자 보기 전 고정):**
- **주판정:** dominant_turnover **fiction 3편평균 Score ≥ present − 0.01 (= ≥ 0.437) AND > perspective(0.377).**
- 통과 → **통합축으로 동결**(fiction present 동급 + 다화자 turnover형태 = LoCoMo 재설계 회피). frozen protagonist 슬롯 = dominant_turnover.
- 실패 → **present로 동결**, dominant_turnover는 LoCoMo용 [[locomo-downstream-deferred]] 이월(폐기 아님).
- **딱 1회 추출. 미달 시 프롬프트 추격·재프롬프트 금지**(D-M5 selection-on-dev 방지). 프롬프트 문구는 사전작성분 1개만 사용.
- 다화자 전이(AMI eval 35편) 실검증은 **현 fiction 동결 scope 밖** — 통과해도 "다화자 전이 확인"이라 주장 금지, 구조적합성 근거로만.
- boundary count≥3(t\*) 구조가 안 움직여도 실패 아님(단일축 교체이므로) — 판정은 최종 Score.

## 수정 로그
- 2026-07-22: §6 dominant-actor turnover 통합축 후보 사전등록(1회, 채택바 ≥0.437 & >0.377).
- 2026-07-22: §6 **결과 = 미채택.** 1회 추출(deepseek, `*_s5_eismv2_domact`). dev-3 3편평균:
  dominant 0.401 < present 0.418(같은 추출) < 바 0.408, 그리고 frozen 바 0.437에도 미달 → **양 기준 모두 실패(강건).**
  perspective(0.362)는 상회. **부작용 발견:** dominant_actor를 protagonist 프롬프트에 번들하자 present가 frozen 0.447→0.418로
  동반 하락(모든 변형 ~0.02~0.03↓) → **코드에서 dominant_actor 추가분 전량 revert**(frozen 캐시 present=0.447 보존, prompt_sha256 정합).
  단서: dominant는 monkey에서만 present 상회(0.283 vs 0.259) = 다인물/서술 문서 강점, 다화자 전이가설과 일관.
  **결정: fiction = present 동결. dominant_turnover = LoCoMo 이월([[locomo-downstream-deferred]]), 추격 안 함.**
  기록: 추출 `outputs/eacl_phase0/*_s5_eismv2_domact/`, 채점 `scripts/eacl_phase0/score_domact_turnover.py`. 추격/재프롬프트 없음(사전등록 준수).
