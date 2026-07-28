# ② 진단 — findings A·H·B①·B② 스케일 게이트

> ①B human 표본이 연 설계 발견의 **문서 스케일 빈도**를 캐시 run1(12편: 유명 6 fam / 무명 6 obs)에서
> gold-free 측정. 예산 0(라벨 재호출 없음, regex 스크리닝). E(time 비대칭)=`time_marker_position_diag.py`
> 별도 완료 → §E 요약. **판정 기준은 결과 보기 전 사전 선언**(no-postresult-criterion-change).

> ⚠️ regex 휴리스틱은 E 진단과 동급 **스크리닝 도구**(정밀도 아닌 방향·규모). cue 감지는 불완전 —
> A(누락)는 하한, H(no-cue non-unspec)는 상한으로 해석.

## 실험 setup

- **목적:** ①B human 표본 발견 A·H·B①·B②가 **문서 스케일에서 흔한가**를 게이트(④ 재설계 포함 여부).
  결함 유형은 이미 표본으로 확정됨 — 여기선 *빈도/규모*만 잰다.
- **데이터:** `outputs/eacl_phase0/repcorpus_s5_eismv2_run1` (frozen `p0s5-eism-v2` run1 라벨) + `outputs/eacl_phase0/repcorpus_master.jsonl`
  (문장 텍스트). 12편(fam 6 유명 / obs 6 무명). **gold 경계 = 전 문서 0개**(repcorpus=QC용, 분절 gold 아님).
- **방법:** 캐시 `{axis}_states.jsonl`(raw, carry 전) + regex cue 휴리스틱. B② leave-one-out 은
  `llm_s5_states.carry_forward`+`reset_intentionality` 재사용. Score=`hi_ontop.dts_scoring.score_dialogues`.
- **판정 기준(사전 동결):** 각 절(§A/§H/§B①/§B②) 상단에 결과 보기 전 명시(rubric-before-results).
- **한계:** cue 감지 불완전(A 하한·H 상한) + repcorpus gold 0 → **Score 무의미**(§한계). 신호 축은 reset율.

## A. space list-miss (under) — 무대(탈것/공중) 목록누락 → unspecified

**판정 기준(사전):** 탈것/공중 목록누락 문장이 ≥1 문서에서 눈에 띄게(해당문서 unspecified의 유의 비율) 발생 + obs>fam 편중 → ④에 A 포함(조건부 충족).

| doc | 장르 | n | list크기 | unspec율 | 최장 unspec run | 탈것/공중 목록누락 문장 |
|---|---|---|---|---|---|---|
| fam_bierce_owlcreek | fam | 201 | 9 | 41.8% | 8 | 3 |
| fam_doyle_speckled | fam | 250 | 23 | 87.2% | 48 | 5 |
| fam_gilman_wallpaper | fam | 250 | 16 | 73.6% | 21 | 0 |
| fam_jewett_heron | fam | 185 | 13 | 68.1% | 22 | 4 |
| fam_london_fire | fam | 250 | 34 | 54.4% | 12 | 5 |
| fam_twain_frog | fam | 92 | 4 | 75.0% | 21 | 2 |
| obs_25744 | obs | 250 | 7 | 94.4% | 67 | 2 |
| obs_28922 | obs | 104 | 2 | 96.2% | 43 | 1 |
| obs_37511 | obs | 46 | 16 | 78.3% | 14 | 3 |
| obs_74140 | obs | 250 | 19 | 93.2% | 51 | 10 |
| obs_74162 | obs | 246 | 16 | 82.1% | 64 | 2 |
| obs_77176 | obs | 121 | 9 | 73.6% | 15 | 26 |

- 탈것/공중 목록누락 문장 합계: **fam 19 / obs 44**.

**예시(unspecified 인데 탈것/공중 어휘 有, 목록엔 없음):**
- `fam_bierce_owlcreek#82` [scout]: He was a Federal scout.
- `fam_bierce_owlcreek#131` [cloud]: He heard a second report, and saw one of the sentinels with his rifle at his shoulder, a light cloud of blue s
- `fam_bierce_owlcreek#189` [air]: His tongue was swollen with thirst; he relieved its fever by thrusting it forward from between his teeth into 
- `fam_doyle_speckled#40` [train]: You have come in by train this morning, I see.
- `fam_doyle_speckled#43` [cart]: You must have started early, and yet you had a good drive in a dog-cart, along heavy roads, before you reached
- `fam_doyle_speckled#48` [cart]: There is no vehicle save a dog-cart which throws up mud in that way, and then only when you sit on the left-ha
- `fam_doyle_speckled#50` [train]: “I started from home before six, reached Leatherhead at twenty past, and came in by the first train to Waterlo
- `fam_doyle_speckled#230` [train]: But I shall return by the twelve o’clock train, so as to be there in time for your coming.
- `fam_jewett_heron#31` [air]: However, it was not much farther to the house, and the air was soft and sweet.
- `fam_jewett_heron#128` [air]: She had always believed that whoever climbed to the top of it could see the ocean; and the little girl had oft
- `fam_jewett_heron#138` [mount,sky]: There was the huge tree asleep yet in the paling moonlight, and small and hopeful Sylvia began with utmost bra
- `fam_jewett_heron#159` [clouds]: Sylvia could see the white sails of ships out at sea, and the clouds that were purple and rose-colored and yel

## H. space context-leak (over) — 장소단서 없는데 non-unspecified (raw, carry 전)

**판정 기준(사전):** no-cue 문장 non-unspecified율(상한)이 전역 non-unspecified율 대비 낮지 않고 절대적으로도 상당(대략 ≥0.2)하면 leak 스케일 실재 → ④ H 포함(이미 ✅, 규모 확인).

| doc | 장르 | n | cue문장 non-unspec | no-cue문장 n | **no-cue non-unspec(=leak 상한)** |
|---|---|---|---|---|---|
| fam_bierce_owlcreek | fam | 201 | 73.4% | 73 | 31.5% |
| fam_doyle_speckled | fam | 250 | 22.9% | 119 | 1.7% |
| fam_gilman_wallpaper | fam | 250 | 39.5% | 131 | 14.5% |
| fam_jewett_heron | fam | 185 | 44.5% | 75 | 13.3% |
| fam_london_fire | fam | 250 | 58.4% | 96 | 25.0% |
| fam_twain_frog | fam | 92 | 31.8% | 48 | 18.8% |
| obs_25744 | obs | 250 | 12.7% | 148 | 0.7% |
| obs_28922 | obs | 104 | 7.0% | 61 | 1.6% |
| obs_37511 | obs | 46 | 34.5% | 17 | 0.0% |
| obs_74140 | obs | 250 | 16.0% | 144 | 0.0% |
| obs_74162 | obs | 246 | 45.6% | 156 | 1.9% |
| obs_77176 | obs | 121 | 33.8% | 44 | 13.6% |

- 전체 no-cue 문장 1112 중 non-unspecified ≈ 98 (8.8% — leak 상한).

**예시(no-cue 인데 위치 부여):**
- `fam_bierce_owlcreek#2` → owl_creek_bridge: A rope closely encircled his neck.
- `fam_bierce_owlcreek#17` → owl_creek_bridge: The captain stood with folded arms, silent, observing the work of his subordinates, but making no sign.
- `fam_bierce_owlcreek#24` → owl_creek_bridge: Evidently this was no vulgar assassin.
- `fam_bierce_owlcreek#29` → owl_creek_bridge: The end upon which the civilian stood almost, but not quite, reached a fourth.
- `fam_bierce_owlcreek#32` → owl_creek_bridge: The arrangement commended itself to his judgement as simple and effective.
- `fam_bierce_owlcreek#35` → railroad_bridge: A piece of dancing driftwood caught his attention and his eyes followed it down the current.
- `fam_bierce_owlcreek#40` → owl_creek: And now he became conscious of a new disturbance.
- `fam_bierce_owlcreek#91` → owl_creek: He was conscious of motion.
- `fam_bierce_owlcreek#98` → owl_creek: He was still sinking, for the light became fainter and fainter until it was a mere glimmer.
- `fam_bierce_owlcreek#101` → owl_creek_bridge: No; I will not be shot; that is not fair.
- `fam_bierce_owlcreek#113` → owl_creek: His whole body was racked and wrenched with an insupportable anguish!
- `fam_bierce_owlcreek#114` → owl_creek: But his disobedient hands gave no heed to the command.

## B①. 대사 화행별 P(goal≠none) — 수행적 vs 논의

**판정 기준(사전):** P(goal≠none|수행적 대사) − P(goal≠none|논의 대사) ≤ 0.10 AND 수행적 절대 P(goal≠none) < 0.5 → 모델이 화행을 무시(수행적도 none) → ① 확정.

**event_type 별 P(goal≠none) (참조 상한/하한):**

| event_type | n | P(goal≠none) |
|---|---|---|
| state_description | 1167 | 19.9% |
| physical_action | 232 | 42.7% |
| dialogue | 226 | 31.4% |
| perception_or_observation | 185 | 27.6% |
| other | 150 | 34.7% |
| conflict_or_violence | 94 | 34.0% |
| thought_or_emotion | 75 | 25.3% |
| movement_or_travel | 63 | 60.3% |
| work_or_creation | 49 | 32.7% |
| transaction_or_exchange | 4 | 50.0% |

**dialogue 안 화행 분해:**

| 화행 | n | P(goal≠none) |
|---|---|---|
| 수행적(명령/질문/약속/계획) | 89 | 41.6% |
| 순수 논의 | 137 | 24.8% |

- 수행적 − 논의 = **+16.8%**, 수행적 절대 = **41.6%**.
- **① 판정: 기각/불확정(수행적이 goal 더 잘 발화)**.

**수행적인데 none(① 증거 후보):**
- `fam_doyle_speckled#11`: “What is it, then—a fire?
- `fam_doyle_speckled#27`: I am glad to see that Mrs.
- `fam_doyle_speckled#40`: You have come in by train this morning, I see.
- `fam_doyle_speckled#102`: “She died just two years ago, and it is of her death that I wish to speak to you.
- `fam_doyle_speckled#189`: “Were there gipsies in the plantation at the time?
- `fam_doyle_speckled#236`: I shall look forward to seeing you again this afternoon.
- `fam_jewett_heron#44`: But who could have foreseen such an accident as this?
- `fam_jewett_heron#50`: Where’d she tucked herself away this time, Sylvy?
- `fam_jewett_heron#73`: “Dan, my boy, was a great hand to go gunning,” she explained sadly.
- `fam_jewett_heron#85`: “So Sylvy knows all about birds, does she?” he exclaimed, as he looked round at the little girl who sat, very 

## B②. goal eventiveness + leave-one-goal-out — 상시(pervasive)목표 실재+영향

**판정 기준(사전):** span≥0.6(문서길이 대비) 목표가 ≥1 문서에 존재(=상시목표 실재) AND leave-one-out 이 그 문서 intentionality reset율 또는 Score 를 |Δ|>0(비영) 변화 → 상시목표가 경계신호 오염 확증 → ④ B② 포함(이미 ✅, 규모 확인).

**문서별 상시목표(span≥0.6) + leave-one-out (Δreset / Δscore, 5축 t\*=3):**

| doc | 장르 | n목표 | 상시목표(span) | base reset | base score |
|---|---|---|---|---|---|
| fam_bierce_owlcreek | fam | 10 | execute_hanging(0.98); escape_execution(0.61) | 17.5% | 0.000 |
| fam_doyle_speckled | fam | 6 | investigate_case(0.94); solve_mystery(0.88); accompany_holmes(0.88) | 8.0% | 0.000 |
| fam_gilman_wallpaper | fam | 16 | get_well(0.94); have_perfect_rest(0.76); understand_wallpaper_pattern(0.75); hide_being_awake(0.72); avoid_opposition(0.70) | 19.7% | 0.000 |
| fam_jewett_heron | fam | 11 | drive_cow_home(0.65) | 12.5% | 0.000 |
| fam_london_fire | fam | 8 | reach_camp(0.70); keep_warm(0.64) | 14.5% | 0.000 |
| fam_twain_frog | fam | 7 | right_himself(0.90); avoid_boredom(0.64) | 23.1% | 0.000 |
| obs_25744 | obs | 12 | evaluate_situation(0.95); test_blood_susceptibility(0.80); maintain_team(0.74); resist_transformation(0.69) | 16.1% | 0.000 |
| obs_28922 | obs | 2 | — | 1.9% | 0.075 |
| obs_37511 | obs | 10 | — | 22.2% | 0.000 |
| obs_74140 | obs | 10 | get_home(0.79) | 6.8% | 0.000 |
| obs_74162 | obs | 7 | escape_capture(0.83) | 6.1% | 0.000 |
| obs_77176 | obs | 5 | bomb_target(0.93); die_gloriously(0.88); escape(0.77) | 20.8% | 0.000 |

**leave-one-goal-out 상세(상시목표 보유 문서):**

| doc | drop goal | span | reset율 | Δreset | score | Δscore |
|---|---|---|---|---|---|---|
| fam_bierce_owlcreek | execute_hanging | 0.98 | 14.0% | -3.5% | 0.000 | +0.000 |
| fam_bierce_owlcreek | escape_execution | 0.61 | 11.5% | -6.0% | 0.000 | +0.000 |
| fam_bierce_owlcreek | ALL_pervasive | — | 8.5% | -9.0% | 0.000 | +0.000 |
| fam_doyle_speckled | investigate_case | 0.94 | 4.0% | -4.0% | 0.000 | +0.000 |
| fam_doyle_speckled | solve_mystery | 0.88 | 5.6% | -2.4% | 0.000 | +0.000 |
| fam_doyle_speckled | accompany_holmes | 0.88 | 6.8% | -1.2% | 0.000 | +0.000 |
| fam_doyle_speckled | ALL_pervasive | — | 2.0% | -6.0% | 0.000 | +0.000 |
| fam_gilman_wallpaper | get_well | 0.94 | 17.7% | -2.0% | 0.000 | +0.000 |
| fam_gilman_wallpaper | have_perfect_rest | 0.76 | 16.9% | -2.8% | 0.000 | +0.000 |
| fam_gilman_wallpaper | understand_wallpaper_pattern | 0.75 | 16.5% | -3.2% | 0.000 | +0.000 |
| fam_gilman_wallpaper | hide_being_awake | 0.72 | 18.9% | -0.8% | 0.000 | +0.000 |
| fam_gilman_wallpaper | avoid_opposition | 0.70 | 18.1% | -1.6% | 0.000 | +0.000 |
| fam_gilman_wallpaper | ALL_pervasive | — | 9.2% | -10.4% | 0.000 | +0.000 |
| fam_jewett_heron | drive_cow_home | 0.65 | 9.2% | -3.3% | 0.000 | +0.000 |
| fam_jewett_heron | ALL_pervasive | — | 9.2% | -3.3% | 0.000 | +0.000 |
| fam_london_fire | reach_camp | 0.70 | 9.2% | -5.2% | 0.000 | +0.000 |
| fam_london_fire | keep_warm | 0.64 | 9.6% | -4.8% | 0.000 | +0.000 |
| fam_london_fire | ALL_pervasive | — | 4.4% | -10.0% | 0.000 | +0.000 |
| fam_twain_frog | right_himself | 0.90 | 19.8% | -3.3% | 0.000 | +0.000 |
| fam_twain_frog | avoid_boredom | 0.64 | 19.8% | -3.3% | 0.000 | +0.000 |
| fam_twain_frog | ALL_pervasive | — | 16.5% | -6.6% | 0.000 | +0.000 |
| obs_25744 | evaluate_situation | 0.95 | 8.8% | -7.2% | 0.000 | +0.000 |
| obs_25744 | test_blood_susceptibility | 0.80 | 12.9% | -3.2% | 0.000 | +0.000 |
| obs_25744 | maintain_team | 0.74 | 12.4% | -3.6% | 0.000 | +0.000 |
| obs_25744 | resist_transformation | 0.69 | 14.9% | -1.2% | 0.000 | +0.000 |
| obs_25744 | ALL_pervasive | — | 3.6% | -12.4% | 0.000 | +0.000 |
| obs_74140 | get_home | 0.79 | 5.6% | -1.2% | 0.000 | +0.000 |
| obs_74140 | ALL_pervasive | — | 5.6% | -1.2% | 0.000 | +0.000 |
| obs_74162 | escape_capture | 0.83 | 4.1% | -2.0% | 0.000 | +0.000 |
| obs_74162 | ALL_pervasive | — | 4.1% | -2.0% | 0.000 | +0.000 |
| obs_77176 | bomb_target | 0.93 | 8.3% | -12.5% | 0.000 | +0.000 |
| obs_77176 | die_gloriously | 0.88 | 15.0% | -5.8% | 0.000 | +0.000 |
| obs_77176 | escape | 0.77 | 15.0% | -5.8% | 0.000 | +0.000 |
| obs_77176 | ALL_pervasive | — | 3.3% | -17.5% | 0.000 | +0.000 |

- 상시목표(span≥0.6) 보유 문서 존재: **True**.
- Δreset/Δscore 가 비영이면 상시목표가 경계신호를 실제로 흔든다(억제 또는 가짜경계).

## E. time opens-with 비대칭 (별도 완료 — `time_marker_position_diag.py`)

요약(FINDINGS §E, 캐시 실증): non-absent 83건 중 sequence 50(문중 24%) vs jump 29(문중 48%).
jump 절반이 문중 발화 = 위치조건 비대칭 실증. **④에 위치조건 대칭화 포함(확정).**

## 해석

### A (space list-miss, under) — 실재하나 **국소집중**(regex FP 주의)
- 탈것/공중 목록누락 문장은 **`obs_77176`=26건으로 압도적**, 2위=10건. `obs_77176`(1차대전 항공 pulp)는
  Stage-1 locations 가 지상 airdrome 만이라 '비행기/조종석/공중' 무대 문장(#8 rearmost plane, #10 tractor
  biplane, #25 leading edge of wing …)이 전부 unspecified — **진짜 어휘공백**(FINDINGS A 사례 재현·확대).
- 타 문서 히트는 대부분 **regex 오검**: 'by **train**'(언급) · '**cloud** of smoke' · Federal '**scout**'(사람) 등
  = 무대 아님. ⇒ A는 **탈것/공중/항해가 무대인 서사에 국소적**, 전 장르 보편 아님. obs>fam(44 vs 19)이나 대부분 FP.
- **판정:** A 확정, 단 빈도는 genre-conditional. fix(Stage-1 physical_setting 정의에 vehicle/airspace 예시)는
  **저비용·정합**(EISM space 와 충돌 안 함)이라 ④ 포함할 가치 有 — 단 '흔해서'가 아니라 '해당 서사에 치명적'이라서.

### H (space context-leak, over) — 기제 실재(측정순수성), 다수는 맥락상 정답
- no-cue 문장의 non-unspecified율(leak 상한) 전체 8.8%, 최고 `fam_bierce_owlcreek` 31.5%.
  = 모델이 **문장 자체 근거 없이도** carry 전 raw 에서 위치를 부여(target-only 소프트 위반) — FINDINGS H 기제 확증.
- 단 예시(#2 'A rope encircled his neck'→owl_creek_bridge 등)는 **맥락상 옳은 위치**가 많음 → boundary 에 꼭
  해롭진 않으나 **space reset 이 순수 문장근거 아닌 맥락기억에 일부 의존**한다는 측정순수성 문제는 실재.
- **판정:** H fix('문장 근거 없으면 반드시 unspecified, 맥락추론 금지') ④ 포함(✅) — 규모는 중간, 순수성 명분.

### B① (대사 화행) — 사전기준 **미충족 → ① 기각**(예시 교체는 선택적 보강으로 강등)
- 수행적 대사 P(goal≠none)=41.6% > 논의 24.8%, gap **+16.8%**(기준 ≤10% 위반). 모델은 화행을 **무시하지 않음**
  (명령·질문·계획 대사에서 goal 을 더 발화). event_type 상 movement 60%·physical 43% 가 상한, dialogue 31% 는 중간.
- ⇒ '대사=none 오독'은 **스케일에서 확증 안 됨**(none-rate 0.73 은 대사 아닌 state_description 19.9%·다수가 주범).
- **판정:** ① 구성타당도 결함으로는 **기각**. FINDINGS 대로 '대사→목표' 예시 추가는 *무해한 명료화*이므로 ④에서
  **선택적 경보강**만(load-bearing 아님). B②를 먼저 치우면 none-gap 상당수가 해소될 것.

### B② (상시목표) — **확정·보편**(10/12 문서), reset 대량 기여
- span≥0.6 상시목표 보유 문서 **10/12**. Yellow Wallpaper 5개(get_well·have_perfect_rest·
  avoid_opposition …)=FINDINGS control_self 계열 재현. 상시목표가 이산·순차 goal 가정을 깸.
- **leave-one-out:** 상시목표 전부 제거 시 intentionality reset율이 대폭 하락(예: fam_bierce_owlcreek 17.5%→8.5%; fam_doyle_speckled 8.0%→2.0%; fam_gilman_wallpaper 19.7%→9.2%; fam_jewett_heron 12.5%→9.2% …)
  = 상시목표가 현재 intentionality reset 의 **상당분을 생성**(산발 re-entry·목표전환). 노이즈 경계 후보.
- **판정:** B② ④ 포함(✅) 확정. Stage-1 goals 를 bounded commitment 로 제한 + 확산적 목표(enjoy_leave류) 규정.

### E (time 비대칭) — 별도 확정(§E). ④ 위치조건 대칭화 포함(✅).

## 판정 (④ 반영 결정 → FINDINGS/HANDOFF 갱신)

| finding | 사전기준 | 진단 결과 | ④ 포함 |
|---|---|---|---|
| A list-miss | 국소 유의+obs편중 | `obs_77176`=26 압도(국소), 타 FP | **포함(정합·저비용, genre국소)** |
| H context-leak | no-cue non-unspec 상당 | 전체 8.8%, 최고 31.5% | **포함 ✅(순수성)** |
| B① 대사화행 | gap≤10%&수행<50% | gap +16.8% (위반) | **기각 → 선택적 보강만** |
| B② 상시목표 | span≥0.6 존재+영향 | 10/12, reset 대량 | **포함 ✅** |
| E time 비대칭 | 별도 | jump 문중 48% | **포함 ✅** |

## 한계

- **repcorpus gold 경계 = 전 문서 0** → Score·Δscore(B② 표) **무의미**(F1 정의 불가, base_score≈0).
  findings 의 **Score 영향은 여기서 측정 불가** → ④ dev-3(pieman/tunnel/monkey, gold 有) Gate C 로 이월. 여기 신호축=reset율.
- cue/화행/탈것 감지 = regex 스크리닝(E 진단 동급). A=하한(누락 과소), H=상한(no-cue 에 암묵무대 혼입), B①=화행분류 noise.
  방향·규모는 견고하나 소수점 정밀 아님.
- 12편·genre 교란(fam=19c 문학 / obs=20c pulp) 잔존 — obs>fam 편중은 A/B② 에서 보이나 era/장르와 얽힘.
- B① 기각은 '스케일 확증 실패'지 '① 절대 없음' 아님 — 개별 오독은 존재(예시 목록). 예시 보강은 무해.

