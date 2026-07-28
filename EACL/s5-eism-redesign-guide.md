# M3-S5-Extractor — EISM/Indexter 정합 프롬프트 재설계 가이드라인 (REDESIGN SPEC v0.1)

> **상태:** 제안(PROPOSAL). 동결 전까지 canonical 아님.
> **트리거:** README 「다음 단계」 #2 — *"저비용 라벨러용 S5 프롬프트 재설계 (D-M13⑥). dev 3편에서만 튜닝, 시험지 봉인. carry-forward 프롬프트 명시 + goal 축 재정의."*
> **동결 시 처리:** 이 문서가 확정되면 `phase0-annotation-guide.md` §1~3의 추출 스펙을 **대체**하고, 결정은 ledger에 **D-B9(가칭)** 로 기록한다. 그 전까지 annotation-guide가 여전히 canonical.
> **선행 미완료 (README):** ① 저비용 라벨러 모델 확정(D-M13⑥: deepseek-v4-pro 기각, fable5 예산밖) ② goal divergence 진단. **이 둘이 끝나기 전에는 프롬프트를 동결하지 않는다.**
> **이론 근거:** Zwaan·Langston·Graesser (1995) EISM 5지표 + Cardona-Rivera et al. (2012) Indexter. Indexter 논문은 인덱스를 "*다른 사건과 무관한 사건 자체의 속성*"(§2.1.3 각주 2)으로, "*이분법(dichotomous)*"(§2.1.3·6.1)으로 정의한다.

---

## 0. 절대 불변 가드레일 (재설계로도 못 건드림 — 위반 시 측정 무효)

`phase0-annotation-guide.md` §0 + §7 + README 전제에서 승계. **프롬프트를 어떻게 바꾸든 아래는 고정.**

| # | 규칙 | 근거 | 재설계에서의 의미 |
|---|---|---|---|
| G1 | LLM에 변화/리셋/경계/이전상태/novelty를 **절대 묻지 않는다** | S1 halo, AUC 1.0 오염 | EISM/Indexter "충실화"를 이유로 pairwise 비교·경계 질의를 넣지 말 것 |
| G2 | Stage 2는 **target 문장 단독.** 왼쪽 문맥·다른 문장·다른 축·CosDrift 점수 미제공 | S2 carryover, AUC 0.55 | **이것이 Indexter 충실성의 핵심**(§1 참조) — 완화 대상 아님 |
| G3 | **닫힌 목록에서만 선택.** 자유서술 금지 | S3 표현 노이즈, AUC 0.39 | 새 필드도 반드시 enum/목록 |
| G4 | 변화 판정·carry-forward는 **100% 코드** | 판정 오염 방지 | reset 규칙은 코드에 남는다. 프롬프트는 *값*만 뽑음 |
| G5 | 모든 enum·규칙·임계값은 **gold 확인 전 동결** | 사후 fitting 방지 | 프롬프트를 한 글자라도 바꾸면 `prompt_sha256`가 바뀌어 **현 동결이 전부 무효** → dev-3 재튜닝 후 재동결 필수 |
| G6 | 튜닝은 **dev 픽션 3편(pieman/tunnel/monkey)에서만.** 시험지(held-out)는 봉인 | §7 selection-on-test(D-M5) 재발 방지 | 7개 전부에 돌려 최고 설정 고르기 = **금지**(그게 D-M5 오류) |

> **🔴 메타 가드레일:** 재설계 → 7개 데이터 재평가 → 최고 variant 선택은 **D-M5 오염의 정확한 반복**이다. 재설계는 dev-3에서 동결하고, 봉인된 held-out(§6)에서 **1회만** 확인한다.

---

## 1. EISM/Indexter ↔ 현 파이프라인 정합 지도 (재설계의 개념적 토대)

### 1.1 왜 target-only가 "Indexter 위반"이 아니라 "Indexter 충실"인가

Indexter의 salience는 두 사건 간 공유 인덱스 수로 계산되는 **관계량**이다. 순진하게 "Indexter를 따른다"고 LLM에게 사건 쌍을 비교시키면 G1·G2를 즉시 위반한다. 그러나 Indexter는 인덱스 자체를 **사건 단독 속성**으로 정의하고, *공유 여부*는 그 속성들을 비교해 별도로 계산한다.

```
Indexter:  event.index_value  (사건 속성)  →  code: shares(e_i, e_j)  →  salience
현 S5:     LLM(target-only)    (사건 속성)  →  code: reset(s_{t-1}, s_t) →  count → boundary
```

**⇒ 완전히 같은 분해다.** LLM=인덱스 값 추출기, 코드=공유/리셋 판정기. target-only는 Indexter의 "인덱스는 사건 속성" 정의를 하드웨어로 강제한 것이다. **따라서 충실화는 target-only를 유지한 채 이뤄진다.**

### 1.2 5지표 정합표 (축 키 · Indexter 정의 · 현 근사 · 격차)

| 축 키 (신규 — 코드·문서 공통) | 구 키 (폐기) | Indexter/Zwaan 공유 기준 | 현 S5 추출·리셋 | 충실도 | 조치 |
|---|---|---|---|---|---|
| **`protagonist`** | (동일) | 둘 다 *단일 지정 주인공*이 관여(전경화 무관, 특수 지표) | `perspective` 단독(charset 제외, D-B7/B8) | 부분 — perspective는 근사, "단일 주인공 관여"는 아님 | §3.1: null 명시화 유지 + **protagonist-present boolean을 실험 후보**로 |
| **`time`** | (동일) | 동일 시간 프레임(Zwaan 1996: 순차 지각 + 명시적 시간 불연속 없음) | `temporal_shift_marker` 단독(tense 제거, §2.3) | **높음** — marker=명시적 불연속. 시제는 EISM 무관 | §3.2: **변경 없음.** tense 제거가 더 충실 |
| **`space`** | ~~`place`~~ | 동일 공간 영역 | `location` 닫힌목록 + carry-forward | **높음** | §3.3: carry-forward 계약을 프롬프트에 명시화 |
| **`causality`** | (동일) | 인과 관계(필요성+반사실 검증), 직접·전이. enablement/motivation/psychological/physical | `event_type` 변화 OR `aspect`==inceptive | **최저** — event_type 변화는 인과가 아님 | §3.4: **문장 단독 인과 접속표지**를 실험 후보로(time축과 동형 절충) |
| **`intentionality`** | ~~`goal`~~ | 동일 목표 달성 계획(같은 frame of commitment) | `active_goal` 닫힌목록 + carry-forward | 부분 — 목표 슬롯. 계획 소속은 아님 | §3.5: null 명시화 + **재정의(선행: goal divergence 진단)** |

> **핵심:** 5지표 집합은 이미 EISM와 1:1이다(구 `place`=space, 구 `goal`=intentionality). 재설계의 실질은 *이름 교체가 아니라* (a) 느린 변수의 null 계약 명시화, (b) causality·protagonist의 이론 충실도 제고 실험, (c) 논문용 정식명 정렬이다.

---

## 2. 명명 — 하드 리네임 확정 (✅ 결정됨)

**결정:** 축 키를 EISM 정식 어휘로 **전량 교체**한다. alias 계층을 두지 않는다. 코드·산출물·문서가 같은 이름을 쓴다.

```
protagonist   (불변)
place       → space
time          (불변)
causality     (불변)
goal        → intentionality
```

**근거:** 저비용 라벨러 모델 교체(D-M13⑥)로 **어차피 전 문서를 재추출**한다. 기존 캐시를 보존할 유인이 사라졌으므로, alias의 유일한 정당화였던 "캐시·재현 체인 보호"가 소멸했다. 리네임 비용이 가장 싼 시점이 지금이다.

### 2.1 치환 대상 (누락 시 §8-6에서 걸림)

| 대상 | 구 | 신 |
|---|---|---|
| `AXES` 튜플 | `"place"`, `"goal"` | `"space"`, `"intentionality"` |
| `AXIS_FIELDS` 키 | 동일 | 동일 |
| 리셋 함수명 | `reset_place`, `reset_goal` | `reset_space`, `reset_intentionality` |
| `stage2_prompt`/`stage2_schema` 분기 | `axis == "place"` 등 | 신규 키 |
| Stage2 캐시 파일 | `place_states.jsonl`, `goal_states.jsonl` | `space_states.jsonl`, `intentionality_states.jsonl` |
| `reset_matrix.jsonl` 컬럼 | `place`, `goal` | `space`, `intentionality` |
| `main()` carry-forward 분기 | `states["place"]`, `states["goal"]` | 신규 키 |
| 다운스트림(평가·ablation·M4~M7 앵커 로더) | 구 컬럼 참조 | 신규 컬럼 |

> **필드명은 안 바꾼다.** `location`, `active_goal`은 축 키가 아니라 축 *내부* 필드다. 축 키만 EISM 어휘로 맞추고, 필드는 프롬프트 문구와 함께 §3에서 다룬다. (`active_goal` 변경은 §3.5 goal 재정의 결정에 종속.)

### 2.2 실행 순서 (재추출과 묶어서 1회)

1. **구 캐시 삭제** — `{out_dir}/{doc_id}/place_states.jsonl`, `goal_states.jsonl`, 구 `reset_matrix.jsonl`. (`story_lists.json`은 Stage1 산출물이라 축 키와 무관하나, Stage1 프롬프트도 §4.1에서 바뀌면 함께 폐기.)
2. **코드 전량 치환** — §2.1 표 + 다운스트림.
3. **`VERSION` 갱신** — 예: `p0s5-story-list-v1` → `p0s5-eism-v2`. 구 버전 산출물과 섞이지 않게.
4. **신규 라벨러 모델로 dev-3 재추출.**
5. **ledger에 기록** — 리네임·VERSION·모델 교체를 **한 D코드**로 묶어 append(D-B9 가칭).

> **🔴 구 수치 인용 금지:** 리네임 + 모델 교체 후 산출물은 기존 동결 수치(dev 0.634 등)와 **다른 실행**이다. 구 수치를 신 파이프라인 성능으로 인용하지 말 것. 비교하려면 §6 절차로 새로 측정한다.

---

## 3. 축별 재설계 스펙 (추출=LLM / 판정=코드)

각 축: **LLM이 뽑는 것(닫힌 목록/enum)** · **프롬프트 필드 문구** · **코드 리셋(불변)** · **EISM 근거**. 리셋 규칙은 G4에 따라 코드 유지이며 대부분 불변.

### 3.1 `protagonist`

- **동결 추출:** `perspective` ∈ {first, second, third, none} — 문장 중심 행위자/경험자의 문법 인칭. (`characters_present`는 §4.2·D-B8로 **count 제외 유지**.)
- **코드 리셋(불변):** `perspective` 변함 AND 둘 다 ≠ none.
- **EISM 근거:** EISM protagonist 지표는 *단일 지정 주인공*의 관여 여부이며, 전경화와 무관하게 salience에 기여하는 특수 지표. perspective는 1인칭 서사(held-out = *Born a Crime* 회고록, 1인칭)에서 "주인공(화자)의 장면 내/외"를 근사한다.
- **⚠️ charset 되돌리지 말 것:** EISM은 전체 캐스트가 아니라 *한* 주인공만 본다. charset이 죽은 이유(조연이 장면 안팎을 수시로 드나듦 → 상시 흔들림, dev 0.538, 병용 시 0.542로 희석)가 곧 **비-EISM적**이었다.
- **🧪 실험 (실행 확정, dev-3 전용):** `protagonist_present` ∈ {true, false} — "이 문장의 문법 주어/경험자가 *지정 주인공*인가"(Stage 1이 뽑은 주인공 1인 대비 boolean). charset의 전체-캐스트 노이즈를 피하면서 EISM "단일 주인공 관여"에 더 충실. 리셋 = boolean 뒤집힘. **perspective와 별도 컬럼으로 로깅해 dev-3에서 승패를 데이터가 결정**(D-B7 절차 준용). 이기지 못하면 perspective 단독 유지.

### 3.2 `time`

- **동결 추출:** `temporal_shift_marker` ∈ {absent, sequence, jump, simultaneity}. (`tense`는 추출은 하되 리셋에서 **제외 유지**.)
- **코드 리셋(불변):** `temporal_shift_marker` ∈ {sequence, jump, simultaneity}.
- **EISM 근거:** Zwaan(1996) — 두 사건은 "순차적으로 지각되고 명시적 시간 불연속이 없으면" 동일 시간 프레임을 공유. marker가 바로 그 *명시적 불연속*이다. **시제는 EISM 어디에도 없다.**
- **⚠️ tense 되돌리지 말 것:** tense 조건 제거는 튜닝이 아니라 **결함 수정**(Monkey time 리셋 119회 중 ~103회가 시제 왕복 가짜알람; dev 0.598→0.634). **버그 수정과 이론 충실성이 같은 방향** — tense 복원은 둘 다 배반한다. `−time`(0.628)보다 `marker만`(0.634)이 우수하므로 축은 유지.

### 3.3 `space` (구 `place`)

- **동결 추출:** `location` ∈ Stage1 `locations` ∪ {`unspecified`}.
- **코드(불변):** carry-forward(`unspecified`면 직전 명시값 유지, 새 명시값이 덮어씀) → 리셋 = carry 후 값 변함 AND 둘 다 ≠ unspecified.
- **EISM 근거:** 동일 공간 영역이면 공유. 느린 변수(target-only 추출 시 86% unspecified)라 carry-forward 없이는 死(reset 1/212)였고 적용 후 부활(0.498→0.572).
- **🔴 재설계 핵심 = 해석 고정 (D-M13⑥이 요구한 바로 그것):** deepseek-v4-pro가 dev 3편 Gate C 전부 실패한 원인은 **능력이 아니라 `location`의 해석 차이**였다 — fable5는 "이 *장면*이 어디서 벌어지나"(연속성), deepseek는 "이 *문장에 장소 단어가 있나*"(문자적). 결과: 연속 장면을 전부 unspecified 처리(monkey 82.4%→**96.3%**) + 스쳐가는 지명("home in England")을 집어 carry-forward 오염 → pieman ΔScore **−0.253**.
- **⇒ 프롬프트가 fable5의 해석을 명문화해야 한다.** 3줄로 고정:
  1. 장소 **단어를 찾는 것이 아니라**, 이 문장이 **그리는 장면의 무대**를 고른다. (예: "그는 침대에서 뒤척였다" → `bedroom`. 단어가 없어도 무대는 침실.)
  2. **언급만 된** 장소는 고르지 않는다. 인물이 화제로 삼거나 회상하는 지명은 무대가 아니다. (예: "영국 집 얘기를 했다" → 무대는 영국이 **아님**.)
  3. 무대를 판단할 근거가 **이 문장 안에 전혀 없을 때만** `unspecified`.
- **G2 유지:** 3번만이 "모르면 비워라"이고, 1·2번은 *이 문장 자체*의 의미 해석이다. 문맥·이전 상태를 참조하라는 것이 아니다. carry는 여전히 코드가 한다.
- **⚠️ 폐기된 초안:** v0.1 최초안의 *"발화하지 않으면 반드시 unspecified, 추론하지 말 것"* 문구는 **삭제**한다. 그것은 deepseek의 실패 해석을 그대로 명문화한 것이며, 실제 실패 모드(환각이 아니라 **과잉 침묵**)를 악화시킨다.
- **⚠️ scope 판별자 주장 금지:** "무대 리셋이 EISM의 핵심 조건"은 표본 부족으로 **철회(D-M13⑤)**. place를 특권화하지 말 것.

### 3.4 `causality`

- **동결 추출:** `event_type`(10종) · `aspect` ∈ {inceptive, ongoing, completive}.
- **코드 리셋(불변):** `event_type` 변함 OR `aspect`==inceptive.
- **EISM 근거·격차(정직하게):** EISM 인과는 사건 간 *인과 관계*(필요성+반사실; enablement/motivation/psychological)다. `event_type` **변화는 인과가 아니다** — "행동 종류가 바뀜"에 가깝다. **5축 중 이론 충실도 최저.** 진짜 인과 링크는 사건 비교가 필요(Indexter는 plan의 causal_link 사용)해서 **target-only에서 원리상 불가.**
- **🧪 실험 (실행 확정, dev-3 전용) — time축과 동형 절충:** `causal_marker` ∈ {absent, cause, consequence} — "이 문장이 명시적 인과 접속을 *자체 문면에* 담는가"(because/so/therefore/as a result/그래서/때문에). time의 `temporal_shift_marker`와 정확히 같은 target-only-안전 전략. §3.5 표에서 causality는 상시발화·저정보(Monkey 70%)이므로 더 날카로운 인과 신호가 도움될 여지. **리셋 규칙 후보:** 기존 OR `causal_marker` ∈ {cause, consequence}. dev-3에서만 검증, 이기면 동결·지면 폐기.
- **⚠️ 주의:** §4.1의 −log p 기저율 가중은 **기각 유지**(rare 축 증폭이 place 노이즈까지 증폭; ami −0.180). 인과 개선은 *신호 교체*로 하고 *가중 재도입*으로 하지 말 것.

### 3.5 `intentionality` (구 `goal`)

- **동결 추출:** `active_goal` ∈ Stage1 `goals` ∪ {`none`}.
- **코드(불변):** carry-forward(`none`이면 직전 유지) → 리셋 = carry 후 값 변함 AND 둘 다 ≠ none.
- **EISM 근거:** Indexter — 두 사건이 *같은 frame of commitment*(동일 목표 달성 계획)에 속하면 공유. 현 슬롯은 "어떤 목표가 활성"이라 근사이나 계획-소속 그 자체는 아님.
- **✅ null 처리 (space와 동일 원리):** ① 목표 **단어 찾기가 아니라** 이 문장의 행위가 *무엇을 위한 것인지*를 고른다. ② 단순히 화제로 **언급된** 목표는 고르지 않는다. ③ 판단 근거가 이 문장 안에 전혀 없을 때만 `none`. — §3.3의 정정과 같은 이유로, "발화 안 하면 무조건 none"식 문자적 지시는 쓰지 않는다.
- **📉 goal은 3편 공통 최저(F1 0.38, D-M13⑥).** 라벨러 교체 검증에서도 가장 약한 축으로 확인됐다 → 재정의 우선순위 1순위.
- **🔴 "goal 축 재정의" (README) — 선행 진단 필수:** README가 이 축을 재정의 대상으로, **goal divergence 진단을 선행조건**으로 지정. 진단 없이 재정의를 동결하지 말 것. 진단이 답해야 할 것:
  1. target-only에서 `active_goal`이 얼마나 자주 식별 가능한가(place의 86% unspecified 유사 문제 여부).
  2. 저비용 라벨러와 fable-5 간 `active_goal` 라벨 발산(divergence)이 얼마나 큰가.
  3. goal은 제거하면 dev가 크게 상함(−goal: pieman 0.598→0.546, dialseg 0.634→0.477) → **신호는 실재하나 취약.** 재정의는 신호를 *죽이지 않으면서* 안정화해야 함.
- **재정의 후보(진단 후 택1, dev-3 동결):** (a) 현행 유지 + null 명시화만; (b) `active_goal`을 "이 문장이 추구/진전시키는 Stage1 목표"로 좁히고 진전 없으면 `none`; (c) Indexter 정합형 — Stage1 `goals`를 "누구의 어떤 목표"로 뽑아 frame 근사. **b/c는 진단 결과가 지지할 때만.**

---

## 4. 재설계 프롬프트 (드롭인 — 변경 시 G5로 재동결 필수)

### 4.1 Stage 1 (이야기당 1회, 경계어 금지 — G1)

기존과 동일 골격. `goals`만 재정의 후보에 맞춰 조정 가능(진단 후). 경계/세그먼트/변화 단어 **불포함** 유지.

```
Below is a complete narrative. List the entities it contains, as normalized
snake_case identifiers. Do not describe events, order, or structure.

- cast: every distinct person/character that appears anywhere.
        Use `narrator` for the first-person teller.
- protagonist: the single central character the narrative is primarily about
        (one id from cast). EISM designates exactly one protagonist per story.
- locations: every distinct physical setting that appears.
- goals: every distinct goal/intention pursued by anyone.

Return only these lists. Do not mention boundaries, segments, changes, or scenes.

NARRATIVE:
{story}
```
> `protagonist` 단일 id 추가는 §3.1 실험 후보(`protagonist_present`)를 위한 것. boolean 실험을 안 하면 이 줄은 빼서 기존 sha 유지.

### 4.2 Stage 2 필드 문구 (축별, target-only — G2)

공통 꼬리말(불변):
```
Items are shuffled and unrelated. Never compare items to each other. Do not reason
about documents, stories, transitions, order, continuity, novelty, changes, resets,
or boundaries. Judge each TARGET sentence on its own — interpret what this sentence
depicts, do not merely string-match for keywords. Use the neutral token
(none/unspecified/absent) only when this sentence gives no basis at all. Never carry a
value over from another item. Return exactly one result per item_id.
```

| 축 | 필드 문구(핵심) |
|---|---|
| protagonist | `perspective`: 이 문장 중심 행위자/경험자의 문법 인칭(first/second/third), 인물 없으면 none. **(실험)** `protagonist_present`: 이 문장의 주어/경험자가 지정 주인공 `{protagonist}`이면 true, 아니면 false. |
| time | `temporal_shift_marker`: 이 문장이 다음-단계 접속(then/next/after that)으로 열리면 sequence; 시간점프(later/the next morning/years later) simultaneity(meanwhile) 표지면 각각; 아니면 absent. **이 문장 자체 문면만.** (`tense`도 뽑되 리셋 미사용.) |
| space | `location`: 이 문장이 **그리는 장면의 무대**. 장소 단어를 찾는 것이 아니다 — 단어가 없어도 무대가 드러나면 고른다. 화제로 **언급만** 된 지명은 고르지 않는다. 근거가 이 문장에 전혀 없을 때만 `unspecified`. 목록 `[{locs}, unspecified]`에서만. |
| causality | `event_type`: 이 문장이 표현하는 사건/행동의 종류. `aspect`: 새 행동/상태 시작=inceptive, 종료=completive, 그 외 ongoing. **(실험)** `causal_marker`: 이 문장이 원인 접속(because/때문에)을 담으면 cause, 결과 접속(so/therefore/그래서)이면 consequence, 아니면 absent. |
| intentionality | `active_goal`: 이 문장의 행위가 **무엇을 위한 것인지**. 화제로 **언급만** 된 목표는 고르지 않는다. 근거가 이 문장에 전혀 없을 때만 `none`. 목록 `[{goals}, none]`에서만. |

> 실험 필드(`protagonist_present`, `causal_marker`)는 **additive 로깅용**. 동결 count에는 안 들어가고, dev-3에서 별도 컬럼으로 승패만 본다(§6).

---

## 5. 재설계로도 바뀌지 않는 동결 사항 (충실화 명목의 과잉수정 방지)

| 항목 | 상태 | 근거 |
|---|---|---|
| target-only, 사건 비교 금지 | **유지** | Indexter 충실성의 하드웨어(§1). 완화 시 S2 carryover 재발 |
| 변화/경계/리셋/이전 질의 금지 | **유지** | S1 halo(AUC 1.0) |
| 닫힌 목록·코드 판정 | **유지** | S3·G4 |
| tense 리셋 제외 | **유지** | 결함 수정 + EISM 충실(§3.2) |
| charset count 제외(perspective 단독) | **유지** | D-B7/B8. charset은 비-EISM적 노이즈 |
| −log p 기저율 가중 | **기각 유지** | §4.1(ami −0.180) |
| t\* = 3, count ≥ t\* 규칙 | **유지**(방법별 자연규칙, D-M3) | 저해상도 정수 count에 μ+cσ 부적합 |
| Score = 0.5·F1+0.25(1−Pk)+0.25(1−WD), 인코더 mpnet frozen | **유지** | §6 |

> **알려진 결함(불변):** t\*=3은 문서 길이 무관 적용이라 긴 문서 과예측. Phase 0의 구조적 한계이며 **M5 sticky-CRP(κ)가 원리적으로 해결**. 재설계로 고치려 하지 말 것 — 기록만.

---

## 6. 동결·평가 프로토콜 (오염 재발 방지 — G5·G6)

1. **선행 완료 확인:** goal divergence 진단(§3.5) + **라벨러 Gate C 재검증(§6.2 — Q1 deepseek)**.

2. **dev-3(pieman/tunnel/monkey)에서만 튜닝.** 실험 필드(§3.1·3.4) 승패, null-명시화 효과, 재정의 후보를 dev-3에서 결정. **held-out은 열지 않는다.**
3. **동결:** 최종 프롬프트·enum·리셋·t\*·축 키를 확정하고 `prompt_sha256` 기록. **이후 무수정.**
4. **오염 스팟체크(§8) 통과.**
5. **봉인된 held-out에서 1회만 평가.** 현 held-out = **Tier1 6편(신규 *Born a Crime* 3편, 사람 20명 주석; D-M13②)**. — **⚠️ `tier2-heldout-spec.md`는 폐기(D-M13①): Gutenberg+GPT-4 silver 제외.** 따라서 그 문서의 GPT-4 gold·교차모델(GPT-4 vs Claude) 순환 규칙은 **더 이상 구속하지 않는다**(Tier1 gold는 사람 주석이라 LLM 공유편향 순환이 성립 안 함). tier2를 이 재설계의 평가 대상으로 삼지 말 것.
6. **doc-level bootstrap CI.** held-out 편수가 다수여야 CI 가능(현 fiction held-out doc 1개 문제의 해소가 존재 이유).
7. 전 과정·수치를 ledger에 **append-only** 기록(D-B9 가칭).

> **🔴 절대 규칙:** held-out에서 성능이 낮게 나와도 dev-3로 돌아가 재튜닝 후 다시 held-out을 열면 그 즉시 selection-on-test(D-M5) 재발. **held-out은 1회.**

### 6.2 🔴 라벨러 Gate C 재검증 (deepseek-v4-pro — Q1)

**성격(D-M13⑥):** 라벨러 교체는 구현 교체가 아니라 **측정도구 변경**이다. M3 경계함수가 라벨러 출력에 직접 의존한다. 따라서 프롬프트 재설계와 **동시에** 검증한다.

| 항목 | 값 |
|---|---|
| 대상 | dev 3편(pieman/tunnel/monkey) |
| 변형 | Stage 1+2 전부 재생성, 신규 프롬프트·동결 설정·t\*=3 |
| **판정** | **Gate C — 사람 gold 대비 ΔScore ≥ −0.03, 3편 모두** |
| 참조 실패치 | 옛 프롬프트: pieman −0.253 / tunnel −0.151 / monkey −0.052 |
| 기술 조건 | `--stage1-max-tokens` 상향(2000이면 monkey 27,545자에서 `content is None`) |

- **통과 → 채택.** 재설계가 해석 차이를 제거했다는 증거이므로, "저비용 라벨러 확보"라는 재설계 목적이 달성된다.
- **실패 → 진짜 기각.** 그때는 원인이 해석이 아니라 능력이라는 뜻이므로 다른 모델을 찾는다.
- **⚠️ place 단독 통계로 판정 금지(D-M13⑥).** 옛 실행에서 pieman place 리셋 F1 0.889·false-move 0이었는데 최종 Score는 최악이었다. **판정은 5축 전체 최종 Score로만.**
- **⚠️ 프롬프트를 Gate C 통과할 때까지 반복 수정하는 것은 dev 튜닝이며 허용**된다(시험지는 안 열림). 단 **회차·수정 내역을 전부 ledger에 기록**한다. 기록 없는 반복은 은닉된 selection이다.

### 6.3 실험 2종을 모두 돌릴 때의 오염 방지 (Q3·Q4 실행 확정에 따름)

실험을 늘리면 **변종 수가 늘어 D-M5 구조로 되돌아갈 위험**이 커진다. 다음을 강제한다.

1. **채택 기준을 실행 전에 문서화한다.** 예: "dev-3 평균 Score가 기존 대비 +0.02 이상이고, 3편 중 2편 이상에서 하락하지 않을 것." 숫자를 본 뒤 기준을 만들지 않는다.
2. **비교는 dev-3에서만.** 7개 데이터 전체 스윕 금지(그것이 D-M5의 경로다).
3. **축별로 독립 판정한다.** `protagonist_present`는 protagonist 축 안에서 perspective와, `causal_marker`는 causality 축 안에서만 겨룬다. 2×2 조합 전수 탐색 금지 — 조합 수가 곧 오염 자유도다.
4. **패자는 즉시 폐기하고 기록한다.** "나중에 다시 볼 후보"로 남기면 held-out 결과를 본 뒤 부활시키고 싶어진다.
5. **동결 count에는 승자만 편입**되고, 실험 필드는 그때까지 별도 컬럼으로만 존재한다.

---

## 7. 동결 전 확인 필요 (미해결 결정 — 사용자 판단 필요)

| # | 결정 | 선택지 | 비고 |
|---|---|---|---|
| Q1 | 저비용 라벨러 모델 | **deepseek-v4-pro 재검증**(사용자 지정) | D-M13⑥ 기각은 **옛 프롬프트 하 판정**이고 원인이 *해석 방식*이므로 재설계가 직격한다. §6.2 Gate C 재통과 시 채택, 실패 시 진짜 기각. `--stage1-max-tokens` 상향 필수 |
| ~~Q2~~ | ~~명명~~ | ✅ **해결 — 하드 리네임 확정** | §2. 라벨러 모델 교체로 전량 재추출 예정이라 캐시 보존 유인 소멸 |
| ~~Q3~~ | ~~protagonist_present 실험~~ | ✅ **실행 확정**(사용자 지정) | §3.1. Stage1에 `protagonist` 단일 id 추가 |
| ~~Q4~~ | ~~causal_marker 실험~~ | ✅ **실행 확정**(사용자 지정) | §3.4. 인과 충실도 최저 축 |
| Q5 | goal 재정의 후보 | a/b/c | §3.5. **goal divergence 진단 후 결정** |

---

## 8. 오염 검증 스팟체크 (재설계 후 필수 — annotation-guide §5 확장)

1. Stage1·2 출력에 목록 밖 값·자유서술·"변화/경계/이전" 흔적이 없는가?
2. Stage2 문장 순서를 바꿔 재추출 시 동일 결과인가?(target 단독 → 순서 무관)
3. Stage1 캐스트/목록이 경계 정보를 흘리지 않았는가?
4. carry-forward가 **코드에서만** 수행되고, null-명시화 문구가 *이전 상태*를 프롬프트에 넣지 않았는가?(unspecified/none은 현재 문장 판단일 뿐)
5. **(신규)** 실험 필드가 동결 count에 유입되지 않고 별도 로깅되는가?
6. **(신규)** 구 키(`place`/`goal`)가 코드·산출물·문서 어디에도 남아 있지 않은가?(§2 완료 체크리스트) 구 캐시가 삭제되었는가?
7. **(신규)** 프롬프트 변경 후 `prompt_sha256`가 갱신·기록되고, 이전 동결 수치를 새 프롬프트에 소급 인용하지 않는가?

---

## 부록 A. 코드 변경 지점 (`scripts/eacl_phase0/llm_s5_states.py`)

| 위치 | 변경 | 가드레일 |
|---|---|---|
| `STAGE1_PROMPT` | (선택) `protagonist` 단일 id 라인 추가 | Q3 실행 시만. sha 변경됨 |
| `STAGE1_SCHEMA` | (선택) `protagonist` 키 추가 | 위와 동일 |
| `stage2_prompt` 각 축 `field` | null-명시화 문구(§4.2), 실험 필드 문구 | G2 유지: 이전 상태 미주입 |
| `stage2_schema` | 실험 필드 enum 추가(additive) | strict schema 유지 |
| `AXIS_FIELDS`/`valid` | 실험 필드 검증 추가 | 동결 count엔 미포함 |
| `reset_*` 함수 | **불변**(실험 리셋은 별도 함수로 분리 로깅) | G4 |
| `AXES`·`AXIS_FIELDS`·`reset_*`·`stage2_*`·`main()` | `place`→`space`, `goal`→`intentionality` 전량 치환 | §2. 구 캐시 삭제 후 재추출 |

> 모든 프롬프트 문자열 변경은 `VERSION` 갱신 + `prompt_sha256` 재기록을 동반한다(재현성).
