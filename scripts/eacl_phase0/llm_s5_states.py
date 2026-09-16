#!/usr/bin/env python3
"""P0-S5: event-boundary (B) measurement — story-level closed lists + target-only match.

EISM redesign (v2, 2026-07-21, spec: EACL/s5-eism-redesign-guide.md):
  Axis keys aligned to EISM/Indexter vocabulary — hard rename place→space, goal→intentionality
  (§2). Field names (`location`, `active_goal`) are axis-internal and unchanged. null-explicitation
  wording for the slow variables (§3.3, §3.5) replaces the earlier literal "if not uttered, blank"
  instruction (which encoded deepseek's over-silence failure mode). Two additive experiment fields
  are logged for dev-3 head-to-head (§3.1, §3.4) but NEVER enter the frozen count:
    - `protagonist_present` (protagonist axis): is this sentence's subject/experiencer the single
      designated protagonist? (vs perspective)
    - `causal_marker` (causality axis): explicit causal connective in THIS sentence's own wording.

Two stages (EntNet tied-keys trick):
  Stage 1 (per story, 1 call, NO boundary/segment words): LLM reads the whole story and enumerates
    cast / protagonist(single id) / locations / goals as normalized closed lists.
  Stage 2 (per sentence, target-only, per-axis separate calls): LLM picks ONLY from the Stage-1
    lists (+ fixed absolute enums). Never compares sentences, never sees boundary/change/reset/prev.

Change/reset judgment is 100% code (below), with boundary-agnostic carry-forward for the slow
variables space/intentionality. All enums/rules/thresholds frozen before seeing gold (G5).

Inherited measurement hygiene (S1~S4): no halo (never ask change), no carryover (target-only),
no free-form (closed lists), code-only change.

D-B8 → **D-B10 (2026-07-22) 로 대체**: 동결 count 의 protagonist 성분은 `protagonist_present`
(`perspective` 대체, `build_anchor_dataset.py` 가 쓰는 조합). `characters_present`·`perspective` 는
ablation 컬럼으로만 남는다. `reset_matrix.jsonl` 은 세 컬럼을 모두 기록하므로 이 결정은 이 스크립트가
아니라 소비 측에서 적용된다.

동결 앵커 = `--prompt-version v3.1` (D-B14) · 라벨러 Qwen3.6-35B-A3B no-thinking (D-M13⑥).
논문 appendix 용 프롬프트 전문은 `--dump-prompts [--dump-format latex]` 로 실제 호출 경로에서
직접 렌더링한다(손으로 옮겨 적지 말 것 — 문구의 유일한 출처는 아래 빌더들이다).
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, time
from pathlib import Path
from dotenv import load_dotenv

VERSION = "p0s5-eism-v2"
# --prompt-version tag. v3 = D-B12 redesign(ALL fixes). v3.1 = D-B14 동결(H+E만, 2026-07-25).
VERSIONS = {"v2": "p0s5-eism-v2", "v3": "p0s5-eism-v3", "v3.1": "p0s5-eism-v3.1"}
AXES = ("protagonist", "space", "time", "causality", "intentionality")
# D-B22 실험 축 — 동결 5축과 완전 분리. AXES 에 넣지 않으므로 reset_matrix·prompt_sha256 불변.
# `--axis chainrole` 로만 추출되고 `chainrole_states.jsonl` 에만 쓰인다.
EXPERIMENT_AXES = ("chainrole",)

# --- fixed absolute enums (story-independent) ---
PERSPECTIVE = ["first", "second", "third", "none"]
TENSE = ["past", "present", "future", "none"]
TEMPORAL_SHIFT = ["absent", "sequence", "jump", "simultaneity"]
EVENT_TYPE = ["dialogue", "physical_action", "movement_or_travel",
              "perception_or_observation", "thought_or_emotion", "transaction_or_exchange",
              "conflict_or_violence", "work_or_creation", "state_description", "other"]
ASPECT = ["inceptive", "ongoing", "completive"]
CAUSAL_MARKER = ["absent", "cause", "consequence"]  # additive experiment field (§3.4)
# D-B22: 의도 사슬 역할. D(commitment)/A(execution)/C(attainment) 3비트의 8조합.
# `unclear` 없음(도피 범주 금지) · `outside_chain` 유지(음성 범주).
CHAIN_ROLE = ["outside_chain", "commitment", "execution", "attainment",
              "commitment_execution", "commitment_attainment", "execution_attainment",
              "commitment_execution_attainment"]

JACCARD_RESET = 0.5  # protagonist active-set reset threshold (frozen, ablation column)


def read(p: Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write(p: Path, rows: list[dict]) -> None:
    Path(p).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def sequence(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda x: int(x["transition"]))
    return [rows[0]["previous"], *[x["current"] for x in rows]]


# ---------------------------------------------------------------- Stage 1
STAGE1_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "cast": {"type": "array", "items": {"type": "string"}},
        "protagonist": {"type": "string"},
        "locations": {"type": "array", "items": {"type": "string"}},
        "goals": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cast", "protagonist", "locations", "goals"],
}

STAGE1_PROMPT = """Below is a complete narrative. List the entities it contains, as normalized
snake_case identifiers. Do not describe events, order, or structure.

- cast: every distinct person/character that appears anywhere in the narrative
  (e.g. narrator, roommate, store_clerk, dean). Use `narrator` for the first-person teller.
- protagonist: the single central character the narrative is primarily about (one id from cast).
  EISM designates exactly one protagonist per story.
- locations: every distinct physical setting that appears (e.g. bedroom, convenience_store, campus).
- goals: every distinct goal/intention pursued by anyone (e.g. get_the_story, avoid_getting_caught).

Return only these lists. Do not mention boundaries, segments, changes, or scenes.

NARRATIVE:
{story}"""

# v3 (D-B12, 2026-07-23): A = locations 정의에 vehicle/interior/airspace 무대 + 한 수준 고정.
# B② = goals 를 bounded scene-level commitment 로 제한, 상시/기질/도덕 목표 제외(상시목표가 label space
# 에 안 들어오면 carry/reset 에서 자연 탈락 — reset 코드 무변경). cast/protagonist 문구는 v2와 동일.
STAGE1_PROMPT_V3 = """Below is a complete narrative. List the entities it contains, as normalized
snake_case identifiers. Do not describe events, order, or structure.

- cast: every distinct person/character that appears anywhere in the narrative
  (e.g. narrator, roommate, store_clerk, dean). Use `narrator` for the first-person teller.
- protagonist: the single central character the narrative is primarily about (one id from cast).
  EISM designates exactly one protagonist per story.
- locations: every distinct physical setting where characters are located or act — the spatial
  stages of the narrative. Include vehicle / interior / airspace stages when characters occupy them
  (e.g. cockpit, ship_deck, carriage_interior, train_compartment), not only named ground places.
  Keep one level per stage: while a character is aboard a conveyance use its interior (cockpit,
  cabin); use an airspace / open-water stage (e.g. open_sky) only for travel described from outside it.
  (e.g. bedroom, convenience_store, campus, cockpit, ship_deck)
- goals: every distinct bounded, scene-level objective someone actively pursues — a commitment that
  can start, progress, and be completed or abandoned (e.g. get_the_story, escape_the_room,
  deliver_the_message). EXCLUDE standing dispositions, character traits, general or lifelong desires,
  and moral principles — these are not events (e.g. NOT stay_calm, NOT be_brave, NOT avoid_opposition).
  A diffuse or open-ended wish counts only if the narrative shows concrete steps taken toward it.

Return only these lists. Do not mention boundaries, segments, changes, or scenes.

NARRATIVE:
{story}"""

# ---------- modular per-fix decomposition (D-B12-fine, 2026-07-24) ----------
# 개별 fix 를 v2 에 하나씩 얹어 기여 분해. 무-fix = v2 문구 정확 재현. v3 = ALL fixes.
V3_FIXES = frozenset({"space_A", "space_H", "intent_2", "intent_1",
                      "time_E", "time_conseq", "time_first"})


# v3.1 (D-B14, 2026-07-25) = **H + E 만**. space_A 제외(빈도 조건 미충족·obs_77176 단독 현상),
# intent_1/intent_2 제외(②진단서 B① 기각·B② 타깃 악화 3→5). dev-3 판정: H 타깃 7.1%→5.0% ·
# E 타깃(repcorpus) 41.2%→22.5% · 평균 ΔScore +0.0146(안전장치 ≥−0.09). 근거 = D-B14.
V31_FIXES = frozenset({"space_H", "time_E", "time_conseq", "time_first"})


def fixes_of(pv: str, fixes_str: str | None) -> frozenset:
    if fixes_str:
        return frozenset(x.strip() for x in fixes_str.split(",") if x.strip())
    if pv == "v3":
        return V3_FIXES
    if pv == "v3.1":
        return V31_FIXES
    return frozenset()


_S1_HEAD = ("Below is a complete narrative. List the entities it contains, as normalized\n"
            "snake_case identifiers. Do not describe events, order, or structure.\n\n"
            "- cast: every distinct person/character that appears anywhere in the narrative\n"
            "  (e.g. narrator, roommate, store_clerk, dean). Use `narrator` for the first-person teller.\n"
            "- protagonist: the single central character the narrative is primarily about (one id from cast).\n"
            "  EISM designates exactly one protagonist per story.\n")
_S1_LOC_V2 = "- locations: every distinct physical setting that appears (e.g. bedroom, convenience_store, campus).\n"
_S1_LOC_A = ("- locations: every distinct physical setting where characters are located or act — the spatial\n"
             "  stages of the narrative. Include vehicle / interior / airspace stages when characters occupy them\n"
             "  (e.g. cockpit, ship_deck, carriage_interior, train_compartment), not only named ground places.\n"
             "  Keep one level per stage: while a character is aboard a conveyance use its interior (cockpit,\n"
             "  cabin); use an airspace / open-water stage (e.g. open_sky) only for travel described from outside it.\n"
             "  (e.g. bedroom, convenience_store, campus, cockpit, ship_deck)\n")
# space_Agen (2026-07-24): 일반화 A — "목록을 제대로". 모든 유형 실제 무대 포함 + 언급-only 제외.
# 좁은 space_A(vehicle 특화)의 상위. post_office 같은 mention-only over-enumeration 을 Stage-1 에서 차단.
_S1_LOC_AGEN = ("- locations: every distinct physical setting where the narrated action actually takes place —\n"
                "  the spatial stages of the narrative, of ANY kind: rooms and outdoor places, and also\n"
                "  vehicle / interior / airspace / vessel stages when characters occupy them (e.g. cockpit,\n"
                "  ship_deck, carriage_interior). Do NOT list a place that is only mentioned, referred to, or\n"
                "  recalled but is never itself a scene of action (e.g. a shop someone is merely said to run,\n"
                "  a home only spoken about). Keep one level per stage: while aboard a conveyance use its\n"
                "  interior; use an airspace / open-water stage only for travel described from outside it.\n")
_S1_GOAL_V2 = "- goals: every distinct goal/intention pursued by anyone (e.g. get_the_story, avoid_getting_caught).\n"
_S1_GOAL_2 = ("- goals: every distinct bounded, scene-level objective someone actively pursues — a commitment that\n"
              "  can start, progress, and be completed or abandoned (e.g. get_the_story, escape_the_room,\n"
              "  deliver_the_message). EXCLUDE standing dispositions, character traits, general or lifelong desires,\n"
              "  and moral principles — these are not events (e.g. NOT stay_calm, NOT be_brave, NOT avoid_opposition).\n"
              "  A diffuse or open-ended wish counts only if the narrative shows concrete steps taken toward it.\n")
_S1_TAIL = "\nReturn only these lists. Do not mention boundaries, segments, changes, or scenes.\n\nNARRATIVE:\n{story}"


def build_stage1(fixes: frozenset, story: str) -> str:
    loc = (_S1_LOC_AGEN if "space_Agen" in fixes else
           _S1_LOC_A if "space_A" in fixes else _S1_LOC_V2)
    goal = _S1_GOAL_2 if "intent_2" in fixes else _S1_GOAL_V2
    return _S1_HEAD + loc + goal + _S1_TAIL.format(story=story)


def build_space_field(locs: str, fixes: frozenset) -> str:
    veh = (" (including a vehicle / interior / airspace stage from the list when the action happens there)"
           if "space_A" in fixes else "")
    head = (f"location: the STAGE this sentence depicts — where the narrated action is set{veh}. You are "
            "NOT looking for a place word: pick the stage even if no place noun appears (e.g. "
            "'he tossed in bed' -> bedroom). A place that is only MENTIONED — talked about or "
            "recalled — is NOT the stage")
    if "space_Hrelax" in fixes:  # 완화 H: situated action 없는 문장만 unspec, 암묵장면은 보존
        tail = (". Use unspecified when THIS sentence depicts no situated action of its own — a bare "
                "emotion, sensation, inner thought, reflection, or an isolated body-part description does "
                "not place the action anywhere -> unspecified. When the sentence DOES depict a situated "
                "action, name the stage that action occupies (even if the place is clear only from the "
                "current scene). Do not attach a location merely because an earlier sentence was set there. ")
    elif "space_H" in fixes:
        tail = (". Judge ONLY this sentence's own content: do NOT infer the stage from neighboring "
                "sentences, earlier context, or where the character 'must' be. If this sentence alone "
                "gives no stage (a bare emotion, an isolated reflection, a body part with no setting), "
                "you MUST answer unspecified. ")
    else:
        tail = (" (e.g. 'they spoke of home in England' -> the stage is not England). Use unspecified only "
                "when THIS sentence gives no basis for a stage at all. ")
    return head + tail + f"Choose only from: [{locs}, unspecified]."


def build_intent_field(goals: str, fixes: frozenset) -> str:
    speech = ("Speech is also action: a command, request, promise, plan, or purposeful question can serve a "
              "goal. " if "intent_1" in fixes else "")
    ex = (" (e.g. 'they talked about whether escape was possible' -> none, only discussed; 'Meet me at "
          "midnight so we can escape' -> escape, the speech serves it; 'he sawed through the fence bars' "
          "-> escape)" if "intent_1" in fixes else
          " (e.g. 'they talked about escaping the compound' -> none, the goal is only discussed; 'he sawed "
          "through the fence bars' -> escape, the action serves it)")
    return ("active_goal: the Stage 1 goal that THIS sentence's action works toward. Read the purpose "
            "from what the action does — the goal's own words need not appear. " + speech +
            "A goal that is only MENTIONED — talked about as a topic — is NOT chosen. Use none only when "
            "the action serves no goal on the list." + ex + f". Choose only from: [{goals}, none].")


def build_time_field(fixes: frozenset) -> str:
    seq = "then, next, after that" + (", first, afterwards" if "time_first" in fixes else "")
    tense = "tense: dominant grammatical tense of THIS sentence (past/present/future); none if unclear.\n"
    if "time_E" in fixes:
        body = ("temporal_shift_marker: judge a shift ONLY if the marker OPENS this sentence (a leading "
                f"coordinating conjunction or quote is allowed). 'sequence' if it opens with a next-step "
                f"connective ({seq}); 'jump' if it opens with a time jump (later, the next morning, years "
                "later, a moment later); 'simultaneity' if it opens with meanwhile / at the same time. A "
                "time word that appears mid-sentence ('he walked home and later ate') is NOT a shift -> "
                "absent.")
    else:
        body = (f"temporal_shift_marker: 'sequence' if THIS sentence opens with a next-step connective "
                f"({seq}); 'jump' if it marks a time jump (later, the next morning, years later); "
                "'simultaneity' (meanwhile, at the same time); otherwise 'absent'.")
    conseq = (" Do NOT treat consequence/contrast connectives (so, therefore, and yet, but) as time markers."
              if "time_conseq" in fixes else "")
    return tense + body + conseq + " Judge only THIS sentence's own wording."


SYSTEM_STAGE1 = ("You enumerate the entities present in a text as normalized identifiers. "
                 "You never discuss structure, order, or boundaries.")
SYSTEM_STAGE2 = ("You assign fixed categorical labels to one sentence at a time from a closed "
                 "list. You never compare sentences and never output free text.")


def stage1(c, model: str, story: str, max_tokens: int = 8000, extra_body: dict | None = None,
           pv: str = "v2", fixes: frozenset | None = None):
    """Stage 1 — one whole-story call producing the closed label space.

    `max_tokens` default 8000 for reasoning-style models: deepseek-v4-pro returns
    `message.content is None` on monkey (27,545 chars, the longest dev story) when reasoning
    tokens exhaust a small budget. `extra_body` (e.g. {"reasoning": {"enabled": False}}) is passed
    through for the --no-reasoning determinism experiment (D-B11 후속, 2026-07-23).
    """
    fx = fixes if fixes is not None else fixes_of(pv, None)
    prompt = build_stage1(fx, story)
    for a in range(3):
        try:
            r = c.chat.completions.create(
                model=model, temperature=0, max_tokens=max_tokens,
                response_format={"type": "json_schema", "json_schema":
                                 {"name": "story_entities", "strict": True, "schema": STAGE1_SCHEMA}},
                messages=[{"role": "system", "content": SYSTEM_STAGE1},
                          {"role": "user", "content": prompt}],
                **({"extra_body": extra_body} if extra_body else {}))
            out = json.loads(r.choices[0].message.content)
            return out, prompt
        except Exception:
            if a == 2:
                raise
            time.sleep(2 * (a + 1))


# ---------------------------------------------------------------- Stage 2
def stage2_schema(axis: str, lists: dict) -> dict:
    cast = sorted(set(lists["cast"]))
    locs = sorted(set(lists["locations"])) + ["unspecified"]
    goals = sorted(set(lists["goals"])) + ["none"]
    props: dict = {"item_id": {"type": "integer"}}
    if axis == "protagonist":
        # cast 가 비면 valid() 는 빈 배열만 통과시킨다(442행). 예전 스키마는 자유 문자열을
        # 허용해 모델이 "narrator" 를 적으면 배치 전체가 무효 → no valid outputs 로 사망했다
        # (2026-07-27 pg26644_002·pg72574_024). 스키마를 검증기에 맞춰 빈 배열만 허용한다.
        # cast 가 있는 문서에서는 분기가 바뀌지 않으므로 기존 라벨에 영향 없음.
        props["characters_present"] = ({"type": "array", "items": {"type": "string", "enum": cast}}
                                       if cast else
                                       {"type": "array", "items": {"type": "string"}, "maxItems": 0})
        props["perspective"] = {"type": "string", "enum": PERSPECTIVE}
        props["protagonist_present"] = {"type": "boolean"}  # experiment (§3.1)
    elif axis == "space":
        props["location"] = {"type": "string", "enum": locs}
    elif axis == "time":
        props["tense"] = {"type": "string", "enum": TENSE}
        props["temporal_shift_marker"] = {"type": "string", "enum": TEMPORAL_SHIFT}
    elif axis == "causality":
        props["event_type"] = {"type": "string", "enum": EVENT_TYPE}
        props["aspect"] = {"type": "string", "enum": ASPECT}
        props["causal_marker"] = {"type": "string", "enum": CAUSAL_MARKER}  # experiment (§3.4)
    elif axis == "intentionality":
        props["active_goal"] = {"type": "string", "enum": goals}
    elif axis == "chainrole":                                   # D-B22 (실험 축)
        props["intentional_chain_role"] = {"type": "string", "enum": CHAIN_ROLE}
    item = {"type": "object", "additionalProperties": False,
            "properties": props, "required": list(props)}
    return {"type": "object", "additionalProperties": False,
            "properties": {"items": {"type": "array", "items": item}}, "required": ["items"]}


# 축 공통 envelope. 축마다 바뀌는 것은 `{axis}` 이름과 `{field}` 블록뿐 — 나머지는 전 축 동일.
STAGE2_TEMPLATE = """Label the {axis} state expressed by each TARGET sentence, independently.
Choose exactly one value per field from its fixed list (arrays may be empty).

Fields:
{field}

{tail}

{payload}"""

# common tail — target-only, no comparison, null only when no basis (§4.2)
STAGE2_TAIL = (
    "Items are shuffled and unrelated. Never compare items to each other. Do not reason about "
    "documents, stories, transitions, order, continuity, novelty, changes, resets, or boundaries. "
    "Judge each TARGET sentence on its own — interpret what this sentence depicts, do not merely "
    "string-match for keywords. Use the neutral token (none/unspecified/absent) only when this "
    "sentence gives no basis at all. Never carry a value over from another item. Return exactly one "
    "result per item_id.")


def stage2_prompt(axis: str, lists: dict, items: list[tuple[int, str]], pv: str = "v2",
                  fixes: frozenset | None = None) -> str:
    fx = fixes if fixes is not None else fixes_of(pv, None)
    cast = ", ".join(sorted(set(lists["cast"]))) or "(none)"
    prot = lists.get("protagonist") or "(unknown)"
    locs = ", ".join(sorted(set(lists["locations"]))) or "(none)"
    goals = ", ".join(sorted(set(lists["goals"]))) or "(none)"
    # protagonist·causality 는 어떤 fix 도 건드리지 않으므로 v2 문구 그대로 고정.
    # space/time/intentionality 는 아래 모듈러 빌더가 만든다(무-fix = v2 문구 정확 재현,
    # v3/v3.1 = 해당 fix 적용). ⚠️ 2026-07-24 이전에 여기 있던 v3 리터럴 사본(`field_v3`)과
    # space/time/intentionality 의 v2 사본은 빌더 도입 후 **호출되지 않는 죽은 코드**였고
    # 실제 전송 문구와도 미세하게 달라져 있었다(예: 'afterwards, first' 순서, 잔여 "Otherwise
    # 'absent'."). 인용 사고를 막기 위해 제거 — 실제 문구의 유일한 출처는 빌더다.
    # 전문 확인: `--dump-prompts --prompt-version v3.1`.
    field = {
        "protagonist": (
            f"characters_present: the subset of this cast that appears/acts in THIS sentence; "
            f"[] if none. Choose only from: [{cast}].\n"
            "perspective: grammatical person of the central actor/experiencer in THIS sentence "
            "(first/second/third); none if no person.\n"
            f"protagonist_present: true if the grammatical subject/experiencer of THIS sentence is "
            f"the designated protagonist `{prot}`, otherwise false."),
        "causality": (
            "event_type: the kind of event/action expressed in THIS sentence.\n"
            "aspect: 'inceptive' if THIS sentence begins a new action/state, 'completive' if it "
            "finishes one, else 'ongoing'. A linguistic property of THIS sentence.\n"
            "causal_marker: 'cause' if THIS sentence carries a cause connective (because, since, "
            "때문에), 'consequence' if a result connective (so, therefore, as a result, 그래서), "
            "else 'absent'. Judge only THIS sentence's own wording."),
    }
    if axis == "space":
        field = build_space_field(locs, fx)
    elif axis == "time":
        field = build_time_field(fx)
    elif axis == "intentionality":
        field = build_intent_field(goals, fx)
    elif axis == "chainrole":
        # D-B22 (실험 축) — codex 확정 문안. G1/G2/G3 준수: 이전상태·변화·경계·novelty 미질의,
        # target 문장 단독, 닫힌 목록. **Stage-1 goals 를 참조하지 않는다**(intentionality 중복 회피).
        field = (
            "intentional_chain_role: Classify the intentional-chain role expressed in THIS sentence.\n"
            "A commitment is a character's decision, choice, resolve, or intention, including one "
            "expressed through thought, emotion, speech, or indirect report.\n"
            "An execution is a purposeful act performed by a character.\n"
            "An attainment is an explicit statement that an intended act was carried out or its "
            "intended result was achieved. Do not treat mere stopping, ending, or a resulting "
            "condition as attainment unless the sentence presents the act or result as intended.\n"
            "A sentence may express more than one role. Choose exactly one value: outside_chain, "
            "commitment, execution, attainment, commitment_execution, commitment_attainment, "
            "execution_attainment, or commitment_execution_attainment.\n"
            "Use outside_chain only when THIS sentence expresses none of commitment, execution, "
            "or attainment.")
    else:
        field = field[axis]
    payload = "\n\n".join(f"ITEM {i}\nTARGET: {s}" for i, s in items)
    return STAGE2_TEMPLATE.format(axis=axis, field=field, tail=STAGE2_TAIL, payload=payload)


def stage2_call(c, model: str, axis: str, lists: dict, items: list[tuple[int, str]],
                extra_body: dict | None = None, pv: str = "v2", fixes: frozenset | None = None):
    prompt = stage2_prompt(axis, lists, items, pv, fixes)
    for a in range(6):  # origin (api.ssunlp) returns 504 under load; back off long, retry more
        try:
            r = c.chat.completions.create(
                model=model, temperature=0, max_tokens=10000,
                response_format={"type": "json_schema", "json_schema":
                                 {"name": f"{axis}_s5_states", "strict": True,
                                  "schema": stage2_schema(axis, lists)}},
                messages=[{"role": "system", "content": SYSTEM_STAGE2},
                          {"role": "user", "content": prompt}],
                **({"extra_body": extra_body} if extra_body else {}))
            return json.loads(r.choices[0].message.content)["items"], prompt
        except Exception:
            if a == 5:
                raise
            time.sleep(min(120, 20 * (a + 1)))  # 20,40,60,80,100s


AXIS_FIELDS = {
    "protagonist": ["characters_present", "perspective", "protagonist_present"],
    "space": ["location"],
    "time": ["tense", "temporal_shift_marker"],
    "causality": ["event_type", "aspect", "causal_marker"],
    "intentionality": ["active_goal"],
    "chainrole": ["intentional_chain_role"],                    # D-B22 (실험 축)
}


def valid(axis: str, x: dict, lists: dict) -> dict | None:
    vals = {}
    for f in AXIS_FIELDS[axis]:
        v = x.get(f)
        if f == "characters_present":
            if not isinstance(v, list) or any(cc not in set(lists["cast"]) for cc in v):
                return None
            vals[f] = sorted(set(v))
        elif f == "perspective" and v not in PERSPECTIVE:
            return None
        elif f == "protagonist_present" and not isinstance(v, bool):
            return None
        elif f == "location" and v not in set(lists["locations"]) | {"unspecified"}:
            return None
        elif f == "tense" and v not in TENSE:
            return None
        elif f == "temporal_shift_marker" and v not in TEMPORAL_SHIFT:
            return None
        elif f == "event_type" and v not in EVENT_TYPE:
            return None
        elif f == "aspect" and v not in ASPECT:
            return None
        elif f == "causal_marker" and v not in CAUSAL_MARKER:
            return None
        elif f == "active_goal" and v not in set(lists["goals"]) | {"none"}:
            return None
        elif f == "intentional_chain_role" and v not in CHAIN_ROLE:   # D-B22
            return None
        if f not in vals:
            vals[f] = v
    return vals


# ---------------------------------------------------------------- carry-forward + reset (code)
def carry_forward(states: dict[int, dict], field: str, null: str) -> dict[int, dict]:
    out, last = {}, None
    for i in sorted(states):
        s = dict(states[i])
        if s.get(field, null) == null:
            if last is not None:
                s[field] = last
        else:
            last = s[field]
        out[i] = s
    return out


def reset_protagonist_charset(a: dict, b: dict) -> int:
    sa, sb = set(a["characters_present"]), set(b["characters_present"])
    if not sa and not sb:
        return 0
    inter, union = sa & sb, sa | sb
    jacc = len(inter) / len(union) if union else 1.0
    return int(len(inter) == 0 or jacc < JACCARD_RESET)


def reset_perspective(a: dict, b: dict) -> int:
    return int(a["perspective"] != b["perspective"] and "none" not in (a["perspective"], b["perspective"]))


def reset_protagonist_present(a: dict, b: dict) -> int:
    # experiment (§3.1): reset = boolean flip. Logged separately, NOT in frozen count.
    return int(bool(a["protagonist_present"]) != bool(b["protagonist_present"]))

# §6 dominant-actor turnover (unified protagonist axis candidate) was tested 2026-07-22 and REJECTED
# for fiction (3-doc avg 0.401 < present 0.418, same extraction; < 0.437 frozen bar). Deferred to
# the LoCoMo/multi-speaker downstream, where perspective/present are dead and turnover is the only
# live protagonist signal. Extraction lives in *_s5_eismv2_domact/; scorer score_domact_turnover.py.
# NOT added to the frozen fiction prompt (bundling it degraded present 0.447->0.418).


def reset_space(a: dict, b: dict) -> int:  # after carry (was reset_place)
    return int(a["location"] != b["location"] and "unspecified" not in (a["location"], b["location"]))


def reset_time(a: dict, b: dict) -> int:
    # time reset = temporal-shift marker ONLY. tense-change 조건 제거(2026-07-13):
    # tense 는 서사 내내 서술↔논평↔대화로 문장마다 뒤집혀 grammatical noise 이고, time 리셋의
    # 87~91%가 tense-change 에서 왔으며 Score 를 깎았다(dev 0.598→0.634로 회복). jump/sequence/
    # simultaneity marker 만이 진짜 시간 불연속 신호. (a 는 시그니처 유지용, 미사용.)
    return int(b["temporal_shift_marker"] in ("sequence", "jump", "simultaneity"))


def _dac(role: str) -> tuple[bool, bool, bool]:
    """intentional_chain_role -> (D=commitment, A=execution, C=attainment) 비트."""
    return ("commitment" in role, "execution" in role, "attainment" in role)


def chainrole_resets(states: list[dict]) -> list[int]:
    """D-B22-A2 (승인된 단일 예외) — `outside_chain` 을 **투명한 ε-전이**로 두는 시퀀스 판정.

    인접 쌍만으로는 표현할 수 없다(`outside -> role` 판단에 outside **이전** 역할이 필요).
    따라서 문서 전체 상태열을 받아 전이별 리셋 [len-1] 을 돌려준다.

        역할 r -> outside : 전이 없음, r 보존
        outside -> outside: 전이 없음, 보존
        outside -> 역할 s : 보존된 r -> s 를 적용
        역할 r -> 역할 s  : r -> s 적용

    역할 전이 규칙(7x7, A1 에서 `outside` 행·열만 제거)은 그대로:
        reset = chain_entry(s) AND trigger(r)
        chain_entry(s) = s 에 commitment 또는 execution 비트
        trigger(r)     = r 에 attainment 비트   (outside 는 이제 trigger 가 아님)
    문서 첫 비-outside 이전 구간은 `last_chain_role` 이 없으므로 **리셋 없음**.
    길이 상한 N 은 두지 않는다(현 자료로 정하면 관측값 최적화 — codex 2026-07-28).

    `raw_role` 은 변경하지 않는다. 이 함수는 유효 상태만 계산한다.
    """
    out: list[int] = []
    last: str | None = None                       # 가장 최근 비-outside 역할
    for i in range(1, len(states)):
        cur = states[i]["intentional_chain_role"]
        prev = states[i - 1]["intentional_chain_role"]
        if prev != "outside_chain":
            last = prev
        if cur == "outside_chain" or last is None:
            out.append(0)                          # 전이 없음 / 사슬 미개시
            continue
        _, _, pc = _dac(last)
        cd, ca, _ = _dac(cur)
        out.append(int((cd or ca) and pc))
    return out


def reset_chainrole(a: dict, b: dict) -> int:
    """D-B22-A1 사전등록 규칙 (100% 코드, G4). 리셋 = **새 의도 사슬로의 진입**.

        chain_entry(cur) = cur 에 commitment 또는 execution 비트가 있음
        trigger(prev)    = prev 가 outside_chain 이거나 attainment 비트를 가짐
        reset = chain_entry(cur) AND trigger(prev)

    ① outside_chain -> commitment/execution 계열 진입 ② attainment 포함 상태 뒤 재시작.
    `attainment` 단독(현재)은 리셋이 아니며, 고립된 outside -> attainment 도 리셋이 아니다
    (문맥 밖 결과 보고일 가능성 — codex: 이론적 선택, 불확실성 있음).
    `event_type` 은 이 규칙에서 완전히 배제한다(D-B21 에서 포화 원인으로 확인).

    🔴 D-B22 원안(`cur.D OR (cur.A AND NOT prev.A) OR (prev.C AND cur.A)`)은 1차 스모크에서
    발화율 0.380 (기준 0.02~0.15) 으로 실패해 교체됐다. dev-3 미개봉 상태에서의 의미론적 수정이다.
    """
    pd, pa, pc = _dac(a["intentional_chain_role"])
    cd, ca, _ = _dac(b["intentional_chain_role"])
    chain_entry = cd or ca
    trigger = (not (pd or pa or pc)) or pc      # outside_chain 이거나 attainment 보유
    return int(chain_entry and trigger)


def reset_causality(a: dict, b: dict) -> int:
    return int(a["event_type"] != b["event_type"] or b["aspect"] == "inceptive")


def reset_causal_marker(a: dict, b: dict) -> int:
    # experiment (§3.4): explicit causal connective. Logged separately, NOT in frozen count.
    # Gate C S3 evaluates the candidate rule (reset_causality OR this) — this column is the signal.
    return int(b["causal_marker"] in ("cause", "consequence"))


def reset_intentionality(a: dict, b: dict) -> int:  # after carry (was reset_goal)
    return int(a["active_goal"] != b["active_goal"] and "none" not in (a["active_goal"], b["active_goal"]))


# ---------------------------------------------------------------- prompt dump (paper appendix)
# 논문 appendix 에 프롬프트 전문을 싣기 위한 렌더러. **프롬프트 문자열을 재작성하지 않고**
# 실제 호출 경로(build_stage1 / stage2_prompt)를 그대로 태워 뽑으므로, 여기 출력과 실제 전송
# 문자열이 어긋날 수 없다(=`prompt_sha256` 동결 불변). 스토리·문장·닫힌목록 자리는 아래 슬롯으로
# 대체된다 — 그 자리에 Stage-1 산출 목록이 쉼표로 이어붙어 들어간다.
SLOT_STORY = "{{STORY}}"                 # Stage-1: 문서 전문
SLOT_SENT = "{{SENTENCE_i}}"             # Stage-2: 한 TARGET 문장
SLOT_LISTS = {"cast": ["{{CAST}}"], "protagonist": "{{PROTAGONIST}}",
              "locations": ["{{LOCATIONS}}"], "goals": ["{{GOALS}}"]}


FIELD_VALUES = {  # `--dump-prompts` 의 [closed value sets] 표기. 실제 제약은 stage2_schema() 가 건다.
    "characters_present": "Stage-1 cast (any subset, possibly empty)",
    "perspective": PERSPECTIVE, "protagonist_present": ["true", "false"],
    "location": "Stage-1 locations + unspecified", "tense": TENSE,
    "temporal_shift_marker": TEMPORAL_SHIFT, "event_type": EVENT_TYPE, "aspect": ASPECT,
    "causal_marker": CAUSAL_MARKER, "active_goal": "Stage-1 goals + none",
    "intentional_chain_role": CHAIN_ROLE,
}


# latex 모드 전용 ASCII 폴딩. verbatim 안에서 pdfLaTeX 가 넘어지지 않게 하되, 무엇을 바꿨는지
# 반드시 헤더 주석으로 같이 찍는다(=조용한 변조 금지). 남은 비-ASCII 가 있으면 예외로 죽는다.
ASCII_FOLD = {"—": "--", "’": "'", "‘": "'", "“": '"', "”": '"',
              "때문에": "ttaemune", "그래서": "geuraeseo"}


def _fold_ascii(s: str) -> str:
    for k, v in ASCII_FOLD.items():
        s = s.replace(k, v)
    bad = sorted({ch for ch in s if ord(ch) > 127})
    if bad:
        raise ValueError(f"latex 덤프에 폴딩 규칙 없는 비-ASCII 문자: {bad} — ASCII_FOLD 에 추가하라")
    return s


# 한 줄로 이어지는 논리 단위의 시작. 이 패턴이 아닌 줄은 앞 단위에 이어 붙인다(소스 코드의
# 임의 줄바꿈 제거). 빈 줄은 문단 구분으로 보존한다.
_UNIT_START = re.compile(r"^(-\s|\[|\{\{|ITEM\b|[A-Za-z_]+:)")
_LABEL = re.compile(r"^\[[A-Za-z][A-Za-z ]*\]$")   # `[System]` / `[User]` / `[Slots]` — 단독 유지


def _pre_wrap(s: str, width: int) -> list[str]:
    """줄 구조를 그대로 두고 넘치는 줄만 접는다(JSON 스키마처럼 형태가 의미인 블록용)."""
    import textwrap
    out: list[str] = []
    for line in s.split("\n"):
        indent = " " * (len(line) - len(line.lstrip(" ")))
        out += textwrap.wrap(line, width=width, subsequent_indent=indent + "    ",
                             break_long_words=False, break_on_hyphens=False) or [""]
    return out


def _reflow_wrap(s: str, width: int) -> list[str]:
    """verbatim 용 재조판: 소스의 임의 줄바꿈을 없애고 `width` 열로 hard-wrap 한다.

    verbatim 에는 자동 줄바꿈이 없어 원문 그대로 두면 컬럼 밖으로 흘러넘친다. 바꾸는 것은
    **공백뿐** — 토큰 열은 실제 전송 문자열과 같다. 바이트 단위 원본이 필요하면 기본(text) 모드.
    """
    import textwrap
    units: list[str] = []
    for line in s.split("\n"):
        st = line.strip()
        if not st:
            units.append("")
        elif not units or units[-1] == "" or _LABEL.match(units[-1]) or _UNIT_START.match(st):
            units.append(st)
        else:
            units[-1] += " " + st
    out: list[str] = []
    for u in units:
        if not u:
            out.append("")
            continue
        hang = "    " if u.startswith("- ") else "  "
        out += textwrap.wrap(u, width=width, subsequent_indent=hang,
                             break_long_words=False, break_on_hyphens=False)
    return out


def _closed_values(axis: str) -> str:
    """축이 고르는 값 집합을 사람이 읽는 한 줄씩으로. 실제 제약은 `stage2_schema()` 가 건다."""
    return "\n".join(
        f"  {f}: " + (FIELD_VALUES[f] if isinstance(FIELD_VALUES[f], str)
                      else ", ".join(FIELD_VALUES[f]))
        for f in AXIS_FIELDS[axis])


def _field_block(axis: str, pv: str, fixes: frozenset) -> str:
    """실제 Stage-2 프롬프트에서 축별 `Fields:` 블록만 잘라낸다(공통 envelope 제외)."""
    body = stage2_prompt(axis, SLOT_LISTS, [], pv, fixes)
    return body.split("Fields:\n", 1)[1].split("\n\n" + STAGE2_TAIL, 1)[0]


def _type_str(prop: dict) -> str:
    """JSON Schema 한 속성을 한 줄 타입 표기로. 원 스키마의 제약을 빠짐없이 옮긴다."""
    if "enum" in prop:
        return "enum[" + ", ".join(prop["enum"]) + "]"
    if prop.get("type") == "array":
        inner = _type_str(prop["items"])
        return f"array of {inner}" + (f" (max {prop['maxItems']})" if "maxItems" in prop else "")
    return str(prop.get("type"))


def _schema_lines(schema: dict) -> str:
    """`stage2_schema()`/`STAGE1_SCHEMA` 를 속성 한 줄씩으로 옮긴다(디코딩 제약, 프롬프트 아님)."""
    props = schema["properties"]
    pad = max(len(k) for k in props) + 2
    body = "\n".join(f'  "{k}":'.ljust(pad + 4) + _type_str(v) for k, v in props.items())
    return "{\n" + body + "\n}  // all keys required, no others allowed"


def _item_schema(axis: str) -> str:
    """응답을 구속하는 실제 JSON Schema(아이템 1개분). 프롬프트 문구가 아니라 디코딩 제약이다."""
    return _schema_lines(stage2_schema(axis, SLOT_LISTS)["properties"]["items"]["items"])


# 슬롯 = 프로그램이 채우는 자리. B.1 의 `Given:` 목록에 해당하는 섹션을 블록마다 붙인다.
SLOTS_STAGE1 = f"- {SLOT_STORY}: the full text of one document"
SLOTS_ENVELOPE = ("- {{DIMENSION}}: protagonist | space | time | causality | intentionality\n"
                  "- {{FIELD_BLOCK}}: that dimension's field block\n"
                  f"- {SLOT_SENT}: one sentence, in the order shuffled for this dimension")
# 축별로 **그 블록에 실제로 등장하는** 슬롯만 싣는다.
SLOT_DESC = {"{{CAST}}": "the Stage-1 cast, comma-joined",
             "{{PROTAGONIST}}": "the Stage-1 protagonist id",
             "{{LOCATIONS}}": "the Stage-1 locations, comma-joined",
             "{{GOALS}}": "the Stage-1 goals, comma-joined"}


def _slots_used(text: str) -> str:
    used = [f"- {s}: {d}" for s, d in SLOT_DESC.items() if s in text]
    return "[Slots]\n" + "\n".join(used) if used else ""


# 축별 라벨 역할 + 캡션. 문구는 **논문 §Large-Scale Situation-State Annotation(Stage 2)** 의
# 서술을 그대로 따른다 — appendix 박스가 본문 설명과 1:1 로 대응해야 하기 때문이다.
# primary = Stage 3 의 상태변화 타깃을 만드는 필드, auxiliary = 기록만 하고 타깃에 안 들어가는 필드.
DIMENSION_NOTE = {
    "protagonist": {
        "primary": [("protagonist_present", "whether the central character is present")],
        "auxiliary": ["perspective", "characters_present"],
        "caption": "Primary label: whether the central character is present in the current "
                   "scene. The narrative perspective of the central acting character is "
                   "annotated as auxiliary information for absolute-state prediction; the "
                   "cast subset is retained as an ablation column only."},
    "space": {
        "primary": [("location", "the candidate location in which the action takes place")],
        "auxiliary": [],
        "caption": "Identifies the candidate location in which the described action takes "
                   "place. The neutral value unspecified is assigned when the setting cannot "
                   "be established from the sentence alone."},
    "time": {
        "primary": [("temporal_shift_marker", "the temporal-shift marker opening the sentence")],
        "auxiliary": ["tense"],
        "caption": "Indicates whether the sentence begins with a temporal-shift marker "
                   "expressing a time jump, sequential progression, or simultaneity. Markers "
                   "appearing mid-sentence, as well as causal or contrastive connectives, are "
                   "not treated as temporal transitions. Tense is auxiliary and unused."},
    "causality": {
        "primary": [("event_type", "the kind of event the sentence expresses"),
                    ("aspect", "whether an action or state begins, ends, or continues")],
        "auxiliary": ["causal_marker"],
        "caption": "Represented by the event type expressed by the sentence and by an aspect "
                   "label indicating whether a new action or state begins, ends, or continues. "
                   "The explicit causal connective is an auxiliary column."},
    "intentionality": {
        "primary": [("active_goal", "the goal the action is actually directed toward")],
        "auxiliary": [],
        "caption": "Identifies the active goal, among the candidate goals, toward which the "
                   "action in the sentence is actually directed. The neutral value none is "
                   "assigned when the action serves no candidate goal."},
}


def _labels_section(axis: str) -> str:
    """`[Labels]` — 어느 필드가 Stage-3 타깃이고 어느 것이 보조인지(논문 Stage 2 서술과 대응)."""
    note = DIMENSION_NOTE.get(axis)
    if not note:
        return ""
    lines = [f"- primary: {f} -- {g}" for f, g in note["primary"]]
    if note["auxiliary"]:
        lines.append("- auxiliary (recorded, not used as a target): "
                     + ", ".join(note["auxiliary"]))
    return "[Labels]\n" + "\n".join(lines)


def _tex_escape(s: str) -> str:
    """verbatim 밖(제목·캡션)에 나가는 문자열의 LaTeX 특수문자 이스케이프."""
    for a, b in (("\\", "\\textbackslash "), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                 ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}")):
        s = s.replace(a, b)
    return s


def dump_prompts(pv: str, fixes: frozenset, axes: tuple[str, ...] = AXES,
                 fmt: str = "text", width: int = 62, box_prefix: str = "A") -> str:
    """Stage-1 + Stage-2 프롬프트 전문을 렌더링한다 (`--dump-prompts`).

    실제 호출 경로(`build_stage1` / `stage2_prompt`)를 그대로 태우므로 여기 출력은 전송 문자열과
    정의상 일치한다. 문서·문장·닫힌목록 자리만 `SLOT_*` 로 치환된다.

    Args:
        pv: 프롬프트 버전 태그(동결 앵커 = `v3.1`).
        fixes: `fixes_of()` 가 돌려준 fix 집합.
        axes: 덤프할 축. 기본은 동결 5축(실험 축 `chainrole` 제외).
        fmt: `text` = 사람이 읽는 덤프, `latex` = 논문 appendix 에 그대로 붙이는 verbatim 블록.
        width: latex 모드의 hard-wrap 열 수(ACL 단일 컬럼 \\footnotesize 기준 62).
        box_prefix: latex 모드 박스 번호의 접두(Appendix A 에 실으면 "A" → Box A.1, A.2 …).

    Returns:
        렌더링된 프롬프트 전문. system/user 메시지와 닫힌 값 집합을 모두 포함한다.
    """
    tag = (("p0s5-eism-fix:" + ",".join(sorted(fixes)))
           if fixes and pv not in VERSIONS else VERSIONS[pv])
    head = (f"prompt_version = {tag}   fixes = {{{', '.join(sorted(fixes)) or '-'}}}\n"
            "decoding = temperature 0, JSON-schema constrained, batch of items per Stage-2 call")
    # 한 블록 = (제목, 캡션, [(kind, 본문)]). kind: prose = 재조판 허용, pre = 줄 구조 보존.
    stage1_seg = [("prose", f"[System]\n{SYSTEM_STAGE1}\n\n[User]\n{build_stage1(fixes, SLOT_STORY)}"),
                  ("prose", f"[Slots]\n{SLOTS_STAGE1}"),
                  ("pre", f"[Response schema]\n{_schema_lines(STAGE1_SCHEMA)}")]
    blocks: list[tuple[str, str, list[tuple[str, str]]]] = [
        ("Stage 1: Candidate Set Construction",
         "One call per document. The normalized inventories it returns serve as the label "
         "spaces used in Stage 2 and the comparison units used in Stage 3. To avoid "
         "introducing event-structural information, the prompt prohibits descriptions of "
         "event order or structure and restricts outputs related to boundaries, segments, "
         "changes, or scenes.",
         stage1_seg)]

    if fmt == "text":                       # 축마다 실제로 보낸 문자열 전체를 그대로
        for axis in axes:
            blocks.append(
                (f"Stage 2 field block --- {axis}", "",
                 [("prose", f"[System]\n{SYSTEM_STAGE2}\n\n[User]\n"
                            f"{stage2_prompt(axis, SLOT_LISTS, [(0, SLOT_SENT)], pv, fixes)}"),
                  ("prose", f"[Closed value sets]\n{_closed_values(axis)}"),
                  ("pre", "[Response schema] the reply is {\"items\": [<item>, ...]}, "
                  f"one <item> per ITEM:\n{_item_schema(axis)}")]))
        bar = "=" * 96
        out = [bar, head, bar]
        for title, _cap, segs in blocks:
            out += ["", f"### {title}", ""] + ["\n".join(b for _k, b in segs)] + ["", bar]
        return "\n".join(out)

    # latex 모드는 **factored**: 축마다 글자 그대로 같은 envelope(system·머리말·공통 꼬리말·아이템
    # 형식)를 5번 싣는 것은 지면 낭비다. envelope 를 한 번만 싣고 축별로는 Fields 블록과 응답
    # 스키마만 싣는다 — 슬롯을 채워 넣으면 실제 전송 문자열이 그대로 복원된다.
    blocks.append((
        "Stage 2: Sentence-Level State Assignment (shared envelope)",
        "Byte-identical for all five dimensions; substituting the dimension name and its field "
        "block reconstructs the exact request sent to the annotator. Each sentence is annotated "
        "independently, no label is carried over from another sentence, and the sentence order "
        "is the one shuffled for that dimension.",
        [("prose", f"[System]\n{SYSTEM_STAGE2}\n\n[User]\n"
                   + STAGE2_TEMPLATE.format(
                       axis="{{DIMENSION}}", field="{{FIELD_BLOCK}}", tail=STAGE2_TAIL,
                       payload="ITEM 0\nTARGET: {{SENTENCE_0}}\n\nITEM 1\n"
                               "TARGET: {{SENTENCE_1}}\n\n[... one batch of items per call]")),
         ("prose", f"[Slots]\n{SLOTS_ENVELOPE}")]))
    for axis in axes:
        field = _field_block(axis, pv, fixes)
        segs = [("prose", f"[Field block: {axis}]\n{field}")]
        for extra in (_slots_used(field), _labels_section(axis)):
            if extra:
                segs.append(("prose", extra))
        segs.append(("pre", "[Response schema] the reply is {\"items\": [<item>, ...]}, "
                     f"one <item> per ITEM:\n{_item_schema(axis)}"))
        cap = DIMENSION_NOTE.get(axis, {}).get("caption", "")
        blocks.append((f"Stage 2 field block --- {axis}",
                       f"Substituted into the envelope of Box {box_prefix}.2. " + cap, segs))

    subs = ", ".join(f"{k} -> {v}" for k, v in ASCII_FOLD.items()
                     if any(k in b for _t, _c, sg in blocks for _k, b in sg))
    out = ["% AUTO-GENERATED -- do not hand-edit. Regenerate with:",
           f"%   python scripts/eacl_phase0/llm_s5_states.py --dump-prompts "
           f"--dump-format latex --prompt-version {pv}",
           f"% {head.splitlines()[0]}",
           f"% ASCII substitutions for verbatim: {subs or '(none)'}"]
    for i, (title, cap, segs) in enumerate(blocks, 1):
        lines: list[str] = []
        for j, (kind, text) in enumerate(segs):
            if j:
                lines.append("")
            wrap = _pre_wrap if kind == "pre" else _reflow_wrap
            lines += wrap(_fold_ascii(text), width)
        out += ["", "\\vspace{0.6em}", "\\noindent\\rule{\\columnwidth}{0.8pt}\\\\[-0.4em]",
                f"{{\\footnotesize\\textbf{{Box {box_prefix}.{i}: "
                f"{_tex_escape(_fold_ascii(title))}}}}}\\\\[-0.7em]",
                "\\noindent\\rule{\\columnwidth}{0.4pt}",
                "\\nopagebreak", "{\\footnotesize", "\\begin{verbatim}", *lines,
                # `\end{verbatim}` 뒤 같은 줄에는 아무것도 두지 않는다(그룹 닫는 중괄호도 다음 줄로).
                "\\end{verbatim}", "}", "\\vspace{-0.7em}",
                "\\noindent\\rule{\\columnwidth}{0.8pt}"]
        if cap:
            out += [f"\\vspace{{0.2em}}\n"
                    f"{{\\footnotesize\\textit{{{_tex_escape(_fold_ascii(cap))}}}}}"]
        out += ["\\vspace{0.6em}"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path, nargs="?")
    ap.add_argument("out_dir", type=Path, nargs="?")
    ap.add_argument("--model", default="openrouter/deepseek/deepseek-v4-pro")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--stage1-max-tokens", type=int, default=8000,
                    help="raise for reasoning models that return null content (see stage1 docstring)")
    ap.add_argument("--seed", type=int, default=20260711)
    # 180 은 개체가 많은 문서의 Stage-1 생성이 못 끝내고 APITimeout 으로 죽는다
    # (2026-07-27 phase1: 4편 소실). 상한일 뿐이라 라벨 내용에는 영향 없음. 기본값은 동결 유지.
    ap.add_argument("--timeout", type=float, default=180, help="HTTP 요청 타임아웃(초)")
    ap.add_argument("--axis", choices=AXES + EXPERIMENT_AXES, action="append")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="disable model reasoning (extra_body={'reasoning':{'enabled':False}}) — "
                         "D-B11 determinism experiment; NOT the frozen config (default keeps reasoning ON)")
    ap.add_argument("--prompt-version", choices=["v2", "v3", "v3.1"], default="v2",
                    help="v2 = p0s5-eism-v2 (default, 이력 재현용). **v3.1 = 동결 앵커 버전(D-B14)** "
                         "= H(space 문장근거 강제)+E(time 문두조건) 만. v3 = D-B12 전체 fix 묶음"
                         "(A/H space, E time, B②/B① intentionality) — 미채택, 실험용. 별도 out_dir 사용.")
    ap.add_argument("--engine", choices=["proxy", "vllm"], default="proxy",
                    help="thinking-off dialect for --no-reasoning: proxy=deepseek/openrouter "
                         "({'reasoning':{'enabled':False}}); vllm=Qwen/local "
                         "({'chat_template_kwargs':{'enable_thinking':False}}). Phase-1 open-model pilot.")
    ap.add_argument("--base-url", default=None, help="override OPENAI_BASE_URL (e.g. local vLLM)")
    ap.add_argument("--fixes", default=None,
                    help="개별 fix CSV (space_A,space_H,intent_2,intent_1,time_E,time_conseq,time_first). "
                         "지정 시 --prompt-version 무시하고 v2+그 fix만. 비지정 시 pv(v2=∅/v3=ALL). 별도 out_dir.")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help="vLLM chat_template_kwargs 를 JSON 으로 직접 지정 (--no-reasoning 의 기본 dialect 를 "
                         "대체). gpt-oss 처럼 enable_thinking 이 없고 reasoning_effort 만 있는 모델용 — "
                         '예: \'{"reasoning_effort":"low"}\'. 미지정 시 동결 동작 그대로.')
    ap.add_argument("--dump-prompts", action="store_true",
                    help="실제 호출 경로로 렌더링한 Stage-1/Stage-2 프롬프트 전문을 stdout 으로 출력하고 "
                         "종료한다(API 호출 없음). 논문 appendix 수록용. `--prompt-version`/`--fixes` 를 "
                         "그대로 따르므로 동결본은 `--dump-prompts --prompt-version v3.1`.")
    ap.add_argument("--dump-format", choices=["text", "latex"], default="text",
                    help="--dump-prompts 출력 형식. latex = appendix 에 그대로 붙이는 verbatim 블록 "
                         "(hard-wrap + ASCII 폴딩, 치환 목록을 헤더 주석에 명시).")
    ap.add_argument("--dump-width", type=int, default=62,
                    help="--dump-format latex 의 hard-wrap 열 수 (기본 62 = ACL 단컬럼 footnotesize).")
    ap.add_argument("--dump-box-prefix", default="A",
                    help="latex 박스 번호 접두. 프롬프트를 Appendix A 에 실으면 A → Box A.1 …")
    args = ap.parse_args()
    if args.dump_prompts:
        axes = tuple(args.axis) if args.axis else AXES
        print(dump_prompts(args.prompt_version, fixes_of(args.prompt_version, args.fixes), axes,
                           args.dump_format, args.dump_width, args.dump_box_prefix))
        return
    if args.master is None or args.out_dir is None:
        ap.error("master 와 out_dir 은 필수다 (--dump-prompts 일 때만 생략 가능)")
    if args.chat_template_kwargs:
        extra_body = {"chat_template_kwargs": json.loads(args.chat_template_kwargs)}
    elif args.no_reasoning:
        extra_body = ({"chat_template_kwargs": {"enable_thinking": False}} if args.engine == "vllm"
                      else {"reasoning": {"enabled": False}})
    else:
        extra_body = None
    pv = args.prompt_version
    fixes = fixes_of(pv, args.fixes)
    vtag = ("p0s5-eism-fix:" + ",".join(sorted(fixes))) if args.fixes else VERSIONS[pv]

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from openai import OpenAI
    base_url = args.base_url or os.environ["OPENAI_BASE_URL"]
    # local vLLM (--base-url) needs no CF-Access headers and accepts any api_key
    if args.base_url:
        c = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"), base_url=base_url,
                   timeout=args.timeout, max_retries=0)
    else:
        c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url,
                   timeout=180, max_retries=0,  # fail hung structured calls fast; own retry loop handles it
                   default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                                    "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]})

    all_rows = read(args.master)
    by_doc: dict[str, list[dict]] = {}
    for r in all_rows:
        by_doc.setdefault(str(r["doc_id"]), []).append(r)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrix_all: list[dict] = []
    for doc_id, rows in by_doc.items():
        rows.sort(key=lambda x: int(x["transition"]))
        texts = sequence(rows)
        ddir = args.out_dir / doc_id
        ddir.mkdir(parents=True, exist_ok=True)

        # Stage 1 (cache per doc)
        lp = ddir / "story_lists.json"
        if lp.exists():
            lists = json.loads(lp.read_text())["lists"]
        else:
            lists, s1_prompt = stage1(c, args.model, "\n".join(texts), args.stage1_max_tokens,
                                      extra_body, pv, fixes)
            lp.write_text(json.dumps({"doc_id": doc_id, "lists": lists, "model": args.model,
                                      "prompt_version": vtag,
                                      "prompt_sha256": hashlib.sha256(s1_prompt.encode()).hexdigest()},
                                     ensure_ascii=False, indent=2))
        print(f"[{doc_id}] stage1 cast={len(lists['cast'])} prot={lists.get('protagonist')} "
              f"loc={len(lists['locations'])} goals={len(lists['goals'])}", flush=True)

        # Stage 2 per axis
        states: dict[str, dict[int, dict]] = {}
        for axis in args.axis or AXES:
            sp = ddir / f"{axis}_states.jsonl"
            done = {x["item_id"]: x for x in read(sp)}
            ids = list(range(len(texts)))
            # 실험 축은 동결 축 뒤 인덱스를 받아 seed 가 겹치지 않는다(동결 5축 seed 불변).
            all_axes = AXES + EXPERIMENT_AXES
            seed = args.seed + all_axes.index(axis)
            random.Random(seed).shuffle(ids)
            while len(done) < len(texts):
                todo = [i for i in ids if i not in done][:args.batch]
                got, prompt = stage2_call(c, args.model, axis, lists,
                                          [(i, texts[i]) for i in todo], extra_body, pv, fixes)
                sha = hashlib.sha256(prompt.encode()).hexdigest()
                accepted = 0
                for x in got:
                    i = int(x["item_id"])
                    if i not in set(todo):
                        continue
                    vals = valid(axis, x, lists)
                    if vals is None:
                        continue
                    done[i] = {"item_id": i, "axis": axis, **vals, "model": args.model,
                               "prompt_version": vtag, "prompt_sha256": sha, "shuffle_seed": seed}
                    accepted += 1
                if not accepted:
                    raise RuntimeError(f"[{doc_id}] {axis}: no valid outputs for {todo}")
                write(sp, [done[i] for i in sorted(done)])
                print(f"[{doc_id}] state {axis}: {len(done)}/{len(texts)}", flush=True)
            states[axis] = done

        # carry-forward slow variables (code, boundary-agnostic)
        if "space" in states:
            states["space"] = carry_forward(states["space"], "location", "unspecified")
        if "intentionality" in states:
            states["intentionality"] = carry_forward(states["intentionality"], "active_goal", "none")

        # reset_matrix (only when all axes present). Frozen columns + additive experiment columns.
        if not args.axis or set(args.axis) == set(AXES):
            by_t = {int(r["transition"]): r for r in rows}
            for i in range(1, len(texts)):
                r = by_t[i]
                pc = reset_protagonist_charset(states["protagonist"][i - 1], states["protagonist"][i])
                pp = reset_perspective(states["protagonist"][i - 1], states["protagonist"][i])
                ppr = reset_protagonist_present(states["protagonist"][i - 1], states["protagonist"][i])
                cm = reset_causal_marker(states["causality"][i - 1], states["causality"][i])
                row = {"doc_id": doc_id, "transition": i, "boundary": int(r["boundary"]),
                       "idm_score": (None if r.get("idm_score") is None else float(r["idm_score"])),
                       # --- frozen ---
                       "protagonist_charset": pc, "protagonist_perspective": pp,
                       "protagonist": int(pc or pp),
                       "space": reset_space(states["space"][i - 1], states["space"][i]),
                       "time": reset_time(states["time"][i - 1], states["time"][i]),
                       "causality": reset_causality(states["causality"][i - 1], states["causality"][i]),
                       "intentionality": reset_intentionality(states["intentionality"][i - 1],
                                                              states["intentionality"][i]),
                       # --- additive experiment (NOT in frozen count) ---
                       "protagonist_present": ppr, "causal_marker": cm}
                matrix_all.append(row)
            print(f"[{doc_id}] reset_matrix rows so far: {len(matrix_all)}", flush=True)

    if matrix_all:
        write(args.out_dir / "reset_matrix.jsonl", matrix_all)
        print(f"reset_matrix: {len(matrix_all)} transitions -> {args.out_dir/'reset_matrix.jsonl'}",
              flush=True)


if __name__ == "__main__":
    main()
