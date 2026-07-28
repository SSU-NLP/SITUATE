#!/usr/bin/env python3
"""P0-S4: target-only LLM closed-enum states; deterministic enum-change resets.

S3의 free-form anchor 노이즈(死因)를 닫힌 enum으로 원천 정규화한다.
- LLM은 target 문장 단독에서 아래 고정 enum 값만 출력(자유서술 금지).
- 변화/리셋/경계/이전 문장은 프롬프트에 절대 없음.
- 리셋 판정은 100% 코드가 인접 문장 enum을 비교(아래 RESET 규칙, gold 전 고정).
스펙 출처: EACL/phase0-annotation-guide.md
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
from dotenv import load_dotenv

VERSION = "p0s4-enum-v1"
AXES = ("protagonist", "place", "time", "causality", "goal")

# 축별 닫힌 스키마 (target 문장의 절대 속성; 상대·비교 표현 금지)
FIELDS: dict[str, dict[str, list[str]]] = {
    "protagonist": {
        "perspective": ["first", "second", "third", "none"],
        "actor_role": ["child_or_student", "white_collar_professional",
                       "service_or_manual_worker", "authority_or_official",
                       "medical_or_caregiver", "family_role", "creative_or_artist",
                       "criminal_or_antagonist", "supernatural_or_nonhuman",
                       "generic_person", "unknown"],
    },
    "place": {
        "setting_type": ["home", "workplace", "school", "commercial",
                         "street_or_urban_outdoor", "nature_outdoor",
                         "vehicle_or_transit", "institution", "entertainment_venue",
                         "online_virtual", "unspecified", "other"],
    },
    "time": {
        "tense": ["past", "present", "future", "none"],
        "new_time_frame_marker": ["present", "absent"],
    },
    "causality": {
        "event_type": ["dialogue", "physical_action", "movement_or_travel",
                       "perception_or_observation", "thought_or_emotion",
                       "transaction_or_exchange", "conflict_or_violence",
                       "work_or_creation", "state_description", "other"],
    },
    "goal": {
        "goal_present": ["explicit", "implicit", "none"],
        "goal_category": ["safety_or_survival", "achievement_or_success",
                          "relationship_or_social", "material_or_acquisition",
                          "knowledge_or_discovery", "escape_or_avoidance",
                          "duty_or_obligation", "none"],
    },
}

# A-검증용: place가 죽은 게 enum 거칠기(coarseness)냐 sparsity냐를 가리기 위한
# 세분화 setting_type. 결과가 여전히 near-dead면 원인=sparsity 확정(gold 무관 진단).
FINE_PLACE = ["home_interior", "workplace_office", "school_or_classroom",
              "shop_or_restaurant_indoor", "entertainment_venue_indoor",
              "institution_indoor", "street_or_road", "urban_plaza_or_yard",
              "forest_or_woods", "water_or_shore", "field_or_open_land",
              "mountain_or_hill", "vehicle_interior", "transit_station",
              "underground_or_indoor_passage", "sky_or_space",
              "online_virtual", "unspecified", "other"]

FIELD_PROMPT = {
    "protagonist": (
        "perspective: grammatical person of the central actor/experiencer in THIS sentence "
        "(first/second/third); none if no person is present.\n"
        "actor_role: the social/identity category of that central actor in THIS sentence; "
        "unknown if not determinable from THIS sentence alone."),
    "place": (
        "setting_type: the kind of location where the action of THIS sentence is situated; "
        "unspecified if none is expressed."),
    "time": (
        "tense: dominant grammatical tense of THIS sentence (past/present/future); none if not determinable.\n"
        "new_time_frame_marker: 'present' only if THIS sentence itself contains an expression that opens a new "
        "time frame (e.g. 'one day', 'the next morning', 'years later', 'in 2019'); otherwise 'absent'."),
    "causality": (
        "event_type: the kind of event/action expressed in THIS sentence."),
    "goal": (
        "goal_present: whether THIS sentence expresses a goal/intention explicitly, implicitly, or none.\n"
        "goal_category: the category of that goal; none if no goal is expressed."),
}


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


def schema(axis: str) -> dict:
    props = {"item_id": {"type": "integer"}}
    for name, vals in FIELDS[axis].items():
        props[name] = {"type": "string", "enum": vals}
    item = {"type": "object", "additionalProperties": False,
            "properties": props, "required": list(props)}
    return {"type": "object", "additionalProperties": False,
            "properties": {"items": {"type": "array", "items": item}}, "required": ["items"]}


def call(c, model: str, axis: str, items: list[tuple[int, str]]):
    payload = "\n\n".join(f"ITEM {i}\nTARGET: {s}" for i, s in items)
    prompt = f"""Label the {axis} state expressed by each TARGET sentence, independently.
Choose exactly one value per field from its fixed list. If unclear, use the neutral value
(none/unknown/unspecified/absent). Never invent values outside the lists.

Fields:
{FIELD_PROMPT[axis]}

Items are shuffled and unrelated. Never compare items to each other. Do not reason about documents,
stories, transitions, order, continuity, novelty, changes, resets, or boundaries. Judge each TARGET
sentence on its own. Return exactly one result for every item_id.

{payload}"""
    for a in range(3):
        try:
            r = c.chat.completions.create(
                model=model, temperature=0, max_tokens=10000,
                response_format={"type": "json_schema", "json_schema":
                                 {"name": f"{axis}_enum_states", "strict": True, "schema": schema(axis)}},
                messages=[{"role": "system",
                           "content": "You assign fixed categorical labels to one sentence at a time. "
                                      "You never compare sentences and never output free text."},
                          {"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content)["items"], prompt
        except Exception:
            if a == 2:
                raise
            time.sleep(2 * (a + 1))


def reset(axis: str, a: dict, b: dict) -> int:
    """인접 문장 a=t-1, b=t 의 enum 비교로 리셋 0/1 (gold 전 고정 규칙)."""
    if axis == "protagonist":
        persp = a["perspective"] != b["perspective"] and "none" not in (a["perspective"], b["perspective"])
        role = a["actor_role"] != b["actor_role"] and "unknown" not in (a["actor_role"], b["actor_role"])
        return int(persp or role)
    if axis == "place":
        return int(a["setting_type"] != b["setting_type"]
                   and "unspecified" not in (a["setting_type"], b["setting_type"]))
    if axis == "time":
        tense = a["tense"] != b["tense"] and "none" not in (a["tense"], b["tense"])
        marker = b["new_time_frame_marker"] == "present"
        return int(tense or marker)
    if axis == "causality":
        return int(a["event_type"] != b["event_type"])
    if axis == "goal":
        return int(a["goal_category"] != b["goal_category"]
                   and "none" not in (a["goal_category"], b["goal_category"]))
    raise ValueError(axis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--model", default="openrouter/anthropic/claude-fable-5")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--axis", choices=AXES, action="append")
    ap.add_argument("--fine-place", action="store_true",
                    help="place setting_type을 세분 enum으로 교체(coarseness vs sparsity 진단용).")
    args = ap.parse_args()
    if args.fine_place:
        FIELDS["place"]["setting_type"] = FINE_PLACE

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
               default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                                "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]})

    rows = read(args.master)
    texts = sequence(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states: dict[str, dict[int, dict]] = {}

    for axis in args.axis or AXES:
        sp = args.out_dir / f"{axis}_states.jsonl"
        done = {x["item_id"]: x for x in read(sp)}
        ids = list(range(len(texts)))
        random.Random(args.seed + AXES.index(axis)).shuffle(ids)
        seed = args.seed + AXES.index(axis)
        while len(done) < len(texts):
            todo = [i for i in ids if i not in done][:args.batch]
            got, prompt = call(c, args.model, axis, [(i, texts[i]) for i in todo])
            sha = hashlib.sha256(prompt.encode()).hexdigest()
            accepted = 0
            for x in got:
                i = int(x["item_id"])
                if i not in set(todo):
                    continue
                vals = {}
                ok = True
                for name, allowed in FIELDS[axis].items():
                    v = x.get(name)
                    if v not in allowed:
                        ok = False
                        break
                    vals[name] = v
                if not ok:
                    continue
                done[i] = {"item_id": i, "axis": axis, **vals, "model": args.model,
                           "prompt_version": VERSION, "prompt_sha256": sha, "shuffle_seed": seed}
                accepted += 1
            if not accepted:
                raise RuntimeError(f"{axis}: no valid enum outputs for {todo}")
            write(sp, [done[i] for i in sorted(done)])
            print(f"state {axis}: {len(done)}/{len(texts)}", flush=True)
        states[axis] = done

        changes = []
        for i in range(1, len(texts)):
            a, b = done[i - 1], done[i]
            v = reset(axis, a, b)
            changes.append({"doc_id": rows[i - 1]["doc_id"], "transition": i, "axis": axis,
                            "suggested_value": v,
                            "before": {k: a[k] for k in FIELDS[axis]},
                            "after": {k: b[k] for k in FIELDS[axis]},
                            "method": "deterministic-enum-compare", "prompt_version": VERSION})
        write(args.out_dir / f"{axis}_changes.jsonl", changes)
        print(f"change {axis}: {len(changes)}/{len(texts) - 1}", flush=True)

    # 평가용 reset_matrix: transition마다 boundary+idm_score+5축 reset 0/1
    if not args.axis or set(args.axis) == set(AXES):
        by_t = {int(r["transition"]): r for r in rows}
        matrix = []
        for i in range(1, len(texts)):
            r = by_t[i]
            row = {"doc_id": r["doc_id"], "transition": i,
                   "boundary": int(r["boundary"]), "idm_score": float(r["idm_score"])}
            for axis in AXES:
                row[axis] = reset(axis, states[axis][i - 1], states[axis][i])
            matrix.append(row)
        write(args.out_dir / "reset_matrix.jsonl", matrix)
        print(f"reset_matrix: {len(matrix)} transitions -> {args.out_dir/'reset_matrix.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
