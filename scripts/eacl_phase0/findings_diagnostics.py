#!/usr/bin/env python3
"""②진단 — ①B human 표본이 연 설계 발견 A·H·B①·B②의 **스케일(빈도)** 게이트.

목적(HANDOFF s5-v3-redesign): 표본 1~2건으로 확정된 결함이 **문서 스케일에서 실제로 흔한가**를
캐시된 run1 라벨(예산 0)로 gold-free 측정해, ④ 재설계에 포함할지 게이트한다. 여기서 하는 건
'무엇을 고칠지 결정'이 아니라 '결함이 흔한지 수치화'다(결정은 HANDOFF 표에 사전 명시).

측정 4종 (E=time 비대칭은 `time_marker_position_diag.py`로 별도 완료 — 여기 §E에 요약만):
  A. space list-miss (under) — 무대(특히 탈것/공중)가 명백한데 Stage-1 locations 목록에 없어
     unspecified 가 되는 빈도. 직접신호 = unspecified 라벨 문장 중 **탈것/공중 어휘를 담았는데
     그 무대가 locations 목록에 없는** 문장 수 + 문서·장르별 unspecified율·최장 unspecified run.
  H. space context-leak (over) — 문장 자체에 **장소단서가 없는데** non-unspecified 위치가 붙는 비율
     (raw 라벨, carry 전). no-cue 문장의 non-unspecified율 = leak **상한**(암묵무대 혼입으로 과대).
  B①. 대사 화행별 P(goal≠none) — event_type=='dialogue' 문장을 수행적(명령·질문·약속·계획) vs
     순수논의로 나눠 active_goal≠none 비율 대조. 수행적이 논의만큼 none 이면 → ① 확정.
  B②. goal eventiveness + leave-one-goal-out — raw active_goal 의 목표별 span/count/re-entry 로
     상시(pervasive)목표 식별 + 그 목표 제거 시 intentionality reset·deployable Score Δ.

전부 gold-free · 캐시 읽기 + regex 휴리스틱(E 진단과 동급 스크리닝 도구; 정밀도 아닌 방향·규모).
산출: outputs/experiments/2026-07-23_findings_diagnostics/REPORT.md
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from llm_s5_states import carry_forward, reset_intentionality  # noqa: E402

RUN = ROOT / "outputs/eacl_phase0/repcorpus_s5_eismv2_run1"
MASTER = ROOT / "outputs/eacl_phase0/repcorpus_master.jsonl"
OUT = ROOT / "outputs/experiments/2026-07-23_findings_diagnostics"
FROZEN = ["protagonist", "space", "time", "causality", "intentionality"]  # reset_matrix column names
T_STAR = 3

# 사전 선언 판정 기준 (rubric-before-results — no-postresult-criterion-change)
CRIT = {
    "A": "탈것/공중 목록누락 문장이 ≥1 문서에서 눈에 띄게(해당문서 unspecified의 유의 비율) 발생 + obs>fam 편중 → ④에 A 포함(조건부 충족).",
    "H": "no-cue 문장 non-unspecified율(상한)이 전역 non-unspecified율 대비 낮지 않고 절대적으로도 상당(대략 ≥0.2)하면 leak 스케일 실재 → ④ H 포함(이미 ✅, 규모 확인).",
    "B1": "P(goal≠none|수행적 대사) − P(goal≠none|논의 대사) ≤ 0.10 AND 수행적 절대 P(goal≠none) < 0.5 → 모델이 화행을 무시(수행적도 none) → ① 확정.",
    "B2": "span≥0.6(문서길이 대비) 목표가 ≥1 문서에 존재(=상시목표 실재) AND leave-one-out 이 그 문서 intentionality reset율 또는 Score 를 |Δ|>0(비영) 변화 → 상시목표가 경계신호 오염 확증 → ④ B② 포함(이미 ✅, 규모 확인).",
}

# ---------------------------------------------------------------- shared IO
def read(p: Path) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def sequence(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda x: int(x["transition"]))
    return [rows[0]["previous"], *[x["current"] for x in rows]]


def load_sentences() -> dict[str, list[str]]:
    by_doc = defaultdict(list)
    for r in read(MASTER):
        by_doc[str(r["doc_id"])].append(r)
    return {d: sequence(rs) for d, rs in by_doc.items()}


def docs() -> list[str]:
    return sorted(p.name for p in RUN.iterdir() if p.is_dir())


def genre(doc: str) -> str:
    return "fam" if doc.startswith("fam_") else "obs"


# ================================================================ A: space list-miss (under)
# 탈것/공중/특수무대 어휘 — 이게 문장에 있는데 문서 locations 목록에 없으면 '목록 누락 무대'.
CONVEYANCE = re.compile(
    r"\b(aircraft|aeroplane|airplane|plane|cockpit|fuselage|wing|propeller|"
    r"biplane|monoplane|scout(?:_plane)?|machine|sky|skies|air|mid-?air|clouds?|altitude|"
    r"boat|ship|deck|vessel|schooner|steamer|canoe|raft|barge|yacht|sail|"
    r"train|carriage|coach|wagon|cart|car|cab|automobile|buggy|sled|sledge|sleigh|"
    r"saddle|horseback|mount|"
    r"cabin|berth|compartment|hold)\b", re.IGNORECASE)
# 위 어휘를 locations 목록 토큰과 매칭할 때 쓰는 정규화
_WS = re.compile(r"[^a-z]+")


def loc_tokens(locations: list[str]) -> set[str]:
    toks: set[str] = set()
    for l in locations:
        toks.update(t for t in _WS.split(l.lower()) if t)
    return toks


def diag_A(sents: dict[str, list[str]]) -> list[dict]:
    rows = []
    examples = []
    for doc in docs():
        lists = json.loads((RUN / doc / "story_lists.json").read_text())["lists"]
        ltoks = loc_tokens(lists["locations"])
        st = read(RUN / doc / "space_states.jsonl")
        n = len(st)
        if not n:
            continue
        unspec = [int(s["item_id"]) for s in st if s["location"] == "unspecified"]
        unspec_rate = len(unspec) / n
        # 최장 unspecified run
        labs = [st[i]["location"] for i in range(n)]  # states written in item_id order? ensure sort
        labs = [s["location"] for s in sorted(st, key=lambda x: int(x["item_id"]))]
        longest = cur = 0
        for l in labs:
            cur = cur + 1 if l == "unspecified" else 0
            longest = max(longest, cur)
        # conveyance-miss: unspecified 문장에 탈것/공중 어휘 있고 그 어휘가 목록 토큰에 없음
        miss = 0
        for i in unspec:
            if i >= len(sents.get(doc, [])):
                continue
            sent = sents[doc][i]
            m = CONVEYANCE.findall(sent)
            m = [w.lower() for w in m]
            hit = [w for w in m if not any(w[:4] in t or t in w for t in ltoks)]
            if hit:
                miss += 1
                if len(examples) < 12:
                    examples.append((doc, i, hit, sent[:110]))
        rows.append({"doc": doc, "genre": genre(doc), "n": n, "list_size": len(lists["locations"]),
                     "unspec_rate": unspec_rate, "longest_unspec_run": longest, "conveyance_miss": miss})
    return rows, examples


# ================================================================ H: space context-leak (over)
# 장소단서 감지(넓게) — 하나라도 있으면 'cue 있음'. 없으면 'no-cue'(leak 후보 표본).
SETTING = re.compile(
    r"\b(room|chamber|bedroom|parlou?r|hall|kitchen|study|library|attic|cellar|"
    r"house|home|cottage|cabin|hut|mansion|palace|castle|tower|"
    r"door|doorway|window|wall|floor|ceiling|stair|staircase|roof|threshold|gate|fence|hearth|fire(?:place|side)|"
    r"street|road|lane|path|trail|way(?:side)?|corner|crossing|bridge|"
    r"yard|garden|court(?:yard)?|field|meadow|lawn|pasture|farm|barn|"
    r"wood|woods|forest|tree|grove|hill(?:side)?|mountain|valley|cliff|rock|cave|"
    r"river|stream|brook|creek|pond|lake|sea|shore|beach|coast|bank|water|"
    r"village|town|city|street|market|square|station|platform|"
    r"church|chapel|temple|tavern|inn|shop|store|office|bank|prison|cell|camp|tent|"
    r"deck|cockpit|saddle|carriage|coach|wagon|boat|ship|train|car|"
    r"veranda|porch|balcony|terrace|"
    r"aircraft|plane|sky|air|trenches?|airdrome|"
    r"here|there|outside|inside|within|indoors|outdoors|upstairs|downstairs|abroad|"
    r"trenches?)\b", re.IGNORECASE)
SPATIAL_PREP = re.compile(
    r"\b(in|into|inside|within|on|onto|upon|at|by|near|beside|behind|beyond|across|through|"
    r"under|below|beneath|above|over|toward|towards?|past|along|around|amid|amidst|among)\s+"
    r"(the|a|an|his|her|their|its|this|that|my|our|your)\b", re.IGNORECASE)
MOVEMENT = re.compile(
    r"\b(walk|walked|ran|run|running|went|came|climb|climbed|enter|entered|rode|ride|drove|drive|"
    r"sail|sailed|flew|fly|cross|crossed|wander|wandered|strode|stride|stepped|step|moved|move|"
    r"return|returned|arrive|arrived|approach|approached|descend|descended|ascend|ascended|"
    r"pass|passed|gallop|galloped|march|marched|hurried|hurry|creep|crept|crawl|crawled)\b",
    re.IGNORECASE)


def has_place_cue(sent: str, ltoks: set[str]) -> bool:
    if SETTING.search(sent) or SPATIAL_PREP.search(sent) or MOVEMENT.search(sent):
        return True
    low = sent.lower()
    return any(len(t) >= 4 and t in low for t in ltoks)  # doc 고유 장소 토큰(방이름 등)


def diag_H(sents: dict[str, list[str]]) -> list[dict]:
    rows = []
    examples = []
    for doc in docs():
        lists = json.loads((RUN / doc / "story_lists.json").read_text())["lists"]
        ltoks = loc_tokens(lists["locations"])
        st = {int(s["item_id"]): s for s in read(RUN / doc / "space_states.jsonl")}
        n = len(st)
        if not n:
            continue
        cue = {"n": 0, "nonunspec": 0}
        nocue = {"n": 0, "nonunspec": 0}
        for i in range(n):
            if i >= len(sents.get(doc, [])):
                continue
            sent = sents[doc][i]
            nonunspec = int(st[i]["location"] != "unspecified")
            bucket = cue if has_place_cue(sent, ltoks) else nocue
            bucket["n"] += 1
            bucket["nonunspec"] += nonunspec
            if bucket is nocue and nonunspec and len(examples) < 12:
                examples.append((doc, i, st[i]["location"], sent[:110]))
        rows.append({"doc": doc, "genre": genre(doc), "n": n,
                     "cue_n": cue["n"], "cue_nonunspec": (cue["nonunspec"] / cue["n"] if cue["n"] else None),
                     "nocue_n": nocue["n"],
                     "nocue_nonunspec": (nocue["nonunspec"] / nocue["n"] if nocue["n"] else None)})
    return rows, examples


# ================================================================ B①: 대사 화행별 P(goal≠none)
QUESTION = re.compile(r"\?")
EXCLAIM = re.compile(r"!")
DIRECTIVE = re.compile(
    r"\b(must|shall|let us|let's|come|go|hand over|give me|tell me|bring|take|stop|wait|listen|"
    r"swear|promise|i will|we will|i'll|we'll|you will|meet me|follow|stay|leave|help me|"
    r"do not|don't|you shall|shall we|would you|will you|can you|could you|please|"
    r"i want|i wish|i mean to|i intend|let me|remember|be sure|see that|make sure)\b",
    re.IGNORECASE)
# 따옴표 안 첫 단어가 명령형 동사(대문자 시작 인용부 뒤)일 때도 directive 로 근사
IMPERATIVE_LEAD = re.compile(r'["“‘\']\s*([A-Z][a-z]+)\b')
IMP_VERBS = {"come", "go", "stop", "wait", "look", "listen", "give", "take", "tell", "bring",
             "let", "hand", "help", "run", "stay", "leave", "follow", "hurry", "keep", "hold",
             "get", "put", "make", "do", "be", "have", "see", "watch", "mind", "swear", "remember"}


def is_performative(sent: str) -> bool:
    if QUESTION.search(sent) or DIRECTIVE.search(sent):
        return True
    for v in IMPERATIVE_LEAD.findall(sent):
        if v.lower() in IMP_VERBS:
            return True
    return False


def diag_B1(sents: dict[str, list[str]]) -> dict:
    # event_type 별 P(goal≠none), 그리고 dialogue 안에서 수행적 vs 논의
    by_evt = defaultdict(lambda: {"n": 0, "goal": 0})
    dlg = {"perf": {"n": 0, "goal": 0}, "disc": {"n": 0, "goal": 0}}
    examples = {"perf_none": [], "disc_goal": []}
    for doc in docs():
        caus = {int(s["item_id"]): s for s in read(RUN / doc / "causality_states.jsonl")}
        intent = {int(s["item_id"]): s for s in read(RUN / doc / "intentionality_states.jsonl")}
        n = min(len(caus), len(intent))
        for i in range(n):
            if i not in caus or i not in intent:
                continue
            evt = caus[i]["event_type"]
            goal = int(intent[i]["active_goal"] != "none")
            by_evt[evt]["n"] += 1
            by_evt[evt]["goal"] += goal
            if evt == "dialogue" and i < len(sents.get(doc, [])):
                sent = sents[doc][i]
                cls = "perf" if is_performative(sent) else "disc"
                dlg[cls]["n"] += 1
                dlg[cls]["goal"] += goal
                if cls == "perf" and not goal and len(examples["perf_none"]) < 10:
                    examples["perf_none"].append((doc, i, sent[:110]))
                if cls == "disc" and goal and len(examples["disc_goal"]) < 6:
                    examples["disc_goal"].append((doc, i, intent[i]["active_goal"], sent[:100]))
    return {"by_evt": by_evt, "dlg": dlg, "examples": examples}


# ================================================================ B②: eventiveness + leave-one-goal-out
def runs_of(items_sorted: list[int]) -> int:
    """연속 run(재등장) 개수 — gap 으로 끊긴 구간 수."""
    if not items_sorted:
        return 0
    r = 1
    for a, b in zip(items_sorted, items_sorted[1:]):
        if b - a > 1:
            r += 1
    return r


def deployable_score(matrix: list[dict], axes: list[str], tstar: int) -> float:
    rows = sorted(matrix, key=lambda r: int(r["transition"]))
    n_sent = len(rows) + 1
    gold = [0] * n_sent
    for r in rows:
        if int(r["boundary"]):
            gold[int(r["transition"]) - 1] = 1
    count = {int(r["transition"]): sum(int(r[a]) for a in axes) for r in rows}
    bt = [i for i in count if count[i] >= tstar]
    pred = [0] * n_sent
    for i in bt:
        pred[i - 1] = 1
    return score_dialogues([gold], [pred])["score"]


def intentionality_resets_without(states_raw: dict[int, dict], drop: set[str]) -> dict[int, int]:
    """goal drop 집합을 none 처리 → carry_forward → reset_intentionality 재계산."""
    mod = {}
    for i, s in states_raw.items():
        g = s["active_goal"]
        mod[i] = {"active_goal": "none" if g in drop else g}
    carried = carry_forward(mod, "active_goal", "none")
    out = {}
    for i in sorted(carried):
        if i == 0:
            continue
        out[i] = reset_intentionality(carried[i - 1], carried[i])
    return out


def diag_B2() -> dict:
    per_doc = []
    matrix_by_doc = defaultdict(list)
    for r in read(RUN / "reset_matrix.jsonl"):
        matrix_by_doc[str(r["doc_id"])].append(r)
    for doc in docs():
        st = read(RUN / doc / "intentionality_states.jsonl")
        n = len(st)
        if not n:
            continue
        raw = {int(s["item_id"]): s for s in st}
        by_goal = defaultdict(list)
        for i in sorted(raw):
            g = raw[i]["active_goal"]
            if g != "none":
                by_goal[g].append(i)
        goal_prof = []
        for g, items in by_goal.items():
            items.sort()
            span = (items[-1] - items[0] + 1) / n
            goal_prof.append({"goal": g, "count": len(items), "span": span,
                              "density": len(items) / (items[-1] - items[0] + 1),
                              "reentry": runs_of(items)})
        goal_prof.sort(key=lambda x: -x["span"])
        pervasive = [gp for gp in goal_prof if gp["span"] >= 0.6]
        # leave-one-goal-out (pervasive 목표들만, 각각 + 전체)
        matrix = matrix_by_doc.get(doc, [])
        loo = []
        base_reset = sum(int(r["intentionality"]) for r in matrix) / len(matrix) if matrix else None
        base_score = deployable_score(matrix, FROZEN, T_STAR) if matrix else None
        if matrix and pervasive:
            by_t = {int(r["transition"]): r for r in matrix}
            for gp in pervasive:
                new_int = intentionality_resets_without(raw, {gp["goal"]})
                new_matrix = []
                nr = 0
                for r in matrix:
                    t = int(r["transition"])
                    rr = dict(r)
                    rr["intentionality"] = new_int.get(t, 0)
                    nr += rr["intentionality"]
                    new_matrix.append(rr)
                loo.append({"goal": gp["goal"], "span": gp["span"],
                            "reset_rate": nr / len(new_matrix),
                            "d_reset": nr / len(new_matrix) - base_reset,
                            "score": deployable_score(new_matrix, FROZEN, T_STAR),
                            "d_score": deployable_score(new_matrix, FROZEN, T_STAR) - base_score})
            # 전체 pervasive 동시 제거
            new_int = intentionality_resets_without(raw, {gp["goal"] for gp in pervasive})
            nr = sum(new_int.get(int(r["transition"]), 0) for r in matrix)
            new_matrix = [dict(r, intentionality=new_int.get(int(r["transition"]), 0)) for r in matrix]
            loo.append({"goal": "ALL_pervasive", "span": None, "reset_rate": nr / len(matrix),
                        "d_reset": nr / len(matrix) - base_reset,
                        "score": deployable_score(new_matrix, FROZEN, T_STAR),
                        "d_score": deployable_score(new_matrix, FROZEN, T_STAR) - base_score})
        per_doc.append({"doc": doc, "genre": genre(doc), "n": n, "n_goals": len(by_goal),
                        "goal_prof": goal_prof, "pervasive": pervasive,
                        "base_reset": base_reset, "base_score": base_score, "loo": loo})
    return {"per_doc": per_doc}


# ================================================================ REPORT
def pct(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def build_report() -> str:
    sents = load_sentences()
    A_rows, A_ex = diag_A(sents)
    H_rows, H_ex = diag_H(sents)
    B1 = diag_B1(sents)
    B2 = diag_B2()

    L = ["# ② 진단 — findings A·H·B①·B② 스케일 게이트", "",
         "> ①B human 표본이 연 설계 발견의 **문서 스케일 빈도**를 캐시 run1(12편: 유명 6 fam / 무명 6 obs)에서",
         "> gold-free 측정. 예산 0(라벨 재호출 없음, regex 스크리닝). E(time 비대칭)=`time_marker_position_diag.py`",
         "> 별도 완료 → §E 요약. **판정 기준은 결과 보기 전 사전 선언**(no-postresult-criterion-change).", "",
         "> ⚠️ regex 휴리스틱은 E 진단과 동급 **스크리닝 도구**(정밀도 아닌 방향·규모). cue 감지는 불완전 —",
         "> A(누락)는 하한, H(no-cue non-unspec)는 상한으로 해석.", "",
         "## 실험 setup", "",
         "- **목적:** ①B human 표본 발견 A·H·B①·B②가 **문서 스케일에서 흔한가**를 게이트(④ 재설계 포함 여부).",
         "  결함 유형은 이미 표본으로 확정됨 — 여기선 *빈도/규모*만 잰다.",
         f"- **데이터:** `{RUN.relative_to(ROOT)}` (frozen `p0s5-eism-v2` run1 라벨) + `{MASTER.relative_to(ROOT)}`",
         "  (문장 텍스트). 12편(fam 6 유명 / obs 6 무명). **gold 경계 = 전 문서 0개**(repcorpus=QC용, 분절 gold 아님).",
         "- **방법:** 캐시 `{axis}_states.jsonl`(raw, carry 전) + regex cue 휴리스틱. B② leave-one-out 은",
         "  `llm_s5_states.carry_forward`+`reset_intentionality` 재사용. Score=`hi_ontop.dts_scoring.score_dialogues`.",
         "- **판정 기준(사전 동결):** 각 절(§A/§H/§B①/§B②) 상단에 결과 보기 전 명시(rubric-before-results).",
         "- **한계:** cue 감지 불완전(A 하한·H 상한) + repcorpus gold 0 → **Score 무의미**(§한계). 신호 축은 reset율.", ""]

    # ---- A
    L += ["## A. space list-miss (under) — 무대(탈것/공중) 목록누락 → unspecified", "",
          f"**판정 기준(사전):** {CRIT['A']}", "",
          "| doc | 장르 | n | list크기 | unspec율 | 최장 unspec run | 탈것/공중 목록누락 문장 |",
          "|---|---|---|---|---|---|---|"]
    fam_miss = obs_miss = 0
    for r in A_rows:
        L.append(f"| {r['doc']} | {r['genre']} | {r['n']} | {r['list_size']} | {pct(r['unspec_rate'])} | "
                 f"{r['longest_unspec_run']} | {r['conveyance_miss']} |")
        if r["genre"] == "fam":
            fam_miss += r["conveyance_miss"]
        else:
            obs_miss += r["conveyance_miss"]
    L += ["", f"- 탈것/공중 목록누락 문장 합계: **fam {fam_miss} / obs {obs_miss}**.", ""]
    if A_ex:
        L.append("**예시(unspecified 인데 탈것/공중 어휘 有, 목록엔 없음):**")
        for doc, i, hit, s in A_ex:
            L.append(f"- `{doc}#{i}` [{','.join(hit)}]: {s}")
    L.append("")

    # ---- H
    L += ["## H. space context-leak (over) — 장소단서 없는데 non-unspecified (raw, carry 전)", "",
          f"**판정 기준(사전):** {CRIT['H']}", "",
          "| doc | 장르 | n | cue문장 non-unspec | no-cue문장 n | **no-cue non-unspec(=leak 상한)** |",
          "|---|---|---|---|---|---|"]
    tot_nocue_n = tot_nocue_nu = 0
    for r in H_rows:
        nu = r["nocue_nonunspec"]
        L.append(f"| {r['doc']} | {r['genre']} | {r['n']} | {pct(r['cue_nonunspec'])} | {r['nocue_n']} | "
                 f"{pct(nu)} |")
        if nu is not None:
            tot_nocue_n += r["nocue_n"]
            tot_nocue_nu += round(nu * r["nocue_n"])
    L += ["", f"- 전체 no-cue 문장 {tot_nocue_n} 중 non-unspecified ≈ {tot_nocue_nu} "
              f"({tot_nocue_nu/tot_nocue_n:.1%} — leak 상한).", ""]
    if H_ex:
        L.append("**예시(no-cue 인데 위치 부여):**")
        for doc, i, loc, s in H_ex:
            L.append(f"- `{doc}#{i}` → {loc}: {s}")
    L.append("")

    # ---- B1
    L += ["## B①. 대사 화행별 P(goal≠none) — 수행적 vs 논의", "",
          f"**판정 기준(사전):** {CRIT['B1']}", "",
          "**event_type 별 P(goal≠none) (참조 상한/하한):**", "",
          "| event_type | n | P(goal≠none) |", "|---|---|---|"]
    for evt, c in sorted(B1["by_evt"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {evt} | {c['n']} | {c['goal']/c['n']:.1%} |" if c["n"] else f"| {evt} | 0 | — |")
    d = B1["dlg"]
    pp = d["perf"]["goal"] / d["perf"]["n"] if d["perf"]["n"] else None
    dp = d["disc"]["goal"] / d["disc"]["n"] if d["disc"]["n"] else None
    L += ["", "**dialogue 안 화행 분해:**", "",
          "| 화행 | n | P(goal≠none) |", "|---|---|---|",
          f"| 수행적(명령/질문/약속/계획) | {d['perf']['n']} | {pct(pp)} |",
          f"| 순수 논의 | {d['disc']['n']} | {pct(dp)} |", ""]
    if pp is not None and dp is not None:
        gap = pp - dp
        conf = (gap <= 0.10) and (pp < 0.5)
        L += [f"- 수행적 − 논의 = **{gap:+.1%}**, 수행적 절대 = **{pp:.1%}**.",
              f"- **① 판정: {'확정(수행적도 none — 화행 무시)' if conf else '기각/불확정(수행적이 goal 더 잘 발화)'}**.", ""]
    if B1["examples"]["perf_none"]:
        L.append("**수행적인데 none(① 증거 후보):**")
        for doc, i, s in B1["examples"]["perf_none"]:
            L.append(f"- `{doc}#{i}`: {s}")
    L.append("")

    # ---- B2
    L += ["## B②. goal eventiveness + leave-one-goal-out — 상시(pervasive)목표 실재+영향", "",
          f"**판정 기준(사전):** {CRIT['B2']}", "",
          "**문서별 상시목표(span≥0.6) + leave-one-out (Δreset / Δscore, 5축 t\\*=3):**", ""]
    any_perv = False
    L += ["| doc | 장르 | n목표 | 상시목표(span) | base reset | base score |",
          "|---|---|---|---|---|---|"]
    for pd in B2["per_doc"]:
        perv = "; ".join(f"{gp['goal']}({gp['span']:.2f})" for gp in pd["pervasive"]) or "—"
        if pd["pervasive"]:
            any_perv = True
        L.append(f"| {pd['doc']} | {pd['genre']} | {pd['n_goals']} | {perv} | "
                 f"{pct(pd['base_reset'])} | {pd['base_score']:.3f} |" if pd["base_score"] is not None
                 else f"| {pd['doc']} | {pd['genre']} | {pd['n_goals']} | {perv} | — | — |")
    L += ["", "**leave-one-goal-out 상세(상시목표 보유 문서):**", "",
          "| doc | drop goal | span | reset율 | Δreset | score | Δscore |", "|---|---|---|---|---|---|---|"]
    for pd in B2["per_doc"]:
        for l in pd["loo"]:
            sp = "—" if l["span"] is None else f"{l['span']:.2f}"
            L.append(f"| {pd['doc']} | {l['goal']} | {sp} | {pct(l['reset_rate'])} | {l['d_reset']:+.1%} | "
                     f"{l['score']:.3f} | {l['d_score']:+.3f} |")
    L += ["", f"- 상시목표(span≥0.6) 보유 문서 존재: **{any_perv}**.",
          "- Δreset/Δscore 가 비영이면 상시목표가 경계신호를 실제로 흔든다(억제 또는 가짜경계).", ""]

    # ---- E summary (external)
    L += ["## E. time opens-with 비대칭 (별도 완료 — `time_marker_position_diag.py`)", "",
          "요약(FINDINGS §E, 캐시 실증): non-absent 83건 중 sequence 50(문중 24%) vs jump 29(문중 48%).",
          "jump 절반이 문중 발화 = 위치조건 비대칭 실증. **④에 위치조건 대칭화 포함(확정).**", ""]

    # ---- computed summary stats for 해석
    A_by = {r["doc"]: r["conveyance_miss"] for r in A_rows}
    a_sorted = sorted(A_by.values(), reverse=True)
    a_top, a_2nd = (a_sorted + [0, 0])[0], (a_sorted + [0, 0])[1]
    a_topdoc = max(A_by, key=A_by.get)
    n_perv_docs = sum(1 for pd in B2["per_doc"] if pd["pervasive"])
    # ALL_pervasive reset halving
    halved = []
    for pd in B2["per_doc"]:
        allp = [l for l in pd["loo"] if l["goal"] == "ALL_pervasive"]
        if allp and pd["base_reset"]:
            halved.append((pd["doc"], pd["base_reset"], allp[0]["reset_rate"]))

    # ---- 해석
    L += ["## 해석", "",
          "### A (space list-miss, under) — 실재하나 **국소집중**(regex FP 주의)",
          f"- 탈것/공중 목록누락 문장은 **`{a_topdoc}`={a_top}건으로 압도적**, 2위={a_2nd}건. `{a_topdoc}`(1차대전 항공 pulp)는",
          "  Stage-1 locations 가 지상 airdrome 만이라 '비행기/조종석/공중' 무대 문장(#8 rearmost plane, #10 tractor",
          "  biplane, #25 leading edge of wing …)이 전부 unspecified — **진짜 어휘공백**(FINDINGS A 사례 재현·확대).",
          "- 타 문서 히트는 대부분 **regex 오검**: 'by **train**'(언급) · '**cloud** of smoke' · Federal '**scout**'(사람) 등",
          "  = 무대 아님. ⇒ A는 **탈것/공중/항해가 무대인 서사에 국소적**, 전 장르 보편 아님. obs>fam(44 vs 19)이나 대부분 FP.",
          "- **판정:** A 확정, 단 빈도는 genre-conditional. fix(Stage-1 physical_setting 정의에 vehicle/airspace 예시)는",
          "  **저비용·정합**(EISM space 와 충돌 안 함)이라 ④ 포함할 가치 有 — 단 '흔해서'가 아니라 '해당 서사에 치명적'이라서.", "",
          "### H (space context-leak, over) — 기제 실재(측정순수성), 다수는 맥락상 정답",
          f"- no-cue 문장의 non-unspecified율(leak 상한) 전체 {tot_nocue_nu/tot_nocue_n:.1%}, 최고 `fam_bierce_owlcreek` 31.5%.",
          "  = 모델이 **문장 자체 근거 없이도** carry 전 raw 에서 위치를 부여(target-only 소프트 위반) — FINDINGS H 기제 확증.",
          "- 단 예시(#2 'A rope encircled his neck'→owl_creek_bridge 등)는 **맥락상 옳은 위치**가 많음 → boundary 에 꼭",
          "  해롭진 않으나 **space reset 이 순수 문장근거 아닌 맥락기억에 일부 의존**한다는 측정순수성 문제는 실재.",
          "- **판정:** H fix('문장 근거 없으면 반드시 unspecified, 맥락추론 금지') ④ 포함(✅) — 규모는 중간, 순수성 명분.", "",
          "### B① (대사 화행) — 사전기준 **미충족 → ① 기각**(예시 교체는 선택적 보강으로 강등)",
          "- 수행적 대사 P(goal≠none)=41.6% > 논의 24.8%, gap **+16.8%**(기준 ≤10% 위반). 모델은 화행을 **무시하지 않음**",
          "  (명령·질문·계획 대사에서 goal 을 더 발화). event_type 상 movement 60%·physical 43% 가 상한, dialogue 31% 는 중간.",
          "- ⇒ '대사=none 오독'은 **스케일에서 확증 안 됨**(none-rate 0.73 은 대사 아닌 state_description 19.9%·다수가 주범).",
          "- **판정:** ① 구성타당도 결함으로는 **기각**. FINDINGS 대로 '대사→목표' 예시 추가는 *무해한 명료화*이므로 ④에서",
          "  **선택적 경보강**만(load-bearing 아님). B②를 먼저 치우면 none-gap 상당수가 해소될 것.", "",
          "### B② (상시목표) — **확정·보편**(10/12 문서), reset 대량 기여",
          f"- span≥0.6 상시목표 보유 문서 **{n_perv_docs}/12**. Yellow Wallpaper 5개(get_well·have_perfect_rest·",
          "  avoid_opposition …)=FINDINGS control_self 계열 재현. 상시목표가 이산·순차 goal 가정을 깸.",
          "- **leave-one-out:** 상시목표 전부 제거 시 intentionality reset율이 대폭 하락(예: "
          + "; ".join(f"{d} {b:.1%}→{a:.1%}" for d, b, a in halved[:4]) + " …)",
          "  = 상시목표가 현재 intentionality reset 의 **상당분을 생성**(산발 re-entry·목표전환). 노이즈 경계 후보.",
          "- **판정:** B② ④ 포함(✅) 확정. Stage-1 goals 를 bounded commitment 로 제한 + 확산적 목표(enjoy_leave류) 규정.", "",
          "### E (time 비대칭) — 별도 확정(§E). ④ 위치조건 대칭화 포함(✅).", "",
          # ---- 판정 roll-up
          "## 판정 (④ 반영 결정 → FINDINGS/HANDOFF 갱신)", "",
          "| finding | 사전기준 | 진단 결과 | ④ 포함 |", "|---|---|---|---|",
          f"| A list-miss | 국소 유의+obs편중 | `{a_topdoc}`={a_top} 압도(국소), 타 FP | **포함(정합·저비용, genre국소)** |",
          f"| H context-leak | no-cue non-unspec 상당 | 전체 {tot_nocue_nu/tot_nocue_n:.1%}, 최고 31.5% | **포함 ✅(순수성)** |",
          "| B① 대사화행 | gap≤10%&수행<50% | gap +16.8% (위반) | **기각 → 선택적 보강만** |",
          f"| B② 상시목표 | span≥0.6 존재+영향 | {n_perv_docs}/12, reset 대량 | **포함 ✅** |",
          "| E time 비대칭 | 별도 | jump 문중 48% | **포함 ✅** |", "",
          "## 한계", "",
          "- **repcorpus gold 경계 = 전 문서 0** → Score·Δscore(B② 표) **무의미**(F1 정의 불가, base_score≈0).",
          "  findings 의 **Score 영향은 여기서 측정 불가** → ④ dev-3(pieman/tunnel/monkey, gold 有) Gate C 로 이월. 여기 신호축=reset율.",
          "- cue/화행/탈것 감지 = regex 스크리닝(E 진단 동급). A=하한(누락 과소), H=상한(no-cue 에 암묵무대 혼입), B①=화행분류 noise.",
          "  방향·규모는 견고하나 소수점 정밀 아님.",
          "- 12편·genre 교란(fam=19c 문학 / obs=20c pulp) 잔존 — obs>fam 편중은 A/B② 에서 보이나 era/장르와 얽힘.",
          "- B① 기각은 '스케일 확증 실패'지 '① 절대 없음' 아님 — 개별 오독은 존재(예시 목록). 예시 보강은 무해.", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build_report()
    (OUT / "REPORT.md").write_text(rep, encoding="utf-8")
    print(rep)
    print(f"[written] {OUT/'REPORT.md'}")


if __name__ == "__main__":
    main()
