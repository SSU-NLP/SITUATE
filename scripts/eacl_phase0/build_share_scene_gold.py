"""SHARE 세션↔대본 씬 정렬 → place/time gold 구성 → M3 reset 대조 (HANDOFF_07).

핵심 질문: SHARE place/time null이 "안 바뀜"(추출기 정답)인가 "바뀌는데 대사에 안 보임"
(측정 실패)인가? 원 대본 슬러그라인으로 세션 경계별 place/time **진짜 변경 여부** gold를
만들고, M3(코드판정) place/time reset이 그 진짜변경을 얼마나 잡는지(recall)와 무변경에서
조용한지(침묵률)를 측정한다.

usage: uv run python scripts/eacl_phase0/build_share_scene_gold.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TIME_TOKENS = {"day", "night", "morning", "evening", "dawn", "dusk", "afternoon",
               "later", "continuous", "moments", "sunset", "sunrise", "midnight",
               "noon", "nighttime", "daytime", "pre-dawn"}
SLUG_RE = re.compile(r"\b(INT\.|EXT\.|INT/EXT\.|INT\./EXT\.)\s*(.+)", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def parse_slug(raw: str) -> tuple[str, str]:
    """슬러그라인 → (location_norm, time_token)."""
    s = raw.strip()
    s = re.sub(r"^\d+\s*", "", s)          # 앞 scene number
    s = re.sub(r"\s*\d+\s*$", "", s)       # 뒤 scene number
    s = SLUG_RE.sub(lambda m: m.group(2), s)
    parts = [p.strip() for p in re.split(r"\s+-\s+|\s+--\s+", s) if p.strip()]
    time_tok = ""
    loc_parts = []
    for p in parts:
        pl = p.lower()
        if any(tk in pl for tk in TIME_TOKENS) and len(pl) <= 16:
            time_tok = next((tk for tk in TIME_TOKENS if tk in pl), pl)
        else:
            loc_parts.append(p)
    loc = norm(re.sub(r"[^\w\s]", " ", " ".join(loc_parts)))
    return loc, time_tok


def load_scenes(script_txt: str) -> list[tuple[int, str, str]]:
    lines = script_txt.split("\n")
    out = []
    for i, l in enumerate(lines):
        if SLUG_RE.search(l) and len(l.strip()) < 75:
            loc, tt = parse_slug(l)
            if loc:
                out.append((i, loc, tt))
    return out, [norm(l) for l in lines]


def scene_at(scenes, line_idx):
    best = None
    for (i, loc, tt) in scenes:
        if i <= line_idx:
            best = (loc, tt)
        else:
            break
    return best


def align_session_scene(sess_turns, script_lines_norm, scenes):
    """세션 대사들을 대본에서 찾아 다수결 씬."""
    from collections import Counter
    hits = []
    for t in sess_turns:
        frag = norm(t.get("text", ""))[:40]
        if len(frag) < 12:
            continue
        key = frag[:30]
        for li, ln in enumerate(script_lines_norm):
            if key in ln:
                sc = scene_at(scenes, li)
                if sc:
                    hits.append(sc)
                break
    if not hits:
        return None, 0
    c = Counter(hits)
    return c.most_common(1)[0][0], len(hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=Path, default=REPO / "benchmarks/SHARE/datasets/valid.json")
    ap.add_argument("--scripts", type=Path, default=REPO / "data/eacl_phase0/share_scripts")
    ap.add_argument("--master", type=Path,
                    default=REPO / "outputs/eacl_phase0/share_valid_probe30_master.jsonl")
    ap.add_argument("--reset-matrix", type=Path,
                    default=REPO / "outputs/eacl_phase0/share_valid_probe30_s5/reset_matrix.jsonl")
    args = ap.parse_args()

    def slug(s):
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    data = json.load(open(args.valid))
    master = [json.loads(l) for l in open(args.master)]
    rm = {(r["doc_id"], r["transition"]): r for r in (json.loads(l) for l in open(args.reset_matrix))}
    m_by_doc: dict[str, list[dict]] = {}
    for r in master:
        m_by_doc.setdefault(r["doc_id"], []).append(r)
    probe_docs = set(m_by_doc)

    # doc_id -> conv (build_share_pack와 동일 규칙)
    seen: set[str] = set()
    doc_to_conv: dict[str, dict] = {}
    for pair_key, conv in data.items():
        did = f"{slug(conv.get('movie','unknown'))}__{slug(re.sub(chr(39)+'|[(),]','',pair_key))}"
        if did in seen:
            k = 2
            while f"{did}_{k}" in seen:
                k += 1
            did = f"{did}_{k}"
        seen.add(did)
        if did in probe_docs:
            doc_to_conv[did] = conv

    # 대본 캐시
    scripts: dict[str, tuple] = {}
    for p in args.scripts.glob("*.txt"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        scenes, lnorm = load_scenes(txt)
        if scenes:
            scripts[p.stem] = (scenes, lnorm)

    # 집계
    tot_bnd = 0; convs_used = 0; docs_no_rm = 0
    gp1_m3hit = gp1 = 0     # gold place-change 경계: M3 place reset 켜진 수
    gp0_m3sil = gp0 = 0     # gold no-change 경계: M3 place reset 꺼진 수(침묵)
    gt1_m3hit = gt1 = 0
    gt0_m3sil = gt0 = 0
    align_fail = 0
    per_movie = {}
    for did, conv in sorted(doc_to_conv.items()):
        ms = did.split("__")[0]
        if ms not in scripts:
            continue
        if (did, 1) not in rm:
            docs_no_rm += 1; continue
        scenes, lnorm = scripts[ms]
        # 세션별 씬 정렬
        sess_scene = {}
        for s in conv["dialogue"]:
            sc, nh = align_session_scene(s["dialogues"], lnorm, scenes)
            sess_scene[s["session"]] = sc
        rows = sorted(m_by_doc[did], key=lambda r: r["transition"])
        convs_used += 1
        mv = per_movie.setdefault(ms, {"bnd": 0, "gp1": 0, "gp1hit": 0, "gt1": 0, "gt1hit": 0})
        for r in rows:
            if not r["boundary"]:
                continue
            ps, cs = r["prev_session"], r["cur_session"]
            a, b = sess_scene.get(ps), sess_scene.get(cs)
            if a is None or b is None:
                align_fail += 1; continue
            tot_bnd += 1; mv["bnd"] += 1
            m3 = rm[(did, r["transition"])]
            # place
            gp = int(a[0] != b[0])
            if gp:
                gp1 += 1; mv["gp1"] += 1
                gp1_m3hit += int(m3["place"] == 1); mv["gp1hit"] += int(m3["place"] == 1)
            else:
                gp0 += 1; gp0_m3sil += int(m3["place"] == 0)
            # time
            gt = int(a[1] != b[1] and bool(a[1]) and bool(b[1]))
            if gt:
                gt1 += 1; mv["gt1"] += 1
                gt1_m3hit += int(m3["time"] == 1); mv["gt1hit"] += int(m3["time"] == 1)
            else:
                gt0 += 1; gt0_m3sil += int(m3["time"] == 0)

    print(f"\n=== SHARE 씬-gold 대조 (HANDOFF_07) ===")
    print(f"대본 有 & reset_matrix 有 conv: {convs_used}  (정렬실패 경계 {align_fail}개 제외)")
    print(f"평가된 세션 경계: {tot_bnd}")
    print(f"\n[진짜 변경 base rate]")
    print(f"  place 변경 경계: {gp1}/{tot_bnd} = {100*gp1/max(1,tot_bnd):.0f}%  |  무변경: {gp0}")
    print(f"  time  변경 경계: {gt1}/{tot_bnd} = {100*gt1/max(1,tot_bnd):.0f}%  |  무변경: {gt0}")
    print(f"\n[M3 place reset 대조]")
    print(f"  진짜 place 변경 경계에서 M3 잡은 비율(recall): {gp1_m3hit}/{gp1} = "
          f"{100*gp1_m3hit/max(1,gp1):.0f}%")
    print(f"  place 무변경 경계에서 M3 침묵 비율(specificity): {gp0_m3sil}/{gp0} = "
          f"{100*gp0_m3sil/max(1,gp0):.0f}%")
    print(f"\n[M3 time reset 대조]")
    print(f"  진짜 time 변경 경계에서 M3 잡은 비율(recall): {gt1_m3hit}/{gt1} = "
          f"{100*gt1_m3hit/max(1,gt1):.0f}%")
    print(f"  time 무변경 경계에서 M3 침묵 비율(specificity): {gt0_m3sil}/{gt0} = "
          f"{100*gt0_m3sil/max(1,gt0):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
