#!/usr/bin/env python3
"""M5 Phase 2 — assemble the slot-alignment dataset from existing M3 (S5) anchor labels.

Per doc, produce: frozen mpnet sentence embeddings + per-transition 5-axis RESET labels (primary
anchor) + per-sentence STATE enums (aux anchor). Reuses the S5 outputs already on disk — no LLM,
no new corpus. Fiction docs (narrative-fiction scope): Pieman/Tunnel/Monkey (michelmann) + Tier2.

Anchor labels (codex 2026-07-15 design):
  primary  = reset_matrix.jsonl per-axis 0/1 (protagonist/place/time/causality/goal)  -> L_delta
  aux      = per-sentence global-enum states (perspective/tense/temporal_shift/event_type/aspect) -> L_state
             story-local (location/active_goal/characters) stored raw -> same/diff pairwise aux at train time

Output: outputs/experiments/m5_phase2_slotalign/slot_dataset.npz  (+ _local.json for story-local states)
Usage: uv run python scripts/eacl_phase0/build_slot_dataset.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/experiments/m5_phase2_slotalign"
AXES = ["protagonist", "space", "time", "causality", "intentionality"]
# global fixed enums (aux absolute-state targets); story-local axes handled separately
PERSPECTIVE = ["first", "second", "third", "none"]
TENSE = ["past", "present", "future", "none"]
TEMPORAL = ["absent", "sequence", "jump", "simultaneity"]
EVENT_TYPE = ["dialogue", "physical_action", "movement_or_travel", "perception_or_observation",
              "thought_or_emotion", "transaction_or_exchange", "conflict_or_violence",
              "work_or_creation", "state_description", "other"]
ASPECT = ["inceptive", "ongoing", "completive"]
ENUMS = {"perspective": PERSPECTIVE, "tense": TENSE, "temporal_shift_marker": TEMPORAL,
         "event_type": EVENT_TYPE, "aspect": ASPECT}

# doc -> (master.jsonl, s5_dir with {doc}/{axis}_states.jsonl, reset_matrix.jsonl)
BASE = REPO / "outputs/eacl_phase0"
SCENE_ALIGN = BASE / "share_turn_scene_align.jsonl"
# 값 = (master.jsonl, s5_doc_dir, reset_matrix.jsonl, doc_id or None)
# doc_id=None -> master 파일에 문서 1개뿐이므로 첫 행에서 추론(fiction). SHARE는 한 파일에 30문서라 명시 필수.
FICTION = {}
for d in ["pieman", "tunnel", "monkey"]:
    FICTION[f"{d}_michelmann"] = (BASE / f"{d}_michelmann_master.jsonl",
                                  BASE / f"{d}_michelmann_s5" / d,
                                  BASE / f"{d}_michelmann_s5/reset_matrix.jsonl", None)
TIER2_DOCS = ["cop_and_anthem", "the_furnished_room", "ionitch", "the_lady_with_the_dog",
              "head_of_the_family", "a_man_and_some_others", "the_wise_men"]
for d in TIER2_DOCS:
    FICTION[d] = (BASE / f"tier2/{d}_master.jsonl", BASE / f"tier2_s5/{d}",
                  BASE / "tier2_s5/reset_matrix.jsonl", None)


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def share_sources() -> dict:
    """SHARE 스크린플레이 문서 (D-M8 §990: place/time만 슬러그라인 외부 라벨, 3축은 S5 silver).

    대상 = build_share_turn_scene.py가 정렬에 성공한 문서만.
    """
    if not SCENE_ALIGN.exists():
        raise SystemExit(f"씬 정렬 파일 없음: {SCENE_ALIGN}\n"
                         f"  먼저: uv run python scripts/eacl_phase0/build_share_turn_scene.py")
    docs = sorted({r["doc_id"] for r in read(SCENE_ALIGN)})
    return {f"share__{d}": (BASE / "share_valid_probe30_master.jsonl",
                            BASE / "share_valid_probe30_s5" / d,
                            BASE / "share_valid_probe30_s5/reset_matrix.jsonl", d)
            for d in docs}


def load_scene_gold() -> dict:
    """(doc_id, transition) -> {gold_place, gold_time, place_mask, time_mask}."""
    return {(r["doc_id"], int(r["transition"])): r for r in read(SCENE_ALIGN)}


def sentences_of(master_rows):
    rows = sorted(master_rows, key=lambda r: int(r["transition"]))
    return [rows[0]["previous"], *[r["current"] for r in rows]], rows


def load_states(s5_doc_dir, axis):
    d = {int(x["item_id"]): x for x in read(s5_doc_dir / f"{axis}_states.jsonl")}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["fiction", "share", "both"], default="both")
    ap.add_argument("--out", default="slot_dataset_all.npz")
    ap.add_argument("--share-silver", action="store_true",
                    help="SHARE place/time에 슬러그라인 gold 대신 M3 silver 사용 (both_silver 통제군)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / args.out
    # 🔒 Phase 2 REPORT의 근거인 기존 데이터셋 보호
    if out_path.name == "slot_dataset.npz" and out_path.exists():
        raise SystemExit(f"거부: {out_path} 는 기존 Phase 2 REPORT의 근거다. --out 으로 다른 이름을 쓸 것.")

    sources = {}
    if args.corpus in ("fiction", "both"):
        sources.update({k: (v, "fiction") for k, v in FICTION.items()})
    if args.corpus in ("share", "both"):
        sources.update({k: (v, "share") for k, v in share_sources().items()})
    scene = load_scene_gold() if args.corpus in ("share", "both") else {}

    sys.path.insert(0, str(REPO / "src"))
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cuda")

    embs, resets, gstates, doc_ids, offsets = [], [], {k: [] for k in ENUMS}, [], [0]
    masks, corpora = [], []
    local = []  # per-sentence story-local states {doc, sent, location, active_goal, characters}
    for doc, ((mp, s5dir, rmp, did), corpus) in sources.items():
        mrows = read(mp)
        if did is not None:                       # SHARE: 한 파일에 30문서 -> 반드시 필터
            mrows = [r for r in mrows if r["doc_id"] == did]
        doc_id = did if did is not None else mrows[0]["doc_id"]
        sents, rows = sentences_of(mrows)
        n = len(sents)
        rm = {int(r["transition"]): r for r in read(rmp) if r["doc_id"] == doc_id}
        # per-transition reset labels (5 axes); transition t in 1..n-1
        R = np.array([[int(rm[t][a]) for a in AXES] for t in range(1, n)], dtype=np.int8)
        M = np.ones((n - 1, 5), dtype=np.float32)
        if corpus == "share" and not args.share_silver:
            # D-M8 §990: space/time 2축만 슬러그라인 외부 gold로 치환. 3축은 모달리티 내 S5 silver 유지.
            for t in range(1, n):
                g = scene.get((doc_id, t))
                if g is None:
                    M[t - 1, 1] = M[t - 1, 2] = 0.0
                    continue
                R[t - 1, 1] = g["gold_place"]
                R[t - 1, 2] = g["gold_time"]
                M[t - 1, 1] = float(g["place_mask"])
                M[t - 1, 2] = float(g["time_mask"])
        # per-sentence states
        st = {axis: load_states(s5dir, axis) for axis in AXES}
        persp = st["protagonist"]; space = st["space"]; time = st["time"]; caus = st["causality"]; intent = st["intentionality"]
        for i in range(n):
            gstates["perspective"].append(PERSPECTIVE.index(persp[i]["perspective"]))
            gstates["tense"].append(TENSE.index(time[i]["tense"]))
            gstates["temporal_shift_marker"].append(TEMPORAL.index(time[i]["temporal_shift_marker"]))
            gstates["event_type"].append(EVENT_TYPE.index(caus[i]["event_type"]))
            gstates["aspect"].append(ASPECT.index(caus[i]["aspect"]))
            local.append({"doc": doc, "sent": i, "location": space[i]["location"],
                          "active_goal": intent[i]["active_goal"],
                          "characters": sorted(persp[i].get("characters_present", []))})
        E = np.asarray(enc.encode(sents, normalize_embeddings=True, batch_size=128), dtype=np.float32)
        embs.append(E); resets.append(R); masks.append(M)
        doc_ids += [doc] * n; corpora += [corpus] * n; offsets.append(offsets[-1] + n)
        print(f"{doc:42s} [{corpus:7s}] {n:4d} sents / {len(R):4d} trans  "
              f"reset-rate {R.mean(0).round(2)}  masked {int((M == 0).any(1).sum()):3d}")

    np.savez(out_path,
             emb=np.concatenate(embs), doc_offsets=np.array(offsets),
             doc_ids=np.array(doc_ids), corpus=np.array(corpora),
             resets=np.concatenate(resets),  # (sum(n-1), 5) — indexed per-transition; use offsets-per-doc
             mask=np.concatenate(masks),     # (sum(n-1), 5) — 0 = 라벨 유보(손실에서 제외)
             reset_offsets=np.array([0] + list(np.cumsum([len(r) for r in resets]))),
             **{f"state_{k}": np.array(v, dtype=np.int16) for k, v in gstates.items()})
    (OUT / f"{out_path.stem}_local.json").write_text(json.dumps(local, ensure_ascii=False))
    tot_s = offsets[-1]; tot_t = sum(len(r) for r in resets)
    R_all = np.concatenate(resets); M_all = np.concatenate(masks)
    print(f"\n{len(sources)} docs · {tot_s} sentences · {tot_t} transitions -> {out_path}")
    for i, a in enumerate(AXES):
        m = M_all[:, i] > 0
        print(f"  {a:12s} pos {int(R_all[m, i].sum()):4d} / {int(m.sum()):4d} = "
              f"{R_all[m, i].mean():.3f}")


if __name__ == "__main__":
    main()
