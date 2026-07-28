#!/usr/bin/env python3
"""①B human 표본 → 한국어 참조 번역 (라벨링 이해용, 측정과 무관).

두 산출물 (둘 다 값싼 non-reasoning 모델, resumable 캐시):
  translations.json  {row_id: ko}                     — 100 문장 번역 (영어 아래 표시)
  list_gloss.json    {identifier: {ko, desc}}          — story_lists 의 cast/locations/goals 식별자를
                                                         **그 이야기 맥락**으로 번역한 한국어 라벨 + 한 줄 설명
                                                         (butte=지명, meet_the_stage=역마차 마중 등 맥락 의존 해소)

번역/글로스는 사람이 영어를 이해하도록 돕는 보조일 뿐 — 모델 라벨을 노출하지 않으므로 블라인드 불변.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

SYS_SENT = ("Translate each numbered English narrative sentence into natural, faithful Korean. "
            "Do not merge or split. Return ONLY a JSON object {\"t\":{\"<idx>\":\"<korean>\"}} "
            "with exactly the given keys.")

SYS_GLOSS = (
    "You are glossing identifiers extracted from ONE early-1900s American short story, using the excerpt "
    "for context. For EACH identifier return a Korean label and a one-line Korean explanation grounded in "
    "THIS story. Disambiguate with the context: place names stay place names (Butte, Too Dry, Ford = 지명), "
    "and period/regional terms by their real sense (a 'stage' is usually a 역마차/stagecoach; a 'nester' is a "
    "개척 정착민/homesteader; 'reservation' = 인디언 보호구역; 'squaw' = 원주민 여성). "
    "desc must be ONE short line (≤45 chars), no trailing period needed. Return ONLY a JSON object "
    "{\"g\":{\"<id>\":{\"ko\":\"<label>\",\"desc\":\"<one line>\"}}} with exactly the given ids as keys.")


def sequence(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda x: int(x["transition"]))
    return [rows[0]["previous"], *[x["current"] for x in rows]]


def make_client():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                  timeout=120, max_retries=2,
                  default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                                   "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]})


def sentence_phase(c, model, rows, batch, out_path):
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"[문장] {len(done)} cached · {len(todo)} to do")
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        payload = "\n".join(f"[{i}] {r['sentence']}" for i, r in enumerate(chunk))
        want = {str(i) for i in range(len(chunk))}
        for attempt in range(3):
            r = c.chat.completions.create(
                model=model, temperature=0, max_tokens=4000, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS_SENT}, {"role": "user", "content": payload}])
            obj = json.loads(r.choices[0].message.content).get("t", {})
            if isinstance(obj, dict) and want <= set(obj):
                for i, row in enumerate(chunk):
                    done[row["row_id"]] = str(obj[str(i)]).strip()
                break
            print(f"  [문장] align retry {attempt+1}/3 (@{s})", flush=True)
        else:
            raise RuntimeError(f"[문장] batch @{s}: alignment failed")
        out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [문장] {min(s+batch, len(todo))}/{len(todo)}", flush=True)


def gloss_phase(c, model, rows, run_dir, master, out_path, ctx_sents=28):
    # doc -> sentences (for context)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in (json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()):
        by_doc[str(r["doc_id"])].append(r)
    sents = {d: sequence(rs) for d, rs in by_doc.items()}

    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    done = {k: v for k, v in done.items() if isinstance(v, dict) and "desc" in v}  # drop old string schema
    docs = sorted({r["doc_id"] for r in rows})
    for doc in docs:
        lp = run_dir / doc / "story_lists.json"
        if not lp.exists():
            continue
        L = json.loads(lp.read_text())["lists"]
        groups = {"cast": sorted(set(L.get("cast", []))),
                  "locations": sorted(set(L.get("locations", []))),
                  "goals": sorted(set(L.get("goals", [])))}
        ids = [i for g in groups.values() for i in g]
        need = [i for i in ids if i not in done]
        if not need:
            print(f"[글로스] {doc}: 전부 캐시")
            continue
        excerpt = " ".join(sents.get(doc, [])[:ctx_sents])
        payload = (f"STORY EXCERPT:\n{excerpt}\n\nIDENTIFIER GROUPS:\n"
                   f"cast: {', '.join(groups['cast']) or '(none)'}\n"
                   f"locations: {', '.join(groups['locations']) or '(none)'}\n"
                   f"goals: {', '.join(groups['goals']) or '(none)'}\n\n"
                   f"REQUIRED KEYS (gloss every one):\n" + "\n".join(need))
        for attempt in range(3):
            r = c.chat.completions.create(
                model=model, temperature=0, max_tokens=4000, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS_GLOSS}, {"role": "user", "content": payload}])
            obj = json.loads(r.choices[0].message.content).get("g", {})
            if isinstance(obj, dict) and set(need) <= set(obj) and all(
                    isinstance(obj[i], dict) and obj[i].get("ko") and obj[i].get("desc") for i in need):
                for i in need:
                    done[i] = {"ko": str(obj[i]["ko"]).strip(), "desc": str(obj[i]["desc"]).strip()}
                break
            print(f"  [글로스] {doc} align retry {attempt+1}/3", flush=True)
        else:
            raise RuntimeError(f"[글로스] {doc}: alignment failed")
        out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [글로스] {doc}: +{len(need)} (총 {len(done)})", flush=True)
    return done


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    base = root / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample"
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=base / "sheet.jsonl")
    ap.add_argument("--out", type=Path, default=base / "translations.json")
    ap.add_argument("--gloss-out", type=Path, default=base / "list_gloss.json")
    ap.add_argument("--run-dir", type=Path,
                    default=root / "outputs/eacl_phase0/repcorpus_s5_eismv2_run1")
    ap.add_argument("--master", type=Path,
                    default=root / "outputs/eacl_phase0/repcorpus_master.jsonl")
    ap.add_argument("--model", default="openrouter/openai/gpt-4.1-mini")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    c = make_client()
    rows = [json.loads(x) for x in args.sheet.read_text(encoding="utf-8").splitlines() if x.strip()]
    sentence_phase(c, args.model, rows, args.batch, args.out)
    gloss = gloss_phase(c, args.model, rows, args.run_dir, args.master, args.gloss_out)
    n_sent = len(json.loads(args.out.read_text()))
    print(f"-> {args.out} ({n_sent}) · {args.gloss_out} ({len(gloss)})")


if __name__ == "__main__":
    main()
