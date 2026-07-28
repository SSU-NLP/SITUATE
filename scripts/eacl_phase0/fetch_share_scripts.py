"""SHARE probe 영화들의 원 대본을 IMSDB에서 확보 (HANDOFF_07).

각 영화의 movie 필드 → IMSDB 제목 변형 후보들을 시도해 다운로드하고, **그 영화의 SHARE
대사가 실제로 대본에 있는지 sanity-match**로 검증한 것만 캐시한다(오답 대본 방지).

출력: data/eacl_phase0/share_scripts/<movie_slug>.txt (+ _coverage.json)

usage: uv run python scripts/eacl_phase0/fetch_share_scripts.py \
         --probe-master outputs/eacl_phase0/share_valid_probe30_master.jsonl
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def imsdb_candidates(movie: str) -> list[str]:
    """movie 필드 → IMSDB 제목 URL 후보들."""
    m = movie.strip()
    cands: list[str] = []
    def hyph(t: str) -> str:
        return re.sub(r"\s+", "-", re.sub(r"[^\w\s,&-]", "", t).strip())
    cands.append(hyph(m))
    if m.lower().startswith("the "):
        cands.append(hyph(m[4:] + ", The"))
    if m.lower().startswith("a "):
        cands.append(hyph(m[2:] + ", A"))
    # snake→title 재구성(파싱 실패 대비)
    cands.append(hyph(m.replace("_", " ").title()))
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out


def fetch(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception:
        return None


def extract_script(page: str) -> str:
    m = re.search(r"<pre>(.*)</pre>", page, re.S | re.I)
    body = m.group(1) if m else page
    return html.unescape(re.sub(r"<[^>]+>", "", body))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-master", type=Path,
                    default=REPO / "outputs/eacl_phase0/share_valid_probe30_master.jsonl")
    ap.add_argument("--valid", type=Path, default=REPO / "benchmarks/SHARE/datasets/valid.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/eacl_phase0/share_scripts")
    ap.add_argument("--min-match", type=float, default=0.15,
                    help="대본 sanity: SHARE 대사 중 이 비율 이상이 대본에 있어야 채택")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    probe_docs = {r["doc_id"] for r in (json.loads(l) for l in open(args.probe_master))}
    data = json.load(open(args.valid))
    # movie_slug -> {movie, dialogue lines(sanity용)}
    movies: dict[str, dict] = {}
    seen: set[str] = set()
    for pair_key, conv in data.items():
        movie = conv.get("movie", "unknown")
        pair = re.sub(r"[()',]", "", pair_key)
        did = f"{slug(movie)}__{slug(pair)}"
        if did in seen:
            k = 2
            while f"{did}_{k}" in seen:
                k += 1
            did = f"{did}_{k}"
        seen.add(did)
        if did not in probe_docs:
            continue
        ms = slug(movie)
        movies.setdefault(ms, {"movie": movie, "lines": []})
        for s in conv["dialogue"]:
            for t in s["dialogues"]:
                tx = (t.get("text") or "").strip()
                if len(tx) >= 20:
                    movies[ms]["lines"].append(norm(tx)[:40])

    cov = {}
    for ms, info in sorted(movies.items()):
        dest = args.out_dir / f"{ms}.txt"
        if dest.exists():
            cov[ms] = {"movie": info["movie"], "status": "cached"}
            print(f"[{ms}] cached"); continue
        best = None
        for cand in imsdb_candidates(info["movie"]):
            url = f"https://imsdb.com/scripts/{cand}.html"
            page = fetch(url)
            time.sleep(0.5)
            if not page:
                continue
            txt = extract_script(page)
            if len(txt) < 8000:
                continue
            tnorm = norm(txt)
            hit = sum(1 for ln in info["lines"] if ln and ln[:30] in tnorm)
            frac = hit / max(1, len(info["lines"]))
            if best is None or frac > best[1]:
                best = (txt, frac, url, cand)
            if frac >= args.min_match:
                break
        if best and best[1] >= args.min_match:
            dest.write_text(best[0], encoding="utf-8")
            n_slug = len(re.findall(r"\b(INT\.|EXT\.|INT/EXT\.)", best[0]))
            cov[ms] = {"movie": info["movie"], "status": "ok", "match": round(best[1], 3),
                       "url": best[2], "n_slug": n_slug}
            print(f"[{ms}] OK match={best[1]:.2f} slugs={n_slug}  ({best[3]})")
        else:
            bf = round(best[1], 3) if best else 0.0
            cov[ms] = {"movie": info["movie"], "status": "fail", "best_match": bf}
            print(f"[{ms}] FAIL best_match={bf}  ({info['movie']})")

    (args.out_dir / "_coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=2))
    ok = sum(1 for v in cov.values() if v["status"] in ("ok", "cached"))
    print(f"\n=== 커버리지: {ok}/{len(cov)} 영화 대본 확보 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
