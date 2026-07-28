#!/usr/bin/env python3
"""SHARE 턴 단위 씬 정렬 → per-transition place/time 외부 gold (슬러그라인).

D-M8 §990 사양: 스크린플레이 데이터에서 **place/time 2축만 슬러그라인(씬 헤딩) 외부 라벨**로
공급하고 protagonist/causality/goal 3축은 모달리티 내 M3(S5) silver 유지. 이 스크립트는 그
place/time 라벨을 만든다. 절차 원본 = HANDOFF_07 §2.0 (SHARE 대사 → 원 대본 씬 → 슬러그라인).

기존 build_share_scene_gold.py는 **세션 경계**(92개)에서만 gold를 냈다. anchor 학습은
**전이 단위**(~970개) 라벨이 필요하므로 정렬을 턴 수준으로 내린다:

  turn_candidates  각 턴 텍스트 앞 30자를 대본 라인에서 부분문자열 검색 (전체 후보 반환)
  monotone_align   후보 중 **라인 인덱스가 비감소**인 최장 경로를 LIS-DP로 선택 (오매칭 제거)
  fill_scenes      미매칭 턴: 좌/우 앵커 씬이 같으면 그 씬(씬 내 불변), 다르면 mask=0
  turn_labels      place = loc 변경, time = time-of-day 토큰 변경 (양쪽 non-empty)
  validate_*       회귀 테스트 — 턴 라벨을 세션으로 재집계해 D-M7 수치(place 76, time 38±) 재현

usage: uv run python scripts/eacl_phase0/build_share_turn_scene.py [--validate]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 원본 미수정 — 파싱 규칙을 그대로 재사용해 D-M7과의 정합 보장
from build_share_scene_gold import load_scenes, norm, scene_at  # noqa: E402

MIN_FRAG = 12   # build_share_scene_gold.align_session_scene와 동일 게이트
KEY_LEN = 30    # 동일


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def turns_of(mrows: list[dict]) -> tuple[list[str], list[int], list[dict]]:
    """master 행 -> (턴 텍스트, 턴별 세션 번호, 정렬된 행).

    transition t <-> (turn t-1, turn t) 이므로 턴 수 = 전이 수 + 1.
    """
    rows = sorted(mrows, key=lambda r: int(r["transition"]))
    texts = [rows[0]["previous"], *[r["current"] for r in rows]]
    sess = [rows[0]["prev_session"], *[r["cur_session"] for r in rows]]
    return texts, sess, rows


def turn_candidates(texts: list[str], lines_norm: list[str]) -> list[list[int]]:
    """턴별 매칭 라인 인덱스 후보 전체. 매칭 규칙은 기존 align_session_scene과 동일."""
    out = []
    for t in texts:
        frag = norm(t)[:40]
        if len(frag) < MIN_FRAG:
            out.append([])
            continue
        key = frag[:KEY_LEN]
        out.append([li for li, ln in enumerate(lines_norm) if key in ln])
    return out


def monotone_align(cands: list[list[int]]) -> list[int | None]:
    """턴당 최대 1개, 라인 인덱스가 **비감소**인 최장 부분수열 (patience sorting, O(M log M)).

    대사는 대본에 순서대로 등장하므로 단조성이 참이다. 첫 히트만 쓰면 흔한 짧은 대사가 엉뚱한
    라인에 붙어 씬이 튄다(실측: bodyguard 비단조). 같은 턴의 후보를 **내림차순**으로 넣으면
    한 턴에서 두 개가 같은 경로에 들어갈 수 없다.
    """
    tails: list[int] = []        # tails[k] = 길이 k+1 경로의 최소 말단 라인 인덱스
    tails_ptr: list[int] = []    # 그 말단의 items 인덱스
    items: list[tuple[int, int, int]] = []   # (turn, line, parent_item_idx)
    for t, cs in enumerate(cands):
        for line in sorted(cs, reverse=True):
            k = bisect.bisect_right(tails, line)
            parent = tails_ptr[k - 1] if k > 0 else -1
            items.append((t, line, parent))
            idx = len(items) - 1
            if k == len(tails):
                tails.append(line)
                tails_ptr.append(idx)
            else:
                tails[k] = line
                tails_ptr[k] = idx
    out: list[int | None] = [None] * len(cands)
    if not tails:
        return out
    i = tails_ptr[-1]
    while i != -1:
        t, line, parent = items[i]
        out[t] = line
        i = parent
    return out


def fill_scenes(scene_per_turn: list[tuple | None]) -> list[tuple | None]:
    """미매칭 턴 보간. 씬 내에서 place/time이 불변(piecewise-constant)이라는 구조를 이용.

    좌/우 최근접 앵커의 씬이 **같으면** 그 사이 턴은 반드시 같은 씬 -> 채움.
    **다르면** 씬 전환 지점을 특정할 수 없음 -> None (라벨 유보; mask=0).
    """
    n = len(scene_per_turn)
    left: list[tuple | None] = [None] * n
    right: list[tuple | None] = [None] * n
    last = None
    for i in range(n):
        if scene_per_turn[i] is not None:
            last = scene_per_turn[i]
        left[i] = last
    nxt = None
    for i in range(n - 1, -1, -1):
        if scene_per_turn[i] is not None:
            nxt = scene_per_turn[i]
        right[i] = nxt
    filled = list(scene_per_turn)
    for i in range(n):
        if filled[i] is not None:
            continue
        l, r = left[i], right[i]
        if l is not None and r is not None:
            filled[i] = l if l == r else None
        elif l is not None:
            filled[i] = l        # 꼬리 외삽
        elif r is not None:
            filled[i] = r        # 머리 외삽
    return filled


def turn_labels(doc_id: str, filled: list[tuple | None], rows: list[dict]) -> list[dict]:
    """전이 t = (turn t-1, turn t). place/time gold + mask."""
    out = []
    for r in rows:
        t = int(r["transition"])
        a, b = filled[t - 1], filled[t]
        ok = int(a is not None and b is not None)
        out.append({
            "doc_id": doc_id,
            "transition": t,
            "gold_place": int(a[0] != b[0]) if ok else 0,
            "gold_time": int(a[1] != b[1] and bool(a[1]) and bool(b[1])) if ok else 0,
            "place_mask": ok,
            "time_mask": ok,
            "scene_prev": list(a) if a else None,
            "scene_cur": list(b) if b else None,
        })
    return out


def normalize_locations(filled: list[tuple | None], thresh: float = 0.8) -> list[tuple | None]:
    """문서 내 location 표면 변이 흡수 (HANDOFF_07 §4 노이즈). token-set Jaccard >= thresh 병합."""
    locs = sorted({f[0] for f in filled if f is not None})
    canon: dict[str, str] = {}
    for loc in locs:
        a = set(loc.split())
        for c in canon.values():
            b = set(c.split())
            if a and b and len(a & b) / len(a | b) >= thresh:
                canon[loc] = c
                break
        else:
            canon[loc] = loc
    return [(canon[f[0]], f[1]) if f is not None else None for f in filled]


def validate_against_dm7(all_rows: dict[str, list[dict]], masters: dict[str, list[dict]],
                         sessions: dict[str, list[int]], filled: dict[str, list]) -> dict:
    """회귀 테스트: 턴 라벨을 세션(다수결)으로 재집계 -> D-M7 세션 수준 수치를 재현하는가.

    D-M7 실측: place 변경 76/92 (83%), time 39/92 (42%).
    """
    gp1 = gt1 = tot = 0
    for did, rows in masters.items():
        sess_turns: dict[int, list[tuple]] = {}
        for i, s in enumerate(sessions[did]):
            if filled[did][i] is not None:
                sess_turns.setdefault(s, []).append(filled[did][i])
        sess_scene = {s: Counter(v).most_common(1)[0][0] for s, v in sess_turns.items() if v}
        for r in rows:
            if not r["boundary"]:
                continue
            a, b = sess_scene.get(r["prev_session"]), sess_scene.get(r["cur_session"])
            if a is None or b is None:
                continue
            tot += 1
            gp1 += int(a[0] != b[0])
            gt1 += int(a[1] != b[1] and bool(a[1]) and bool(b[1]))
    return {"boundaries": tot, "place_change": gp1, "time_change": gt1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", type=Path, default=REPO / "data/eacl_phase0/share_scripts")
    ap.add_argument("--master", type=Path,
                    default=REPO / "outputs/eacl_phase0/share_valid_probe30_master.jsonl")
    ap.add_argument("--reset-matrix", type=Path,
                    default=REPO / "outputs/eacl_phase0/share_valid_probe30_s5/reset_matrix.jsonl")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/eacl_phase0/share_turn_scene_align.jsonl")
    ap.add_argument("--min-match-ratio", type=float, default=0.15)
    ap.add_argument("--normalize-locations", action="store_true")
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    args = ap.parse_args()

    master = read_jsonl(args.master)
    m_by_doc: dict[str, list[dict]] = {}
    for r in master:
        m_by_doc.setdefault(r["doc_id"], []).append(r)
    rm_docs = {r["doc_id"] for r in read_jsonl(args.reset_matrix)}   # S5 5축 완료 문서만
    cov = json.loads((args.scripts / "_coverage.json").read_text())

    out_rows: list[dict] = []
    masters: dict[str, list[dict]] = {}
    sessions: dict[str, list[int]] = {}
    filled_all: dict[str, list] = {}
    skipped: list[str] = []

    for did in sorted(m_by_doc):
        if did not in rm_docs:
            skipped.append(f"{did}: no reset_matrix (S5 미완)")
            continue
        movie = did.split("__")[0]
        c = cov.get(movie, {})
        if c.get("status") != "ok" or c.get("match", 0) < args.min_match_ratio:
            skipped.append(f"{did}: script fail/low-match ({c.get('status')}, {c.get('match')})")
            continue
        sp = args.scripts / f"{movie}.txt"
        if not sp.exists():
            skipped.append(f"{did}: script file missing")
            continue
        scenes, lnorm = load_scenes(sp.read_text(encoding="utf-8", errors="ignore"))
        if not scenes:
            skipped.append(f"{did}: 0 sluglines parsed")
            continue

        texts, sess, rows = turns_of(m_by_doc[did])
        cands = turn_candidates(texts, lnorm)
        lines = monotone_align(cands)
        scene_per_turn = [scene_at(scenes, li) if li is not None else None for li in lines]
        filled = fill_scenes(scene_per_turn)
        if args.normalize_locations:
            filled = normalize_locations(filled)

        n_match = sum(1 for x in lines if x is not None)
        rws = turn_labels(did, filled, rows)
        out_rows += rws
        masters[did] = rows
        sessions[did] = sess
        filled_all[did] = filled
        n_mask = sum(1 for r in rws if not r["place_mask"])
        print(f"{did:42s} turns {len(texts):4d}  matched {n_match:4d} ({100*n_match/len(texts):3.0f}%)"
              f"  trans {len(rws):4d}  masked {n_mask:3d}"
              f"  place+ {sum(r['gold_place'] for r in rws):3d}"
              f"  time+ {sum(r['gold_time'] for r in rws):3d}")

    n_tr = len(out_rows)
    n_ok = sum(r["place_mask"] for r in out_rows)
    p_pos = sum(r["gold_place"] for r in out_rows)
    t_pos = sum(r["gold_time"] for r in out_rows)
    print(f"\n=== SHARE 턴 단위 씬 정렬 ===")
    print(f"docs {len(masters)}  transitions {n_tr}  usable {n_ok} (masked {n_tr-n_ok} = "
          f"{100*(n_tr-n_ok)/max(1,n_tr):.0f}%)")
    print(f"  SLUGLINE place_reset  {p_pos}/{n_ok} = {p_pos/max(1,n_ok):.3f}")
    print(f"  SLUGLINE time_reset   {t_pos}/{n_ok} = {t_pos/max(1,n_ok):.3f}")
    if skipped:
        print(f"\n제외 {len(skipped)}:")
        for s in skipped:
            print(f"  - {s}")

    if args.validate:
        v = validate_against_dm7(out_rows, masters, sessions, filled_all)
        print(f"\n=== 회귀 테스트 (D-M7 세션 수준 재현) ===")
        print(f"  세션 경계 {v['boundaries']}  (D-M7: 92)")
        print(f"  place 변경 {v['place_change']} ({100*v['place_change']/max(1,v['boundaries']):.0f}%)"
              f"  (D-M7: 76 = 83%)")
        print(f"  time  변경 {v['time_change']} ({100*v['time_change']/max(1,v['boundaries']):.0f}%)"
              f"  (D-M7: 39 = 42%)")
        ok = 76 <= v["place_change"] <= 80 and 36 <= v["time_change"] <= 41
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  턴 정렬이 D-M7 확립 수치와 어긋남 — 진행 금지.")
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
