#!/usr/bin/env python3
"""causality 반사실+인과 2단계 창(window) 추출 — 사전등록
`outputs/experiments/2026-07-25_causality_cf_window/PRE_REGISTRATION.md`.

Trabasso & Sperry(1985)는 서로 다른 두 기준을 썼다:
  (a) 반사실 필요성 — "A가 없었다면 B도 안 일어났을까". 물리·심리·동기·**가능화 전부**에 높은 강도.
  (b) 인과 기준     — 앞의 셋과 **가능화(enablement)**를 가르는 것은 이 두 번째 기준이다.
따라서 질의를 두 단계로 나눈다. 한 질문에 합치면 P2ⓑ 실패 시 스킴이 틀린 건지 프롬프트가 틀린 건지
구분할 수 없다(prisoner_in_yard→sprint 는 반사실로는 "그렇다"이므로).

  1단계 y_still_happens=true  -> none (2단계 안 물음)
  1단계 false -> 2단계 physical / psychological / motivational / enabling

링크 = {physical, psychological, motivational}. enabling·none 은 링크 아님(문헌 기준: 가능화는
인과 기준으로는 인과가 아니다). reset(i)=1 ⟺ 창 [i-W, i-1] 안에 링크가 하나도 없음(막다른 사건).

모드:
  --probe <jsonl> [--probe ...]  probe 채점 (k=3 다수결). P1/P2 사전기준 대비 판정.
  --dev --master-dir <dir>       코퍼스 추출 (단일 draw). transition 별 열 + 거리별 관계 라벨 저장.

boundary/change/segment/reset/이전상태 어휘는 프롬프트에 일절 넣지 않는다(S1 halo 회피). 쌍에는
인덱스·거리·중간 문장을 주지 않으므로 모델은 두 사건이 얼마나 떨어져 있는지 알 수 없다.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # llm_s5_states import
import llm_s5_states as s5  # noqa: E402
from openai import OpenAI  # noqa: E402

EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
RELATIONS = ("none", "physical", "psychological", "motivational", "enabling")
LINK = frozenset({"physical", "psychological", "motivational"})

S1_SYS = (
    "You judge whether one event in a story was necessary for another. "
    "Given EVENT X and EVENT Y from the same story, answer one question: "
    "if X had not happened, in the circumstances of this story, would Y still have happened? "
    'Return JSON {"y_still_happens": true or false}.'
)

S2_SYS = (
    "You classify how one event in a story bears on another. You are told that Y depended on X. "
    "X did not merely precede Y. Decide which of these holds:\n"
    '  "physical" - X physically brought Y about.\n'
    '  "psychological" - X produced Y as a mental or emotional response.\n'
    '  "motivational" - the motive for doing Y came from X.\n'
    '  "enabling" - X only made Y possible: it set up a condition, means, or opportunity that Y '
    "depended on, without bringing Y about.\n"
    'Return JSON {"relation": "<one of the four labels>"}.'
)

_stats = collections.Counter()


def _pair(x: str, y: str) -> str:
    return f"EVENT X: {x}\nEVENT Y: {y}"


def relation(client, model: str, x: str, y: str) -> str:
    """반사실(1단계) -> 인과(2단계). 파싱 실패는 1회 재시도 후 보수적으로 none."""
    for attempt in (0, 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=200,
                response_format={"type": "json_object"}, extra_body=EXTRA,
                messages=[{"role": "system", "content": S1_SYS},
                          {"role": "user", "content": _pair(x, y)}])
            still = json.loads(r.choices[0].message.content).get("y_still_happens")
            if not isinstance(still, bool):
                raise ValueError("y_still_happens not boolean")
            break
        except Exception:
            if attempt:
                _stats["parse_fail_s1"] += 1
                return "none"
    _stats["s1_true" if still else "s1_false"] += 1
    if still:
        return "none"

    for attempt in (0, 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=200,
                response_format={"type": "json_object"}, extra_body=EXTRA,
                messages=[{"role": "system", "content": S2_SYS},
                          {"role": "user", "content": _pair(x, y)}])
            lab = str(json.loads(r.choices[0].message.content).get("relation", "")).strip().lower()
            if lab not in RELATIONS[1:]:
                raise ValueError(f"bad relation {lab!r}")
            return lab
        except Exception:
            if attempt:
                _stats["parse_fail_s2"] += 1
                return "none"
    return "none"


def relation_k(client, model: str, x: str, y: str, k: int) -> tuple[str, list[str]]:
    """k회 표집 후 다수결. 동률이면 보수적으로 none."""
    votes = [relation(client, model, x, y) for _ in range(k)]
    cnt = collections.Counter(votes)
    top = cnt.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "none", votes
    return top[0][0], votes


# ------------------------------------------------------------------ probe
def run_probe(client, model: str, paths: list[Path], k: int) -> None:
    for path in paths:
        rows = [json.loads(l) for l in path.open() if l.strip()]
        with ThreadPoolExecutor(max_workers=12) as ex:
            res = list(ex.map(lambda r: relation_k(client, model, r["prev"], r["cur"], k), rows))
        print(f"\n=== {path.name} (n={len(rows)}, k={k} 다수결) ===")
        cells: dict[str, list] = collections.defaultdict(list)
        unan = 0
        for r, (lab, votes) in zip(rows, res):
            cell = r.get("cell") or ("discordant_break" if r.get("causal_break") else
                                     "discordant_continuous")
            cells[cell].append((r, lab))
            if len(set(votes)) == 1:
                unan += 1
            ok = _item_ok(cell, lab)
            print(f"  {r['id']:<6} {cell:<22} -> {lab:<14} {'✓' if ok else '✗'}  "
                  f"votes={'/'.join(votes)}")
        print(f"  만장일치 {unan}/{len(rows)} · 분열 {len(rows) - unan}")
        for cell, items in sorted(cells.items()):
            hit = sum(_item_ok(cell, lab) for _, lab in items)
            print(f"  [{cell}] {hit}/{len(items)} = {hit / len(items):.3f}   {_CELL_BAR[cell]}")
        # P1 원본 14문항 부분집합 (halo 7/7·0/7 비교용)
        sub = [(r, lab) for r, lab in zip(rows, [x[0] for x in res])
               if r.get("origin") == "halo_v1"]
        if sub:
            for cell in ("discordant_break", "discordant_continuous"):
                it = [(r, lab) for r, lab in sub
                      if (r.get("cell") or "") == cell]
                if it:
                    hit = sum(_item_ok(cell, lab) for _, lab in it)
                    print(f"  (원본14 부분집합) [{cell}] {hit}/{len(it)} = {hit / len(it):.3f}")


_CELL_BAR = {
    "discordant_break": "P1 단절율 ≥0.80",
    "discordant_continuous": "P1 오발현 ≤0.20 (= 표시된 값은 '연속으로 옳게 본 비율' ≥0.80)",
    "cutaway": "P2ⓐ 링크율 ≥0.70",
    "enabling": "P2ⓑ enabling 라벨율 ≥0.70",
}


def _item_ok(cell: str, lab: str) -> bool:
    if cell == "discordant_break":
        return lab not in LINK          # 인과 단절 -> 링크 아님
    if cell == "discordant_continuous":
        return lab in LINK              # 인과 연쇄 -> 링크 (오발현의 여집합)
    if cell == "cutaway":
        return lab in LINK
    if cell == "enabling":
        return lab == "enabling"
    raise KeyError(cell)


# -------------------------------------------------------------------- dev
def run_dev(client, model: str, master_dir: Path, out: Path, window: int,
            decide: int, only: str | None, workers: int) -> None:
    masters = sorted(master_dir.glob("*.jsonl"))
    if only:
        masters = [m for m in masters if m.stem == only]
    done = {(r["doc_id"], int(r["transition"])) for r in s5.read(out)}
    if done:
        print(f"[resume] 기존 {len(done)} 전이 건너뜀", flush=True)
    all_rows = list(s5.read(out))

    for m in masters:
        rows = sorted((json.loads(l) for l in m.open() if l.strip()),
                      key=lambda r: int(r["transition"]))
        doc_id = rows[0]["doc_id"]
        sents = s5.sequence(rows)
        tr = [int(r["transition"]) for r in rows if (doc_id, int(r["transition"])) not in done]
        if not tr:
            continue
        # 창 안의 모든 (거리 d, 전이 i) 쌍. 겹치는 쌍은 없다(문장은 재등장하나 쌍은 아니다).
        jobs = [(i, d) for i in tr for d in range(1, window + 1) if i - d >= 0]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            labs = list(ex.map(lambda jd: relation(client, model, sents[jd[0] - jd[1]],
                                                   sents[jd[0]]), jobs))
        by_tr: dict[int, dict[str, str]] = collections.defaultdict(dict)
        for (i, d), lab in zip(jobs, labs):
            by_tr[i][str(d)] = lab
        for i in tr:
            rel = by_tr[i]
            cand = [d for d in range(1, decide + 1) if str(d) in rel]
            all_rows.append({
                "doc_id": doc_id, "transition": i,
                "causality_cf": int(not any(rel[str(d)] in LINK for d in cand)),
                "n_cand": len(cand), "relations": rel,
            })
        s5.write(out, sorted(all_rows, key=lambda r: (r["doc_id"], int(r["transition"]))))
        fire = sum(r["causality_cf"] for r in all_rows if r["doc_id"] == doc_id)
        n = sum(1 for r in all_rows if r["doc_id"] == doc_id)
        print(f"  {doc_id:<24} n={n:>4}  발현율={fire / n:.3f}", flush=True)

    summarize(all_rows, decide)
    print(f"-> {out}")


def summarize(rows: list[dict], decide: int) -> None:
    n = len(rows)
    fire = sum(r["causality_cf"] for r in rows)
    labs = collections.Counter(l for r in rows for l in r["relations"].values())
    tot = sum(labs.values())
    print(f"\n=== 요약 ({n} 전이, 판정 창 W={decide}) ===")
    print(f"발현율(reset) = {fire / n:.3f}   [P3 대역 0.15~0.40]")
    print("관계 유형 분포 (창 안 전체 쌍 기준):")
    for lab in RELATIONS:
        print(f"  {lab:<15} {labs[lab]:>6}  {labs[lab] / tot:.3f}")
    links = sum(labs[l] for l in LINK)
    dep = tot - labs["none"]
    print(f"1단계 false(반사실 필요성 인정) = {dep}/{tot} = {dep / tot:.3f}")
    if dep:
        print(f"  그중 enabling = {labs['enabling']}/{dep} = {labs['enabling'] / dep:.3f}"
              f"   (두 기준이 실제로 갈리는지)")
    if links:
        print("링크 내부 구성:", {l: round(labs[l] / links, 3) for l in sorted(LINK)})
    pf = _stats["parse_fail_s1"] + _stats["parse_fail_s2"]
    print(f"파싱 실패 {pf} (s1 {_stats['parse_fail_s1']} / s2 {_stats['parse_fail_s2']})"
          f"  [>2% 면 런 무효]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, action="append")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--master-dir", type=Path,
                    default=REPO / "outputs/eacl_phase0/repcorpus_masters_stage0")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/experiments/2026-07-25_causality_cf_window/cf_dev.jsonl")
    ap.add_argument("--window", type=int, default=5, help="수집 창 (판정은 --decide-window)")
    ap.add_argument("--decide-window", type=int, default=3, help="사전등록 판정 창 = 3")
    ap.add_argument("--k", type=int, default=1, help="다중샘플 수 (probe=3, 코퍼스=1)")
    ap.add_argument("--only", help="단일 문서 스모크용 master stem")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default="qwen36-35b")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    a = ap.parse_args()

    client = OpenAI(api_key="dummy", base_url=a.base_url, timeout=180, max_retries=2)
    if a.probe:
        run_probe(client, a.model, a.probe, a.k)
    if a.dev:
        run_dev(client, a.model, a.master_dir, a.out, a.window, a.decide_window,
                a.only, a.workers)


if __name__ == "__main__":
    main()
