#!/usr/bin/env python3
"""D-M11 ② — gold-swap 랭킹보존 검사: Tier2의 B3-gold가 인간 gold의 대리로 성립하는가?

질문. Tier2 주장은 "M3 > M1/B1 (B3-gold 기준)"이다. 이것이 "M3 > M1/B1 (인간 기준)"의 대리이려면
**같은 문서에서 gold만 인간→B3로 갈아끼워도 랭킹과 격차의 부호가 보존**돼야 한다.
기존 데이터로는 답이 안 나온다 — dev(인간 gold)와 Tier2(B3 gold)는 **gold 출처와 문서가 동시에**
달라 완전 교란이다. 같은 문서에서 gold만 바꿔야 gold 효과가 분리된다.

설계 (D-M11 ②에서 실행 전 동결 — 결과 보고 변경 금지):
  - 데이터: dev 인간 gold 3편 (pieman/tunnel/monkey michelmann, author-filtered = 메인)
  - 방법: M3(t*=3) / M1(c=1.0) / B1(newsent_max, c=0.75) — 동결 설정, 예측 재생성 없이 동일 예측을
    두 reference로 채점
  - reference 2벌: (a) 인간 gold  (b) B3 reference = Tier2와 **동일 프로토콜**
    (GPT-4.1 · B2 프롬프트 verbatim · temp 0 × 20회 · B3 top-n 합의)
  - 1차 지표 = M3-M1, M3-B1 두 pairwise 차이의 **부호**
  - 보존 = 두 차이 모두 부호 동일 AND 문서별 3/3 일관 / 역전 = 하나라도 부호 반전
  - **불충분** = 부호는 같으나 문서별 불일치 -> "통과"로 처리 금지 (dev 3편뿐이므로 기본값)

⚠️ raw run을 반드시 캐시한다 (dev_b2b3_topline.py가 저장을 안 해 재지불이 발생한 전례).
   --reuse 로 캐시에서 채점만 재실행 (API 0).

부산물: D-M11 ⑤ split-half 합의 안정성 (20 run을 10/10으로 갈라 합의 일치도) — 같은 지출에서 무료.

usage:
  uv run python scripts/eacl_phase0/dev_goldswap.py            # 생성 + 채점 (~$2.5)
  uv run python scripts/eacl_phase0/dev_goldswap.py --reuse    # 캐시에서 채점만 (API 0)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from build_tier2_gold import PROMPT_HEAD, PROMPT_TAIL, GOLD_MODEL, WORD, client, run_to_transitions  # noqa: E402
from dev_b2b3_topline import DEV, AX, C_M1, T_M3, C_B1, read, endturn, adaptive, count_pred  # noqa: E402

BASE = REPO / "outputs/eacl_phase0"
CACHE = BASE / "dev_b3ref"
OUT = REPO / "outputs/experiments/dev_goldswap"


def consensus(runs: list[list[int]]) -> list[int]:
    """B3 top-n 합의 (D-M9 규칙 그대로): 위치별 득표 -> n=round(평균 경계수) -> 상위 n. 동률=SHA-256."""
    tally: dict[int, int] = {}
    for tr in runs:
        for t in tr:
            tally[t] = tally.get(t, 0) + 1
    n = int(math.floor(sum(len(tr) for tr in runs) / len(runs) + 0.5))
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], hashlib.sha256(str(kv[0]).encode()).hexdigest()))
    return sorted(t for t, _ in ranked[:n])


def gen_runs(c, story: str, sents: list[str], reps: int, temp: float, workers: int) -> list[list[int]]:
    orig = len(WORD.findall(story))

    def one():
        for _ in range(6):
            try:
                r = c.chat.completions.create(
                    model=GOLD_MODEL, temperature=temp, max_tokens=32000, timeout=300,
                    messages=[{"role": "user", "content": PROMPT_HEAD + story + PROMPT_TAIL}])
            except Exception:
                continue
            out = r.choices[0].message.content or ""
            if r.choices[0].finish_reason != "length" and len(WORD.findall(out)) >= 0.8 * orig:
                return run_to_transitions(out, sents)
        raise RuntimeError("fail")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [f.result() for f in as_completed([ex.submit(one) for _ in range(reps)])]


def preds(name: str, rows: list[dict], n: int) -> dict[str, list[int]]:
    """동결 설정의 방법별 예측 (gold와 무관 — 두 reference에 동일 예측을 채점)."""
    _, rmp, b1p = DEV[name]
    rm = {int(r["transition"]): r for r in read(BASE / rmp)}
    cnt = [sum(int(rm[int(r["transition"])][a]) for a in AX) for r in rows]
    idm = [float(r["idm_score"]) for r in rows]
    Wd = read(BASE / b1p)
    g0: dict[int, int] = {}
    for gi, w in enumerate(Wd):
        g0.setdefault(w["sent"], gi)
    V = [w["transient"] for w in Wd]
    b1 = [float(max(V[g0.get(t, 0):g0.get(t + 1, len(V))]) if g0.get(t) is not None else 0)
          for t in range(1, n)]
    return {"M3": count_pred(cnt, T_M3), "M1": adaptive(idm, C_M1), "B1": adaptive(b1, C_B1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="캐시에서 채점만 (API 0)")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    c = None if args.reuse else client()

    per_doc = {}
    for name, (mp, _, _) in DEV.items():
        rows = sorted(read(BASE / mp), key=lambda r: int(r["transition"]))
        n = len(rows) + 1
        human = [int(r["transition"]) for r in rows if int(r["boundary"])]
        sents = [rows[0]["previous"], *[r["current"] for r in rows]]
        story = re.sub(r"\s+", " ", " ".join(sents)).strip()

        cf = CACHE / f"{name}_runs.json"
        if cf.exists():
            runs = json.loads(cf.read_text())["runs"]
            print(f"[{name}] 캐시 재사용: {len(runs)} runs")
        else:
            if args.reuse:
                raise SystemExit(f"--reuse 인데 캐시 없음: {cf}")
            print(f"[{name}] B3 생성 중 (reps={args.reps}, temp={args.temp}) ...")
            runs = gen_runs(c, story, sents, args.reps, args.temp, args.workers)
            cf.write_text(json.dumps({"doc": name, "model": GOLD_MODEL, "reps": args.reps,
                                      "temp": args.temp, "n_sentences": n, "runs": runs},
                                     ensure_ascii=False))
            print(f"[{name}] -> {cf}")

        b3 = consensus(runs)
        P = preds(name, rows, n)
        sc = {}
        for ref_name, ref in [("human", human), ("b3", b3)]:
            g = endturn(n, ref)
            sc[ref_name] = {m: score_dialogues([g], [endturn(n, p)])["score"] for m, p in P.items()}
        # split-half (D-M11 ⑤)
        h1, h2 = consensus(runs[:len(runs) // 2]), consensus(runs[len(runs) // 2:])
        inter = len(set(h1) & set(h2))
        sh_f1 = 2 * inter / (len(h1) + len(h2)) if (h1 or h2) else float("nan")
        per_doc[name] = {
            "n_trans": n - 1, "human_n": len(human), "b3_n": len(b3),
            "human_dens": len(human) / (n - 1), "b3_dens": len(b3) / (n - 1),
            "scores": sc, "uniq_seg": len({tuple(r) for r in runs}), "reps": len(runs),
            "per_run_counts": sorted(len(r) for r in runs),
            "splithalf_f1": sh_f1, "splithalf_n": [len(h1), len(h2)],
        }

    (OUT / "goldswap.json").write_text(json.dumps(per_doc, ensure_ascii=False, indent=1))

    L = []
    L.append("=== 참조 밀도 (같은 문서, gold만 교체) ===")
    L.append(f"{'doc':8s} {'전이':>5s} | {'인간 n':>6s} {'인간밀도':>8s} | {'B3 n':>5s} {'B3밀도':>7s}")
    for d, r in per_doc.items():
        L.append(f"{d:8s} {r['n_trans']:5d} | {r['human_n']:6d} {r['human_dens']:8.3f} | "
                 f"{r['b3_n']:5d} {r['b3_dens']:7.3f}")

    L.append("\n=== Score (동일 예측, 두 reference) ===")
    L.append(f"{'doc':8s} | {'M3(H)':>6s} {'M1(H)':>6s} {'B1(H)':>6s} | {'M3(B3)':>7s} {'M1(B3)':>7s} {'B1(B3)':>7s}")
    for d, r in per_doc.items():
        h, b = r["scores"]["human"], r["scores"]["b3"]
        L.append(f"{d:8s} | {h['M3']:6.3f} {h['M1']:6.3f} {h['B1']:6.3f} | "
                 f"{b['M3']:7.3f} {b['M1']:7.3f} {b['B1']:7.3f}")

    L.append("\n=== 🔒 판정: pairwise 차이의 부호 (D-M11 ② 사전 고정 기준) ===")
    L.append(f"{'doc':8s} | {'M3-M1(H)':>9s} {'M3-M1(B3)':>10s} {'부호':>5s} | {'M3-B1(H)':>9s} {'M3-B1(B3)':>10s} {'부호':>5s}")
    ok = {"M3-M1": [], "M3-B1": []}
    for d, r in per_doc.items():
        h, b = r["scores"]["human"], r["scores"]["b3"]
        row = [f"{d:8s} |"]
        for pair, x, y in [("M3-M1", "M3", "M1"), ("M3-B1", "M3", "B1")]:
            dh, db = h[x] - h[y], b[x] - b[y]
            same = (dh > 0) == (db > 0)
            ok[pair].append(same)
            row.append(f" {dh:9.3f} {db:10.3f} {'같음' if same else '반전':>5s} |")
        L.append("".join(row))

    L.append("")
    allsame = all(all(v) for v in ok.values())
    for pair, v in ok.items():
        L.append(f"  {pair}: 문서별 부호 일치 {sum(v)}/{len(v)}")
    L.append("")
    if allsame:
        L.append("  판정 = **보존** (두 pairwise 모두 3/3 부호 일치)")
        L.append("     -> Tier2 = 비교 랭킹의 silver proxy로 제한적 사용 가능 (D-M11 ②)")
    elif any(not all(v) for v in ok.values()):
        L.append("  판정 = **역전** (부호 반전 발생)")
        L.append("     -> 기존 3/3 Go 및 'held-out 승리' 주장 철회 (D-M11 ②)")
    L.append("  ⚠️ dev 문서 3편뿐 — 모호하면 '통과'가 아니라 '불충분'이 기본값 (D-M11 ②)")

    L.append("\n=== D-M11 ⑤ B3 합의 안정성 (부산물) ===")
    L.append(f"{'doc':8s} {'고유분절':>8s} {'run별 경계수':>14s} {'split-half F1':>13s} {'half n':>10s}")
    for d, r in per_doc.items():
        pc = r["per_run_counts"]
        L.append(f"{d:8s} {r['uniq_seg']:5d}/{r['reps']:<2d} {min(pc):6d} ~ {max(pc):<6d} "
                 f"{r['splithalf_f1']:13.3f} {str(r['splithalf_n']):>10s}")

    txt = "\n".join(L)
    print("\n" + txt)
    (OUT / "goldswap_table.txt").write_text(txt)
    print(f"\n-> {OUT}/goldswap.json · goldswap_table.txt · 캐시 {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
