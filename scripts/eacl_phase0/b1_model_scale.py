#!/usr/bin/env python3
"""B1 모델 강건성 — 같은 B1 절차를 서로 다른 LM 으로 돌린 결과를 나란히 채점.

기존 표(ledger D-M11 부속 "B1 모델 강건성")의 확장이다. **readout·규칙·동결 프로토콜을
`b1_deployable.py` 에서 그대로 import 해 쓴다** — 재구현하지 않는다. 바뀌는 것은 단어 덤프 경로뿐.

  readout = newsent_max transient Bayesian surprise
  규칙    = 적응 임계 mu + c*sigma, c 는 **pieman(dev) argmax 로 동결** (기존 행과 동일 절차)

사용:
  .venv/bin/python scripts/eacl_phase0/b1_model_scale.py \
      --label qwen36-35b-a3b --words-dir outputs/eacl_phase0/b1_bayes/qwen36_35b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_deployable as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="이 LM 의 표기 이름 (표에 그대로 들어간다)")
    ap.add_argument("--words-dir", required=True, type=Path,
                    help="{doc}_words.jsonl 이 있는 디렉터리")
    ap.add_argument("--docs", nargs="*", default=["pieman", "tunnel", "monkey"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    wd = args.words_dir if args.words_dir.is_absolute() else REPO / args.words_dir
    missing = [d for d in args.docs if not (wd / f"{d}_words.jsonl").exists()]
    if missing:
        raise SystemExit(f"🔴 단어 덤프 없음: {missing} (--dump-words 로 재실행 필요)")

    # gold 는 master jsonl 에서 직접 읽는다.
    # 🔴 `b1_deployable.load_doc` 은 쓰지 않는다 — 그 함수는 reset_matrix 의 M3/M1 열까지 읽는데
    #    S5 v3.1 축 개명(protagonist_perspective -> protagonist_present)으로 stale 해 KeyError 가 난다.
    #    이 스크립트는 **B1 열만** 쓰므로 그 경로를 타지 않는다. readout(b1_newsent_max)·규칙
    #    (adaptive_pred)·채점기는 동결 스크립트 것을 그대로 import 해 쓴다.
    def load(d):
        rm = sorted(B.read(B.BASE / f"{d}_michelmann_master.jsonl"),
                    key=lambda r: int(r["transition"]))
        words = B.read(wd / f"{d}_words.jsonl")
        return {"n": len(rm) + 1,
                "gold": [int(r["transition"]) for r in rm if int(r["boundary"])],
                "b1": B.b1_newsent_max(words, len(rm)),
                "y": [int(r["boundary"]) for r in rm]}

    data = {d: load(d) for d in args.docs}
    dev = data["pieman"]
    c_b1 = max(B.C_GRID, key=lambda c: B.score_docs([dev], "b1", c, "adapt"))

    rows, golds, preds = {}, [], []
    for d in args.docs:
        x = data[d]
        rows[d] = {
            "score": B.score_docs([x], "b1", c_b1, "adapt"),
            "auc": B.auc(x["y"], x["b1"]),
            "n_pred": len(B.adaptive_pred(x["b1"], c_b1)),
            "n_gold": len(x["gold"]),
        }
        rows[d]["bor"] = rows[d]["n_pred"] / rows[d]["n_gold"]
        golds.append(B.endturn(x["n"], x["gold"]))
        preds.append(B.endturn(x["n"], B.adaptive_pred(x["b1"], c_b1)))
    pooled = B.score_dialogues(golds, preds)

    out = {"label": args.label, "words_dir": str(wd), "c_frozen_on_pieman": c_b1,
           "readout": "newsent_max transient Bayesian surprise",
           "per_doc": rows, "pooled": pooled}
    print(f"[{args.label}] c(dev=pieman argmax) = {c_b1}")
    print(f"{'doc':8s} {'Score':>8s} {'AUC':>7s} {'pred':>6s} {'gold':>5s} {'BOR':>6s}")
    for d in args.docs:
        r = rows[d]
        print(f"{d:8s} {r['score']:8.4f} {r['auc']:7.4f} {r['n_pred']:6d} "
              f"{r['n_gold']:5d} {r['bor']:6.2f}")
    print(f"{'pooled':8s} {pooled['score']:8.4f}  (F1 {pooled['f1']:.3f} "
          f"Pk {pooled['pk']:.3f} WD {pooled['wd']:.3f})")

    dest = args.out or (REPO / "outputs/experiments/2026-07-31_b1_model_scale"
                        / f"b1_{args.label.replace('/', '_')}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n→ {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
