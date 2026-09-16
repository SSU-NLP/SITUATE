#!/usr/bin/env python3
"""baseline 작동점을 **우리와 같은 데이터에서** 동결한다 (🔒 D-B40 사전 등록).

## 왜
우리 M5 는 임계를 **앵커 val 5-fold OOF** 에서 골라 동결했는데(D-B29/D-B30) baseline 은 아무
데서도 고르지 않은 값으로 돌고 있었다. 한쪽만 작동점을 맞춘 비교다.

⚠️ 다만 대부분의 baseline 상수는 **발표된 값 그대로**다(GreedySeg 5개 전부 일치, GraphSeg 2개,
TextTiling 3개). 실제 시정 대상은 **CSM 의 α**(원저자는 dev 에서 `alpha_search(-2,2,0.1)` 하는데
우리는 0.0 고정)와 **우리가 발명한 스트리밍 knob 3개**(window_d · min_gap · warmup)뿐이다.

## 규약 (사후 변경 금지)
- 선택셋 = 서사 **앵커 val**(의사-gold `count≥t*=3`) · 대화 **superseg/validation**. 둘 다 앞 60편.
- **dev-3 와 test 는 열지 않는다.** 도메인별로 따로 동결해 dev-3 / DTS-3 에 그대로 적용한다.
- 선택 지표 = 하네스 (A) Score. **동률이면 발표 기본값에 가까운 쪽**을 택한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-29_baseline_compare"
EXPD = REPO / "outputs/experiments/m4m5_phase1_anchor/data"
MASTER = REPO / "data/eacl_phase1/all_master.jsonl"
SDS = REPO / "benchmarks/superdialseg_data"
T_STAR = 3
SPLIT_SEED = 20260728          # 🔒 train_anchor_m4m5.py 와 동일 — 같은 val 분할을 재현한다
VAL_FRAC = 0.05

# 🔒 발표된 기본값 (as-published). 표의 대조군이자 동률 시 선택 기준.
PUBLISHED = {
    "TextTiling": {"min_gap": 4, "warmup_gaps": 3},
    "GraphSeg": {"window_d": 10},
    # 🔴 α=1.0 이 as-published 다 — 2026-05-24 실행(`2026-05-24_csm_online_delay2_fixed/REPORT.md`)이
    # 쓴 값이자 SuperDialseg `CSMSegmenter(cut_rate=1.0)` 기본값(`thr = mean + cut_rate·std`).
    # `csm_online.py` dataclass 의 `alpha=0.0` 은 폐기된 버그판
    # (`2026-05-23_csm_online_delay2_buggy_alpha0_logitdiff/`) 값이 코드에 남은 것이다.
    "CSM": {"alpha": 1.0, "min_gap": 2, "delay": 2},
}


# ── 선택셋 ────────────────────────────────────────────────────────────────────
def anchor_val_doc_ids() -> list[str]:
    """M5 의 r=0.0642 를 고른 바로 그 val 분할을 재현한다."""
    ix = np.load(EXPD / "phase1_index.npz", allow_pickle=True)
    doc_ids = [str(x) for x in ix["doc_ids"]]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(doc_ids))
    n_val = max(1, int(len(doc_ids) * VAL_FRAC))
    return [doc_ids[i] for i in perm[:n_val].tolist()]


def load_narrative(n_doc: int) -> list[tuple[str, list[str], np.ndarray]]:
    """🔴 **사용 금지** — 앵커에는 **사람 경계 라벨이 없다**(`count ≥ t*=3` 의사-gold 뿐).

    D-B40 초안이 여기서 baseline 작동점을 고르려 했는데 **틀렸다**(사용자 지적, D-B41).
    의사-gold 에 baseline 을 맞추는 것은 튜닝이 아니라 **대리지표 오염**이다. 실측도 그랬다 —
    TextTiling 이 경계를 2,000~2,400개 찍고 전 15셀이 0.236~0.241 에 몰려 선택 신호가 없었다.
    ⇒ 서사 작동점은 **dev-3(사람 gold)** 에서 고른다. 함수는 재현·감사용으로만 남긴다.
    """
    want = set(anchor_val_doc_ids()[:n_doc])
    ix = np.load(EXPD / "phase1_index.npz", allow_pickle=True)
    doc_ids, to, resets = [str(x) for x in ix["doc_ids"]], ix["trans_offsets"], ix["resets"]
    gold = {d: (resets[int(to[i]):int(to[i + 1])].sum(1) >= T_STAR).astype(np.int8)
            for i, d in enumerate(doc_ids) if d in want}
    rows: dict[str, list] = {}
    with MASTER.open(encoding="utf-8") as f:                 # 792MB — 한 줄씩 훑는다
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["doc_id"] in want:
                rows.setdefault(r["doc_id"], []).append(r)
    out = []
    for d, rs in rows.items():
        rs.sort(key=lambda r: r["transition"])
        sents = [rs[0]["previous"]] + [r["current"] for r in rs]
        g = gold[d]
        if len(g) == len(rs) and len(sents) >= 3:
            out.append((d, sents, g))
    return out


def load_dialogue(n_doc: int) -> list[tuple[str, list[str], np.ndarray]]:
    """superseg/validation 앞 n_doc 대화. 라벨 = 세그먼트의 **마지막** turn 이 1."""
    raw = json.loads((SDS / "superseg/segmentation_file_validation.json").read_text(encoding="utf-8"))
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for i, d in enumerate(arr):
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if len(utts) >= 3:
            out.append((f"val{i}", utts, np.asarray(yt[:-1], np.int8)))
        if len(out) >= n_doc:
            break
    return out


# ── 실행 ──────────────────────────────────────────────────────────────────────
def make(kind: str, device: str, **kw):
    if kind == "TextTiling":
        from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling as C
        return C(**kw)
    if kind == "GraphSeg":
        from hi_ontop.baselines.graphseg_window import GraphSegWindowD as C
        return C(**kw)
    if kind == "CSM":
        from hi_ontop.baselines.csm_online import CSMOnlineDelay2 as C
        return C(device=device, **kw)
    raise ValueError(kind)


def run(kind: str, data, device: str, **kw) -> dict:
    preds = {}
    for did, sents, g in data:
        seg = make(kind, device, **kw)
        bset: set[int] = set()
        for u in sents:
            for b in seg.push(u):
                bset.add(int(b))
        v = np.zeros(len(g), np.int8)
        for b in bset:
            if 1 <= b <= len(g):
                v[b - 1] = 1
        preds[did] = v
    r = score_dialogues([np.concatenate([g, [0]]).tolist() for _, _, g in data],
                        [np.concatenate([preds[d], [0]]).tolist() for d, _, _ in data])
    return {"score": r["score"], "pk": r["pk"], "wd": r["wd"], "f1": r["f1"],
            "n_pred": int(sum(int(v.sum()) for v in preds.values()))}


def grid(kind: str, ckpt: str | None) -> list[dict]:
    """🔒 D-B40 격자. 사후 확장 금지."""
    if kind == "TextTiling":
        return [{"min_gap": m, "warmup_gaps": w} for m in (1, 2, 3, 4, 5) for w in (2, 3, 5)]
    if kind == "GraphSeg":
        return [{"window_d": d} for d in (5, 10, 15, 20, 30)]
    if kind == "CSM":
        # 🔴 D-B41 정정: **탐색하지 않는다.** α=1.0 은 SuperDialseg `cut_rate=1.0` = **발표된 값**이며
        # TextTiling w/k/c · GraphSeg 임계를 고정한 것과 같은 규약이다(우리가 발명한 knob 만 탐색).
        # 원저자 `alpha_search(-2,2,0.1)` 은 *같은 코퍼스의 dev→test* 오프라인 절차라 코퍼스를
        # 건너 전이시키는 우리 설정에 그대로 옮기는 것이 "충실"이 되지 않는다. 41점을 훑으면
        # GraphSeg 서사에서 본 것과 같은 **퇴화 구성이 1위로 올라올 위험**도 있다.
        raise SystemExit("CSM 은 발표값 동결(α=1.0 · delay=2 · min_gap=2). 탐색 대상이 아니다 — D-B41.")
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["dev3", "dialogue", "narrative_ANCHOR_INVALID"])
    ap.add_argument("--methods", default="TextTiling,GraphSeg")
    ap.add_argument("--n_doc", type=int, default=60)
    ap.add_argument("--ckpt", default=None, help="CSM 체크포인트 (CSM 탐색 시 필수)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.domain == "dev3":
        from baseline_compare import load_dataset
        data = load_dataset("dev3")                      # 사람 gold (Michelmann author-filtered)
    elif args.domain == "dialogue":
        data = load_dialogue(args.n_doc)                 # superseg/validation — test 와 disjoint
    else:
        raise SystemExit("앵커 선택셋은 무효다 (D-B41). --domain dev3 또는 dialogue 를 쓸 것.")
    n_tr = sum(len(g) for _, _, g in data)
    print(f"[선택셋 · {args.domain}] 문서 {len(data)} · 전이 {n_tr} · "
          f"gold {sum(int(g.sum()) for _, _, g in data)}  🔒 dev-3·test 미개봉", flush=True)

    rp = OUT / "baseline_opt_freeze.json"
    allr = json.loads(rp.read_text()) if rp.exists() else {}
    for kind in [x.strip() for x in args.methods.split(",") if x.strip()]:
        cells, t0 = grid(kind, args.ckpt), time.perf_counter()
        rows = []
        for i, kw in enumerate(cells):
            m = run(kind, data, args.device, **kw)
            tag = {k: v for k, v in kw.items() if k not in ("backbone", "ckpt_path")}
            rows.append({"hp": tag, "metrics": m})
            print(f"  [{i+1:>3}/{len(cells)}] {tag} -> Score {m['score']:.4f} "
                  f"(경계 {m['n_pred']})", flush=True)
        # 동률(±1e-9)이면 발표 기본값과의 거리로 가른다
        pub = PUBLISHED.get(kind, {})
        def dist(r: dict) -> float:
            return sum(abs(r["hp"].get(k, v) - v) for k, v in pub.items() if k in r["hp"])
        best = max(rows, key=lambda r: (round(r["metrics"]["score"], 9), -dist(r)))
        base = next((r for r in rows if all(r["hp"].get(k) == v for k, v in pub.items())), None)
        allr[f"{args.domain}|{kind}"] = {
            "n_doc": len(data), "n_trans": n_tr, "grid": len(cells),
            "best": best, "as_published": base,
            "delta": (best["metrics"]["score"] - base["metrics"]["score"]) if base else None,
            "wall_s": time.perf_counter() - t0}
        rp.write_text(json.dumps(allr, indent=2, default=float))
        d = allr[f"{args.domain}|{kind}"]["delta"]
        print(f"  ⇒ {kind} 동결 {best['hp']}  Score {best['metrics']['score']:.4f}"
              + (f"  (발표기본값 {base['metrics']['score']:.4f}, Δ {d:+.4f})" if base else "")
              + f"  {time.perf_counter()-t0:.0f}s\n", flush=True)
    print(f"-> {rp}")


if __name__ == "__main__":
    main()
