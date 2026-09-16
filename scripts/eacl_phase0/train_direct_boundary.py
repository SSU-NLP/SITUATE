#!/usr/bin/env python3
r"""M8-DirectBoundary — 동결 인코더 + 이진 경계 헤드 **직행** 베이스라인.

## 무엇을 / 왜
5축 구조(축별 라벨·축별 dynamics·축별 확률 합)를 **전혀 거치지 않고**, 앵커에서 코드로 유도된
**경계 라벨 하나**(동시 리셋 ≥ t*=3)만으로 문장 전이 이진분류기를 학습한다.
M5 와 동률이면 **5축 분해의 예측적 기여는 0**이고 남는 주장은 해석 가능성뿐이다.
사전등록: `outputs/experiments/2026-07-31_m8_direct_boundary/PREREG.md` (실행 전 작성, 수정 금지).

## 공정 비교 조건 (전부 M4/M5 와 동일)
- 인코더 frozen mpnet 768D · 앵커 phase1 10,598편 · **분할 seed 20260728 · val 5%**
- plain BCE(pos_weight=1, W3) · Adam lr 3e-4 · 8 epoch · seed 0/1/2 · 마지막 epoch 사용
- dev-3 채점 = `dts_scoring.score_dialogues`, deployable(경계 개수 미제공)
- 임계 = 적응임계(`online_nf`, c=1.0, warmup 5) · 문서 내 고정분위수 r=0.0642(D-B30 동결)
- 🔴 dev-3 로 고르는 것 없음(HP 탐색 없음, 모델 선택 없음). 봉인 시험지 미개봉.

## 실행
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/train_direct_boundary.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-31_m8_direct_boundary"
T_STAR = 3          # 🔒 동결
C_MAIN = 1.0        # 🔒 D-B39 적응임계 계수 (출처 = SuperDialseg 공식 test). 재탐색 금지.
ANCHOR_R = 0.0642   # 🔒 D-B30 앵커 val 동결 경계율. dev-3 미사용.
SPLIT_SEED = 20260728
VAL_FRAC = 0.05
CAVEAT = "앵커 k=1 · S_self 0.659(상한) · dev-3 = 연습장 3편 — Δ·방향만 읽는다"


# ── 모델 ─────────────────────────────────────────────────────────────────────
class DirectBoundaryHead(nn.Module):
    """전이 특징 -> 경계 로짓 1개. 상태 없음 · 축 없음 · dynamics 없음."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def feats(zl: torch.Tensor, zr: torch.Tensor, cl: torch.Tensor | None) -> torch.Tensor:
    """전이 특징. `cl` 이 있으면 과거 누적문맥 항을 덧붙인다(과거만 사용 = 온라인 적법)."""
    parts = [zl, zr, zr - zl, zl * zr]
    if cl is not None:
        parts.append(zr - cl)
    return torch.cat(parts, -1)


# ── 데이터 ───────────────────────────────────────────────────────────────────
def doc_split(n_docs: int, doc_ids: list[str] | None = None,
              author_map: str = "") -> tuple[list[int], list[int]]:
    """🔒 M4/M5 (`train_anchor_m4m5.main`) 와 **동일한** 문서 분할.

    author_map 이 주어지면 **작가 단위**로 나눈다 — M5 와 조건을 맞추기 위함이다.
    무작위 분할은 val 의 96.1% 가 train 에 같은 작가를 둔다(2026-08-03 실측).
    """
    rng = np.random.default_rng(SPLIT_SEED)
    n_val = max(1, int(n_docs * VAL_FRAC))
    if not author_map:
        perm = rng.permutation(n_docs)
        return perm[n_val:].tolist(), perm[:n_val].tolist()
    import sys as _s, json as _j
    _s.path.insert(0, str(REPO / "scripts/eacl_phase1"))
    from author_review_list import parse_person  # noqa: E402
    amap = {}
    for _l in open(author_map, encoding="utf-8"):
        _r = _j.loads(_l)
        _n, _y, _ = parse_person(_r["author"].split(";")[0])
        amap[_r["doc_id"]] = (_n.split("(")[0].strip().rstrip(",").lower(), _y)
    groups: dict = {}
    for i in range(n_docs):
        groups.setdefault(amap.get(doc_ids[i], ("__unk__", doc_ids[i])), []).append(i)
    keys = sorted(groups, key=lambda k: (str(k[0]), str(k[1])))
    va: list[int] = []
    for b in rng.permutation(len(keys)):
        if len(va) >= n_val:
            break
        va.extend(groups[keys[b]])
    vs = set(va)
    return [i for i in range(n_docs) if i not in vs], va


def causal_context(emb: torch.Tensor, so: np.ndarray, dev: str) -> torch.Tensor:
    """문서별 **과거 포함 누적평균** (L2 정규화). c_t = normalize(mean(z_0..z_t))."""
    ctx = torch.empty_like(emb)
    for i in range(len(so) - 1):
        s0, s1 = int(so[i]), int(so[i + 1])
        if s1 <= s0:
            continue
        seg = emb[s0:s1].float()
        cs = torch.cumsum(seg, 0) / torch.arange(
            1, s1 - s0 + 1, device=dev, dtype=torch.float32).unsqueeze(1)
        ctx[s0:s1] = torch.nn.functional.normalize(cs, dim=1).to(emb.dtype)
    return ctx


def trans_left_index(so: np.ndarray, docs: list[int]) -> np.ndarray:
    """문서 목록 -> 전이의 왼쪽 문장 전역 인덱스 배열."""
    return np.concatenate([np.arange(int(so[i]), int(so[i + 1]) - 1) for i in docs])


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train_one(variant: str, seed: int, emb: torch.Tensor, ctx: torch.Tensor | None,
              y_all: torch.Tensor, tr_ix: np.ndarray, va_ix: np.ndarray,
              args, dev: str) -> tuple[DirectBoundaryHead, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = 768 * (5 if variant == "ctx" else 4)
    model = DirectBoundaryHead(in_dim, args.hidden, args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()          # pos_weight = 1 (W3 승계)
    tr_t = torch.from_numpy(tr_ix).to(dev)
    rng = np.random.default_rng(seed)
    hist = []
    best_vl, best_ep, bad, best_state = float("inf"), 0, 0, None
    for ep in range(args.epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(tr_ix))).to(dev)
        tot, nb = 0.0, 0
        for b in range(0, len(order), args.batch):
            j = tr_t[order[b:b + args.batch]]
            x = feats(emb[j].float(), emb[j + 1].float(),
                      ctx[j].float() if ctx is not None else None)
            loss = lossf(model(x), y_all[j].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        ap, auc = anchor_val(model, emb, ctx, y_all, va_ix, dev, args.batch)
        vl = val_loss(model, emb, ctx, y_all, va_ix, dev, args.batch, lossf)
        hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1), "val_loss": vl,
                     "val_auprc": ap, "val_auc": auc})
        print(f"[M8-{variant} seed{seed}] ep{ep+1}/{args.epochs} train {tot/max(nb,1):.4f} "
              f"val {vl:.4f} | 앵커 val AUPRC {ap:.4f} AUC {auc:.4f}", flush=True)
        # 🔒 조기종료 — M5/M4 와 **같은 규약**(val 손실 · patience · 최적 가중치 복원).
        #    M8 만 100 epoch 완주하면 동일조건 비교가 깨진다.
        if vl < best_vl - args.min_delta:
            best_vl, best_ep, bad = vl, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                print(f"[M8-{variant} seed{seed}] ⏹ 조기종료 ep{ep+1} "
                      f"(best ep{best_ep} val {best_vl:.4f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"n_params": n_par, "history": hist,
                   "val_loss_best": best_vl, "val_loss_best_epoch": best_ep}


@torch.no_grad()
def val_loss(model, emb, ctx, y_all, va_ix, dev: str, bs: int, lossf) -> float:
    """val 손실 — 학습과 **같은 목적함수**. 조기종료 기준(M5/M4 와 동일 규약)."""
    model.eval()
    ix = torch.from_numpy(va_ix).to(dev)
    tot, nb = 0.0, 0
    for b in range(0, len(ix), bs):
        j = ix[b:b + bs]
        x = feats(emb[j].float(), emb[j + 1].float(),
                  ctx[j].float() if ctx is not None else None)
        tot += float(lossf(model(x), y_all[j].float())); nb += 1
    model.train()
    return tot / max(nb, 1)


@torch.no_grad()
def anchor_val(model, emb, ctx, y_all, va_ix: np.ndarray, dev: str, bs: int) -> tuple[float, float]:
    """앵커 val 의사-gold(count≥3) 대비 AUPRC/AUC. 사람 경계가 아니다 — 학습 진단용."""
    model.eval()
    ix = torch.from_numpy(va_ix).to(dev)
    ps = []
    for b in range(0, len(ix), 65536):
        j = ix[b:b + 65536]
        x = feats(emb[j].float(), emb[j + 1].float(),
                  ctx[j].float() if ctx is not None else None)
        ps.append(torch.sigmoid(model(x)).cpu().numpy())
    p = np.concatenate(ps)
    y = y_all[ix].cpu().numpy()
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


# ── dev-3 평가 (M5/M4 와 동일한 규칙 함수) ────────────────────────────────────
def cuts_adaptive(s: np.ndarray) -> np.ndarray:
    """동결 규약: 0번 sentinel 을 앞에 붙여 넘긴다. 재구현 금지 — 기존 함수 호출."""
    seq = np.concatenate([[0.0], s]).tolist()
    return np.asarray(adaptive_boundaries(seq, c=C_MAIN, mode="online_nf", warmup=5),
                      np.int8)[:len(s)]


def cuts_pct(s: np.ndarray, rate: float = ANCHOR_R) -> np.ndarray:
    return (s >= np.quantile(s, 1.0 - rate)).astype(np.int8)


def score_signals(S_list: list[np.ndarray], G_list: list[np.ndarray]) -> dict:
    """연속 신호 리스트 -> 두 임계 규칙의 deployable Score + 보조 AUC."""
    out = {}
    for rule, fn in (("adaptive", cuts_adaptive), ("percentile", cuts_pct)):
        ys, ps, n = [], [], 0
        for s, g in zip(S_list, G_list):
            b = fn(s)[:len(g)]
            ys.append(np.concatenate([g, [0]]).tolist())
            ps.append(np.concatenate([b, [0]]).tolist())
            n += int(b.sum())
        out[rule] = {"score": float(score_dialogues(ys, ps)["score"]), "n_pred": n}
    out["auc_pooled"] = float(roc_auc_score(np.concatenate(G_list), np.concatenate(S_list)))
    return out


@torch.no_grad()
def m8_dev3_signal(model, variant: str, dev3: Corpus, dev: str) -> list[np.ndarray]:
    model.eval()
    emb = torch.from_numpy(np.asarray(dev3.emb, dtype=np.float32)).to(dev)
    ctx = causal_context(emb, dev3.so, dev) if variant == "ctx" else None
    sig = []
    for i in range(dev3.n_docs):
        s0, s1 = int(dev3.so[i]), int(dev3.so[i + 1])
        j = torch.arange(s0, s1 - 1, device=dev)
        x = feats(emb[j], emb[j + 1], ctx[j] if ctx is not None else None)
        sig.append(torch.sigmoid(model(x)).cpu().numpy())
    return sig


def eval_reference(tag: str, ns: str, seeds, dev3: Corpus, G: list[np.ndarray], dev: str) -> dict:
    """M5/M4 를 **같은 규칙 함수**로 그 자리에서 재평가(문서 수치 전사 금지)."""
    from db29_final_eval import load
    res = []
    for sd in seeds:
        ck = REPO / f"outputs/experiments/m4m5_phase1_anchor/{tag}{ns}_seed{sd}.pt"
        if not ck.exists():
            print(f"  ⚠️ 없음: {ck.name}", flush=True)
            continue
        m = load(tag, sd, dev, ns=ns)[0]
        with torch.no_grad():
            P = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=4, with_anchor=False)
        r = score_signals([p.sum(1) for p in P], G)
        r["seed"] = sd
        res.append(r)
        print(f"  [{tag}{ns} seed{sd}] adaptive {r['adaptive']['score']:.4f} "
              f"({r['adaptive']['n_pred']}) · percentile {r['percentile']['score']:.4f} "
              f"({r['percentile']['n_pred']}) · AUC {r['auc_pooled']:.4f}", flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    return {"runs": res, **agg(res)}


def agg(runs: list[dict]) -> dict:
    if not runs:
        return {}
    o = {}
    for rule in ("adaptive", "percentile"):
        v = [r[rule]["score"] for r in runs]
        o[rule] = {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                   "n_pred_mean": float(np.mean([r[rule]["n_pred"] for r in runs]))}
    o["auc_mean"] = float(np.mean([r["auc_pooled"] for r in runs]))
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description="M8-DirectBoundary 학습 + dev-3 평가")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3,
                    help="조기종료 — val 손실 기준(0=끄기). M5/M4 와 동일 규약.")
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--variants", nargs="*", default=["pair", "ctx"])
    # 🔒 M5/M4 와 **같은 앵커**(의도 축 I2). 셋의 조건이 어긋나면 비교가 성립하지 않는다.
    ap.add_argument("--split", default="phase1__personkey_i2")
    ap.add_argument("--author-map",
                    default=str(REPO / "data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl"),
                    help="작가 단위 분할용 doc_id→저자 매핑. 빈 문자열이면 무작위 분할(구 조건).")
    ap.add_argument("--anchor-rate", type=float, default=0.0660,
                    help="적응임계 분위수 = I2 앵커 경계율. frozen 축은 0.0657, 구 앵커는 0.0642(D-B30).")
    ap.add_argument("--limit-docs", type=int, default=0, help="스모크용. 0 = 전량")
    ap.add_argument("--out-suffix", default="",
                    help="🔒 조건 네임스페이스. 스모크·변형 런은 반드시 이걸로 산출을 분리한다")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global OUT
    OUT = OUT.with_name(OUT.name + (f"_{a.out_suffix}" if a.out_suffix else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    global ANCHOR_R
    ANCHOR_R = a.anchor_rate          # 새 앵커 경계율로 갈아끼운다(조건이 산출에 기록된다)
    print(f"device={dev} · split={a.split} · anchor_r={ANCHOR_R}", flush=True)

    c = Corpus(a.split)
    dev3 = Corpus("dev3")
    n = c.n_docs if not a.limit_docs else min(a.limit_docs, c.n_docs)
    tr_docs, va_docs = doc_split(n, c.doc_ids, a.author_map)
    tr_ix, va_ix = trans_left_index(c.so, tr_docs), trans_left_index(c.so, va_docs)
    n_sent = int(c.so[n])
    print(f"[data] 문서 train {len(tr_docs):,} / val {len(va_docs):,} · "
          f"전이 train {len(tr_ix):,} / val {len(va_ix):,}", flush=True)

    emb = torch.from_numpy(np.asarray(c.emb[:n_sent], dtype=np.float16)).to(dev)
    y_all = torch.zeros(n_sent, dtype=torch.int8, device=dev)
    # 전역 문장 인덱스 -> 라벨. 전이 t 의 라벨을 왼쪽 문장 인덱스에 싣는다.
    for i in range(n):
        s0 = int(c.so[i])
        t0, t1 = int(c.to[i]), int(c.to[i + 1])
        yb = (c.resets[t0:t1].sum(1) >= T_STAR).astype(np.int8)
        y_all[s0:s0 + len(yb)] = torch.from_numpy(yb).to(dev)
    rate = float(y_all[torch.from_numpy(tr_ix).to(dev)].float().mean())
    print(f"[data] 앵커 경계율(train) {rate:.4f} · 동결 분위수 r={ANCHOR_R}", flush=True)
    ctx = causal_context(emb, c.so[:n + 1], dev) if "ctx" in a.variants else None

    G = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8) for i in range(dev3.n_docs)]
    print(f"[data] dev-3 {dev3.n_docs}편 · gold 경계 {sum(int(g.sum()) for g in G)}", flush=True)

    results: dict = {"_meta": {"caveat": CAVEAT, "t_star": T_STAR, "c_adaptive": C_MAIN,
                               "anchor_r": ANCHOR_R, "split_seed": SPLIT_SEED,
                               "anchor_boundary_rate_train": rate,
                               "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
                               "hidden": a.hidden, "dropout": a.dropout,
                               "n_docs_train": len(tr_docs), "n_docs_val": len(va_docs),
                               "n_trans_train": int(len(tr_ix)), "n_trans_val": int(len(va_ix))}}

    for variant in a.variants:
        runs = []
        for sd in a.seeds:
            model, info = train_one(variant, sd, emb, ctx if variant == "ctx" else None,
                                    y_all, tr_ix, va_ix, a, dev)
            sig = m8_dev3_signal(model, variant, dev3, dev)
            r = score_signals(sig, G)
            r.update({"seed": sd, "n_params": info["n_params"],
                      "val_auprc": info["history"][-1]["val_auprc"],
                      "val_auc": info["history"][-1]["val_auc"], "history": info["history"]})
            runs.append(r)
            torch.save({"state": model.state_dict(), "variant": variant, "seed": sd,
                        "args": vars(a)}, OUT / f"M8_{variant}_seed{sd}.pt")
            print(f"  [M8-{variant} seed{sd}] dev-3 adaptive {r['adaptive']['score']:.4f} "
                  f"({r['adaptive']['n_pred']}) · percentile {r['percentile']['score']:.4f} "
                  f"({r['percentile']['n_pred']}) · AUC {r['auc_pooled']:.4f}", flush=True)
        results[f"M8_{variant}"] = {"runs": runs, **agg(runs)}

    print("\n[참조군 재평가 — 같은 규칙 함수]", flush=True)
    results["M5_W3"] = eval_reference("M5", "_per_slot_slot_pwone", a.seeds, dev3, G, dev)
    results["M4_W0"] = eval_reference("M4", "_per_slot_slot", a.seeds, dev3, G, dev)

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'results.json'}")

    # 사전등록 §4 판정 (기준은 실행 전에 고정됨 — 여기서 바꾸지 않는다)
    best = max((k for k in results if k.startswith("M8_")),
               key=lambda k: results[k]["adaptive"]["mean"] + results[k]["percentile"]["mean"])
    print(f"\n=== 판정 (⚠️ {CAVEAT}) · M8 대표 = {best} (두 변종 중 높은 쪽) ===")
    for rule in ("adaptive", "percentile"):
        d = results["M5_W3"][rule]["mean"] - results[best][rule]["mean"]
        print(f"  {rule:11s}  M5 {results['M5_W3'][rule]['mean']:.4f}"
              f" · M8 {results[best][rule]['mean']:.4f} · Δ(M5−M8) {d:+.4f}")


if __name__ == "__main__":
    main()
