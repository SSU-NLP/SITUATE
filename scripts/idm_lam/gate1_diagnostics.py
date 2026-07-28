"""Gate 1: does the IDM code carry TRANSITION CHARACTER, not next-utterance CONTENT?

Runs on superseg_validation transitions (labels used ONLY for diagnosis, not training).

Diagnostics:
  1. Codebook utilization: usage histogram + perplexity (collapse check).
  2. CODE CONTRIBUTION (is the code even used?): MSE of FDM with the real code vs
     MSE with the code index SHUFFLED across transitions. If shuffling barely hurts,
     the FDM ignores the code -> IDM learned nothing -> NO-GO.
  3. LEAKAGE: normalized MI(code; content-topic-cluster) vs MI(code; boundary-label).
     + linear-probe R^2 of one-hot code -> z_{t+1} (content) vs -> (z_{t+1}-z_t) (change).
     High content-MI / content-R^2 = code smuggles content = leakage = fail.
  4. Qualitative: 12 sample transitions per code (text), written to a file to READ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dts_common import embed_split  # noqa: E402
from lam import LAM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"


def entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_info(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (MI, normalized MI) between discrete labels a and b."""
    ka, kb = a.max() + 1, b.max() + 1
    joint = np.zeros((ka, kb))
    for x, y in zip(a, b):
        joint[x, y] += 1
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = joint * np.log(joint / (pa @ pb))
    mi = np.nansum(mi[joint > 0])
    ha = entropy(joint.sum(1))
    hb = entropy(joint.sum(0))
    nmi = mi / max(min(ha, hb), 1e-9)
    return float(mi), float(nmi)


def ridge_r2(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> float:
    """Multi-output ridge R^2 (train==eval, in-sample upper bound of recoverability)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb.T @ Y)
    pred = Xb @ W
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="K16")
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(OUT / f"lam_{args.tag}.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
                commitment_weight=a["commitment"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    K = a["codebook_size"]

    va = embed_split("superseg_validation")
    # gather transitions: z_t, z_tp1, boundary label (turn t+1 starts new seg), texts
    Zt, Ztp1, blabel, txt_t, txt_tp1 = [], [], [], [], []
    for d in va:
        e, turns, bnd = d["emb"], d["turns"], d["boundaries"]
        for t in range(len(turns) - 1):
            Zt.append(e[t]); Ztp1.append(e[t + 1])
            blabel.append(1 if (t + 1) in bnd else 0)
            txt_t.append(turns[t]); txt_tp1.append(turns[t + 1])
    Zt = np.stack(Zt); Ztp1 = np.stack(Ztp1); blabel = np.array(blabel)
    zt = torch.tensor(Zt, device=dev); ztp1 = torch.tensor(Ztp1, device=dev)

    with torch.no_grad():
        _, q, idx, _ = model.encode_transition(zt, ztp1)
        idx = idx.cpu().numpy()
        # real-code recon
        zhat_real = model.fdm(torch.cat([zt, q], -1))
        mse_real = ((zhat_real - ztp1) ** 2).sum(-1).mean().item()
        # shuffled-code recon (break code<->transition link)
        perm = torch.randperm(len(q), device=dev)
        zhat_shuf = model.fdm(torch.cat([zt, q[perm]], -1))
        mse_shuf = ((zhat_shuf - ztp1) ** 2).sum(-1).mean().item()
        # zero-code recon (code vec = mean code)
        qmean = q.mean(0, keepdim=True).expand_as(q)
        zhat_zero = model.fdm(torch.cat([zt, qmean], -1))
        mse_zero = ((zhat_zero - ztp1) ** 2).sum(-1).mean().item()

    # ---- 1. utilization
    counts = np.bincount(idx, minlength=K)
    ppl = float(np.exp(entropy(counts)))

    # ---- content-topic cluster proxy (kmeans on z_{t+1}) for MI/leakage
    from sklearn.cluster import KMeans
    ntop = 30
    topic = KMeans(n_clusters=ntop, n_init=4, random_state=0).fit_predict(Ztp1)

    mi_topic, nmi_topic = mutual_info(idx, topic)
    mi_bnd, nmi_bnd = mutual_info(idx, blabel)

    # ---- leakage via linear probe R^2 (one-hot code -> content vs change)
    onehot = np.eye(K)[idx]
    r2_content = ridge_r2(onehot, Ztp1)          # code -> next content
    r2_change = ridge_r2(onehot, Ztp1 - Zt)       # code -> transition delta
    # reference ceilings: z_t -> next content (how much content is just carryover)
    r2_zt_content = ridge_r2(Zt, Ztp1)

    # ---- per-code boundary rate (which codes are "shift" codes)
    code_bnd_rate = {int(k): float(blabel[idx == k].mean()) if (idx == k).any() else 0.0
                     for k in range(K)}
    base_rate = float(blabel.mean())

    # ---- FAIR CONTROL: compare code vs a surprise signal binned to K bins.
    #   delta = 1 - cos(z_t, z_{t+1}); quantile-bin into K bins.
    #   If code has HIGHER boundary-NMI than delta-bin at similar topic-NMI, the
    #   code is a *better transition signal* than surprise, not just content.
    delta = 1.0 - (Zt * Ztp1).sum(1)  # embeddings are L2-normalized -> dot=cos
    qs = np.quantile(delta, np.linspace(0, 1, K + 1)[1:-1])
    dbin = np.digitize(delta, qs)
    mi_d_topic, nmi_d_topic = mutual_info(dbin, topic)
    mi_d_bnd, nmi_d_bnd = mutual_info(dbin, blabel)

    # ---- CONDITIONAL: boundary-NMI computed WITHIN each topic then averaged.
    #   Removes the content confound: does code separate transition type *given* topic?
    def conditional_nmi(code, tgt, cond):
        vals, tot = [], 0
        for c in np.unique(cond):
            m = cond == c
            if m.sum() < 30 or len(np.unique(tgt[m])) < 2 or len(np.unique(code[m])) < 2:
                continue
            _, nmi = mutual_info(code[m].astype(int), tgt[m].astype(int))
            vals.append(nmi * m.sum()); tot += m.sum()
        return float(sum(vals) / tot) if tot else 0.0

    cnmi_code_bnd = conditional_nmi(idx, blabel, topic)      # code->boundary | topic
    cnmi_delta_bnd = conditional_nmi(dbin, blabel, topic)    # delta->boundary | topic (surprise ref)

    report = {
        "tag": args.tag, "K": K, "n_transitions": len(idx),
        "utilization": {"counts": counts.tolist(), "perplexity": ppl, "used": int((counts > 0).sum())},
        "code_contribution": {
            "mse_real_code": mse_real, "mse_shuffled_code": mse_shuf, "mse_mean_code": mse_zero,
            "contribution_vs_shuffle": mse_shuf - mse_real,
            "contribution_vs_mean": mse_zero - mse_real,
        },
        "leakage": {
            "nmi_code_topic(content)": nmi_topic, "nmi_code_boundary(transition)": nmi_bnd,
            "r2_code->content": r2_content, "r2_code->change": r2_change,
            "r2_zt->content(ref)": r2_zt_content,
        },
        "boundary": {"base_rate": base_rate, "per_code_boundary_rate": code_bnd_rate},
        "fair_control": {
            "nmi_code_topic": nmi_topic, "nmi_code_boundary": nmi_bnd,
            "nmi_deltabin_topic": nmi_d_topic, "nmi_deltabin_boundary": nmi_d_bnd,
            "cond_nmi_code_boundary|topic": cnmi_code_bnd,
            "cond_nmi_deltabin_boundary|topic": cnmi_delta_bnd,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gate1_{args.tag}.json").write_text(json.dumps(report, indent=2))

    # ---- qualitative dump
    lines = [f"# Gate-1 qualitative: sample transitions per code ({args.tag}, K={K})\n"]
    rng = np.random.default_rng(0)
    order = sorted(range(K), key=lambda k: -code_bnd_rate[k])
    for k in order:
        pos = np.where(idx == k)[0]
        lines.append(f"\n## code {k}  (n={len(pos)}, boundary_rate={code_bnd_rate[k]:.2f} vs base {base_rate:.2f})")
        for j in rng.choice(pos, size=min(12, len(pos)), replace=False):
            b = "  <<BND" if blabel[j] else ""
            lines.append(f"  [{txt_t[j][:70]}]  ->  [{txt_tp1[j][:70]}]{b}")
    (OUT / f"gate1_samples_{args.tag}.txt").write_text("\n".join(lines))

    # ---- console summary
    print(f"=== Gate-1 {args.tag} (K={K}, {len(idx)} transitions) ===")
    print(f"[1] utilization: {int((counts>0).sum())}/{K} used, perplexity {ppl:.2f}/{K}")
    print(f"    counts: {counts.tolist()}")
    print(f"[2] CODE CONTRIBUTION: mse real={mse_real:.4f} shuffled={mse_shuf:.4f} mean={mse_zero:.4f}")
    print(f"    -> code helps by {mse_shuf-mse_real:.4f} (vs shuffle), {mse_zero-mse_real:.4f} (vs mean)")
    print(f"[3] LEAKAGE:  NMI(code;topic-content)={nmi_topic:.3f}  NMI(code;boundary-transition)={nmi_bnd:.3f}")
    print(f"    R2 code->content={r2_content:.3f}  code->change={r2_change:.3f}  [ref z_t->content={r2_zt_content:.3f}]")
    print(f"[4] boundary base rate {base_rate:.3f}; per-code rate range "
          f"{min(code_bnd_rate.values()):.2f}..{max(code_bnd_rate.values()):.2f}")
    print(f"[5] FAIR CONTROL vs surprise(delta-bin, K bins):")
    print(f"    topic-NMI:    code={nmi_topic:.3f}  delta-bin={nmi_d_topic:.3f}")
    print(f"    boundary-NMI: code={nmi_bnd:.3f}  delta-bin={nmi_d_bnd:.3f}")
    print(f"    COND boundary-NMI | topic:  code={cnmi_code_bnd:.3f}  delta-bin={cnmi_delta_bnd:.3f}")
    print(f"    samples -> gate1_samples_{args.tag}.txt")


if __name__ == "__main__":
    main()
