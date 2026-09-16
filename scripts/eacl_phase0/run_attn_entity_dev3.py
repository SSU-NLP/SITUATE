#!/usr/bin/env python3
"""M9-AttnEntityDrift dev-3 실험 러너 (사전등록: `PREREG.md`).

단계:
  --selftest        배선 검증. GPU 불필요, ~5s. **Score 출력 없음.**
                    오라클 주입 → Score=1.0, no-boundary floor 0.27896 재현, 개수 assert.
  --stage cache     LM 1회 로드 → 문서별 attention mass / hidden / EM-LLM surprise 캐시.
  --stage precheck  attention 없는 통계만. **Score 출력 없음.** (F2 퇴화 사전탐지)
  --stage primary   사전등록 격자 36셀 + 통제군 5종. 이 시점 이후 기준 변경 금지.
  --stage secondary 부록 격자.

🔴 dev-3 = 연습장. 절대 Score 를 성능으로 인용하지 않는다. Tier1 봉인 시험지는 열지 않는다.
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
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

import baseline_compare as bc  # noqa: E402
import segment_attn_entity as sae  # noqa: E402
from segment_manual_emllm import DEFAULT_LM, encode_doc, refine_boundaries  # noqa: E402

DEFAULT_NAME = "2026-07-31_m9_attn_entity"
DEV3 = ["pieman", "tunnel", "monkey"]
REPR_LAYERS = (14, 18, 28)
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
OPS = ("OP-1", "OP-2", "OP-3")
SELECTIONS = ("attn-topk", "attn-spacy", "all-core")   # all-core = attention 없는 쌍둥이
FLOOR = 0.2789608866237916                              # 사전 확인값 (PREREG §2)


def out_dir(name: str) -> Path:
    return REPO / "outputs/experiments" / name


def cache_dir(name: str) -> Path:
    return out_dir(name) / "cache"


# ── 캐시 I/O ──────────────────────────────────────────────────────────────────
def save_tok(name: str, doc: str, dt: sae.DocTok) -> None:
    (cache_dir(name) / f"{doc}__tok.json").write_text(json.dumps({
        "input_ids": dt.input_ids, "spans": dt.spans, "core_spans": dt.core_spans,
        "core_mask": dt.core_mask.astype(int).tolist(), "sent_of": dt.sent_of.tolist(),
        "char_start": dt.char_start.tolist(), "text_of": dt.text_of, "sents": dt.sents,
    }, ensure_ascii=False), encoding="utf-8")


def load_tok(name: str, doc: str) -> sae.DocTok:
    j = json.loads((cache_dir(name) / f"{doc}__tok.json").read_text(encoding="utf-8"))
    return sae.DocTok(j["input_ids"], [tuple(x) for x in j["spans"]],
                      [tuple(x) for x in j["core_spans"]],
                      np.asarray(j["core_mask"], bool), np.asarray(j["sent_of"], np.int32),
                      np.asarray(j["char_start"], np.int32), j["text_of"], j["sents"])


def a_path(name: str, doc: str, scope: str, norm: str) -> Path:
    return cache_dir(name) / f"{doc}__A__{scope.replace(':', '')}__{norm}.npz"


# ── spaCy 필터 ────────────────────────────────────────────────────────────────
def spacy_allow(name: str, doc: str, dt: sae.DocTok):
    """spaCy 내용어/NER 마스크 [T] + lemma 맵. 문장 sha1 대조로 로딩 일치를 강제한다."""
    import hashlib

    p = cache_dir(name) / "dev3_spacy_tags.json"
    tags = json.loads(p.read_text(encoding="utf-8"))["docs"][doc]
    sha = [hashlib.sha1(s.encode()).hexdigest()[:12] for s in dt.sents]
    assert sha == tags["sent_sha"], (
        f"{doc}: spaCy 캐시의 문장이 baseline_compare.load_dataset 와 다르다 "
        f"(별도 venv 로딩 복제가 어긋남). 캐시를 다시 만들 것.")
    allow = np.zeros(dt.n_tok, bool)
    lemma: dict[int, dict[int, str]] = {}
    n_hit = n_miss = 0
    for p_i in range(dt.n_tok):
        c = int(dt.char_start[p_i])
        if c < 0:
            continue
        i = int(dt.sent_of[p_i])
        st = tags["tags"][i]
        m = next((t for t in st if t["s"] <= c < t["e"]), None)
        if m is None:
            n_miss += 1
            continue
        n_hit += 1
        if m["pos"] in CONTENT_POS or m["ent"]:
            allow[p_i] = True
        lemma.setdefault(i, {})[c] = m["lemma"]
    return allow, lemma, {"n_hit": n_hit, "n_miss": n_miss}


# ── 신호 조립 ─────────────────────────────────────────────────────────────────
def build_d(dt, A, Hr, allow, *, signal: str, selection: str, layer_idx, head_agg="mean",
            topk=sae.DEFAULT_TOPK, topk_mode="fixed", frac=0.25, entity_unit="word",
            cap=sae.DEFAULT_CAP, l2=True, center=False, lemma=None, seed=None,
            setw=sae.DEFAULT_SETW):
    """한 문서의 거리 신호 d [n] + 진단치."""
    a_t = sae.aggregate(A, layer_idx=layer_idx, head_agg=head_agg)
    rank, alw = {"attn-topk": ("attn", None), "attn-spacy": ("attn", allow),
                 "all-core": ("all", None), "shuffled": ("shuffled", None),
                 # all-spacy = spaCy 내용어/NER **전부** (attention 무관, top-k 없음).
                 # "완벽에 가까운 엔티티 추출기를 가정했을 때의 천장" 측정용 — 사후 추가(§ceiling).
                 "all-spacy": ("all", allow)}[selection]
    rng = np.random.default_rng(seed) if rank == "shuffled" else None
    picks, st = sae.select_topk(a_t, dt, k=topk, mode=topk_mode, frac=frac,
                                allow=alw, rank=rank, rng=rng)
    keys = sae.entity_keys(dt, picks, unit=entity_unit, lemma=lemma)
    d = sae.build_signal(signal, Hr=Hr, picks=picks, keys=keys, cap=cap, l2=l2, center=center,
                         setw=setw)
    r = np.asarray(d, np.float64)[1:]
    diag = dict(st)
    diag.update({
        "std": float(np.std(r)), "n_distinct": int(len(np.unique(np.round(r, 9)))),
        "frac_ties_max": float(np.mean(np.isclose(r, r.max()))) if r.size else 1.0,
        "degenerate": bool(sae.is_degenerate(d)),
        "mean_entities": float(np.mean([len(k) for k in keys])),
    })
    return d, diag, keys, picks


def agg_diag(per_doc: dict) -> dict:
    """문서별 진단치 → 셀 요약 (최악값 기준 — 한 문서만 퇴화해도 셀은 퇴화)."""
    ks = ["frac_truncated", "std", "n_distinct", "frac_ties_max", "mean_entities"]
    o = {}
    for k in ks:
        v = [d[k] for d in per_doc.values()]
        o[k] = (min(v) if k in ("std", "n_distinct") else max(v))
    o["degenerate"] = any(d["degenerate"] for d in per_doc.values())
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}


def is_degenerate_cell(diag: dict) -> str | None:
    """PREREG §5 DEGENERATE 규칙. 해당하면 사유 문자열."""
    if diag["degenerate"]:
        return "std<1e-9"
    if diag["frac_truncated"] > 0.30:
        return f"frac_truncated={diag['frac_truncated']:.2f}>0.30"
    if diag["frac_ties_max"] > 0.5:
        return f"frac_ties_max={diag['frac_ties_max']:.2f}>0.5"
    return None


# ── 통제군 ────────────────────────────────────────────────────────────────────
def preds_random(data, counts: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for did, _, g in data:
        v = np.zeros(len(g), np.int8)
        c = int(min(counts.get(did, 0), len(g)))
        if c:
            v[rng.choice(len(g), size=c, replace=False)] = 1
        out[did] = v
    return out


def preds_m1(data) -> dict[str, np.ndarray]:
    """M1-CosDrift δ_eff — `baseline_compare.preds_m1:127-134` 수식 verbatim, 3 OP 용 raw 신호."""
    e_all = np.load(REPO / "outputs/experiments/m4m5_phase1_anchor/data/dev3_emb.npy")
    idx = np.load(REPO / "outputs/experiments/m4m5_phase1_anchor/data/dev3_index.npz")
    so, ids = idx["sent_offsets"], [str(x) for x in idx["doc_ids"]]
    out = {}
    for i, did in enumerate(ids):
        e = np.asarray(e_all[int(so[i]):int(so[i + 1])], np.float32)
        d = np.zeros(len(e))
        for t in range(1, len(e)):
            dp = 1.0 - float(e[t - 1] @ e[t])
            lo = max(0, t - bc.M1["m"])
            w = np.array([bc.M1["rho"] ** j for j in range(t - lo)])[::-1]
            c = (w[:, None] * e[lo:t]).sum(0)
            nrm = float(np.linalg.norm(c))
            dc = 1.0 - float(c @ e[t] / nrm) if nrm > 0 else dp
            d[t] = bc.M1["a"] * dp + (1 - bc.M1["a"]) * dc
        out[did] = d
    return out


# ── 1단계: 배선 self-test ─────────────────────────────────────────────────────
def selftest(data) -> None:
    """F3(off-by-one) 방지. Score 를 출력하지 않는다 — 통과/실패만."""
    exp = {"pieman": (96, 18), "tunnel": (504, 18), "monkey": (408, 30)}
    for did, sents, g in data:
        n, k = exp[did]
        assert len(sents) == n, f"{did}: 문장 {len(sents)} != {n}"
        assert len(g) == n - 1, f"{did}: 전이 {len(g)} != {n - 1}"
        assert int(g.sum()) == k, f"{did}: gold {int(g.sum())} != {k}"
    tot = sum(len(g) for _, _, g in data)
    pos = sum(int(g.sum()) for _, _, g in data)
    assert (tot, pos) == (1005, 66), f"전이/gold 합 불일치: {tot}/{pos}"

    r = bc.score_A(data, {d: np.zeros(len(g), np.int8) for d, _, g in data})
    assert abs(r["score"] - FLOOR) < 1e-12, f"no-boundary floor 재현 실패: {r['score']}"

    # (a) **정렬 검사 (F3 본체)** — 순수 임계 0.5 로 오라클 신호를 통과시킨다.
    #     d[i] = dist(i-1,i) 규약이 맞다면 결과는 gold 와 **정확히** 같아야 한다.
    #     OP 를 끼우지 않는 이유: 작동점의 발화 규칙이 정렬 검증을 오염시키기 때문(아래 (b) 참조).
    preds = {}
    for did, _, g in data:
        d = np.concatenate([[0.0], g.astype(np.float64)])
        preds[did] = (d[1:] >= 0.5).astype(np.int8)
    for did, _, g in data:
        assert np.array_equal(preds[did], g), f"{did}: 전이 정렬 어긋남 (off-by-one)"
    s = bc.score_A(data, preds)
    assert abs(s["score"] - 1.0) < 1e-12 and abs(s["f1"] - 1.0) < 1e-12, (
        f"오라클 Score {s['score']:.6f} != 1.0 — score_A 배선이 어긋났다")

    # (b) **OP 배선 검사 — shift 대조.** 각 OP 에 대해 "올바른 정렬"과 "1칸 민 정렬"의 Score 를
    #     둘 다 재고, 올바른 쪽이 크게 앞서야 한다.
    #     절대 Score 로 문턱을 두지 않는 이유 — 둘 다 배선이 아니라 **동결 작동점의 성질**이다:
    #       · OP-1(`adaptive_boundaries` online_nf): running std=0 인 접두부에서 `>=` 때문에
    #         0 에도 발화 → 이진 오라클에서 과예측(72 vs 66). REPORT §3 에 성질로 기록한다.
    #       · OP-3(고정 분위수 6.42%): 문서별 gold 밀도(pieman 0.19 / tunnel 0.036)와 원리적으로
    #         안 맞으므로 이진 오라클에서 1.0 이 나올 수 없다.
    #     반면 1칸 shift 는 F1 을 붕괴시키므로 이 대조는 off-by-one 에 결정적이다.
    for op in OPS:
        for fn, tag in ((sae.signal_to_boundaries, "signal"),
                        (sae.surprisal_to_boundaries, "surprisal")):
            ok, sh = {}, {}
            for did, _, g in data:
                d = np.concatenate([[0.0], g.astype(np.float64)])
                v = fn(d, op)
                assert len(v) == len(g), f"{tag} {op} {did}: 길이 {len(v)} != {len(g)}"
                ok[did] = v
                sh[did] = np.concatenate([[0], v[:-1]]).astype(np.int8)   # 1칸 민 대조군
            s_ok = bc.score_A(data, ok)["score"]
            s_sh = bc.score_A(data, sh)["score"]
            assert s_ok > s_sh + 0.2, (
                f"{tag} {op}: 정렬 Score {s_ok:.4f} vs 1칸-shift {s_sh:.4f} — 차이가 작다. "
                f"transition 정렬을 의심할 것")

    # 퇴화 가드: 상수 신호가 전 구간 경계로 터지지 않는가
    for op in OPS:
        p = {d: sae.signal_to_boundaries(np.r_[0.0, np.ones(len(g))], op) for d, _, g in data}
        assert sum(int(v.sum()) for v in p.values()) == 0, f"{op}: 상수 신호가 경계를 냈다"

    (out_dir("_selftest_marker").parent / ".m9_selftest_ok").write_text(str(time.time()))
    print("[self-test] PASS — 문장/gold 개수(96/504/408, 18/18/30), floor 0.27896 재현, "
          "정렬 오라클 Score=1.0(임계 0.5), shift 대조 3 OP × 2 경로, 상수신호 퇴화가드")


def selftest_ok() -> bool:
    return (REPO / "outputs/experiments/.m9_selftest_ok").exists()


# ── 2단계: 캐시 구축 ──────────────────────────────────────────────────────────
def build_cache(name: str, data, lm: str, docs: list[str], scopes: list[str],
                norms: list[str]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir(name).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(lm)
    model = AutoModelForCausalLM.from_pretrained(lm, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    n_layer = model.config.num_hidden_layers
    print(f"[LM] {lm} loaded in {time.time()-t0:.0f}s  layers={n_layer} "
          f"heads={model.config.num_attention_heads} sliding={model.config.use_sliding_window}")
    all_layers = list(range(n_layer))       # 전 레이어 저장 → 레이어대역 ablation 재추출 불필요

    for did, sents, _ in data:
        if did not in docs:
            continue
        dt = sae.tokenize_doc(sents, tok)
        assert_token_offsets(dt)
        save_tok(name, did, dt)
        print(f"[{did}] {dt.n_sent}문장 T={dt.n_tok} core={int(dt.core_mask.sum())}")

        hp = cache_dir(name) / f"{did}__hidden.npz"
        ep = cache_dir(name) / f"{did}__emllm.npz"
        for scope in scopes:
            for norm in norms:
                p = a_path(name, did, scope, norm)
                if p.exists():
                    print(f"  · skip {p.name} (존재)")
                    continue
                t1 = time.time()
                A, hs = sae.attention_mass(dt, model, device, layers=all_layers,
                                           query_scope=scope, norm=norm,
                                           repr_layers=REPR_LAYERS)
                np.savez_compressed(p, A=A.astype(np.float16))
                if not hp.exists():
                    np.savez_compressed(hp, **{f"L{k}": v for k, v in hs.items()})
                print(f"  · {p.name}  {time.time()-t1:.0f}s  A{A.shape}")
                del A, hs
        if not ep.exists():
            t1 = time.time()
            surprise, A_sim = encode_doc(sents, tok, model, device, refine_from_layer=0)
            np.savez_compressed(ep, surprise=surprise, A_sim=A_sim.float().cpu().numpy())
            print(f"  · {ep.name}  {time.time()-t1:.0f}s (EM-LLM surprisal 통제군)")

        dg = diagnostics(name, did, scopes[0], norms[0])
        print(f"  · [F1 진단] top1 위치 첫bin={dg['rel_first_bin']:.3f} 끝bin={dg['rel_last_bin']:.3f} "
              f"| top20점유={dg['top20_share']:.3f} | 불용어·구두점={dg['stop_or_punct_frac']:.3f} "
              f"| 절단={dg['frac_truncated']:.3f}  → {dg['REDLINE'] or 'PASS'}")
        print(f"    최빈 선택: {', '.join(w for w, _ in dg['top20'][:12])}")
        dp = out_dir(name) / "diagnostics.json"
        allj = json.loads(dp.read_text()) if dp.exists() else {}
        allj[f"{did}|{scopes[0]}|{norms[0]}"] = dg
        dp.write_text(json.dumps(allj, ensure_ascii=False, indent=2), encoding="utf-8")


STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those he she they them his her their i you we me my your our not no
so than then there here what which who whom when where how all any both each few more most other
some such only own same too very can will just do does did done have has had having would could
should i'm it's don't""".split())


def diagnostics(name: str, doc: str, scope: str, norm: str, topk: int = sae.DEFAULT_TOPK) -> dict:
    """F1(선택이 내용이 아니라 **위치 슬롯**) 사전탐지. Score 를 계산하지 않는다."""
    dt = load_tok(name, doc)
    A = np.load(a_path(name, doc, scope, norm))["A"].astype(np.float32)
    li = sae.parse_layers(sae.DEFAULT_LAYERS, A.shape[0])
    a_t = sae.aggregate(A, layer_idx=li, head_agg="mean")
    picks, st = sae.select_topk(a_t, dt, k=topk)

    rel, surf, n_stop, n_punct, n_all = [], [], 0, 0, 0
    for i, sel in enumerate(picks):
        a, b = dt.spans[i]
        pool = np.nonzero(dt.core_mask[a:b])[0] + a
        if pool.size == 0 or sel.size == 0:
            continue
        top1 = int(sel[np.argmax(a_t[sel])])
        rel.append((top1 - pool.min()) / max(pool.max() - pool.min(), 1))
        for p in sel:
            w = dt.text_of[p].strip().lower()
            surf.append(w)
            n_all += 1
            n_stop += w in STOPWORDS
            n_punct += sae._is_punct_or_space(w)
    hist = np.histogram(rel, bins=10, range=(0, 1))[0] / max(len(rel), 1)
    from collections import Counter
    top20 = Counter(surf).most_common(20)
    d = {
        "doc": doc, "scope": scope, "norm": norm, "topk": topk, "n_sent": dt.n_sent,
        "T": dt.n_tok, "rel_pos_hist10": [round(float(x), 4) for x in hist],
        "rel_first_bin": round(float(hist[0]), 4), "rel_last_bin": round(float(hist[-1]), 4),
        "top20": [[w, c] for w, c in top20],
        "top20_share": round(sum(c for _, c in top20) / max(n_all, 1), 4),
        "stop_or_punct_frac": round((n_stop + n_punct) / max(n_all, 1), 4),
        "frac_truncated": round(st["frac_truncated"], 4),
    }
    d["REDLINE"] = [r for r, hit in [
        ("rel_first_bin>0.5", d["rel_first_bin"] > 0.5),
        ("rel_last_bin>0.5", d["rel_last_bin"] > 0.5),
        ("top20_share>0.4", d["top20_share"] > 0.4),
        ("stop_or_punct>0.3", d["stop_or_punct_frac"] > 0.3),
    ] if hit]
    return d


def assert_token_offsets(dt: sae.DocTok) -> None:
    """문장-로컬 char offset 이 실제 원문과 맞는지 (sep 오프셋 어긋남 = 조용한 오염)."""
    bad = []
    for p in range(dt.n_tok):
        c = int(dt.char_start[p])
        if c < 0:
            continue
        i, frag = int(dt.sent_of[p]), dt.text_of[p].lstrip()
        if dt.sents[i][c:c + len(frag)] != frag:
            bad.append((p, i, c, repr(dt.text_of[p])))
    assert not bad, f"char offset 불일치 {len(bad)}건, 예: {bad[:3]}"


# ── 3단계: 사전점검 (Score 없음) ──────────────────────────────────────────────
def precheck(name: str, data) -> None:
    """F2(동적 범위 없음) 사전탐지. attention 불필요 → GPU 없이 CPU 로 돈다."""
    import hashlib

    p = cache_dir(name) / "dev3_spacy_tags.json"
    tags = json.loads(p.read_text(encoding="utf-8"))["docs"]
    rows = []
    for did, sents, g in data:
        st = tags[did]
        assert [hashlib.sha1(s.encode()).hexdigest()[:12] for s in sents] == st["sent_sha"]
        keys = [[t["lemma"] if t["lemma"] else "" for t in s
                 if t["pos"] in CONTENT_POS or t["ent"]] for s in st["tags"]]
        keys = [sorted(set(k for k in ks if k)) for ks in keys]
        cands = [("setdrift", sae.signal_setdrift(keys)),
                 ("setdrift_w6", sae.signal_setdrift_w(keys, w=6)),
                 ("recency", sae.signal_recency(keys, cap=sae.DEFAULT_CAP))]
        for nm, d in cands:
            r = d[1:]
            npred = {op: int(sae.signal_to_boundaries(d, op).sum()) for op in OPS}
            rows.append({
                "doc": did, "signal": nm, "n_sent": len(sents), "gold": int(g.sum()),
                "mean_entities": round(float(np.mean([len(k) for k in keys])), 2),
                "P(d=1)": round(float(np.mean(np.isclose(r, 1.0))), 4),
                "n_distinct": int(len(np.unique(np.round(r, 9)))),
                "std": round(float(np.std(r)), 5),
                "frac_ties_max": round(float(np.mean(np.isclose(r, r.max()))), 4),
                **{f"n_{o}": npred[o] for o in OPS},
            })
    print("\n=== 사전점검 (ALL-content-words, attention 없음, Score 미계산) ===")
    _print_rows(rows)
    print("\n레드라인(PREREG §6 F2): 어느 문서든 P(d=1) > 0.7 이면 그 집합신호 사양은 죽은 것.")
    print("DEGENERATE 문턱(PREREG §5): frac_ties_max > 0.5 / frac_truncated > 0.30 / std < 1e-9.")
    (out_dir(name) / "precheck.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def _print_rows(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    w = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" | ".join(str(c).ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


# ── 4·5단계: 격자 ─────────────────────────────────────────────────────────────
def _load_doc_ctx(name: str, data, scope: str, norm: str, repr_layer: int):
    """문서별 (dt, A, Hr, allow, lemma) 로드."""
    ctx = {}
    for did, _, _ in data:
        dt = load_tok(name, did)
        A = np.load(a_path(name, did, scope, norm))["A"].astype(np.float32)
        Hr = np.load(cache_dir(name) / f"{did}__hidden.npz")[f"L{repr_layer}"]
        allow, lemma, _ = spacy_allow(name, did, dt)
        ctx[did] = (dt, A, Hr, allow, lemma)
    return ctx


def run_grid(name: str, data, cells: list[dict], scope: str, norm: str, tag: str) -> dict:
    """셀 목록 → results dict. 신호는 OP 와 무관하므로 문서별로 한 번만 만든다."""
    results: dict[str, dict] = {}
    ctx_cache: dict[tuple, dict] = {}
    for cell in cells:
        sc, nm = cell.get("scope", scope), cell.get("norm", norm)
        rl = cell.get("repr_layer", sae.DEFAULT_REPR_LAYER)
        key = (sc, nm, rl)
        if key not in ctx_cache:
            ctx_cache[key] = _load_doc_ctx(name, data, sc, nm, rl)
        ctx = ctx_cache[key]
        n_layer = next(iter(ctx.values()))[1].shape[0]
        li = sae.parse_layers(cell.get("layers", sae.DEFAULT_LAYERS), n_layer)
        ds, dg = {}, {}
        for did, _, _ in data:
            dt, A, Hr, allow, lemma = ctx[did]
            d, diag, _, _ = build_d(
                dt, A, Hr, allow, signal=cell["signal"], selection=cell["selection"],
                layer_idx=li, head_agg=cell.get("head_agg", "mean"),
                topk=cell.get("topk", sae.DEFAULT_TOPK), topk_mode=cell.get("topk_mode", "fixed"),
                frac=cell.get("frac", 0.25), entity_unit=cell.get("entity_unit", "word"),
                cap=cell.get("cap", sae.DEFAULT_CAP), l2=cell.get("l2", True),
                center=cell.get("center", False),
                lemma=lemma if cell.get("lemma") else None, seed=cell.get("seed"))
            ds[did], dg[did] = d, diag
        diag = agg_diag(dg)
        deg = is_degenerate_cell(diag)
        pd_dir = out_dir(name) / "preds"
        pd_dir.mkdir(parents=True, exist_ok=True)
        for op in OPS:
            preds = {did: sae.signal_to_boundaries(ds[did], op) for did, _, _ in data}
            r = bc.score_A(data, preds)
            k = f"{cell['label']}|{op}"
            results[k] = {**cell, "op": op, **r, "diag": diag, "degenerate_reason": deg,
                          "n_pred_by_doc": {d: int(preds[d].sum()) for d, _, _ in data}}
            # 🔒 저장소 원칙: **예측 파일이 원자료**, 집계 Score 는 파생물 (재채점·감사 가능하게)
            (pd_dir / f"{k.replace('|', '__')}.json").write_text(
                json.dumps({d: preds[d].tolist() for d, _, _ in data}), encoding="utf-8")
        # 거리 신호 원자료도 함께 (임계 규칙만 바꿔 재분석할 수 있게)
        (pd_dir / f"signal__{cell['label'].replace('|', '__')}.json").write_text(
            json.dumps({d: [round(float(x), 6) for x in ds[d]] for d, _, _ in data}),
            encoding="utf-8")
    (out_dir(name) / f"results_{tag}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return results


def controls(name: str, data, primary_results: dict) -> dict:
    """통제군 5종."""
    res: dict[str, dict] = {}
    res["floor|no-boundary"] = bc.score_A(data, {d: np.zeros(len(g), np.int8)
                                                 for d, _, g in data})
    res["floor|all-boundary"] = bc.score_A(data, {d: np.ones(len(g), np.int8)
                                                  for d, _, g in data})
    # ④ EM-LLM surprisal — **dev-3 수치가 아직 존재하지 않으므로 여기서 생산한다.**
    #   refine=none  : surprise 를 raw 신호로 보고 3 OP 를 다 적용 (우리 신호와 like-for-like)
    #   refine=modul/conduct : EM-LLM 원본 파이프라인(γ=1.1 → 그래프 스냅)이 이미 경계 **집합**을
    #                          내므로 OP 재적용이 무의미하다. 라벨을 `|EMLLM` 로 구분해 둔다.
    import torch
    for did, _, _ in data:
        z = np.load(cache_dir(name) / f"{did}__emllm.npz")
        res.setdefault("_surp", {})[did] = (z["surprise"], z["A_sim"])
    surp = res.pop("_surp")
    for op in OPS:
        res[f"surprisal-none|{op}"] = bc.score_A(
            data, {d: sae.surprisal_to_boundaries(surp[d][0], op) for d, _, _ in data})
    for refine in ("modularity", "conductance"):
        preds = {}
        for did, _, g in data:
            s, A_sim = surp[did]
            base = _surp_events(s)                       # γ=1.1 full-past (원본 기본값)
            v = np.zeros(len(g), np.int8)
            if base:
                ref = refine_boundaries(base, torch.from_numpy(A_sim), metric=refine, min_size=1)
                for b in ref:
                    if 1 <= int(b) <= len(g):
                        v[int(b) - 1] = 1
            preds[did] = v
        res[f"surprisal-{refine}|EMLLM"] = bc.score_A(data, preds)
    # ⑤ M1-CosDrift
    m1 = preds_m1(data)
    for op in OPS:
        res[f"M1-CosDrift|{op}"] = bc.score_A(
            data, {d: sae.signal_to_boundaries(m1[d], op) for d, _, _ in data})
    # ② random @ matched density (셀별 예측 개수에 맞춰 20 seed)
    for k, v in primary_results.items():
        if v.get("degenerate_reason"):
            continue
        cnt = v["n_pred_by_doc"]
        ss = [bc.score_A(data, preds_random(data, cnt, s))["score"] for s in range(20)]
        res[f"random@{k}"] = {"score": float(np.mean(ss)), "sd": float(np.std(ss)),
                              "n_pred": sum(cnt.values())}
    (out_dir(name) / "results_controls.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return res


def _surp_events(s: np.ndarray) -> list[int]:
    from segment_manual_emllm import surprise_boundaries
    return surprise_boundaries(np.asarray(s, np.float64), sae.DEFAULT_GAMMA, win=0, warmup=3)


def primary_cells() -> list[dict]:
    cells = []
    for sig in sae.SIGNALS:
        for sel in SELECTIONS:
            cells.append({"signal": sig, "selection": sel, "label": f"{sig}|{sel}"})
    return cells


def shuffled_cells(n_seed: int = 5) -> list[dict]:
    return [{"signal": s, "selection": "shuffled", "seed": z,
             "label": f"{s}|shuffled-s{z}"} for s in sae.SIGNALS for z in range(n_seed)]


def main() -> int:
    ap = argparse.ArgumentParser(description="M9-AttnEntityDrift dev-3 러너")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--stage", choices=["cache", "precheck", "primary", "secondary", "ceiling"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--docs", default=",".join(DEV3))
    ap.add_argument("--scopes", default="sentnext")
    ap.add_argument("--norms", default="rowsent")
    args = ap.parse_args()

    data = bc.load_dataset("dev3")
    out_dir(args.name).mkdir(parents=True, exist_ok=True)

    if args.selftest:
        selftest(data)
        if not args.stage:
            return 0
    if args.stage == "cache":
        build_cache(args.name, data, args.lm, args.docs.split(","),
                    args.scopes.split(","), args.norms.split(","))
    elif args.stage == "precheck":
        precheck(args.name, data)
    elif args.stage == "primary":
        assert selftest_ok(), "self-test 를 먼저 통과할 것 (--selftest). PREREG §6 F3."
        t0 = time.time()
        pr = run_grid(args.name, data, primary_cells(), "sentnext", "rowsent", "primary")
        sh = run_grid(args.name, data, shuffled_cells(), "sentnext", "rowsent", "shuffled")
        ct = controls(args.name, data, pr)
        print(f"\n[primary] {len(pr)}셀 + shuffled {len(sh)} + 통제군 {len(ct)} "
              f"— {time.time()-t0:.0f}s")
        report_tables(args.name, pr, sh, ct)
    elif args.stage == "secondary":
        assert selftest_ok(), "self-test 를 먼저 통과할 것"
        run_grid(args.name, data, secondary_cells(), "sentnext", "rowsent", "secondary")
    elif args.stage == "ceiling":
        assert selftest_ok(), "self-test 를 먼저 통과할 것"
        ceiling(args.name, data)
    return 0


def ceiling(name: str, data) -> None:
    """🔶 **사후 추가 측정 (primary 결과를 본 뒤 실행 — 판정에 쓰지 않는다).**

    묻는 것: *"엔티티 추출을 attention 이 아니라 잘 하는 추출기로 대체했다면, 애초에
    '엔티티 집합 거리' 라는 틀의 천장이 얼마인가?"*
    → selection = `all-spacy` (spaCy 내용어/NER **전부**, top-k 없음, attention 무관).
    이 값이 무경계 하한 근처면 추출기를 고쳐도 소용없다는 뜻이다.
    """
    cells = [{"signal": s, "selection": "all-spacy", "label": f"{s}|all-spacy"}
             for s in sae.SIGNALS]
    r = run_grid(name, data, cells, "sentnext", "rowsent", "ceiling")
    pr = json.loads((out_dir(name) / "results_primary.json").read_text())
    ct = json.loads((out_dir(name) / "results_controls.json").read_text())
    rows = []
    for sig in sae.SIGNALS:
        for op in OPS:
            c = r[f"{sig}|all-spacy|{op}"]
            a = pr.get(f"{sig}|all-core|{op}", {})
            t = pr.get(f"{sig}|attn-topk|{op}", {})
            rows.append({
                "signal": sig, "op": op,
                "all-spacy(천장)": round(c["score"], 4), "n_pred": c["n_pred"],
                "all-core": round(a.get("score", float("nan")), 4),
                "attn-topk": round(t.get("score", float("nan")), 4),
                "vs floor": round(c["score"] - FLOOR, 4),
                "DEGEN": c["degenerate_reason"] or "",
            })
    print("\n=== 🔶 사후 천장 측정: spaCy 내용어/NER 전체 (attention 무관) ===")
    _print_rows(rows)
    best = max(r.values(), key=lambda v: v["score"])
    print(f"\n무경계 하한 {FLOOR:.4f} · G0 문턱 {G0_BAR:.4f} · M1-CosDrift(OP-1) "
          f"{ct['M1-CosDrift|OP-1']['score']:.4f} · M5(기존 표) 0.4190")
    print(f"천장 최고 = {best['score']:.4f} ({best['label']}|{best['op']}) "
          f"→ floor 대비 {best['score']-FLOOR:+.4f}")


def secondary_cells() -> list[dict]:
    """부록 격자 — 탐색적. 판정에 쓰지 않는다."""
    base = {"signal": "setdrift", "selection": "attn-topk"}
    out = []
    for lay in ("all", "0-8", "9-17"):
        out.append({**base, "layers": lay, "label": f"setdrift|attn|layers={lay}"})
    for sc in ("next", "sent"):
        out.append({**base, "scope": sc, "label": f"setdrift|attn|scope={sc}"})
    for k in (5,):
        out.append({**base, "topk": k, "label": f"setdrift|attn|k={k}"})
    out.append({**base, "topk_mode": "frac", "label": "setdrift|attn|k=frac0.25"})
    out.append({**base, "entity_unit": "token", "label": "setdrift|attn|unit=token"})
    out.append({**base, "lemma": True, "label": "setdrift|attn|lemma"})
    out.append({**base, "head_agg": "perhead-vote", "label": "setdrift|attn|perhead-vote"})
    for cap in (10, 50):
        out.append({**base, "signal": "recency", "cap": cap,
                    "label": f"recency|attn|cap={cap}"})
    for rl in (14, 28):
        out.append({**base, "signal": "embed", "repr_layer": rl,
                    "label": f"embed|attn|repr={rl}"})
    out.append({**base, "signal": "embed", "center": True, "label": "embed|attn|center"})
    out.append({**base, "signal": "embed", "l2": False, "label": "embed|attn|no-l2"})
    return out


NOISE = 0.06        # 🔒 PREREG §2 노이즈 바닥
G0_BAR = FLOOR + NOISE


def gates(name: str, pr: dict, sh: dict, ct: dict) -> list[dict]:
    """PREREG §5 게이트 G0~G3 자동 판정. 손으로 표를 읽고 판정하지 않는다."""
    rows = []
    for sig in sae.SIGNALS:
        for sel in SELECTIONS:
            for op in OPS:
                k = f"{sig}|{sel}|{op}"
                v = pr.get(k)
                if v is None:
                    continue
                sc = v["score"]
                deg = v["degenerate_reason"]
                row = {"signal": sig, "selection": sel, "op": op,
                       "Score": round(sc, 4), "n_pred": v["n_pred"], "DEGEN": deg or ""}
                row["G0"] = "PASS" if sc > G0_BAR else "FAIL"
                s_surp = ct.get(f"surprisal-none|{op}", {}).get("score")
                row["d_surp"] = round(sc - s_surp, 4) if s_surp is not None else None
                base = pr.get(f"{sig}|all-core|{op}")
                row["d_sel"] = round(sc - base["score"], 4) if base else None
                shs = [sh[x]["score"] for x in sh
                       if x.startswith(f"{sig}|shuffled-s") and x.endswith(f"|{op}")]
                row["d_rand"] = round(sc - float(np.mean(shs)), 4) if shs else None
                rows.append(row)
    # G1 은 OP-1·OP-2 양쪽에서 +0.06 초과 + 부호 일치를 요구한다 (셀 단위가 아니라 arm 단위)
    for r in rows:
        d1 = next((x["d_surp"] for x in rows if x["signal"] == r["signal"]
                   and x["selection"] == r["selection"] and x["op"] == "OP-1"), None)
        d2 = next((x["d_surp"] for x in rows if x["signal"] == r["signal"]
                   and x["selection"] == r["selection"] and x["op"] == "OP-2"), None)
        r["G1"] = ("PASS" if (d1 is not None and d2 is not None
                              and d1 > NOISE and d2 > NOISE) else
                   "노이즈" if (d1 is not None and d2 is not None and d1 * d2 < 0) else "FAIL")
        r["G2"] = ("기여없음" if r["d_sel"] is not None and abs(r["d_sel"]) < NOISE
                   else "PASS" if r["d_sel"] is not None and r["d_sel"] > 0 else "FAIL")
        r["G3"] = ("PASS" if r["d_rand"] is not None and r["d_rand"] > NOISE else "FAIL")
        if r["selection"] == "all-core":
            r["G2"] = r["G3"] = "—"        # 쌍둥이 자신에게는 G2/G3 가 정의되지 않는다
    (out_dir(name) / "gates.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def report_tables(name: str, pr: dict, sh: dict, ct: dict) -> None:
    """결과 먼저 — primary 표 + 게이트 판정 재료."""
    rows = []
    for k, v in sorted(pr.items(), key=lambda kv: -kv[1]["score"]):
        rows.append({"cell": k, "Score": round(v["score"], 4), "F1": round(v["f1"], 4),
                     "Pk": round(v["pk"], 4), "WD": round(v["wd"], 4), "n_pred": v["n_pred"],
                     "DEGEN": v["degenerate_reason"] or ""})
    print("\n=== PRIMARY 36셀 (Score 내림차순) ===")
    _print_rows(rows)
    print("\n=== 통제군 ===")
    _print_rows([{"key": k, "Score": round(v["score"], 4),
                  "n_pred": v.get("n_pred", "")} for k, v in ct.items()
                 if not k.startswith("random@")])
    gr = gates(name, pr, sh, ct)
    print(f"\n=== 게이트 판정 (G0 문턱 {G0_BAR:.4f} = floor {FLOOR:.4f} + 노이즈 {NOISE}) ===")
    _print_rows([{k: r[k] for k in ("signal", "selection", "op", "Score", "n_pred",
                                    "d_surp", "d_sel", "d_rand", "G0", "G1", "G2", "G3",
                                    "DEGEN")} for r in gr])


if __name__ == "__main__":
    raise SystemExit(main())
