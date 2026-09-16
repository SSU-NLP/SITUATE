#!/usr/bin/env python3
"""M5-FactoredSlot 으로 GutenQA 100편을 청킹 (D-B39 Phase 2).

## 파이프라인
문단 시퀀스(`books.jsonl`) → **문단마다** 앵커와 동일한 문장 분할
(`build_narrative_pack.split_sentences` + `build_pg_shorts.stage0_piece`) → 책 전체 문장열 →
`hi_ontop.m5_default.segment()` → 전이별 연속 점수 `s = Σ_k P(reset_k)` → **문단 경계에서만** 절단.

## 🔴 문단 경계 제약 (D-B39 §6) — 정확도가 아니라 **측정 타당성**
GutenQA 정답 판정은 `Chunk Must Contain` 부분문자열이 **한 청크 안에 통째로** 있느냐다.
문단 중간을 자르면 gold 문자열이 두 청크로 쪼개져 **어느 청크도 포함하지 않게 되고**, 방법론과
무관한 이유로 0점이 난다. 공개 베이스라인 5종은 전부 문단이 최소 단위다.

구현은 **스냅이 아니라 후보 제한**이다: 문단 경계에 해당하는 전이만 절단 후보로 두고 그중
점수 상위 r 비율을 자른다. 스냅(아무 데나 자른 뒤 가까운 문단으로 이동)은 충돌·중복 절단이
생기지만 후보 제한은 그런 게 없고, 합법 절단점들 사이의 **상대 순위 정보를 하나도 안 버린다.**

## 임계 (D-B39 §4) — 적응임계도, 동결 분위수도 쓰지 않는다
`segment()` 는 임계와 무관하게 항상 `scores` 를 돌려준다. 그것만 받아 **호출부에서** 분위수를 건다.
`configs/m5_default.json` 은 읽기만 하고 **변경하지 않는다.**

- `--rates R...` : 문단 경계 중 상위 R 비율을 절단 → 청크 크기 knob. **이게 주 sweep.**
  ⚠️ 여기서 R 의 분모는 **문단 경계 수**다. config 의 `r=0.0642` 는 분모가 **전체 문장 전이 수**라
  같은 수가 아니다. 헷갈리지 않도록 이름을 `rate` 로 따로 쓴다.
- `--frozen` : 참고 셀. 동결 규칙 그대로(전체 문장 전이 분위수 0.0642)로 경계를 낸 뒤 가장 가까운
  문단 경계로 스냅·중복 제거. "출하된 분절기를 그대로 쓰면 어떻게 되는가"에 대한 답.

## 산출
`data/chunks/m5__rate{R}.jsonl` (+ `m5__frozen.jsonl`) — 다른 방법과 동일 스키마라 `02_eval_harness.py`
에 그대로 물린다. 점수 캐시 `m5_scores/book{ID}.npz` (2회차부터 인코딩 생략).

    uv run python scripts/gutenqa/03_segment_m5.py --rates 0.05 0.10 0.15 0.20 0.30 0.50 --frozen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase0"))
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase1"))

BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
CACHE = BASE / "m5_scores"
# 🔴 체크포인트 세대별로 산출을 갈라 쓰기 위한 전역. main() 이 CLI 값으로 덮어쓴다.
# 기본값은 구 동작(W0 산출)과 100% 동일하므로 기존 호출은 영향받지 않는다.
OUT_CHUNKS = DATA / "chunks"
CKPT: str | None = None


def load_books():
    with (DATA / "books.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def paragraphs_to_sentences(paras: list[str]) -> tuple[list[str], list[int]]:
    """문단마다 앵커와 **동일한** 분할을 적용. 문단 j 가 시작하는 문장 인덱스도 함께 돌려준다.

    앵커(`build_pg_shorts.py`)는 `split_sentences(body)` 후 `stage0_piece` 를 걸었다.
    `stage0_piece` 는 문장 단위 연산이므로 문단별로 적용해도 결과가 같고, 이렇게 해야
    문단↔문장 정렬이 유지된다.
    """
    from build_narrative_pack import split_sentences
    from build_pg_shorts import stage0_piece

    sents: list[str] = []
    starts: list[int] = []
    for p in paras:
        starts.append(len(sents))
        s, _ops = stage0_piece(split_sentences(p))
        sents.extend(s)
    return sents, starts


def book_scores(bid: int, sents: list[str], device: str, force: bool = False) -> np.ndarray:
    """전이별 s = Σ_k P(reset_k). 인코딩이 비싸므로 캐시한다.

    🔴 캐시 적중 조건은 `n_sent` 일치뿐이라 **체크포인트를 바꿔도 구 점수를 조용히 재사용한다.**
    그래서 세대마다 `--scores-dir` 로 캐시를 분리한다(`--force` 만 쓰면 W0 캐시가 파괴된다).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"book{bid:03d}.npz"
    if p.exists() and not force:
        z = np.load(p)
        if int(z["n_sent"]) == len(sents):
            return z["scores"]
    from hi_ontop.m5_default import segment
    r = segment(sents, device=device, checkpoint=CKPT)
    np.savez_compressed(p, scores=np.asarray(r["scores"], np.float32), n_sent=len(sents))
    return np.asarray(r["scores"], np.float32)


def cuts_by_rate(scores: np.ndarray, starts: list[int], rate: float) -> list[int]:
    """문단 경계 후보 중 점수 상위 `rate` 비율을 절단점으로. 반환 = 절단이 일어나는 **문단 인덱스**."""
    # 문단 j(j>=1) 를 여는 전이 = scores[starts[j]-1]
    cand = [(j, starts[j] - 1) for j in range(1, len(starts))
            if 0 <= starts[j] - 1 < len(scores)]
    if not cand:
        return []
    n_cut = max(1, int(round(len(cand) * rate)))
    ranked = sorted(cand, key=lambda t: -float(scores[t[1]]))[:n_cut]
    return sorted(j for j, _ in ranked)


def snap_to_paragraph(raw: np.ndarray, starts: list[int]) -> list[int]:
    """문장 인덱스 경계열을 가장 가까운 문단 경계로 스냅·중복 제거 → 문단 인덱스 리스트."""
    opens = np.asarray(starts[1:], dtype=np.int64)   # 문단을 여는 문장 인덱스
    if opens.size == 0:
        return []
    idx = np.searchsorted(opens, raw)
    out = set()
    for r, i in zip(raw, idx):
        best, bd = None, None
        for cand_i in (i - 1, i):
            if 0 <= cand_i < opens.size:
                d = abs(int(opens[cand_i]) - int(r))
                if bd is None or d < bd:
                    best, bd = cand_i, d
        if best is not None:
            out.add(best + 1)                        # starts 인덱스로 환산 (문단 인덱스)
    return sorted(out)


def cuts_fixedq(scores: np.ndarray, starts: list[int], rate: float) -> list[int]:
    """⛔ **폐기된 규칙** — `fixed_quantile r=0.0642`.
    D-B39 시점의 `configs/m5_default.json` 값이었으나 2026-07-29 15:28 에 `online_nf c=1.0` 으로
    교체됐다. 폐기 사유(config 원문): *"provenance is void: r was the anchor boundary rate under
    count >= t*=3, and t*=3 was discarded"*. **임계 규칙 애블레이션용으로만 유지한다.**"""
    if len(scores) == 0:
        return []
    thr = float(np.quantile(scores, 1.0 - rate))
    return snap_to_paragraph(np.flatnonzero(scores >= thr) + 1, starts)


def cuts_adaptive(scores: np.ndarray, starts: list[int], c: float, mode: str,
                  warmup: int = 5) -> list[int]:
    """✅ **현행 채택 규칙** — 적응임계 `online_nf` μ + c·σ (config `adopted 2026-07-29`).
    `hi_ontop.hi_ontop_v2.adaptive_boundaries` 를 그대로 호출한다(재구현 금지). 규약상 0번
    sentinel 을 앞에 붙인 시퀀스를 받는다."""
    if len(scores) == 0:
        return []
    from hi_ontop.hi_ontop_v2 import adaptive_boundaries
    b = np.asarray(adaptive_boundaries(np.concatenate([[0.0], scores]).tolist(),
                                       c=c, mode=mode, warmup=warmup), np.int8)[:len(scores)]
    return snap_to_paragraph(np.flatnonzero(b) + 1, starts)


def emit(name: str, per_book: dict[int, list[str]]) -> Path:
    out = OUT_CHUNKS / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for bid in sorted(per_book):
            f.write(json.dumps({"book_id": bid, "chunks": per_book[bid]},
                               ensure_ascii=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="*",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
                    help="문단 경계 중 절단할 비율 (분모 = 문단 경계 수)")
    ap.add_argument("--adaptive", action="store_true",
                    help="✅ 현행 채택 규칙 `online_nf` μ+c·σ + 문단 스냅 셀 생성 (config adopted)")
    ap.add_argument("--fixedq", action="store_true",
                    help="⛔ 폐기된 `fixed_quantile r=0.0642` + 문단 스냅 셀 (임계 규칙 애블레이션용)")
    ap.add_argument("--ewma-spans", type=int, nargs="*", default=[],
                    help="EWMA 임계 셀 생성. span=warmup → alpha=2/(span+1). "
                         "누적(online_nf)과 달리 **국소** 임계이고 현재 값을 판정에 넣지 않는다.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="점수 캐시 무시하고 재계산")
    # ── 체크포인트 세대 분리 (2026-08-01, 하류 재검증) ──────────────────────
    ap.add_argument("--ckpt", default=None,
                    help="M5 체크포인트 경로. 생략하면 configs/m5_default.json 의 동결 경로. "
                         "학습 잡이 디폴트를 실행 중 교체하는 사고를 막으려면 명시하는 쪽이 안전하다.")
    ap.add_argument("--scores-dir", default=None,
                    help="점수 캐시 디렉터리(<out-base> 기준 상대). 세대마다 분리해야 "
                         "구 점수 재사용 사고가 없다. 기본 = m5_scores (W0)")
    ap.add_argument("--cell-prefix", default="m5__",
                    help="셀 이름 접두. 기본 m5__ = W0. 예: m5i2__ / m5w3__")
    ap.add_argument("--out-base", default=None,
                    help="청크·점수 산출 루트. 기본 = 2026-07-29_gutenqa_chunking")
    ap.add_argument("--books", default=None, help="books.jsonl 경로 (기본 = 기존 실험 data/)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="책을 N등분해 I번째만 처리. 점수 캐시 생성을 여러 GPU로 쪼갤 때 쓴다. "
                         "샤드는 청크를 **일부만** 갖게 되므로 반드시 --scores-only 와 함께 쓴다.")
    ap.add_argument("--scores-only", action="store_true",
                    help="점수 캐시만 만들고 청크를 쓰지 않는다. 샤드 병렬 → 마지막에 전체 1회 패스로 청크 생성.")
    a = ap.parse_args()

    shard_i, shard_n = (0, 1)
    if a.shard:
        shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    if a.shard and not a.scores_only:
        raise SystemExit("🔴 --shard 는 --scores-only 와 함께만 쓴다 (부분 청크가 전체를 덮어쓴다)")

    global CACHE, OUT_CHUNKS, CKPT, DATA
    out_base = Path(a.out_base) if a.out_base else BASE
    CACHE = out_base / (a.scores_dir or "m5_scores")
    OUT_CHUNKS = out_base / "chunks"
    OUT_CHUNKS.mkdir(parents=True, exist_ok=True)
    if a.books:
        DATA = Path(a.books).parent
    if a.ckpt:
        import hashlib
        CKPT = str(Path(a.ckpt).resolve())
        md5 = hashlib.md5(Path(CKPT).read_bytes()).hexdigest()
        print(f"[ckpt] {CKPT}\n[ckpt] md5={md5}")

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((REPO_ROOT / "configs/m5_default.json").read_text(encoding="utf-8"))
    th = cfg["threshold"]
    # 🔴 config 는 2026-07-29 15:28 에 fixed_quantile → online_nf 로 교체됐다. 두 규칙 다 지원한다.
    ad_mode = th.get("mode", "online_nf")
    ad_c = float(th.get("c", 1.0))
    FIXEDQ_RATE = 0.0642            # 폐기된 값. config 에서 사라졌으므로 애블레이션용으로 상수 고정.
    print(f"[m5] device={device} · config 현행 임계 = {ad_mode} c={ad_c} (읽기만, 변경 없음)")

    P = a.cell_prefix
    variants: dict[str, dict[int, list[str]]] = {f"{P}rate{r:g}": {} for r in a.rates}
    if a.adaptive:
        variants[f"{P}adaptive_{ad_mode}_c{ad_c:g}"] = {}
    if a.fixedq:
        variants[f"{P}fixedq_superseded"] = {}
    for sp in a.ewma_spans:
        variants[f"{P}ewma{sp}_c{ad_c:g}"] = {}
    stats: dict[str, list] = {k: [] for k in variants}

    for n, bk in enumerate(load_books()):
        if n % shard_n != shard_i:
            continue
        bid, paras = bk["book_id"], bk["paragraphs"]
        sents, starts = paragraphs_to_sentences(paras)
        if len(sents) < 2:
            continue
        sc = book_scores(bid, sents, device, force=a.force)
        if a.scores_only:
            if (n + 1) % 10 == 0:
                print(f"  [shard {shard_i}/{shard_n}] {n+1}/100 books", flush=True)
            continue

        for name in variants:
            kind = name[len(P):]          # 접두를 벗겨 규칙만 본다 (m5__ / m5i2__ / m5w3__)
            if kind.startswith("ewma"):
                sp = int(kind.split("ewma")[1].split("_")[0])
                cut_paras = cuts_adaptive(sc, starts, ad_c, "ewma", warmup=sp)
            elif kind.startswith("adaptive"):
                cut_paras = cuts_adaptive(sc, starts, ad_c, ad_mode)
            elif kind == "fixedq_superseded":
                cut_paras = cuts_fixedq(sc, starts, FIXEDQ_RATE)
            else:
                cut_paras = cuts_by_rate(sc, starts, float(name.split("rate")[1]))
            edges = [0] + cut_paras + [len(paras)]
            chunks = ["\n".join(paras[x:y]).strip() for x, y in zip(edges[:-1], edges[1:])]
            chunks = [c for c in chunks if c]
            variants[name][bid] = chunks
            stats[name].append((len(chunks), sum(len(c.split()) for c in chunks)))

        if (n + 1) % 10 == 0:
            print(f"  {n+1}/100 books · 마지막 책 문장 {len(sents):,} · 문단 {len(paras):,}",
                  flush=True)

    if a.scores_only:
        print(f"\n[shard {shard_i}/{shard_n}] 점수 캐시만 생성 완료 -> {CACHE}")
        return

    print()
    summary = {}
    for name, per_book in variants.items():
        p = emit(name, per_book)
        nc = sum(c for c, _ in stats[name])
        nw = sum(w for _, w in stats[name])
        summary[name] = {"chunks": nc, "chunks_per_book": round(nc / len(per_book), 1),
                         "mean_words_per_chunk": round(nw / max(1, nc), 1)}
        print(f"  {name:18s} {nc:>7,} 청크 ({summary[name]['chunks_per_book']:>7.1f}/book, "
              f"평균 {summary[name]['mean_words_per_chunk']:>6.1f} 단어) -> {p.name}")

    # 🔴 세대별로 갈라 쓴다 — 같은 경로면 2회차가 1회차 통계를 덮어쓴다(b1 리포트에서 실제 발생)
    stats_name = "m5_chunk_stats.json" if P == "m5__" else f"chunk_stats_{P.rstrip('_')}.json"
    (out_base / stats_name).write_text(
        json.dumps({"config_threshold": {"mode": ad_mode, "c": ad_c},
                    "superseded_fixedq_rate": FIXEDQ_RATE, "variants": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n참고: LumberChunker = 369.2 청크/book · 327.3 단어/청크 (우리 실측)")


if __name__ == "__main__":
    main()
