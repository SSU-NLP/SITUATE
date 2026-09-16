#!/usr/bin/env python3
r"""LoCoMo 10대화를 **M5 로만** 분절해 보는 사전점검 (LLM 호출 0회 · 비용 0).

왜 먼저 도나: M5 는 서사 dev-3 에서 약 2.2배 과분절한다. 대화에서도 그러면 에피소드 수가 늘어
하류 LLM 비용·지연이 그만큼 곱해진다. **본 실행 전에** 에피소드 개수를 알아야 예산을 잡을 수 있다.

🔴 여기서 나오는 건 **개수·길이 분포뿐**이다. 점수를 보고 임계 c 를 고르는 일은 하지 않는다
(c=1.0 은 DTS 공식 test 가 고른 값 · D-B39 · LoCoMo 에서 재탐색 금지).

    .venv/bin/python scripts/nemori_locomo/segment_dryrun.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemori_env import REPO, NEMORI, add_nemori_to_path  # noqa: E402

add_nemori_to_path()

DATA = REPO / "benchmarks/locomo/locomo10.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--unit", default="message", choices=["message", "sentence"])
    ap.add_argument("--c", type=float, default=None,
                    help="적응임계 c. 미지정이면 동결값(1.0). 여러 값은 --c-sweep")
    ap.add_argument("--c-sweep", default="",
                    help="쉼표구분 c 목록. 예: 0.5,1.0,1.5,2.0 (그룹 수만 비교, 점수 없음)")
    ap.add_argument("--out", default=str(REPO / "outputs/experiments/2026-08-01_nemori_locomo_m5/segment_dryrun.json"))
    args = ap.parse_args()

    # add.py 와 **동일한** 메시지 구성 로직을 재사용한다(따로 만들지 않는다).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "locomo_add", NEMORI / "evaluation/locomo/add.py")
    add_mod = importlib.util.module_from_spec(spec)
    sys.modules["locomo_add"] = add_mod
    spec.loader.exec_module(add_mod)

    from hi_ontop.nemori_segmenter import M5SegmenterConfig, _boundaries

    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    # 대화별 텍스트는 한 번만 만든다 (c 를 바꿔도 입력은 같다)
    convs = []
    for item in dataset:
        msgs = add_mod.build_messages(item.get("conversation", {}))
        convs.append([" ".join(str(m["content"]).split()) or "." for m in msgs])

    c_list = ([float(x) for x in args.c_sweep.split(",") if x.strip()]
              if args.c_sweep else [args.c])

    cells = {}
    for c in c_list:
        cfg = M5SegmenterConfig(device=args.device, unit=args.unit, c=c)
        label = f"c={cfg.resolved()['c']}"
        rows = []
        t0 = time.time()
        for idx, texts in enumerate(convs):
            bnd, stats = _boundaries(texts, cfg)
            sizes = [b - a for a, b in zip([0] + bnd, bnd + [len(texts)])]
            rows.append({
                "conv": idx, "n_messages": len(texts), "n_groups": len(sizes),
                "boundary_rate": round(len(bnd) / max(1, len(texts) - 1), 4),
                "median_group": statistics.median(sizes),
                "mean_group": round(statistics.mean(sizes), 2),
                "max_group": max(sizes), "min_group": min(sizes),
                "seg_latency_ms": stats["seg_latency_ms"],
            })
        total_msgs = sum(r["n_messages"] for r in rows)
        total_groups = sum(r["n_groups"] for r in rows)
        cells[label] = {
            "config": cfg.resolved(),
            "total_messages": total_msgs,
            "total_groups": total_groups,
            "overall_boundary_rate": round((total_groups - len(rows)) / (total_msgs - len(rows)), 4),
            "mean_group_size": round(total_msgs / total_groups, 2),
            "median_group_size": statistics.median(
                [s for r in rows for s in [r["median_group"]]]),
            "wall_s": round(time.time() - t0, 1),
            "per_conversation": rows,
        }
        print(f"{label}: {total_groups} groups · rate "
              f"{cells[label]['overall_boundary_rate']:.4f} · mean size "
              f"{cells[label]['mean_group_size']}", flush=True)

    summary = cells[list(cells)[0]] if len(cells) == 1 else {"sweep": cells}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "per_conversation"}
         for k, v in cells.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
