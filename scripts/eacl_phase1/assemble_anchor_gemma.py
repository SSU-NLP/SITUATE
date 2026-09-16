#!/usr/bin/env python3
r"""Phase-1 앵커 회수분 무결성 검사 + 조립용 master jsonl 생성 (D-B53, Gemma-4-26B-A4B).

## 왜 이 단계가 필요한가
`llm_s5_states.py` 는 축 파일이 **부족하면 그 자리에서 모델을 호출**한다(:722 while 루프).
회수분에 결손 문서가 섞인 채로 전량 full 패스를 돌리면 서버가 없는 상태에서 죽거나,
서버가 있으면 **일부 문서만 다른 시점·다른 조건으로 라벨**된다. 둘 다 코퍼스 오염이다.

⇒ 조립 전에 **완비 문서만 골라낸다.** 완비 = story_lists.json + 5축 파일이 모두 있고
각 축의 `item_id` 개수가 그 문서의 문장 수와 정확히 일치.

## 산출
- `_assembly_master.jsonl` — 완비 문서만 담은 전이 레코드(조립 full 패스 입력)
- `_assembly_audit.json` — 결손 사유별 문서 목록(손실을 숨기지 않는다)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

AXES = ("protagonist", "space", "time", "causality", "intentionality")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="축 파일이 풀린 디렉터리")
    ap.add_argument("--shard-dir", default="/shared/hiontop/data/shards_all")
    a = ap.parse_args()
    out = Path(a.out_dir)

    # 원본 전이 레코드 (doc_id -> rows). 문장 수 = 전이 수 + 1.
    by_doc: dict[str, list[dict]] = {}
    for sh in sorted(Path(a.shard_dir).glob("*.jsonl")):
        for line in sh.open():
            r = json.loads(line)
            by_doc.setdefault(r["doc_id"], []).append(r)
    print(f"원본 문서 {len(by_doc)}편 · 전이 {sum(len(v) for v in by_doc.values()):,}", flush=True)

    ok, bad = [], Counter()
    detail: dict[str, list[str]] = {}
    for doc, rows in by_doc.items():
        d = out / doc
        n_sent = len(rows) + 1                       # 문장 수 = 전이 + 1
        if not (d / "story_lists.json").exists():
            bad["stage1_없음"] += 1; detail.setdefault("stage1_없음", []).append(doc); continue
        miss = [ax for ax in AXES if not (d / f"{ax}_states.jsonl").exists()]
        if miss:
            bad["축파일_없음"] += 1
            detail.setdefault("축파일_없음", []).append(f"{doc}:{','.join(miss)}"); continue
        short = []
        for ax in AXES:
            n = sum(1 for _ in (d / f"{ax}_states.jsonl").open())
            if n != n_sent:
                short.append(f"{ax}({n}/{n_sent})")
        if short:
            bad["축_불완전"] += 1
            detail.setdefault("축_불완전", []).append(f"{doc}:{','.join(short)}"); continue
        ok.append(doc)

    n_tr = sum(len(by_doc[d]) for d in ok)
    print(f"완비 {len(ok)}편 · 전이 {n_tr:,}  |  결손 {dict(bad)}", flush=True)

    mp = out / "_assembly_master.jsonl"
    with mp.open("w") as f:
        for doc in ok:
            for r in sorted(by_doc[doc], key=lambda x: int(x["transition"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "_assembly_audit.json").write_text(json.dumps(
        {"n_source": len(by_doc), "n_ok": len(ok), "n_transitions_ok": n_tr,
         "loss_counts": dict(bad), "loss_detail": detail}, indent=2, ensure_ascii=False))
    print(f"→ {mp}", flush=True)


if __name__ == "__main__":
    main()
