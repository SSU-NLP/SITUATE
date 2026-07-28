#!/usr/bin/env python3
"""SuperDialseg 공식 test 전체 → master.jsonl (D-B24 부분집합 문제 해소).

## 왜
`outputs/eacl_phase0/dts_*_master.jsonl` 은 S5 추출용 **샘플**이었다 —
tiage 20 대화 · dialseg711 12 · superseg 15, 게다가 **대화당 15 전이로 절단**.
공식 test 는 **tiage 100 / dialseg711 711 / superseg 1322**(`2026-06-13_dts_official_rescore`).
그 차이 때문에 M1-CosDrift 가 공식 0.4675 vs 본 파이프라인 0.3431 로 어긋났다(D-B24).

## 데이터
`benchmarks/superdialseg_data/{set}/segmentation_file_test.json`
(SuperDialseg 공식 배포본, Google Drive. README 의 Download Dataset 링크.)
형식: `[{dial_id, turns:[{utterance, segmentation_label, ...}]}]`

## 경계 규약 (프로젝트 공식과 일치)
`segmentation_label` 은 **세그먼트의 마지막 턴에 1**(끝-turn 규약, `dts_scoring` 과 동일).
전이 t = (turn[t-1] → turn[t]) 이며 `boundary = segmentation_label[t-1]`.
마지막 턴의 라벨은 문서 끝이므로 전이가 없다.

산출: `outputs/eacl_phase0/dts_full_{set}_master.jsonl` — 기존 샘플 파일을 **덮어쓰지 않는다**.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "benchmarks/superdialseg_data"
OUT = REPO / "outputs/eacl_phase0"
SETS = ("tiage", "dialseg711", "superseg")


def load(name: str) -> list[dict]:
    raw = json.loads((SRC / name / "segmentation_file_test.json").read_text())
    dials = raw if isinstance(raw, list) else (raw.get("dial_data") or raw)
    if isinstance(dials, dict):
        dials = [x for v in dials.values() for x in v]
    return dials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    args = ap.parse_args()
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        dials = load(name)
        rows, nb, nt = [], 0, 0
        for d in dials:
            turns = d.get("turns") or d.get("utterances") or []
            utts = [str(t.get("utterance", "")).strip() for t in turns]
            labs = [int(t.get("segmentation_label", 0)) for t in turns]
            if len(utts) < 2:
                continue
            did = str(d.get("dial_id", f"{name}_{len(rows)}"))
            for i in range(1, len(utts)):
                rows.append({"doc_id": did, "transition": i,
                             "previous": utts[i - 1], "current": utts[i],
                             "previous_ko_reference": "", "current_ko_reference": "",
                             # 끝-turn 규약: 세그먼트 마지막 턴에 1 → 그 턴에서 다음으로 넘어가는 전이가 경계
                             "boundary": int(labs[i - 1]), "idm_score": None,
                             "protagonist": None, "place": None, "time": None,
                             "causality": None, "goal": None, "notes": ""})
                nb += int(labs[i - 1])
            nt += len(utts)
        p = OUT / f"dts_full_{name}_master.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
        ndoc = len({r["doc_id"] for r in rows})
        print(f"{name:12s} 대화 {ndoc:5d} · 턴 {nt:6d} · 전이 {len(rows):6d} · "
              f"경계 {nb:5d} (밀도 {nb/max(len(rows),1):.4f})  -> {p.name}", flush=True)


if __name__ == "__main__":
    main()
