#!/usr/bin/env python3
"""개별 인간 주석자 기준선 — Michelmann 원자료의 피험자별 버튼 누름을 우리 하네스로 채점.

사전등록 = outputs/experiments/2026-07-31_human_annotator_baseline/PREREG.md

무엇을 하나:
  피험자 1명 = 하나의 "분절기" 로 보고, 그 사람의 누름을 gold A(저자 필터 합의) 로 채점한다.
  경계 도출 규칙·채점기 전부 동결본 재사용. 새로 만드는 것은 "피험자 행 → 누름 시각" 하나뿐.

  🔴 이 행은 **해석 기준선**이다. 성능 우위 주장에 쓰지 않는다 (PREREG §지위).

사용:
  .venv/bin/python scripts/eacl_phase0/human_annotator_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_michelmann_pack import GOLD_MS, load_mat_col, merge_5s  # noqa: E402
from build_pieman_pack import align_words_to_sentences, split_sentences  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

DATA = REPO / "data" / "eacl_phase0"
PACK = REPO / "outputs" / "eacl_phase0"
OUT = REPO / "outputs" / "experiments" / "2026-07-31_human_annotator_baseline"

# PREREG §대상 — 문서별 피험자 원자료. (rate_hz 는 §검증 2 에서 실측 확인한다.)
SUBJ_SRC = {
    "pieman": [("run1", "alldata_run1.mat", 1000.0), ("run2", "alldata_run2.mat", 1000.0)],
    "tunnel": [("run1", "alldata.mat", 10.0)],
    # monkey: 피험자 원자료 없음 (집계 MonkeyBounds.txt 뿐) — PREREG 에 불가로 기록.
}
PRIMARY_RUN = "run1"  # 🔒 사전 지정


def build_doc(story: str):
    """동결 빌더와 동일한 문장 분할·정렬·ms→전이 매핑을 재현해 반환."""
    sd = DATA / f"{story}_michelmann" / "sourcedata"
    ddir = DATA / f"{story}_michelmann"
    words = [str(w[0]).strip().lower() if hasattr(w, "__len__") and len(w) else str(w).strip().lower()
             for w in load_mat_col(sd / "words.mat", "words")]
    onsets = [float(x) for x in load_mat_col(sd / "word_onsets.mat", "onsets")]
    offsets = [float(x) for x in load_mat_col(sd / "word_offsets.mat", "offsets")]
    n = min(len(words), len(onsets), len(offsets))
    words, onsets = words[:n], onsets[:n]
    text = (ddir / f"{story.capitalize()}Clean.txt").read_text(encoding="utf-8-sig")
    sentences = split_sentences(text)
    word_sent = align_words_to_sentences(words, sentences)

    def time_to_word(t_s: float) -> int:
        lo = 0
        for i, on in enumerate(onsets):
            if on == on and on <= t_s:
                lo = i
        return lo

    def ms_to_transitions(ms_list) -> list[int]:
        """ms 목록 -> 경계 전이 인덱스 집합 (빌더와 동일: 충돌 붕괴 + 마지막 문장 drop)."""
        after = set()
        for ms in ms_list:
            after.add(word_sent[time_to_word(ms / 1000.0)])
        after.discard(len(sentences) - 1)
        return sorted(s + 1 for s in after)  # 전이 인덱스 = 문장 s 다음

    return sentences, onsets, ms_to_transitions


def to_vec(trans: list[int], n_tr: int) -> list[int]:
    v = [0] * n_tr
    for t in trans:
        if 1 <= t <= n_tr:
            v[t - 1] = 1
    return v


def presses_ms(row: np.ndarray, rate_hz: float) -> list[float]:
    """0->1 상승 에지만 누름 1회로 센다 (버튼 유지 중복 제거)."""
    b = (row > 0).astype(np.int8)
    edges = np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1
    if b[0] == 1:
        edges = np.concatenate([[0], edges])
    return (edges / rate_hz * 1000.0).tolist()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"prereg": "PREREG.md", "gates": {}, "stories": {}}

    for story, srcs in SUBJ_SRC.items():
        sentences, onsets, ms_to_transitions = build_doc(story)
        n_tr = len(sentences) - 1

        # 🔒 관문 1 — 매핑 회귀 확인
        meta = json.loads((PACK / f"{story}_michelmann_pack_meta.json").read_text())
        repro = ms_to_transitions(GOLD_MS[story])
        ok = repro == meta["boundary_transitions"]
        report["gates"][f"{story}_mapping"] = {
            "pass": ok, "frozen": meta["boundary_transitions"], "reproduced": repro}
        print(f"[gate1] {story}: mapping repro = {'PASS' if ok else '🔴 FAIL'} "
              f"({len(repro)} vs {len(meta['boundary_transitions'])})")
        if not ok:
            print("  🔴 관문 1 실패 — PREREG 에 따라 이 문서의 개인 수치는 해석하지 않는다.")

        gold_vec = to_vec(meta["boundary_transitions"], n_tr)
        n_gold = sum(gold_vec)
        st = {"n_transitions": n_tr, "n_gold": n_gold, "gate_mapping_pass": ok, "runs": {}}

        for run, fname, rate in srcs:
            m = sio.loadmat(str(DATA / f"{story}_michelmann" / "sourcedata" / fname))
            arr = m[[k for k in m if not k.startswith("__")][0]]
            n_subj, n_samp = arr.shape

            # 🔒 관문 2 — 샘플레이트 확인
            dur_story = float(np.nanmax(onsets))
            dur_rec = n_samp / rate
            last_press = max((presses_ms(arr[i], rate)[-1] / 1000.0
                              for i in range(n_subj) if arr[i].any()), default=0.0)
            rate_ok = 0.7 * dur_story <= last_press <= 1.3 * dur_story
            report["gates"][f"{story}_{run}_rate"] = {
                "pass": rate_ok, "assumed_hz": rate, "story_dur_s": round(dur_story, 1),
                "record_dur_s": round(dur_rec, 1), "last_press_s": round(last_press, 1)}
            print(f"[gate2] {story}/{run}: rate={rate}Hz  story={dur_story:.0f}s "
                  f"rec={dur_rec:.0f}s last_press={last_press:.0f}s -> "
                  f"{'PASS' if rate_ok else '🔴 FAIL'}")

            rows, n_zero = [], 0
            for i in range(n_subj):
                p = presses_ms(arr[i], rate)
                if not p:
                    n_zero += 1
                    continue
                tr = ms_to_transitions(merge_5s(p))
                if not tr:
                    n_zero += 1
                    continue
                s = score_dialogues([gold_vec], [to_vec(tr, n_tr)])
                rows.append({"subject": i, "n_press_raw": len(p), "n_pred": len(tr),
                             "bor": len(tr) / n_gold, "score": s["score"],
                             "f1": s["f1"], "pk": s["pk"], "wd": s["wd"]})

            sc = np.array([r["score"] for r in rows])
            bor = np.array([r["bor"] for r in rows])
            st["runs"][run] = {
                "file": fname, "rate_hz": rate, "n_subject_total": int(n_subj),
                "n_excluded_zero_press": int(n_zero), "n_scored": len(rows),
                "score_mean": float(sc.mean()), "score_sd": float(sc.std(ddof=1)),
                "score_pcts": {k: float(np.percentile(sc, q)) for k, q in
                               [("min", 0), ("p25", 25), ("median", 50), ("p75", 75), ("max", 100)]},
                "bor_mean": float(bor.mean()), "bor_sd": float(bor.std(ddof=1)),
                "bor_pcts": {k: float(np.percentile(bor, q)) for k, q in
                             [("min", 0), ("p25", 25), ("median", 50), ("p75", 75), ("max", 100)]},
                "f1_mean": float(np.mean([r["f1"] for r in rows])),
                "pk_mean": float(np.mean([r["pk"] for r in rows])),
                "wd_mean": float(np.mean([r["wd"] for r in rows])),
                "subjects": rows,
            }
            tag = " (주값)" if run == PRIMARY_RUN else " (판독)"
            print(f"  {story}/{run}{tag}: n={len(rows)} (0누름 제외 {n_zero}) "
                  f"Score {sc.mean():.4f} ± {sc.std(ddof=1):.4f}  "
                  f"[{sc.min():.3f} .. {sc.max():.3f}]  BOR {bor.mean():.2f}")

        report["stories"][story] = st

    report["monkey_note"] = ("피험자 원자료 없음 — MonkeyBounds.txt(집계)만 존재. "
                            "개인 행 산출 불가. PREREG §대상 에 사전 기록됨.")
    (OUT / "human_baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT / 'human_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
