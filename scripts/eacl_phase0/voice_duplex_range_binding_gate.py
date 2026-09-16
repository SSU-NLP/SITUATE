#!/usr/bin/env python3
"""게이트 §6-A A — 범위 제약이 M5 의 `speak` 발화율을 실제로 묶는가.

## 왜 이것이 게이트인가
트랙의 원 논거였던 "160ms 격자가 입도를 외생적으로 고정한다"는 **틀렸다**(사전등록 §0 개정 1).
격자가 고정하는 것은 결정 주기이지 `speak` 발화율이 아니다. 범위 제약 디코딩은 그 논거의 **복구**이며,
제약이 실제로 발화율을 묶지 못하면 입도 교란이 그대로 돌아온다.

## 측정
조건 ② (적응 임계 단독) 의 `speak` 중 조건 ②′ (+ 범위 제약) 에서 **최소 간격 800 ms 로 억제되는 비율**.

| 결속률 | 판정 |
|---|---|
| **10% 미만** | 제약 무작동 ⇒ ②′ ≈ ② ⇒ **봉인 개봉 전 종료** |
| 유의하게 억제 | 논거 복구 확인 ⇒ 진행 |

점수가 아니라 발화 억제 횟수이므로 금지 목록에 걸리지 않는다.

## 한계 (기록)
개발 세트 동봉 **gold 단어 타임스탬프**로 IPU 를 만든다. 본실험은 SoulX 자체 스트리밍 ASR 의
인과 전사를 쓴다(사전등록 §5.2). 텍스트는 다르지만 IPU 타이밍은 같은 오디오에서 오므로
**결정 밀도**의 자릿수는 보존된다고 보고 게이트에 쓴다. 통과 시 인과 전사로 재확인한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.m5_default import segment  # noqa: E402  🔒 동결 진입점

DEV = Path("/root/voiceduplex/fdb_v1_data/v1.0/ex/synthetic_pause_handling")
OUT = REPO / "outputs/experiments/2026-07-31_voice_duplex"

MIN_GAP_MS = 800.0        # 🔒 사전등록 §3 — 벤치마크 config max_wait_num 5 × 160 ms
MAX_GAP_MS = 32000.0      # 🔒 사전등록 §3 — history_len 200 × 160 ms
WARMUP_MS = 1160.0        # 🔒 사전등록 §3 — get_chunk() back+chunk+ahead
IPU_PAUSE_MS = 200.0      # 허용 목록 ① — 개발 세트에서 정해도 되는 값


def build_ipus(words: list[dict], pause_ms: float) -> list[tuple[str, float, float]]:
    """단어 열 -> IPU. 단어 간 휴지가 `pause_ms` 이상이면 IPU 를 끊는다.

    Returns: [(text, start_ms, end_ms), ...]
    """
    ipus, cur, t0 = [], [], None
    prev_end = None
    for w in words:
        a, b = w["timestamp"]
        if a is None or b is None:
            continue
        a, b = a * 1000.0, b * 1000.0
        if prev_end is not None and a - prev_end >= pause_ms and cur:
            ipus.append((" ".join(cur), t0, prev_end))
            cur, t0 = [], None
        if t0 is None:
            t0 = a
        cur.append(w["text"].strip())
        prev_end = b
    if cur:
        ipus.append((" ".join(cur), t0, prev_end))
    return [x for x in ipus if x[0]]


def apply_range(times: list[float], min_gap: float, max_gap: float,
                warmup: float) -> tuple[list[float], list[float]]:
    """범위 제약 (a) 기준 — 연속한 두 `speak` 사이 간격. 사전등록 §0 개정 5 용어 확정.

    Returns: (채택된 시각, 억제된 시각)
    """
    kept, dropped, last = [], [], None
    for t in times:
        if t < warmup:                      # 첫 판정 웜업 전에는 결정하지 않는다
            dropped.append(t); continue
        if last is not None and t - last < min_gap:
            dropped.append(t); continue
        kept.append(t); last = t
    # 최대 간격 초과 시 강제 발화는 결속률 분자에 넣지 않는다(억제가 아니라 추가이므로).
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipu-pause-ms", type=float, default=IPU_PAUSE_MS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dirs = sorted([d for d in DEV.iterdir() if d.is_dir()],
                  key=lambda p: int(p.name) if p.name.isdigit() else 0)
    if args.limit:
        dirs = dirs[:args.limit]

    n_c2 = n_kept = n_drop_gap = n_drop_warm = 0
    n_ipus = n_docs = n_skipped = 0
    forced = 0
    per_doc = []
    for d in dirs:
        tf = d / "transcription.json"
        if not tf.exists():
            n_skipped += 1; continue
        words = json.loads(tf.read_text())
        ipus = build_ipus(words, args.ipu_pause_ms)
        if len(ipus) < 2:
            n_skipped += 1; continue
        r = segment([x[0] for x in ipus])
        # 경계 i = IPU i 가 새 구간을 연다 -> 결정 시각 = 직전 IPU 의 끝
        times = [ipus[i - 1][2] for i in r["boundaries"]]
        kept, dropped = apply_range(times, MIN_GAP_MS, MAX_GAP_MS, WARMUP_MS)
        n_warm = sum(1 for t in dropped if t < WARMUP_MS)
        n_c2 += len(times); n_kept += len(kept)
        n_drop_gap += len(dropped) - n_warm; n_drop_warm += n_warm
        n_ipus += len(ipus); n_docs += 1
        # 최대 간격 강제 발화 (참고)
        dur = ipus[-1][2]
        forced += int(dur // MAX_GAP_MS) if not kept else 0
        per_doc.append({"doc": d.name, "n_ipu": len(ipus), "n_speak_c2": len(times),
                        "n_speak_c2p": len(kept), "n_suppressed": len(dropped)})

    bind = (n_c2 - n_kept) / n_c2 if n_c2 else 0.0
    bind_gap = n_drop_gap / n_c2 if n_c2 else 0.0
    verdict = "**진행**" if bind >= 0.10 else "🔴 **종료** (제약 무작동)"

    print(f"문서 {n_docs}편 (제외 {n_skipped}) · IPU {n_ipus}개 "
          f"(문서당 {n_ipus / max(n_docs,1):.1f})")
    print(f"② speak {n_c2} -> ②′ speak {n_kept}")
    print(f"  최소 간격 800ms 억제 {n_drop_gap} · 웜업 1160ms 억제 {n_drop_warm}")
    print(f"결속률 {bind:.1%} (최소 간격만 {bind_gap:.1%})  ->  {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"# 게이트 §6-A A — 범위 제약 결속률 (IPU 휴지 {args.ipu_pause_ms:.0f} ms)\n",
         f"**결속률 {bind:.1%} → {verdict}**  (기준: 10% 미만이면 종료)\n",
         "| 항목 | 값 |", "|---|---|",
         f"| 문서 | {n_docs}편 (제외 {n_skipped}) |",
         f"| IPU | {n_ipus}개 · 문서당 {n_ipus / max(n_docs,1):.1f} |",
         f"| ② `speak` | {n_c2} |",
         f"| ②′ `speak` | {n_kept} |",
         f"| 최소 간격 800 ms 억제 | {n_drop_gap} ({bind_gap:.1%}) |",
         f"| 웜업 1,160 ms 억제 | {n_drop_warm} |",
         f"| **결속률 (전체 억제/②)** | **{bind:.1%}** |",
         "\n## 고정값 (사전등록 §3, 금지 목록)\n",
         f"- 최소 간격 {MIN_GAP_MS:.0f} ms — 벤치마크 config `max_wait_num 5` × 160 ms",
         f"- 최대 간격 {MAX_GAP_MS:.0f} ms — `history_len 200` × 160 ms",
         f"- 웜업 {WARMUP_MS:.0f} ms — `get_chunk()` back+chunk+ahead",
         f"- IPU 휴지 임계 {args.ipu_pause_ms:.0f} ms — **허용 목록 ①**(개발 세트에서 정해도 되는 값)",
         "\n## 한계\n",
         "- 개발 세트 동봉 **gold 단어 타임스탬프**로 IPU 를 만들었다. 본실험은 SoulX 자체",
         "  스트리밍 ASR 의 인과 전사를 쓴다(사전등록 §5.2). 통과 시 인과 전사로 재확인한다.",
         "- 이것은 점수가 아니라 **발화 억제 횟수**다. 성능 주장에 쓰지 않는다."]
    (OUT / "GATE_range_binding.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"binding_rate": bind, "binding_rate_gap_only": bind_gap,
               "n_speak_c2": n_c2, "n_speak_c2p": n_kept, "n_docs": n_docs,
               "n_ipus": n_ipus, "ipu_pause_ms": args.ipu_pause_ms, "per_doc": per_doc},
              (OUT / "gate_range_binding.json").open("w"), ensure_ascii=False, indent=2)
    print("->", OUT / "GATE_range_binding.md")


if __name__ == "__main__":
    main()
