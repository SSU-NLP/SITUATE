#!/usr/bin/env python3
"""온라인 변형이 **미래를 보지 않는지** 코드로 검증한다.

## 왜 필요한가
"온라인"이라는 주장은 구현 실수 하나로 조용히 깨진다 — 슬라이딩 창이 뒤를 한 칸 더 보거나,
정규화가 전체 문서 통계를 쓰거나, 캐시가 이전 전체 실행 결과를 재사용하면 끝이다.
그런 코드도 **점수는 멀쩡히 나온다.** 그래서 점수가 아니라 **불변성**을 직접 검사한다.

## 검사 규약 — prefix-불변성
prefix-causal 이 참이면 다음이 성립해야 한다:

    forward(u_1..u_t)[t-2]  ==  forward(u_1..u_T)[t-2]   가 **성립하지 않아도 된다**
                                                          (오프라인은 미래를 보므로 달라야 정상)
    forward(u_1..u_t)[t-2]  ==  forward(u_1..u_{t+k})[t-2]  ← 🔴 이것도 성립할 필요 없다

핵심은 다른 것이다. **온라인 예측이 미래에 의존하지 않는다** 는 것은
*"턴 t 의 결정을 내릴 때 입력에 u_{>t} 가 들어가지 않았다"* 를 뜻한다. 이는 값 비교가 아니라
**입력 감사**로만 확정된다. 그래서 두 가지를 본다:

1. **입력 감사** — `forward` 에 넘어간 `utterances` 길이가 정확히 t 인가 (몰래 전체를 넘기지 않는가).
2. **미래 교란 불변성** — 문서 **뒤쪽을 바꿔치기**해도 턴 t 의 온라인 판정이 **변하지 않는가**.
   변하면 어딘가에서 미래를 읽고 있다는 뜻이다. 이게 실질적인 판정 기준이다.

## 대조군
같은 검사를 **오프라인** 경로에 걸면 **반드시 실패해야** 한다(미래를 보니까).
실패하지 않으면 검사 자체가 무력하다는 뜻이므로 그것도 함께 확인한다.

    PYTHONPATH=benchmarks/superdialseg/src .venv-superdialseg/bin/python \\
      EACL/methods/common/verify_causality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHODS))

from common.superdialseg_harness import (segmentation_file,  # noqa: E402
                                         shim_broken_relative_import)


def _load_many(dataset: str, k: int, min_turns: int = 20) -> list[list[str]]:
    """검사용 대화 k 개 — 서로의 뒤쪽을 갈아끼워 교란한다.

    같은 대화를 뒤집거나 대문자로 바꾸는 정도의 교란은 어휘 분포가 그대로라
    어휘 응집도 기반 방법(TextTiling)의 오프라인 판정을 **안 움직인다**. 그러면 대조군이
    죽어 검사 자체가 무력해진다. **다른 대화의 발화**로 갈아끼워야 분포가 실제로 바뀐다.
    """
    shim_broken_relative_import()
    from super_dialseg.utils.data.load_dialseg import DialsegDataset

    ds = DialsegDataset(segmentation_file=str(segmentation_file(dataset)))
    found = []
    for i in range(len(ds)):
        s = ds[i]
        if len(s.utterances) >= min_turns:
            found.append(list(s.utterances))
        if len(found) == k:
            return found
    if len(found) < 2:
        raise RuntimeError(f"{dataset}: {min_turns}턴 이상 대화가 2개 미만이다")
    return found


def _safe_flags(segmenter, utts: list[str]):
    """짧은 prefix 에서 분절기가 던지는 예외를 하네스와 **같은 규약**으로 흡수한다."""
    try:
        return segmenter.forward({"utterances": utts})
    except Exception:  # noqa: BLE001
        return None


def online_decision(segmenter, utts: list[str], t: int) -> int:
    """prefix-causal 규약: u_1..u_t 만 넣고 최전선 전이(t−1→t) 판정을 읽는다."""
    flags = _safe_flags(segmenter, utts[:t])
    if flags is None or len(flags) < t - 1:
        return -1                      # miss — 하네스에서는 경계 없음(0) 으로 떨어진다
    return int(flags[t - 2])


def offline_decision(segmenter, utts: list[str], t: int) -> int:
    """대조군: 전체 문서를 넣고 같은 전이 판정을 읽는다 (미래 사용)."""
    flags = _safe_flags(segmenter, utts)
    if flags is None or len(flags) < t - 1:
        return -1
    return int(flags[t - 2])


def check(segmenter_factory, name: str, dataset: str = "dialseg711",
          n_docs: int = 12) -> bool:
    """여러 문서에 교란 검사를 걸어 집계한다.

    🔴 **한 문서로는 판정할 수 없다.** 실측해 보면 어떤 문서에서는 뒤쪽을 통째로 갈아끼워도
    오프라인 판정이 하나도 안 바뀐다(그 문서에서는 미래가 원래 영향을 안 준다는 뜻).
    그런 문서에서는 **미래를 읽는 잘못된 코드도 검사를 통과**하므로 증거가 되지 못한다.
    따라서 문서를 여러 개 돌려 ① 온라인은 **전 문서에서** 불변 ② 대조군은 **일부 문서에서라도**
    움직임 — 둘 다 충족해야 통과로 본다.
    """
    docs = _load_many(dataset, n_docs + 1)
    seg = segmenter_factory()
    print(f"\n### {name} · {dataset} · 문서 {len(docs) - 1}개")

    audit_ok = True
    n_online_moved = n_control_moved = 0
    per_doc = []

    for di in range(len(docs) - 1):
        utts, other = docs[di], docs[di + 1]
        n = len(utts)
        cut = max(3, n // 2)
        # 교란 = 뒤 절반을 **다른 대화의 발화**로 갈아끼움 (어휘 분포가 실제로 바뀐다)
        tail = (other * ((n - cut) // max(1, len(other)) + 1))[: n - cut]
        scrambled = utts[:cut] + tail

        # 1) 입력 감사 — forward 가 정확히 t 개만 받는가
        seen, orig = {}, seg.forward

        def spy(inputs, _o=orig, _s=seen):
            _s["len"] = len(inputs["utterances"])
            return _o(inputs)

        seg.forward = spy
        online_decision(seg, utts, cut)
        seg.forward = orig
        audit_ok &= seen.get("len") == cut

        # 2) 온라인 불변성  3) 대조군(오프라인) 변화
        rng = range(2, cut + 1)
        on_d = sum(online_decision(seg, utts, t) != online_decision(seg, scrambled, t)
                   for t in rng)
        off_d = sum(offline_decision(seg, utts, t) != offline_decision(seg, scrambled, t)
                    for t in rng)
        n_online_moved += int(on_d > 0)
        n_control_moved += int(off_d > 0)
        per_doc.append((n, cut, on_d, off_d))

    print(f"  1) 입력 감사(전 문서)        : {'✅ forward 가 t 개만 받음' if audit_ok else '🔴 초과 입력'}")
    print(f"  2) 온라인 미래-교란 불변      : 변한 문서 {n_online_moved}/{len(per_doc)}"
          f"  {'✅ 전부 불변' if n_online_moved == 0 else '🔴 미래 의존'}")
    print(f"  3) 대조군(오프라인) 변화      : 변한 문서 {n_control_moved}/{len(per_doc)}"
          f"  {'✅ 검사 유효' if n_control_moved else '⚠️ 전 문서 무변화 → 검사 무력'}")
    print("     문서별 (턴, 교란시작, 온라인변화, 오프라인변화):")
    print("       " + " ".join(f"({a},{b},{c},{d})" for a, b, c, d in per_doc))

    ok = audit_ok and n_online_moved == 0 and n_control_moved > 0
    if not n_control_moved:
        print("     ⚠️ 대조군이 전 문서에서 안 변했다 → 2) 의 통과가 인과성의 증거가 못 된다.")
    print(f"  → {name}: {'✅ prefix-causal 확인' if ok else '🔴 미확정/실패'}")
    return ok


def main() -> int:
    from super_dialseg.models.texttiling.modeling_texttiling import TexttilingSegmenter

    # dialseg711 을 쓴다 — 중앙 26턴으로 뒤 절반을 갈아끼울 여유가 있다.
    # (tiage 는 최대 16턴, superseg 는 중앙 13턴이라 교란 폭이 부족하다.)
    results = {"TextTiling": check(TexttilingSegmenter, "TextTiling", "dialseg711")}

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k:<20} {'✅' if v else '🔴'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
