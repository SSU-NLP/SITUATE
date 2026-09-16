#!/usr/bin/env python3
"""B3-EventRecall 을 **저자 모델(Llama)·저자 파이프라인**으로 dev-3 에 실행한다 — API 0.

## 왜 이걸 돌리나
저장소에는 이미 B3 방법(전문 복사 + 사건 경계마다 줄바꿈)을 돌린 기록이 있다
(`outputs/eacl_phase0/dev_b3ref/`, **gpt-4.1** · temp 0 · 20회). 그런데 두 가지가 저자와 다르다:

1. **모델** — 논문은 GPT-4 와 **Meta-Llama-3-8B-Instruct** 를 쓴다. 우리는 gpt-4.1 을 썼다.
2. **파싱** — 우리는 `build_tier2_gold.py::run_to_transitions`(세그먼트별 누적 단어수를
   문장 경계에 스냅)를 썼고, 저자는 `parse_llm_output` 정규식 7단계 → `\\n\\n` 분리를 쓴다.

Llama 경로는 **로컬 HF pipeline** 이라 API 예산과 무관하다. 이 스크립트가 그 구멍을 닫는다.

## 🔴 미리 밝히는 세 가지 제약
1. **모델이 정확히 같지 않다.** 논문은 `Meta-Llama-3-8B-Instruct`(3.0), 우리 HF 캐시에 있는 것은
   **Llama-3.1-8B-Instruct** 다. 다른 모델이므로 "저자 설정 그대로"가 아니다.
2. **temperature 는 Llama 경로에서 아무 효과가 없다.** 저자 코드가 `temperature==0 → 0.1` 로
   강제하지만, 정작 `do_sample` 을 넘기지 않아 HF 기본값(`False`)대로 **greedy 디코딩**이 된다
   (`module/segmentation.py:54-57`). 즉 이 경로는 애초에 **결정적**이고 temp 0 여부를 논할 대상이
   아니다. 우리도 `do_sample` 을 넘기지 않아 같은 동작을 따른다. `--sample` 로 켜면
   **저자와 다른 조건**이 되며 산출 파일명에 `sample` 로 표시된다.
3. **논문 수치와 대조하지 않는다.** Hamming 이 경계 희소성에 지배되어 우리 gold 에서는
   무경계 하한보다 나쁘게도 나온다는 것을 이미 실측했다(`rescore_existing_runs.py`).
   목표는 **"저자 모델·저자 파싱으로도 돌렸다"** 까지다.

## 출력 길이 주의
이 방법은 **원문 전체를 복사**하게 하므로 출력이 입력보다 길다. 저자 기본값
`max_new_tokens=4096` 은 dev-3 에서 tunnel(≈4.6k토큰)·monkey(≈6.8k토큰)를 **자른다**.
기본은 저자값 그대로 두고, `--max-new-tokens` 로 올린 조건을 **별도 이름**으로 함께 낸다
([[no-script-edit-during-run]]: 조건을 산출 파일명에 박는다).

    .venv-dialstart/bin/python EACL/methods/event_recall/run_llama_dev3.py \\
        --model /root/.cache/huggingface/... --max-new-tokens 4096
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHODS))

from event_recall.segment import segment_text_llama  # noqa: E402

BASE = REPO / "outputs/eacl_phase0"
OUT = REPO / "outputs/experiments/2026-08-03_eventrecall_llama_dev3"
DOCS = ("pieman", "tunnel", "monkey")


def load_doc(doc: str) -> tuple[list[str], list[int]]:
    rows = [json.loads(l) for l in (BASE / f"{doc}_michelmann_master.jsonl").open()]
    sents = [rows[0]["previous"]] + [r["current"] for r in rows]
    gold = [int(r["boundary"]) for r in rows]        # 전이 i = 문장 i+1 앞
    return sents, gold


def events_to_boundaries(events: list[str], sents: list[str]) -> list[int]:
    """저자 사건 리스트 → 문장 경계 인덱스.

    사건은 원문의 연속 구간이므로 **누적 단어 수**로 문장 경계에 스냅한다
    (`build_tier2_gold.py::run_to_transitions` 와 같은 규약 — 우리 채점기와 축을 맞추기 위함).
    """
    import re

    word = re.compile(r"\w+")
    cum, acc = [], 0
    for s in sents:
        acc += len(word.findall(s))
        cum.append(acc)
    total = cum[-1]

    bounds, off = [], 0
    for ev in events[:-1]:
        off += len(word.findall(ev))
        if off <= 0 or off >= total:
            continue
        # 가장 가까운 문장 끝으로 스냅 → 그 다음 문장이 새 사건의 시작
        i = min(range(len(cum)), key=lambda k: abs(cum[k] - off))
        if 0 <= i < len(sents) - 1:
            bounds.append(i + 1)
    return sorted(set(bounds))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="🔴 논문은 Meta-Llama-3-8B-Instruct(3.0). 캐시에 있는 것은 3.1 이다")
    ap.add_argument("--iters", type=int, default=1, help="저자 기본값 1")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="저자 코드가 0 을 0.1 로 강제한다")
    ap.add_argument("--max-new-tokens", type=int, default=4096, help="저자 기본값 4096")
    ap.add_argument("--docs", default=",".join(DOCS))
    ap.add_argument("--sample", action="store_true",
                    help="🔴 저자와 다른 조건. 저자 코드는 do_sample 을 안 넘겨 greedy 다")
    args = ap.parse_args()

    from transformers import pipeline as hf_pipeline

    tag = f"maxnew{args.max_new_tokens}_{'sample' if args.sample else 'greedy'}"
    print(f"모델 {args.model} · iters {args.iters} · max_new_tokens {args.max_new_tokens} · "
          f"디코딩 {'sampling(저자와 다름)' if args.sample else 'greedy(저자 기본)'}")
    pipe = hf_pipeline("text-generation", args.model, device_map="auto")

    rows = {}
    for doc in args.docs.split(","):
        sents, gold = load_doc(doc)
        text = " ".join(sents)
        print(f"\n=== {doc} · 문장 {len(sents)} · gold 경계 {sum(gold)} ===")
        events, raws = segment_text_llama(
            text, model=args.model, iters=args.iters, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, pipe=pipe, sample=args.sample)

        per_iter = []
        for ev, raw in zip(events, raws):
            b = events_to_boundaries(ev, sents)
            # 복사 충실도 = 출력 단어수 / 원문 단어수. 1.0 에 못 미치면 잘렸다는 뜻
            import re
            w = re.compile(r"\w+")
            fidelity = len(w.findall(raw)) / max(1, len(w.findall(text)))
            per_iter.append({"n_events": len(ev), "boundaries": b,
                             "copy_fidelity": round(fidelity, 3)})
            print(f"  사건 {len(ev)}개 · 경계 {len(b)}개 · 복사충실도 {fidelity:.2f}"
                  f"{'  🔴 잘림' if fidelity < 0.8 else ''}")
        rows[doc] = {"n_sentences": len(sents), "n_gold": sum(gold), "iters": per_iter}

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"llama_dev3_{tag}.json"
    f.write_text(json.dumps({"config": vars(args), "results": rows},
                            ensure_ascii=False, indent=2))
    print(f"\n저장: {f}")
    print("🔴 논문 수치와 대조하지 않는다 — 근거는 이 파일 docstring §제약 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
