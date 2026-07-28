"""Prepare LoCoMo for the segmenter-swap experiment (evidence + category aware).

Unlike ``01b_prepare_locomo.py`` (which only kept text for SeCom's schema), this
records the *turn-level structure* needed to measure **gold-evidence recall@k**:

- Each conversation keeps its ordered ``sessions`` where every turn is a dict
  ``{dia_id, speaker, text}`` (dia_id = LoCoMo's ``D{session}:{turn}`` id — the
  key the QA ``evidence`` field points at).
- Each QA keeps ``evidence`` (list of gold dia_ids) + ``category`` (1-5).

Output: ``outputs/experiments/2026-07-15_locomo_segmenter_swap/data/locomo_turns.jsonl``
one row per conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "benchmarks/locomo/locomo10.json"
OUT = (
    REPO_ROOT
    / "outputs/experiments/2026-07-15_locomo_segmenter_swap/data/locomo_turns.jsonl"
)


def session_keys(conv: dict) -> list[str]:
    keys = [
        k
        for k in conv
        if k.startswith("session_")
        and not k.endswith("date_time")
        and isinstance(conv[k], list)
    ]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def turn_text(turn: dict) -> str:
    text = (turn.get("text") or "").strip()
    if not text:
        cap = (turn.get("blip_caption") or "").strip()
        text = f"[shares an image] {cap}" if cap else "[non-text turn]"
    return text


def qa_answer(q: dict) -> str:
    ans = q.get("answer")
    if ans is None:
        ans = q.get("adversarial_answer")
    return "" if ans is None else str(ans)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC.read_text())
    n_conv = n_turn = n_qa = 0
    cat_counter: dict = {}
    with OUT.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(data):
            conv = sample["conversation"]
            sessions: list[list[dict]] = []
            all_dia = set()
            for sk in session_keys(conv):
                sess = []
                for t in conv[sk]:
                    d = {
                        "dia_id": t.get("dia_id"),
                        "speaker": t.get("speaker", "Unknown"),
                        "text": turn_text(t),
                    }
                    sess.append(d)
                    all_dia.add(d["dia_id"])
                if sess:
                    sessions.append(sess)
                    n_turn += len(sess)
            qa_rows = []
            for q in sample.get("qa", []):
                ev = q.get("evidence") or []
                # keep only evidence ids that actually resolve to a turn
                ev = [e for e in ev if e in all_dia]
                cat = q.get("category")
                cat_counter[cat] = cat_counter.get(cat, 0) + 1
                qa_rows.append(
                    {
                        "question": str(q["question"]),
                        "answer": qa_answer(q),
                        "evidence": ev,
                        "category": cat,
                    }
                )
            row = {
                "conversation_id": str(sample.get("sample_id", idx)),
                "sessions": sessions,
                "qa": qa_rows,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_conv += 1
            n_qa += len(qa_rows)
    print(f"wrote {n_conv} conversations -> {OUT}")
    print(f"  turns={n_turn}  questions={n_qa}")
    print(f"  categories={dict(sorted(cat_counter.items(), key=lambda x: str(x[0])))}")


if __name__ == "__main__":
    main()
