#!/usr/bin/env python3
"""Zehe et al. (2025) 프롬프팅 베이스라인 — 저자 `prompting/classify.py` 이식.

## 왜 이것만 실행 가능한가
2025 논문의 주력은 Train-Full 로 학습한 BERT-SSC 와 파인튜닝 Llama 인데, 학습 코퍼스는
저작권 자료이고 체크포인트도 공개되지 않았다(`REPRO.md` §2). **프롬프팅 baseline 만
학습이 필요 없어** 우리가 가진 STSS-Test-2 2편에서 그대로 돌릴 수 있다.

## 방법 (저자 코드 그대로)
문장마다 **이진 질의**를 한 번씩 던진다. 대화 기억은 매 호출 뒤 지워지므로 사실상 무상태다.

    프롬프트: "Does the sentence in <sentence>...</sentence> introduce the beginning of a new
    scene and a significant break in time, location or characters? Answer 'True' or 'False' ..."

문맥은 대상 문장을 가운데 두고 **앞뒤로 대칭 확장**하며 모델의 `context_size` 토큰
(= 공시 창 × 0.8, `INPUT_PERCENTAGE`)을 채운다. 응답 첫 줄에 `True`/`False` 가 없으면
최대 10회 재시도한다.

## 대조할 발표 수치 (논문 Table 2 · Test-Full · 완화 F1 t=3)
    llama3:8b 0.13 · llama3:70b 0.34 · llama3.1:405b 0.34 · gpt-4o-mini 0.14 · **gpt-4o 0.45**

🔴 **직접 대조는 성립하지 않는다.** 논문은 Test-Full(= Test-1 4편 + Test-2 2편 + OOD 3편)만
보고하는데 우리는 그중 **Test-2 2편**만 가지고 있다. Test-2 는 고전 문학이라 통속소설 중심의
Test-1 과 성격이 다르다(같은 모델이 test set 별로 크게 갈린다는 것이 논문 Table 1 의 관찰이다).
따라서 이 스크립트의 산출은 **참고 수치**이며 "0.45 를 재현했다/못 했다"로 쓸 수 없다.

## 🔴 현재 실행 불가
2026-08-03 기준 우리 LLM 프록시는 예산 소진(429 ExceededBudget)이다. 예산이 복구되면
아래 명령이 바로 돈다. 다른 모델로 바꿔 돌린 값은 이 논문의 수치와 무관하다.

    .venv/bin/python EACL/methods/ssc_gbert_2025/prompting.py --model gpt-4o --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHODS))

from common.llm_client import chat, make_client  # noqa: E402
from common.stss_data import DUMP_TO_DOC, load_doc  # noqa: E402
from common.stss_eval import (NOBORDER, iou_zehe2025, labels_from_boundaries,  # noqa: E402
                              relaxed_f1_zehe2025)

# 저자 코드 L34-40 문자열 그대로
PROMPT_CLASSIFY = (
    "Does the sentence in <sentence>...</sentence> introduce the beginning of a new scene and a "
    "significant break in time, location or characters? Answer 'True' or 'False' and provide a reason "
    "for your decision. A scene is defined as a segment of text with a coherent structure across the "
    "dimensions 'characters' (which characters are present in the narration), 'location' (where does the"
    " narration take place), and 'time' (continuous time in the narration). A significant break in any"
    " of these dimensions corresponds to a scenes change. ")

INPUT_PERCENTAGE = 0.8            # 저자 상수
# 저자 `Model` 정의의 context_size (창 × INPUT_PERCENTAGE). 논문 Table 2 에 나오는 것만.
CONTEXT_SIZE = {
    "gpt-4o": int(512 * INPUT_PERCENTAGE),
    "gpt-4o-mini": int(512 * INPUT_PERCENTAGE),
    "llama3:8b": int(512 * INPUT_PERCENTAGE),
    "llama3:70b": int(512 * INPUT_PERCENTAGE),
    "llama3.1:405b": int(512 * INPUT_PERCENTAGE),
}


def build_sentence_sample(sentences: list[str], i: int, context_size: int, encoding) -> str:
    """저자 `build_sentence_sample` 이식 — 대상 문장을 가운데 두고 앞뒤 대칭 확장.

    🔴 저자 루프는 '현재 문맥 + 다음 후보 2개'의 토큰 수로 조건을 검사한 뒤 후보를 넣는다.
    그래서 실제 길이가 `context_size` 를 살짝 넘을 수 있다. 그대로 옮긴다.
    """
    context = [f"<sentence>{sentences[i]}</sentence>"]
    prev_i, next_i = i - 1, i + 1
    while True:
        prev_t = sentences[prev_i] if prev_i >= 0 else ""
        next_t = sentences[next_i] if next_i < len(sentences) else ""
        probe = " ".join(context + [prev_t, next_t])
        if len(encoding.encode(probe)) >= context_size:
            break
        if prev_i < 0 and next_i >= len(sentences):
            break
        if prev_i >= 0:
            context.insert(0, sentences[prev_i])
            prev_i -= 1
        if next_i < len(sentences):
            context.append(sentences[next_i])
            next_i += 1
    return " ".join(context)


def classify_document(client, sentences: list[str], model: str, *, cache: Path | None = None,
                      limit: int | None = None) -> list[bool]:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    ctx = CONTEXT_SIZE.get(model, int(512 * INPUT_PERCENTAGE))
    labels: list[bool] = []
    if cache and cache.exists():
        labels = json.loads(cache.read_text())["labels"]
        print(f"    캐시에서 {len(labels)}개 이어서 시작")

    n = len(sentences) if limit is None else min(limit, len(sentences))
    for i in range(len(labels), n):
        sample = build_sentence_sample(sentences, i, ctx, encoding)
        decision = None
        for _ in range(10):                      # 저자의 retry_count < 10
            text = chat(client, model, PROMPT_CLASSIFY, sample, temperature=0.0)
            first = text.split("\n")[0]
            if "True" in first:
                decision = True
                break
            if "False" in first:
                decision = False
                break
        if decision is None:
            raise RuntimeError(f"문장 {i}: 10회 시도에도 True/False 를 못 받았다")
        labels.append(decision)
        if cache and (i + 1) % 25 == 0:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"labels": labels}))
            print(f"    {i + 1}/{n}", flush=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"labels": labels}))
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o", help="논문 Table 2 의 모델명")
    ap.add_argument("--limit", type=int, default=None, help="문장 수 상한(스모크용)")
    ap.add_argument("--tolerance", type=int, default=3, help="논문 표는 전부 t=3")
    args = ap.parse_args()

    out = METHODS / "ssc_gbert_2025/results"
    client = make_client()
    rows = []
    for name in DUMP_TO_DOC.values():
        doc = load_doc(name)
        sents = [doc.text[a:b] for a, b in zip(doc.sent_starts, doc.sent_ends)]
        print(f"  {name}: 문장 {len(sents)}")
        cache = out / f"cache_{args.model.replace('/', '_')}_{name}.json"
        labels = classify_document(client, sents, args.model, cache=cache, limit=args.limit)

        n = len(labels)
        pred = [(i, "Scene-to-Scene") for i, v in enumerate(labels) if v and i > 0]
        pred_lab = labels_from_boundaries(pred, n)
        for ver, gold in (("2025", doc.gold2025), ("2021", doc.gold2021)):
            gold_lab = labels_from_boundaries([(i, t) for i, t in gold if i < n], n)
            m = relaxed_f1_zehe2025(gold_lab, pred_lab, args.tolerance)
            rows.append({"doc": name, "gold": ver, "model": args.model, **m,
                         "iou": iou_zehe2025(gold_lab, pred_lab),
                         "n_pred": sum(1 for x in pred_lab if x != NOBORDER),
                         "n_gold": sum(1 for x in gold_lab if x != NOBORDER)})
            print(f"    [{ver} gold] 완화F1(t={args.tolerance}) {m['f1']:.3f} "
                  f"(P {m['precision']:.3f} R {m['recall']:.3f}) · 예측경계 {rows[-1]['n_pred']}")

    out.mkdir(parents=True, exist_ok=True)
    (out / f"prompting_{args.model.replace('/', '_')}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    print("\n🔴 논문 Table 2 는 Test-Full 수치다. 위 값은 Test-2 2편만이라 직접 대조 불가.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
