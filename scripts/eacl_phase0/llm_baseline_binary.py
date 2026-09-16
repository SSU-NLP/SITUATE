#!/usr/bin/env python3
"""SeCom-binary LLM baseline — **온라인·인과적**. 예측을 `baseline_compare` 와 같은 형식으로 덤프.

## 무엇이고 왜 이 형태인가
`scripts/segmentation_prompts/` 의 SeCom 계열 중 **binary 판만** 쓴다:
- `baseline_segment_v1.md`(원본 partition)은 **입력 전체를 한 번에 보고** 자른다 = **오프라인**.
  우리 주장은 *온라인·LLM-free 실시간*이므로 **오프라인은 경쟁상대가 아니다** → 쓰지 않는다.
- `baseline_segment_binary_v1.md` = SeCom 의 Context/Goal/기준을 **그대로 두고** Question 만
  *"현재 턴이 NEW 인가 SAME 인가"* 로 바꾼 것. **full-past = 미래 미사용** → 온라인 peer.
- `judge_binary_neutral_v1.md` 는 **우리 자체 judge** 이므로 baseline 이 아니다 — 쓰지 않는다.

## 🔴 도메인 적응 (정직 기록)
SeCom 프롬프트는 *"multi-turn **conversation**"* · *"**turn** ... by one **speaker**"* 로 쓰여 있다.
그런데 우리 주 무대 dev-3 는 **화자도 턴도 없는 서사 산문**이다. 원문을 그대로 쓰면 LLM 이
*대화적* 화제전환을 찾게 유도되어 **SeCom 에 불리한 허수아비**가 된다.
→ `baseline_segment_binary_narrative_v1.md` = **기준·출력형식·요구사항을 한 글자도 안 바꾸고**
도메인 명사만 교체(conversation→narrative · turn→sentence · speaker 제거).
**SeCom 원문이 아니라 도메인 적응판**임을 결과에 반드시 명시한다.
대화 데이터(DTS)에는 **원문 그대로**(`baseline_segment_binary_v1.md`) 쓴다.

## 배포 규약
경계 개수를 LLM 이 스스로 정한다(gold 개수 미사용) → **deployable**. 미래 문장도 안 본다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from baseline_compare import OUT, load_dataset, score_A, score_B  # noqa: E402

PROMPTS = REPO / "scripts/segmentation_prompts"


def build(sents: list[str], cur: int, past: int, narrative: bool,
          criterion: str = "topic") -> str:
    """cur 번째 문장까지의 full-past 프롬프트. `past` 로 최대 되돌아볼 문장 수를 제한(0=무제한)."""
    # 🔴 2026-08-02(D-B62): 이전 판은 non-narrative 일 때 `criterion` 을 **무시**해 DTS 가 항상
    # topic 문구를 받았다. event 통일(사용자 결정) 위해 대화 도메인 event 판을 추가했다.
    ev = "event_" if criterion == "event" else ""
    dom = "narrative_" if narrative else ""
    tmpl = (PROMPTS / f"baseline_segment_binary_{ev}{dom}v1.md").read_text(encoding="utf-8")
    lo = 0 if past <= 0 else max(0, cur - past)
    unit = "Sentence" if narrative else "Turn"
    body = "\n\n".join(f"[{unit} {i}]: {sents[i]}" for i in range(lo, cur + 1))
    return tmpl.replace("{text_to_be_segmented}", body).replace("{cur}", str(cur))


def build_stateful(sents: list[str], cur: int, seg_of: list[int], past: int,
                   criterion: str = "topic", narrative: bool = True) -> str:
    """상태 있는 판 — **자기가 이미 확정한 경계**를 Segment 번호로 함께 보여준다.

    무상태판은 매 문장을 독립 판정해 *"이 문서에 경계가 대략 몇 개여야 하는가"* 감각이 없다
    (dev-3 에서 gold 66 대비 166개 = 2.5배 과분절). 여기서는 직전까지의 자기 분절을 문맥에
    넣어 **방금 경계를 찍었다는 사실**과 **현 세그먼트 길이**를 알려준다.
    ⚠️ 그래서 문서 내에서는 **순차 실행**이다(결정 t 가 결정 <t 에 의존). 미래는 여전히 안 본다.
    """
    ev = "event_" if criterion == "event" else ""
    # 🔴 2026-08-02(D-B62): 이전 판은 대화에도 **서사 템플릿**을 먹였다. 도메인 분기 추가.
    name = (f"baseline_segment_binary_{ev}stateful_narrative_v1.md" if narrative
            else f"baseline_segment_binary_{ev}stateful_v1.md")
    tmpl = (PROMPTS / name).read_text(encoding="utf-8")
    lo = 0 if past <= 0 else max(0, cur - past)
    unit = "Sentence" if narrative else "Turn"
    body = "\n\n".join(f"[Segment {seg_of[i]} | {unit} {i}]: {sents[i]}" for i in range(lo, cur)) \
        + f"\n\n[Segment ? | {unit} {cur}]: {sents[cur]}"
    cur_seg = seg_of[cur - 1]
    return (tmpl.replace("{text_to_be_segmented}", body)
            .replace("{cur_seg_len}", str(sum(1 for s in seg_of[:cur] if s == cur_seg)))
            .replace("{cur_seg}", str(cur_seg)).replace("{next_seg}", str(cur_seg + 1))
            .replace("{cur}", str(cur)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dev3")
    # 🔴 criterion=event: SeCom 의 **topic** 기준 대신 **B3-EventRecall(Panela 2025)의 사건 정의**
    #    ("An event is an ongoing coherent situation ... one event ends, and another begins")를 쓴다.
    #    바뀌는 것은 기준 문구 2줄뿐이며 Data·상태서술·출력형식은 동일하다.
    #    ⚠️ 이것은 **B3 가 아니다** — B3 원본은 오프라인(전문 복사)·20회 합의다. 사건 정의만 채용한
    #    **온라인 이진판**이며, 라벨·보고에 반드시 그렇게 적는다.
    # 🔒 B3 절차 정렬: 오프라인 B3 가 reps 회 반복 + 합의를 쓰므로 온라인도 같은 집계를 써야
    #    "시점(온라인/오프라인) 효과"와 "집계(1회/합의) 효과"가 안 섞인다.
    #    전이 단위 다수결 — B3 의 top-n 합의와 형태는 다르나(이진 판정이라 개수 개념이 없다)
    #    "R회 독립 실행 후 합의"라는 절차는 동일하다. 이 차이는 보고에 명시한다.
    ap.add_argument("--reps", type=int, default=1, help="전이당 반복 호출 수. B3 정렬시 10")
    ap.add_argument("--criterion", default="topic", choices=["topic", "event"],
                    help="topic = SeCom 원기준 · event = B3 사건 정의 채용")
    ap.add_argument("--model", default="openrouter/openai/gpt-4o")
    # 🔴 GPT-5 계열 대응 (2026-07-30): 추론 토큰을 먼저 쓰는 모델은 max_tokens=4 면 답이 안 나오고
    #    예외 처리로 전부 SAME 이 되어 **무경계 하한이 "모델 성능"으로 오독된다.**
    #    D-B31 의 `gpt-4 -> gpt-4.1` 교정과 같은 범주 = 성능 목적이 아니라 **비실행성 교정**이다.
    #    출력은 어느 모델이든 한 단어이므로 상한을 늘려도 gpt-4o 결과는 안 바뀐다.
    ap.add_argument("--max-tokens", type=int, default=4,
                    help="응답 상한. 추론 모델은 크게(예: 2048). 기본 4 = 기존 gpt-4o 조건 그대로")
    ap.add_argument("--reasoning-effort", default="",
                    help="'minimal' 등. 게이트웨이가 안 받으면 자동으로 빼고 재시도한다(로그에 기록)")
    ap.add_argument("--label", default=None, help="표에 쓸 이름 (기본 SeCom-binary/<model 끝>)")
    ap.add_argument("--past", type=int, default=0, help="0=full-past. >0 이면 최근 N문장으로 제한")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="문서당 전이 수 상한 (스모크용)")
    ap.add_argument("--stateful", action="store_true",
                    help="자기가 이미 확정한 경계를 문맥에 함께 제시(문서 내 순차 실행)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from secom_llm_eval import make_client
    client = make_client()
    narrative = args.dataset == "dev3" or args.dataset.startswith("test3")
    label = args.label or (f"SeCom-binary{'-stateful' if args.stateful else ''}"
                           f"/{args.model.split('/')[-1]}")
    print(f"[{label}] dataset={args.dataset} · criterion={args.criterion} · prompt="
          f"{'narrative 적응판' if narrative else ('대화 event 판' if args.criterion == 'event' else 'SeCom 원문')}"
          f"{' + 자기경계 상태' if args.stateful else ''} · past={'full' if args.past<=0 else args.past}",
          flush=True)

    data = load_dataset(args.dataset)
    cache_p = OUT / f"cache_{label.replace('/','_')}_{args.dataset}.json"
    cache: dict[str, str] = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    _no_re = {"v": False}          # reasoning_effort 미지원 판정 (1회 감지 후 전역 비활성)
    _empty = {"v": 0}              # 본문이 빈 응답 수 (추론 토큰 소진 징후)
    _fail = {"v": 0}               # 5회 재시도 후 실패 -> SAME 로 떨어진 수

    def ask(key: str, prompt: str) -> str:
        if key in cache:
            return cache[key]
        if args.reps > 1:                                  # R회 독립 호출 -> 다수결
            with ThreadPoolExecutor(max_workers=min(args.reps, 10)) as ex:
                votes = list(ex.map(lambda _: _ask_once(prompt), range(args.reps)))
            new_n = sum(1 for v in votes if v.startswith("NEW"))
            return "NEW" if new_n * 2 > len(votes) else "SAME"
        return _ask_once(prompt)

    def _ask_once(prompt: str) -> str:
        use_re = bool(args.reasoning_effort) and not _no_re["v"]
        for attempt in range(5):
            try:
                kw = {"reasoning_effort": args.reasoning_effort} if use_re else {}
                r = client.chat.completions.create(
                    model=args.model, max_tokens=args.max_tokens, temperature=0,
                    messages=[{"role": "user", "content": prompt}], **kw)
                txt = (r.choices[0].message.content or "").strip().upper()
                if not txt:                               # 추론에 토큰을 다 써 본문이 빈 경우
                    _empty["v"] += 1
                return txt
            except Exception as e:
                if use_re and attempt == 0:               # reasoning_effort 미지원 -> 빼고 재시도
                    _no_re["v"] = True
                    use_re = False
                    print(f"  [warn] reasoning_effort 미지원 -> 제외하고 진행 ({type(e).__name__})",
                          flush=True)
                    continue
                time.sleep(2 * (attempt + 1))
        _fail["v"] += 1
        return "SAME"                                     # 실패는 보수적으로 SAME

    def run_doc(item):
        did, sents, gold = item
        n_tr = len(gold) if args.limit <= 0 else min(len(gold), args.limit)
        v = np.zeros(len(gold), np.int8)
        if args.stateful:
            # 순차 — 결정 t 가 결정 <t 에 의존. 문서 간에는 병렬.
            seg_of = [0]
            for j in range(n_tr):
                cur = j + 1
                o = ask(f"{did}|{j}",
                        build_stateful(sents, cur, seg_of, args.past, args.criterion, narrative))
                new = o.startswith("NEW")
                v[j] = 1 if new else 0
                seg_of.append(seg_of[-1] + 1 if new else seg_of[-1])
        else:
            # 전이 j = (sent_j -> sent_{j+1}) 이므로 "sent_{j+1} 가 NEW 인가" 를 묻는다
            jobs = [(f"{did}|{j}", build(sents, j + 1, args.past, narrative, args.criterion)) for j in range(n_tr)]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                outs = list(ex.map(lambda kv: ask(*kv), jobs))
            v[:n_tr] = np.asarray([1 if o.startswith("NEW") else 0 for o in outs], np.int8)
        print(f"  {did}: {n_tr} 콜 · NEW {int(v.sum())} / gold {int(gold.sum())}", flush=True)
        return did, v

    t0 = time.perf_counter()
    if args.stateful:
        with ThreadPoolExecutor(max_workers=len(data)) as ex:   # 문서 간 병렬, 문서 내 순차
            preds = dict(ex.map(run_doc, data))
    else:
        preds = dict(run_doc(d) for d in data)
    cache_p.write_text(json.dumps(cache), encoding="utf-8")

    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    (OUT / "preds" / f"{label.replace('/','_')}_{args.dataset}.json").write_text(
        json.dumps({d: preds[d].tolist() for d in preds}), encoding="utf-8")
    A, B = score_A(data, preds), score_B(data, preds)
    print(f"\n{label}  A-Score {A['score']:.4f} (Pk {A['pk']:.3f} WD {A['wd']:.3f} "
          f"F1 {A['f1']:.3f}, 경계 {A['n_pred']})  | B {B['score']:.4f}  "
          f"| {time.perf_counter()-t0:.0f}s")
    rp = OUT / "llm_results.json"
    allr = json.loads(rp.read_text()) if rp.exists() else {}
    allr[f"{args.dataset}|{label}"] = {"A": A, "B": B, "model": args.model,
                                       "prompt": ("stateful_narrative_v1" if args.stateful else
                                                  ("narrative_v1" if narrative else "secom_v1")),
                                       "past": args.past, "limit": args.limit}
    rp.write_text(json.dumps(allr, indent=2, default=float))
    print(f"-> {rp}")


if __name__ == "__main__":
    main()
