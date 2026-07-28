#!/usr/bin/env python3
"""100문장 human-verify 세트를 Qwen3.6-35B **thinking-on**으로 재라벨 + 동시성 램프 측정.

- 라벨: 실제 Stage-2 프롬프트/스키마(llm_s5_states) 재사용, extra_body enable_thinking=True,
  서버는 --reasoning-parser qwen3 로 기동(추론 토큰 생성 후 guided JSON). off와 동일 프롬프트라 clean.
- 동시성 램프: client offered concurrency 를 8→16→32→64→100 으로 키우며 서버 /metrics
  (num_requests_running/waiting, gpu_cache_usage)를 0.5s 폴링 → 실제 도달 동시성 곡선.
- 비교: thinking 라벨 vs 사람(중립/비중립) vs off-pipeline Qwen(compare_data qwen). 축별 over/under·flip.
산출: outputs/experiments/2026-07-24_qwen_human_verify/THINKING_COMPARE.md (+ ramp csv, labels json)
"""
from __future__ import annotations
import json, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
import llm_s5_states as S  # noqa: E402
from openai import OpenAI  # noqa: E402

OUT = ROOT / "outputs/experiments/2026-07-24_qwen_human_verify"
BASE = "http://localhost:8001/v1"
MODEL = "qwen36-35b"
METRICS = "http://localhost:8001/metrics"
NEUT = {"space": "unspecified", "intentionality": "none", "time": "absent",
        "protagonist": "none", "causality": "ongoing"}
FIELD = {"space": "location", "intentionality": "active_goal", "time": "temporal_shift_marker",
         "protagonist": "perspective", "causality": "aspect"}


def make_lists(axis: str, choices: list[str]) -> dict:
    locs = [c for c in choices if c != "unspecified"] if axis == "space" else []
    goals = [c for c in choices if c != "none"] if axis == "intentionality" else []
    return {"cast": [], "locations": locs, "goals": goals, "protagonist": None}


def label_one(c: OpenAI, case: dict) -> dict:
    axis = case["axis"]; lists = make_lists(axis, case["choices"])
    item = [(0, case["sentence"])]
    eb = {"chat_template_kwargs": {"enable_thinking": True}}
    t = time.time()
    r = c.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=8000,
        response_format={"type": "json_schema", "json_schema":
                         {"name": f"{axis}_s5_states", "strict": True,
                          "schema": S.stage2_schema(axis, lists)}},
        messages=[{"role": "system", "content":
                   "You assign fixed categorical labels to one sentence at a time from a closed "
                   "list. You never compare sentences and never output free text."},
                  {"role": "user", "content": S.stage2_prompt(axis, lists, item)}],
        extra_body=eb)
    dt = time.time() - t
    obj = json.loads(r.choices[0].message.content)["items"][0]
    val = obj.get(FIELD[axis])
    return {"row_id": case["row_id"], "axis": axis, "think": val, "lat": dt,
            "out_tok": r.usage.completion_tokens}


def poll_metrics(stop_ev, samples, offered_ref):
    while not stop_ev.is_set():
        try:
            txt = urllib.request.urlopen(METRICS, timeout=2).read().decode()
            run = wait = cache = None
            for ln in txt.splitlines():
                if ln.startswith("#"):
                    continue
                if "num_requests_running{" in ln: run = float(ln.split()[-1])
                elif "num_requests_waiting{" in ln: wait = float(ln.split()[-1])
                elif "gpu_cache_usage_perc{" in ln: cache = float(ln.split()[-1])
            samples.append((round(time.time(), 2), offered_ref[0], run, wait, cache))
        except Exception:
            pass
        time.sleep(0.5)


def main():
    cases = json.load(open(OUT / "compare_data.json"))
    c = OpenAI(base_url=BASE, api_key="x")

    # 동시성 램프: offered cap 을 단계적으로 늘림. semaphore permits 를 controller 가 증설.
    LEVELS = [8, 16, 32, 64, 100]
    sem = threading.Semaphore(LEVELS[0])
    offered_ref = [LEVELS[0]]
    granted = LEVELS[0]

    stop_ev = threading.Event()
    samples: list = []
    poller = threading.Thread(target=poll_metrics, args=(stop_ev, samples, offered_ref), daemon=True)
    poller.start()

    results: list = []
    res_lock = threading.Lock()
    errors: list = []

    def work(case):
        sem.acquire()
        try:
            r = label_one(c, case)
            with res_lock:
                results.append(r)
        except Exception as e:
            with res_lock:
                errors.append((case["row_id"], f"{type(e).__name__}: {str(e)[:120]}"))
        finally:
            sem.release()

    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=100)
    futs = [ex.submit(work, cs) for cs in cases]

    # controller: 매 3초마다 다음 레벨로 permit 증설 (offered concurrency 램프)
    def ramp():
        nonlocal granted
        for lvl in LEVELS[1:]:
            time.sleep(3.0)
            add = lvl - granted
            for _ in range(add):
                sem.release()  # permit 추가 → 유효 동시성 상한 증가
            granted = lvl
            offered_ref[0] = lvl
    ramper = threading.Thread(target=ramp, daemon=True)
    ramper.start()

    for f in futs:
        f.result()
    wall = time.time() - t0
    ex.shutdown()
    stop_ev.set(); poller.join(timeout=2)

    # ramp 곡선: offered 레벨별 관측된 running peak/mean, waiting peak
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "thinking_ramp.csv", "w") as f:
        f.write("t,offered,running,waiting,gpu_cache\n")
        for s in samples:
            f.write(",".join("" if v is None else str(v) for v in s) + "\n")
    json.dump(results, open(OUT / "thinking_labels.json", "w"), ensure_ascii=False, indent=1)

    # 레벨별 실제 동시성 요약
    lvl_stat = {}
    for (t, off, run, wait, cache) in samples:
        if run is None:
            continue
        lvl_stat.setdefault(off, {"run": [], "wait": []})
        lvl_stat[off]["run"].append(run)
        lvl_stat[off]["wait"].append(wait or 0)

    # === 비교: think vs 사람 vs off-qwen ===
    by_rid = {c["row_id"]: c for c in cases}
    think = {r["row_id"]: r["think"] for r in results}
    axes = ["space", "intentionality", "time", "protagonist", "causality"]

    def is_neut(axis, lab):
        return lab == NEUT[axis]

    rows_md = []
    per_axis = {}
    for ax in axes:
        rs = [c for c in cases if c["axis"] == ax and c["row_id"] in think]
        n = len(rs)
        agree_t = over_t = under_t = 0        # think vs 사람
        agree_off = 0                          # off vs 사람 (재계산, 대조)
        flips = 0                              # think != off
        flip_to_correct = flip_to_wrong = 0
        for c in rs:
            hn = (c["human"] == "no")          # 사람=중립
            tl = think[c["row_id"]]; ofl = c["qwen"]
            t_neut = is_neut(ax, tl); o_neut = is_neut(ax, ofl)
            if t_neut == hn: agree_t += 1
            elif hn and not t_neut: over_t += 1
            elif (not hn) and t_neut: under_t += 1
            if o_neut == hn: agree_off += 1
            if tl != ofl:
                flips += 1
                # off가 사람과 불일치였는데 think가 일치로 바뀜?
                off_ok = (o_neut == hn); think_ok = (t_neut == hn)
                if think_ok and not off_ok: flip_to_correct += 1
                if off_ok and not think_ok: flip_to_wrong += 1
        per_axis[ax] = dict(n=n, agree_t=agree_t, agree_off=agree_off, over_t=over_t,
                            under_t=under_t, flips=flips, ftc=flip_to_correct, ftw=flip_to_wrong)

    # 리포트
    L = ["# Qwen3.6-35B **thinking-on** 재라벨 + 동시성 램프 (human-verify 100문장)", "",
         f"> 모델 Qwen/Qwen3.6-35B-A3B(fp8), `--reasoning-parser qwen3`, enable_thinking=True, "
         f"guided json_schema. off와 동일 Stage-2 프롬프트. n={len(results)}/{len(cases)} 완료, "
         f"에러 {len(errors)}. wall {wall:.1f}s.", ""]
    tot_out = sum(r["out_tok"] for r in results)
    lats = sorted(r["lat"] for r in results)
    L += ["## 처리량", "",
          f"- 총 출력 {tot_out} tok / {wall:.1f}s = **{tot_out/wall:.0f} tok/s**(집계).",
          f"- 호출당 지연 median {lats[len(lats)//2]:.1f}s / p90 {lats[int(len(lats)*0.9)]:.1f}s / "
          f"max {lats[-1]:.1f}s. 평균 출력 {tot_out/max(1,len(results)):.0f} tok/call.", "",
          "## 동시성 램프 (offered cap → 서버 실제 running/waiting)", "",
          "| offered | running mean | running peak | waiting peak | gpu_cache peak |",
          "|---|---|---|---|---|"]
    for lvl in LEVELS:
        st = lvl_stat.get(lvl)
        if not st or not st["run"]:
            L.append(f"| {lvl} | (표본없음) | | | |"); continue
        rmean = sum(st["run"]) / len(st["run"]); rpk = max(st["run"]); wpk = max(st["wait"])
        cpk = max((c for (_, o, r, w, c) in samples if o == lvl and c is not None), default=0)
        L.append(f"| {lvl} | {rmean:.1f} | {rpk:.0f} | {wpk:.0f} | {cpk:.2f} |")
    # 전역 peak
    all_run = [r for (_, _, r, _, _) in samples if r is not None]
    L += ["", f"- **관측된 최대 동시 실행(running peak) = {max(all_run):.0f}**"
          f" (offered 100까지 올렸을 때). 그 이상은 waiting 으로 큐잉.", ""]

    L += ["## thinking vs 사람(중립/비중립) — off 대비", "",
          "| 축 | n | think 일치 | (off 일치) | Δ | think over | think under | flips | →정답 | →오답 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    T = dict(n=0, at=0, ao=0, ov=0, un=0, fl=0, ftc=0, ftw=0)
    for ax in axes:
        p = per_axis[ax]
        d = p["agree_t"] - p["agree_off"]
        L.append(f"| {ax} | {p['n']} | {p['agree_t']} | {p['agree_off']} | {d:+d} | "
                 f"{p['over_t']} | {p['under_t']} | {p['flips']} | {p['ftc']} | {p['ftw']} |")
        T["n"] += p["n"]; T["at"] += p["agree_t"]; T["ao"] += p["agree_off"]
        T["ov"] += p["over_t"]; T["un"] += p["under_t"]; T["fl"] += p["flips"]
        T["ftc"] += p["ftc"]; T["ftw"] += p["ftw"]
    L.append(f"| **전체** | {T['n']} | **{T['at']}** | {T['ao']} | {T['at']-T['ao']:+d} | "
             f"{T['ov']} | {T['un']} | {T['fl']} | {T['ftc']} | {T['ftw']} |")
    L += ["", "> Δ = think 일치 − off 일치(둘 다 vs 사람). flips=off 대비 라벨 바뀐 수, "
          "→정답=off가 틀렸는데 think가 맞춤, →오답=off 맞았는데 think가 틀림.", ""]
    if errors:
        L += ["## 에러", ""] + [f"- {rid}: {e}" for rid, e in errors[:20]]

    (OUT / "THINKING_COMPARE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[written] {OUT/'THINKING_COMPARE.md'}  labels+ramp csv 동봉")


if __name__ == "__main__":
    main()
