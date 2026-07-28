#!/usr/bin/env python3
"""no-thinking 지속 처리량 실측 — 운영 동시성에서 완료 콜/초·tok/s.

실전 배치20 Stage-2 프롬프트를 목표 동시성으로 계속 유지(프롬프트 순환), 정상구간(warmup 후)에서
완료 콜 수·출력토큰을 세어 sustained throughput 을 잰다. Stage-1(전체 스토리 입력) 콜도 별도 소표본 측정.
"""
from __future__ import annotations
import json, sys, time, threading, itertools, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
import llm_s5_states as S  # noqa: E402
from openai import OpenAI  # noqa: E402

import os
# 사용법: throughput_probe.py [동시성] [base_url]   — base_url 은 env PROBE_BASE 로도 지정 가능
BASE = (sys.argv[2] if len(sys.argv) > 2
        else os.environ.get("PROBE_BASE", "http://localhost:8001/v1")); MODEL = "qwen36-35b"
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 280
WARMUP = 10.0; MEASURE = 30.0
REP = ROOT / "outputs/eacl_phase0"


def s2_prompts():
    out = []
    rng = random.Random(3)
    docs = sorted({p.parent.name for p in REP.glob("repcorpus_s5_q36_35b_v2/*/story_lists.json")})
    for doc in docs:
        try:
            lists = json.load(open(REP / "repcorpus_s5_q36_35b_v2" / doc / "story_lists.json"))["lists"]
        except Exception:
            continue
        m = next(REP.glob(f"{doc}*_master.jsonl"), None) or (REP / f"{doc}_michelmann_master.jsonl")
        if not m.exists():
            continue
        sents = S.sequence(S.read(m)); ids = list(range(len(sents))); rng.shuffle(ids)
        for ax in ["space", "intentionality", "time", "protagonist", "causality"]:
            for k in range(0, len(ids), 20):
                ch = ids[k:k + 20]
                if len(ch) < 8:
                    continue
                out.append((ax, lists, S.stage2_prompt(ax, lists, [(i, sents[i]) for i in ch])))
    rng.shuffle(out)
    return out


def call(c, ax, lists, pr):
    r = c.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=2000,
        response_format={"type": "json_schema", "json_schema":
                         {"name": f"{ax}_s", "strict": True, "schema": S.stage2_schema(ax, lists)}},
        messages=[{"role": "system", "content":
                   "You assign fixed categorical labels to one sentence at a time from a closed "
                   "list. You never compare sentences and never output free text."},
                  {"role": "user", "content": pr}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.usage.completion_tokens, r.usage.prompt_tokens


def main():
    prompts = s2_prompts()
    print(f"[probe] {len(prompts)} 배치20 프롬프트, 목표동시성 {CONC}, warmup {WARMUP}s + 측정 {MEASURE}s")
    c = OpenAI(base_url=BASE, api_key="x")
    inflight = {"n": 0}; done = {"calls": 0, "otok": 0, "ptok": 0, "lat": []}
    lock = threading.Lock(); stop = threading.Event()
    measuring = {"on": False}
    pi = itertools.cycle(range(len(prompts)))

    def worker(ax, lists, pr):
        t = time.time()
        try:
            ot, pt = call(c, ax, lists, pr)
            if measuring["on"]:
                with lock:
                    done["calls"] += 1; done["otok"] += ot; done["ptok"] += pt
                    done["lat"].append(time.time() - t)
        except Exception:
            pass
        finally:
            with lock:
                inflight["n"] -= 1

    ex = ThreadPoolExecutor(max_workers=CONC + 50)

    def feeder():
        while not stop.is_set():
            with lock:
                room = CONC - inflight["n"]
            for _ in range(max(0, room)):
                with lock:
                    inflight["n"] += 1
                ax, lists, pr = prompts[next(pi)]
                ex.submit(worker, ax, lists, pr)
            time.sleep(0.03)
    threading.Thread(target=feeder, daemon=True).start()

    time.sleep(WARMUP)
    measuring["on"] = True; t0 = time.time()
    time.sleep(MEASURE)
    measuring["on"] = False; dt = time.time() - t0
    stop.set()

    lat = sorted(done["lat"])
    cps = done["calls"] / dt
    print(f"\n=== no-thinking sustained @conc {CONC} ({dt:.0f}s 측정) ===")
    print(f"- 완료 콜: {done['calls']} → **{cps:.2f} 콜/s**")
    print(f"- 출력 tok/s: {done['otok']/dt:.0f}  |  입력(prefill) tok/s: {done['ptok']/dt:.0f}  "
          f"|  총 tok/s: {(done['otok']+done['ptok'])/dt:.0f}")
    print(f"- 콜당 출력 {done['otok']/max(1,done['calls']):.0f}tok / 입력 {done['ptok']/max(1,done['calls']):.0f}tok")
    if lat:
        print(f"- 콜 지연 median {lat[len(lat)//2]:.1f}s / p90 {lat[int(len(lat)*0.9)]:.1f}s")
    # 1만편 외삽
    S2 = 130530; S1 = 10000
    print(f"\n=== 1만편 외삽 (Stage-2 {S2:,}배치 @ {cps:.2f}콜/s) ===")
    s2_sec = S2 / cps
    print(f"- Stage-2 wall ≈ {s2_sec:.0f}s = {s2_sec/3600:.2f}h")
    ex.shutdown(wait=False)


if __name__ == "__main__":
    main()
