#!/usr/bin/env python3
"""배치20 + thinking-on 동시성 램프 — 실전 프롬프트로 KV 포화 지점 탐색.

실전 Stage-2 프롬프트(문장 20개/프롬프트, 실제 repcorpus 문장·Stage-1 목록)를 thinking-on으로
서버에 부하. offered 동시성을 8→160 단계로 올리며 각 레벨을 HOLD초 유지, 서버 /metrics
(running/waiting/kv_cache_usage/preemptions)를 폴링해 실제 도달 동시성과 포화 지점을 잰다.
프롬프트는 유니크 풀을 순환 재사용(부하 측정 목적, 라벨 품질 아님). 산출: 콘솔 표 + ramp csv.
"""
from __future__ import annotations
import json, sys, time, threading, urllib.request, random, itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
import llm_s5_states as S  # noqa: E402
from openai import OpenAI  # noqa: E402

OUT = ROOT / "outputs/experiments/2026-07-24_qwen_human_verify"
BASE = "http://localhost:8001/v1"; MODEL = "qwen36-35b"
METRICS = "http://localhost:8001/metrics"
LEVELS = [256, 320, 384, 448, 512]
HOLD = 15.0     # 각 레벨 유지 시간(초)
MAXTOK = 2000
THINKING = False   # no-thinking 벽 측정

REP = ROOT / "outputs/eacl_phase0"


def build_prompts() -> list[tuple]:
    """실전 배치20 프롬프트 풀 (docs × axes). 반환: [(axis, lists, prompt), ...]"""
    docs = sorted({p.parent.name for p in REP.glob("repcorpus_s5_q36_35b_v2/*/story_lists.json")})
    prompts = []
    rng = random.Random(7)
    for doc in docs:
        d = REP / "repcorpus_s5_q36_35b_v2" / doc
        try:
            lists = json.load(open(d / "story_lists.json"))["lists"]
        except Exception:
            continue
        master = REP / f"{doc}_michelmann_master.jsonl"
        if not master.exists():
            master = next(REP.glob(f"{doc}*_master.jsonl"), None)
        if not master or not master.exists():
            continue
        sents = S.sequence(S.read(master))
        ids = list(range(len(sents))); rng.shuffle(ids)
        for ax in ["space", "intentionality", "time", "protagonist", "causality"]:
            for k in range(0, len(ids), 20):
                chunk = ids[k:k + 20]
                if len(chunk) < 8:
                    continue
                items = [(i, sents[i]) for i in chunk]
                prompts.append((ax, lists, S.stage2_prompt(ax, lists, items)))
    rng.shuffle(prompts)
    return prompts


def call(c, axis, lists, prompt):
    r = c.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=MAXTOK,
        response_format={"type": "json_schema", "json_schema":
                         {"name": f"{axis}_s", "strict": True, "schema": S.stage2_schema(axis, lists)}},
        messages=[{"role": "system", "content":
                   "You assign fixed categorical labels to one sentence at a time from a closed "
                   "list. You never compare sentences and never output free text."},
                  {"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": THINKING}})
    return r.usage.completion_tokens


def read_metrics():
    run = wait = kv = preempt = None
    try:
        txt = urllib.request.urlopen(METRICS, timeout=2).read().decode()
        for ln in txt.splitlines():
            if ln.startswith("#"):
                continue
            if "num_requests_running{" in ln: run = float(ln.split()[-1])
            elif "num_requests_waiting{" in ln: wait = float(ln.split()[-1])
            elif "kv_cache_usage_perc{" in ln: kv = float(ln.split()[-1])
            elif "num_preemptions_total{" in ln: preempt = float(ln.split()[-1])
    except Exception:
        pass
    return run, wait, kv, preempt


def main():
    prompts = build_prompts()
    print(f"[build] {len(prompts)} 배치20 프롬프트 (docs×5축). HOLD {HOLD}s/level, max_tokens {MAXTOK}")
    c = OpenAI(base_url=BASE, api_key="x")

    target = {"n": LEVELS[0]}          # 목표 동시성(램프가 갱신)
    inflight = {"n": 0}
    lock = threading.Lock()
    stop = threading.Event()
    done_tok = {"n": 0, "reqs": 0}
    pi = itertools.cycle(range(len(prompts)))

    # 워커 풀: target 만큼 in-flight 유지하며 프롬프트 순환 발사
    def feeder():
        with ThreadPoolExecutor(max_workers=600) as ex:
            while not stop.is_set():
                with lock:
                    room = target["n"] - inflight["n"]
                if room <= 0:
                    time.sleep(0.05); continue
                for _ in range(room):
                    with lock:
                        inflight["n"] += 1
                    idx = next(pi)
                    ax, lists, pr = prompts[idx]
                    def job(ax=ax, lists=lists, pr=pr):
                        try:
                            t = call(c, ax, lists, pr)
                            with lock:
                                done_tok["n"] += t; done_tok["reqs"] += 1
                        except Exception:
                            pass
                        finally:
                            with lock:
                                inflight["n"] -= 1
                    ex.submit(job)
                time.sleep(0.05)
    fth = threading.Thread(target=feeder, daemon=True); fth.start()

    rows = []   # (level, run_mean, run_peak, wait_mean, wait_peak, kv_peak, preempt_delta, tok_s)
    base_pre = read_metrics()[3] or 0
    for lvl in LEVELS:
        with lock:
            target["n"] = lvl
        # 램프-업 안정화 잠깐
        t_end = time.time() + HOLD
        runs, waits, kvs = [], [], []
        tok0 = done_tok["n"]; pre0 = read_metrics()[3] or 0; tw0 = time.time()
        # 첫 3초는 안정화(측정 제외)
        time.sleep(min(3.0, HOLD * 0.2))
        while time.time() < t_end:
            run, wait, kv, pre = read_metrics()
            if run is not None: runs.append(run)
            if wait is not None: waits.append(wait)
            if kv is not None: kvs.append(kv)
            time.sleep(0.5)
        tok1 = done_tok["n"]; pre1 = read_metrics()[3] or 0; tw1 = time.time()
        rmean = sum(runs)/len(runs) if runs else 0
        rpeak = max(runs) if runs else 0
        wmean = sum(waits)/len(waits) if waits else 0
        wpeak = max(waits) if waits else 0
        kvpk = max(kvs) if kvs else 0
        toks = (tok1 - tok0)/(tw1 - tw0) if tw1 > tw0 else 0
        rows.append((lvl, rmean, rpeak, wmean, wpeak, kvpk, pre1 - pre0, toks))
        print(f"offered {lvl:3d} | running {rmean:5.1f} (peak {rpeak:3.0f}) | "
              f"waiting {wmean:5.1f} (peak {wpeak:3.0f}) | KV {kvpk*100:4.1f}% | "
              f"preempt +{pre1-pre0:.0f} | {toks:5.0f} tok/s")
        # 포화 판정: waiting 이 offered 의 25% 이상 지속 또는 KV>0.95 또는 preempt 발생
        if wmean > lvl * 0.25 or kvpk > 0.95 or (pre1 - pre0) > 0:
            print(f"  ^ 포화 신호(waiting↑ 또는 KV>95% 또는 preemption). offered {lvl}에서 서버가 다 못 돌림.")
    stop.set(); time.sleep(1)

    # 요약
    print("\n=== 요약 ===")
    ceil = max((r for r in rows), key=lambda r: r[2])   # running peak 최대
    # 실제 지속가능 동시성 = waiting 이 낮게 유지된 마지막 레벨의 running mean
    sustainable = [r for r in rows if r[3] < r[0] * 0.15 and r[6] == 0]
    print(f"- KV풀 1,297,785 tok. running peak 최대 관측 = {ceil[2]:.0f} (offered {ceil[0]}).")
    if sustainable:
        s = sustainable[-1]
        print(f"- 대기·선점 없이 안정적으로 돈 최대 = running~{s[1]:.0f} (offered {s[0]}).")
    save = OUT / "batch20_thinking_ramp.csv"
    with open(save, "w") as f:
        f.write("offered,run_mean,run_peak,wait_mean,wait_peak,kv_peak,preempt_delta,tok_s\n")
        for r in rows:
            f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"[written] {save}")


if __name__ == "__main__":
    main()
