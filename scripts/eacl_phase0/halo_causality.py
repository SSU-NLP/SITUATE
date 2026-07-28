#!/usr/bin/env python3
"""causality Zacks-halo 추출 + 사전등록 3조건 측정.

Zacks et al.(2009) causal 코딩: "CURRENT 절의 주 활동이 PREVIOUS 절의 주 활동에 의해 직접
야기되지 않았을 때 = 경계(단절)". 두 절을 함께 보는 halo 판단(관계형). boundary/change/reset
언급 없이 인과관계만 물음. frozen Qwen3.6-35B-A3B no-thinking v2, temp=0.

모드:
  --probe  discordant_probe.jsonl → primary(장면연속·인과단절을 halo가 단절로 잡나) 판정
  --dev    repcorpus_masters_stage0/*.jsonl → transition별 halo 값 out 저장 + 발현율(sanity),
           동시에 현행 event_type 프록시 발현율(s5-dir causality_states)도 계산해 대조.
"""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # llm_s5_states import
from openai import OpenAI  # noqa: E402

c = OpenAI(api_key="dummy", base_url="http://localhost:8001/v1", timeout=180, max_retries=2)
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}

SYS = ("You judge the causal link between two consecutive narrative clauses. "
       "Given PREVIOUS and CURRENT, decide ONE thing: is the main activity or event in CURRENT a "
       "DIRECT causal consequence of the main activity in PREVIOUS — i.e., does PREVIOUS's action "
       "directly bring about, enable, or cause what happens in CURRENT? Judge direct causation only; "
       "mere time order or being in the same place is NOT causation. "
       'Return JSON {"caused_by_previous": true or false}.')


def halo_break(prev: str, cur: str) -> int:
    """1 = 단절(직전에 의해 야기되지 않음), 0 = 연속(직접 인과)."""
    r = c.chat.completions.create(model="qwen36-35b", temperature=0, max_tokens=200,
        response_format={"type": "json_object"}, extra_body=EXTRA,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": f"PREVIOUS: {prev}\nCURRENT: {cur}"}])
    obj = json.loads(r.choices[0].message.content)
    return int(not bool(obj.get("caused_by_previous", False)))


def run_probe(path: Path) -> None:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        vals = list(ex.map(lambda r: halo_break(r["prev"], r["cur"]), rows))
    brk = [(r, v) for r, v in zip(rows, vals) if r["causal_break"] == 1]
    con = [(r, v) for r, v in zip(rows, vals) if r["causal_break"] == 0]
    brk_fire = sum(v for _, v in brk) / len(brk)
    con_fire = sum(v for _, v in con) / len(con)
    print("=== discordant probe (장면 space/protag 전부 연속) ===")
    for r, v in zip(rows, vals):
        gt = "단절" if r["causal_break"] else "연속"
        hd = "단절" if v else "연속"
        mark = "✓" if (v == r["causal_break"]) else "✗"
        print(f"  {r['id']:<5} gt={gt} halo={hd} {mark}  | {r['note']}")
    print(f"\nwithin-scene 인과단절 → halo 단절율 = {brk_fire:.2f} (통과 ≥0.80)")
    print(f"within-scene 인과연속 → halo 오발현율 = {con_fire:.2f} (통과 ≤0.20)")
    primary = brk_fire >= 0.80 and con_fire <= 0.20
    print(f"[primary] {'PASS' if primary else 'FAIL'}")


def proxy_fire_rate(s5_dir: Path, docs: list[str]) -> float:
    """현행 event_type 프록시 발현율 (reconstruct와 동일 규칙)."""
    import llm_s5_states as s5
    num = den = 0
    for d in docs:
        p = s5_dir / d / "causality_states.jsonl"
        if not p.exists():
            continue
        st = {int(x["item_id"]): x for x in s5.read(p)}
        n = max(st) + 1
        for i in range(1, n):
            if i - 1 in st and i in st:
                num += s5.reset_causality(st[i - 1], st[i]); den += 1
    return num / den if den else float("nan")


def run_dev(master_dir: Path, s5_dir: Path, out: Path) -> None:
    masters = sorted(master_dir.glob("*.jsonl"))
    docs = [m.stem for m in masters]
    all_rows: list[dict] = []
    fire_num = fire_den = 0
    for m in masters:
        rows = [json.loads(l) for l in m.open() if l.strip()]
        rows.sort(key=lambda r: int(r["transition"]))
        with ThreadPoolExecutor(max_workers=12) as ex:
            vals = list(ex.map(lambda r: halo_break(r["previous"], r["current"]), rows))
        for r, v in zip(rows, vals):
            all_rows.append({"doc_id": r["doc_id"], "transition": int(r["transition"]),
                             "causality_halo": int(v)})
            fire_num += v; fire_den += 1
        dr = sum(vals) / len(vals)
        print(f"  {m.stem:<24} n={len(rows):>4}  halo 발현율={dr:.3f}", flush=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n")
    halo_rate = fire_num / fire_den
    proxy_rate = proxy_fire_rate(s5_dir, docs)
    print(f"\n=== sanity (dev {len(docs)}편, {fire_den} transitions) ===")
    print(f"halo 발현율   = {halo_rate:.3f}")
    print(f"현행 프록시   = {proxy_rate:.3f}")
    print(f"경계율(참고)  ~ 0.07")
    ok = 0.15 <= halo_rate <= 0.40 and halo_rate < proxy_rate
    print(f"[sanity] 대역[0.15,0.40] & <프록시 : {'PASS' if ok else 'FAIL'}")
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--master-dir", type=Path,
                    default=REPO / "outputs/eacl_phase0/repcorpus_masters_stage0")
    ap.add_argument("--s5-dir", type=Path,
                    default=REPO / "outputs/eacl_phase0/repcorpus_s5_q36_35b_v2_stage0")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/experiments/2026-07-25_causality_halo_zacks/halo_dev.jsonl")
    a = ap.parse_args()
    if a.probe:
        run_probe(a.probe)
    if a.dev:
        run_dev(a.master_dir, a.s5_dir, a.out)


if __name__ == "__main__":
    main()
