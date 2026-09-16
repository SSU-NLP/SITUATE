#!/usr/bin/env python3
r"""인과 사슬(causalchain) 축 — 샤드 단위 추출기. VESSL Job 안에서 돈다.

🔒 **프롬프트는 `causal_chain_redundancy_diag.py` 에서 import 한다** — dev-3 사슬 라벨(F12 의 근거)을
만든 것과 **같은 문구**임을 코드로 보장한다. 여기서 문구를 다시 쓰지 않는다.

동결 5축 경로(`llm_s5_states.py`)는 **건드리지 않는다.** 자체 Stage-1(`chain_lists.json`)을 쓰므로
동결 `story_lists.json` 캐시와도 겹치지 않는다.

산출: {out}/{doc}/{chain_lists.json, causalchain_states.jsonl}
      causalchain_states.jsonl = {"item_id": int, "causal_chain": str, "model":…, "prompt_sha256":…}

    python chain_shard.py <shard.jsonl ...> <out_dir> --model qwen36-35b --base-url http://127.0.0.1:8001/v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 🔒 프롬프트 원본 = `causal_chain_redundancy_diag.py`(dev-3 사슬 라벨을 만든 파일).
#    모듈 import 는 numpy/dotenv 등 전이 의존을 끌고 오므로, **필요한 정의만 AST 로 뽑아 실행**한다.
#    문구가 그 파일에서 온다는 보장은 유지되고 Job 이미지 의존은 0 이다.
def _load_prompts():
    import ast
    src = (REPO / "scripts/eacl_phase0/causal_chain_redundancy_diag.py").read_text()
    tree = ast.parse(src)
    want = {"CHAIN_STAGE1", "CHAIN_SCHEMA", "stage2_chain_prompt"}
    keep = [n for n in tree.body
            if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in want)]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<chain_prompts>", "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise SystemExit(f"🔴 프롬프트 정의를 못 찾았다: {missing}")
    return ns["CHAIN_STAGE1"], ns["CHAIN_SCHEMA"], ns["stage2_chain_prompt"]


CHAIN_STAGE1, CHAIN_SCHEMA, stage2_chain_prompt = _load_prompts()

COST = {"usd": 0.0, "in": 0, "out": 0, "calls": 0}

S1_SYS = ("You enumerate causal chains present in a text as normalized identifiers. "
          "You never discuss structure, order, or boundaries.")
SEED = 20260711 + 5          # 동결 규약: 20260711 + 축 인덱스. 5축 뒤 = 실험축 첫 번째
PROMPT_SHA = hashlib.sha256((CHAIN_STAGE1 + S1_SYS).encode()).hexdigest()


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def docs_of(shards: list[Path]) -> dict[str, list[str]]:
    """샤드 → {doc_id: 문장 리스트}. 동결 규약과 동일(seq = [rows[0].previous, *currents])."""
    by: dict[str, list[dict]] = {}
    for sh in shards:
        for r in read_jsonl(sh):
            by.setdefault(r["doc_id"], []).append(r)
    out = {}
    for doc, rows in by.items():
        rows.sort(key=lambda r: int(r["transition"]))
        out[doc] = [rows[0]["previous"], *[r["current"] for r in rows]]
    return out


def chat(client, model, sys_msg, user, schema=None, max_tokens=3000, timeout=900, extra=None):
    """추론형 모델(gpt-oss 등)은 reasoning 토큰이 예산을 먹어 content 가 None 이거나 잘린다.
    ⇒ 실패 시 예산을 2배로 늘려 두 번 더 시도한다."""
    last = None
    for mult in (1, 2, 4):
        kw = dict(model=model, temperature=0, max_tokens=int(max_tokens * mult), timeout=timeout,
                  messages=[{"role": "system", "content": sys_msg},
                            {"role": "user", "content": user}])
        if schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": schema}
        if extra:
            kw["extra_body"] = extra
        r = client.chat.completions.create(**kw)
        u = getattr(r, "usage", None)
        if u:                                   # 프록시가 usage.cost 를 준다 — 실비 집계
            COST["calls"] += 1
            COST["in"] += getattr(u, "prompt_tokens", 0) or 0
            COST["out"] += getattr(u, "completion_tokens", 0) or 0
            COST["usd"] += float(getattr(u, "cost", 0) or 0)
        txt = r.choices[0].message.content
        if txt:
            try:
                return json.loads(txt)
            except json.JSONDecodeError as e:      # 잘림 — 예산 늘려 재시도
                last = e
        else:
            last = ValueError("content=None (reasoning 토큰이 예산 소진)")
    raise last


def run_doc(client, model, doc: str, sents: list[str], out: Path, batch: int,
            extra: dict | None, timeout: int, a_s1: int = 8000,
            a_s2: int = 3000) -> tuple[str, int, str]:
    d = out / doc
    d.mkdir(parents=True, exist_ok=True)
    lp, sp = d / "chain_lists.json", d / "causalchain_states.jsonl"

    # ── Stage 1 (문서당 1회, 캐시) ──────────────────────────────────────
    if lp.exists():
        chains = json.loads(lp.read_text())["chains"]
    else:
        r = chat(client, model, S1_SYS, CHAIN_STAGE1.format(story="\n".join(sents)),
                 schema={"name": "causal_chains", "strict": True, "schema": CHAIN_SCHEMA},
                 max_tokens=a_s1, timeout=timeout, extra=extra)
        chains = [str(x) for x in r.get("chains", [])]
        lp.write_text(json.dumps({"doc_id": doc, "chains": chains, "model": model,
                                  "prompt_sha256": PROMPT_SHA}, ensure_ascii=False))

    # ── Stage 2 (문장 단독, 배치, resume) ───────────────────────────────
    done = {int(json.loads(l)["item_id"]) for l in sp.read_text().splitlines()
            if l.strip()} if sp.exists() else set()
    todo = [(i, s) for i, s in enumerate(sents) if i not in done]
    if not todo:
        return doc, 0, "cached"
    rng = random.Random(SEED)
    rng.shuffle(todo)
    enum = {"type": "string", "enum": [*chains, "none"]} if chains else {"type": "string"}
    n_new = 0
    with sp.open("a") as f:
        for k in range(0, len(todo), batch):
            items = todo[k:k + batch]
            schema = {"name": "chain_items", "strict": True, "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {"items": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"idx": {"type": "integer"}, "causal_chain": enum},
                    "required": ["idx", "causal_chain"]}}},
                "required": ["items"]}}
            r = chat(client, model, "You label sentences independently.",
                     stage2_chain_prompt(chains, items), schema=schema,
                     max_tokens=a_s2, timeout=timeout, extra=extra)
            got = {int(x["idx"]): x["causal_chain"] for x in r.get("items", [])}
            for i, _s in items:
                if i in got:
                    f.write(json.dumps({"item_id": i, "axis": "causalchain",
                                        "causal_chain": got[i], "model": model,
                                        "prompt_sha256": PROMPT_SHA}, ensure_ascii=False) + "\n")
                    n_new += 1
            f.flush()
    return doc, n_new, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-tokens-s1", type=int, default=8000)
    ap.add_argument("--max-tokens-s2", type=int, default=3000)
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--reasoning-effort", default=None,
                    help="추론형 모델의 추론량. gpt-oss 는 minimal 로 추론 토큰이 333->30 으로 준다. "
                         "🔴 조건이 달라지므로 산출 네임스페이스에 반드시 표기한다.")
    ap.add_argument("--proxy", action="store_true",
                    help="Crts 프록시(OpenAI 호환)로 호출 — .env 의 키·CF-Access 헤더를 쓴다. "
                         "모델명은 `openrouter/...` 형식.")
    a = ap.parse_args()
    from openai import OpenAI
    if a.proxy:
        import os
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                        base_url=a.base_url if a.base_url != "http://127.0.0.1:8001/v1"
                        else os.environ["OPENAI_BASE_URL"],
                        timeout=a.timeout, max_retries=2,
                        default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                                         "CF-Access-Client-Secret":
                                             os.environ["CF_ACCESS_CLIENT_SECRET"]})
    else:
        client = OpenAI(api_key="dummy", base_url=a.base_url, timeout=a.timeout, max_retries=1)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_reasoning else None
    if a.reasoning_effort:
        extra = {**(extra or {}), "reasoning": {"effort": a.reasoning_effort}}
    a.out.mkdir(parents=True, exist_ok=True)

    D = docs_of(a.shards)
    print(f"[chain] 문서 {len(D):,} · 문장 {sum(len(v) for v in D.values()):,} · "
          f"workers={a.workers} batch={a.batch}", flush=True)
    t0 = time.time()
    ok = fail = 0

    def work(item):
        doc, sents = item
        for attempt in range(a.retries):
            try:
                return run_doc(client, a.model, doc, sents, a.out, a.batch, extra, a.timeout,
                               a.max_tokens_s1, a.max_tokens_s2)
            except Exception as e:                                    # noqa: BLE001
                if attempt == a.retries - 1:
                    return doc, 0, f"FAIL {type(e).__name__}: {str(e)[:120]}"
                time.sleep(3 * (attempt + 1))
        return doc, 0, "FAIL"

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (doc, n, st) in enumerate(ex.map(work, D.items()), 1):
            if st.startswith("FAIL"):
                fail += 1
                print(f"[chain] 🔴 {doc}: {st}", flush=True)
            else:
                ok += 1
            if i % 20 == 0:
                print(f"[chain] {i}/{len(D)} · ok {ok} fail {fail} · {time.time()-t0:.0f}s · "
                      f"${COST['usd']:.2f} ({COST['calls']:,}콜 in {COST['in']:,} out {COST['out']:,})",
                      flush=True)
    print(f"[chain] 완료 ok {ok} fail {fail} · {time.time()-t0:.0f}s · "
          f"실비 ${COST['usd']:.2f} ({COST['calls']:,}콜)", flush=True)
    sys.exit(1 if fail and not ok else 0)


if __name__ == "__main__":
    main()
