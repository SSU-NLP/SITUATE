"""typed-episodic-memory Go/No-Go 실험 공용 유틸 (gonogo_expreiment.md Tier 0).

- LLM 클라이언트: openrouter/* via Crts 프록시(CF Access 헤더 필수).
- 임베딩: 로컬 mpnet (multi-qa-mpnet-base-dot-v1, GPU) — 무과금.
- jsonl IO.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

QA_MODEL = "openrouter/openai/gpt-4o-mini"
JUDGE_MODEL = "openrouter/openai/gpt-4o-mini"    # judge도 mini(비용). gpt-4o로 올릴 수 있음.
LABEL_MODEL = "openrouter/openai/gpt-4o-mini"

# LoCoMo QA 카테고리 → 이름 (LoCoMo 논문 표준)
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                  4: "single-hop", 5: "adversarial"}


def get_client():
    import openai
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ["OPENAI_BASE_URL"]
    hdr = {"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
           "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]}
    return openai.OpenAI(api_key=key, base_url=base, default_headers=hdr)


def chat(client, model, messages, max_tokens=512, temperature=0.0, retries=4):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if a == retries - 1:
                print(f"  [chat err final] {str(e)[:120]}", flush=True)
                return ""
            time.sleep(2 * (a + 1))
    return ""


def chat_many(model, jobs, build_messages, max_tokens=512, temperature=0.0, workers=8):
    """jobs: list of items. build_messages(item)->messages. 반환: item순서대로 응답 list."""
    client = get_client()
    out = [""] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(chat, client, model, build_messages(it), max_tokens, temperature): i
                for i, it in enumerate(jobs)}
        done = 0
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(jobs)}", flush=True)
    return out


def parse_json_lenient(txt: str):
    """```json ... ``` / 앞뒤 잡음 제거 후 첫 {...} 파싱."""
    import re
    if not txt:
        return None
    t = txt.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ── 임베딩 (로컬 mpnet) ───────────────────────────────────────────────
_ENCODER = None


def get_encoder(model="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
    global _ENCODER
    if _ENCODER is None:
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from hi_ontop.embedding import make_encoder
        _ENCODER = make_encoder(model=model)
    return _ENCODER


def embed(texts, model="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
    import numpy as np
    enc = get_encoder(model)
    v = np.asarray(enc.encode(list(texts)), dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


# ── jsonl ─────────────────────────────────────────────────────────────
def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
