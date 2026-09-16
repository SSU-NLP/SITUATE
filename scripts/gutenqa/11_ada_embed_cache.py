#!/usr/bin/env python3
"""ada-002 임베딩 캐시 — 논문과 **같은 인코더**로 재평가하기 위한 것 (D-B47).

## 왜 필요한가
논문 Table 1 은 `text-embedding-ada-002` 로 산출됐다(각주 3). 우리 하네스는 로컬
`multi-qa-mpnet-base-dot-v1` 을 써서 **5종 베이스라인이 전부 5~6점 낮게** 나왔다(순위는 일치).
같은 인코더로 돌리면 ① 논문 수치를 실제로 재현하는지 검증되고 ② M5 와 LumberChunker 의 격차가
인코더 탓인지 방법 탓인지 분리된다.

🔴 두 인코더는 **유사도 정의가 다르다**:
- ada-002 = OpenAI 가 단위정규화해서 준다(실측 노름 1.000000) ⇒ 레포의 `np.dot` = 코사인
- multi-qa-mpnet-dot-v1 = 정규화 없음 ⇒ 같은 `np.dot` 에 벡터 크기 항이 섞인다
  (실측 노름 5.62~7.12, 노름–길이 상관 −0.285)
따라서 ada 쪽이 **논문 조건 그 자체**다. 우리 mpnet 결과를 버리는 게 아니라 나란히 보고한다.

## 접속 경로
`.env` 의 `OPENAI_BASE_URL` 은 Cloudflare Access 뒤의 LiteLLM→OpenRouter 프록시다.
모델명은 반드시 `openrouter/openai/text-embedding-ada-002` — 앞의 두 형태는 404 다.
Cloudflare 가 `Python-urllib` UA 를 403 으로 막으므로 `requests` + UA 를 쓴다(실측).

## 캐시
sha1(텍스트) → float32[1536] 을 sqlite 에 저장한다. `_b1`/`cd` 셀들은 원본 셀과 청크를
대량 공유하므로 내용주소 캐시가 **중복 결제를 막는다**. 재실행은 공짜다.

    uv run python scripts/gutenqa/11_ada_embed_cache.py --cells lumberchunker recursive ...
    uv run python scripts/gutenqa/11_ada_embed_cache.py --questions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
CACHE_DB = BASE / "ada_cache.sqlite"

MODEL = "openrouter/openai/text-embedding-ada-002"
DIM = 1536
MAX_TOK = 8191               # ada-002 입력 상한
BATCH_ITEMS = 128            # 요청당 최대 입력 개수
BATCH_TOKENS = 120_000       # 요청당 최대 총 토큰
PRICE_PER_1M = 0.10          # ada-002 공시가 ($/1M tokens) — 프록시 마크업은 별개


def sha(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


class Cache:
    """sha1 → float32[1536]. 스레드 안전(쓰기는 락으로 직렬화)."""

    def __init__(self, path: Path):
        self.lock = threading.Lock()
        self.con = sqlite3.connect(str(path), check_same_thread=False, timeout=120.0)
        self.con.execute("PRAGMA journal_mode=WAL")
        # 🔴 busy_timeout 없이 두 프로세스를 동시에 띄웠다가 `database is locked` 로 잡이 죽었다
        #    (실측 2026-07-30). 결제한 벡터가 커밋 직전에 날아간다. 프로세스는 하나만 띄우고,
        #    그래도 겹칠 때를 대비해 대기시간을 길게 잡는다.
        self.con.execute("PRAGMA busy_timeout=120000")
        self.con.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")
        self.con.commit()

    def have(self, hs: list[str]) -> set[str]:
        out: set[str] = set()
        with self.lock:
            for i in range(0, len(hs), 900):
                chunk = hs[i:i + 900]
                q = ",".join("?" * len(chunk))
                out |= {r[0] for r in
                        self.con.execute(f"SELECT h FROM emb WHERE h IN ({q})", chunk)}
        return out

    def put(self, items: list[tuple[str, np.ndarray]]) -> None:
        rows = [(h, v.astype(np.float32).tobytes()) for h, v in items]
        for attempt in range(8):                     # 결제한 벡터를 락 때문에 버리지 않는다
            try:
                with self.lock:
                    self.con.executemany("INSERT OR REPLACE INTO emb VALUES (?,?)", rows)
                    self.con.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) and "busy" not in str(e):
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("sqlite 잠금 8회 재시도 실패 — 임베딩 프로세스가 둘 이상인지 확인할 것")

    def get_many(self, hs: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self.lock:
            for i in range(0, len(hs), 900):
                chunk = hs[i:i + 900]
                q = ",".join("?" * len(chunk))
                for h, v in self.con.execute(f"SELECT h,v FROM emb WHERE h IN ({q})", chunk):
                    out[h] = np.frombuffer(v, dtype=np.float32)
        return out


def load_env() -> dict:
    env = {}
    for line in (REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class Client:
    def __init__(self, env: dict):
        import requests
        self.s = requests.Session()
        self.url = env["OPENAI_BASE_URL"].rstrip("/") + "/embeddings"
        self.h = {
            "Authorization": f"Bearer {env['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
            "CF-Access-Client-Id": env["CF_ACCESS_CLIENT_ID"],
            "CF-Access-Client-Secret": env["CF_ACCESS_CLIENT_SECRET"],
            "User-Agent": "curl/8.0",       # 🔴 Cloudflare 가 Python UA 를 403 으로 막는다
        }
        self.tokens = 0
        self.lock = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        last = None
        for attempt in range(7):
            try:
                r = self.s.post(self.url, headers=self.h, timeout=180,
                                json={"model": MODEL, "input": texts})
                if r.status_code == 200:
                    d = r.json()
                    rows = sorted(d["data"], key=lambda x: x["index"])
                    with self.lock:
                        self.tokens += d.get("usage", {}).get("total_tokens", 0)
                    return np.asarray([x["embedding"] for x in rows], dtype=np.float32)
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in (408, 429, 500, 502, 503, 504, 529):
                    raise RuntimeError(last)
            except Exception as e:                      # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"임베딩 실패(7회 재시도): {last}")


def batches(texts: list[str], toks: list[int]):
    cur: list[str] = []
    cur_t = 0
    for t, n in zip(texts, toks):
        if cur and (len(cur) >= BATCH_ITEMS or cur_t + n > BATCH_TOKENS):
            yield cur
            cur, cur_t = [], 0
        cur.append(t)
        cur_t += n
    if cur:
        yield cur


def ensure_embedded(texts: list[str], cache: Cache, cli: Client, enc, workers: int,
                    label: str) -> None:
    """캐시에 없는 텍스트만 임베딩해 채운다."""
    uniq: dict[str, str] = {}
    for t in texts:
        uniq.setdefault(sha(t), t)
    have = cache.have(list(uniq))
    todo = [(h, t) for h, t in uniq.items() if h not in have]
    if not todo:
        print(f"  [{label}] 전부 캐시됨 ({len(uniq):,} 고유) — 결제 0")
        return
    tl = [enc.encode(t) for _, t in todo]
    trunc = sum(1 for x in tl if len(x) > MAX_TOK)
    txts = [enc.decode(x[:MAX_TOK]) if len(x) > MAX_TOK else t
            for (_, t), x in zip(todo, tl)]
    toks = [min(len(x), MAX_TOK) for x in tl]
    hs = [h for h, _ in todo]
    total_tok = sum(toks)
    print(f"  [{label}] 고유 {len(uniq):,} · 캐시적중 {len(have):,} · 신규 {len(todo):,} "
          f"({total_tok:,} 토큰 ≈ ${total_tok*PRICE_PER_1M/1e6:.2f})"
          + (f" · 8191토큰 초과 절단 {trunc}" if trunc else ""))

    # 8191 토큰 초과분은 절단되어 해시가 바뀐다 → **원래 해시**로 되돌려 저장해야
    # 다음 조회에서 적중한다. 절단 안 된 텍스트는 sha(t) == 원래 해시라 항등이다.
    trunc_map = {sha(t): h for h, t in zip(hs, txts)}
    done = [0]
    dlock = threading.Lock()

    def work(bt: list[str]):
        V = cli.embed(bt)
        cache.put([(trunc_map[sha(t)], v) for t, v in zip(bt, V)])
        with dlock:
            done[0] += len(bt)
            if done[0] % 20000 < len(bt):
                print(f"    {done[0]:,}/{len(todo):,}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, batches(txts, toks)))

    stored = cache.have(hs)          # 🔴 루프 밖에서 한 번만 — 안에서 부르면 O(n²) 로 멈춘다(실측)
    miss = [h for h in hs if h not in stored]
    if miss:
        print(f"  🔴 [{label}] 저장 누락 {len(miss)}개 — 재실행하면 그만큼만 다시 결제한다")


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--questions", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="비용만 계산하고 결제하지 않는다")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    cache = Cache(CACHE_DB)
    cli = Client(load_env())

    if a.questions:
        qs = [q["question"] for q in load_jsonl(DATA / "questions.jsonl")]
        if a.dry_run:
            n = sum(len(enc.encode(t)) for t in qs)
            print(f"[dry] 질문 {len(qs):,} = {n:,}토큰 ≈ ${n*PRICE_PER_1M/1e6:.4f}")
        else:
            ensure_embedded(qs, cache, cli, enc, a.workers, "questions")

    for cell in a.cells:
        p = DATA / f"chunks/{cell}.jsonl"
        if not p.exists():
            print(f"  skip {cell} (없음)")
            continue
        texts = [c for r in load_jsonl(p) for c in r["chunks"]]
        if a.dry_run:
            uniq = {sha(t): t for t in texts}
            have = cache.have(list(uniq))
            new = [t for h, t in uniq.items() if h not in have]
            n = sum(len(enc.encode(t)) for t in new)
            print(f"[dry] {cell:32s} 청크 {len(texts):,} · 고유 {len(uniq):,} · "
                  f"신규 {len(new):,} = {n:,}토큰 ≈ ${n*PRICE_PER_1M/1e6:.2f}")
        else:
            ensure_embedded(texts, cache, cli, enc, a.workers, cell)

    if not a.dry_run:
        print(f"\n🔒 이번 실행 결제 토큰 = {cli.tokens:,} ≈ ${cli.tokens*PRICE_PER_1M/1e6:.2f}")
    print(f"캐시 -> {CACHE_DB}")


if __name__ == "__main__":
    main()
