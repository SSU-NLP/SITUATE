"""Stage-1(story_lists.json)만 미리 생성 — 분산 라벨링 전 캐시 예열.

왜: Stage-1 캐시는 축 간 공유 파일이라, 같은 문서의 여러 축이 다른 머신에 흩어지면
경합한다. 미리 뽑아 두면 축 순서 제약도 경합도 사라져 축 5개를 완전 병렬화할 수 있다.
동결 `stage1()` 을 그대로 호출하며 파일 형식·필드도 llm_s5_states 와 동일하다.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eacl_phase0"))
import llm_s5_states as M
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("master", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--ports", default="8001,8002,8003,8004")
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--stage1-max-tokens", type=int, default=8000)
ap.add_argument("--timeout", type=float, default=900)
a = ap.parse_args()

fixes, vtag = M.V31_FIXES, M.VERSIONS["v3.1"]
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
ports = [int(x) for x in a.ports.split(",")]
clients = [OpenAI(api_key="dummy", base_url=f"http://localhost:{p}/v1",
                  timeout=a.timeout, max_retries=0) for p in ports]

by_doc: dict[str, list] = {}
for r in M.read(a.master):
    by_doc.setdefault(r["doc_id"], []).append(r)
docs = sorted(by_doc)
lock, done, fail = threading.Lock(), [0], []

def job(k: int) -> None:
    d = docs[k]
    ddir = a.out_dir / d
    lp = ddir / "story_lists.json"
    if lp.exists():
        with lock: done[0] += 1
        return
    rows = sorted(by_doc[d], key=lambda x: int(x["transition"]))
    texts = M.sequence(rows)
    try:
        lists, prompt = M.stage1(clients[k % len(clients)], "qwen36-35b", "\n".join(texts),
                                 a.stage1_max_tokens, EXTRA, "v3.1", fixes)
    except Exception as e:
        with lock: fail.append((d, type(e).__name__))
        return
    ddir.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({"doc_id": d, "lists": lists, "model": "qwen36-35b",
                              "prompt_version": vtag,
                              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
                             ensure_ascii=False, indent=2))
    with lock:
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  {done[0]}/{len(docs)} (실패 {len(fail)})", flush=True)

print(f"Stage-1 예열: {len(docs)}편 · 복제 {len(ports)} · 동시성 {a.workers}", flush=True)
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    list(ex.map(job, range(len(docs))))
print(f"[완료] 성공 {done[0]} · 실패 {len(fail)}")
for d, e in fail[:20]:
    print(f"  FAIL {d} {e}")
json.dump([d for d, _ in fail], open(a.out_dir / "_stage1_failed.json", "w"))
