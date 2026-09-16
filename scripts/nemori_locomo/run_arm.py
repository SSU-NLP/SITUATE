#!/usr/bin/env python3
r"""LoCoMo × Nemori — 한 팔(arm)을 끝까지 돌리는 러너.

    # 발표판 그대로 (LLM 배치 분절기)
    .venv/bin/python scripts/nemori_locomo/run_arm.py --arm baseline

    # 분절기만 M5 로 교체
    .venv/bin/python scripts/nemori_locomo/run_arm.py --arm m5

두 팔의 **차이는 분절기 클래스 하나뿐**이다. 모델(gpt-4.1-mini · gemini-embedding-001),
프롬프트, 에피소드 서술 생성, 시맨틱 메모리, 병합, 검색(vector, top-k 10/20), QA 프롬프트,
채점(BLEU·F1·LLM judge) 전부 `external/nemori` 원본 코드를 **그대로 실행**한다.
원본 파일은 수정하지 않고 `runpy` 로 실행하며, 교체는 모듈 전역 이름 치환으로만 한다.

## 팔 격리
팔마다 `agent_id` 와 Qdrant 컬렉션 접두사를 따로 준다(nemori 의 멀티테넌트 기능).
같은 PostgreSQL/Qdrant 안에서 서로 안 섞인다. `--reset` 이면 그 팔의 데이터만 지운다.

## 주입 프로토콜 (`--ingest`)
| 값 | 동작 | 왜 |
|---|---|---|
| `single-flush` (기본) | `buffer_size_min` 을 크게 둬 자동 트리거를 끄고, 대화 하나를 **한 번에** flush | 두 팔이 **완전히 같은 입력**을 분절기에 준다. 재현 가능 |
| `published` | 저장소 기본값(`buffer_size_min=1`, `--batch-size 1`) 그대로 | 발표 프로토콜 재현용. 단 버퍼 적재 타이밍이 LLM 지연에 좌우돼 **run 마다 배치 구성이 달라진다** |

🔴 `single-flush` 는 **방법을 바꾸는 게 아니라 스케줄 경합을 없애는 것**이고 두 팔에 **동일하게**
적용된다. REPORT 에 공시한다.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemori_env import REPO, NEMORI, bootstrap, add_nemori_to_path  # noqa: E402

KST = timezone(timedelta(hours=9))
DATA = REPO / "benchmarks/locomo/locomo10.json"
PUBLISHED_CONFIG = NEMORI / "evaluation/locomo/config.json"

ARMS = {
    # 팔 이름 -> (설명, M5 분절기 사용 여부)
    "baseline": ("Nemori 발표판 — LLM 배치 분절기(gpt-4.1-mini, JSON 모드)", False),
    "m5": ("분절기만 교체 — M5-FactoredSlot(I2) + 적응임계 online_nf c=1.0", True),
}


class _Tee:
    """stdout/stderr 를 콘솔과 run.log 에 동시에 쓴다 (add.py 가 찍는 LLM 사용 요약을
    `collect_stats.py` 가 나중에 다시 읽어야 한다)."""

    def __init__(self, stream, path: Path) -> None:
        self._stream = stream
        self._file = path.open("a", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def now() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def arm_config(arm: str, ingest: str, out_dir: Path, label: str | None = None) -> Path:
    """발표 config.json 에서 출발해 팔 격리·주입 설정만 덮어쓴 config 를 만든다."""
    cfg = json.loads(PUBLISHED_CONFIG.read_text(encoding="utf-8"))
    published = dict(cfg)

    # 게이트웨이 경로 접두사 (같은 모델, 경로 표기만 다름)
    env = bootstrap()
    cfg["llm_model"] = env["llm_model"]
    cfg["embedding_model"] = env["embedding_model"]

    # 팔 격리 — label 로 조건을 네임스페이스에 박는다(조건이 조용히 섞이는 사고 방지)
    ns = label or arm
    cfg["agent_id"] = f"locomo_{ns}"
    cfg["qdrant_collection_prefix"] = f"nemori_locomo_{ns}"

    if ingest == "single-flush":
        # 자동 트리거 차단 -> flush() 한 번에 대화 전체가 분절기로 간다.
        # buffer_size_max 는 코드 어디서도 안 쓰이지만 `MemoryConfig.__post_init__` 이
        # max >= min 을 검사하므로 같이 올린다.
        cfg["buffer_size_min"] = 10 ** 9
        cfg["buffer_size_max"] = 10 ** 9

    path = out_dir / "config.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config_published_reference.json").write_text(
        json.dumps(published, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def reset_arm(cfg_path: Path) -> None:
    """이 팔의 PostgreSQL 행 + Qdrant 컬렉션만 지운다."""
    import asyncio
    import asyncpg
    from qdrant_client import QdrantClient

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    agent_id = cfg["agent_id"]
    prefix = cfg["qdrant_collection_prefix"]

    async def _pg() -> None:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            for table in ("semantic_memories", "episodes", "message_buffer"):
                try:
                    n = await conn.execute(f"DELETE FROM {table} WHERE agent_id = $1", agent_id)
                    log(f"  reset pg {table}: {n}")
                except asyncpg.exceptions.UndefinedTableError:
                    log(f"  reset pg {table}: (없음)")
        finally:
            await conn.close()

    asyncio.run(_pg())

    qc = QdrantClient(url=os.environ.get("QDRANT_URL", "127.0.0.1"),
                      port=int(os.environ.get("QDRANT_PORT", "6333")))
    for name in (f"{prefix}_episodes", f"{prefix}_semantic"):
        try:
            if qc.collection_exists(name):
                qc.delete_collection(name)
                log(f"  reset qdrant {name}")
        except Exception as e:
            log(f"  reset qdrant {name} 실패: {e}")
    qc.close()


def install_segmenter(arm: str, out_dir: Path, m5_c: float | None = None,
                      m5_ckpt: str | None = None) -> dict:
    """M5 팔이면 분절기를 교체하고, 어느 쪽이든 실제 사용 분절기를 기록한다."""
    import nemori.core.memory_system as ms

    if not ARMS[arm][1]:
        return {"segmenter": "nemori.llm.generators.segmenter.BatchSegmenter (원본)",
                "class": ms.BatchSegmenter.__name__}

    from hi_ontop.nemori_segmenter import M5SegmenterConfig, install

    resolved = install(M5SegmenterConfig(
        device=os.environ.get("M5_DEVICE", "cpu"),
        c=m5_c,
        checkpoint=m5_ckpt,          # None 이면 configs/m5_default.json 동결본
        trace_path=str(out_dir / "m5_segmentation_trace.jsonl"),
    ))
    import nemori.core.memory_system as ms2
    return {"segmenter": "hi_ontop.nemori_segmenter.M5BatchSegmenter",
            "class": ms2.BatchSegmenter.__name__, "m5": resolved}


def run_script(script: Path, argv: list[str], cwd: Path | None = None) -> None:
    """원본 스크립트를 수정 없이 `__main__` 으로 실행한다."""
    old_argv, old_cwd = sys.argv, Path.cwd()
    sys.argv = [str(script)] + argv
    if cwd:
        os.chdir(cwd)
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def stage_add(arm: str, cfg_path: Path, out_dir: Path, ingest: str, max_workers: int,
              m5_c: float | None = None, m5_ckpt: str | None = None) -> None:
    info = install_segmenter(arm, out_dir, m5_c, m5_ckpt)
    log(f"분절기: {info}")
    (out_dir / "segmenter_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    batch_size = "1" if ingest == "published" else "10000"
    t0 = time.time()
    run_script(NEMORI / "evaluation/locomo/add.py",
               ["--data", str(DATA), "--config", str(cfg_path),
                "--batch-size", batch_size, "--max-workers", str(max_workers)])
    log(f"add 완료 — {time.time() - t0:.0f}s")


def stage_search(arm: str, cfg_path: Path, out_dir: Path, max_workers: int) -> None:
    t0 = time.time()
    run_script(NEMORI / "evaluation/locomo/search.py",
               ["--data", str(DATA), "--config", str(cfg_path),
                "--output", str(out_dir / "results.json"),
                "--max-workers", str(max_workers)])
    log(f"search 완료 — {time.time() - t0:.0f}s")


def stage_evals(out_dir: Path, max_workers: int) -> None:
    t0 = time.time()
    run_script(NEMORI / "evaluation/locomo/evals.py",
               ["--input_file", str(out_dir / "results.json"),
                "--output_file", str(out_dir / "metrics.json"),
                "--max_workers", str(max_workers)])
    log(f"evals 완료 — {time.time() - t0:.0f}s")


def stage_scores(out_dir: Path) -> None:
    """원본 generate_scores.py 는 `locomo/metrics.json` 상대경로를 하드코딩했다 —
    그 레이아웃을 만들어 주고 cwd 를 옮겨 **원본 그대로** 실행한다."""
    shim = out_dir / "_gs" / "locomo"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "metrics.json").write_bytes((out_dir / "metrics.json").read_bytes())
    run_script(NEMORI / "evaluation/locomo/generate_scores.py", [], cwd=out_dir / "_gs")


def main() -> None:
    ap = argparse.ArgumentParser(description="LoCoMo × Nemori 한 팔 실행")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--name", default="2026-08-01_nemori_locomo_m5",
                    help="outputs/experiments/<name>/ 아래에 팔별 디렉토리 생성")
    ap.add_argument("--stages", default="add,search,evals,scores")
    ap.add_argument("--ingest", default="single-flush", choices=["single-flush", "published"])
    ap.add_argument("--reset", action="store_true", help="실행 전 이 팔의 저장소를 비운다")
    ap.add_argument("--add-workers", type=int, default=10, help="add.py 동시 대화 수 (원본 기본 10)")
    ap.add_argument("--search-workers", type=int, default=50, help="search.py 기본 50")
    ap.add_argument("--eval-workers", type=int, default=20, help="evals.py 기본 20")
    ap.add_argument("--limit", type=int, default=0,
                    help="대화 N개만 (스모크용). 0 = 전체 10대화")
    ap.add_argument("--limit-qa", type=int, default=0,
                    help="대화당 질문 N개만 (스모크용). 0 = 전체")
    ap.add_argument("--m5-c", type=float, default=None,
                    help="M5 적응임계 c. 미지정이면 동결값(1.0). 🔴 LoCoMo 점수를 보고 고르면 "
                         "그 순간 LoCoMo 는 test 가 아니라 dev 다 — REPORT 에 명시할 것")
    ap.add_argument("--m5-ckpt", default="",
                    help="M5 체크포인트 경로. 미지정이면 configs/m5_default.json 동결본. "
                         "🔴 다른 체크포인트를 쓰면 --label 도 반드시 다르게 준다")
    ap.add_argument("--label", default=None,
                    help="산출/저장소 네임스페이스 이름. 미지정이면 --arm 값. "
                         "조건이 다르면 반드시 다르게 준다(예: m5_c1p5)")
    args = ap.parse_args()

    label = args.label or args.arm
    out_dir = REPO / "outputs/experiments" / args.name / label
    out_dir.mkdir(parents=True, exist_ok=True)

    import logging
    sys.stdout = _Tee(sys.stdout, out_dir / "run.log")
    sys.stderr = _Tee(sys.stderr, out_dir / "run.log")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    global DATA
    if args.limit or args.limit_qa:
        data = json.loads(DATA.read_text(encoding="utf-8"))
        if args.limit:
            data = data[: args.limit]
        if args.limit_qa:
            for item in data:
                item["qa"] = item.get("qa", [])[: args.limit_qa]
        subset = out_dir / f"locomo_subset_c{args.limit}_q{args.limit_qa}.json"
        subset.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        DATA = subset
        log(f"🔴 스모크 모드 — 대화 {args.limit or '전체'} · 질문 {args.limit_qa or '전체'} ({subset})")

    env = bootstrap()
    add_nemori_to_path()
    log(f"팔 = {label} (arm={args.arm}) · {ARMS[args.arm][0]}"
        + (f" · c={args.m5_c}" if args.m5_c is not None else ""))
    log(f"게이트웨이 {env['base_url']} · llm={env['llm_model']} · emb={env['embedding_model']}")

    cfg_path = arm_config(args.arm, args.ingest, out_dir, label)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "arm": args.arm, "label": label, "m5_c": args.m5_c,
        "description": ARMS[args.arm][0], "ingest": args.ingest,
        "started_kst": now(), "env": env, "data": str(DATA),
        "stages": args.stages, "add_workers": args.add_workers,
        "search_workers": args.search_workers, "eval_workers": args.eval_workers,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.reset:
        log("저장소 리셋")
        reset_arm(cfg_path)

    if "add" in stages:
        stage_add(args.arm, cfg_path, out_dir, args.ingest, args.add_workers, args.m5_c,
                  args.m5_ckpt or None)
    if "search" in stages:
        stage_search(args.arm, cfg_path, out_dir, args.search_workers)
    if "evals" in stages:
        stage_evals(out_dir, args.eval_workers)
    if "scores" in stages:
        stage_scores(out_dir)

    log(f"팔 {label} 완료 — 산출 {out_dir}")


if __name__ == "__main__":
    main()
