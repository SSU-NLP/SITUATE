"""Live, process-shared request limits with rate-limit backoff."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
import os
from pathlib import Path
import time


def category(model):
    return "judge" if "gpt-oss" in model else "actor"


def control_path():
    return Path(os.environ["STAGE_USAGE_LEDGER"]).parent / "concurrency.json"


async def start_ticket(control, key):
    """Space request starts across processes; concurrency alone cannot limit RPM."""
    rate_path = control.parent / f"{key}_request_rate.json"
    while True:
        settings = json.loads(control.read_text())[key]
        rps = float(settings.get("requests_per_second", 8 if key == "actor" else 32))
        with rate_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            next_start = json.loads(rate_path.read_text()).get("next_start", 0) if rate_path.exists() else 0
            ready = max(next_start, settings.get("cooldown_until", 0))
            wait = ready - time.time()
            if wait <= 0:
                rate_path.write_text(json.dumps({"next_start": time.time() + 1 / rps}))
                return
        await asyncio.sleep(min(1, wait))


@asynccontextmanager
async def request_slot(model):
    control = control_path()
    key = category(model)
    directory = control.parent / "request_slots" / key
    directory.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    while True:
        settings = json.loads(control.read_text()) if control.exists() else {}
        limit = int(settings.get(key, {}).get("limit", 64))
        for offset in range(limit):
            index = (os.getpid() + offset) % limit
            fd = os.open(directory / f"{index}.lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            try:
                await start_ticket(control, key)
                yield {"global_limit": limit, "queue_seconds": round(time.monotonic()-began, 3)}
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            return
        await asyncio.sleep(.1)


def record_error(model, status, attempt, duration, error_type=None):
    control = control_path()
    entry = {"time": time.time(), "model": model, "status": status,
             "attempt": attempt, "duration": duration, "error_type": error_type}
    with (control.parent / "transport_errors.jsonl").open("a") as log:
        log.write(json.dumps(entry) + "\n")
    if status not in (429, 502, 503, 504):
        return
    with control.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        settings = json.loads(control.read_text())
        section = settings[category(model)]
        now = time.time()
        recent = [value for value in section.get("recent_errors", []) if value > now - 10]
        recent.append(now)
        section["recent_errors"] = recent
        # A single transient rejection must not halve throughput for the entire run.
        if len(recent) >= 5 and now - section.get("last_throttle_at", 0) > 60:
            section["limit"] = max(8, int(section["limit"]) // 2)
            if status == 429:
                section["requests_per_second"] = max(1, section.get("requests_per_second", 8) / 2)
                section["cooldown_until"] = now + 60
            section["last_throttle_at"] = now
            section["last_throttle_status"] = status
        temporary = control.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings, indent=2) + "\n")
        temporary.replace(control)
