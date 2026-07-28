#!/usr/bin/env python3
"""Blinded one-at-a-time web annotator for Phase-0 packs.

Serves a single-axis annotation UI. Each 0/1 click is written straight to the
axis jsonl on disk, so `merge_blinded_packs.py` can read the result directly --
no download/re-upload. Blinding is enforced: only the requested axis is served,
and `boundary` is refused until all five slot files are complete.

Usage:
    uv run python scripts/eacl_phase0/annotate_server.py [axis] \
        [--dir outputs/eacl_phase0/narrative10_axis_packs] [--port 8042]

`axis` defaults to the first incomplete slot in the recommended order.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nltk.tokenize.treebank import TreebankWordDetokenizer

SLOTS = ("protagonist", "place", "time", "causality", "goal")
LABELS = (*SLOTS, "boundary")

AXIS_INFO = {
    "protagonist": ("주인공(행위자) 교체?",
                    "중심 행위자/인물 집합이 교체되면 1. 새 인물 언급만으로는 1이 아니다."),
    "place": ("장소(주 무대) 교체?",
              "서사의 주 무대가 교체되면 1. 장소 단어가 새로 나오는 것만으로는 1이 아니다."),
    "time": ("시간 불연속 이동?",
             "서사 기준 시간이 불연속적으로 이동하면 1. 동일 사건 안의 정상 진행은 0."),
    "causality": ("인과 사슬 단절?",
                  "직전 인과 사슬과 끊기고 새 사건 사슬이 시작하면 1."),
    "goal": ("목표 교체?",
             "중심 인물/화자의 활성 목표가 교체되면 1."),
    "boundary": ("새 서사 경계?",
                 "current가 새 완결 서사(새 memory-orb)를 시작하면 1. 같은 이야기의 장면·설명·예시는 0."),
}

_LOCK = threading.Lock()
_DIR: Path = Path(".")
_SILVER_DIR: Path | None = None
_DETOK = TreebankWordDetokenizer()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def axis_counts() -> dict[str, dict]:
    out = {}
    for label in LABELS:
        path = _DIR / f"{label}.jsonl"
        if not path.exists():
            out[label] = {"filled": 0, "total": 0}
            continue
        rows = read_jsonl(path)
        filled = sum(1 for r in rows if r.get(label) in (0, 1, False, True))
        out[label] = {"filled": filled, "total": len(rows)}
    return out


def available_labels() -> tuple[str, ...]:
    """Only labels physically present in this blinded pack may be served."""
    return tuple(label for label in LABELS if (_DIR / f"{label}.jsonl").is_file())


def display_english(text: str) -> str:
    """Undo WritingPrompts' Penn-Treebank spacing for display only."""
    out = _DETOK.detokenize(str(text).split()).replace("``", '"').replace("''", '"')
    return re.sub(r"\s+([.,!?;:])", r"\1", out)


def slots_complete() -> bool:
    c = axis_counts()
    return all(c[s]["total"] > 0 and c[s]["filled"] == c[s]["total"] for s in SLOTS)


def first_incomplete() -> str:
    c = axis_counts()
    labels = available_labels()
    for label in labels:
        if c[label]["filled"] < c[label]["total"] or c[label]["total"] == 0:
            return label
    return labels[-1] if labels else ""


def public_rows(axis: str) -> list[dict]:
    if axis not in available_labels():
        raise ValueError("axis is not available")
    rows = read_jsonl(_DIR / f"{axis}.jsonl")
    # Explicit allowlist: labels, boundary, IDM, story ids, and provenance cannot leak.
    silver = {}
    if _SILVER_DIR and (_SILVER_DIR / f"{axis}.jsonl").exists():
        silver = {int(x["transition"]): x for x in read_jsonl(_SILVER_DIR / f"{axis}.jsonl")}
    return [{"doc_id": r["doc_id"], "transition": r["transition"],
             "previous": display_english(r["previous"]),
             "previous_ko_reference": r.get("previous_ko_reference", ""),
             "current": display_english(r["current"]),
             "current_ko_reference": r.get("current_ko_reference", ""),
             "value": r.get(axis), "notes": r.get("notes", ""),
             "silver": silver.get(int(r["transition"]))} for r in rows]


def save_annotation(axis: str, doc_id: str, transition: int, value, notes: str) -> dict:
    if axis not in available_labels():
        raise ValueError("unknown or unavailable axis")
    if value not in (None, 0, 1, False, True):
        raise ValueError("value must be 0, 1, or null")
    path = _DIR / f"{axis}.jsonl"
    rows = read_jsonl(path)
    hit = False
    for r in rows:
        if str(r["doc_id"]) == str(doc_id) and int(r["transition"]) == int(transition):
            r[axis] = None if value is None else int(value)
            r["notes"] = str(notes)
            hit = True
            break
    if not hit:
        raise KeyError("transition not found")
    write_jsonl(path, rows)
    filled = sum(1 for r in rows if r.get(axis) in (0, 1, False, True))
    return {"ok": True, "filled": filled, "total": len(rows)}


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase-0 주석 · __AXIS__</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #090d14; --fg: #f5f7fb; --muted: #9aa6b5;
    --card: #111823; --line: #263244; --accent: #62a8ff;
    --prevfg: #d5dbe5; --ko: #ffd166; --yes: #3ddc97; --no: #ff6b6b;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg: #edf2f8; --fg: #111827; --muted: #64748b;
            --card: #ffffff; --line: #d8e0ea; --accent: #1769e0; --prevfg: #334155;
            --ko: #9a5c00; --yes: #087f5b; --no: #c92a2a; }
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", "Malgun Gothic", sans-serif;
         margin: 0; background: var(--bg); color: var(--fg);
         font-size: 18px; line-height: 1.65; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 12px 28px 170px; }
  header { position: sticky; top: 0; z-index: 5;
           background: color-mix(in srgb, var(--bg) 88%, transparent);
           backdrop-filter: blur(16px); padding: 12px 0 10px;
           border-bottom: 1px solid var(--line); margin-bottom: 6px; }
  .axis { font-size: 28px; font-weight: 850; letter-spacing: -.025em; }
  .axis .tag { opacity: .5; font-weight: 500; font-size: 16px; }
  .qdef { font-size: 16px; color: var(--muted); margin-top: 5px; line-height: 1.55; }
  .qdef b { color: var(--fg); }
  .bar { height: 6px; border-radius: 8px; background: var(--line); margin-top: 10px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--accent); width: 0%; transition: width .25s; }
  .meta { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted);
          margin-top: 8px; font-variant-numeric: tabular-nums; }
  .card { background: transparent; border: 0; padding: 10px 0; margin-top: 10px; box-shadow: none; }
  .lbl { font-size: 12px; letter-spacing: .1em; text-transform: uppercase;
         color: var(--muted); font-weight: 700; margin-bottom: 8px; }
  .compare { display: grid; grid-template-columns: minmax(0,1fr) 54px minmax(0,1fr);
             align-items: stretch; gap: 0; min-height: 430px; }
  .sentence { display: flex; flex-direction: column; justify-content: center;
              padding: 34px; border: 1px solid var(--line); border-radius: 22px;
              background: linear-gradient(145deg, color-mix(in srgb, var(--card) 96%, white 4%), var(--card));
              box-shadow: 0 14px 44px #00000030; }
  .sentence.current { border-color: color-mix(in srgb, var(--accent) 65%, var(--line));
                      box-shadow: 0 14px 48px #1769e026; }
  .prev { color: var(--prevfg); font-size: 22px; line-height: 1.72; font-weight: 520; }
  .arrow { display: flex; align-items: center; justify-content: center; color: var(--accent);
           margin: 0; font-size: 28px; font-weight: 900; }
  .curr { font-size: 24px; line-height: 1.72; font-weight: 680; color: var(--fg); }
  .ko { margin-top: 22px; padding: 18px 19px; border-radius: 13px;
        border: 1px solid color-mix(in srgb, var(--ko) 55%, transparent);
        background: color-mix(in srgb, var(--ko) 13%, transparent);
        color: var(--fg); font-size: 19px; line-height: 1.68; font-weight: 560; }
  .ko b { display: block; margin-bottom: 7px; color: var(--ko);
          font-size: 12px; letter-spacing: .12em; text-transform: uppercase; }
  @media (prefers-color-scheme: light) {
    .ko { background: #fff6d8; }
  }
  .silver { margin: 20px auto 0; max-width: 900px; padding: 20px 22px; border-radius: 16px;
            background: color-mix(in srgb, var(--accent) 11%, var(--card));
            border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--line)); }
  .silver-head { display:flex; align-items:center; gap:10px; font-weight:800; color:var(--accent); }
  .silver-v { font-size:20px; padding:2px 11px; border-radius:20px; background:var(--accent); color:#07111f; }
  .silver-body { margin-top:10px; font-size:17px; line-height:1.65; }
  .silver-ev { margin-top:8px; color:var(--muted); font-size:15px; }
  @media (max-width: 640px) {
    body { font-size: 16px; }
    .wrap { padding: 8px 10px 150px; }
    .card { padding: 6px 0; }
    .compare { grid-template-columns: 1fr; min-height: 0; }
    .sentence { padding: 16px; }
    .arrow { padding: 7px; transform: rotate(90deg); }
    .prev { font-size: 17px; } .curr { font-size: 19px; }
    .ko { font-size: 15px; }
  }
  .btns { position: sticky; bottom: 12px; z-index: 8; display: grid;
          grid-template-columns: 1fr 1fr; gap: 14px; margin: 22px auto 0; max-width: 760px;
          padding: 10px; border: 1px solid var(--line); border-radius: 18px;
          background: color-mix(in srgb, var(--bg) 90%, transparent); backdrop-filter: blur(18px); }
  button.ans { padding: 20px 18px; font-size: 22px; font-weight: 850; border-radius: 13px;
               border: 2px solid transparent; cursor: pointer; transition: .12s; letter-spacing: -.02em; }
  .b0 { background: color-mix(in srgb, var(--no) 18%, var(--card)); color: var(--fg); }
  .b1 { background: color-mix(in srgb, var(--yes) 20%, var(--card)); color: var(--fg); }
  .b0:hover { border-color: var(--no); } .b1:hover { border-color: var(--yes); }
  .b0.sel { border-color: var(--no); } .b1.sel { border-color: var(--yes); }
  @media (prefers-color-scheme: light) {
    .b0 { background: #e9edf2; color: #2a3240; } .b1 { background: #d6e5ff; color: #1451a3; }
    .b0:hover { background: #dfe4ea; } .b1:hover { background: #c6dbff; }
    .b0.sel { border-color: #7b8492; } .b1.sel { border-color: var(--accent); }
  }
  .notes { width: 100%; margin-top: 18px; padding: 12px 14px; border-radius: 10px;
           border: 1px solid var(--line); background: color-mix(in srgb, var(--bg) 60%, transparent);
           color: inherit; font-family: inherit; font-size: 15px; line-height: 1.6; resize: vertical; }
  .nav { display: flex; gap: 10px; margin-top: 18px; }
  .nav button { flex: 1; padding: 13px; border-radius: 10px; border: 1px solid var(--line);
                background: transparent; color: var(--fg); cursor: pointer; font-size: 14px; font-weight: 600; }
  .nav button:hover { background: color-mix(in srgb, var(--fg) 8%, transparent); }
  .nav button:disabled { opacity: .3; cursor: default; }
  .hint { text-align: center; font-size: 13px; color: var(--muted); margin-top: 18px; line-height: 1.8; }
  .done { text-align: center; padding: 44px 20px; }
  .done .big { font-size: 52px; } .done h2 { margin: 14px 0 6px; font-size: 24px; }
  .status { display: flex; flex-wrap: wrap; gap: 7px; justify-content: center; margin-top: 22px; font-size: 13px; }
  .chip { padding: 5px 12px; border-radius: 20px; background: color-mix(in srgb, var(--fg) 8%, transparent);
          color: var(--muted); font-variant-numeric: tabular-nums; }
  .chip.cur { outline: 2px solid var(--accent); color: var(--fg); font-weight: 700; }
  .chip.ok { background: #1f7a4522; color: #4cc47e; }
  code { background: color-mix(in srgb, var(--fg) 14%, transparent); padding: 2px 7px;
         border-radius: 6px; font-size: 13px; }
</style></head><body>
<div class="wrap">
  <header>
    <div class="axis">__TITLE__ <span style="opacity:.5;font-weight:400;font-size:15px">· __AXIS__</span></div>
    <div class="qdef">__DEF__ <b>애매하면 0.</b></div>
    <div class="bar"><i id="fill"></i></div>
    <div class="meta"><span id="pos"></span><span id="cnt"></span></div>
  </header>
  <div id="app"></div>
  <div class="status" id="status"></div>
</div>
<script>
const AXIS = "__AXIS__";
let rows = [], idx = 0;

async function load() {
  const r = await fetch(`/api/rows?axis=${AXIS}`);
  rows = await r.json();
  idx = rows.findIndex(x => x.value === null);
  if (idx < 0) idx = rows.length - 1;
  render(); loadStatus();
}
async function loadStatus() {
  const s = await (await fetch('/api/status')).json();
  const el = document.getElementById('status');
  el.innerHTML = s.order.map(a => {
    const c = s.counts[a]; const done = c.total>0 && c.filled===c.total;
    return `<span class="chip ${a===AXIS?'cur':''} ${done?'ok':''}">${a} ${c.filled}/${c.total}</span>`;
  }).join('');
}
function render() {
  const total = rows.length;
  const filled = rows.filter(x => x.value !== null).length;
  document.getElementById('fill').style.width = (100*filled/total) + '%';
  document.getElementById('pos').textContent = `전이 ${idx+1} / ${total}`;
  document.getElementById('cnt').textContent = `완료 ${filled} / ${total}`;
  const app = document.getElementById('app');
  if (idx >= total) { renderDone(filled, total); return; }
  const r = rows[idx];
  app.innerHTML = `
    <div class="card">
      <div class="compare"><div class="sentence">
        <div class="lbl">이전 문장 · Previous</div>
        <div class="prev">${esc(r.previous)}</div>
        ${r.previous_ko_reference ? `<div class="ko"><b>한국어 참고</b>${esc(r.previous_ko_reference)}</div>` : ''}
      </div><div class="arrow">→</div>
      <div class="sentence current">
        <div class="lbl">현재 문장 · Current</div>
        <div class="curr">${esc(r.current)}</div>
        ${r.current_ko_reference ? `<div class="ko"><b>한국어 참고</b>${esc(r.current_ko_reference)}</div>` : ''}
      </div></div>
      ${r.silver ? `<div class="silver">
        <div class="silver-head"><span>LLM 제안</span><span class="silver-v">${r.silver.suggested_value}</span>
          <span>확신 ${Math.round(100*r.silver.confidence)}%</span></div>
        <div class="silver-body">${esc(r.silver.explanation)}</div>
        <div class="silver-ev"><b>근거:</b> ${esc(r.silver.evidence)}</div>
      </div>` : ''}
      <div class="btns">
        <button class="ans b0 ${r.value===0?'sel':''}" onclick="answer(0)">0 · 아니오</button>
        <button class="ans b1 ${r.value===1?'sel':''}" onclick="answer(1)">1 · 예</button>
      </div>
      <textarea class="notes" id="notes" placeholder="메모 (선택)" oninput="dirty=true">${esc(r.notes||'')}</textarea>
      <div class="nav">
        <button onclick="go(idx-1)" ${idx===0?'disabled':''}>← 이전</button>
        <button onclick="answer(null)">미입력으로 원복</button>
        <button onclick="jumpNext()">다음 미입력 →</button>
        <button onclick="go(idx+1)" ${idx>=total-1?'disabled':''}>다음 →</button>
      </div>
    </div>
    <div class="hint">키보드: <code>0</code>/<code>1</code> 응답 · <code>←</code>/<code>→</code> 이동 · 응답 시 자동 저장·다음 이동</div>`;
}
function renderDone(filled, total) {
  document.getElementById('app').innerHTML = `
    <div class="card done"><div class="big">✓</div>
      <h2>${AXIS} 축 ${filled===total?'완료':'검토 끝'}</h2>
      <p style="opacity:.7">${filled} / ${total} 저장됨. 파일에 이미 기록되어 있습니다.</p>
      <div class="nav" style="max-width:320px;margin:16px auto 0">
        <button onclick="go(0)">처음부터 검토</button>
        <button onclick="jumpNext()">미입력으로</button>
      </div>
    </div>`;
}
let dirty = false;
async function answer(v) {
  await saveNotes();
  const r = rows[idx];
  r.value = v;
  await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({axis:AXIS, doc_id:r.doc_id, transition:r.transition, value:v, notes:r.notes||''})});
  loadStatus();
  go(idx+1);
}
async function saveNotes() {
  const t = document.getElementById('notes');
  if (!t || !dirty) return;
  const r = rows[idx]; r.notes = t.value; dirty = false;
  await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({axis:AXIS, doc_id:r.doc_id, transition:r.transition, value:r.value, notes:r.notes})});
}
async function go(i) { await saveNotes(); idx = Math.max(0, Math.min(rows.length, i)); render(); }
function jumpNext() { const i = rows.findIndex(x=>x.value===null); go(i<0?rows.length:i); }
function esc(s){ return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key === '0') answer(0);
  else if (e.key === '1') answer(1);
  else if (e.key === 'ArrowLeft') go(idx-1);
  else if (e.key === 'ArrowRight') go(idx+1);
});
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            axis = (q.get("axis", [None])[0]) or first_incomplete()
            if axis not in available_labels():
                return self._send(400, "unknown axis", "text/plain")
            if axis == "boundary" and not slots_complete():
                return self._send(403,
                    "<h2>boundary는 아직 열 수 없습니다</h2><p>다섯 슬롯 축을 먼저 끝내세요.</p>",
                    "text/html")
            title, definition = AXIS_INFO[axis]
            html = (PAGE.replace("__AXIS__", axis).replace("__TITLE__", title)
                    .replace("__DEF__", definition))
            return self._send(200, html, "text/html")
        if u.path == "/api/rows":
            axis = q.get("axis", [""])[0]
            if axis not in available_labels():
                return self._send(400, json.dumps({"error": "unknown axis"}))
            if axis == "boundary" and not slots_complete():
                return self._send(403, json.dumps({"error": "boundary locked"}))
            with _LOCK:
                out = public_rows(axis)
            return self._send(200, json.dumps(out, ensure_ascii=False))
        if u.path == "/api/status":
            return self._send(200, json.dumps({"order": list(available_labels()), "counts": axis_counts()},
                                               ensure_ascii=False))
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        axis = payload.get("axis")
        if axis not in available_labels():
            return self._send(400, json.dumps({"error": "unknown axis"}))
        did, tr = str(payload.get("doc_id")), int(payload.get("transition"))
        val, notes = payload.get("value"), payload.get("notes", "")
        try:
            with _LOCK:
                result = save_annotation(axis, did, tr, val, notes)
        except ValueError as e:
            return self._send(400, json.dumps({"error": str(e)}))
        except KeyError as e:
            return self._send(404, json.dumps({"error": str(e)}))
        return self._send(200, json.dumps(result))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("axis", nargs="?", default=None, help="축 이름 (기본: 첫 미완료 슬롯)")
    ap.add_argument("--dir", type=Path,
                    default=Path("outputs/eacl_phase0/narrative10_axis_packs"))
    ap.add_argument("--port", type=int, default=8042)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--silver-dir", type=Path,
                    default=Path("outputs/eacl_phase0/narrative10_llm_silver"))
    args = ap.parse_args()
    global _DIR, _SILVER_DIR
    _DIR = args.dir
    _SILVER_DIR = args.silver_dir
    global PAGE
    page_path = Path(__file__).with_name("annotate_page.html")
    PAGE = page_path.read_text(encoding="utf-8")
    if not _DIR.exists():
        raise SystemExit(f"블라인드 폴더가 없습니다: {_DIR}")
    axis = args.axis or first_incomplete()
    if axis not in available_labels():
        raise SystemExit(f"알 수 없거나 없는 축: {axis} (가능: {', '.join(available_labels())})")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/?axis={axis}"
    print(f"주석 서버 시작 → {url}")
    print(f"  폴더: {_DIR}")
    print("  브라우저에서 위 주소를 열어 0/1을 입력하세요. 각 입력은 즉시 파일에 저장됩니다.")
    print("  종료: Ctrl-C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료됨.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
