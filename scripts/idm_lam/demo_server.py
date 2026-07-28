"""Real-time topic-segmentation demo — REAL mpnet encoder + online segmenter (delta_eff / IDM FDM_best)
with adaptive mu+c*sigma threshold. Type utterances live; see topics split in real time.

Run:  uv run python scripts/idm_lam/demo_server.py            # serves http://0.0.0.0:8000
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request


def split_sentences(text):
    """Split any input (sentence / paragraph / multi-line) into sentences (KO/EN).
    Split on line breaks and on sentence-ending punctuation . ? ! … 。 (keep the mark)."""
    out = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in re.split(r'(?<=[.!?…。])\s+', line):
            s = s.strip()
            # further split runs like "a. b. c" with no space already handled; drop empties
            if s:
                out.append(s)
    return out

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lam import LAM  # noqa: E402

ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RESULTS = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("[load] encoder ...", flush=True)
from sentence_transformers import SentenceTransformer

ENC = SentenceTransformer(ENCODER_NAME, device=DEV)
print("[load] LAM (IDM/FDM_best) ...", flush=True)
_ck = torch.load(RESULTS / "lam_corpus_f1.0.pt", weights_only=False)
_a = _ck["args"]
LAM_M = LAM(code_dim=_a["code_dim"], codebook_size=_a["codebook_size"], commitment_weight=_a["commitment"]).to(DEV)
LAM_M.load_state_dict(_ck["state"]); LAM_M.eval()
_CB = LAM_M.vq._codebook.embed.reshape(-1, LAM_M.code_dim)  # (K, d)

STATE = {"texts": [], "embs": []}  # single-session demo


def embed(text):
    z = ENC.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    return z


def fdm_best(z_prev, z_cur):
    with torch.no_grad():
        zt = torch.tensor(z_prev, device=DEV).unsqueeze(0)
        z1 = torch.tensor(z_cur, device=DEV).unsqueeze(0)
        errs = [((LAM_M.fdm(torch.cat([zt, _CB[k:k+1]], -1)) - z1) ** 2).sum(-1).item()
                for k in range(_CB.shape[0])]
    return float(min(errs))


def delta_eff(embs, t, m=2, rho=0.7, a=0.5):
    z = embs[t]; zp = embs[t - 1]
    d_prev = 1.0 - float(z @ zp)
    ctx = np.zeros_like(z); w = 0.0
    for k in range(1, m + 1):
        if t - k >= 0:
            wk = rho ** (k - 1); ctx += wk * embs[t - k]; w += wk
    if w > 0:
        ctx = ctx / (np.linalg.norm(ctx) + 1e-9)
        d_ctx = 1.0 - float(z @ ctx)
    else:
        d_ctx = d_prev
    return a * d_prev + (1 - a) * d_ctx


FALLBACK = {"cosine": 0.55}  # delta_eff* fallback (HANDOFF_06); idm has no fixed scale -> adaptive only


def run_online(embs, signal, c, warmup=3):
    """Online (strict causal): adaptive mu+c*sigma once >=warmup past values, else fixed fallback."""
    n = len(embs)
    sigs = [0.0] * n
    for t in range(1, n):
        sigs[t] = fdm_best(embs[t - 1], embs[t]) if signal == "idm" else delta_eff(embs, t)
    out = []
    seg = 0
    past = []
    for t in range(n):
        thr = None; bnd = False
        if t >= 1:
            s = sigs[t]
            if len(past) >= warmup:
                mu = float(np.mean(past)); sd = float(np.std(past))
                thr = mu + c * sd
            else:
                thr = FALLBACK.get(signal)          # None for idm during warmup
            if thr is not None and s > thr:
                bnd = True; seg += 1
            past.append(s)
        out.append({"sig": round(sigs[t], 4), "thr": (round(thr, 4) if thr is not None else None),
                    "boundary": bnd, "seg": seg})
    return out


app = Flask(__name__)


@app.route("/")
def index():
    return HTML


@app.route("/reset", methods=["POST"])
def reset():
    STATE["texts"] = []; STATE["embs"] = []
    return jsonify(ok=True)


@app.route("/add", methods=["POST"])
def add():
    d = request.get_json(force=True)
    text = (d.get("text") or "").strip()
    signal = d.get("signal", "cosine")
    c = float(d.get("c", 1.0))
    if text:
        for sent in split_sentences(text):      # split any input into sentence-level turns
            STATE["texts"].append(sent)
            STATE["embs"].append(embed(sent))
    seg = run_online(STATE["embs"], signal, c)
    turns = [{"text": t, **s} for t, s in zip(STATE["texts"], seg)]
    return jsonify(turns=turns, n_seg=(seg[-1]["seg"] + 1 if seg else 0))


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>실시간 화제분절 데모</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--fg:#e6e8ee;--mut:#8b93a7;--acc:#5b9dff;--line:#2a2e38}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,'Apple SD Gothic Neo',sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:860px;margin:0 auto;padding:20px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px;font-size:13px}
select,input[type=range]{accent-color:var(--acc)}
.inrow{display:flex;gap:8px;margin-bottom:16px}
textarea{flex:1;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:10px;font-size:15px;resize:none;height:44px}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:0 18px;font-size:14px;cursor:pointer}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--mut)}
.seg{border-left:3px solid var(--acc);background:var(--card);border-radius:0 10px 10px 0;margin:10px 0;padding:8px 12px}
.seghd{font-size:11px;color:var(--acc);letter-spacing:.04em;margin-bottom:6px}
.turn{display:flex;justify-content:space-between;gap:12px;padding:4px 0;border-top:1px dashed var(--line);font-size:14px}
.turn:first-of-type{border-top:0}
.tx{flex:1}.meta{color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.bnd{color:#ff8a5b;font-weight:600}
.stat{color:var(--mut);font-size:12px;margin:6px 0 14px}
.hue{--h:210}
</style></head><body><div class="wrap">
<h1>실시간 화제분절 데모 <span style="color:var(--mut);font-weight:400">· 실제 mpnet + 온라인 적응임계</span></h1>
<div class="sub">문장을 하나씩 입력(Enter)하면 스트리밍으로 화제 경계를 판정합니다. 미래 문장 안 봄(strict causal).</div>
<div class="controls">
 <label>신호:
  <select id="sig"><option value="cosine">δ_eff (cosine 문맥)</option><option value="idm">IDM FDM_best</option></select></label>
 <label>민감도 c: <input id="c" type="range" min="0.3" max="2.5" step="0.1" value="1.0"><span id="cv">1.0</span></label>
 <button class="ghost" id="rst">초기화</button>
 <span class="stat" id="stat"></span>
</div>
<div class="inrow">
 <textarea id="in" placeholder="문장 하나든, 문단 전체든 붙여넣고 Enter — 내부에서 문장 단위로 쪼개 경계 판정"></textarea>
 <button id="send">전송</button>
</div>
<div id="out"></div>
</div>
<script>
const HUES=[210,150,30,280,340,110,190,50,260,10];
const $=s=>document.querySelector(s);
$('#c').oninput=e=>{$('#cv').textContent=e.target.value; render(LAST);};
$('#sig').onchange=()=>refresh();
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json();}
let LAST={turns:[],n_seg:0};
async function send(){const t=$('#in').value.trim();if(!t)return;$('#in').value='';
 LAST=await post('/add',{text:t,signal:$('#sig').value,c:parseFloat($('#c').value)});render(LAST);}
async function refresh(){LAST=await post('/add',{text:'',signal:$('#sig').value,c:parseFloat($('#c').value)});render(LAST);}
async function reset(){await post('/reset');LAST={turns:[],n_seg:0};render(LAST);}
$('#send').onclick=send;$('#rst').onclick=reset;
$('#in').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
reset();  // 페이지 열 때 깨끗한 상태로 시작
function render(d){const turns=d.turns||[];let html='';let cur=-1;
 turns.forEach((t,i)=>{
  if(t.seg!==cur){if(cur!==-1)html+='</div>';cur=t.seg;const h=HUES[cur%HUES.length];
   html+=`<div class="seg" style="border-left-color:hsl(${h} 70% 60%)"><div class="seghd" style="color:hsl(${h} 70% 65%)">화제 ${cur+1}</div>`;}
  const mark=t.boundary?'<span class="bnd">▲경계</span> ':'';
  const meta=t.thr!=null?`s=${t.sig} / thr=${t.thr}`:`s=${t.sig}`;
  html+=`<div class="turn"><div class="tx">${mark}${escapeHtml(t.text)}</div><div class="meta">${meta}</div></div>`;
 });
 if(cur!==-1)html+='</div>';
 $('#out').innerHTML=html;
 $('#stat').textContent=turns.length?`${turns.length}턴 · ${d.n_seg}개 화제`:'';
}
function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
</script></body></html>"""


if __name__ == "__main__":
    print(f"[ready] http://0.0.0.0:8000  (device={DEV})", flush=True)
    app.run(host="0.0.0.0", port=8000, threaded=True)
