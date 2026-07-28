"""Voice + real-time topic segmentation demo.
Mic audio -> Qwen3-ASR-1.7B (transcription) -> sentence split -> online segmenter
(delta_eff / IDM FDM_best, adaptive mu+c*sigma). Also accepts typed text.

Run (isolated ASR venv):
  <asrenv>/bin/python scripts/idm_lam/demo_voice.py    # serves http://0.0.0.0:8000
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lam import LAM  # noqa: E402

ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RESULTS = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
ASR_ID = "Qwen/Qwen3-ASR-1.7B-hf"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("[load] mpnet encoder ...", flush=True)
from sentence_transformers import SentenceTransformer

ENC = SentenceTransformer(ENCODER_NAME, device=DEV)
print("[load] LAM (IDM/FDM_best) ...", flush=True)
_ck = torch.load(RESULTS / "lam_corpus_f1.0.pt", weights_only=False)
_a = _ck["args"]
LAM_M = LAM(code_dim=_a["code_dim"], codebook_size=_a["codebook_size"], commitment_weight=_a["commitment"]).to(DEV)
LAM_M.load_state_dict(_ck["state"]); LAM_M.eval()
_CB = LAM_M.vq._codebook.embed.reshape(-1, LAM_M.code_dim)
print("[load] Qwen3-ASR-1.7B ...", flush=True)
from transformers import Qwen3ASRForConditionalGeneration, AutoProcessor
import librosa
import subprocess

ASR_PROC = AutoProcessor.from_pretrained(ASR_ID)
ASR = Qwen3ASRForConditionalGeneration.from_pretrained(ASR_ID, dtype=torch.bfloat16,
                                                       attn_implementation="eager").to(DEV).eval()

STATE = {"texts": [], "embs": []}
FALLBACK = {"cosine": 0.55}


import tempfile, os


def decode_audio(raw: bytes) -> np.ndarray:
    """Decode ANY browser-encoded audio (webm/opus, ogg, wav, mp4...) to 16k mono f32.
    MediaRecorder produces a *streaming* webm (no duration/seek header) -> decode from a
    real temp file (not pipe:0, which can truncate/garble non-seekable webm)."""
    tf = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    try:
        tf.write(raw); tf.close()
        p = subprocess.run(
            ["ffmpeg", "-i", tf.name, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    finally:
        os.unlink(tf.name)
    pcm = np.frombuffer(p.stdout, dtype=np.float32).copy()
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    dur = len(pcm) / 16000.0
    rms = float(np.sqrt(np.mean(pcm ** 2))) if pcm.size else 0.0
    print(f"[audio] {len(raw)}B blob -> {dur:.2f}s pcm  peak={peak:.3f} rms={rms:.4f}", flush=True)
    if 0.02 < peak < 0.5:                      # amplify quiet speech only; skip near-silence/noise
        pcm *= min(0.9 / peak, 10.0)           # cap gain so noise floor isn't blown up
    return pcm


_REP = re.compile(r"(\b[\w가-힣]{1,4}\b)(?:\s*[,、]?\s*\1){3,}")


def collapse_repeats(txt: str) -> str:
    """Safety net: collapse pathological decode loops like '열, 열, 열, 열 …' to a single token.
    Only fires on 4+ identical consecutive short tokens (normal text never does this)."""
    return _REP.sub(r"\1", txt)


def transcribe(pcm: np.ndarray, sr: int, lang=None) -> str:
    if sr != 16000:
        pcm = librosa.resample(pcm.astype(np.float32), orig_sr=sr, target_sr=16000)
    if len(pcm) < 1600:  # <0.1s
        return ""
    inp = ASR_PROC.apply_transcription_request(audio=pcm.astype(np.float32), sampling_rate=16000,
                                               language=lang)

    def _cast(v):
        if not hasattr(v, "to"):
            return v
        if getattr(v, "is_floating_point", None) and v.is_floating_point():
            return v.to(DEV, dtype=ASR.dtype)      # match model dtype (bfloat16)
        return v.to(DEV)
    inp = {k: _cast(v) for k, v in inp.items()}
    with torch.no_grad():
        out = ASR.generate(**inp, max_new_tokens=440, do_sample=False)
    try:
        txt = ASR_PROC.batch_decode(out, skip_special_tokens=True)[0]
    except Exception:
        txt = ASR_PROC.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    if "<asr_text>" in txt:
        txt = txt.split("<asr_text>")[-1]
    return collapse_repeats(txt.strip())


def split_sentences(text):
    out = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in re.split(r'(?<=[.!?…。])\s+', line):
            s = s.strip()
            if s:
                out.append(s)
    return out or ([text.strip()] if text.strip() else [])


def embed(text):
    return ENC.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)


def fdm_best(zp, zc):
    with torch.no_grad():
        zt = torch.tensor(zp, device=DEV).unsqueeze(0); z1 = torch.tensor(zc, device=DEV).unsqueeze(0)
        errs = [((LAM_M.fdm(torch.cat([zt, _CB[k:k+1]], -1)) - z1) ** 2).sum(-1).item() for k in range(_CB.shape[0])]
    return float(min(errs))


def delta_eff(embs, t, m=2, rho=0.7, a=0.5):
    z = embs[t]; d_prev = 1.0 - float(z @ embs[t - 1])
    ctx = np.zeros_like(z); w = 0.0
    for k in range(1, m + 1):
        if t - k >= 0:
            wk = rho ** (k - 1); ctx += wk * embs[t - k]; w += wk
    d_ctx = 1.0 - float(z @ (ctx / (np.linalg.norm(ctx) + 1e-9))) if w > 0 else d_prev
    return a * d_prev + (1 - a) * d_ctx


def run_online(embs, signal, c, warmup=3):
    n = len(embs); sigs = [0.0] * n
    for t in range(1, n):
        sigs[t] = fdm_best(embs[t - 1], embs[t]) if signal == "idm" else delta_eff(embs, t)
    out = []; seg = 0; past = []
    for t in range(n):
        thr = None; bnd = False
        if t >= 1:
            s = sigs[t]
            thr = (float(np.mean(past)) + c * float(np.std(past))) if len(past) >= warmup else FALLBACK.get(signal)
            if thr is not None and s > thr:
                bnd = True; seg += 1
            past.append(s)
        out.append({"sig": round(sigs[t], 4), "thr": (round(thr, 4) if thr is not None else None),
                    "boundary": bnd, "seg": seg})
    return out


def segmentation(signal, c):
    seg = run_online(STATE["embs"], signal, c)
    turns = [{"text": t, **s} for t, s in zip(STATE["texts"], seg)]
    return {"turns": turns, "n_seg": (seg[-1]["seg"] + 1 if seg else 0)}


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
    for sent in split_sentences(text) if text else []:
        STATE["texts"].append(sent); STATE["embs"].append(embed(sent))
    return jsonify(**segmentation(d.get("signal", "idm"), float(d.get("c", 1.0))))


@app.route("/transcribe", methods=["POST"])
def transcribe_ep():
    signal = request.args.get("signal", "idm"); c = float(request.args.get("c", 1.0))
    lang = request.args.get("lang") or None
    pcm = decode_audio(request.get_data())      # browser sends encoded (webm/opus) blob
    text = transcribe(pcm, 16000, lang)
    added = []
    for sent in split_sentences(text):
        STATE["texts"].append(sent); STATE["embs"].append(embed(sent)); added.append(sent)
    return jsonify(transcript=text, added=added, **segmentation(signal, c))


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>음성 실시간 화제분절</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--fg:#e6e8ee;--mut:#8b93a7;--acc:#5b9dff;--rec:#ff5b6e;--line:#2a2e38}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,'Apple SD Gothic Neo',sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:860px;margin:0 auto;padding:20px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px;font-size:13px}
.inrow{display:flex;gap:8px;margin-bottom:8px}
textarea{flex:1;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:10px;font-size:15px;resize:none;height:44px}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:0 16px;font-size:14px;cursor:pointer;height:44px}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--mut)}
button.mic{background:var(--card);border:1px solid var(--line);font-size:20px;width:52px}
button.mic.rec{background:var(--rec);border-color:var(--rec);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.seg{border-left:3px solid var(--acc);background:var(--card);border-radius:0 10px 10px 0;margin:10px 0;padding:8px 12px}
.seghd{font-size:11px;letter-spacing:.04em;margin-bottom:6px}
.turn{display:flex;justify-content:space-between;gap:12px;padding:4px 0;border-top:1px dashed var(--line);font-size:14px}
.turn:first-of-type{border-top:0}.tx{flex:1}.meta{color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.bnd{color:#ff8a5b;font-weight:600}.stat{color:var(--mut);font-size:12px}.busy{color:var(--acc)}
</style></head><body><div class="wrap">
<h1>🎙️ 음성 실시간 화제분절 <span style="color:var(--mut);font-weight:400">· Qwen3-ASR + mpnet 온라인 분절</span></h1>
<div class="sub">🎤 눌러 말하고 다시 눌러 멈추면 → 전사 → 문장 단위 화제분절. 텍스트 입력도 가능.</div>
<div class="controls">
 <label>신호: <select id="sig"><option value="idm" selected>IDM FDM_best</option><option value="cosine">δ_eff (cosine)</option></select></label>
 <label>언어: <select id="lang"><option value="ko">한국어</option><option value="en">English</option><option value="">자동</option></select></label>
 <label>민감도 c: <input id="c" type="range" min="0.3" max="2.5" step="0.1" value="1.0"><span id="cv">1.0</span></label>
 <button class="ghost" id="rst">초기화</button> <span class="stat" id="stat"></span>
</div>
<div class="inrow">
 <button class="mic" id="mic" title="녹음">🎤</button>
 <textarea id="in" placeholder="문장/문단 입력 후 Enter — 또는 🎤로 말하기"></textarea>
 <button id="send">전송</button>
</div>
<div id="out"></div></div>
<script>
const HUES=[210,150,30,280,340,110,190,50,260,10];const $=s=>document.querySelector(s);
$('#c').oninput=e=>{$('#cv').textContent=e.target.value;refresh();};
$('#sig').onchange=refresh;let LAST={turns:[],n_seg:0};
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});return r.json();}
async function send(){const t=$('#in').value.trim();if(!t)return;$('#in').value='';setBusy('인코딩·분절…');LAST=await post('/add',{text:t,signal:$('#sig').value,c:parseFloat($('#c').value)});render(LAST);setBusy('');}
async function refresh(){LAST=await post('/add',{text:'',signal:$('#sig').value,c:parseFloat($('#c').value)});render(LAST);}
async function reset(){await post('/reset');LAST={turns:[],n_seg:0};render(LAST);}
$('#send').onclick=send;$('#rst').onclick=reset;
$('#in').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
function setBusy(m){$('#stat').innerHTML=m?'<span class="busy">'+m+'</span>':(LAST.turns.length?LAST.turns.length+'문장 · '+LAST.n_seg+'화제':'');}
// ---- mic (MediaRecorder → webm/opus; 서버가 ffmpeg로 디코드) ----
let rec=false,mr,chunks=[],stream;
$('#mic').onclick=async()=>{ if(!rec){startRec();} else {stopRec();} };
async function startRec(){
 // ASR용 원음: 에코제거/노이즈억제/자동게인 OFF (통화용 처리는 음성을 왜곡)
 try{stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:true}});}
 catch(e){alert('마이크 권한 필요: '+e.message);return;}
 chunks=[];
 let mime='';for(const m of ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4']){if(MediaRecorder.isTypeSupported(m)){mime=m;break;}}
 mr=new MediaRecorder(stream, mime?{mimeType:mime}:undefined);
 mr.ondataavailable=e=>{if(e.data&&e.data.size)chunks.push(e.data);};
 mr.onstop=onStop;
 mr.start();rec=true;$('#mic').classList.add('rec');setBusy('🔴 녹음 중…');}
function stopRec(){rec=false;$('#mic').classList.remove('rec');setBusy('🧠 Qwen3-ASR 전사 중…');mr.stop();}
async function onStop(){
 stream.getTracks().forEach(t=>t.stop());
 const blob=new Blob(chunks,{type:mr.mimeType||'audio/webm'});
 if(blob.size<1200){setBusy('너무 짧음');return;}
 const q=`/transcribe?signal=${$('#sig').value}&c=${$('#c').value}&lang=${$('#lang').value}`;
 const r=await fetch(q,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:blob});
 LAST=await r.json();render(LAST);setBusy(LAST.transcript?('전사: '+LAST.transcript.slice(0,40)+'…'):'전사 없음');}
function render(d){const turns=d.turns||[];let html='',cur=-1;
 turns.forEach(t=>{if(t.seg!==cur){if(cur!==-1)html+='</div>';cur=t.seg;const h=HUES[cur%HUES.length];
  html+=`<div class="seg" style="border-left-color:hsl(${h} 70% 60%)"><div class="seghd" style="color:hsl(${h} 70% 65%)">화제 ${cur+1}</div>`;}
  const mark=t.boundary?'<span class="bnd">▲경계</span> ':'';const meta=t.thr!=null?`s=${t.sig}/thr=${t.thr}`:`s=${t.sig}`;
  html+=`<div class="turn"><div class="tx">${mark}${esc(t.text)}</div><div class="meta">${meta}</div></div>`;});
 if(cur!==-1)html+='</div>';$('#out').innerHTML=html;
 if(!$('#stat').querySelector('.busy'))setBusy('');}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
reset();
</script></body></html>"""


if __name__ == "__main__":
    print("[warmup] Qwen3-ASR triton compile ...", flush=True)
    try:
        transcribe(np.zeros(16000, np.float32), 16000)
    except Exception as e:
        print("[warmup] skip:", e, flush=True)
    print(f"[ready] http://0.0.0.0:8000  (device={DEV})", flush=True)
    app.run(host="0.0.0.0", port=8000, threaded=True)
