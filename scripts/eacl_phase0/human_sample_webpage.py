#!/usr/bin/env python3
"""①B human 층화표본 — 브라우저 블라인드 라벨링 페이지 생성.

sheet.jsonl(블라인드: assigned_label·is_neutral 없음)을 self-contained HTML 로 굽는다. 사람이 브라우저에서
문장+choices 만 보고 라벨을 매기고(모델 답 미포함), 진행상황은 localStorage 자동저장, 끝나면 CSV 로 내보낸다.
내보낸 CSV(row_id·human_label·notes) 를 `human_sample_score.py --stage1 --labels <csv>` 로 채점.

Artifact 로 publish 하거나 파일로 직접 열어도 됨. 데이터에 모델 라벨이 없으므로 anchoring 원천 차단.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TEMPLATE = r"""<style>
:root{
  --ground:#f4f2ec; --surface:#fffdf9; --ink:#1c1f24; --muted:#6b717a; --faint:#8d9199;
  --border:#e5e1d6; --line:#efece3; --accent:#1f7a70; --accent-ink:#ffffff; --accent-soft:#dcefea;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --radius:10px;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#14171b; --surface:#1b1f25; --ink:#e9eaec; --muted:#99a0a8; --faint:#767d85;
  --border:#2a2f36; --line:#23282f; --accent:#5cbcb0; --accent-ink:#0b1013; --accent-soft:#1d2b2a;
}}
:root[data-theme="dark"]{
  --ground:#14171b; --surface:#1b1f25; --ink:#e9eaec; --muted:#99a0a8; --faint:#767d85;
  --border:#2a2f36; --line:#23282f; --accent:#5cbcb0; --accent-ink:#0b1013; --accent-soft:#1d2b2a;
}
:root[data-theme="light"]{
  --ground:#f4f2ec; --surface:#fffdf9; --ink:#1c1f24; --muted:#6b717a; --faint:#8d9199;
  --border:#e5e1d6; --line:#efece3; --accent:#1f7a70; --accent-ink:#ffffff; --accent-soft:#dcefea;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:900px;margin:0 auto;padding:20px 20px 96px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px;padding-bottom:14px;
  border-bottom:1px solid var(--border)}
h1{font-size:19px;margin:0;letter-spacing:-.01em;font-weight:650}
.sub{color:var(--muted);font-size:13px}
.count{margin-left:auto;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:13px;
  color:var(--muted)}
.count b{color:var(--ink)}
.save{font-size:12px;color:var(--faint);font-family:var(--mono)}

.map{display:flex;flex-direction:column;gap:7px;margin:16px 0 6px}
.maprow{display:flex;align-items:center;gap:9px}
.maplabel{width:104px;flex:none;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);text-align:right;font-family:var(--mono)}
.cells{display:grid;grid-template-columns:repeat(20,1fr);gap:3px;flex:1;min-width:0}
.cell{aspect-ratio:1;border-radius:3px;background:transparent;border:1px solid var(--border);cursor:pointer;
  min-width:0;display:flex;align-items:center;justify-content:center;line-height:1;
  font-family:var(--mono);font-size:11px;font-variant-numeric:tabular-nums;color:var(--faint)}
.cell:hover{border-color:var(--accent)}
.cell.done{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-weight:600}
.cell.note{box-shadow:inset 0 0 0 1px var(--surface),inset 0 0 0 2px var(--accent)}
.cell.cur{outline:2px solid var(--ink);outline-offset:1px}
.mapcap{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  font-family:var(--mono);margin:16px 0 8px;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.mapcap .lg{font-size:10px;color:var(--faint);text-transform:none;letter-spacing:0}
.mapcap .sw{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:-1px;margin-right:3px}

.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  margin-top:14px;padding:20px 22px 18px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}
.chip{font-size:11px;font-family:var(--mono);text-transform:uppercase;letter-spacing:.05em;
  padding:3px 8px;border-radius:999px;border:1px solid var(--border);color:var(--muted)}
.chip.axis{color:var(--accent-ink);background:var(--accent);border-color:var(--accent);letter-spacing:.06em}
.idx{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--faint);
  font-variant-numeric:tabular-nums}
.ctxcap{margin:20px 0 7px;padding-top:15px;border-top:1px solid var(--line);
  font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-family:var(--mono)}
.ctx{border:1px solid var(--border);border-radius:8px;background:var(--ground);padding:5px 8px;
  display:flex;flex-direction:column;gap:1px}
.ctx details{border-radius:6px}
.ctx summary{cursor:pointer;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);font-family:var(--mono);padding:4px 2px;list-style:none}
.ctx summary::-webkit-details-marker{display:none}
.ctx summary::before{content:"▸";color:var(--faint);margin-right:5px;font-size:9px}
.ctx details[open] summary::before{content:"▾"}
.ctx summary .n{color:var(--faint);margin-left:4px}
.ctx .gitem{font-size:12.5px;line-height:1.5;padding:1px 6px 1px 20px}
.ctx .gitem b{font-weight:650;color:var(--ink)}
.ctx .gitem span{color:var(--muted)}
.ctx .gitem .en{font-family:var(--mono);font-size:11px;color:var(--faint)}
.ctx .gwrap{margin:2px 0 5px}
.sentence{font-family:var(--serif);font-size:22px;line-height:1.45;margin:2px 0 6px;text-wrap:pretty}
.ko{font-family:var(--serif);font-size:15px;line-height:1.55;color:var(--muted);margin:0 0 20px;
  padding-left:11px;border-left:2px solid var(--border)}
.q{font-size:12px;color:var(--muted);margin:0 0 9px;text-transform:uppercase;letter-spacing:.05em}
.filter{width:100%;margin-bottom:11px;padding:8px 11px;border:1px solid var(--border);
  border-radius:8px;background:var(--ground);color:var(--ink);font-family:var(--mono);font-size:13px}
.filter::placeholder{color:var(--faint)}
.filter:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.choices{display:flex;flex-wrap:wrap;gap:8px}
.choice{font-family:var(--mono);font-size:13px;padding:7px 12px;border-radius:8px;
  border:1px solid var(--border);background:var(--ground);color:var(--ink);cursor:pointer;
  transition:transform .04s}
.choice:hover{border-color:var(--accent)}
.choice:active{transform:translateY(1px)}
.choice.sel{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-weight:600}
.choice:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.choice.yn{font-family:var(--sans);font-size:15px;padding:12px 20px;flex:1;min-width:120px;justify-content:center;text-align:center}
body[data-mode="yesno"] #q{font-family:var(--serif);font-size:16px;color:var(--ink);text-transform:none;
  letter-spacing:0;margin-bottom:13px;line-height:1.45}
body[data-mode="yesno"] #q b{color:var(--accent);font-weight:700}
.hint{font-size:12px;color:var(--muted);border:1px dashed var(--border);border-radius:6px;
  padding:7px 10px;margin:0 0 13px;line-height:1.5}
.hint:empty{display:none}
.hint b{color:var(--ink);font-weight:600}
.notes{width:100%;margin-top:16px;padding:8px 11px;border:1px solid var(--border);border-radius:8px;
  background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:13px;resize:vertical;
  min-height:38px}
.notes:focus-visible{outline:2px solid var(--accent);outline-offset:1px}

.nav{display:flex;align-items:center;gap:10px;margin-top:18px}
.btn{font:inherit;font-size:13px;padding:8px 15px;border-radius:8px;border:1px solid var(--border);
  background:var(--surface);color:var(--ink);cursor:pointer}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-weight:600}
.btn:disabled{opacity:.4;cursor:default}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.kbd{margin-left:auto;font-size:11px;color:var(--faint);font-family:var(--mono)}
.kbd b{color:var(--muted);font-weight:600}

.exportbar{position:fixed;left:0;right:0;bottom:0;background:var(--surface);
  border-top:1px solid var(--border);padding:11px 20px;display:flex;gap:10px;align-items:center}
.exportbar .info{font-size:12px;color:var(--muted)}
.exportbar .spacer{margin-left:auto}
dialog{border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);
  color:var(--ink);max-width:680px;width:92vw;padding:0}
dialog::backdrop{background:rgba(10,12,14,.5)}
.dlg-h{padding:16px 20px 0;font-weight:650}
.dlg-p{padding:6px 20px 0;font-size:13px;color:var(--muted)}
.csv{margin:14px 20px;width:calc(100% - 40px);height:220px;font-family:var(--mono);font-size:12px;
  border:1px solid var(--border);border-radius:8px;background:var(--ground);color:var(--ink);padding:10px;
  white-space:pre;overflow:auto}
.dlg-nav{display:flex;gap:10px;padding:0 20px 18px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media (max-width:620px){.maplabel{width:64px}.sentence{font-size:19px}.wrap{padding:16px 14px 96px}}
</style>

<div class="wrap">
  <header>
    <h1>①B 블라인드 라벨링</h1>
    <span class="sub">문장만 보고 라벨을 매기세요 · 모델 답은 표시되지 않습니다</span>
    <span class="count"><b id="ndone">0</b>/<span id="ntot">0</span></span>
    <span class="save" id="save"></span>
  </header>

  <div class="mapcap">문항 1–100 · 눌러서 이동
    <span class="lg"><span class="sw" style="background:var(--accent)"></span>완료
      &nbsp;<span class="sw" style="border:1px solid var(--border)"></span>미완료</span>
  </div>
  <div class="map" id="map"></div>

  <div class="card">
    <div class="meta">
      <span class="chip axis" id="m-axis">axis</span>
      <span class="chip" id="m-field" title="이 축에서 고르는 라벨 필드"></span>
      <span class="chip" id="m-bin">bin</span>
      <span class="chip" id="m-doc" style="font-family:var(--mono)"></span>
      <span class="idx" id="m-idx"></span>
    </div>
    <p class="sentence" id="sentence"></p>
    <p class="ko" id="ko"></p>
    <p class="q" id="q">이 문장의 라벨을 고르세요</p>
    <p class="hint" id="hint"></p>
    <input class="filter" id="filter" placeholder="선택지 필터 (타이핑) · Enter=단일후보 선택" autocomplete="off">
    <div class="choices" id="choices"></div>
    <textarea class="notes" id="notes" placeholder="메모 (선택) — 애매하면 왜 애매한지"></textarea>
    <div class="nav">
      <button class="btn" id="prev">← 이전</button>
      <button class="btn primary" id="next">다음 →</button>
      <span class="kbd"><b>←/→</b> 이동 · <b>필터+Enter</b> 선택 · 선택 시 자동 다음</span>
    </div>
    <div class="ctxcap">아래 목록 — 가능한 답 후보 (참고)</div>
    <div class="ctx" id="ctx"></div>
  </div>

</div>

<div class="exportbar">
  <span class="info">진행은 이 브라우저에 자동저장됩니다. <b>내보내기</b>로 CSV를 받아 돌려주세요.</span>
  <span class="spacer"></span>
  <button class="btn" id="jump">미완료로 이동</button>
  <button class="btn primary" id="export">CSV 내보내기</button>
</div>

<dialog id="dlg">
  <div class="dlg-h">CSV 내보내기</div>
  <div class="dlg-p" id="dlg-p"></div>
  <textarea class="csv" id="csv" readonly></textarea>
  <div class="dlg-nav">
    <button class="btn primary" id="copy">클립보드에 복사</button>
    <button class="btn" id="download">파일로 저장</button>
    <button class="btn" id="close" style="margin-left:auto">닫기</button>
  </div>
</dialog>

<script>
const ITEMS = __DATA__;
const DOCS = __DOCS__;              // doc_id -> {cast, locations, goals} (LLM Stage-1 맥락)
const GLOSS = __GLOSS__;            // identifier -> 한국어 gloss (맥락 패널 가독성)
const MODE = __MODE__;              // "yesno" | "full"
const AXIS_Q = __AXISQ__;           // 축별 예/아니오 질문(yesno)
const FIELD = {space:"location", intentionality:"active_goal", time:"temporal_shift_marker",
               protagonist:"perspective", causality:"aspect"};
const YESNO = [["yes","예 — 근거 있음"],["no","아니오 — 근거 없음"],["unclear","애매"]];
const KEY = "eacl_gate1b_human_v" + __VER__;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { state = {}; }
let cur = 0, filt = "";

// axis groups in sheet order
const axes = [...new Set(ITEMS.map(x=>x.axis))];
const $ = s => document.querySelector(s);

function saveState(){
  try { localStorage.setItem(KEY, JSON.stringify(state)); $("#save").textContent="저장됨"; }
  catch(e){ $("#save").textContent="⚠ 저장실패 — 자주 내보내세요"; }
  setTimeout(()=>{ if($("#save").textContent==="저장됨") $("#save").textContent=""; }, 1200);
}
function rec(i){ const id=ITEMS[i].row_id; return state[id] || (state[id]={label:"",notes:""}); }
function doneCount(){ return ITEMS.filter(x=>state[x.row_id]&&state[x.row_id].label).length; }

function buildMap(){
  const map=$("#map"); map.innerHTML="";
  axes.forEach(ax=>{
    const row=document.createElement("div"); row.className="maprow";
    const lab=document.createElement("div"); lab.className="maplabel"; lab.textContent=ax; row.appendChild(lab);
    const cells=document.createElement("div"); cells.className="cells";
    ITEMS.forEach((it,i)=>{ if(it.axis!==ax) return;
      const b=document.createElement("button"); b.className="cell"; b.dataset.i=i;
      b.textContent=i+1; b.title=(i+1)+" · "+it.axis; b.onclick=()=>go(i); cells.appendChild(b);
    });
    row.appendChild(cells); map.appendChild(row);
  });
}
function refreshMap(){
  document.querySelectorAll(".cell").forEach(c=>{
    const i=+c.dataset.i, r=state[ITEMS[i].row_id];
    c.classList.toggle("done", !!(r&&r.label));
    c.classList.toggle("note", !!(r&&r.notes&&r.notes.trim()));
    c.classList.toggle("cur", i===cur);
  });
}
function render(){
  const it=ITEMS[cur], r=rec(cur);
  $("#m-axis").textContent=it.axis;
  $("#m-field").textContent=FIELD[it.axis]||"";
  $("#m-field").style.display=(MODE==="yesno")?"none":"";
  $("#m-bin").textContent="대사밀도 "+it.bin;
  $("#m-doc").textContent=it.doc_id;
  $("#m-idx").textContent="문항 "+(cur+1)+" / "+ITEMS.length;
  renderCtx(it.doc_id, it.axis);
  $("#sentence").textContent=it.sentence;
  $("#ko").textContent=it.ko||"";
  $("#notes").value=r.notes||"";
  filt=""; $("#filter").value=""; $("#filter").style.display=(MODE==="yesno")?"none":"";
  if(MODE==="yesno"){ $("#q").innerHTML=AXIS_Q[it.axis]||"판단해 주세요";
    $("#hint").innerHTML="💡 <b>이 문장만</b> 보고 판단하세요. 뒤 내용으로 ‘여기·그·거기’가 어디/누구인지 "
      +"추론하지 마세요 — 아래 목록은 ‘가능한 답 후보’일 뿐입니다."; }
  else { $("#q").textContent="이 문장의 라벨을 고르세요"; $("#hint").textContent=""; }
  renderChoices();
  $("#prev").disabled=(cur===0); $("#next").disabled=(cur===ITEMS.length-1);
  $("#ndone").textContent=doneCount();
  refreshMap();
}
const AXIS_LIST = {space:"locations", intentionality:"goals", protagonist:"cast"};
function renderCtx(doc, axis){
  const d=DOCS[doc]||{}; const box=$("#ctx"); const open=AXIS_LIST[axis];
  const item=x=>{ const g=GLOSS[x]||{}; const ko=(g.ko||x), desc=g.desc||"";
    return `<div class="gitem" title="${x}"><b>${ko}</b> <span class="en">(${x})</span>`
         + `${desc?` — <span>${desc}</span>`:""}</div>`; };
  const sec=(lab,key,arr)=>{ if(!arr||!arr.length) return "";
    return `<details${key===open?" open":""}><summary>${lab}<span class="n">${arr.length}</span></summary>`
         + `<div class="gwrap">${arr.map(item).join("")}</div></details>`; };
  box.innerHTML = sec("등장인물","cast",d.cast) + sec("장소","locations",d.locations)
                + sec("목표","goals",d.goals);
}
function visibleChoices(){
  const cs=ITEMS[cur].choices; const f=filt.trim().toLowerCase();
  return f? cs.filter(c=>c.toLowerCase().includes(f)) : cs;
}
function renderChoices(){
  const box=$("#choices"); box.innerHTML=""; const sel=rec(cur).label;
  const opts = (MODE==="yesno") ? YESNO : ITEMS[cur].choices.map(c=>[c,c]);
  const src = (MODE==="yesno") ? opts : opts.filter(([v])=>{
    const f=filt.trim().toLowerCase(); return !f || v.toLowerCase().includes(f); });
  src.forEach(([val,txt])=>{
    const b=document.createElement("button"); b.className="choice"+(val===sel?" sel":"");
    if(MODE==="yesno") b.classList.add("yn");
    b.textContent=txt; b.onclick=()=>pick(val); box.appendChild(b);
  });
}
function pick(c){
  rec(cur).label=c; saveState(); renderChoices(); refreshMap(); $("#ndone").textContent=doneCount();
  if(cur<ITEMS.length-1) setTimeout(()=>go(cur+1), 140);
}
function go(i){ if(i<0||i>=ITEMS.length) return; rec(cur).notes=$("#notes").value; saveState(); cur=i; render();
  document.querySelector(".card").scrollIntoView({block:"nearest"}); }

$("#prev").onclick=()=>go(cur-1);
$("#next").onclick=()=>go(cur+1);
$("#notes").addEventListener("input", ()=>{ rec(cur).notes=$("#notes").value; });
$("#notes").addEventListener("blur", saveState);
$("#filter").addEventListener("input", e=>{ filt=e.target.value; renderChoices(); });
$("#filter").addEventListener("keydown", e=>{
  if(e.key==="Enter"){ const v=visibleChoices(); if(v.length>=1){ e.preventDefault(); pick(v[0]); } }
});
document.addEventListener("keydown", e=>{
  if(e.target.tagName==="TEXTAREA"||e.target.id==="filter"){
    if(e.target.id==="filter" && (e.key==="ArrowLeft"||e.key==="ArrowRight")) return;
    if(e.target.tagName==="TEXTAREA") return;
  }
  if(e.key==="ArrowLeft"){ go(cur-1); return; }
  if(e.key==="ArrowRight"){ go(cur+1); return; }
  if(MODE==="yesno"){
    const m={"1":"yes","2":"no","3":"unclear","y":"yes","Y":"yes","n":"no","N":"no"};
    if(m[e.key]){ e.preventDefault(); pick(m[e.key]); }
  }
});
$("#jump").onclick=()=>{ const i=ITEMS.findIndex(x=>!(state[x.row_id]&&state[x.row_id].label));
  if(i<0){ alert("모든 문항 완료!"); } else go(i); };

function csvText(){
  const esc=s=>{ s=(s==null?"":String(s)); return /[",\n]/.test(s)? '"'+s.replace(/"/g,'""')+'"' : s; };
  const lines=["row_id,axis,human_label,notes"];
  ITEMS.forEach(it=>{ const r=state[it.row_id]||{}; lines.push([it.row_id,it.axis,r.label||"",r.notes||""].map(esc).join(",")); });
  return lines.join("\n");
}
$("#export").onclick=()=>{ rec(cur).notes=$("#notes").value; saveState();
  const n=doneCount(); $("#dlg-p").textContent=`라벨 ${n}/${ITEMS.length} 완료. 아래 CSV를 복사해 돌려주거나 파일로 저장하세요.`;
  $("#csv").value=csvText(); $("#dlg").showModal(); };
$("#copy").onclick=async()=>{ try{ await navigator.clipboard.writeText($("#csv").value); $("#copy").textContent="복사됨 ✓";
  setTimeout(()=>$("#copy").textContent="클립보드에 복사",1200);}catch(e){ $("#csv").select(); document.execCommand&&document.execCommand("copy"); } };
$("#download").onclick=()=>{ const blob=new Blob([csvText()],{type:"text/csv"}); const u=URL.createObjectURL(blob);
  const a=document.createElement("a"); a.href=u; a.download="gate1b_human_labels.csv"; a.click(); URL.revokeObjectURL(u); };
$("#close").onclick=()=>$("#dlg").close();

$("#ntot").textContent=ITEMS.length;
document.body.dataset.mode=MODE;
$(".kbd").innerHTML = (MODE==="yesno")
  ? '<b>1</b> 예 · <b>2</b> 아니오 · <b>3</b> 애매 · <b>←/→</b> 이동'
  : '<b>←/→</b> 이동 · <b>필터+Enter</b> 선택 · 선택 시 자동 다음';
if(MODE==="yesno") $(".sub").textContent="문장을 읽고 예/아니오만 누르세요 · 모델 답은 표시되지 않습니다";
buildMap(); render();
</script>
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path,
                    default=root / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/sheet.jsonl")
    ap.add_argument("--out", type=Path,
                    default=root / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/label_web.html")
    ap.add_argument("--run-dir", type=Path,
                    default=root / "outputs/eacl_phase0/repcorpus_s5_eismv2_run1",
                    help="story_lists.json 위치 (LLM Stage-1 맥락 패널용)")
    ap.add_argument("--translations", type=Path,
                    default=root / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/translations.json",
                    help="{row_id: 한국어} — 있으면 문장 아래 번역 표시")
    ap.add_argument("--gloss", type=Path,
                    default=root / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/list_gloss.json",
                    help="{identifier: 한국어} — 맥락 패널 목록을 한국어로 표시")
    ap.add_argument("--mode", choices=("yesno", "full"), default="yesno",
                    help="yesno=축별 예/아니오(중립 정당성만, 쉬움) · full=20지선다 정밀")
    ap.add_argument("--version", default="20260723v2", help="localStorage 키 버전(시트/모드 바뀌면 올려 캐시 무효화)")
    args = ap.parse_args()

    # 축별 예/아니오 질문 (yesno): 예=근거/신호 있음(모델 non-neutral), 아니오=없음(모델 중립토큰)
    axis_q = {
        "space": "이 문장이 아래 <b>장소 목록 중 한 곳</b>에서 벌어지나요? "
                 "(무대가 목록에 없거나 알 수 없으면 ‘아니오’)",
        "intentionality": "이 문장의 행동이 아래 <b>목표 목록 중 하나</b>를 위한 것으로 읽히나요? "
                          "(해당 목표가 목록에 없거나 목표가 안 보이면 ‘아니오’)",
        "time": "이 문장에 <b>시간이 건너뛰거나 이어지는 신호</b>가 있나요? (그다음·나중에·그때·동시에 등)",
        "protagonist": "이 문장에 <b>행위의 주체(누가 하는지·인칭)</b>가 드러나 있나요?",
        "causality": "이 문장이 <b>행동·상태의 시작이나 완료</b>를 나타내나요? (막 시작/막 끝남 ↔ 그냥 진행 중)",
    }

    ko = json.loads(args.translations.read_text()) if args.translations.exists() else {}
    gloss = json.loads(args.gloss.read_text()) if args.gloss.exists() else {}
    rows = [json.loads(x) for x in args.sheet.read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [{"row_id": r["row_id"], "axis": r["axis"], "doc_id": r["doc_id"],
              "bin": r["density_bin"], "sentence": r["sentence"], "choices": r["choices"],
              "ko": ko.get(r["row_id"], "")} for r in rows]

    # per-doc Stage-1 lists (cast/locations/goals) = the context the LLM had before labeling
    docs = {}
    for doc in sorted({r["doc_id"] for r in rows}):
        lp = args.run_dir / doc / "story_lists.json"
        L = json.loads(lp.read_text())["lists"] if lp.exists() else {}
        docs[doc] = {"cast": sorted(set(L.get("cast", []))),
                     "locations": sorted(set(L.get("locations", []))),
                     "goals": sorted(set(L.get("goals", [])))}

    # inject as valid JS; escape '<' so nothing can break out of <script>
    def js(o): return json.dumps(o, ensure_ascii=False).replace("<", "\\u003c")
    html = (TEMPLATE.replace("__DATA__", js(items)).replace("__DOCS__", js(docs))
            .replace("__GLOSS__", js(gloss))
            .replace("__VER__", json.dumps(args.version)).replace("__MODE__", json.dumps(args.mode))
            .replace("__AXISQ__", js(axis_q)))
    args.out.write_text(html, encoding="utf-8")
    print(f"{len(items)} items ({args.mode}) · {sum(bool(i['ko']) for i in items)} 번역 · "
          f"{len(docs)} docs · {len(gloss)} gloss -> {args.out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
