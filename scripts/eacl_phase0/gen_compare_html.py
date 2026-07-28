#!/usr/bin/env python3
"""compare_enriched.json → 브라우징 HTML. 세리프 본문 + 정렬된 대조. self-contained."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/experiments/2026-07-24_qwen_human_verify"
data = json.loads((D / "compare_enriched.json").read_text())
DATA_JSON = json.dumps(data, ensure_ascii=False)

TMPL = r'''<div id="app"></div>
<style>
:root{
  --paper:#f4f5f7; --card:#ffffff; --ink:#20242b; --sub:#565d68; --faint:#9aa0aa;
  --line:#e5e7ec; --line2:#eef0f3; --accent:#3a5a8c;
  --agree:#3f7d5a; --agree-w:#e9f2ec; --diverge:#b0483c; --diverge-w:#f7e9e6; --un:#8a8f99; --un-w:#eef0f2;
  --serif:"Iowan Old Style","Charter",Georgia,"Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#16181c; --card:#1e2127; --ink:#e8eaee; --sub:#a2a9b4; --faint:#6b7280;
  --line:#2b2f37; --line2:#23272e; --accent:#7ba0d4;
  --agree:#5fb487; --agree-w:#18271f; --diverge:#e07a6e; --diverge-w:#2c1d1b; --un:#8b93a0; --un-w:#242830;
}}
:root[data-theme="dark"]{--paper:#16181c;--card:#1e2127;--ink:#e8eaee;--sub:#a2a9b4;--faint:#6b7280;--line:#2b2f37;--line2:#23272e;--accent:#7ba0d4;--agree:#5fb487;--agree-w:#18271f;--diverge:#e07a6e;--diverge-w:#2c1d1b;--un:#8b93a0;--un-w:#242830;}
:root[data-theme="light"]{--paper:#f4f5f7;--card:#ffffff;--ink:#20242b;--sub:#565d68;--faint:#9aa0aa;--line:#e5e7ec;--line2:#eef0f3;--accent:#3a5a8c;--agree:#3f7d5a;--agree-w:#e9f2ec;--diverge:#b0483c;--diverge-w:#f7e9e6;--un:#8a8f99;--un-w:#eef0f2;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:760px;margin:0 auto;padding:40px 22px 100px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:640;margin-bottom:10px}
h1{font-family:var(--serif);font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 8px;line-height:1.15}
.sub{color:var(--sub);margin:0 0 26px;font-size:14px;max-width:60ch;line-height:1.6}
.overview{display:flex;flex-direction:column;gap:0;border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-bottom:14px;background:var(--card)}
.orow{display:grid;grid-template-columns:120px 1fr 1fr;align-items:center;gap:16px;padding:12px 18px;border-top:1px solid var(--line2)}
.orow:first-child{border-top:none}
.orow.head{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);font-weight:640;padding-top:13px;padding-bottom:9px}
.orow .an{font-weight:600;font-size:14px}.orow .an small{display:block;font-weight:400;color:var(--faint);font-size:11px;margin-top:1px;letter-spacing:0;text-transform:none}
.meter{display:flex;align-items:center;gap:10px}
.meter .track{flex:1;height:6px;border-radius:4px;background:var(--line);overflow:hidden}
.meter .fill{height:100%;border-radius:4px;background:var(--accent)}
.meter .pv{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:13px;font-weight:600;min-width:38px;text-align:right}
.filters{display:flex;flex-wrap:wrap;gap:14px 20px;align-items:center;position:sticky;top:0;z-index:5;background:linear-gradient(var(--paper) 82%,transparent);padding:14px 2px 12px;margin-top:6px}
.fg{display:flex;gap:6px;align-items:center}.fg .fl{font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.07em;margin-right:2px}
button.pill{font-family:var(--sans);font-size:12.5px;padding:5px 12px;border-radius:8px;cursor:pointer;border:1px solid var(--line);background:var(--card);color:var(--sub);transition:.13s}
button.pill:hover{border-color:var(--accent);color:var(--ink)}
button.pill[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--paper)}
button.pill:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.count{color:var(--faint);font-size:12.5px;margin:2px 2px 4px;font-variant-numeric:tabular-nums}
.axhead{font-family:var(--serif);font-size:19px;font-weight:600;margin:34px 0 3px}
.axhead .q{display:block;font-family:var(--sans);font-weight:400;font-size:12.5px;color:var(--sub);margin-top:5px;line-height:1.5}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin-top:14px;position:relative}
.card.dis{border-color:color-mix(in srgb,var(--diverge) 32%,var(--line))}
.chd{display:flex;align-items:baseline;gap:12px;margin-bottom:12px}
.num{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--faint);letter-spacing:.02em}
.rid{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-left:auto}
.en{font-family:var(--serif);font-size:18px;line-height:1.55;letter-spacing:-.003em}
.ko{font-size:14px;line-height:1.65;color:var(--sub);margin-top:9px}
.rows{margin-top:18px;padding-top:16px;border-top:1px solid var(--line2);display:flex;flex-direction:column;gap:11px}
.r{display:grid;grid-template-columns:88px 1fr auto;align-items:center;gap:14px}
.r .role{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);font-weight:640}
.r .v{display:flex;align-items:center;gap:9px;min-width:0}
.r .v .dot{width:8px;height:8px;border-radius:50%;flex:none;background:var(--faint)}
.r .v .txt{font-size:14.5px;line-height:1.3}.r .v .txt b{font-weight:640}
.r .v .id{font-family:var(--mono);font-size:11px;color:var(--faint);margin-left:2px}
.r .v .ne{font-size:12px;color:var(--faint)}
.tag{font-size:11px;font-weight:640;padding:2px 9px;border-radius:999px;white-space:nowrap;letter-spacing:.01em}
.t-agree{color:var(--agree);background:var(--agree-w)}.t-diverge{color:var(--diverge);background:var(--diverge-w)}.t-un{color:var(--un);background:var(--un-w)}
.d-agree{background:var(--agree)}.d-diverge{background:var(--diverge)}.d-un{background:var(--un)}
.human .v .txt{font-weight:640}
.empty{color:var(--faint);text-align:center;padding:56px;font-family:var(--serif);font-size:17px}
</style>
<script>
const DATA=__DATA__;
const NEUT={space:"unspecified",intentionality:"none",time:"absent",protagonist:"none",causality:"ongoing"};
const NEUTKO={space:"미지정",intentionality:"목표 없음",time:"전환 없음",protagonist:"인칭 없음",causality:"진행중"};
const AXK={space:"장소 · 무대",intentionality:"목표",time:"시간전환",protagonist:"인칭",causality:"상(aspect)"};
const AXQ={space:"이 문장의 행동이 벌어지는 무대가 목록의 장소 중 하나인가?",intentionality:"이 문장의 행동이 목록의 목표 중 하나를 향하나?",time:"이 문장이 시간전환 표지(then/later/meanwhile…)로 시작하나?",protagonist:"이 문장에 인물의 문법 인칭이 드러나나?",causality:"이 문장의 상(aspect)이 시작·완료(inceptive/completive)인가?"};
const AXORDER=["space","intentionality","time","protagonist","causality"];
function hstate(h){return h==="no"?"neutral":h==="yes"?"nonneutral":"unclear";}
function mstate(ax,l){return l===NEUT[ax]?"neutral":"nonneutral";}
function verdict(ax,h,l){const hs=hstate(h);if(hs==="unclear")return"un";const ms=mstate(ax,l);if(hs===ms)return"agree";return"diverge";}
const TAG={agree:"일치",diverge:"불일치",un:"애매"};
let fAxis="all",fView="all";
function summary(){const s={};for(const ax of AXORDER){const rows=DATA.filter(d=>d.axis===ax);let ds=0,q=0,n=0;for(const r of rows){if(hstate(r.human)!=="unclear"){n++;if(verdict(ax,r.human,r.deepseek)==="agree")ds++;if(verdict(ax,r.human,r.qwen)==="agree")q++;}}s[ax]={ds,q,n,tot:rows.length};}return s;}
function passView(r){const v=verdict(r.axis,r.human,r.qwen);if(fView==="all")return true;if(fView==="dis")return v==="diverge";if(fView==="over"){return v==="diverge"&&hstate(r.human)==="neutral";}if(fView==="under"){return v==="diverge"&&hstate(r.human)==="nonneutral";}if(fView==="un")return v==="un";return true;}
function vrow(role,ident,ko,ax,human){
  let vd,txt,idl="",ne="";
  if(role==="사람"){txt=human==="no"?"아니오 · <b>중립</b>":human==="yes"?"예 · <b>비중립</b>":"애매";return `<div class="r human"><div class="role">${role}</div><div class="v"><span class="txt">${txt}</span></div><div></div></div>`;}
  vd=verdict(ax,human,ident);const neu=mstate(ax,ident)==="neutral";
  txt=neu?`<b>${NEUTKO[ax]}</b>`:(ko?`<b>${esc(ko)}</b>`:`<b>${esc(ident)}</b>`);
  idl=(!neu&&ko)?`<span class="id">${esc(ident)}</span>`:'';
  ne=`<span class="ne">${neu?'중립':'비중립'}</span>`;
  return `<div class="r"><div class="role">${role}</div><div class="v"><span class="dot d-${vd}"></span><span class="txt">${txt}${idl} ${ne}</span></div><span class="tag t-${vd}">${TAG[vd]}</span></div>`;
}
function render(){
  const s=summary();
  let ov=`<div class="orow head"><div>축</div><div>deepseek 일치율</div><div>Qwen 일치율</div></div>`;
  for(const ax of AXORDER){const t=s[ax];const dp=t.n?Math.round(t.ds/t.n*100):0,qp=t.n?Math.round(t.q/t.n*100):0;
    ov+=`<div class="orow"><div class="an">${AXK[ax]}<small>${ax} · 애매 ${t.tot-t.n} 제외 · n=${t.n}</small></div>
      <div class="meter"><span class="track"><span class="fill" style="width:${dp}%"></span></span><span class="pv">${dp}%</span></div>
      <div class="meter"><span class="track"><span class="fill" style="width:${qp}%"></span></span><span class="pv">${qp}%</span></div></div>`;}
  const rows=DATA.filter(r=>(fAxis==="all"||r.axis===fAxis)&&passView(r));
  let list='';
  for(const ax of AXORDER){const rr=rows.filter(r=>r.axis===ax);if(!rr.length)continue;
    list+=`<div class="axhead">${AXK[ax]}<span class="q">사람이 판단한 질문 — ${AXQ[ax]} <span style="color:var(--faint)">(아니오 = ${NEUTKO[ax]})</span></span></div>`;
    for(const r of rr){const vq=verdict(r.axis,r.human,r.qwen);const cls=vq==="diverge"?"dis":"";
      list+=`<div class="card ${cls}"><div class="chd"><span class="num">No.${r.no}</span><span class="rid">${r.row_id}</span></div>
        <div class="en">${esc(r.sentence)}</div><div class="ko">${esc(r.ko||'')}</div>
        <div class="rows">${vrow("사람",null,null,r.axis,r.human)}${vrow("deepseek",r.deepseek,r.ds_ko,r.axis,r.human)}${vrow("Qwen",r.qwen,r.q_ko,r.axis,r.human)}</div></div>`;}}
  if(!rows.length)list=`<div class="empty">해당 필터에 케이스가 없습니다</div>`;
  document.getElementById("app").innerHTML=`<div class="wrap">
    <div class="eyebrow">Gate ①B · Human verification</div>
    <h1>사람 판정과 두 라벨러</h1>
    <p class="sub">대표 코퍼스 100문장을 사람이 블라인드로 <b>예 / 아니오 / 애매</b>로 매긴 뒤, deepseek·Qwen 라벨과 대조합니다. 각 문장은 영어 원문과 한국어 번역, 그리고 세 판정을 나란히 보여줍니다. 일치율은 애매를 제외한 값입니다.</p>
    <div class="overview">${ov}</div>
    <div class="filters"><div class="fg"><span class="fl">축</span><div id="axf" class="fg"></div></div><div class="fg"><span class="fl">보기</span><div id="vf" class="fg"></div></div></div>
    <div class="count">${rows.length}건 · Qwen 기준</div>${list}</div>`;
  mkpills("axf",[["all","전체"],["space","장소"],["intentionality","목표"],["time","시간"],["protagonist","인칭"],["causality","상"]],fAxis,v=>{fAxis=v;render();});
  mkpills("vf",[["all","전체"],["dis","불일치"],["over","과잉(over)"],["under","회피(under)"],["un","애매"]],fView,v=>{fView=v;render();});
}
function mkpills(id,opts,cur,cb){const el=document.getElementById(id);el.innerHTML='';for(const[v,lab]of opts){const b=document.createElement("button");b.className="pill";b.textContent=lab;b.setAttribute("aria-pressed",v===cur);b.onclick=()=>cb(v);el.appendChild(b);}}
function esc(s){return(s||'').replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
render();
</script>'''

out = D / "human_compare_review.html"
out.write_text(TMPL.replace("__DATA__", DATA_JSON), encoding="utf-8")
print("written", out, len(out.read_text()), "bytes")
