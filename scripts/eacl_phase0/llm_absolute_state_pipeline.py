#!/usr/bin/env python3
"""De-circularized P0: shuffled left-only absolute states, then state-only comparison."""
from __future__ import annotations
import argparse,json,os,random,time
from pathlib import Path
from dotenv import load_dotenv

SLOTS=("protagonist","place","time","causality","goal")
DEFS={
"protagonist":"the currently central acting person or cast, including enough event-specific identity to distinguish different first-person narrators",
"place":"the physical or narrative stage where the current action is situated",
"time":"the current narrative time frame, only when grounded by the text",
"causality":"the currently active cause-to-consequence chain, stated concretely",
"goal":"the currently active intention of the central actor, only if supported by the text",
}
def read(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
def write(p,rows):Path(p).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
def seq(rows):
 rows=sorted(rows,key=lambda x:int(x["transition"]));return [rows[0]["previous"],*[x["current"] for x in rows]]
def schema_state():
 item={"type":"object","additionalProperties":False,"properties":{"item_id":{"type":"integer"},"observed":{"type":"boolean"},"state":{"type":"string"},"evidence":{"type":"string"},"confidence":{"type":"number"}},"required":["item_id","observed","state","evidence","confidence"]}
 return {"type":"object","additionalProperties":False,"properties":{"items":{"type":"array","items":item}},"required":["items"]}
def schema_compare():
 item={"type":"object","additionalProperties":False,"properties":{"transition":{"type":"integer"},"comparable":{"type":"boolean"},"value":{"type":"integer","enum":[0,1]},"reason":{"type":"string"}},"required":["transition","comparable","value","reason"]}
 return {"type":"object","additionalProperties":False,"properties":{"items":{"type":"array","items":item}},"required":["items"]}
def call(client,model,name,system,user,schema):
 for a in range(3):
  try:
   r=client.chat.completions.create(model=model,temperature=0,max_tokens=10000,response_format={"type":"json_schema","json_schema":{"name":name,"strict":True,"schema":schema}},messages=[{"role":"system","content":system},{"role":"user","content":user}]);return json.loads(r.choices[0].message.content)["items"]
  except Exception:
   if a==2:raise
   time.sleep(2*(a+1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("master",type=Path);ap.add_argument("out_dir",type=Path);ap.add_argument("--model",default="openrouter/anthropic/claude-fable-5");ap.add_argument("--context",type=int,default=6);ap.add_argument("--batch",type=int,default=20);ap.add_argument("--seed",type=int,default=20260711);ap.add_argument("--axis",choices=SLOTS,action="append");args=ap.parse_args()
 load_dotenv(Path(__file__).resolve().parents[2]/".env");from openai import OpenAI
 c=OpenAI(api_key=os.environ["OPENAI_API_KEY"],base_url=os.environ["OPENAI_BASE_URL"],default_headers={"CF-Access-Client-Id":os.environ["CF_ACCESS_CLIENT_ID"],"CF-Access-Client-Secret":os.environ["CF_ACCESS_CLIENT_SECRET"]})
 rows=read(args.master);texts=seq(rows);args.out_dir.mkdir(parents=True,exist_ok=True)
 for axis in args.axis or SLOTS:
  sp=args.out_dir/f"{axis}_states.jsonl";done={x["item_id"]:x for x in read(sp)} if sp.exists() else {}
  ids=list(range(len(texts)));random.Random(args.seed+SLOTS.index(axis)).shuffle(ids)
  for b in range(0,len(ids),args.batch):
   todo=[i for i in ids[b:b+args.batch] if i not in done]
   if not todo:continue
   blocks=[]
   for i in todo:
    ctx="\n".join(f"[{j}] {texts[j]}" for j in range(max(0,i-args.context+1),i+1));blocks.append(f"ITEM {i}\n{ctx}")
   user=f"""Extract ONLY the current {axis} state: {DEFS[axis]}.
Each ITEM is independent and deliberately shuffled. Do not compare items or infer document/story boundaries. Do not label change or reset.
Use only the item's left-to-right context. If the state is not textually supported, observed=false, state='unknown', and do not invent it.
Evidence must quote the supplied English text. Return exactly one result per item_id.\n\n"""+"\n\n".join(blocks)
   got=call(c,args.model,f"absolute_{axis}","You extract grounded narrative states without comparing documents.",user,schema_state());gm={int(x["item_id"]):x for x in got}
   unexpected=set(gm)-set(todo)
   if unexpected:raise RuntimeError(f"{axis} unexpected state ids: {sorted(unexpected)}")
   if not gm:raise RuntimeError(f"{axis} returned no requested state ids")
   for i,x in gm.items():done[i]={"item_id":i,"axis":axis,"observed":bool(x["observed"]),"state":x["state"],"evidence":x["evidence"],"confidence":max(0,min(1,float(x["confidence"]))),"model":args.model}
   write(sp,[done[i] for i in sorted(done)]);print(f"state {axis}: {len(done)}/{len(texts)}",flush=True)
  cp=args.out_dir/f"{axis}_changes.jsonl";cdone={x["transition"]:x for x in read(cp)} if cp.exists() else {}
  tids=list(range(1,len(texts)));random.Random(args.seed+100+SLOTS.index(axis)).shuffle(tids)
  for b in range(0,len(tids),args.batch*2):
   todo=[i for i in tids[b:b+args.batch*2] if i not in cdone]
   if not todo:continue
   blocks=[]
   for i in todo:
    a,z=done[i-1],done[i];blocks.append(f"TRANSITION {i}\nBEFORE: observed={a['observed']}; state={a['state']}\nAFTER: observed={z['observed']}; state={z['state']}")
   user=f"""Compare ONLY two extracted {axis} states per transition. You cannot see original text or document boundaries.
value=1 only when both states are observed and the axis state genuinely changes incompatibly. If either is unknown, comparable=false and value=0. Paraphrases/coreferent descriptions are continuity. Return exactly one result per transition.\n\n"""+"\n\n".join(blocks)
   got=call(c,args.model,f"compare_{axis}","You compare one structured narrative axis conservatively.",user,schema_compare());gm={int(x["transition"]):x for x in got}
   unexpected=set(gm)-set(todo)
   if unexpected:raise RuntimeError(f"{axis} unexpected comparison ids: {sorted(unexpected)}")
   if not gm:raise RuntimeError(f"{axis} returned no requested comparison ids")
   for i,x in gm.items():cdone[i]={"doc_id":rows[i-1]["doc_id"],"transition":i,"axis":axis,"comparable":bool(x["comparable"]),"suggested_value":int(x["value"]),"reason":x["reason"],"model":args.model,"method":"absolute-left-only-state-compare"}
   write(cp,[cdone[i] for i in sorted(cdone)]);print(f"change {axis}: {len(cdone)}/{len(texts)-1}",flush=True)
if __name__=="__main__":main()
