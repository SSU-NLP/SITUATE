#!/usr/bin/env python3
"""P0-S3: target-only LLM states; deterministic anchor-overlap changes."""
from __future__ import annotations
import argparse,hashlib,json,os,random,re,time
from pathlib import Path
from dotenv import load_dotenv

AXES=("protagonist","place","time","causality","goal")
DEFS={
"protagonist":"Who is centrally acting or experiencing in this sentence. Identity anchors should combine perspective with grounded role/identity when present (e.g. first-person narrator|IT-company newcomer), not merely 'I'.",
"place":"Where the action in this sentence is physically or narratively situated. Use explicit place/stage anchors only.",
"time":"The temporal frame expressed in this sentence: explicit time expressions plus grammatical tense/aspect anchors.",
"causality":"The causal state expressed in this sentence: agent/action and explicit cause or consequence anchors. Do not infer a wider story.",
"goal":"The intention expressed in this sentence: agent/intent/object anchors. If no intention is expressed or strongly entailed by this sentence alone, mark unobserved.",
}
VERSION="p0s3-target-only-v1"
def read(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
def write(p,rows):Path(p).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
def sequence(rows):
 rows=sorted(rows,key=lambda x:int(x["transition"]));return [rows[0]["previous"],*[x["current"] for x in rows]]
def norm(s):return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
def schema():
 item={"type":"object","additionalProperties":False,"properties":{"item_id":{"type":"integer"},"observed":{"type":"boolean"},"anchors":{"type":"array","items":{"type":"string"}},"summary":{"type":"string"},"evidence":{"type":"string"},"confidence":{"type":"number"}},"required":["item_id","observed","anchors","summary","evidence","confidence"]}
 return {"type":"object","additionalProperties":False,"properties":{"items":{"type":"array","items":item}},"required":["items"]}
def call(c,model,axis,items):
 payload="\n\n".join(f"ITEM {i}\nTARGET: {s}" for i,s in items)
 prompt=f"""Extract the {axis} state expressed by each TARGET sentence independently.
Axis definition: {DEFS[axis]}
Items are shuffled and unrelated. Never compare items. Do not reason about documents, stories, transitions, continuity, novelty, changes, resets, or boundaries.
Use TARGET only. anchors must be short canonical lowercase identity/state phrases grounded in TARGET, not generic filler. evidence must be an exact contiguous quote from TARGET.
If TARGET does not support this axis, observed=false, anchors=[], summary='unknown', evidence=''. Do not guess from world knowledge.
Return one result for every item_id.\n\n{payload}"""
 for a in range(3):
  try:
   r=c.chat.completions.create(model=model,temperature=0,max_tokens=10000,response_format={"type":"json_schema","json_schema":{"name":f"{axis}_target_states","strict":True,"schema":schema()}},messages=[{"role":"system","content":"You extract grounded state from one sentence only. You never compare sentences."},{"role":"user","content":prompt}]);return json.loads(r.choices[0].message.content)["items"],prompt
  except Exception:
   if a==2:raise
   time.sleep(2*(a+1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("master",type=Path);ap.add_argument("out_dir",type=Path);ap.add_argument("--model",default="openrouter/anthropic/claude-fable-5");ap.add_argument("--batch",type=int,default=20);ap.add_argument("--seed",type=int,default=20260711);ap.add_argument("--axis",choices=AXES,action="append");args=ap.parse_args()
 load_dotenv(Path(__file__).resolve().parents[2]/".env");from openai import OpenAI
 c=OpenAI(api_key=os.environ["OPENAI_API_KEY"],base_url=os.environ["OPENAI_BASE_URL"],default_headers={"CF-Access-Client-Id":os.environ["CF_ACCESS_CLIENT_ID"],"CF-Access-Client-Secret":os.environ["CF_ACCESS_CLIENT_SECRET"]})
 rows=read(args.master);texts=sequence(rows);args.out_dir.mkdir(parents=True,exist_ok=True)
 for axis in args.axis or AXES:
  sp=args.out_dir/f"{axis}_states.jsonl";done={x["item_id"]:x for x in read(sp)} if sp.exists() else {}
  ids=list(range(len(texts)));random.Random(args.seed+AXES.index(axis)).shuffle(ids)
  rounds=0
  while len(done)<len(texts):
   pending=[i for i in ids if i not in done];todo=pending[:args.batch]
   got,prompt=call(c,args.model,axis,[(i,texts[i]) for i in todo]);gm={int(x["item_id"]):x for x in got if int(x["item_id"]) in set(todo)}
   accepted=0
   for i,x in gm.items():
    observed=bool(x["observed"]);ev=str(x["evidence"]);anchors=sorted(set(norm(a) for a in x["anchors"] if norm(a)))
    grounded=(not observed) or (bool(ev) and norm(ev) in norm(texts[i]) and bool(anchors))
    if not grounded:continue
    done[i]={"item_id":i,"axis":axis,"observed":observed,"anchors":anchors if observed else [],"summary":x["summary"] if observed else "unknown","evidence":ev if observed else "","confidence":max(0,min(1,float(x["confidence"]))),"model":args.model,"prompt_version":VERSION,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"shuffle_seed":args.seed+AXES.index(axis)};accepted+=1
   if not accepted:raise RuntimeError(f"{axis}: no grounded outputs accepted for {todo}")
   write(sp,[done[i] for i in sorted(done)]);rounds+=1;print(f"state {axis}: {len(done)}/{len(texts)}",flush=True)
  changes=[]
  for i in range(1,len(texts)):
   a,b=done[i-1],done[i];comparable=a["observed"] and b["observed"]
   overlap=sorted(set(a["anchors"])&set(b["anchors"])) if comparable else []
   value=int(comparable and not overlap)
   changes.append({"doc_id":rows[i-1]["doc_id"],"transition":i,"axis":axis,"comparable":comparable,"suggested_value":value,"before_anchors":a["anchors"],"after_anchors":b["anchors"],"overlap":overlap,"method":"deterministic-exact-anchor-overlap","prompt_version":VERSION})
  write(args.out_dir/f"{axis}_changes.jsonl",changes);print(f"change {axis}: {len(changes)}/{len(texts)-1}",flush=True)
if __name__=="__main__":main()
