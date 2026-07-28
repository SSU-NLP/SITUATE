#!/usr/bin/env python3
"""Translate LLM silver explanations to Korean without changing judgments/evidence."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from dotenv import load_dotenv

def read(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("silver_dir",type=Path);ap.add_argument("--model",default="openrouter/openai/gpt-4o-mini");args=ap.parse_args()
 load_dotenv(Path(__file__).resolve().parents[2]/".env");from openai import OpenAI
 c=OpenAI(api_key=os.environ["OPENAI_API_KEY"],base_url=os.environ["OPENAI_BASE_URL"],default_headers={"CF-Access-Client-Id":os.environ["CF_ACCESS_CLIENT_ID"],"CF-Access-Client-Secret":os.environ["CF_ACCESS_CLIENT_SECRET"]})
 for p in sorted(args.silver_dir.glob("*.jsonl")):
  rows=read(p)
  for lo in range(0,len(rows),40):
   batch=[r for r in rows[lo:lo+40] if not r.get("explanation_ko")]
   if not batch: continue
   payload="\n".join(f"[{r['transition']}] {r['explanation']}" for r in batch)
   resp=c.chat.completions.create(model=args.model,temperature=0,max_tokens=4000,response_format={"type":"json_object"},messages=[{"role":"system","content":"Translate each indexed explanation into clear, friendly Korean for a human annotation reviewer. Preserve the exact meaning; do not add judgments. Return JSON {\"translations\":{\"transition_id\":\"Korean\"}}."},{"role":"user","content":payload}])
   tr=json.loads(resp.choices[0].message.content)["translations"]
   for r in batch:r["explanation_ko"]=str(tr[str(r["transition"])]).strip()
   p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
   print(f"{p.stem}: {min(lo+40,len(rows))}/{len(rows)}",flush=True)
if __name__=="__main__":main()
