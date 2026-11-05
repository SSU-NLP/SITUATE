"""Three-stage annotation: closed vocabulary, independently shuffled axes, code targets."""
import hashlib,json,os,random
from pathlib import Path
from . import annotation_prompts as p

def annotate(input_path,output_path,model,base_url,seed=20260711):
    from openai import OpenAI
    target=Path(output_path)
    if target.exists():raise FileExistsError(target)
    data=json.loads(Path(input_path).read_text());texts=data['sentences']
    if len(texts)<2 or any(not isinstance(x,str) or not x.strip() for x in texts):raise ValueError('Invalid sentences')
    client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url=base_url)
    extra={'chat_template_kwargs':{'enable_thinking':False}}
    lists,prompt=p.stage1(client,model,'\n'.join(texts),8000,extra,'v3.1')
    hashes=[hashlib.sha256(prompt.encode()).hexdigest()];states={}
    for index,axis in enumerate(p.AXES):
        ids=list(range(len(texts)));random.Random(seed+index).shuffle(ids);done={}
        for offset in range(0,len(ids),20):
            batch=ids[offset:offset+20]
            rows,prompt=p.stage2_call(client,model,axis,lists,[(i,texts[i]) for i in batch],extra,'v3.1')
            hashes.append(hashlib.sha256(prompt.encode()).hexdigest())
            for row in rows:
                i=int(row['item_id']);value=p.valid(axis,row,lists)
                if i not in batch or i in done or value is None:raise ValueError('Invalid or duplicate annotation')
                done[i]=value
            if any(i not in done for i in batch):raise ValueError('Missing annotation')
        states[axis]=[done[i] for i in range(len(texts))]
    result=dict(sentences=texts,group=data['group'],states=states,annotation=dict(model=model,
        prompt_version='v3.1',seed=seed,temperature=0,thinking=False,prompt_sha256=hashes,lists=lists))
    with target.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return {'saved':str(target),'sentences':len(texts)}
