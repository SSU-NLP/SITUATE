"""Paper narrative metrics; input gold is binary consensus on n-1 transitions.

The terminal sentinel is appended only for Score, never for BOR or Lift.
No gold count is used for the model's threshold or density-matched controls.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from .inference import segment_embeddings
from .metrics import score_dialogues

def metrics(gold,pred):
    g=np.asarray(gold);p=np.asarray(pred)
    if g.ndim!=1 or p.shape!=g.shape or not len(g) or not np.isin(g,[0,1]).all() or not np.isin(p,[0,1]).all():
        raise ValueError('Aligned binary transition vectors required')
    g=g.astype(int);p=p.astype(int)
    out=score_dialogues([g.tolist()+[0]],[p.tolist()+[0]])
    out.update(bor=float(p.sum()/g.sum()) if g.sum() and p.sum() else None,
        lift=float(g[p==1].mean()/g.mean()) if g.sum() and p.sum() else None,
        predicted=int(p.sum()),reference=int(g.sum()))
    return out

def evaluate(manifest,checkpoint,seed=20260731):
    path=Path(manifest);documents=json.loads(path.read_text())
    if not documents or len({d['id'] for d in documents})!=len(documents):
        raise ValueError('Unique nonempty document IDs required')
    rng=np.random.default_rng(seed);results=[]
    for doc in documents:
        ep=path.parent/doc['embeddings'];z=np.load(ep,allow_pickle=False)
        if len(doc['gold'])!=len(z)-1:raise ValueError('Expected one consensus label per transition')
        result=segment_embeddings(z,checkpoint=checkpoint)
        if result.get('profile')!='paper':raise ValueError('Paper evaluation requires newly trained paper-v1 weights')
        n=len(z)-1;p=np.zeros(n,dtype=int)
        p[np.asarray(result['boundaries'],dtype=int)-1]=1;k=int(p.sum())
        regular=np.zeros(n,dtype=int);random=np.zeros(n,dtype=int)
        if k:
            regular[np.floor((np.arange(k)+.5)*n/k).astype(int)]=1
            random[rng.choice(n,k,replace=False)]=1
        results.append(dict(id=doc['id'],embedding_sha256=hashlib.sha256(ep.read_bytes()).hexdigest(),
            boundaries=result['boundaries'],situate=metrics(doc['gold'],p),
            fixed_interval=metrics(doc['gold'],regular),random=metrics(doc['gold'],random),
            no_boundary=metrics(doc['gold'],np.zeros(n,dtype=int))))
    macro={}
    for method in ('situate','fixed_interval','random','no_boundary'):
        macro[method]={key:float(np.mean([r[method][key] for r in results if r[method][key] is not None]))
                       if any(r[method][key] is not None for r in results) else None
                       for key in ('score','bor','lift')}
    return dict(checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),control_seed=seed,documents=results,macro=macro)
