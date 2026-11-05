"""Independent Adam training on explicitly disjoint train/validation NPZ directories."""
from pathlib import Path
import json, random, hashlib
import numpy as np
import torch
from .factored_slot import FactoredSlot

def load_dataset(directory, profile):
    if profile != 'pretrained-i2':
        raise ValueError('Paper candidates are isolated on the paper-candidate branch')
    docs = []
    for path in sorted(Path(directory).glob('*.npz')):
        with np.load(path, allow_pickle=False) as f:
            if str(f['profile']) != profile:
                raise ValueError(f'{path}: label profile mismatch')
            z=np.asarray(f['embeddings'],np.float32); r=np.asarray(f['resets'],np.int64)
            s=np.asarray(f['states'],np.int64)
            if z.ndim!=2 or z.shape[1]!=768 or len(z)<2 or r.shape!=(len(z)-1,5) or s.shape!=(len(z),3):
                raise ValueError(f'{path}: invalid dimensions')
            if not np.isfinite(z).all() or not np.isin(r,[0,1]).all():
                raise ValueError(f'{path}: invalid values')
            norm=np.linalg.norm(z,axis=1,keepdims=True)
            if (norm<=1e-12).any() or any(((s[:,i]<0)|(s[:,i]>=k)).any() for i,k in enumerate((4,4,10))):
                raise ValueError(f'{path}: invalid vectors or state labels')
            docs.append(dict(z=z/norm,r=r,s=s,group=str(f['group']),
                digest=hashlib.sha256(z.tobytes()).hexdigest(),path=str(path)))
    if not docs: raise ValueError(f'No training documents in {directory}')
    return docs

def batch(docs, device):
    b=len(docs); t=max(len(d['z']) for d in docs)
    z=torch.zeros(b,t,768,device=device);r=torch.zeros(b,t-1,5,dtype=torch.long,device=device)
    s=torch.zeros(b,t,3,dtype=torch.long,device=device);sm=torch.zeros(b,t,device=device);tm=torch.zeros(b,t-1,device=device)
    for i,d in enumerate(docs):
        n=len(d['z']);z[i,:n]=torch.tensor(d['z'],device=device);r[i,:n-1]=torch.tensor(d['r'],device=device)
        s[i,:n]=torch.tensor(d['s'],device=device);sm[i,:n]=1;tm[i,:n-1]=1
    return z,r,s,sm,tm

def train(train_dir,val_dir,output,profile='pretrained-i2',epochs=100,patience=3,batch_size=48,seed=0,device='cpu'):
    if epochs<1 or patience<1 or batch_size<1: raise ValueError('Training counts must be positive')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    tr=load_dataset(train_dir,profile);va=load_dataset(val_dir,profile)
    for key in ('group','digest'):
        if {d[key] for d in tr}&{d[key] for d in va}: raise ValueError(f'Train/validation overlap: {key}')
    counts=np.concatenate([d['r'] for d in tr]); pos=counts.sum(0)
    weights=np.clip((len(counts)-pos)/np.maximum(pos,1),1,30).tolist()
    kw=dict(slot_dim=128,bottleneck=8,dropout=.4,dynamics='per_slot',pred_space='obs',
            pred_weight=1.,pos_weight=weights)
    model=FactoredSlot(**kw).to(device); opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    best=float('inf');bad=0;history=[]
    for epoch in range(epochs):
        model.train();random.shuffle(tr);losses=[]
        for i in range(0,len(tr),batch_size):
            opt.zero_grad();loss=model.forward_batched(*batch(tr[i:i+batch_size],device))['loss']
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vals=[float(model.forward_batched(*batch([d],device))['loss']) for d in va]
        value=float(np.mean(vals));history.append(dict(epoch=epoch+1,train=float(np.mean(losses)),validation=value))
        if value<best:
            best=value;bad=0
            torch.save(dict(state=model.state_dict(),model_kwargs=kw,profile=profile,seed=seed,epoch=epoch+1,
                            train_documents=len(tr),validation_documents=len(va)),output)
        else:bad+=1
        if bad>=patience:break
    output.with_suffix('.training.json').write_text(json.dumps(history,indent=2))
    return history
