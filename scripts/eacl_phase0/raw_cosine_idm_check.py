import json, sys
import numpy as np
sys.path.insert(0,'src')
from hi_ontop.embedding import make_encoder
enc=make_encoder(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
AX=['protagonist_perspective','place','time','causality','goal']
def auc(y,s):
    pos=[a for yy,a in zip(y,s) if yy]; neg=[a for yy,a in zip(y,s) if not yy]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
def seq(rows):
    rows=sorted(rows,key=lambda r:r['transition']); return [rows[0]['previous']]+[r['current'] for r in rows]
def raw_idm_for(master_path):
    rows=[json.loads(l) for l in open(master_path)]
    bydoc={}
    for r in rows: bydoc.setdefault(r['doc_id'],[]).append(r)
    out={}
    for doc,rs in bydoc.items():
        rs=sorted(rs,key=lambda r:r['transition'])
        E=np.asarray(enc.encode(seq(rs)))
        En=E/ (np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
        for r in rs:
            i=r['transition']
            out[(doc,i)]=float(1-np.dot(En[i],En[i-1]))
    return out
cfg={'pieman':'outputs/eacl_phase0/pieman_michelmann_master.jsonl',
     'tunnel':'outputs/eacl_phase0/tunnel_michelmann_master.jsonl',
     'monkey':'outputs/eacl_phase0/monkey_michelmann_master.jsonl',
     'tiage':'outputs/eacl_phase0/dts_tiage_master.jsonl'}
rmpath={'pieman':'pieman_michelmann_s5','tunnel':'tunnel_michelmann_s5','monkey':'monkey_michelmann_s5','tiage':'dts_tiage_s5'}
print(f"{'dataset':8s} {'type':12s} | {'ours':>6s} {'IDM_tuned':>9s} {'IDM_raw':>8s}")
for name in ['pieman','tunnel','monkey','tiage']:
    raw=raw_idm_for(cfg[name])
    M=[json.loads(l) for l in open(f'outputs/eacl_phase0/{rmpath[name]}/reset_matrix.jsonl')]
    y=[r['boundary'] for r in M]; cnt=[sum(r[a] for a in AX) for r in M]
    idmt=[r['idm_score'] for r in M]; idmr=[raw[(r['doc_id'],r['transition'])] for r in M]
    typ={'pieman':'fiction(dev)','tunnel':'fiction','monkey':'journalism','tiage':'dialogue'}[name]
    print(f"{name:8s} {typ:12s} | {auc(y,cnt):6.3f} {auc(y,idmt):9.3f} {auc(y,idmr):8.3f}")
