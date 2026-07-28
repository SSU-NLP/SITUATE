import json, sys
import numpy as np
sys.path.insert(0,'src')
from hi_ontop.dts_scoring import score_dialogues
from hi_ontop.embedding import make_encoder
enc=make_encoder(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
AX=['protagonist_perspective','place','time','causality','goal']
def endturn(n,idxs):
    y=[0]*n
    for i in idxs: y[i-1]=1
    return y
def raw_idm(rows):
    seq=[rows[0]['previous']]+[r['current'] for r in rows]
    E=np.asarray(enc.encode(seq)); En=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
    return {rows[k]['transition']: float(1-np.dot(En[k+1],En[k])) for k in range(len(rows))}
def eval_dataset(master_path, rm_path, name):
    master=[json.loads(l) for l in open(master_path)]
    rm={(r['doc_id'],r['transition']):r for r in (json.loads(l) for l in open(rm_path))}
    bydoc={}
    for r in master: bydoc.setdefault(r['doc_id'],[]).append(r)
    golds=[]; ours_p=[]; idmraw_p=[]; idmtuned_p=[]
    for doc,rows in bydoc.items():
        rows=sorted(rows,key=lambda r:r['transition'])
        if not all((doc,r['transition']) in rm for r in rows): continue
        n=len(rows)+1
        gt=[r['transition'] for r in rows if r['boundary']]; K=len(gt)
        if K==0: continue
        cnt={r['transition']:sum(rm[(doc,r['transition'])][a] for a in AX) for r in rows}
        idmr=raw_idm(rows); idmt={r['transition']:rm[(doc,r['transition'])]['idm_score'] for r in rows}
        tx=[r['transition'] for r in rows]
        golds.append(endturn(n,gt))
        ours_p.append(endturn(n,sorted(tx,key=lambda i:(cnt[i],-i),reverse=True)[:K]))
        idmraw_p.append(endturn(n,sorted(tx,key=lambda i:idmr[i],reverse=True)[:K]))
        idmtuned_p.append(endturn(n,sorted(tx,key=lambda i:idmt[i],reverse=True)[:K]))
    o=score_dialogues(golds,ours_p)['score']; ir=score_dialogues(golds,idmraw_p)['score']; it=score_dialogues(golds,idmtuned_p)['score']
    print(f'{name:22s} matched@gold Score  ours {o:.3f}  IDM_raw {ir:.3f}  IDM_tuned {it:.3f}')
print("=== Score 기준 crossover (matched@gold, 공정 렌즈) ===")
eval_dataset('outputs/eacl_phase0/tunnel_michelmann_master.jsonl','outputs/eacl_phase0/tunnel_michelmann_s5/reset_matrix.jsonl','tunnel (fiction)')
eval_dataset('outputs/eacl_phase0/dts_tiage_master.jsonl','outputs/eacl_phase0/dts_tiage_s5/reset_matrix.jsonl','tiage (dialogue)')
