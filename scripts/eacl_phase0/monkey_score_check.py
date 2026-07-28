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
def eval_gold(master_path, rm_path, gold_master, label):
    master=[json.loads(l) for l in open(master_path)]
    rm={(r['doc_id'],r['transition']):r for r in (json.loads(l) for l in open(rm_path))}
    gb={r['transition']:int(r['boundary']) for r in (json.loads(l) for l in open(gold_master))}
    rows=sorted(master,key=lambda r:r['transition'])
    n=len(rows)+1
    gt=[r['transition'] for r in rows if gb.get(r['transition'],0)]; K=len(gt)
    cnt={r['transition']:sum(rm[(r['doc_id'],r['transition'])][a] for a in AX) for r in rows}
    idmr=raw_idm(rows); idmt={r['transition']:rm[(r['doc_id'],r['transition'])]['idm_score'] for r in rows}
    tx=[r['transition'] for r in rows]
    gold=[endturn(n,gt)]
    o=score_dialogues(gold,[endturn(n,sorted(tx,key=lambda i:(cnt[i],-i),reverse=True)[:K])])['score']
    ir=score_dialogues(gold,[endturn(n,sorted(tx,key=lambda i:idmr[i],reverse=True)[:K])])['score']
    it=score_dialogues(gold,[endturn(n,sorted(tx,key=lambda i:idmt[i],reverse=True)[:K])])['score']
    print(f'monkey {label:14s} (K={K})  matched@gold Score  ours {o:.3f}  IDM_raw {ir:.3f}  IDM_tuned {it:.3f}')
MM='outputs/eacl_phase0/monkey_michelmann_master.jsonl'; RM='outputs/eacl_phase0/monkey_michelmann_s5/reset_matrix.jsonl'
eval_gold(MM,RM,MM,'goldA(filtered)')
eval_gold(MM,RM,'outputs/eacl_phase0/monkey_michelmann_raw_master.jsonl','goldB(raw+5s)')
