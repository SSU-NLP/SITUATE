import json, sys
sys.path.insert(0,'src')
from hi_ontop.dts_scoring import score_dialogues
AX=['protagonist_perspective','place','time','causality','goal']
def rm(doc): return sorted([json.loads(l) for l in open(f'outputs/eacl_phase0/{doc}_michelmann_s5/reset_matrix.jsonl')], key=lambda r:r['transition'])
def auc(y,s):
    pos=[a for yy,a in zip(y,s) if yy]; neg=[a for yy,a in zip(y,s) if not yy]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
def endturn(n,bt): 
    y=[0]*n; 
    for i in bt: y[i-1]=1
    return y
def eval_doc(doc, t=3):
    M=rm(doc); n=len(M)+1
    y=[r['boundary'] for r in M]; cnt={r['transition']:sum(r[a] for a in AX) for r in M}; idm={r['transition']:r['idm_score'] for r in M}
    gold=endturn(n,[r['transition'] for r in M if r['boundary']]); K=sum(y)
    # free-threshold t*=3
    ours_bt=[i for i in cnt if cnt[i]>=t]; ours_free=score_dialogues([gold],[endturn(n,ours_bt)])['score']
    # IDM free (argmax over thresholds)
    idm_free=max((score_dialogues([gold],[endturn(n,[i for i in idm if idm[i]>=thr])])['score'] for thr in set(idm.values()) if any(idm[i]>=thr for i in idm)))
    # matched N=gold
    tx=list(cnt)
    o_top=sorted(tx,key=lambda i:(cnt[i],-i),reverse=True)[:K]; d_top=sorted(idm,key=lambda i:idm[i],reverse=True)[:K]
    o_m=score_dialogues([gold],[endturn(n,o_top)])['score']; d_m=score_dialogues([gold],[endturn(n,d_top)])['score']
    return dict(n=n-1,K=K,ours_free=ours_free,idm_free=idm_free,ours_matched=o_m,idm_matched=d_m,
                ours_auc=auc(y,[cnt[r['transition']] for r in M]),idm_auc=auc(y,[idm[r['transition']] for r in M]))
docs={d:eval_doc(d) for d in ['pieman','tunnel','monkey']}
print(f"{'doc':8s} {'N/K':>8s} | {'free ours/IDM':>16s} | {'matched@K ours/IDM':>20s} | {'AUC ours/IDM':>15s}")
for d,r in docs.items():
    print(f"{d:8s} {r['n']:>4}/{r['K']:<3} | {r['ours_free']:.3f}/{r['idm_free']:.3f} ({r['ours_free']-r['idm_free']:+.3f}) | "
          f"{r['ours_matched']:.3f}/{r['idm_matched']:.3f} ({r['ours_matched']-r['idm_matched']:+.3f}) | {r['ours_auc']:.3f}/{r['idm_auc']:.3f} ({r['ours_auc']-r['idm_auc']:+.3f})")
def grp(name,ds):
    import statistics as st
    for lens in ['free','matched','auc']:
        o=st.mean(docs[d][f'ours_{lens}'] for d in ds); i=st.mean(docs[d][f'idm_{lens}'] for d in ds)
        print(f"  [{name}] {lens:8s} ours {o:.3f}  IDM {i:.3f}  gain {o-i:+.3f}")
print("\n=== 장르 그룹 (per-doc mean) ===")
grp("narrative-fiction (Pieman+Tunnel)", ['pieman','tunnel'])
grp("journalism (Monkey)", ['monkey'])
