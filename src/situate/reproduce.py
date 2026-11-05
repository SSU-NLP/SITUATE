"""Four-seed paper run. Refuses substitute corpus sizes or legacy targets."""
import json
from pathlib import Path
import numpy as np
from .training import train,load_dataset
from .narrative_evaluation import evaluate

def reproduce(train_dir,val_dir,evaluation_manifest,output_dir,device='cuda'):
    tr=load_dataset(train_dir,'paper');va=load_dataset(val_dir,'paper')
    if (len(tr),len(va))!=(10430,1406):
        raise ValueError(f'Paper requires 10,430 train / 1,406 validation documents; found {len(tr)} / {len(va)}')
    docs=json.loads(Path(evaluation_manifest).read_text())
    if len(docs)!=6 or len({d['id'] for d in docs})!=6:
        raise ValueError('Paper evaluation requires all six independently identified narratives')
    del tr,va
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    # Seed identities and unspecified hyperparameters are implementation choices,
    # not values explicitly reported by the paper.
    plan=dict(seeds=[0,1,2,3],epochs=100,patience=3,batch_size=48,lr=.0003,
        profile='paper',architecture='paper-v1',train_documents=10430,validation_documents=1406,
        unreported_choices=['seed identities','batch size','alpha','beta','gamma','variance floor','dropout','MLP hidden widths'])
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    scores=[]
    for seed in plan['seeds']:
        checkpoint=output/f'paper-seed{seed}.pt'
        train(train_dir,val_dir,checkpoint,profile='paper',epochs=100,patience=3,batch_size=48,seed=seed,device=device)
        result=evaluate(evaluation_manifest,checkpoint)
        (output/f'evaluation-seed{seed}.json').write_text(json.dumps(result,indent=2))
        scores.append(result['macro']['situate']['score'])
    result=dict(scores=scores,mean=float(np.mean(scores)),std=float(np.std(scores)),
        parameter_count=705887,reported_paper_parameter_count_approx=1430000,
        downstream_longbench='not run by this command')
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    return result
