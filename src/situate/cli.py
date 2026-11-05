import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description='SITUATE five-axis narrative segmentation')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('segment');q.add_argument('input');q.add_argument('--embeddings',action='store_true')
    q.add_argument('--checkpoint');q.add_argument('--device',default='cpu');q.add_argument('--c',type=float,default=1.)
    q=sub.add_parser('prepare');q.add_argument('input',help='JSON: embeddings, axis states, group')
    q.add_argument('output');q.add_argument('--profile',choices=['paper','pretrained-i2'],default='paper')
    q=sub.add_parser('train');q.add_argument('train_dir');q.add_argument('val_dir');q.add_argument('output')
    q.add_argument('--profile',choices=['paper','pretrained-i2'],default='paper');q.add_argument('--epochs',type=int,default=100)
    q.add_argument('--patience',type=int,default=3);q.add_argument('--batch-size',type=int,default=48)
    q.add_argument('--seed',type=int,default=0);q.add_argument('--device',default='cpu')
    q=sub.add_parser('annotate');q.add_argument('input');q.add_argument('output');q.add_argument('--model',required=True)
    q.add_argument('--tokenizer',required=True);q.add_argument('--base-url',required=True);q.add_argument('--seed',type=int,default=20260711)
    q=sub.add_parser('evaluate');q.add_argument('input')
    q=sub.add_parser('evaluate-narratives');q.add_argument('input');q.add_argument('--checkpoint',required=True)
    q=sub.add_parser('reproduce');q.add_argument('train_dir');q.add_argument('val_dir');q.add_argument('evaluation_manifest');q.add_argument('output_dir');q.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.command=='segment':
        from .inference import segment,segment_embeddings
        kw=dict(checkpoint=a.checkpoint,device=a.device,c=a.c)
        out=segment_embeddings(np.load(a.input,allow_pickle=False),**kw) if a.embeddings else segment(Path(a.input).read_text().splitlines(),**kw)
    elif a.command=='prepare':
        from .labels import targets
        data=json.loads(Path(a.input).read_text());r,s=targets(data['states'],a.profile)
        if 'embeddings' not in data:
            from sentence_transformers import SentenceTransformer
            from .inference import ENCODER, ENCODER_REVISION
            data['embeddings']=SentenceTransformer(ENCODER,device='cpu',revision=ENCODER_REVISION).encode(data['sentences'],normalize_embeddings=True)
        z=np.asarray(data['embeddings'],np.float32)
        if z.shape!=(len(s),768):raise ValueError('Embedding/state length or dimension mismatch')
        with open(a.output,'xb') as f:np.savez_compressed(f,embeddings=z,resets=r,states=s,group=data['group'],profile=a.profile,target_schema='protagonist_present-v1' if a.profile=='paper' else 'perspective-v1')
        out={'saved':a.output}
    elif a.command=='train':
        from .training import train
        out=train(a.train_dir,a.val_dir,a.output,a.profile,a.epochs,a.patience,a.batch_size,a.seed,a.device)
    elif a.command=='annotate':
        from .annotation import annotate
        out=annotate(a.input,a.output,a.model,a.base_url,a.seed,a.tokenizer)
    elif a.command=='reproduce':
        from .reproduce import reproduce
        out=reproduce(a.train_dir,a.val_dir,a.evaluation_manifest,a.output_dir,a.device)
    elif a.command=='evaluate-narratives':
        from .narrative_evaluation import evaluate
        out=evaluate(a.input,a.checkpoint)
    else:
        from .metrics import score_dialogues
        data=json.loads(Path(a.input).read_text());out=score_dialogues(data['golds'],data['preds'])
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
