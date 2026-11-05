"""Standalone causal inference. The bundled checkpoint retains its original settings."""
from pathlib import Path
import numpy as np
import torch
from .factored_slot import FactoredSlot, AXES
from .paper_model import PaperSITUATE

ENCODER = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
ENCODER_REVISION = '4328cf26390c98c5e3c738b4460a05b95f4911f5'
DEFAULT_CHECKPOINT = Path(__file__).parent / 'checkpoints/paper.pt'

def load_model(checkpoint=None, device='cpu'):
    path=Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError('Paper-trained checkpoint is not available yet. Supply --checkpoint explicitly; legacy weights are not a paper reproduction.')
    ck = torch.load(path, map_location=device, weights_only=True)
    if 'model_kwargs' in ck:
        kw = ck['model_kwargs']
    else:
        a = ck['args']
        if ck['mono'] or a.get('tagger') not in (None, 'none'):
            raise ValueError('Expected a five-axis, embedding-input SITUATE checkpoint')
        kw = dict(slot_dim=a['slot_dim'], mono=False, pos_weight=ck['pos_weight'],
                  alpha=a.get('_alpha', (1., .5, 1., .4, .25)), dynamics=ck['dynamics'],
                  pred_space=a.get('pred_space', 'obs'), pred_weight=a.get('pred_weight', 1.),
                  bottleneck=a.get('bottleneck', 8), dropout=a.get('dropout', .4))
    if ck.get('profile')=='paper' and ck.get('architecture')!='paper-v1':
        raise ValueError('Superseded paper profile: rebuild targets and retrain with paper-v1')
    m = (PaperSITUATE(**kw) if ck.get('architecture')=='paper-v1' else FactoredSlot(**kw)).to(device)
    m.load_state_dict(ck['state'], strict=True)
    return m.eval(), ck.get('profile', 'pretrained-i2')

def adaptive_boundaries(scores, c=1., min_transitions=2):
    """Current-inclusive prefix mean/population std; no look-ahead or gold count."""
    if not np.isfinite(c) or c < 0 or min_transitions not in (1, 2):
        raise ValueError('Invalid threshold setting')
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('scores must be finite and one dimensional')
    count = 0; total = 0.; square = 0.; boundaries = []
    for i, value in enumerate(values):
        count += 1; total += value; square += value * value
        mean = total/count; std = max(0., square/count-mean*mean)**.5
        if count >= min_transitions and value >= mean+c*std:
            boundaries.append(i+1)
    return boundaries

@torch.no_grad()
def segment_embeddings(embeddings, checkpoint=None, device='cpu', c=1.):
    z = np.asarray(embeddings, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 768 or not np.isfinite(z).all():
        raise ValueError('Expected finite embeddings with shape [sentences, 768]')
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('Zero embeddings are invalid')
    n = len(z)
    if n < 2:
        return dict(boundaries=[], segments=[[0, n]] if n else [], scores=[], axis_probs=[])
    m, profile = load_model(checkpoint, device)
    E = torch.tensor(z/norms, device=device).unsqueeze(0)
    o = m.forward_batched(E, torch.zeros(1,n-1,5,dtype=torch.long,device=device),
        torch.zeros(1,n,3,dtype=torch.long,device=device), torch.ones(1,n,device=device),
        torch.ones(1,n-1,device=device))
    probs = torch.stack([o['change_logits'][a].sigmoid() for a in AXES], -1)[0].cpu().numpy()
    scores = probs.sum(1)
    boundaries = adaptive_boundaries(scores, c, min_transitions=1 if profile == 'paper' else 2)
    cuts = [0]+boundaries+[n]
    return dict(boundaries=boundaries,segments=[list(x) for x in zip(cuts[:-1],cuts[1:])],
                scores=scores.tolist(),axis_probs=probs.tolist(),profile=profile)

def segment(sentences, checkpoint=None, device='cpu', c=1.):
    if not sentences:
        return dict(boundaries=[],segments=[],scores=[],axis_probs=[])
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER, device=device, revision=ENCODER_REVISION)
    z = encoder.encode(sentences, normalize_embeddings=True)
    return segment_embeddings(z, checkpoint, device, c)
