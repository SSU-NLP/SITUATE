import numpy as np
import pytest
from pathlib import Path
from situate.inference import __file__ as inference_file
LEGACY=Path(inference_file).parent / "checkpoints/pretrained_i2.pt"
from situate.inference import segment_embeddings,load_model,adaptive_boundaries
from situate.labels import targets
from situate.training import train

def test_pretrained_causality():
    z=np.random.default_rng(17).normal(size=(7,768)).astype('float32')
    full=segment_embeddings(z,checkpoint=LEGACY)
    assert full['profile']=='pretrained-i2'
    assert len(full['axis_probs'])==6
    for n in (2,4,6):
        r=segment_embeddings(z[:n],checkpoint=LEGACY)
        np.testing.assert_allclose(r['axis_probs'],full['axis_probs'][:n-1],atol=1e-6)
        assert r['boundaries']==[b for b in full['boundaries'] if b<n]
    assert segment_embeddings(np.empty((0,768)))['segments']==[]
    assert adaptive_boundaries([1,1,1])==[2,3]

def test_label_profiles():
    states=dict(protagonist=[dict(perspective='third',protagonist_present=True)]*4,
      space=[dict(location=x) for x in ['a','unspecified','b','b']],
      time=[dict(temporal_shift_marker='absent')]*4,
      causality=[dict(event_type='dialogue',aspect='ongoing')]*4,
      intentionality=[dict(active_goal=x) for x in ['a','none','b','b']])
    r,s=targets(states,'paper');i,_=targets(states,'pretrained-i2')
    assert r[:,1].tolist()==[0,1,0]
    assert r[:,4].tolist()==[0,1,0]
    states['intentionality']=[dict(active_goal=x) for x in ['none','a','b','none']]
    r,_=targets(states,'paper');i,_=targets(states,'pretrained-i2')
    assert r[:,4].tolist()==[0,1,0] and i[:,4].tolist()==[1,0,0]

def test_training_reload_and_split_guard(tmp_path):
    rng=np.random.default_rng(2)
    for split in ('train','val'):
        d=tmp_path/split;d.mkdir()
        np.savez(d/'doc.npz',embeddings=rng.normal(size=(4,768)).astype('float32'),
          resets=np.zeros((3,5),dtype='int64'),states=np.zeros((4,3),dtype='int64'),group=split,profile='paper',target_schema='protagonist_present-v1')
    target=tmp_path/'model.pt'
    history=train(tmp_path/'train',tmp_path/'val',target,epochs=1,batch_size=1)
    model,profile=load_model(target)
    assert profile=='paper' and model.pred_space=='slot' and np.isfinite(history[0]['validation'])
    r=segment_embeddings(rng.normal(size=(3,768)),checkpoint=target)
    assert r['profile']=='paper'
    with pytest.raises(ValueError,match='overlap'):
        train(tmp_path/'train',tmp_path/'train',tmp_path/'bad.pt',epochs=1)


def test_paper_equation9_and_no_observation_decoder():
    import torch
    from situate.paper_model import PaperSITUATE
    torch.manual_seed(71)
    model=PaperSITUATE(dropout=0.)
    z=torch.nn.functional.normalize(torch.randn(2,4,768),dim=-1)
    out=model.forward_batched(z,torch.zeros(2,3,5,dtype=torch.long),
        torch.zeros(2,4,3,dtype=torch.long),torch.ones(2,4),torch.ones(2,3))
    S=out['S']
    expected=sum((model.forward_models[k](S[:,:-1,k])-S[:,1:,k]).square().sum(-1) for k in range(5)).mean()
    torch.testing.assert_close(out['l_pred'],expected)
    assert not hasattr(model,'readout')
    assert model.state_heads['0'][-1].out_features==2
    out['loss'].backward()
    assert all(p.grad is not None for p in model.parameters())
    assert adaptive_boundaries([1.],min_transitions=1)==[1]


def test_paper_auxiliary_target_is_presence():
    states=dict(protagonist=[dict(perspective='third',protagonist_present=v) for v in [True,False]],
        space=[dict(location='unspecified')]*2,time=[dict(temporal_shift_marker='absent')]*2,
        causality=[dict(event_type='dialogue',aspect='ongoing')]*2,
        intentionality=[dict(active_goal='none')]*2)
    r,s=targets(states,'paper')
    assert s[:,0].tolist()==[1,0] and r[0,0]==1


def test_narrative_metrics_and_undefined_controls():
    from situate.narrative_evaluation import metrics
    result=metrics([0,1,0,0],[0,1,0,0])
    assert result['score']==1 and result['bor']==1 and result['lift']==4
    result=metrics([0,1,0,0],[0,0,0,0])
    assert result['bor'] is None and result['lift'] is None
