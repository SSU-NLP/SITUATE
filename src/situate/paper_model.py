"""SITUATE equations (1)–(11), without the legacy observation decoder.

Widths/regularization unspecified by the paper retain explicitly recorded legacy
values. The protagonist auxiliary target retains the original four narrative perspectives.
"""
import torch
from torch import nn
from torch.nn import functional as F

AXES = ('protagonist', 'space', 'time', 'causality', 'intentionality')
TARGET_SCHEMA = 'perspective-v1'

def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a,b), nn.GELU(), nn.Linear(b,c))

class PaperSITUATE(nn.Module):
    def __init__(self, z_dim=768, slot_dim=128, bottleneck=8, dropout=.4,
                 alpha=(1.,.5,1.,.4,.25), beta=.25, gamma=.05,
                 var_floor=.1, pred_weight=1.):
        super().__init__()
        self.alpha=tuple(alpha);self.beta=beta;self.gamma=gamma
        self.var_floor=var_floor;self.pred_weight=pred_weight
        self.pred_space='slot';self.slot_dim=slot_dim
        self.extractors=nn.ModuleList([nn.Sequential(nn.Linear(z_dim,32),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(32,bottleneck)) for _ in AXES])
        self.keys=nn.Parameter(torch.randn(5,slot_dim)*.1)
        self.A=nn.Linear(slot_dim,slot_dim,bias=False)
        self.B=nn.Linear(slot_dim,slot_dim,bias=False)
        self.C=nn.Linear(bottleneck,slot_dim,bias=False)
        self.gate_in=nn.Linear(bottleneck,slot_dim,bias=False)
        self.phi=nn.PReLU(slot_dim)
        self.change_heads=nn.ModuleList([mlp(4*slot_dim,slot_dim,1) for _ in AXES])
        self.state_heads=nn.ModuleDict({str(k):mlp(slot_dim,slot_dim,size)
                                       for k,size in ((0,4),(2,4),(3,10))})
        self.forward_models=nn.ModuleList([mlp(slot_dim,slot_dim,slot_dim) for _ in AXES])

    def encode(self,z):
        u=torch.stack([extractor(z) for extractor in self.extractors],dim=2)
        h=self.keys.unsqueeze(0).expand(len(z),-1,-1)
        sequence=[]
        for t in range(z.shape[1]):
            g_in=self.gate_in(u[:,t])
            g=torch.sigmoid((g_in*h).sum(-1)+(g_in*self.keys).sum(-1))
            pre=self.A(h)+self.B(self.keys)+self.C(u[:,t])
            candidate=self.phi(pre.reshape(-1,self.slot_dim)).reshape_as(pre)
            h=F.normalize(h+g.unsqueeze(-1)*candidate,p=2,dim=-1,eps=1e-8)
            sequence.append(h)
        return torch.stack(sequence,dim=1)

    def forward_batched(self,z,resets,states,sent_mask,trans_mask):
        S=self.encode(z);a,b=S[:,:-1],S[:,1:]
        features=torch.cat((a,b,b-a,a*b),dim=-1)
        logits={axis:self.change_heads[k](features[:,:,k]).squeeze(-1)
                for k,axis in enumerate(AXES)}
        tm=trans_mask.sum().clamp(min=1);sm=sent_mask.sum().clamp(min=1)
        delta=z.new_zeros(());absolute=z.new_zeros(())
        for k,axis in enumerate(AXES):
            per_transition=F.binary_cross_entropy_with_logits(logits[axis],resets[:,:,k].float(),reduction='none')
            delta=delta+self.alpha[k]*(per_transition*trans_mask).sum()/tm
        for column,k in enumerate((0,2,3)):
            per_sentence=F.cross_entropy(self.state_heads[str(k)](S[:,:,k]).transpose(1,2),states[:,:,column],reduction='none')
            absolute=absolute+self.alpha[k]*(per_sentence*sent_mask).sum()/sm
        predicted=torch.stack([self.forward_models[k](a[:,:,k]) for k in range(5)],dim=2)
        # Eq. 9: sum over latent coordinates AND five axes, mean over transitions.
        errors=(predicted-b).square().sum(dim=(-1,-2))
        prediction=(errors*trans_mask).sum()/tm
        mask=sent_mask[:,:,None,None];n=mask.sum(dim=1)
        mean=(S*mask).sum(dim=1)/n.clamp(min=1)
        variance=((S-mean[:,None]).square()*mask).sum(dim=1)/(n-1).clamp(min=1)
        collapse=F.relu(self.var_floor-variance.clamp(min=1e-12).sqrt()).square().mean()
        loss=delta+self.beta*absolute+self.gamma*collapse+self.pred_weight*prediction
        return dict(loss=loss,change_logits=logits,S=S,l_delta=delta,l_state=absolute,
                    l_collapse=collapse,l_pred=prediction,pred_err=errors)
