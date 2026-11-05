"""Operation and memory-pointer heads with optional persistent FactoredSlot state."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hi_ontop.factored_slot import FactoredSlot

OPS = ("ADD", "UPDATE", "DELETE", "NONE")


class UpdatePolicy(nn.Module):
    def __init__(self, z_dim=768, width=128, slot_dim=32, stateful=True):
        super().__init__()
        self.spec = dict(z_dim=z_dim, width=width, slot_dim=slot_dim, stateful=stateful)
        self.stateful = stateful
        source = FactoredSlot(z_dim=z_dim, slot_dim=slot_dim, dropout=0,
                              dynamics=None, beta=0, gamma=0)
        if stateful:
            # Reuse the existing recurrence; segmentation/auxiliary heads are not trained.
            self.keys = source.keys
            for name in ("extractors", "A", "B", "C", "gate_in", "phi"):
                setattr(self, name, getattr(source, name))
            self.state_proj = nn.Linear(5 * slot_dim, width)
        else:
            # Match trainable capacity to within one hidden unit (<1% overall).
            budget = source.keys.numel() + sum(p.numel() for name in
                ('extractors', 'A', 'B', 'C', 'gate_in', 'phi') for p in getattr(source, name).parameters())
            budget += 5 * slot_dim * width + width
            hidden = max(1, round((budget - width) / (3 * width + 1)))
            self.local_proj = nn.Sequential(nn.Linear(2 * width, hidden), nn.GELU(),
                                            nn.Linear(hidden, width))
        self.fact_proj = nn.Linear(z_dim, width)
        self.mem_proj = nn.Linear(z_dim, width)
        self.context = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.decoder = nn.GRUCell(2 * width, width)
        self.op_embedding = nn.Embedding(5, width)  # four operations and start
        self.op_head = nn.Linear(width, 4)
        self.pointer = nn.Linear(width, width, bias=False)
        self.null_memory = nn.Parameter(torch.zeros(width))

    def observe(self, turns, state=None):
        """Consume each new timestamped turn once; caller owns conversation state."""
        if not self.stateful:
            return None
        h = self.keys.clone() if state is None else state
        for z in turns:
            u = torch.stack([layer(z) for layer in self.extractors])
            gi = self.gate_in(u)
            gate = torch.sigmoid((gi * h).sum(-1) + (gi * self.keys).sum(-1))
            candidate = self.phi(self.A(h) + self.B(self.keys) + self.C(u))
            h = h + gate[:, None] * candidate
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        return h

    def prepare(self, facts, memories, state):
        f = self.fact_proj(facts)
        m = self.mem_proj(memories)
        fmean = f.mean(0) if len(f) else self.null_memory * 0
        mmean = m.mean(0) if len(m) else self.null_memory * 0
        context = (self.state_proj(state.flatten()) if self.stateful else
                   self.local_proj(torch.cat([fmean, mmean])))
        h = self.context(torch.cat([fmean, mmean, context]))
        candidates = torch.cat([self.null_memory[None], m], dim=0)
        return h, candidates

    def step(self, h, candidates, previous_op, previous_target):
        token = torch.cat([self.op_embedding.weight[previous_op], candidates[previous_target]])
        h = self.decoder(token, h)
        return h, self.op_head(h), candidates @ self.pointer(h) / h.shape[-1] ** 0.5

    def loss(self, facts, memories, state, actions):
        """NONE/null terminates a variable-length program; no auxiliary losses."""
        h, candidates = self.prepare(facts, memories, state)
        previous_op, previous_target = 4, 0
        op_losses, target_losses = [], []
        for operation, target in [*actions, (3, 0)]:
            h, logits, pointers = self.step(h, candidates, previous_op, previous_target)
            op_losses.append(F.cross_entropy(logits[None], logits.new_tensor([operation], dtype=torch.long)))
            # ADD has no old-memory target. NONE/null is the explicit stop decision.
            if operation != 0:
                target_losses.append(F.cross_entropy(pointers[None], pointers.new_tensor([target], dtype=torch.long)))
            previous_op, previous_target = operation, target
        op_loss = torch.stack(op_losses).mean()
        target_loss = torch.stack(target_losses).mean() if target_losses else op_loss * 0
        return op_loss + target_loss, op_loss.detach(), target_loss.detach()

    @torch.no_grad()
    def decide(self, facts, memories, state, max_actions=128):
        h, candidates = self.prepare(facts, memories, state)
        previous_op, previous_target = 4, 0
        result, deleted = [], set()
        for _ in range(max_actions):
            h, logits, pointers = self.step(h, candidates, previous_op, previous_target)
            if len(candidates) == 1 or len(deleted) == len(memories):
                logits[1:3] = -torch.inf
            operation = int(logits.argmax())
            if operation == 0:
                target = 0
            else:
                pointers = pointers.clone()
                for k in deleted:
                    pointers[k] = -torch.inf
                if operation in (1, 2):
                    pointers[0] = -torch.inf
                target = int(pointers.argmax())
            if operation == 3 and target == 0:
                return result
            result.append((operation, target))
            if operation == 2:
                deleted.add(target)
            previous_op, previous_target = operation, target
        raise RuntimeError("Policy did not terminate; refusing a silently truncated program")


def labels(actions, memories):
    positions = {str(m["id"]): i + 1 for i, m in enumerate(memories)}
    result, deleted = [], set()
    for action in actions:
        op = OPS.index(action["event"])
        target = 0 if op == 0 else positions[str(action["id"])]
        if target and target in deleted:
            raise ValueError("Target was already deleted in the program")
        if op == 2:
            deleted.add(target)
        result.append((op, target))
    return result
