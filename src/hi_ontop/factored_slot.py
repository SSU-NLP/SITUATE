"""M5-FactoredSlot — Phase 2 slot module (EntNet typed keyed slots + anchor loss).

Aligns K=5 typed slots to the EISM axes (protagonist/space/time/causality/intentionality) on FROZEN sentence
embeddings, supervised by M3 (S5) anchor labels. NO dynamics/switching yet (Phase 2 de-risk only).

Anchor loss (codex 2026-07-15):  L = Σ_k α_k [L_Δ,k + β L_state,k] + γ L_collapse
  - L_Δ,k  : primary — per-axis RESET (change) BCE on the slot's OWN transition (head locality).
  - L_state: aux — per-sentence global-enum state CE, from the slot alone (perspective/temporal/event_type).
  - L_collapse : per-slot temporal variance floor (no inter-slot orthogonality — would kill real EISM corr).
  - readout heads see ONLY their own slot (D3 anchored latent; anchor = target, never inference input).
M4-MonoAnchor control = same heads reading a SINGLE shared latent (mono=True) → cannot separate axes.

Design template: scripts/idm_lam/lam.py (nn.Module + forward()->{"loss",...} + dz change-input idiom).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

AXES = ["protagonist", "space", "time", "causality", "intentionality"]
# axis_idx -> (state field, enum size) for the aux absolute-state head (global-enum axes only;
# space/intentionality are story-local -> no absolute head in Phase 2, rely on reset supervision).
STATE_AXES = {0: ("perspective", 4), 2: ("temporal_shift_marker", 4), 3: ("event_type", 10)}
ALPHA = (1.0, 0.5, 1.0, 0.4, 0.25)  # per-axis weight (protagonist/space/time/causality/intentionality)


def mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class FactoredSlot(nn.Module):
    """
    `dynamics` 인자 (2026-07-28 추가) — method-registry §3 의 명세를 구현한 것이며 **새 메커니즘을
    발명한 것이 아니다.** registry 공통 전제: *"학습 신호 = 자기지도(next-state 예측) + anchor"*,
    그리고 (dynamics 열) M4 = 1개 · **M5 = 축별 5개** · M6 = 공유 1개.

        dynamics=None  : Phase 2 de-risk 그대로 (anchor-only). **기본값 = 기존 수치 완전 재현.**
        "per_slot"     : 슬롯마다 별도 forward model f_k  -> **M5 사양**
        "shared"       : 모든 슬롯이 forward model 하나를 공유 -> **M6 사양**
        (mono=True 면 슬롯이 1개이므로 어느 쪽이든 f 1개 = **M4 사양**)

    dynamics 를 켜면 두 가지가 생긴다 — 이것이 EISM/SEM 의 "기대 → 오차":
      ① **기대 생성**: f_k 가 ŝ_{k,t+1} 을 예측하고, readout g 가 그 예측 슬롯들로부터 다음 문장
         임베딩 **ẑ_{t+1}** 을 만든다 (아직 보지 않은 입력에 대한 기대).
      ② **오차 측정**: `L_pred = ‖ẑ_{t+1} − z_{t+1}‖²` — 라벨 없는 자기지도. forward() 가
         per-transition 예측오차 `pred_err` 를 함께 돌려주므로 분절 신호로 쓸 수 있다.
    ⚠️ **③ 경계에서의 상태 리셋(event switching)은 여전히 미구현이다.** registry §3 의 M5 사양에
    switching 이 없으므로 넣지 않았다(사양 밖 메커니즘 도입 금지 — CLAUDE.md SEM 계승 원칙).
    """

    def __init__(self, z_dim: int = 768, slot_dim: int = 32, mono: bool = False,
                 alpha: tuple = ALPHA, beta: float = 0.25, gamma: float = 0.05,
                 var_floor: float = 0.1, pos_weight: tuple | None = None,
                 d1: bool = True, bottleneck: int = 8, dropout: float = 0.4,
                 dynamics: str | None = None, pred_weight: float = 1.0,
                 pred_space: str = "obs"):
        super().__init__()
        self.mono = mono
        self.dynamics = dynamics
        self.pred_weight = pred_weight
        # §4.6 Axis2 — 예측/비교를 어느 공간에서 하나
        #   "obs"  : 공용 768D 의미공간 (기존 기본값). ⚠️ §Risks 의 "semantic-space prediction
        #            re-importing IDM" 에 해당. 인과 churn 이 전 슬롯으로 새어나간다.
        #   "slot" : **slot-native** — f_k 가 자기 슬롯의 다음 상태를 직접 예측(§4.6 D3 [v1]).
        #            "causal churn is quarantined to the causal slot".
        #   "both" : 둘 다 (등가중)
        if pred_space not in ("obs", "slot", "both"):
            raise ValueError(f"pred_space must be obs/slot/both, got {pred_space!r}")
        self.pred_space = pred_space
        self.d1 = d1 and not mono   # D1 = per-axis extractor routing (codex 2026-07-15); mono uses full input
        self.n_slots = 1 if mono else 5
        self.slot_dim = slot_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.var_floor = var_floor
        self.register_buffer("pos_weight",
                             torch.tensor(pos_weight if pos_weight else [1.0] * 5, dtype=torch.float32))
        # input path: D1 = per-axis low-dim extractor E_k (768->32->GELU->Dropout->bottleneck), routed to
        # slot k ONLY (separates the INFORMATION path, not just parameters). Mono = one full-capacity proj.
        if self.d1:
            self.extractors = nn.ModuleList([
                nn.Sequential(nn.Linear(z_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, bottleneck))
                for _ in range(5)])
            in_dim = bottleneck
        else:
            self.proj = nn.Linear(z_dim, slot_dim)
            in_dim = slot_dim
        # EntNet keyed slots: distinct learnable keys break slot exchangeability (Henaff 2017)
        self.keys = nn.Parameter(torch.randn(self.n_slots, slot_dim) * 0.1)
        # EntNet gated-write params (shared across slots); C maps input path -> slot_dim
        self.A = nn.Linear(slot_dim, slot_dim, bias=False)
        self.B = nn.Linear(slot_dim, slot_dim, bias=False)
        self.C = nn.Linear(in_dim, slot_dim, bias=False)
        self.gate_in = nn.Linear(in_dim, slot_dim, bias=False)  # project input to slot_dim for the gate
        self.phi = nn.PReLU(slot_dim)
        # change-readout head per AXIS (5), input d_Δ = [s_{t-1}, s_t, Δ, s_t⊙s_{t-1}] = 4*slot_dim
        self.change_heads = nn.ModuleList([mlp([4 * slot_dim, slot_dim, 1]) for _ in range(5)])
        # aux absolute-state heads (global-enum axes only)
        self.state_heads = nn.ModuleDict(
            {str(a): mlp([slot_dim, slot_dim, sz]) for a, (_, sz) in STATE_AXES.items()})
        # ── dynamics (registry §3): forward model + next-observation readout ────────
        if dynamics is not None:
            if dynamics not in ("per_slot", "shared"):
                raise ValueError(f"dynamics must be None/'per_slot'/'shared', got {dynamics!r}")
            n_f = 1 if (mono or dynamics == "shared") else 5
            self.forward_models = nn.ModuleList(
                [mlp([slot_dim, slot_dim, slot_dim]) for _ in range(n_f)])
            # 예측된 슬롯들 -> 다음 문장 임베딩 (M2-MonoLAM 의 FDM 과 같은 역할)
            self.readout = mlp([self.n_slots * slot_dim, slot_dim * 4, z_dim])

    def _f(self, k: int) -> nn.Module:
        """슬롯 k 의 forward model. per_slot 이면 f_k, shared/mono 면 f_0."""
        return self.forward_models[0 if len(self.forward_models) == 1 else k]

    def predict_next(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """S [..., T, n_slots, slot_dim] -> (Ŝ_next, ẑ_next), 둘 다 시점 t 에서 t+1 을 향한 기대.

        반환 길이는 T-1 (마지막 시점은 대응할 실제값이 없다).
        """
        Sh = torch.stack([self._f(k)(S[..., :-1, k, :]) for k in range(self.n_slots)], dim=-2)
        z_hat = self.readout(Sh.flatten(start_dim=-2))
        return Sh, z_hat

    def slot_for_axis(self, k: int) -> int:
        return 0 if self.mono else k

    def encode_seq(self, z: torch.Tensor) -> torch.Tensor:
        """z [T, z_dim] frozen embeddings -> slot states S [T, n_slots, slot_dim] (EntNet recurrence).

        D1: slot k is fed ONLY its per-axis extractor E_k(z) (separated input path). Mono: full proj(z).
        """
        if self.d1:
            U = torch.stack([self.extractors[k](z) for k in range(5)], dim=1)  # [T, 5, bottleneck]
        else:
            U = self.proj(z).unsqueeze(1)  # [T, 1, slot_dim]
        h = self.keys.clone()  # [n_slots, slot_dim]; memory initialized to keys (EntNet)
        out = []
        for t in range(z.shape[0]):
            ut = U[t]                                  # [n_slots, in_dim]
            gi = self.gate_in(ut)                      # [n_slots, slot_dim]
            g = torch.sigmoid((gi * h).sum(-1) + (gi * self.keys).sum(-1))   # [n_slots] gate
            cand = self.phi(self.A(h) + self.B(self.keys) + self.C(ut))      # [n_slots, slot_dim]
            h = h + g.unsqueeze(-1) * cand
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)                    # normalize memory
            out.append(h)
        return torch.stack(out)  # [T, n_slots, slot_dim]

    def encode_seq_batched(self, z: torch.Tensor) -> torch.Tensor:
        """배치판 — z [B, T, z_dim] -> S [B, T, n_slots, slot_dim].

        Phase 1 스케일(10,598편)용. 시간축 순차 루프는 그대로 두되 **문서를 배치로 묶어**
        에폭당 순차 스텝 수를 (총 문장 수)에서 (문서당 최대 길이 × 배치 수)로 줄인다.
        수식은 `encode_seq` 와 **동일**하며 배치 차원만 추가됐다 — B=1 이면 수치가 일치한다
        (`tests/test_factored_slot_batched.py` 에서 검증).

        ⚠️ 패딩 처리는 하지 않는다. 호출부가 길이 내림차순으로 정렬해 **뒤쪽에만 패딩**을 두고,
        패딩 구간 출력은 손실 마스크로 제외해야 한다(패딩은 뒤에만 있으므로 유효 구간의 h 를
        오염시키지 않는다).
        """
        B, T = z.shape[0], z.shape[1]
        if self.d1:
            U = torch.stack([self.extractors[k](z) for k in range(5)], dim=2)  # [B,T,5,bottleneck]
        else:
            U = self.proj(z).unsqueeze(2)  # [B,T,1,slot_dim]
        h = self.keys.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, n_slots, slot_dim]
        keys = self.keys.unsqueeze(0)  # [1, n_slots, slot_dim] — broadcast
        out = []
        for t in range(T):
            ut = U[:, t]                               # [B, n_slots, in_dim]
            gi = self.gate_in(ut)                      # [B, n_slots, slot_dim]
            g = torch.sigmoid((gi * h).sum(-1) + (gi * keys).sum(-1))        # [B, n_slots]
            pre = self.A(h) + self.B(keys) + self.C(ut)                      # [B, n_slots, slot_dim]
            # PReLU(num_parameters=slot_dim)는 2D 입력의 dim=1 을 채널로 본다 -> 평탄화 후 적용.
            cand = self.phi(pre.reshape(-1, pre.shape[-1])).view_as(pre)
            h = h + g.unsqueeze(-1) * cand
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
            out.append(h)
        return torch.stack(out, dim=1)  # [B, T, n_slots, slot_dim]

    def forward_batched(self, z: torch.Tensor, resets: torch.Tensor, states: torch.Tensor,
                        sent_mask: torch.Tensor, trans_mask: torch.Tensor) -> dict:
        """배치판 forward. 패딩 구간은 손실에서 완전히 제외한다.

        z [B,T,z_dim] · resets [B,T-1,5] · states [B,T,3] (STATE_AXES 순서: 0,2,3) ·
        sent_mask [B,T] float(1=유효) · trans_mask [B,T-1] float(1=유효).
        반환은 `forward` 와 동형 {loss, change_logits, S, l_collapse}.
        """
        S = self.encode_seq_batched(z)                 # [B,T,n_slots,slot_dim]
        loss = z.new_zeros(())
        change_logits = {}
        tm_sum = trans_mask.sum().clamp(min=1.0)
        for k in range(5):
            sk = S[:, :, self.slot_for_axis(k), :]     # [B,T,slot_dim] — head sees ONLY this slot
            a, b = sk[:, :-1], sk[:, 1:]
            d = torch.cat([a, b, b - a, b * a], dim=-1)          # [B,T-1,4*slot_dim]
            logit = self.change_heads[k](d).squeeze(-1)          # [B,T-1]
            change_logits[AXES[k]] = logit
            l = F.binary_cross_entropy_with_logits(
                logit, resets[:, :, k].float(), pos_weight=self.pos_weight[k], reduction="none")
            loss = loss + self.alpha[k] * (l * trans_mask).sum() / tm_sum
        # aux absolute-state heads (전역 enum 축만) — states 열 순서 = STATE_AXES 키 순서
        sm_sum = sent_mask.sum().clamp(min=1.0)
        for col, a in enumerate(STATE_AXES):
            sk = S[:, :, self.slot_for_axis(a), :]               # [B,T,slot_dim]
            logits = self.state_heads[str(a)](sk)                # [B,T,n_enum]
            l_state = F.cross_entropy(logits.transpose(1, 2), states[:, :, col], reduction="none")
            loss = loss + self.alpha[a] * self.beta * (l_state * sent_mask).sum() / sm_sum
        # collapse: 유효 구간만으로 시간축 표준편차 (패딩 제외)
        m = sent_mask.unsqueeze(-1).unsqueeze(-1)                # [B,T,1,1]
        n = m.sum(dim=1).clamp(min=2.0)                          # [B,1,1]
        mean = (S * m).sum(dim=1) / n                            # [B,n_slots,slot_dim]
        var = ((S - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / (n - 1).clamp(min=1.0)
        std = var.clamp(min=1e-12).sqrt()
        l_collapse = (F.relu(self.var_floor - std) ** 2).mean()
        loss = loss + self.gamma * l_collapse
        out = {"loss": loss, "change_logits": change_logits, "S": S, "l_collapse": l_collapse}
        # ── 자기지도 next-state 예측 (registry §3) ─────────────────────────────
        if self.dynamics is not None:
            Sh, z_hat = self.predict_next(S)                     # 기대 (슬롯 / 관측 공간)
            terms = []
            if self.pred_space in ("obs", "both"):
                terms.append(((z_hat - z[:, 1:]) ** 2).sum(-1))              # [B,T-1]
            if self.pred_space in ("slot", "both"):
                # §4.6 Axis2 — 슬롯별 자기 공간 예측오차를 슬롯에 대해 평균(축별 가중 없음)
                terms.append(((Sh - S[:, 1:]) ** 2).sum(-1).mean(-1))        # [B,T-1]
            pred_err = sum(terms) / len(terms)
            l_pred = (pred_err * trans_mask).sum() / tm_sum
            out["loss"] = loss + self.pred_weight * l_pred
            out["l_pred"] = l_pred
            out["pred_err"] = pred_err
        return out

    def forward(self, z: torch.Tensor, resets: torch.Tensor, states: dict,
                mask: torch.Tensor | None = None, axis_mask: tuple | None = None) -> dict:
        """z [T,z_dim]; resets [T-1,5] int; states {field:[T] int}. Returns {loss, change_logits, S}.

        mask [T-1,5] float: 0 = 그 전이의 그 축 라벨 유보 -> 손실에서 제외 (SHARE place/time 미정렬 구간).
        axis_mask (5,) bool: False인 축은 손실에서 통째 제외 (라벨이 구조적으로 무의미한 축).
        둘 다 None(기본)이면 기존 경로와 수치 동일 -> 기존 체크포인트/REPORT 재현성 보존.
        """
        S = self.encode_seq(z)
        loss = z.new_zeros(())
        change_logits = {}
        for k in range(5):
            sk = S[:, self.slot_for_axis(k), :]  # [T, slot_dim] — head sees ONLY this slot
            d = torch.cat([sk[:-1], sk[1:], sk[1:] - sk[:-1], sk[1:] * sk[:-1]], dim=-1)  # [T-1,4d]
            logit = self.change_heads[k](d).squeeze(-1)  # [T-1]
            change_logits[AXES[k]] = logit
            if axis_mask is not None and not axis_mask[k]:
                continue
            y = resets[:, k].float()
            l = F.binary_cross_entropy_with_logits(logit, y, pos_weight=self.pos_weight[k],
                                                   reduction="none")
            if mask is None:
                l_delta = l.mean()
            else:
                m = mask[:, k]
                l_delta = (l * m).sum() / m.sum().clamp(min=1.0)
            loss = loss + self.alpha[k] * l_delta
        # aux absolute-state heads (global-enum axes)
        for a, (field, _) in STATE_AXES.items():
            sk = S[:, self.slot_for_axis(a), :]  # [T, slot_dim]
            logits = self.state_heads[str(a)](sk)  # [T, n_enum]
            l_state = F.cross_entropy(logits, states[field])
            loss = loss + self.alpha[a] * self.beta * l_state
        # collapse: per-slot temporal std floor (no inter-slot independence reg)
        std = S.std(dim=0)  # [n_slots, slot_dim]
        l_collapse = (F.relu(self.var_floor - std) ** 2).mean()
        loss = loss + self.gamma * l_collapse
        out = {"loss": loss, "change_logits": change_logits, "S": S, "l_collapse": l_collapse}
        if self.dynamics is not None:
            Sh, z_hat = self.predict_next(S)
            terms = []
            if self.pred_space in ("obs", "both"):
                terms.append(((z_hat - z[1:]) ** 2).sum(-1))                 # [T-1]
            if self.pred_space in ("slot", "both"):
                terms.append(((Sh - S[1:]) ** 2).sum(-1).mean(-1))           # [T-1]
            pred_err = sum(terms) / len(terms)
            l_pred = pred_err.mean()
            out["loss"] = loss + self.pred_weight * l_pred
            out["l_pred"] = l_pred
            out["pred_err"] = pred_err
        return out
