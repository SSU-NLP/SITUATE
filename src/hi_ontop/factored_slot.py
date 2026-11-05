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

    `d1` 인자 — **입력 경로**의 분리 여부. ⚠️ 두 가지를 혼동하지 말 것:
      d1=True  : 축별 저차원 extractor E_k(768→32→GELU→Dropout→bottleneck)가 슬롯 k **에만**
                 들어간다. 정보 경로 자체가 분리된다 (현행 기본값, codex 2026-07-15).
      d1=False : **공유 투영 proj(z) 한 벡터가 5슬롯 전부에 그대로 브로드캐스트**된다.
                 (블록으로 쪼개 나눠주는 것이 **아니다** — 모든 슬롯이 같은 입력을 본다.)
                 슬롯은 각자의 key 와 자기 상태(EntNet 재귀)로만 갈라진다 ⇒ "표현은 하나의
                 벡터, 스키마 구조는 슬롯 상태·dynamics 에만" = SEM 식 구성에 가까운 대조군.
                 폭은 `shared_dim`(None 이면 slot_dim). d1 라우팅의 총 입력폭 5×bottleneck 과
                 맞추려면 shared_dim=5*bottleneck 을 준다.
    ⚠️ 설계문서 `factored-slot-eism-v0_1.md` §4.6 의 **D1**(실제 NER/시간태거로 정의되는
       extractor-defined space)과 이 코드의 `d1`(학습형 축별 저차원 MLP)은 **다른 것**이다.
       코드의 d1 은 §4.6 분류로는 D2(learned projection π_k)에 가깝다. 이름만 D1 을 물려받았다.

    `tagger` 인자 (D-B35) — **설계문서 §4.6 의 진짜 D1**. 슬롯 k 의 입력을 spaCy NER/POS/morph +
    어휘 사전이 만든 **축별 8차원**으로 준다(`extract_axis_features.py`). 근거 = D-B34: M5−M4
    이득의 실질 전부가 입력경로에서 왔고, 현 경로는 축이 무엇인지 아무도 안 알려주는 학습형이다.
        None      : 태거 미사용 (d1/shared_proj 경로 그대로)
        "replace" : 입력 = 태거 피처 **그대로** (학습형 E_k 제거) — 문서의 D1 문자 그대로,
                    *"discovery 는 dynamics 에만 산다"*
        "hybrid"  : 입력 = [E_k(z), 태거 피처] concat — 태거가 **정보를 더하는가**를 따로 잰다
    `tagger_axes` : 태거로 대체할 축 수. **3** = protagonist/space/time 만(설계문서 §10:
        *"Causal/goal indices lack clean taggers"* — causality/intentionality 는 학습형 유지,
        hybrid 에서는 태거 자리를 0 으로 채워 폭을 맞춘다) · **5** = 전 축.
    """

    def __init__(self, z_dim: int = 768, slot_dim: int = 32, mono: bool = False,
                 alpha: tuple = ALPHA, beta: float = 0.25, gamma: float = 0.05,
                 var_floor: float = 0.1, pos_weight: tuple | None = None,
                 d1: bool = True, bottleneck: int = 8, dropout: float = 0.4,
                 shared_dim: int | None = None,
                 tagger: str | None = None, tagger_axes: int = 5, tagger_dim: int = 8,
                 block_readout: bool = False,
                 dynamics: str | None = None, pred_weight: float = 1.0,
                 pred_space: str = "obs", head_feat: str = "delta",
                 psi: str | None = None, psi_beta: float = 0.9, psi_dropout: float = 0.3,
                 slot_reset: str | None = None, slot_reset_dropout: float = 0.3):
        super().__init__()
        self.mono = mono
        self.dynamics = dynamics
        self.pred_weight = pred_weight
        # ── §4.6 Axis1 predict-then-compare (D-B29 P묶음) ──────────────────────
        # 설계 원문: *"compare the observation to what dynamics **predicted**, not to x_{t-1}"*
        # (make-or-break). 기존 `delta` 는 compare-to-previous 였다.
        # ⚠️ 누출 금지: ŝ_t = f_k(s_{t-1}) 는 **t-1 까지의 정보만** 쓴다. s_t 를 f 에 넣지 않는다.
        # ⚠️ 잔차를 **임계 규칙**으로 쓰는 건 이미 기각됐다(0.32~0.33). 여기서는 **지도 head 의
        #    입력 특징**으로만 쓴다 — 다른 실험이다(D-B29).
        #   delta        : [s_{t-1}, s_t, Δ, s_t⊙s_{t-1}]              (현행 = compare-to-previous)
        #   resid_signed : delta + (s_t − ŝ_t)                          부호 있는 예측 잔차
        #   resid_sq     : delta + (s_t − ŝ_t)²                         크기만
        #   pred_centric : [ŝ_t, s_t, s_t − ŝ_t, s_t⊙ŝ_t]              **순수** predict-then-compare
        if head_feat not in ("delta", "resid_signed", "resid_sq", "pred_centric"):
            raise ValueError(f"bad head_feat {head_feat!r}")
        if head_feat != "delta" and dynamics is None:
            raise ValueError("head_feat 가 잔차를 쓰려면 dynamics 가 있어야 한다 (ŝ 가 필요)")
        self.head_feat = head_feat
        # ── §4.3 에피소드 스키마 조건부 dynamics ψₑ (D-B29 Ψ묶음) ──────────────
        # 설계: sₜ^(k) = f_θ^(k)(s_{t-1}^(k); **ψₑ**). ψₑ = 현 에피소드의 슬롯상태 요약.
        #   None     : 조건화 없음 (현행)
        #   cummean  : 에피소드 내 누적평균     (O(1): 합·개수만)
        #   ema      : 에피소드 내 EMA(psi_beta) (O(1): 값 하나)
        # 학습 = 앵커 경계로 리셋하되 **teacher reset dropout** 로 일부 경계를 무시해
        # train–inference mismatch 를 줄인다. 추론 = 자기 예측 경계로 리셋(rollout).
        if psi not in (None, "cummean", "ema"):
            raise ValueError(f"bad psi {psi!r}")
        if psi is not None and dynamics is None:
            raise ValueError("psi 는 dynamics 와 함께만 의미가 있다")
        self.psi = psi
        self.psi_beta = psi_beta
        self.psi_dropout = psi_dropout
        # ── D-B63: 경계에서의 슬롯 상태 리셋(event switching) ────────────────────
        #   None = 현행(문서 전체 누적) · "hard" = 경계 다음 문장부터 메모리를 key 로 되돌린다.
        #   🔴 **리셋은 한 문장 지연**된다: 전이 j 의 경계 판정 자체가 s_{j+1} 을 필요로 하므로
        #   s_{j+1} 은 연속 갱신으로 만들고, 리셋은 s_{j+2} 부터 적용한다(인과적·비순환).
        if slot_reset not in (None, "hard"):
            raise ValueError(f"bad slot_reset {slot_reset!r}")
        self.slot_reset = slot_reset
        self.slot_reset_dropout = slot_reset_dropout
        # §4.6 Axis2 — 예측/비교를 어느 공간에서 하나
        #   "obs"  : 공용 768D 의미공간 (기존 기본값). ⚠️ §Risks 의 "semantic-space prediction
        #            re-importing IDM" 에 해당. 인과 churn 이 전 슬롯으로 새어나간다.
        #   "slot" : **slot-native** — f_k 가 자기 슬롯의 다음 상태를 직접 예측(§4.6 D3 [v1]).
        #            "causal churn is quarantined to the causal slot".
        #   "both" : 둘 다 (등가중)
        if pred_space not in ("obs", "slot", "both"):
            raise ValueError(f"pred_space must be obs/slot/both, got {pred_space!r}")
        self.pred_space = pred_space
        # ── D-B36: **하나의 상태 벡터 + 구조화된 스키마** (SEM 식 장면 표현) ──────
        # 상태도 재귀도 **하나**(M4 와 동일한 단일 EntNet 셀)이되, 그 벡터를 5블록으로 나눠
        # **head k 는 자기 블록만 읽는다**. 스키마 구조가 슬롯이 아니라 **벡터의 좌표 구획**에
        # 산다 — M5/공유입력(둘 다 슬롯마다 독립 게이트·독립 메모리)과는 다른 구성이다.
        # 앵커 손실이 "0~b-1 은 인물, b~2b-1 은 장소" 를 학습으로 새긴다.
        self.block_readout = block_readout
        if block_readout:
            if not mono:
                raise ValueError("block_readout 은 단일 상태 벡터 전용이다 (mono=True)")
            if slot_dim % 5:
                raise ValueError(f"block_readout 은 slot_dim 이 5의 배수여야 한다 (받은 값 {slot_dim})")
            self.block_dim = slot_dim // 5
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
        # ── D-B35 §4.6 D1: 태거가 정의한 축 공간 ────────────────────────────────
        if tagger not in (None, "replace", "hybrid"):
            raise ValueError(f"tagger must be None/'replace'/'hybrid', got {tagger!r}")
        if tagger is not None and mono:
            raise ValueError("tagger 경로는 5슬롯 전용이다 (mono 는 축 개념이 없다)")
        if tagger is not None and tagger_axes not in (3, 5):
            raise ValueError(f"tagger_axes must be 3 or 5, got {tagger_axes}")
        self.tagger = tagger
        self.tagger_axes = tagger_axes
        self.tagger_dim = tagger_dim
        if tagger == "replace":
            # 학습형 E_k 는 앞 tagger_axes 개 축에서 제거된다. 나머지 축은 E_k 유지(폭이 같아야
            # 하므로 bottleneck == tagger_dim 을 요구한다).
            if tagger_axes < 5 and bottleneck != tagger_dim:
                raise ValueError("tagger_axes=3 replace 는 bottleneck == tagger_dim 이어야 한다"
                                 f" (현재 {bottleneck} vs {tagger_dim})")
            if tagger_axes < 5:
                self.extractors = nn.ModuleList([
                    nn.Sequential(nn.Linear(z_dim, 32), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(32, bottleneck)) for _ in range(5)])
            in_dim = tagger_dim
        elif tagger == "hybrid":
            self.extractors = nn.ModuleList([
                nn.Sequential(nn.Linear(z_dim, 32), nn.GELU(), nn.Dropout(dropout),
                              nn.Linear(32, bottleneck)) for _ in range(5)])
            in_dim = bottleneck + tagger_dim
        elif self.d1:
            self.extractors = nn.ModuleList([
                nn.Sequential(nn.Linear(z_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, bottleneck))
                for _ in range(5)])
            in_dim = bottleneck
        else:
            # shared_dim=None 이면 slot_dim -> 기존 경로와 수치 동일(재현성 보존).
            self.shared_dim = shared_dim or slot_dim
            self.proj = nn.Linear(z_dim, self.shared_dim)
            in_dim = self.shared_dim
        # EntNet keyed slots: distinct learnable keys break slot exchangeability (Henaff 2017)
        self.keys = nn.Parameter(torch.randn(self.n_slots, slot_dim) * 0.1)
        # EntNet gated-write params (shared across slots); C maps input path -> slot_dim
        self.A = nn.Linear(slot_dim, slot_dim, bias=False)
        self.B = nn.Linear(slot_dim, slot_dim, bias=False)
        self.C = nn.Linear(in_dim, slot_dim, bias=False)
        self.gate_in = nn.Linear(in_dim, slot_dim, bias=False)  # project input to slot_dim for the gate
        self.phi = nn.PReLU(slot_dim)
        # change-readout head per AXIS (5). 입력 폭은 head_feat / psi 에 따라 달라진다.
        n_blk = 5 if head_feat in ("resid_signed", "resid_sq") else 4
        # head 가 실제로 읽는 폭 — block_readout 이면 자기 블록(slot_dim/5)만 본다.
        w = self.block_dim if block_readout else slot_dim
        self.head_width = w
        self.head_in = (n_blk + (1 if psi is not None else 0)) * w
        self.change_heads = nn.ModuleList([mlp([self.head_in, w, 1]) for _ in range(5)])
        # aux absolute-state heads (global-enum axes only)
        self.state_heads = nn.ModuleDict(
            {str(a): mlp([w, w, sz]) for a, (_, sz) in STATE_AXES.items()})
        # ── dynamics (registry §3): forward model + next-observation readout ────────
        if dynamics is not None:
            if dynamics not in ("per_slot", "shared"):
                raise ValueError(f"dynamics must be None/'per_slot'/'shared', got {dynamics!r}")
            n_f = 1 if (mono or dynamics == "shared") else 5
            f_in = slot_dim * (2 if psi is not None else 1)      # ψₑ 를 이어붙여 조건화
            self.forward_models = nn.ModuleList(
                [mlp([f_in, slot_dim, slot_dim]) for _ in range(n_f)])
            # 예측된 슬롯들 -> 다음 문장 임베딩 (M2-MonoLAM 의 FDM 과 같은 역할)
            self.readout = mlp([self.n_slots * slot_dim, slot_dim * 4, z_dim])

    def _f(self, k: int) -> nn.Module:
        """슬롯 k 의 forward model. per_slot 이면 f_k, shared/mono 면 f_0."""
        return self.forward_models[0 if len(self.forward_models) == 1 else k]

    def predict_next(self, S: torch.Tensor, Psi: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """S [..., T, n_slots, slot_dim] -> (Ŝ_next, ẑ_next), 둘 다 시점 t 에서 t+1 을 향한 기대.

        반환 길이는 T-1 (마지막 시점은 대응할 실제값이 없다).
        `Psi` 를 주면 §4.3 대로 `f_k(s_{t}; ψₑ)` 로 조건화한다 — `Psi[..., t, k, :]` 는
        **t 까지의 정보만** 담아야 한다(그렇지 않으면 누출).
        """
        def _x(k: int) -> torch.Tensor:
            x = S[..., :-1, k, :]
            return x if Psi is None else torch.cat([x, Psi[..., :-1, k, :]], dim=-1)
        Sh = torch.stack([self._f(k)(_x(k)) for k in range(self.n_slots)], dim=-2)
        z_hat = self.readout(Sh.flatten(start_dim=-2))
        return Sh, z_hat

    def psi_seq(self, S: torch.Tensor, bnd: torch.Tensor | None) -> torch.Tensor:
        """에피소드 스키마 ψₑ 시퀀스. S [B,T,n_slots,d] · bnd [B,T-1] (1 = 전이 j 가 경계).

        `Psi[:, t]` = **현 에피소드의 t 까지** 슬롯상태 요약 (t 포함, t+1 미포함) → 인과적.
        `bnd[:, t-1]` 이 1 이면 s_t 부터 새 에피소드이므로 그 시점에 요약을 비운다.
        비용 = 스텝당 상수(합·개수 또는 EMA 값 하나) → **온라인 O(1)**.
        """
        B, T = S.shape[0], S.shape[1]
        run = torch.zeros_like(S[:, 0])                       # [B,n_slots,d]
        cnt = S.new_zeros(B, 1, 1)
        out = []
        for t in range(T):
            keep = S.new_ones(B, 1, 1) if t == 0 else (1.0 - bnd[:, t - 1]).view(B, 1, 1)
            if self.psi == "ema":
                run = keep * (self.psi_beta * run + (1 - self.psi_beta) * S[:, t]) \
                    + (1.0 - keep) * S[:, t]
                out.append(run)
            else:                                              # cummean
                run = run * keep + S[:, t]
                cnt = cnt * keep + 1.0
                out.append(run / cnt)
        return torch.stack(out, dim=1)                         # [B,T,n_slots,d]

    def head_input(self, a: torch.Tensor, b: torch.Tensor, sh: torch.Tensor | None,
                   psi: torch.Tensor | None) -> torch.Tensor:
        """change head 입력. a=s_{t-1} · b=s_t · sh=ŝ_t(=f(s_{t-1})) · psi=ψ at t-1. 전부 [...,d]."""
        if self.head_feat == "pred_centric":
            parts = [sh, b, b - sh, b * sh]
        else:
            parts = [a, b, b - a, b * a]
            if self.head_feat == "resid_signed":
                parts.append(b - sh)
            elif self.head_feat == "resid_sq":
                parts.append((b - sh) ** 2)
        if psi is not None:
            parts.append(psi)
        return torch.cat(parts, dim=-1)

    def slot_for_axis(self, k: int) -> int:
        return 0 if self.mono else k

    def axis_state(self, S: torch.Tensor, k: int) -> torch.Tensor:
        """축 k 의 head 가 읽는 상태. S [..., n_slots, slot_dim] -> [..., head_width].

        - factored(M5/M6): 슬롯 k 전체
        - mono(M4)       : 유일한 슬롯 전체 (5 head 가 같은 잠재를 전부 읽는다)
        - **block_readout**: 유일한 슬롯의 **k 번째 블록만** — 하나의 벡터 안에 스키마 구획.
        """
        if self.block_readout:
            b = self.block_dim
            return S[..., 0, k * b:(k + 1) * b]
        return S[..., self.slot_for_axis(k), :]

    def _axis_input(self, z: torch.Tensor, feat: torch.Tensor | None) -> torch.Tensor:
        """z [...,z_dim] · feat [...,5,tagger_dim] -> 슬롯별 입력 U [...,n_slots,in_dim].

        단일/배치 양쪽에서 쓴다(슬롯 축을 항상 -2 에 만든다). 태거 경로에서 축 순서는
        `AXES` = protagonist·space·time·causality·intentionality 이며, `tagger_axes=3` 은
        **앞 3축**(태거가 직접 잡는 축)만 대체한다.
        """
        if self.tagger is None:
            if self.d1:
                return torch.stack([self.extractors[k](z) for k in range(5)], dim=-2)
            return self.proj(z).unsqueeze(-2)          # 한 벡터를 전 슬롯에 브로드캐스트
        if feat is None:
            raise ValueError("tagger 모델은 축별 태거 피처가 필요하다 (axisfeat)")
        parts = []
        for k in range(5):
            tk = feat[..., k, :]
            use = k < self.tagger_axes
            if self.tagger == "replace":
                parts.append(tk if use else self.extractors[k](z))
            else:                                       # hybrid: [E_k(z), 태거]
                parts.append(torch.cat(
                    [self.extractors[k](z), tk if use else torch.zeros_like(tk)], dim=-1))
        return torch.stack(parts, dim=-2)

    def encode_seq(self, z: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
        """z [T, z_dim] frozen embeddings -> slot states S [T, n_slots, slot_dim] (EntNet recurrence).

        D1: slot k is fed ONLY its per-axis extractor E_k(z) (separated input path).
        d1=False: ONE shared proj(z) broadcast to every slot (mono -> 1 slot; factored -> all 5 see
        the SAME vector and differ only via keys + their own recurrence).
        """
        U = self._axis_input(z, feat)      # [T, n_slots, in_dim]
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

    def encode_seq_batched(self, z: torch.Tensor, feat: torch.Tensor | None = None,
                           reset_bnd: torch.Tensor | None = None) -> torch.Tensor:
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
        U = self._axis_input(z, feat)                              # [B,T,n_slots,in_dim]
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
            if reset_bnd is not None and t >= 1:
                # transition t-1 이 경계면 **다음 스텝(t+1)** 의 이전 상태를 key 로 되돌린다.
                b = reset_bnd[:, t - 1].view(-1, 1, 1)
                h = b * keys + (1.0 - b) * h
        return torch.stack(out, dim=1)  # [B, T, n_slots, slot_dim]

    def forward_batched(self, z: torch.Tensor, resets: torch.Tensor, states: torch.Tensor,
                        sent_mask: torch.Tensor, trans_mask: torch.Tensor,
                        psi_bnd: torch.Tensor | None = None,
                        feat: torch.Tensor | None = None,
                        slot_bnd: torch.Tensor | None = None,
                        obs_target: torch.Tensor | None = None) -> dict:
        """배치판 forward. 패딩 구간은 손실에서 완전히 제외한다.

        z [B,T,z_dim] · resets [B,T-1,5] · states [B,T,3] (STATE_AXES 순서: 0,2,3) ·
        sent_mask [B,T] float(1=유효) · trans_mask [B,T-1] float(1=유효).
        반환은 `forward` 와 동형 {loss, change_logits, S, l_collapse}.
        """
        rb = None
        if self.slot_reset is not None:
            if slot_bnd is None:
                raise ValueError("slot_reset 모델은 slot_bnd 가 필요하다 (학습=앵커 경계, 추론=rollout)")
            rb = slot_bnd.float()
            if self.training and self.slot_reset_dropout > 0:
                # teacher reset dropout — 추론에서 경계를 놓치는 상황을 학습에 노출(ψ 와 동일 처방)
                rb = rb * (torch.rand_like(rb) >= self.slot_reset_dropout).float()
        S = self.encode_seq_batched(z, feat, reset_bnd=rb)   # [B,T,n_slots,slot_dim]
        loss = z.new_zeros(())
        change_logits = {}
        tm_sum = trans_mask.sum().clamp(min=1.0)
        # ── ψₑ (§4.3) — 학습은 앵커 경계 + teacher reset dropout ────────────────
        Psi = None
        if self.psi is not None:
            if psi_bnd is None:
                raise ValueError("psi 모델은 psi_bnd 가 필요하다 (학습=앵커 경계, 추론=rollout)")
            bnd = psi_bnd.float()
            if self.training and self.psi_dropout > 0:
                # 일부 teacher 경계를 무시 -> 추론 때 경계를 놓치는 상황을 학습에 노출
                bnd = bnd * (torch.rand_like(bnd) >= self.psi_dropout).float()
            Psi = self.psi_seq(S, bnd)                          # [B,T,n_slots,slot_dim]
        # ── ŝ (§4.6 Axis1) — head 가 잔차를 쓰면 미리 계산 ─────────────────────
        Sh_head = None
        if self.head_feat != "delta":
            Sh_head, _ = self.predict_next(S, Psi)              # [B,T-1,n_slots,slot_dim]
        for k in range(5):
            sk = self.axis_state(S, k)                 # [B,T,head_width] — 자기 축 몫만 본다
            a, b = sk[:, :-1], sk[:, 1:]
            d = self.head_input(a, b,
                                None if Sh_head is None else self.axis_state(Sh_head, k),
                                None if Psi is None else self.axis_state(Psi, k)[:, :-1])
            logit = self.change_heads[k](d).squeeze(-1)          # [B,T-1]
            change_logits[AXES[k]] = logit
            l = F.binary_cross_entropy_with_logits(
                logit, resets[:, :, k].float(), pos_weight=self.pos_weight[k], reduction="none")
            loss = loss + self.alpha[k] * (l * trans_mask).sum() / tm_sum
        # aux absolute-state heads (전역 enum 축만) — states 열 순서 = STATE_AXES 키 순서
        sm_sum = sent_mask.sum().clamp(min=1.0)
        for col, a in enumerate(STATE_AXES):
            sk = self.axis_state(S, a)                           # [B,T,head_width]
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
            Sh, z_hat = (self.predict_next(S, Psi) if Sh_head is None
                         else (Sh_head, self.readout(Sh_head.flatten(start_dim=-2))))
            terms = []
            if self.pred_space in ("obs", "both"):
                target = z[:, 1:] if obs_target is None else obs_target
                terms.append(((z_hat - target) ** 2).sum(-1))              # [B,T-1]
            if self.pred_space in ("slot", "both"):
                # §4.6 Axis2 — 슬롯별 자기 공간 예측오차를 슬롯에 대해 평균(축별 가중 없음)
                terms.append(((Sh - S[:, 1:]) ** 2).sum(-1).mean(-1))        # [B,T-1]
            pred_err = sum(terms) / len(terms)
            l_pred = (pred_err * trans_mask).sum() / tm_sum
            out["loss"] = loss + self.pred_weight * l_pred
            out["l_pred"] = l_pred
            out["pred_err"] = pred_err
        return out

    @torch.no_grad()
    def change_probs_slot_rollout(self, z: torch.Tensor, t_star: int = 3,
                                  feat: torch.Tensor | None = None) -> torch.Tensor:
        """slot_reset 모델의 **추론 경로** — 자기 예측 경계로 슬롯 메모리를 리셋하며 전진.

        학습은 앵커 경계로 리셋하지만 추론에는 앵커가 없다. 여기서는 모델 자신의 전이별 판정
        (`Σ_k 1[p_k > 0.5] ≥ t*`)으로 리셋한다 — 온라인 정보만 사용(미래·gold 미사용).

        🔴 리셋은 **한 문장 지연**된다. 전이 t-1 의 판정이 s_t 를 필요로 하므로 s_t 는 연속
        갱신으로 만들고, 그 판정 결과는 s_{t+1} 을 만들 때의 이전 상태에 적용한다. 학습 경로
        (`encode_seq_batched(reset_bnd=...)`)와 **같은 지연**이라 train–inference 정렬이 맞다.
        """
        B, T = z.shape[0], z.shape[1]
        U = self._axis_input(z, feat)
        h = self.keys.unsqueeze(0).expand(B, -1, -1).contiguous()
        keys = self.keys.unsqueeze(0)
        prev = None
        probs = []
        for t in range(T):
            ut = U[:, t]
            gi = self.gate_in(ut)
            g = torch.sigmoid((gi * h).sum(-1) + (gi * keys).sum(-1))
            pre = self.A(h) + self.B(keys) + self.C(ut)
            cand = self.phi(pre.reshape(-1, pre.shape[-1])).view_as(pre)
            st = h + g.unsqueeze(-1) * cand
            st = st / (st.norm(dim=-1, keepdim=True) + 1e-8)
            h = st
            if prev is not None:                       # 전이 t-1 판정
                p = []
                for k in range(5):
                    ks = self.slot_for_axis(k)
                    d = self.head_input(prev[:, ks], st[:, ks], None, None)
                    p.append(torch.sigmoid(self.change_heads[k](d).squeeze(-1)))
                pt = torch.stack(p, -1)                # [B,5]
                probs.append(pt)
                b = ((pt > 0.5).sum(-1) >= t_star).float().view(B, 1, 1)
                h = b * keys + (1.0 - b) * st          # 다음 스텝의 이전 상태
            prev = st
        return torch.stack(probs, dim=1)               # [B,T-1,5]

    def change_probs_rollout(self, z: torch.Tensor, t_star: int = 3,
                             feat: torch.Tensor | None = None) -> torch.Tensor:
        """ψ 모델의 **추론 경로** — 자기 예측 경계로 ψₑ 를 리셋하며 전진. 반환 P [B,T-1,5].

        학습 때는 앵커 경계로 ψ 를 리셋하지만 추론에는 앵커가 없다. 여기서는 **모델 자신의**
        전이별 판정(`Σ_k 1[P(reset_k) > 0.5] ≥ t*`)으로 리셋한다 — 온라인에서 계산 가능한
        정보만 쓴다(미래·gold 미사용).

        ⚠️ 이 판정은 **ψ 리셋 전용**이며 최종 경계 규칙이 아니다. 최종 경계는 바깥에서
        `s = Σ_k P(reset_k)` + 적응 임계로 정한다(D-B29: 단일 규칙만 보정).
        """
        S = self.encode_seq_batched(z, feat)                    # [B,T,n_slots,d]
        B, T = S.shape[0], S.shape[1]
        run = torch.zeros_like(S[:, 0])
        cnt = S.new_zeros(B, 1, 1)
        keep = S.new_ones(B, 1, 1)
        probs = []
        for t in range(T):
            if self.psi == "ema":
                run = keep * (self.psi_beta * run + (1 - self.psi_beta) * S[:, t]) \
                    + (1.0 - keep) * S[:, t]
                psi_t = run
            else:
                run = run * keep + S[:, t]
                cnt = cnt * keep + 1.0
                psi_t = run / cnt
            if t == T - 1:
                break
            a, b = S[:, t], S[:, t + 1]                         # 전이 t: s_t -> s_{t+1}
            p = []
            for k in range(5):
                ks = self.slot_for_axis(k)
                sh = None
                if self.head_feat != "delta":
                    x = a[:, ks]
                    sh = self._f(ks)(torch.cat([x, psi_t[:, ks]], -1) if self.psi else x)
                d = self.head_input(a[:, ks], b[:, ks], sh, psi_t[:, ks] if self.psi else None)
                p.append(torch.sigmoid(self.change_heads[k](d).squeeze(-1)))
            pt = torch.stack(p, -1)                             # [B,5]
            probs.append(pt)
            keep = 1.0 - ((pt > 0.5).sum(-1) >= t_star).float().view(B, 1, 1)
        return torch.stack(probs, dim=1)                        # [B,T-1,5]

    def forward(self, z: torch.Tensor, resets: torch.Tensor, states: dict,
                mask: torch.Tensor | None = None, axis_mask: tuple | None = None,
                feat: torch.Tensor | None = None) -> dict:
        """z [T,z_dim]; resets [T-1,5] int; states {field:[T] int}. Returns {loss, change_logits, S}.

        mask [T-1,5] float: 0 = 그 전이의 그 축 라벨 유보 -> 손실에서 제외 (SHARE place/time 미정렬 구간).
        axis_mask (5,) bool: False인 축은 손실에서 통째 제외 (라벨이 구조적으로 무의미한 축).
        둘 다 None(기본)이면 기존 경로와 수치 동일 -> 기존 체크포인트/REPORT 재현성 보존.
        """
        S = self.encode_seq(z, feat)
        loss = z.new_zeros(())
        change_logits = {}
        for k in range(5):
            sk = self.axis_state(S, k)  # [T, head_width] — head sees ONLY its own share
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
            sk = self.axis_state(S, a)  # [T, head_width]
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
