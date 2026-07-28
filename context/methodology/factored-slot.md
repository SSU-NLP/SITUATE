# factored-slot — EACL M4~M7 잠재 상황모델 (알고리즘 정의)

> **최초 작성 2026-07-28.** 이 모듈(`src/hi_ontop/factored_slot.py`)은 2026-07-15 Phase 2 de-risk 때
> 만들어졌으나 **methodology/ 에 한 번도 기록되지 않았다.** 2026-07-28 dynamics 추가를 계기로 신설한다.
> 설계 본체 = `EACL/factored-slot-eism-v0_1.md` §4 · 명명 = `EACL/method-registry.md` §3 ·
> 실행 수치 = `EACL/phase0-decisions-and-runs.md` (D-B17).

## 1. 무엇인가

**frozen 문장 임베딩 위에서 상황모델(EISM 5축)을 슬롯에 나눠 추적하고, 축별 리셋을 판정하는 작은
순환 모듈.** 인코더는 학습하지 않는다(`paraphrase-multilingual-mpnet-base-v2`, 768D, L2 정규화 —
M1~M7·baseline 공통이라 공정 비교가 성립).

**5축 (D-B10 동결):** `protagonist_present · space · time · causality · intentionality`.
학습 앵커 = M3-S5 가 만든 축별 0/1 리셋 라벨(Phase-1 코퍼스 10,598편, D-B15).

## 2. 상태 갱신 — EntNet 타입 슬롯

문장 임베딩 `z_t` 를 순차로 읽으며 슬롯 상태 `h ∈ R^{n_slots × slot_dim}` 를 갱신한다.
슬롯 키 `keys` 는 학습 파라미터이며 **슬롯 교환가능성을 깨는 장치**(Henaff 2017 EntNet).

```
U_t   = [E_k(z_t)]_{k=1..5}                     # D1 라우팅 (아래 §3)
g     = σ( ⟨W_in u, h⟩ + ⟨W_in u, keys⟩ )        # 슬롯별 게이트 (이 문장이 나와 상관있나)
cand  = φ( A·h + B·keys + C·u )                  # φ = PReLU(slot_dim)
h     ← h + g ⊙ cand
h     ← h / (‖h‖ + 1e-8)                         # 슬롯별 정규화
```

**D1 = 축별 추출기 라우팅** (codex 2026-07-15): `E_k : 768 → 32 → GELU → Dropout(0.4) → bottleneck(8)`
를 **슬롯 k 에만** 먹인다. 파라미터가 아니라 **정보 경로**를 분리하는 것이 요점 —
v1(전 슬롯 공유 입력)은 모든 슬롯이 모든 축을 동일 디코딩해 M5≈M4 로 붕괴했다(0/5 분리).
`mono=True`(M4)면 D1 없이 `proj: 768 → slot_dim` 하나를 단일 슬롯에 먹인다.

## 3. 판정 헤드

- **변화(리셋) 헤드** — 축 k 마다 `d = [s_{t-1}, s_t, s_t − s_{t-1}, s_t ⊙ s_{t-1}]` (4·slot_dim)
  → MLP → logit. **헤드 k 는 슬롯 k 만 본다**(D3 anchored latent: 앵커는 타깃이지 추론 입력이 아니다).
- **절대 상태 보조 헤드** — 전역 고정 enum 축만: `perspective`(4) · `temporal_shift_marker`(4) ·
  `event_type`(10). story-local 축(space/intentionality)은 제외.

## 4. 손실

```
L = Σ_k α_k [ L_Δ,k + β·L_state,k ] + γ·L_collapse   ( + pred_weight·L_pred )
```
- `L_Δ` = 축별 리셋 BCE(pos_weight = 축별 `(1−p)/p` clip[1,30]) — **primary anchor**
- `L_state` = 절대 상태 CE (보조), β = 0.25
- `L_collapse` = 슬롯별 **시간축 표준편차 하한**, `Σ ReLU(var_floor − std)²`, γ = 0.05, var_floor 0.1.
  슬롯 간 직교화는 **쓰지 않는다**(실제 EISM 축 상관을 죽이므로).
- `α = (1.0, 0.5, 1.0, 0.4, 0.25)` — 축별 가중.
  ⚠️ **설계 이탈**: `factored-slot-eism-v0_1.md` §4.5 는 *"No ad-hoc per-slot weights; normalization
  is emergent"*(우도비의 곱)를 명시한다. 현 구현의 α 는 그 원칙에 반하는 임시 조치다.

### 4b. dynamics + next-state 예측 (2026-07-28 추가)
`dynamics` 인자로 켠다. **registry §3 의 명세를 구현한 것이며 새 메커니즘 도입이 아니다** —
공통 전제 *"학습 신호 = 자기지도(next-state 예측) + anchor"*, dynamics 열 M4=1 / **M5=축별 5** /
M6=공유 1.

```
Ŝ_{t+1,k} = f_k( S_{t,k} )                 # f = MLP[slot_dim→slot_dim→slot_dim]
ẑ_{t+1}   = g( concat_k Ŝ_{t+1,k} )        # readout → 768D  (= ①기대 생성)
L_pred    = ‖ẑ_{t+1} − z_{t+1}‖²           # 라벨 불필요   (= ②오차 측정)
```
`dynamics=None`(기본) → 항이 붙지 않아 **Phase 2 수치를 완전 재현**한다.
`forward()` 는 전이별 `pred_err` 를 함께 반환해 분절 신호로 쓸 수 있다.

## 5. 인프라 — 배치 경로 (2026-07-28)

시간축 재귀가 파이썬 루프라 Phase-1 스케일(2,788,159문장)에서 에폭당 279만 순차 스텝이 된다.
→ `encode_seq_batched` / `forward_batched` 추가: **문서를 배치로 묶어** 스텝 수를 (배치 수 × 최대
문서길이)로 축소(batch 32 기준 약 60배 단축). 길이 내림차순 버킷팅 + **뒤쪽 패딩**만 사용하고
패딩 구간은 `sent_mask`/`trans_mask` 로 손실에서 제외한다(패딩이 뒤에만 있으므로 유효 구간의 h 를
오염시키지 않는다).
- PReLU(num_parameters=slot_dim)는 2D 입력의 dim=1 을 채널로 보므로 **평탄화 후 적용**해야 한다.
- `l_collapse` 는 배치판에서 **마스크 가중 표준편차**로 계산(패딩 제외).
- 기존 단일 경로는 **변경하지 않았다**. 동치성은 `tests/test_factored_slot_batched.py`(6 tests)로
  검증: B=1 수치 일치 · 패딩 무관성 · dynamics 기본 off · 슬롯/forward-model 개수(M5=5·M6=1·M4=1).

## 6. SEM 계승 측면 (있는 것 / 없는 것)

| SEM 요소 | 현 구현 |
|---|---|
| scene dynamics (예측) | ⚠️ 2026-07-28 추가. 단 **의미공간(768D) 점예측**이며 SEM 의 scene-space 예측과 다름 |
| prediction error 기반 분절 | ⚠️ `pred_err` 노출됨. **실측 결과 리셋확률 합보다 나빴다**(D-B17) |
| sticky-CRP / latent-cause 전환 | ❌ **미구현** — 설계 §4.4 의 κ. `method-registry.md` §5 가 "t\* 작동점 결함을 원리적으로 해결할 것"으로 지목한 바로 그 요소 |
| event switching (경계에서 상태 리셋) | ❌ **미구현**. 경계를 *예측*만 하고 슬롯을 비우지 않는다 |
| local MAP / particle smoothing | ❌ 미구현 (설계 §4.7) |
| 슬롯별 이종 확률분포족 · surprise = −log p | ❌ 미구현 (설계 §4.2). 현 구현은 L2 점오차 |

⇒ **현 모듈은 설계된 M5 가 아니다.** `plan-factored-slot.md` line 142/201 의 **fallback 경로
(시나리오 B: "정렬 O, dynamics 한계")** 에 해당한다. 이름을 "M5"로 쓰지 말 것 —
**"P2 정렬 검증 + fallback 분절기"** 가 정확하다.

## 7. 알려진 한계 / 고려 중인 변형

- **정수 임계 병목.** 최종 경계가 5축 합의 정수 임계(t\*=3) 하나로 압축되어, 슬롯 분리 이점이
  통과할 자리가 없다. 앵커 재현성 실측에서 **경계 불일치의 77.7% 가 count 2↔3**(D-B16 결과).
  → **1순위 변형 = §4.5 우도비 스위칭 + §4.4 sticky-CRP κ.**
- **예측 공간 누출.** 설계 §Risks 의 *"semantic-space prediction re-importing IDM"* 을 현
  `readout → 768D` 가 그대로 범한다. → 변형 = **slot-native 예측 공간 T_k**(§4.6 Axis2).
- `pred_weight` 는 튜닝되지 않았다(1.0). dev 에서 튜닝하면 selection-on-test 이므로 확정 패스에서
  사전등록 후 정해야 한다.
- 앵커가 **k=1 단일 draw**(S_self 0.659 = 상한). 라벨 노이즈가 학습 타깃에 얼마나 전이되는지 미측정.

## 8. 관련 파일

`src/hi_ontop/factored_slot.py` · `scripts/eacl_phase0/{build_anchor_dataset,train_anchor_m4m5,
probe_slot_alignment_phase1}.py` · Phase 2 원본 `scripts/eacl_phase0/{build_slot_dataset,
train_factored_slot,probe_slot_alignment,slot_seed_sweep}.py` ·
결과 `outputs/experiments/{m5_phase2_slotalign,m4m5_phase1_anchor}/REPORT.md`
