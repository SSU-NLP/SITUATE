# causality 반사실+인과 2단계 창(window) — 사전등록 (PRE-REGISTRATION)

> **결과 보기 전 봉인.** D-M5 / `[[no-postresult-criterion-change]]` 규약: 아래 통과조건·문항·규약은
> 결과를 보고 고치지 않는다. 위반 시 기각으로 적고 사유만 남긴다.
> 작성 2026-07-25. 상위 계획 `/root/.claude/plans/snuggly-discovering-reddy.md`.
> 수정은 하단 "수정 로그"에 append(삭제 금지).

## 0. 배경 — 네 번의 실패와 공통 뿌리

| 시도 | 결과 | 어긋난 지점 |
|---|---|---|
| event_type 프록시 (현행 frozen) | 발현 0.674(repcorpus)·0.660(dev-3), lift 1.135 | 관계를 안 봄 — 라벨 두 개 비교 |
| requires/produces strict | 사전등록 FAIL (빈집합 55%·인과연쇄→연속 0.69) | 인접 쌍만 봄 |
| requires/produces carry-forward | t=10(prisoner→sprint) 오연결 | 가능화를 인과로 계산 |
| Zacks-halo (`2026-07-25_causality_halo_zacks`) | primary 7/7 PASS, **sanity 0.869 FAIL** | 인접 + "직접 야기" 1종만 |

넷 다 **관계 그래프를 인접 비교로** 만들려 했다. halo의 0.869는 "직전 문장이 이걸 **직접** 야기했나"
= 4관계 중 1종·거리 1만 묻는 질문이라 "아니오"가 기본값이 된 결과다.

**Trabasso & Sperry (1985)는 두 개의 서로 다른 기준을 썼다.**
(a) **반사실 필요성** — "A가 일어나지 않았다면, 이 이야기의 정황에서, B는 일어나지 않았을 것이다."
    이 기준은 물리·심리·동기·**가능화 네 관계 전부에 높은 강도를 준다.**
(b) **인과 기준** — 이것이 앞의 셋과 **가능화(enablement)**를 갈라낸다.
그 외: 인접 쌍이 아니라 **모든 쌍**(인과 거리 ⊥ 문장 거리), 이진이 아니라 **그래프**(사슬 밖 = 막다른 사건).

**가설**: 반사실(1단계) → 인과(2단계) 2단계 질의를 창 W 안의 모든 선행 문장에 적용하고 `enabling`을
링크에서 제외하면, 발현율이 정상 대역(20~30%, Zacks 22%)으로 내려오고 판별력이 생긴다.

## 1. 조작화 (동결)

- **라벨러**: frozen **Qwen3.6-35B-A3B no-thinking**, 로컬 vLLM `http://localhost:8001/v1`,
  `temperature=0`, `response_format={"type":"json_object"}`,
  `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`.
- **질의 = 쌍 단위. 인덱스·거리·중간 문장 미제공.** boundary/change/segment/reset/이전상태 어휘 금지.

**1단계 — 반사실(필요성):**
> `If X had not happened, in the circumstances of this story, would Y still have happened?`
> → `{"y_still_happens": true | false}`

`true` → **`none` 확정, 2단계 안 물음.**

**2단계 — 인과(가능화가 갈리는 지점), 1단계 `false`인 쌍에만:**
> `X did not merely precede Y. Did X bring Y about, or did X only make Y possible — setting up a`
> `condition or opportunity that Y depended on, without bringing Y about?`
> → 일으킴: `physical` / `psychological` / `motivational` · 조건만: `enabling`
> → `{"relation": "<label>"}`

**왜 2단계인가.** 반사실만 물으면 `prisoner_in_yard → sprint`의 답은 "그렇다"이다(마당에 없었으면 못
달렸으므로). 그 답으로 `enabling` 라벨을 기대하는 건 질문이 요구하지 않은 구분을 채점하는 것이고,
P2ⓑ 실패 시 **스킴이 틀린 건지 프롬프트가 틀린 건지 구분이 안 된다.** 두 단계로 나누면 문헌 구조를
그대로 재현하므로 "Trabasso의 두 기준을 두 단계로 구현"으로 방어된다.

**판정 규칙:**
- **링크** = `relation ∈ {physical, psychological, motivational}`. `enabling`·`none` = 링크 아님.
- **`reset(i) = 1` ⟺ `j ∈ [i−3, i−1]` 중 링크가 하나도 없음** (W=3 동결). i<3이면 가능한 후보만,
  `n_cand` 기록.
- **W=5까지 수집하되 판정은 W=3.** W=4·5는 **서술적 민감도 전용**, 판정을 바꾸는 데 쓰지 않는다.
- 파싱 실패/enum 밖 라벨 → 1회 재시도 후 `none` 처리(보수적)·건수 기록.
  **파싱 실패율 > 2%면 런 무효.**

**k(다중샘플) 규약** — reasoning-off 실패 후 채택한 k=3 완화책 승계:

| 대상 | 규약 | 근거 |
|---|---|---|
| probe (64문항) | **k=3 다수결 라벨로 판정** | 소표본 + 배치 numerics flip 25~30% → 단일 draw는 판정이 뒤집힘 |
| 코퍼스 (repcorpus 2222 · dev-3 1005) | **단일 draw** | 전이 평균이라 노이즈 상쇄 |

## 2. 재료

- **probe (신규 2종, 셀당 16문항 = 총 64):**
  - `discordant_probe_v2.jsonl` — 장면 고정·인과만 변별. 원본
    `2026-07-25_causality_halo_zacks/discordant_probe.jsonl` 14문항(brk1~7·con1~7)을 **한 글자도
    바꾸지 않고 부분집합으로 포함** + 각 셀 9문항 추가 → 16 break + 16 continuous.
  - `cutaway_probe.jsonl` — 16 cutaway(장면 바뀌나 인과 이어짐) + 16 enabling 통제(장면 동일·순수 가능화).
- **코퍼스:** repcorpus stage0 12편 `outputs/eacl_phase0/repcorpus_masters_stage0/*.jsonl` (2222 전이) ·
  dev-3 `{pieman,tunnel,monkey}_michelmann_master.jsonl` (1005 전이, 사람 gold 66 경계).
- **frozen 5축 matrix (스왑 대상):** `outputs/eacl_phase0/{doc}_s5_q36_35b_v2/reset_matrix.jsonl`.
- **스크립트:** `scripts/eacl_phase0/causality_cf_window.py`,
  `scripts/eacl_phase0/halo_guardrail_score.py`(일반화판). 원자료 이 디렉토리.

### 🔴 probe 셀 크기 = 16인 이유 (사전 계산)

셀당 6문항에 ≥0.70을 걸면 사실상 5/6이다(4/6=0.667 미달). k=3 다수결 후에도 문항당 오류가 ~18% 남으면
**스킴이 완벽해도 통과율 ~70%**, 셀 18이면 ~92%. FAIL 분기가 축을 **영구히** 죽이는 구조와 결합하면
치명적이므로 셀당 16으로 올린다(문항 작성은 로컬 런보다 싸다).

### 🔴 baseline (사전등록 시점 확정, API 0) — 현행 event_type 프록시, dev-3

| doc | n | 경계 | 발현율 | P(reset\|경계) | P(reset\|비경계) | **lift(비율)** |
|---|---|---|---|---|---|---|
| pieman | 95 | 18 | 0.800 | 0.833 | 0.792 | 1.052 |
| tunnel | 503 | 18 | 0.716 | 0.833 | 0.711 | 1.171 |
| monkey | 407 | 30 | 0.558 | 0.633 | 0.552 | 1.148 |
| **pooled** | 1005 | 66 | **0.660** | 0.712 | 0.627 | **1.135** |

**lift 정의 통일 = 비율** `P(reset|경계)/P(reset|비경계)`. ledger의 "lift +0.007"은 **차이값**이라
단위가 다르므로 인용하지 않고, 위와 같이 같은 비율 정의로 재계산한 값을 병기한다.

## 3. 🔒 통과 조건

> **채택 규칙 (한 줄, 모호성 제거):
> P1·P2·P3·P5 전부 PASS + P4가 −0.09 이내. 다섯 중 하나라도 실패하면 기각.**
> P1·P2·P3·P5 = primary 타깃지표, P4 = 안전장치. "primary만 세면 4개"라는 사후 독해를 막기 위해
> **다섯을 모두 하드로 못박는다.**

| # | 조건 | 기준 | 데이터 |
|---|---|---|---|
| **P1** | 인과 판별 | within-scene 인과단절 → 단절율 **≥0.80** AND 인과연쇄 → 오발현 **≤0.20** | `discordant_probe_v2.jsonl` 16+16, k=3 |
| **P2** | **지연halo 판별타당도** | ⓐ cutaway → 링크율 **≥0.70** AND ⓑ enabling 통제 → `enabling` 라벨 **≥0.70** | `cutaway_probe.jsonl` 16+16, k=3 |
| **P3** | 발현율 정상대역 | **repcorpus pooled ∈ [0.15,0.40] AND dev-3 pooled ∈ [0.15,0.40]** | 2222 + 1005 전이 |
| **P4** | Score 가드레일 (안전장치) | causality 열만 스왑, 4축 + **t\*=3 동결** → **ΔScore ≥ −0.09** | **dev-3만** |
| **P5** | 판별력 | **dev-3 pooled lift 부트스트랩 95% CI 하한 > 1.0** | dev-3 (baseline 1.135 병기) |

**P2 = ledger D-B9 "지연halo 게이트"의 조작화** (사용자 결정 2026-07-25). 창이 경계를 가로지를 때의
링크밀도 차이는 **기각조건이 아니라 서술 통계**로 보고한다 — 밀도 차이는 우리가 찾는 신호 자체라 문자
그대로 적용하면 작동하는 어떤 축도 기각되기 때문. 대신 "모델이 인과가 아니라 *같은 장면인가*를
답하는가"를 P2로 직접 잰다.

**P3에 dev-3를 넣는 이유:** repcorpus 0.25인데 dev-3 0.60이면 P3는 통과하고 축은 망가진 것이다.
gold가 있는 쪽이 dev-3이고 어차피 돌리므로 공짜.

**P4의 t\*:** dev-3에서 argmax를 **새로 뽑지 않는다.** annotation-guide §3이 t\*=3을 dev(pieman)
argmax로 동결·재선택 없음으로 못박았으므로 **t\*=3 고정**. (halo의 t\*=5는 repcorpus config argmax로
설정이 다르다.) 매칭 비교라는 방어가 있어도 D-M5 논쟁거리를 만들지 않는다.

### 🔴 P5 최소검출효과(MDE) — 이 기준은 관대하지 않다 (사전 계산)

n₁=66 · n₀=939 · p₀≈0.25 에서 `SE(log lift) = 0.1862` → CI 하한이 1.0을 넘으려면
**관측 lift ≥ 1.440**이어야 한다. **즉 게이트에서 뺀 1.3보다 이 기준이 더 엄격하다.**

| 참값 lift | 1.3 | 1.5 | 2.0 |
|---|---|---|---|
| 검정력 | **0.291** | 0.586 | 0.961 |

> **P5 FAIL은 "판별력 없음"이 아니라 "n=66에서 중간 크기 효과를 검출할 수 없었음"으로 읽는다.**
> FAIL이 축을 영구히 죽이는 만큼 이 해석을 미리 못박아 둔다. RESULT에 이 문장을 그대로 인용한다.

**부트스트랩 규약(동결):** 전이 단위 resample, 문서별 층화, 10,000회, seed 20260725.
**이 CI는 전이 수준(transition-level)이며 문서 간 변동을 포함하지 않는다.** 같은 문서 안 전이는 서로
독립이 아니므로 CI가 실제보다 좁다. 문서 3편이라 doc-level 부트스트랩은 불가(fiction held-out 1편
문제와 같은 뿌리). **실제 불확실성은 보고된 CI보다 크다.**

## 4. 🔒 FAIL 분기 — 여섯 번째 시도 없음

> 하나라도 실패 → **기각으로 적고 기준을 고치지 않는다.** 3단계(사건 노드 + 사슬 전환 상태축) 착수 없음.
> **그리고 causality를 약축으로 확정한다.** 조작화 실패 5건(event_type 프록시 · reqprod strict ·
> reqprod carry-forward · Zacks-halo · 반사실 창)을 **논문 발견**("관계적 구성개념이 계산 프레임 이식에
> 특히 저항")으로 기록하고 held-out으로 이동한다.
> **causality 재조작화 시도는 여기서 종료** — "3단계는 안 하지만 프롬프트만 살짝"도 포함해서 없다.

PASS여도 채택 아님 — 3단계 착수는 **별도 승인**, 최종 채택은 Gate C DTS Score(별도 실험).

## 5. 필수 보고 항목 (RESULT.md)

- P1~P5 사전기준 대비 판정표. P1은 **확장 셀(게이트)** 과 **원본 14문항 부분집합**(halo 7/7·0/7 비교용)
  둘 다 보고.
- **관계 유형 분포** — 링크 중 physical/psychological/motivational 비율. 한 종류가 ~95%면 4종 스킴이
  사실상 이진으로 퇴화한 것이고 3단계 설계에 직결된다.
- 1단계 `false` 비율(= 반사실 필요성 인정률)과 그중 `enabling` 비율 — **두 기준이 실제로 갈리는지.**
- W=4·5 민감도(서술) · 경계횡단 링크밀도(서술, 기각조건 아님) · 문서별 발현율·lift · k=3 만장일치/2:1 문항 수.
- repcorpus Score는 **계산하지 않는다**(gold 10/2222 = 0.5% → 코드가 거부).

## 6. 한계 (제거 금지)

- probe 64문항 = hand-crafted·낙관 편향(셀 16으로 키워도 남는다). halo가 probe PASS 후 실전 sanity에서
  뒤집힌 전례가 바로 이 코퍼스에 있다 — **P3를 P1·P2보다 무겁게 읽는다.**
- dev-3 경계 n=66이 P5 정밀도를 지배(위 MDE). CI 하한 기준은 한계 인정이지 극복이 아니다.
- P5 CI = 전이 수준, 문서 간 변동 미포함.
- **G2(target-only) 위반은 이 축에서 의도적이다.** causality는 관계적 구성개념이라 단독 문장으로
  정의되지 않는다(D-B9). P2가 그 대가로 치르는 위험을 재는 유일한 장치다.
- dev-3는 pieman 기준 선택압 9회 누적(D-B9). 이번이 10번째 — **확인적 성능 아님.** 시험지(Tier1
  held-out 6편)는 열지 않는다.
- 배치 numerics flip 25~30%(infrastructure §4b) — 코퍼스는 단일 draw 기준.

## 수정 로그
- (없음 — 초판, 봉인 2026-07-25)
