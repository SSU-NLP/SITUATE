# M3-S5-Extractor — 최종 확정 스펙 (CANONICAL)

> **이 문서는 M3(S5-Extractor)의 재현 가능한 유일 스펙이다.** 구현: `scripts/eacl_phase0/llm_s5_states.py`
> **역할 (2가지):** ① Phase 0 kill-switch 측정기 ② **M4~M7의 anchor 라벨 공급원** (LLM은 학습 데이터를 만들 뿐, 최종 추론에는 등장하지 않음)
> **타깃:** 사건 경계(event boundary, B) — 이야기 *안*의 사건 단위. 썰 경계(A) 아님.
> **명명:** `method-registry.md` 기준. baseline은 **CosDrift(M1)** — "IDM"이라 부르지 않는다.

---

## 0. 설계 대원칙 (S1~S3 실패에서 도출 — 절대 위반 금지)

| 함정 | 실패 사례 | 해법 (M3에서 유지) |
|---|---|---|
| **boundary halo** | S1: LLM에 "변했나"를 직접 물음 → AUC 1.0 (오염, 무효) | **LLM에 변화/리셋/경계/이전상태를 절대 묻지 않는다** |
| **carryover** | S2: 왼쪽 문맥 제공 → 이전 상태 이월 (AUC 0.55) | **Stage2는 target 문장 단독** |
| **표현 노이즈** | S3: free-form 상태 추출 → 표현 변이 (AUC 0.39) | **닫힌 목록(closed list)에서만 선택. 자유서술 금지** |
| **판정 오염** | — | **변화 판정·carry-forward는 100% 코드** |
| **사후 fitting** | — | **모든 enum·규칙·임계값은 gold 확인 전 고정** |

---

## 1. 추출 파이프라인 (2-stage)

**모델(동결, D-M13⑥·2026-07-24):** `Qwen3.6-35B-A3B` **no-thinking**, 로컬 vLLM(:8001), **temp 0**,
batch 20, shuffle seed = `20260711 + axis_index`. **LLM frozen** — 파라미터 학습 없음. 순수 추출기.
**프롬프트 버전(동결, D-B14·2026-07-25): `p0s5-eism-v3.1` = v2 + H + E.**
`llm_s5_states.py --prompt-version v3.1` (`V31_FIXES = {space_H, time_E, time_conseq, time_first}`).
sha256: stage1 `7ea7abff71…`(=v2) · space `4d9fe79cdd…` · time `a6a00bc7dd…` · 나머지 3축 = v2 동일.
> ⚠️ 이전 표기 `claude-fable-5` / `p0s5-eism-v2` 는 **이력**이다. §3.5 의 7개 데이터셋 표는 fable5
> 시절 수치이며 **현 동결 설정과 비교 금지**(라벨러가 다르다 — 같은 pieman 이 0.634 → 0.549).

### Stage 1 — 이야기별 닫힌 목록 추출 (문서당 1회)

이야기/대화 **전체**를 입력하고 다음을 추출한다:

- `cast`: 등장 인물 목록 (정규화 ID: `narrator`, `roommate`, …)
- `locations`: 등장 장소 목록 (`bedroom`, `convenience_store`, …)
- `goals`: 추구되는 목표 목록

**설계 근거 (EntNet tied-keys 차용):** EntNet이 슬롯 키를 **데이터에 이미 주어진 엔티티 목록**으로 시드했듯, 이야기 단위로 목록을 먼저 확보하고 문장 단위에서는 **대조만** 한다. 이것이 S3의 표현 노이즈를 원천 차단하면서 **이야기 내 구별력**을 유지한다.

**오염 안전성:** 경계·변화·세그먼트 관련 단어를 프롬프트에 **일절 넣지 않는다.** 이 단계는 엔티티 나열일 뿐 경계 위치와 무관하다.

### Stage 2 — 문장 단독 대조 (축별 별도 호출)

각 문장에 대해, **Stage1 목록 + 고정 enum만을 선택지로** 제시. LLM은 **고르기만** 한다.
**target 문장만 제공.** 왼쪽 문맥·다른 문장·다른 축·경계·CosDrift 점수 **미제공.**

---

## 2. 축별 스키마 + 리셋 규칙 (리셋 판정은 100% 코드)

### 2.1 protagonist
- **추출:** `perspective` ∈ {first, second, third, none} · `characters_present` ⊆ Stage1 `cast`
- **리셋 (동결):** `perspective` 변함 **AND** 둘 다 ≠ none
- **⚠️ `characters_present`는 count에서 제외** — ablation 전용 (D-B8).
  - **근거:** dev에서 charset(Jaccard<0.5) 단독 AUC 0.538 (경계 0.57 vs 내부 0.49 — 거의 안 갈림). perspective 단독 0.609. **둘 병용 시 0.542로 오히려 하락** — charset이 깨끗한 perspective 신호를 희석.
  - **원인:** 조연이 장면 안에서 수시로 들락날락 → charset이 상시 흔들림.
  - **결정:** protagonist = **perspective 단독** (D-B7). EISM 5지표를 유지하면서 dev 실측을 만족.

### 2.2 place — 느린 변수 (carry-forward 필수)
- **추출:** `location` ∈ Stage1 `locations` ∪ {`unspecified`}
- **🔒 v3.1 fix H (D-B14):** Stage-2 는 **이 문장 자체의 근거만** 본다 — 이웃 문장·앞 문맥·"있어야 할
  곳"으로 추론 금지. 문장 단독으로 무대가 안 잡히면(순수 감정·고립된 회상·설정 없는 신체묘사)
  **반드시 `unspecified`.** 효과: no-cue 문장 non-unspecified 율 dev-3 7.1% → 5.0%.
- **carry-forward (코드, 경계-무관):** `unspecified`면 **직전 명시값 유지**, 새 명시값 나오면 덮어씀
- **리셋:** carry 후 값 변함 **AND** 둘 다 ≠ unspecified
- **근거:** target-only 추출 시 place는 86% `unspecified`(느린 변수 — 한 번 설정되면 지속되나 가끔만 명시). carry-forward 없이는 死(reset 1/212). carry 적용 후 부활 (0.498 → 0.572).
- **⚠️ ~~이 축이 scope 판별자~~ → D-M13⑤로 철회.** ~~"무대 리셋이 EISM 작동의 핵심 조건"~~은 표본 부족(경계에서 place 발화 pieman 3/18·tunnel 6/18·monkey 1/30)으로 철회. Monkey는 "死"가 아니라 **명시 문장이 셋 중 최다(72)인데 전환이 사건경계와 무정렬**(리셋/명시 0.22 vs pieman 0.56·tunnel 0.75). 상세: ledger D-M13⑤ + `outputs/experiments/2026-07-20_axis_liveness/REPORT.md`.

### 2.3 time — **tense 조건 제거됨** (버그 수정)
- **추출:** `tense` ∈ {past, present, future, none} · `temporal_shift_marker` ∈ {absent, sequence, jump, simultaneity}
- **리셋 (최종 확정):** `temporal_shift_marker` ∈ {sequence, jump, simultaneity}
- **🔒 v3.1 fix E (D-B14):** 표지가 **문장 첫머리에 올 때만** shift 로 인정(선행 접속사·인용부호 허용).
  문중 표지('he walked home and later ate')는 `absent`. **consequence/contrast 접속사(so, therefore,
  and yet, but)는 시간표지가 아니다.** 'First' 등 순서표지 목록 보강. 효과: 문중 marker 율
  repcorpus 41.2% → 22.5%. ⚠️ dev-3 에서는 라벨 4개만 바뀌어 **Score 기여 0**(근거는 repcorpus 타깃).
- **🔴 `tense` 변화는 리셋 조건에서 제거한다.**
  - **근거 (튜닝이 아니라 결함 수정):** Monkey에서 time 축 119회 리셋 중 **약 103회가 시제 뒤집힘**(과거↔현재 화자 코멘트)이라는 **가짜 알람**이었다. 진짜 시간점프 표지는 8~12회뿐. 서사에서 시제는 시간 이동과 무관하게 왕복한다("사진작가가 갔다"(과거) → "얘는 원숭이예요"(현재 코멘트) → "원숭이가 잡았다"(과거)).
  - **효과:** dev 0.598 → **0.634**. test 전반 개선 (tunnel 0.294→0.383, monkey 0.256→0.359).
  - `−time`(축 완전 제거, dev 0.628)보다 **`time=marker만`(0.634)이 우수** → 진짜 시간점프 신호는 살아있다.

### 2.4 causality
- **추출:** `event_type` (10종: dialogue, physical_action, movement_or_travel, perception_or_observation, thought_or_emotion, transaction_or_exchange, conflict_or_violence, work_or_creation, state_description, other) · `aspect` ∈ {inceptive, ongoing, completive}
- **리셋:** `event_type` 변함 **OR** `aspect` == `inceptive`
- **특성:** 상시 발화 축 (Monkey 284/407 = 70%). **정보량이 낮다** — 아래 §3 가중치 실험 참조.

### 2.5 goal — 느린 변수 (carry-forward)
- **추출:** `active_goal` ∈ Stage1 `goals` ∪ {`none`}
- **carry-forward (코드):** `none`이면 직전 명시값 유지
- **리셋:** carry 후 값 변함 **AND** 둘 다 ≠ none

---

## 3. 경계 판정 (deployable 규칙 — 방법마다 다름)

### coordinated count
```
count_t = Σ_k reset_k(t)      # k ∈ {protagonist, place, time, causality, goal}, 0~5
boundary ⟺ count_t ≥ t*
```
**t\* = 3** (dev=Pieman argmax-Score 동결. test에서 재선택 없음.)

### ⚠️ 방법별 자연 규칙 — 동일 규칙 강요 금지 (D-M3)
| 방법 | 신호 성질 | 자연 규칙 |
|---|---|---|
| **M3-S5** | **정수 count (0~5, 6단계)** | **count ≥ t\*** (t\*=3) |
| M1-CosDrift | 연속 스칼라 | μ + cσ (c=1.0) |
| M2-MonoLAM | 연속 스칼라 (FDM 예측오차) | μ + cσ (c=0.0) |

M3에 μ+cσ를 적용하면 **저해상도 정수 신호에 분포 임계가 안 맞아** 불공정하다. (선행 오적용으로 인한 "판정 하향" 보고는 철회됨.)

### 🔴 알려진 결함 — t\* 작동점 (Phase 0의 구조적 한계)
정수 임계값이 **문서 길이에 무관하게** 적용된다. dev(96문장)에서 고른 t\*=3을 긴 test 문서(504·408문장)에 적용하면 **과예측**(49개)이 발생. 사전등록 free-threshold Score에서 3/3 패한 원인.
**→ M5의 sticky-CRP 지속성 사전(κ)이 이를 원리적으로 해결한다** (세그먼트 길이가 모델에서 유도됨). Phase 0에서는 한계로 기록.

---

## 3.5 최종 확정 설정의 성능 (deployable, 방법별 자연 규칙)

**설정: 5축 (time = `temporal_shift_marker`만) · t\* = 3 · protagonist = perspective 단독**

| 데이터 | 유형 | **M3-S5** | M1-CosDrift | M2-MonoLAM | 승자 |
|---|---|---|---|---|---|
| pieman (dev) | fiction | **0.634** | 0.568 | 0.391 | **M3** |
| tunnel | fiction | **0.383** | 0.227 | 0.141 | **M3** |
| monkey | journalism | **0.359** | 0.298 | 0.206 | **M3** |
| tiage | dialogue | 0.288 | **0.433** | 0.268 | M1 |
| dialseg711 | dialogue (concat) | **0.655** | 0.616 | 0.285 | **M3** |
| superseg | dialogue | 0.304 | 0.292 | **0.312** | M2 (근소) |
| ami | meeting | **0.425** | 0.143 | 0.129 | **M3** |

**⇒ M3가 7개 중 5~6개 우위. M2-MonoLAM은 전반적으로 최하 — 관문 실패가 deployable에서도 확증.**

> ⚠️ **§7 참조: 이 표는 held-out 성능이 아니다.** 7개 전부 dev로 강등됨(selection-on-test). **탐색적(exploratory) 결과로만 보고할 것.**

### time 축 수정의 효과 (버그 수정 전후)

| variant | t\* | pieman | tunnel | monkey | tiage | dialseg | superseg | ami |
|---|---|---|---|---|---|---|---|---|
| full-5 (원래 time: marker **또는** tense) | 3 | 0.598 | 0.294 | 0.256 | 0.277 | 0.634 | 0.459 | 0.385 |
| **time = marker만 (★확정)** | 3 | **0.634** | 0.383 | 0.359 | 0.287 | 0.655 | 0.304 | 0.425 |
| −time (축 완전 제거) | 3 | 0.628 | 0.398 | 0.340 | 0.287 | 0.655 | 0.304 | 0.425 |

**판단:** `−time`(축 제거)이 tunnel에서 미세하게 높으나(0.398 vs 0.383), **dev에서는 `marker만`이 우수**(0.634 vs 0.628)하고 **EISM 5지표 구조를 보존**한다. 진짜 시간점프 신호가 살아있음을 의미하므로 **`time = marker만`으로 확정.**

### 축 ablation 스윕 (참고 — dev 선택 근거)

| subset | t\* | pieman | tunnel | monkey | tiage | dialseg | superseg | ami |
|---|---|---|---|---|---|---|---|---|
| full-5 | 3 | 0.598 | 0.294 | 0.256 | 0.277 | 0.634 | 0.459 | 0.385 |
| −time | 3 | 0.628 | 0.398 | 0.340 | 0.287 | 0.655 | 0.304 | 0.425 |
| −goal | 3 | 0.546 | 0.387 | 0.273 | 0.343 | 0.477 | 0.267 | 0.353 |
| −place | 3 | 0.566 | 0.287 | 0.259 | 0.265 | 0.561 | 0.483 | 0.331 |
| −causality | 2 | 0.582 | 0.202 | 0.233 | 0.301 | 0.584 | 0.516 | 0.292 |
| −perspective | 2 | 0.471 | 0.213 | 0.199 | 0.310 | 0.643 | 0.487 | 0.297 |
| −time,−causality | 2 | 0.609 | 0.294 | 0.313 | 0.311 | 0.600 | 0.573 | 0.310 |

**⚠️ 이 스윕이 곧 test 오염의 경로다** (§7). 9개 변종의 test 점수를 본 뒤 선택함. **기록은 남기되, 이 수치를 held-out 성능으로 인용하지 말 것.**

---
## 4. 기각된 변형 (재시도 금지 — 근거 있음)

### 4.1 축별 기저율 가중 (−log p) — **기각**
- **의도:** 드문 축의 리셋을 크게, 흔한 축을 작게. pieman 가중치: place 2.94 (p=.05, 드묾) · goal 1.72 · time 1.42 · protagonist 1.03 · **causality 0.28** (p=.76, 흔함). 가중치는 **의도대로 계산됐다**(place 크게, causality 작게).
- **결과: 전반적 악화. 기각.**

| 데이터 | count (기존) | **weighted** | Δ |
|---|---|---|---|
| pieman (dev) | 0.598 | 0.599 | +0.001 |
| tunnel | 0.294 | 0.269 | **−0.025** |
| monkey | 0.256 | 0.243 | −0.013 |
| tiage | 0.277 | 0.327 | +0.050 |
| dialseg711 | 0.634 | 0.632 | −0.002 |
| superseg | 0.459 | 0.373 | **−0.086** |
| ami | 0.385 | 0.205 | **−0.180** |

- **원인:** rare 축 증폭이 **place의 노이즈까지 증폭.**
- **교훈:** **노이즈 축 제거는 이득, rare 축 증폭은 손해.** 둘 다 "축 중요도"를 건드리나 방향이 반대다.

### 4.2 charset(characters_present) 포함 — **기각** (§2.1 참조)

---

## 5. 오염 검증 스팟체크 (필수)

1. Stage1·2 출력에 목록 밖 값·자유서술·"변화/경계/이전" 흔적이 없는가?
2. Stage2에서 문장 순서를 바꿔 다시 뽑으면 동일 결과인가? (target 단독 → 순서 무관해야 함)
3. Stage1 캐스트 목록이 경계 정보를 흘리지 않았는가?
4. carry-forward가 코드에서만 수행되고 LLM 프롬프트에 이전 상태가 유입되지 않았는가?

---

## 6. 평가 프로토콜

- **메인 지표:** DTS Score = `0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)`
- **보조:** AUC (참고용, 메인 아님 — matched-N/AUC는 **사후 진단**이며 "졌다"를 뒤집는 데 쓰지 않는다)
- **인코더 (전 방법 공통, frozen):** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, 768D, L2정규화
- **두 렌즈 모두 보고:**
  - **free-threshold** (각 방법의 자연 규칙, 경계 개수 자가결정) ← **deployable, 사전등록 메인**
  - **matched@gold** (경계 개수를 gold에 맞춤) ← 신호 품질 진단용

---

## 7. ⚠️ 현재 평가 기반의 상태 (필독 — canonical: ledger 스냅숏 + D-M5)

**7개 데이터 전부가 dev로 강등되었다.** 9개 변종의 test 점수를 본 뒤 최종 설정을 골랐으므로 **selection-on-test 오염**이다.

- **현재 통계적으로 방어 가능한 held-out 승리 = 0개.**
- "Tunnel +0.178 승"은 **탐색적(exploratory) 결과**이지 held-out 검증이 아니다.
- fiction held-out doc이 1개뿐 → **bootstrap CI 불가.**

**→ Tier 2 held-out(새 서사 + B2/B3 silver gold, doc 다수) 구축 후, 위 동결 설정으로 1회만 평가한 수치가 논문의 실제 held-out 숫자가 된다.**

---

## 8. 데이터 현황 (전부 dev)

| 데이터 | 유형 | docs | 전이 | gold |
|---|---|---|---|---|
| pieman_michelmann | fiction | 1 | 95 | Michelmann author-filtered (18) |
| tunnel_michelmann | fiction | 1 | 503 | Michelmann author-filtered (18) |
| monkey_michelmann | journalism | 1 | 407 | Michelmann author-filtered (30) |
| tiage | dialogue | 20 | 288 | 화제 전환 |
| dialseg711 | dialogue (concat) | 12 | 310 | 화제 전환 |
| superseg | dialogue | 15 | 167 | 화제 전환 |
| ami | meeting | 2 | 436 | 화제 전환 |

**🔴 Monkey 주석 교란 (미해결):** Pieman/Tunnel은 전체 이야기를 주석했으나, **Monkey는 참가자를 4그룹으로 나눠 각자 1/4씩만 주석**(30/35/31/42명). 합의 구조가 근본적으로 다르다. **"저널리즘이라 실패"라는 scope 주장의 대안 설명이 존재하므로, 4구간 진단이 선행되어야 한다** (ledger D-M5-d).

---

## 9. 실행 인프라 — 대량 라벨링 서버 설정 (2026-07-28 실측)

§1 의 동결 모델(`Qwen3.6-35B-A3B` no-thinking / temp 0 / batch 20)을 **대규모로 돌릴 때의 서버·규모 설정**이다.
스펙(§1~§4)이 *무엇을 뽑는가*라면 이 절은 *어떻게 돌리는가*다. 전부 A100-SXM4-80GB / VESSL betelgeuse-na 실측.
라벨 정의를 바꾸지 않으므로 동결 대상이 아니며, **최신 실측으로 갱신한다.**

### ① 서버 (GPU 1장당 1개)

```bash
# flashinfer 샘플러가 JIT 빌드에 ninja 를 exec 로 찾는다. PATH 없으면 기동 즉사.
# vllm/vllm-openai 이미지에는 python 이 없고 python3 만 있다.
command -v ninja >/dev/null || pip install -q ninja
export PATH="$(python3 -c 'import sys,os;print(os.path.dirname(sys.executable))'):$PATH"

python3 -m vllm.entrypoints.openai.api_server \
  --model /shared/models/Qwen3.6-35B-A3B \   # HF ID 아닌 절대경로 → 허브 접근 0
  --served-model-name qwen36-35b \
  --quantization fp8 --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 512 \
  --reasoning-parser qwen3 --no-enable-log-requests \
  --enable-prefix-caching --load-format runai_streamer --port 8001
```

vLLM 0.25.1 / 이미지 `vllm/vllm-openai:v0.25.1` (pull+import 19초).
모델 = `Qwen/Qwen3.6-35B-A3B` rev `995ad96eacd98c81ed38be0c5b274b04031597b0`. **체크포인트는 bf16 67GB 하나뿐이며 fp8 은 별도 파일이 아니라 로드 시 옵션이다.**
클라이언트 = `--engine vllm --model qwen36-35b --no-reasoning --prompt-version v3.1` (frozen = no-thinking).
**로그는 반드시 stdout.** geesefs 는 파일을 닫기 전까지 다른 마운트에서 안 보여 파일 리다이렉트 시 실시간 관찰이 불가능하다.

### ② 실측치

| 항목 | 값 |
|---|---|
| 가중치 (TP=1) | **33.89 GiB** → 80GB 1장에 적재. 40GB 불가, L40S 48GB 는 KV 7GiB 만 남아 비권장 |
| KV | 29.99 GiB = **1,283,125 토큰**. KV 벽 ≈ 동시 320 |
| 복제 1개 처리량 | TP=1 **6.01** / TP=2 7.88 / TP=4 9.45 콜/s (@conc 200) → **TP 를 키우면 통신이 지배, 복제를 늘리는 쪽이 선형** |
| 실전 처리량 | 동시성 250, 20장 합계 **58 콜/s = 복제당 2.90 콜/s** (본run 실측; 단일 스모크 3.55 보다 낮음 = 꼬리) |
| Stage-2 프롬프트 | 약 1,000 tok, 배치 간 공유 접두 **27%** (본문 없음 — 스토리 전문은 Stage-1 만) |
| 스토리지 | cephfs 871 MB/s · geesefs **162 MB/s(워크스페이스) / 30 MB/s(Job 컨테이너)** |

### ③ 동시성 — 가장 자주 틀리는 부분

**동시 요청 수 = 병렬로 도는 (샤드×축) 작업 수. `CONC` 는 상한선일 뿐이다.**
`llm_s5_states.py` 프로세스 하나는 요청을 한 번에 하나씩만 보낸다.

```
작업 10개에 CONC=250  →  실제 동시성 10. GPU 놀음.
Job N개 × 동시성 250  →  필요 작업 250N개
예) 20장 → 5,000작업 → 1,000샤드 × 5축 → 10,645편이면 샤드당 약 11편
```

**GPU 를 늘릴 때 샤드도 같이 잘게 쪼갠다.** 안 하면 20장이 10장만도 못하다.
권장 동시성 **250** (300에서 KV 98.5%. 포화 구간이라 250과 처리량이 사실상 같으면서 여유를 얻는다).

### ④ 출력 — 컨테이너 로컬에 쓰고 주기 동기화

geesefs 에 직접 쓰면 클라이언트가 막혀 **동시성이 300 → 78 로 주저앉는다.**

```bash
WORK_OUT=/tmp/hiontop_out
mkdir -p "$WORK_OUT"
[ -d "$OUT_DIR" ] && cp -ru "$OUT_DIR"/*/ "$WORK_OUT"/ 2>/dev/null          # ① resume seed
( while true; do sleep 1200; cp -ru "$WORK_OUT"/*/ "$OUT_DIR"/ 2>/dev/null; done ) &  # ② 20분 주기
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; cp -ru "$WORK_OUT"/*/ "$OUT_DIR"/ 2>/dev/null' EXIT  # ③ 종료 시
```

없으면 Job 이 죽을 때 그 몫 전부 소실, 있으면 최대 손실 20분.
resume 자체는 `llm_s5_states.py` 에 이미 있다 — 배치 20문장마다 축 파일을 통째로 다시 쓰고 재시작 때 `item_id` 로 이어간다. ①②③은 **그 파일이 컨테이너와 함께 사라지지 않게 하는 것**뿐이다.
N개 Job 이 같은 `OUT_DIR` 에 써도 (문서×축) 파일이 겹치지 않아 안전하다.

### ⑤ Job 제출

```bash
vesslctl job create -n <RUN>-shard<i> -r resourcespec-a100x1 \
  -i vllm/vllm-openai:v0.25.1 \
  --object-volume objvol-gsvyr0eu87wt:/shared --working-dir /shared/hiontop/code \
  -e SHARD_ID=<i> -e NUM_SHARDS=<N> -e CONC=250 \
  -e MODEL_PATH=/shared/models/Qwen3.6-35B-A3B -e OUT_DIR=/shared/hiontop/out/<RUN> \
  --cmd "bash /shared/hiontop/job_shard_v<버전>.sh"
```

$1.55/hr/장 (20장 = $31/hr). **20장 동시 확보 검증됨** (제출 2분 내 전원 running).
private repo 라 Job 안에서 git clone 불가 → 코드·데이터를 `/shared` 에 스테이징한다.

### ⑥ 소요시간 참고표 — 다음 런 규모산정용

**실적 (2026-07-28, 10,645편 · 696,590콜 · A100 20장 · 동시성 250, KST):**

| 구간 | 시각 | 소요 |
|---|---|---|
| 제출 → 전원 ready | 05:58 → 06:09 | **11분** (20장 동시 로딩, runai_streamer + geesefs) |
| 라벨링 | 06:09 → 09:18 | **3시간 20분** |
| 결과 회수 (축파일 53,184개 복사) | 09:18 → 13:28 | **4시간 10분** ← 라벨링보다 오래 걸림 |
| `reset_matrix` 조립 | 13:28 → 13:32 | 4분 |
| **총** | 05:58 → 13:32 | **7시간 34분** |

**추정 공식 (다음 런에 그대로 대입):**

```
총 콜 수    ≈ 문서수 × 65          (5축 · 배치20 기준. 실적 696,590 / 10,645 = 65.4)
라벨링 시간  ≈ 총 콜 수 / (2.9 × GPU수)  초
기동         11분 (GPU 수와 거의 무관 — 20장 동시에도 8→11분)
회수 시간    ≈ 축파일수 × 0.285초   (축파일수 = 문서수 × 5)  ← tar 로 묶으면 제거됨
```

예) 10,000편 · 20장 → 콜 65만 → 라벨링 3.1시간 + 기동 11분 + 회수 4시간(tar 시 수 분).

**⚠️ 회수 비용은 바이트가 아니라 파일 수에 비례한다.** geesefs 는 파일당 약 285ms 가 든다.
53,184개를 하나씩 복사해 4시간 10분이 걸렸고, 이는 "30분" 추정의 8배였다.
**다음 런은 Job 안에서 `tar` 로 묶어 한 덩어리로 올린다** (D-52).

### ⑦ 건드리면 안 되는 것

| 항목 | 이유 |
|---|---|
| `--max-model-len 32768` | 16384 면 Stage-1 컨텍스트 초과로 **278편 탈락** |
| `--reasoning-parser qwen3` | 없으면 guided `json_schema` 가 첫 토큰부터 JSON 을 강제해 추론이 억제됨 |
| `--quantization fp8` (on-the-fly) | 사전 양자화본으로 바꾸지 말 것. 이득 60초인데 수치가 달라짐 |
| `frequency_penalty` | **쓰지 말 것.** 품질검사 12편 전부 cast 변경(동일 0/11, Jaccard 0.498), 진짜 개체 대량 삭제(24→5), 멀쩡한 문서가 파싱 실패로 전환 |

**축·코퍼스 중간에 설정 변경 금지.** vLLM 연속 배칭은 비트 결정적이 아니라 배치 구성·캐시 히트·복제 수가 바뀌면 출력이 미세하게 달라진다. 변경은 새 코퍼스 시작 시점에만. **이전 런 결과를 이어받는 것도 같은 기준으로 판단한다.**

### ⑧ 운영 함정 (전부 실제로 밟음)

| 함정 | 증상 / 대처 |
|---|---|
| ninja / PATH | venv 활성화 없이 python 직접 호출 시 FlashInfer JIT 실패로 **기동 즉사** |
| `python` 없음 | vllm 이미지에는 `python3` 만 존재 |
| 실행 중 스크립트 수정 | bash 가 바이트 오프셋으로 읽어 엉뚱한 지점으로 튄다. **버전 파일명 고정** |
| `pkill -f` | 자기 명령줄까지 매치해 자폭(exit 144). PID 로 죽이고 확인은 bracket 트릭 `grep "api_serve[r]"` |
| `job list` 20행 제한 | `--per-page 30` 필수. 안 붙이면 `run-0` 이 안 보여 죽은 줄 오해 |
| `job logs` | 이 클러스터에서 사실상 무응답. 관찰은 stdout 마커나 파일로 |
| geesefs 가시성 | 파일을 닫기 전까지 다른 마운트에서 안 보임 → **컨테이너 간 실시간 파일 관찰 불가** |
| `terminate` 차단 | 워크스페이스 안에서 불가. 사이드카가 PID 1 에 `VESSLCTL_ACCESS_TOKEN` 을 주입하고 credentials 보다 우선하며 `env -u` 로도 우회 불가. **밖이나 웹 콘솔에서만**, `-y` 없으면 프롬프트에서 멈춤 |

### ⑨ 미검증

- TP=1 다복제 합계 처리량의 선형성 (복제당 값의 산술 투영만 있음)
- prefix caching 의 end-to-end 실효 (공유 접두 27% 는 정적 분석치)
- 클러스터 볼륨(cephfs) 경로 — 복사 Job 80분째 검증 불가로 중단. 이득은 기동 11분 → 2~3분(20장 기준 약 $4)뿐이라 우선순위 낮음

---
