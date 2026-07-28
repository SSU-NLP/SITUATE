# REPORT — Monkey 4구간 진단 (ledger D-M5-d)

## 1. 실험 setup

- **목적.** Monkey 실패의 두 경쟁 설명을 판별한다.
  - (H-genre) 저널리즘(스튜디오 내레이션)이라 상황모델(무대·시간)이 사람 경계에서 리셋되지 않음 → 우리 방법이 검출할 신호가 없음 (우리 scope 주장).
  - (H-confound) Monkey는 참가자를 **4그룹으로 나눠 각 그룹이 이야기의 1/4씩만** 주석(30/35/31/42명)했다. 합의 구조가 구간마다 달라 성능이 교란됐을 수 있음 (scope 주장의 대안 설명).
- **판정 규약 (실행 전 고정).** 구간별 신호질이 **들쭉날쭉 → H-confound**(scope 재작성) / **균일하게 약함 → H-genre**(scope 지지).
- **데이터.** `outputs/eacl_phase0/monkey_michelmann_s5/reset_matrix.jsonl` (407 전이, S5 frozen config). gold A = author-filtered (30 경계), gold B = raw+5s (40 경계).
- **구간.** 전이 위치 균등 4분할: Q1 1–102 / Q2 103–204 / Q3 205–306 / Q4 307–407. (⚠️ 실제 주석 그룹 경계는 미확인 — 위치 균등분할로 근사. 아래 §5.)
- **방법.** ours = coordinated 5축 count(protagonist_perspective+place+time+causality+goal). IDM_raw = 인접 문장 mpnet 코사인 거리(fit 0개). IDM_tuned = HiOnTop delta_eff.
- **지표.** matched@gold Score(`dts_scoring.score_dialogues`, oracle 경계 개수 = **ranking 진단**) + AUC(보조) + 축별 AUC/lift. deployable free-threshold는 구간별 K가 작아 노이즈가 커 제외.
- **인코더.** `paraphrase-multilingual-mpnet-base-v2`, 768D, L2정규화 (전 방법 공통).
- **스크립트.** `scripts/eacl_phase0/monkey_quarter_diagnosis.py`. seed 무관(결정론). 단일 실행.

## 2. 결과

### 전체 doc 기준선

| gold | ours Score | IDM_raw | IDM_tuned | ours AUC | IDM_raw AUC | IDM_tuned AUC |
|---|---|---|---|---|---|---|
| A (K=30) | 0.396 | 0.274 | 0.275 | 0.597 | 0.506 | 0.525 |
| B (K=40) | 0.377 | 0.335 | 0.381 | 0.608 | 0.538 | 0.556 |

### 구간별 ours AUC (핵심 균일성 지표)

| | Q1 | Q2 | Q3 | Q4 | range |
|---|---|---|---|---|---|
| gold A | 0.564 | 0.597 | 0.603 | 0.677 | **0.113** |
| gold B | 0.618 | 0.599 | 0.601 | 0.677 | **0.078** |

**ours는 4구간 모두 near-chance~약(0.56–0.68)로 균일.** 어느 구간도 "작동" 영역(Pieman 0.739·Tunnel 0.642 수준의 깨끗한 신호)에 도달하지 않음. Q4가 최고(0.677)나 그 신호는 place가 아니라 protagonist/causality에서 옴(아래).

### 구간별 축 AUC (place = scope 판별자)

| gold A | protagonist | **place** | time | causality | goal |
|---|---|---|---|---|---|
| Q1 | 0.537 | **0.500** | 0.523 | 0.563 | 0.480 |
| Q2 | 0.597 | **0.480** | 0.480 | 0.638 | 0.439 |
| Q3 | 0.523 | **0.473** | 0.500 | 0.608 | 0.525 |
| Q4 | 0.647 | **0.540** | 0.540 | 0.599 | 0.476 |

- **place는 4구간 전부 死** (AUC 0.47–0.54, reset 총계 0/4/5/7로 거의 침묵). 무대가 어느 구간에서도 리셋 안 함.
- **time도 4구간 전부 死** (≤0.54).
- causality만 유일하게 미약한 lift(0.56–0.64)이나 상시 발화 축(경계율≈내부율, 거의 안 갈림).

### IDM의 구간 변동은 오히려 ours보다 큼

IDM_raw AUC: Q1 0.409 → Q2 0.329 → **Q3 0.685** → Q4 0.527 (range 0.356). Q3에서 IDM만 튀는데, ours는 Q3에서도 평평(0.603) — 주석 품질이 구간별로 달랐다면 두 방법이 함께 흔들려야 하나, **ours는 평평하고 IDM만 Q3에서 상승** → Q3 경계가 "임베딩으로 감지되는 내용 전환"일 뿐 상황모델 리셋은 아님.

## 3. 해석

- **판정 규약 적용: ours가 4구간 균일하게 약함(range 0.078–0.113, 전부 near-chance) → H-genre 지지, H-confound largely 기각.**
- 주석 품질 교란이 실패의 지배 원인이라면, 참가자 많은 구간(Q4=42명)은 좋고 적은 구간(Q3=31명)은 나빠야 한다. 실제로는 **모든 구간이 균일하게 약하고, 특히 scope 판별자 place가 4구간 전부 死.**
- Q4의 상대적 상승(0.677)은 place가 아니라 protagonist/causality 기여이며, **이론이 지목한 메커니즘(무대 리셋)은 어느 구간에서도 작동하지 않음.**
- ⇒ Monkey 실패는 "우리가 못 잡은 것"이 아니라 **"거기엔 상황모델 리셋이 없다"**는 구조적 사실. 이전 눈검증(경계 10개 중 9개 place 무변화)과 일치.

## 4. 판정

- **H-confound(주석 프로토콜 교란) = 실패의 지배 원인 아님.** scope 주장("narrative fiction에서 작동, 저널리즘/화제전환에서 이론상 불성립")의 **대안 설명이 배제됨** → scope 해석 **강화·유지**.
- Monkey는 사전 예측된 실패조건(상황모델 정지 텍스트)으로 정직 보고 가능. D-M5-d 해소.

## 5. 한계 / 검증 미해결

1. **구간 경계 근사.** 위치 균등 4분할이며, 실제 4개 주석 그룹의 정확한 story 분할점은 미확인(`sourcedata/`의 per-participant 매핑 미파싱). 정밀화하려면 원 주석 그룹 경계로 재분할해 재확인. 단 어떤 합리적 분할에서도 place 全구간 死라는 결론은 견고할 것으로 예상.
2. **구간당 K 작음** (4–14) → 구간별 matched Score는 노이즈 큼. 그래서 판정은 **AUC 균일성**을 주지표로 삼음(Score는 보조).
3. **약한 Q1→Q4 gradient(0.11)** 존재. 주석자 수와 약상관 가능성 배제 못 하나, 모든 구간이 near-chance라 실패의 지배 원인은 아님.
4. gold A/B 결론 동일(둘 다 균일 약) → gold 정의에 견고.
