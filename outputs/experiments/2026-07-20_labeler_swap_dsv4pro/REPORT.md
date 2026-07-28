# anchor 라벨러 교체 검증 — claude-fable-5 → deepseek-v4-pro

> 실행 2026-07-20~21. 판정 스크립트 `scripts/eacl_phase0/labeler_agreement.py`, 원자료
> `agreement.json`. 추출 러너 `llm_s5_states.py`(모델만 교체). 발단 = fable5($10/$50 per MTok)
> 비용 절감. **판정: 교체 불가. 3편 전부 FAIL.**

## 1. 실험 setup

**목적.** M3-S5의 anchor 라벨러를 claude-fable-5에서 openrouter/deepseek/deepseek-v4-pro로
교체 가능한지 검증. 라벨러는 M4/M5가 학습하는 측정 도구이므로(codex 2026-07-20, D-M13 참조),
교체는 "구현 교체"가 아니라 "측정도구 변경".

**데이터.** dev 3편(pieman/tunnel/monkey), Michelmann 인간 gold. dev는 D-M5-a로 이미 소진돼
무제한 열람 가능. **신규 3편(시험지)은 봉인 유지 — 이 검증에 미사용.**

**방법.** "변형 B" = DeepSeek이 Stage 1(장소/인물/목표 후보 목록) + Stage 2(문장별 상태)를 전부
재생성. 동결 설정(5축, time=temporal_shift_marker만, protagonist=perspective 단독, t\*=3) 그대로.
새 out_dir `*_michelmann_s5_dsv4pro`(resume 오염 방지). Stage 1 max_tokens는 monkey/tunnel에서만
2000→8000(deepseek reasoning 토큰이 예산 초과 → null content; pieman은 2000으로 통과).

**판정 구조(codex 검토 반영, `outputs/reports/2026-07-20_codex_labeler_swap_review.md`).**
Gate A(실행 무결성) → Gate B(5축 동등성) → **Gate C(최종 M3 경계 + 사람 gold 대비 Score 비열등성).**
Gate C가 승인 게이트, place 4종 진단은 실패 원인 규명용으로 격하.

**Gate C 사전 고정 기준(bidirectional):** boundary_f1(A↔B) ≥ 0.80, ΔScore(사람 gold) ≥ −0.03,
t\* 횡단률 ≤ 0.10.

## 2. 결과

### 2-1. 주지표 — 사람 gold 대비 DTS Score

| doc | fable5-M3 | deepseek-M3 | ΔScore | A↔B 경계F1 | 판정 |
|---|---|---|---|---|---|
| pieman | 0.6344 | 0.3811 | **−0.2534** | 0.480 | ❌ |
| tunnel | (agreement.json) | | **−0.1508** | 0.387 | ❌ |
| monkey | 0.3586 | 0.3067 | **−0.0519** | 0.261 | ❌ |

세 편 모두 ΔScore가 기준(−0.03)을 초과. pieman은 사람 gold 대비 boundary F1이 **0.545 → 0.214**로
반토막.

### 2-2. Gate B — 축별 리셋 F1 (동등성)

| 축 | pieman | tunnel | monkey |
|---|---|---|---|
| protagonist_perspective | 0.800 | — | 0.848 |
| place | 0.889 | — | 0.467 |
| time | 0.778 | — | 0.788 |
| causality | 0.730 | — | 0.686 |
| **goal** | **0.437** | — | **0.381** |

**5비트 리셋 벡터 완전일치율: pieman 0.432 / monkey 0.447.** 축별 총량은 비슷한데 같은 전이에서
함께 터지는 조합이 다름(co-reset 붕괴). 예: pieman `protagonist+goal` 동시리셋 11→4.

### 2-3. 핵심 실패 사례 — monkey place

monkey 도입부는 전체가 인도네시아 숲(사진가가 원숭이 추적) 장면. fable5는 `sulawesi_rainforest`
유지, DeepSeek은:

| 문장 | 원문 | fable5 | deepseek |
|---|---|---|---|
| 5 | "traveled there from his home in **England**" | sulawesi_rainforest | **england** |
| 6,10,18 | (숲 장면 계속, 장소 단어 없음) | sulawesi_rainforest | **unspecified** |

문장5의 잘못된 `england`가 carry-forward로 뒤 문장에 전파 → 숲 장면이 영국으로 오염. place
unspecified가 82.4% → 96.3%로 폭증(축이 사실상 사망), reset/specified 0.22 → 0.93.

## 3. 해석

**모델만 바뀐 것이 검증됨(오염 아님).**
1. 리셋 규칙 코드 = 동일: fable5 라벨을 현재 코드로 재계산(`reconstruct_reset_matrix.py`) →
   기존 fable5 matrix와 전이 95개 중 0개 차이(`pieman_recon.jsonl`).
2. 입력 master = 동일(md5 확인, 같은 문장·gold).
3. prompt_sha256 차이는 설계상 정상 — Stage 2 프롬프트에 Stage 1 목록이 인라인되므로 목록이
   다르면 sha가 다름(변형 B의 정의). 프롬프트 템플릿 코드는 불변.

**근본 원인 = 스펙 해석 차이.** S5 프롬프트 "where THIS sentence is situated; unspecified if none
is expressed"를 두 모델이 다르게 읽음:
- fable5: "이 **장면**이 어디서 벌어지나" (서사 연속성 이해 → carry-forward와 정합)
- deepseek: "이 **문장**에 어떤 장소 단어가 있나" (문자적 → 연속 장면을 unspecified, 스쳐가는
  지명을 집음)

S5는 장면 연속성 추적 과제라 전자가 맞음. 이 차이가 5축 전반에 퍼져 co-reset 구조를 무너뜨림.

**⚠️ 어제 예측의 반증.** 사전 우려는 "똑똑한 모델일수록 추론해서 채운다(냉장고→부엌)"였으나,
DeepSeek은 반대로 **덜 채우고**(unspecified 폭증) 문자적으로 집었다. 실패 방향이 예상과 반대.

**place만 보면 통과처럼 보이는 함정.** pieman place 리셋 F1 0.889, false-move 0으로 place 축만
보면 우수. 그러나 5축 전체·최종 경계를 보면 pieman이 ΔScore −0.25로 셋 중 최악. codex/사용자의
"Score부터 보라"가 실증됨 — 주변 통계(place 4종)로는 라벨러 교체를 판정할 수 없음.

## 4. 판정

- **deepseek-v4-pro 교체 = 기각.** 3편 전부 Gate C FAIL, 최악 문서 기준(codex) pieman ΔScore −0.25.
- **더 작은 모델(Flash 등)로 개선 기대 = 낮음.** 실패가 −0.05가 아니라 −0.25 규모이고, 원인이
  능력이 아니라 스펙 해석 방식이라 모델 크기와 무관.
- **다음 = 프롬프트 재설계.** fable5가 예산 밖이므로 저비용 모델이 장면 연속성을 지키도록 프롬프트를
  손볼 수밖에 없음. 단 이는 fable5 재현성을 깨므로 **별도 사전등록 + confirmatory/robustness 분리
  보고 필요**(codex Q5). 프롬프트 튜닝은 dev 3편에서만, 시험지는 봉인 유지.
  - 핵심 수정 방향(가설): place/goal에 carry-forward 규칙을 프롬프트 안에 명시(현재는 사후 코드만),
    "스쳐가는 지명 vs 현재 무대" 구분 지시, goal 축 재정의(F1 최저 0.38).

## 5. 한계 / 검증 미해결

1. **tunnel fable5 Score 미기재** — 표 2-1에서 agreement.json 참조로 대체(스크립트가 tunnel의
   fable5 절대 Score를 콘솔에 출력했으나 본 REPORT 작성 시 미전사). json에 있음.
2. **Stage 1 max_tokens 비대칭** — fable5 전편 2000, deepseek는 pieman 2000 / tunnel·monkey 8000.
   max_tokens는 상한이라 출력 내용 불변(truncation만 방지)이고, 둘 다 2000인 pieman이 최악이므로
   결과를 뒤집지 않음. 그러나 엄밀한 동일 조건은 아님.
3. **place 문자열 리터럴 비교** — `duane_library_steps` vs `duane_library`를 불일치로 계수.
   의미 정렬 미수행이라 specified-only agreement는 하한(codex Q1). 단 Gate C(최종 Score)는 이
   문제와 무관하게 FAIL이므로 판정에 영향 없음.
4. **DeepSeek이 "틀렸다"는 증명은 아님** — Gate C는 fable5 재현이 아니라 사람 gold 성능으로 판정.
   deepseek-M3가 사람 경계를 못 맞추는 것(boundary F1 vs human: pieman 0.545→0.214)이 기각 근거.
5. **단일 실행** — temperature=0이나 DeepSeek 비결정성 미측정. 재현 변동성 미확인.
