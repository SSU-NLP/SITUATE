# causality Zacks-halo 채택 테스트 — 사전등록 (PRE-REGISTRATION)

> **결과 보기 전 봉인.** D-M5 / no-postresult-criterion-change 규약: 아래 통과조건은 결과를 보고
> 고치지 않는다. 위반 시 기각으로 적고 사유만 남긴다.

## 배경 / 가설

현행 causality reset = `a["event_type"] != b["event_type"] or b["aspect"]=="inceptive"` — 문장별
target-only 라벨의 **코드 비교 프록시**. event_type 불일치가 인과 단절의 노이즈 프록시라 대화↔행동
교대만으로 발현 → **~70% 포화**(경계율 ~7%의 10배).

Zacks et al.(2009, *One Boy's Day* / Red Balloon)의 필드 표준 causal 코딩 = **halo**: "그 절의 활동이
**직전 절의 활동에 의해 직접 야기되지 않았을 때** 경계." 두 절을 함께 보고 인과 관계를 직접 판단.
(Zacks는 target-only가 아니라 (a) 차원코딩·경계지각 참가자 분리, (b) 훈련코더+조작정의, (c) 시간점프·
컷어웨이 없는 재료로 순환을 회피 — 우리 "no halo"의 LLM 확인편향 회피와는 별개 기전.)

**가설**: halo-causality(직전 절 대비 직접 인과 판단)가 event_type 프록시보다 인과를 정확히 따라가고,
발현율이 정상 범위로 내려오며, 최종 Score를 유의하게 깎지 않는다.

## 재료 / 설정 (동결)

- 라벨러: frozen Qwen3.6-35B-A3B no-thinking v2, local vLLM :8001, temperature=0.
- halo 질의: transition별 `previous`/`current`(마스터 그대로)에 대해
  **"CURRENT의 주 활동이 PREVIOUS의 주 활동에 의해 직접 야기/직접적 인과결과인가?"** → JSON
  `{"caused_by_previous": true/false}`. **boundary/change/segment/reset/이전 정보 언급 없음** — 인과
  관계만 물음(halo이되 "바뀌었나?"가 아님). break = `not caused_by_previous`.
- 데이터: (primary) 통제 discordant probe `discordant_probe.jsonl`(축별 정답 주석).
  (sanity/guardrail) repcorpus stage0 12편 dev 추출(`repcorpus_masters_stage0/*.jsonl`).
- 스크립트: `scripts/eacl_phase0/halo_causality.py`. 원자료 이 디렉토리.

## 통과조건 (사전선언 — 3개 전부 충족해야 채택 검토)

1. **primary = discordant에서 인과를 따라감**
   장면축(space/protagonist)이 "연속"인데 인과가 끊긴 자리(**within-scene 인과단절**)를 halo가 "단절"로 판정.
   - 조작화: within-scene-causal-break 항목에서 halo 단절율 **≥ 0.80**,
     AND within-scene-causal-continuous(같은 장면 인과연쇄) 항목에서 오발현율 **≤ 0.20**.
   - (대조로 현행 event_type 프록시를 같은 항목에 적용해 병기 — 프록시는 인과가 아니라 유형변화를 따라감을 보임.)

2. **sanity = 발현율 정상대역**
   dev 12편 halo 발현율이 경계율(~7%)보다 **확실히 높고**, 현행 프록시(~70%)보다 **확실히 낮음** — 대략 **[0.15, 0.40]**.

3. **가드레일 = Score 방어**
   halo-causality를 reset_matrix의 causality 열에만 스왑(나머지 4축·threshold 동결) 후 deployable Gate C
   DTS Score 하락이 **−0.09 이내**(halo Score ≥ 현행 Score − 0.09, 동일 docs·동일 t*).

## 판정 규약

- 3개 전부 충족 → 채택 후보(Gate C 정식 재확인). 하나라도 위반 → **기각**, 결과대로 기록(기준 불변경).
- fallback = 현행 event_type 프록시 유지 또는 causal_marker 후퇴(별도).

## 한계 (사전 명시)

- discordant probe = hand-crafted 소표본(낙관 편향). 통과해도 실전은 더 어렵다.
- halo는 컷어웨이-복귀(장면점프 후 인과 재개)를 "단절"로 발현 — Zacks 재료엔 없던 구조. 희소(divider ~0.45%
  수준)라 known limitation으로 수용. primary는 이 케이스가 아니라 **within-scene 인과단절**로 판정.
- LLM 확인편향(halo가 "단절"을 과보고) 리스크는 sanity 발현율 대역 + Gate C Score로 통제.
