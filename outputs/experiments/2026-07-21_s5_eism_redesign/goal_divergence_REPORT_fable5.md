# goal-divergence 진단 (관문①-a)

- primary dirs: `outputs/eacl_phase0/pieman_michelmann_s5`, `outputs/eacl_phase0/tunnel_michelmann_s5`, `outputs/eacl_phase0/monkey_michelmann_s5` (keys: legacy place/goal)
- dir-b: (없음 — 발산 측정 skip)
- docs: pieman, tunnel, monkey
- t\*=3

## 1. 식별가능성 (active_goal none-rate; space 대비)

| doc | int none-rate | int specified | space none-rate |
|---|---|---|---|
| pieman | 0.594 | 0.406 | 0.906 |
| tunnel | 0.506 | 0.494 | 0.921 |
| monkey | 0.730 | 0.270 | 0.824 |

## 3. 취약성 (deployable Score: 5축 vs intentionality 제거 4축)

| doc | 5축 Score (t*) | −int 4축 Score (t*) | Δ(=int 기여) | 5축@t\*=3 | −int@t\*=3 | Δ@3 |
|---|---|---|---|---|---|---|
| pieman | 0.634 (t*=3) | 0.513 (t*=2) | +0.122 | 0.634 | 0.486 | +0.149 |
| tunnel | 0.383 (t*=3) | 0.419 (t*=3) | -0.037 | 0.383 | 0.419 | -0.037 |
| monkey | 0.359 (t*=3) | 0.366 (t*=3) | -0.008 | 0.359 | 0.366 | -0.008 |

## 요약 (Q5 결정은 codex 위임; 수치만)

- 식별성: intentionality none-rate 0.51~0.73 (space 死변수 기준 ~0.86 대비 참고).
- 취약성: intentionality 기여 ΔScore -0.037~+0.122 (양수·유의하면 '신호 실재하나 취약' → 죽이지 않는 재정의 필요).
- 발산: dir-b 제공 시 κ 낮으면 라벨러 불안정 → 재정의가 안정화해야 함.

> ⚠️ 이 REPORT는 **fable5 구 산출(place/goal, D-M5 오염·dev-strength) baseline**이다. Q5 최종 결정은 deepseek eismv2 재추출 + 발산(κ) 측정 후 codex 위임.

## 해석·한계 (사용자 정정 2026-07-21)

1. **부호 패턴 (+,−,−)은 새 발견이 아니다** — `phase0-annotation-guide.md` 축 ablation 표에 이미 있음(pieman +0.052 / tunnel −0.093 / monkey −0.017). **새로운 것은 크기**: pieman +0.052→+0.149(≈3×), tunnel −0.093→−0.037. 원인 = **base config 차이**(옛 표는 time=marker-or-tense 0.598 위, 이번은 동결 config 0.634 위). **time에서 tense 제거가 goal 중요도를 3배로 키움 = 축들이 독립적이지 않다는 증거.** 가이드 §6.3 "축별 독립 판정·2×2 조합 전수탐색 금지"(오염통제)와 **긴장 관계**. → **한계 기록: 축 기여는 다른 축 설정에 의존한다**(지금은 통제 우선이 맞으나 상호작용 존재를 명시).

2. **pieman-단독 이득 = 오염 신호 가능성.** D-B6: Pieman=dev, t\*=3도 dev=Pieman argmax-Score 동결, goal carry-forward도 pieman 관찰 유래. goal이 유독 pieman에서만 크게 이득인 것은 **선택이 pieman에 맞춰졌을 때 정확히 나타나는 패턴**. → **Q5 재정의 판단에 pieman 개선을 근거로 쓰지 말 것. 실질 기준 = tunnel/monkey 음수를 ≥0으로 끌어올리는가.**

3. **none-rate는 건강 지표가 아니다.** space는 none-rate가 더 높은데도(0.82~0.92) **carry-forward가 공백을 메워 더 잘 작동**. goal의 낮은 none-rate가 곧 건강이 아님. **다만 결론("식별은 되나 신호 품질이 문제")은 유효** — ledger goal F1 0.38이 그것을 가리키고, 이 진단은 **"식별 불가 가설"을 배제**해줌(그 프레이밍으로 사용).
