# 노이즈 floor 재분석 — temp-0 재현성이 관문① 결정들을 무효화하는가 (D-B11 부록)

> 작성 2026-07-23. 촉발 = D-B11 결정성 발견(경계재현 유명 0.865/무명 0.981 = temp-0인데 ~14% 뒤집힘).
> 질문(codex): "이 노이즈가 기존 판단들을 무효화하나? 노이즈 크기를 모르면 앞으로 어떤 숫자도 해석 못 한다."
> 러너 = `scripts/eacl_phase0/noise_floor_run2run.py`(self-agreement) · `noise_propagate_pieman.py`(전파).
> **예산 0** — 기존 run1/run2(D-B11) + dev-3 frozen reset_matrix(gold 보유)만 사용.

## 1. 두 노이즈 추정치 — 반드시 구분

| 추정 | 값 | 무엇을 재나 | ΔScore 판단에 적합? |
|---|---|---|---|
| **self-agreement floor** = 1−S(run1↔run2) | **≈0.44** (유명 3편) | 두 run의 경계집합이 서로 Score-공간서 얼마나 벌어지나 | ❌ **과대평가** — run1을 target 삼으면 run1=1.0 자동, 격차 부풀림 |
| **전파 √2σ** (gold 대비) | **≈0.060** (single-run σ≈0.043) | 같은 config를 재추출했을 때 **외부 gold 대비 Score**가 얼마나 움직이나 | ✅ **결정 판단용 정답** — Gate C의 gold는 두 run과 독립 |

**왜 self-agreement를 쓰면 안 되나:** Gate C의 ΔScore = S(deepseek,gold) − S(fable5,gold)는 **외부 human
gold** 대비 차이다. self-agreement는 run1 자신을 target으로 써서 boundary-F1의 원천적 가혹함(off-by-one이
FP+FN 이중계상)까지 노이즈로 오인한다. 전파법은 **관측된 축별 flip rate를 실제 gold-scored dev-3에 주입**해
"재추출 시 Score가 얼마나 흔들리나"를 직접 잰다.

## 2. 측정 (전파법, 몬테카를로 N=3000, seed 고정)

**축별 flip rate** (유명 3편 run1 vs run2 공유전이):
protagonist 0.206 · space 0.181 · time 0.027 · **causality 0.321** · intentionality 0.145.
(causality가 최대 flip — over-fire + 최저 재현의 이중약점 재확인.)

**dev-3 gold 대비 Score 분포** (실제 frozen 축컬럼에 위 flip 주입, count≥t*=3, 각 doc human gold 채점):

| doc | gold경계 | base Score | MC평균 | **σ** | 2.5% | 97.5% |
|---|---|---|---|---|---|---|
| pieman | 18 | 0.593 | 0.478 | **0.061** | 0.358 | 0.599 |
| tunnel | 18 | 0.461 | 0.258 | 0.034 | 0.197 | 0.328 |
| monkey | 30 | 0.286 | 0.264 | 0.033 | 0.206 | 0.333 |

**mean single-run σ = 0.043 → two-run ΔScore SD ≈ √2σ = 0.060 → ~2σ_pair ≈ 0.12.**

**독립 교차검증(flip 모델 무관):** 유명 3편 실제 경계수 run1→run2 = london **54→29**, doyle 15→10,
gilman 32→33. 일부 doc은 재추출만으로 경계수 **~2× 요동** → 그런 doc의 σ는 0.06보다 훨씬 큼(꼬리 존재).

## 3. 판정 — 관문① 결정들 vs √2σ=0.060

| 결정 | \|Δ\| | z=Δ/√2σ | 판정 | 함의 |
|---|---|---|---|---|
| **Gate C pieman** (ds 0.593 vs fb 0.634) | 0.041 | 0.7 | **노이즈 내** | "strict 미달"이 **실재하는 회귀가 아닐 수 있음** — 수용 강화 |
| Gate C monkey | 0.072 | 1.2 | ~1–2σ 모호 | 미달이 확정적이지 않음 |
| Gate C tunnel (+pass) | 0.078 | 1.3 | ~1–2σ 모호 | pass도 확정적이지 않음 |
| **protagonist_present** vs perspective | 0.070 | 1.2 | ~1–2σ 모호 | 크기만으론 약함 — **3/3 방향일치**가 실제 근거(부호검정) |
| **causality 변형밴드** (frozen/OR/marker/flip) | 0.006 | 0.1 | **깊은 노이즈** | §5의 "전부 노이즈 ±0.006" **경험적 확증**. 변형 순위 무의미 |

## 4. 결론 (정직)

1. **causality 변형 순위(±0.006)는 노이즈다 — 확증됨.** "4축 제거가 근소 최선", "inceptive가 최악" 같은
   **미세 순위 결론은 신뢰 불가**. ⚠️ **단 over-fire 판정은 생존** — 그건 ΔScore가 아니라 **기저율**(0.52~0.76
   vs 경계밀도 0.036~0.19 = 3~20× 비율, n=수백이라 SE≈0.02)이므로 노이즈 floor와 무관. FAIL 유지.

2. **Gate C의 −0.041 "미달"은 노이즈 내(0.7σ_pair).** 회귀가 실재하는지조차 불확실. 관문① 수용 근거였던
   "RQ1은 앵커품질과 독립"에 더해, **미달 자체가 노이즈일 수 있다**는 근거가 추가됨 → 동결 수용 강화.
   (주의: pieman은 다중 selection 오염[REPORT §4]이라 어차피 확인적 성능 인용 금지 — 노이즈든 아니든 결론 동일.)

3. **protagonist_present 채택은 크기(+0.070)로는 ~1.2σ = 약함.** 채택을 지탱하는 건 **3/3 doc 방향일치**
   (부호검정 p=0.125, 여전히 약하나 크기보다 나음). D-B10 재검토 대상은 아니나 "강한 우위"로 서술 금지.

4. **🔴 앞으로의 해석 규칙 (핵심 산출):** **단일 추출 ΔScore가 ~0.06(√2σ) 미만이면 해석 불가**,
   불안정 doc(london형 2× 요동)은 **~0.10 미만도 불가**. 따라서:
   - **암기 재검정**: Score 기반 유명/무명 비교는 ΔScore>0.10 아니면 무의미 → **base-rate 격차**(space 2.3×)로
     프레이밍해야 함(base rate는 SE 작음). Score로 암기 주장하려면 다중샘플 필수.
   - **다중샘플 anchor**: k샘플 다수결 → σ→σ/√k. σ=0.043를 0.04 델타를 2σ로 분해하려면 **k≈9** 필요(k=5면
     √2σ≈0.026, 0.04는 1.5σ로 여전히 약함). **다중샘플은 필요조건이나 k=3~5로는 부족** — 정량 지침.
   - **M5 vs M4(RQ1)**: factoring gain이 0.06 미만이면 단일추출로 주장 불가 → 다중샘플 앵커 위에서 평가.

## 5. 한계 / 검증 미해결 (제거 금지)

1. **flip 모델 = 독립 Bernoulli** (축별 주변 flip rate). 실제 jitter는 (a) 축 간 상관, (b) 비대칭(유명 run2가
   경계 적게 발화 — london 54→29)이라 **참 σ는 다소 클 수 있음**(보수적으로 √2σ≈0.06은 하한 근처).
2. **MC 평균이 base보다 하락**(pieman 0.593→0.478) = 독립flip이 count를 2.5로 밀어 경계 과예측 유발하는
   모델 artifact. **σ(분산)은 유효하나 평균이동은 신뢰 말 것.**
3. **σ는 dev-3(3편)에서만.** flip rate는 유명 3편(문학, dev-3와 register 유사)서 이식 — register 다른 코퍼스는 재측정 필요.
4. **base-rate 오염**: pieman은 다중 selection(REPORT §4)이라 base Score 자체가 오염 — σ 추정엔 무관(분산만 씀)하나 절대값 인용 금지.
