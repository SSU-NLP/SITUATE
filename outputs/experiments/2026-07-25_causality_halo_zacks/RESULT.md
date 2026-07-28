# causality Zacks-halo 채택 테스트 — 결과 (RESULT)

> 사전등록 `PRE_REGISTRATION.md`(3조건 결과 전 봉인). frozen Qwen3.6-35B-A3B no-thinking v2, temp=0.
> primary=discordant probe(14문항), sanity/guardrail=repcorpus stage0 dev 12편(2222 transitions).
> 원자료 `halo_dev.jsonl`, `dev_run.log`. 스크립트 `scripts/eacl_phase0/halo_causality.py`,
> `halo_guardrail_score.py`.

## 판정 = **기각** (3-of-3 필요, sanity 위반)

| 조건(사전선언) | 결과 | 판정 |
|---|---|---|
| **primary** — within-scene 인과단절을 halo가 단절로 (단절율≥0.80, 오발현≤0.20) | 단절 **7/7=1.00**, 오발현 **0/7=0.00** | **PASS** |
| **sanity** — 발현율 [0.15,0.40] & <현행프록시(0.67) | **0.869** (프록시보다 오히려 **높음**) | **FAIL** |
| **guardrail** — Score 하락 −0.09 이내 (t* 동결) | Δ = **+0.000** | PASS(속 빈) |

## 무슨 일이 일어났나

- **primary PASS (완벽)**: 장면(space·protagonist) 전부 연속인 clean 쌍에서, halo는 "직전에 야기됐나"를
  정확히 갈랐다 — 인과단절 7/7 단절, 인과연쇄 7/7 연속. **halo는 인과가 변별변수일 때 인과를 옳게 따라간다.**
- **sanity FAIL (결정적)**: 그런데 실제 서사에서 halo 발현율 = **0.869** (편별 0.74~0.95). event_type
  프록시(0.674)를 **고치기는커녕 더 포화**시켰다. 이유: **인접 문장 간 "직접 인과"는 prose에서 예외**다 —
  대부분의 이웃 문장은 묘사·부연·시간순서·공지시·병렬행동·대화턴으로 이어지지 직접 인과가 아니다. 그래서
  "직전에 야기되지 않음"(=단절)이 **기본값** → 포화. probe가 인위적으로 인과-포화라 primary는 통과했지만
  실전 분포는 정반대.
- **guardrail 속 빈 PASS**: t*=5(5축 전부 발현해야 경계)에서 87% 포화 causality는 **거의 상시-on**이라
  경계 결정에 관여하지 못한다 → Δ=0.000. t*-sweep 전체에서도 halo는 **어느 threshold에서도 Score를 못
  높인다**(Δ −0.007~+0.002). 즉 가드레일 통과는 품질이 아니라 **포화된 축이 적응 threshold에 흡수**된 결과.

```
t   현행    halo    Δ
1  0.0315  0.0310  -0.0006
2  0.0345  0.0336  -0.0009
3  0.0583  0.0518  -0.0065
4  0.2634  0.2650  +0.0016
5  0.3959  0.3959  +0.0000   <- 동결 t*
```

## 핵심 교훈

**포화 문제는 halo-vs-프록시가 아니라 축 자체다.** 두 조작화 모두 이 코퍼스에서 causality를 포화시킨다:
- event_type 프록시 → 대화↔행동 유형 교대로 0.67 포화
- 직접-인과 halo → "prose에서 직접 인과는 희소"라 0.87 포화

**포화된 축 = 비변별.** halo는 포화를 **없애는 게 아니라 옮긴다**(그리고 primary가 보이듯 인과 자체는
옳게 읽는다). Zacks가 성공한 건 (a) *One Boy's Day*가 인과-조밀한 행동 로그이고, (b) 훈련코더의 느슨한
"서사적으로 이어짐" 판독, (c) causal을 단독 임계가 아니라 **6차원 집계의 상관 성분**으로 썼기 때문 —
우리처럼 단일 축을 임계하지 않았다.

## 다음 (별도 사전등록 필요 — 이 기준 이동 금지, D-M5)

- **연속성 뒤집기 후보**: halo 질문을 "직접 인과"(희소→포화)에서 **"CURRENT가 PREVIOUS의 진행 활동/상황을
  이어가나"(연속성, 흔함→그 부재=단절이 희소)**로 바꾸면 발현율이 대역[0.15,0.40]에 들 여지. 이전에 논의한
  thread-continuity를 halo로 물음. primary(인과변별)는 유지될지 재확인 필요.
- **또는** causality를 약축으로 인정하고 space/protagonist/time에 의존 + Zacks식 "집계 성분" 취급.
- 어느 쪽이든 채택 판정은 Gate C DTS Score(별도 실험).

## 한계

- discordant probe = hand-crafted 14문항·소표본(낙관 편향). 통과했어도 실전 분포(sanity)가 뒤집었다 —
  오히려 probe-실전 괴리가 이번 핵심 발견.
- guardrail은 12편을 per-doc deployable(t* 동결)로 채점·평균. t*는 현행 config argmax로 dev 동결 후
  halo에 동일 적용(스왑 외 전부 동결). 단일 dev 세트라 Gate C 정식 재확인은 별도.
