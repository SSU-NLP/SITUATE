# Tier 2 Held-out 구축 스펙 (⛔ SUPERSEDED — D-M13①, 2026-07-21)

> **🔴 이 문서 전체는 폐기되었다.** Tier2(Gutenberg 12편 + GPT-4.1 B3 silver gold)는 held-out에서
> **제외**됐다(ledger D-M13①). 사유: gold가 사람 아닌 silver·인간 대비 4.5× 희소·19세기 문어로
> 타깃(구어 썰) 불일치. 대체 = **Tier1 6편**(신규 *Born a Crime* 3편, 사람 20명 주석 — D-M13②).
> D-M6·D-M9(이 문서의 근거 결정)도 폐기. 아래 내용은 **역사 기록**으로만 보존한다.

> ~~**왜 필요한가:** ledger D-M5에서 7개 데이터 전부가 dev로 강등되었다... 논문의 실제 held-out 숫자가 된다.~~
> ~~**⚠️ 실행 전 M3 설정 동결 필요** (`phase0-annotation-guide.md` §2~3).~~

---

## 0. 핵심 결정 — gold를 LLM으로 만든다 (차선책이나 불가피)

**이유:** 사람이 사건 경계를 주석한 **텍스트** 데이터는 이 분야에 사실상 **3편뿐**(Pieman/Monkey/Tunnel)이며, 그 3편은 모두 dev로 소진되었다. 새 사람 주석은 크라우드소싱 비용·시간이 크다.

**정당화 (선행연구 2편이 LLM 경계의 타당성을 검증했다):**

| 논문 | 모델 | 보고한 것 |
|---|---|---|
| **B2** Michelmann et al. (Behav Res Methods) | GPT-3 | LLM 경계가 사람 주석과 유의하게 상관하며, **개별 인간 주석자보다 오히려 합의(consensus)에 더 가까웠다** (= 정확성) |
| **B3** Panela et al. (Comm Psychology 2025) | GPT-4 / LLaMA-3 | **LLM이 인간보다 더 일관되게(consistently)** 사건 경계를 식별한다 (= 재현성. 20회 반복·온도 스윕으로 검증) |

**⇒ 두 논문 모두 "LLM이 사람보다 낫다"는 취지를 보고했다.** B2는 정확성(합의 근접), B3는 일관성(분산 낮음). **B3가 더 최신이고 더 엄밀하게 검증했다.**

---

## 1. 어느 프로토콜을 쓰는가 — **프롬프트는 B2(원조), 절차는 B3(개선)**

**⚠️ 이 둘은 경쟁 관계가 아니다. B3가 B2를 개선한 것이다** — B3 논문이 명시적으로 "이전에 GPT-3로 기술된(Michelmann et al.) zero-shot 프롬프트를 사용했다"고 밝힌다. **즉 B3의 프롬프트 = B2의 프롬프트.**

| 항목 | 채택 | 출처 | 이유 |
|---|---|---|---|
| **프롬프트** | B2 원문 그대로 | `s-michelmann/GPT_event_segmentation` | 원조. 우리 dev 3편(Pieman/Monkey/Tunnel)에 직접 적용된 검증 프로토콜 |
| **모델** | **GPT-4** | B3 | GPT-3(`davinci`)는 **deprecated**. 그리고 **GPT-4는 B3에서 사람 대비 검증된 모델**이다 (§1.5 참조 — 최신 모델을 안 쓰는 이유) |
| **반복** | **20회, temp 0** | B3 | B3가 실제로 GPT-4 temp 0으로 20회 실행(API 비결정성 잔여 변이) + temp 0이 run 간 일치 최고·저자 권고. temp 1은 과분절·정렬 저하 |
| **합의** | ~~다수결~~ → **B3 top-n** | B3 | 🔒 **정정(codex 2026-07-15, D-M9):** 단순 다수결(≥10/20=동률) 아님. `n=round_half_up(20 run 평균 경계 수)` → 득표 상위 n개, 동률=SHA-256(transition) 순. **경계밀도를 dev에 맞추는 임계조정 금지** |

**⇒ 결론: "B2 프롬프트 + B3 절차(GPT-4, temp 0, 20회, top-n 합의)".** 동결·근거 = ledger **D-M9**.

### 검증된 zero-shot 프롬프트 (B2/B3 공통 — 그대로 사용)

```
An event is an ongoing coherent situation. The following story needs to be
copied and segmented into large events. Copy the following story word-for-word
and start a new line whenever one event ends, and another begins.
This is the story:

<STORY>

This is a word-for-word copy of the same story that is segmented into
large event units:
```

**⚠️ 프롬프트를 임의로 수정하지 않는다.** 수정하면 "검증된 프로토콜"이라는 정당화가 무효가 된다.

---

## 1.5 🔴 왜 최신 모델(GPT-5)이 아니라 GPT-4인가 — 방어 논리

**리뷰어 예상 질문: "왜 더 강한 최신 모델을 안 쓰고 GPT-4를 썼나?"**

### 선행연구가 GPT-3/GPT-4를 쓴 이유는 "그 시점의 가용 최신 모델"이었을 뿐이다

- B2(2023)는 당시 GPT-3가 최신이라 썼다.
- B3는 **GPT-3가 deprecated되어** GPT-4/LLaMA-3로 갈아탔다 (논문이 이를 명시).
- **즉 특정 모델이어야 할 이론적 이유는 없다.**
- **그리고 B3가 오히려 이 프로토콜의 model-agnostic함을 증명한다** — 모델을 GPT-4로 바꾸고 LLaMA-3까지 추가해도 **여전히 인간 수준 이상**의 일관성이 나왔다.

### 그럼에도 GPT-4를 쓰는 이유 = **성능이 아니라 인용 가능성**

**우리는 gold를 만드는 것이지 성능을 겨루는 것이 아니다.** gold의 요건은 "가장 똑똑한 모델"이 아니라 **"사람 경계와의 일치가 peer-review로 검증된 모델"**이다.

| 선택 | 방어 논리 |
|---|---|
| **GPT-4 gold** | "Panela et al. (Comm Psych 2025)이 **이 모델·이 프로토콜이 인간보다 일관되게** 사건 경계를 식별함을 검증했다." → **방어 완료** |
| GPT-5 gold | *"GPT-5가 사람처럼 분절한다는 근거는?"* → **인용할 논문이 없다.** 그 타당성을 우리가 증명해야 하며, 이는 **별도 연구**다 |

**⇒ 우리 주장(RQ1/RQ2)은 gold 품질에 의존한다. 그 품질을 남의 peer-review로 담보할 수 있으면 훨씬 강하다.**
**⇒ 메인 gold = GPT-4로 확정. 이 결정은 결과를 보기 전에 동결한다.**

### GPT-5는 Appendix용 robustness check로 1회 실행

**목적 2가지:**
1. **Gold robustness:** "gold 생성 모델을 바꿔도 결론이 유지되는가?" → 유지되면 **silver gold의 최대 약점(모델 의존성)이 완화**된다.
2. **재현성·미래 대비:** GPT-4도 언젠가 deprecated된다(GPT-3가 그랬듯). 프로토콜이 최신 모델에서도 작동함을 보이면 재현성이 올라간다.

**🔴 절대 규칙: GPT-5 gold를 메인으로 승격하지 않는다.**
GPT-5에서 성능이 더 잘 나온다고 메인을 바꾸면, **그것이 바로 selection-on-test(D-M5에서 겪은 오류)의 반복**이다. **메인은 GPT-4로 사전 동결, GPT-5는 부록 고정.**

## 2. 🔴 교차 모델 계열 규칙 (LLM 공유 편향 완화)

**문제:** M3(우리 추출기)도 LLM을 쓴다(`claude-fable-5`). gold도 LLM이 만들면 **"LLM이 만든 걸 LLM이 맞히는"** 구조가 된다.

**완화 논거 (완전 순환은 아님):**
- **B2/B3:** LLM에게 **"경계가 어디냐"를 직접** 묻는다.
- **M3:** LLM에게 **"이 문장의 장소·인물이 뭐냐"만** 묻고, **경계는 코드가 계산**한다. (경계·변화·리셋 단어를 프롬프트에 넣지 않는 것이 M3의 대원칙)
- ⇒ **메커니즘이 다르다.**

**그러나 무해하지도 않다.** 같은 계열 LLM이면 서사 구조에 대한 **공통 편향**이 있을 수 있다.

**🔴 필수 규칙:**
- **gold 생성 = GPT-4 (OpenAI 계열)**
- **M3 추출 = claude-fable-5 (Anthropic 계열)**
- **⇒ 서로 다른 계열을 사용하여 편향 중첩 확률을 낮춘다.**
- 이 규칙을 논문 Limitations에 명시한다.

---

## 3. 서사 소스

- **공개도메인 단편 서사** (Gutenberg 등). **각 단편 *안*의 사건 경계**가 타깃 — 이어붙이기(concatenation) **금지**(그건 A 경계이며 타깃 불일치).
- **편수:** **최소 10편 이상** — 목적이 **doc-level bootstrap CI 확보**이므로 doc 수가 많을수록 좋다. (현재 fiction held-out doc이 1개뿐이라 CI가 불가능한 것이 핵심 결함.)
- **길이:** dev(Pieman 96문장 / Tunnel 504문장)와 유사한 범위. 극단적으로 짧거나 긴 것은 제외.
- **장르:** **서사 픽션**(scope 내). 저널리즘·다큐·대화는 제외 — 이미 scope 밖으로 판정됨.
- **문체:** 다양하게(단일 작가 편중 금지). 일반화 주장을 위해.

---

## 4. 문장 분할·정렬 정책 (dev와 동일하게 — 고정)

- **문장 분할:** dev(Michelmann 3편)와 **동일한 분할기·규칙** 사용. (텍스트 기반이므로 word-onset 정렬은 불필요.)
- **경계 정렬:** LLM이 개행으로 표시한 경계를 **문장 인덱스로 스냅**. 문장 중간에 떨어지면 가장 가까운 문장 경계로.
- **⚠️ 이 정책을 데이터 생성 시점에 고정하고 ledger에 기록한다.** 나중에 바꾸면 재현 불가.

---

## 5. 평가 프로토콜

### 5.1 사용 가능 / 금지 베이스라인 (🔴 순환 방지)

| 베이스라인 | Tier 2에서 | 이유 |
|---|---|---|
| **B1-BayesSurprise** (Kumar, GPT-2) | ✅ **사용** | gold 생성에 관여하지 않음. **핵심 상대**(EST vs EISM) |
| **M1-CosDrift** | ✅ 사용 | 학습·LLM 무관 |
| **M2-MonoLAM** | ✅ 사용 | gold와 무관 |
| **B2-LLM-Direct** | ❌ **금지** | **gold 생성기 → 순환** |
| **B3-EventRecall** | ❌ **금지** | **gold 생성기 → 순환** |

### 5.2 평가 절차

1. M3 설정을 **동결 상태 그대로** 적용 (time=marker만, t\*=3, perspective 단독). **Tier 2에서 재선택·재튜닝 금지.**
2. **1회만** 평가한다. 여러 변종을 돌려 비교하면 Tier 2도 오염된다.
3. **doc-level bootstrap CI** 산출 (이것이 Tier 2의 존재 이유).
4. 메인 지표 = **DTS Score**(free-threshold, 각 방법 자연 규칙). 보조 = matched@gold, AUC.

### 5.3 판정

- **Go:** M3가 M1·B1을 유의하게(CI 기준) 상회.
- **회색지대:** CI가 걸침 → 편수 확대 후 재판정.
- **No-Go:** 우위 없음 → RQ1의 "EISM이 서사에서 작동한다" 주장 재검토.

---

## 6. ⚠️ 논문에 반드시 명시할 한계

1. **Silver gold.** 사람 주석이 아니다. 단, B2/B3가 LLM 경계의 타당성(합의 근접·인간 이상의 일관성)을 검증했다.
2. **LLM 공유 편향 가능성.** gold 생성(GPT-4)과 M3 추출(Claude)이 서로 다른 계열이나, LLM 일반의 공통 편향은 배제할 수 없다. **메커니즘이 다르다는 점**(직접 경계 질의 vs 상태 추출 후 코드 계산)이 부분적 완화 근거.
3. **B2/B3를 이 데이터의 베이스라인으로 쓸 수 없다** (순환).
4. **⇒ Tier 3(한국어 썰, 사람 주석)가 최종 방어선.** silver gold의 한계를 근본적으로 해소하려면 사람 주석이 필요하다.

---

## 7. 실행 체크리스트

- [ ] M3 설정 동결 확인 (`phase0-annotation-guide.md` §2~3)
- [ ] 서사 소스 확보 (공개도메인 단편 10편+, 서사 픽션, 문체 다양)
- [ ] 문장 분할 정책 고정 + ledger 기록
- [ ] **GPT-4**로 B2 프롬프트 × **20회**(temp 0) → 다수결 합의 → silver gold
- [ ] gold sanity check (경계 밀도가 dev 3편과 유사한 범위인가)
- [ ] M3(동결) + M1 + B1 평가 — **1회만**
- [ ] doc-level bootstrap CI 산출
- [ ] ledger에 전 과정·수치 기록
- [ ] **(Appendix)** GPT-5로 gold 재생성 → robustness check **1회**. **메인으로 승격 금지.**
