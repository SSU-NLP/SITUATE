# S5 EISM 재설계 → 동결 REPORT (관문①)

> 작성 2026-07-22. 사전등록 = 이 디렉토리 `PRE_REGISTRATION.md`. 방법 = EISM(Zwaan 1995)+Indexter 5축.
> VERSION `p0s5-eism-v2`, 앵커 모델 deepseek-v4-pro. **주지표 = 사람 gold 대비 DTS Score**(AUC 보조).
> S5 = M4~M7 anchor 공급원. 이 동결로 관문① 종료 → 관문②(코퍼스+Phase-2 재검증) 진입.

## 0. 동결 대상
5축 = **protagonist / space / time / causality / intentionality** (구 place→space, goal→intentionality 리네임).
frozen count = 각 축 reset 합산 ≥ t*(=3, D-B6 pieman 동결). 실험 additive 컬럼(count 미포함): `protagonist_present`, `causal_marker`, (`dominant_turnover`는 미채택으로 코드에서 제거·데이터만 보존).

## 1. 되돌리기 + SHA 일치 (동결 안전성)
dominant_actor 실험(§6)을 protagonist 프롬프트에 번들했다가 **전량 revert**. 현 코드가 frozen 추출을
**비트 단위로 재현**함을 확인:
- pieman protagonist 첫 배치 prompt SHA(현 코드) = `1ed776…b913c` = frozen 캐시 SHA. **✅ 일치.**
- protagonist 프롬프트에 `dominant_actor` 없음 확인. → 동결 캐시(`*_s5_eismv2`)와 코드 정합, prompt_sha256 유효.
- 유지된 것: API timeout=180·max_retries=0·재시도 백오프(504 origin 과부하 대응, 프롬프트 불변이라 sha 무관).

## 2. 🔴 Gate C 결과 = strict 미달 (2/3), 정직 명시
deepseek present-config vs fable5, 사람 gold 대비 Score, t*=3 pieman 동결:

| doc | deepseek | fable5 | Δ(ds−fb) | strict(≥−0.03) |
|---|---|---|---|---|
| pieman | 0.593 | 0.634 | **−0.041** | ❌ 미달 |
| tunnel | 0.461 | 0.383 | +0.078 | ✅ |
| monkey | 0.286 | 0.359 | **−0.072** | ❌ 미달 |
| 평균 | 0.447 | 0.459 | −0.012 | — |

**strict Gate C(3편 모두 ≥−0.03) 통과 아님.** pieman −0.041, monkey −0.072 미달. **"통과"라 부르지 않는다.**

## 3. 그럼에도 동결(수용) 근거 — RQ1은 앵커 품질과 독립
- 논문 심장 = **RQ1: M5(FactoredSlot) vs M4(MonoAnchor)**. **둘이 같은 S5 anchor를 먹으므로** anchor의 절대 품질은
  M5−M4 **대조의 내적 타당성에 영향 없음**(앵커 품질은 RQ2/절대성능 S1에만 영향).
- 파국적 실패(구 −0.25) 없음 — 재설계로 −0.04~−0.07까지 회복(구조 복원). deepseek은 **살아있는 앵커**.
- Gate C 기준선 fable5(0.634 등)는 **D-M5 selection-on-test로 dev 강등된 오염값** → Δ는 *재현* 지표일 뿐,
  통과 실패가 "성능 부족"을 뜻하지 않음. **논문에 fable5·deepseek 절대 Score를 성능 근거로 인용 금지.**
- ⇒ 수용 = (i) 비파국 + (ii) RQ1-강건 + (iii) Gate C 오염 기준선 caveat. **strict 바 미달을 숨기지 않고 병기.**

## 4. 🔴 pieman 선택압 누적 — 오염 공개 (성능 인용 금지)
pieman Score를 **본 결정들**(=pieman은 dev로 반복 열람, 성능 인용 불가):
1. t* threshold 동결(t*=argmax Score on pieman, D-B6)
2. time 축 tense-change 제거(pieman dev 0.598→0.634)
3. space 축 granularity/침묵 튜닝
4. intentionality Q5·스모크 문면
5. causal_marker **OR 규칙** 테스트
6. causal_marker **flip(극성반전)** 테스트
7. protagonist_present 채택(pieman 포함 3편)
8. dominant_turnover 테스트(t* pieman 동결 + pieman Score)
9. 축별 기여도/liveness/ablation 열람
⇒ **pieman은 다중 selection 누적 = 강한 dev 오염.** pieman 단독 Score를 확인적 성능으로 쓰지 않는다.
확인평가는 봉인 held-out 1회 개봉으로만.

## 5. causality — 관계적 구성개념은 오염-안전 프레임과 원리적으로 양립 불가 (논문 기여)

**이건 변명(measurement gap)이 아니라 경험적 결과다.** EISM 5지표 중 **causality만 관계적 정의**를 갖는다:
원전(Zwaan 1995 / Trabasso 인과 네트워크)의 causality = "이 사건이 직전 사건에 의해 가능해졌/야기됐는가"의 **사건 간 연결**,
불연속 = **인과 선행자의 부재**. 우리의 `event_type 변화 OR inceptive`는 문장 내부 분류의 churn을 세는 **다른 구성개념**이고,
발화율 52~76%(경계밀도 0.036~0.19의 3~20배)가 그 괴리를 그대로 보여준다 = **기저율 잡음.**

**표면 표지 근사(causal_marker)도 성능을 못 올린다 (재현 가능, dev-3 3편평균):**

| causality 변형 | pieman | 3편평균 |
|---|---|---|
| frozen(event_type churn) | 0.593 | **0.447** |
| OR causal_marker | 0.569 | 0.440 |
| marker-only | 0.581 | 0.438 |
| flip(marker 부재시 reset) | 0.544 | 0.433 |
| **causality 제거(4축)** | 0.605 | **0.448** |

이론적으로 더 충실한 명시 표지(because/so 등)를 넣을수록 **오히려 하락**, 통째로 빼는 게 근소 최선. causality lift는
서사 dev-3서 **+0.007(死)** — 단, 화제분절(dialseg711 +0.354)선 살아있어 **"서사 event-B 한정 死"**.

**왜 튜닝으로 못 고치나 (핵심):** 진짜 EISM causality는 **두 사건을 동시에 봐야** 판정 가능 → G2(no-carryover)와 충돌.
그런데 "A가 B를 enable하나"의 판정은 **"A·B가 같은 장면인가"와 강상관** = **경계 정보가 링크에 각인**됨. 따라서 창(window)을
줘서 관계를 뽑으면, 경계를 명시적으로 안 물어도 **LLM의 경계 지식이 출력에 유입**(S1이 죽은 그 정보흐름) → 코드가 그걸 읽는 건
우회가 아니라 **한 단계 지연된 halo**. **coref(정답이 경계 불변)와 결정적으로 다르다** — 인과 링크는 경계 상관이라 분리 패스로도 안전하지 않다.

**⇒ 논문 서술(정직):** "EISM 5지표 중 causality만 관계적 정의라, 오염-안전 추출 프레임(target-only)에서는 관계를 볼 수 없어
표면 표지로 근사했고, 이론적으로 더 충실한 표지를 넣어도 성능이 하락했다(0.447→0.440, flip 0.433). **인지과학 이론을 계산
프레임에 옮길 때 관계적 구성개념이 특히 저항한다**는 것이 발견이다." (Indexter가 plan의 causal_link=구조를 쓴 것도 같은 이유 —
관계는 구조에서 오지 텍스트 표면에서 안 온다.)

**분해 진단(2026-07-22, 공짜) — "event_type 걷고 inceptive만 쓰면?" 유혹 차단:** frozen reset = `event_type변화 OR inceptive`.
inceptive는 Zacks 지지(새 행동 시작)나 event_type변화는 우리 발명. 발화율 분해(dev-3 pooled): **event_type변화 66%**(과발화 주범 확정),
**inceptive 9%**. lift: event_type −0.041 / inceptive −0.019 (**둘 다 반신호**). Score: inceptive만 **0.408(최악**, tunnel 0.461→0.341),
event_type만 0.454 ≈ frozen 0.447 ≈ 제거 0.448 (**전부 노이즈 ±0.006**). ⇒ **프레임 내 건질 성분 없음** — 이론충실 성분(inceptive)조차
서사 경계와 misalign. event_type만의 문제가 아니라 causality 통째로 event-B와 안 맞음. "싼 in-frame 수선" 가능성 닫힘.

**Phase X 후보(관계적 causality) — 지금 안 함, 검증조건과 함께 이월:**
- 후보 = 경계-blind 인과그래프 추출 + 코드 불연속. **단 이는 S2 완화가 아니라 G2 위반**(창=carryover, 좌문맥 6문장 AUC 0.55로 이미 死).
- **사전등록 필수 게이트(실행 전):** *창을 경계 가로지르게 vs 안 가로지르게 배치했을 때 링크 밀도가 다른가?* 다르면 = 링크가 경계
  정보를 담음 = 코드계산은 되읽기(지연 halo) → 기각. (결과 보고 판단하면 늘 "우회"로 읽히므로 반드시 사전등록.)
- **투자가치 낮음(명시):** causality 상한 ≈ 0.016(제거해도 그만큼만 하락), 새 HP(창·링크임계)+추출비용+위생논쟁 대비 소득 미미.
  Gate C 이미 실패·수용(iii)·동결 마무리·추격금지 상태 → **LoCoMo/후속 논문에서 제대로 설계할 사안.**

## 6. dominant_turnover 미채택 + 모달리티 분기 (LoCoMo 이월)
- §6 통합 protagonist 축 후보(지배행위자 turnover). dev-3 3편평균 **0.401 < present 0.408 바 미달**(사전등록 기준).
  **판정 근거 = 사전등록 1회 바 미달**(노이즈 논증 불사용).
- **⚠️ 0.401은 핸디캡 하한값 — 측정 오염(none 이중의미) 확인됨.** dominant_actor를 **cast 객관식(closed list)**로 강제한 탓에
  none이 두 가지를 섞음: (a) 진짜 행위자 없음(서술), (b) **행위자 있으나 cast로 해소 못 함**(대명사·집합주어·대화화자).
  tunnel none률 0.87의 문면 표본검사 결과 **대부분 (b)**: `[4]"He sat up."`, `[212]"I shut it off."`,
  `[351]"They seem to be computing something."` = 명백한 행위(교체)인데 none→none으로 기록돼 **turnover 미발화.**
  characters_present도 대부분 []=대명사 coref 자체를 안 함. ⇒ **개념이 present에 진 게 아니라, 이 추출이 turnover를 못 잰 것.**
- **따라서 주장 범위:** "이 closed-list 추출은 사전등록 바 미달로 미채택"까지만. **"turnover 개념 < present"는 주장 불가**(측정 confounded).
- **present가 덜 휘둘린 이유:** "주인공 하나 있냐" boolean이라 대명사 he를 그 한 명에만 판정하면 돼 cast 완전성/coref에 강건.
- **모달리티 분기 증거(핵심):** dominant는 **monkey(14인 다행위자)서 present 상회**(0.283>0.259). AMI 회의서도
  perspective gold발화율 **0.000(死)**, charset(집합변화) **+0.116(유일 신호)**. ⇒ 단일주인공=present, 다화자=turnover.
- **결정: fiction=present 동결. turnover=LoCoMo 다화자용 이월**([[locomo-downstream-deferred]]). 착수 시 재설계 필수:
  **대명사 coref·집합주어("they")·대화 화자귀속 해소**(현 추출은 이걸 안 해 none 87% 오염) + cast 확장/개방 + AMI eval 35편 실검증(현 2편만 추출).

### 6.1 none 87% 원인 분석 — "목록 김"(해석1) 반증, "target-only 하 coref 불가"(해석2) 확정
cast 크기 vs none률(예산 0, story_lists):

| doc | 문장 | cast수 | none률 | 대화율 |
|---|---|---|---|---|
| pieman | 96 | 6 | 0.28 | 0.40 |
| tunnel | 504 | 6 | **0.87** | 0.60 |
| monkey | 408 | **14** | 0.64 | 0.28 |

- **해석1(cast 목록이 길어 none 도피) 반증:** tunnel cast=6=pieman인데 none 3배(0.87 vs 0.28); monkey는 cast 최대(14)인데 none은 tunnel보다 낮음. **none률과 cast 크기 무관.** ⇒ 목록 좁히기/상위N 접근은 무효.
- **해석2(개념이 target-only서 어려움) 확정, 단 정교화:** none률이 **대화율과 정렬**(pieman0.40→0.28, tunnel0.60→0.87). 실 원인 = **대화·대명사("He sat up"/"They compute") 문장은 행위자를 붙이려면 앞문장 coref가 필요한데 S2 target-only·G2 no-carryover가 금지** = turnover(문장 넘는 행위자 정체성) ↔ 측정위생의 구조적 긴장. present는 문장내 boolean이라 생존.
- **fix 방향(codex+별도 사전등록):** (a) protagonist 축만 target-only 완화(coref 허용, 위생 예외 정당화) 또는 (b) present식 2단계 캐스케이드. **목록 조정 아님.**
- 데이터 `outputs/eacl_phase0/*_s5_eismv2_domact/`, 채점 `scripts/eacl_phase0/score_domact_turnover.py`.

## 7. Q3 protagonist_present 채택 = 동결 config 변경 (별도 기록 D-B10)
- 규정 통과: protagonist 슬롯을 **perspective 단독 → protagonist_present**로 교체 시 dev-3 3편평균
  **0.447(present) vs 0.377(perspective)**, 3편 모두 개선 → 사전등록 채택바 통과.
- **⚠️ 이건 동결 config 변경(D-B8 "perspective 단독"에서 이탈)이므로 조용히 넘기지 않고 별도 D-B10으로 명시.**
  frozen count의 protagonist 성분 = 이제 `protagonist_present`(perspective 아님). eval_seg_score FROZEN_AXES도 갱신 대상.

## 8. 남은 것 (관문② 진입 전)
- eval_seg_score.py FROZEN_AXES: `protagonist_perspective`→`protagonist_present`로 갱신(D-B10 반영).
- ①B 비시험 파일럿 QC(파싱률·schema율·비용/문장·축 liveness)는 **관문② 스케일 추출 시** 수행(대표 코퍼스 필요).
  **🔴 순서 역전 주의:** 사전등록 §4는 ①B를 freeze *선행* 게이트로 뒀으나 여기선 미뤄 **검증 전 동결**임 →
  관문②에서 **①B QC 실패 = 프롬프트 패치 아님, frozen v2 폐기 + 새 VERSION 재등록**(사전등록 §4, ledger D-B9 명문화).
- ledger D-B9(동결)·D-B10(config) append. methodology p0s5-eism-v2 반영.
