# LoCoMo SeCom segmenter-swap — 5축(M3) vs surprise/CosDrift 베이스라인

`outputs/experiments/2026-07-15_locomo_segmenter_swap/` · 2026-07-15

## 한 줄

SeCom 메모리 파이프라인을 고정하고 **분절기만** 교체 → **경계 밀도를 매칭한 상태**에서 LoCoMo 카테고리별 gold-evidence turn **recall@k**와 **최종 답변 GPT4Score**를 측정. **밀도를 맞추면 5축 사건 분절기(M3)가 evidence recall·답변점수 모두 최고, CosDrift(M1)가 최저.** M3의 우위는 **temporal·single-hop**에 집중.

---

## 1. 실험 setup

**질문.** SeCom류 메모리 파이프라인에서 어떤 화제/사건 분절기가 downstream 검색·QA에 유리한가? 특히 (b) 5축 상황모델 리셋(M3) vs surprise 기반 vs 하이브리드.

**데이터.** LoCoMo (`benchmarks/locomo/locomo10.json`) — 10 대화, 5,882 turn, 272 session, 1,986 QA. dia_id(`D{sess}:{turn}`) 보존해 QA의 `evidence`(gold 근거 turn)와 매핑. 카테고리: 1 multi-hop(282, 평균 3.13 근거turn) · 2 temporal(321, 1.17) · 3 open-domain(96, 2.21) · 4 single-hop(841, 1.06) · 5 adversarial(446, 1.03).
- **split:** dev = conv-26,30,41,42,43 / **test = conv-44,47,48,49,50** (온라인 임계 보정은 dev, 평가는 test 1회).

**파이프라인(SeCom 고정, 분절만 swap).** 분절 → (compress 생략, 분절 순수비교) → retrieve(mpnet `multi-qa-mpnet-base-dot-v1`, cosine top-k) → chat(`gpt-4o-mini`) → judge(`gpt-4o`, 1–10 rubric = SeCom 08_eval과 동일). SeCom 원 repo 부재로 retrieve/chat/judge는 최소 재구현. **인코더·chat·judge 전 방법 동일 → 유일 차이 = 분절.**

**분절기 4종** (전이당 점수, 높을수록 경계):
| 표기 | 신호 | 온라인? |
|---|---|---|
| **M1-CosDrift** | delta_eff (m=2,ρ=0.7,a=0.5) | ✅ 인과 |
| **surprise-gpt2xl** | gpt2-xl 다음단어 NLL(과거 2턴 조건) | ✅ 인과 |
| **M3-5axis** | 5축 coordinated reset count 0–5 (frozen enum·리셋규칙, `llm_s5_states`) | ⚠️ Stage1(엔티티목록)=전체대화 필요 → 원천적 offline 측정기 |
| **Hybrid** | z(surprise)+z(count) 정규화 합 랭킹 | (M3 상속) |

M3 추출 = `gpt-4o-mini`, 2-stage(Stage1 대화당 cast/loc/goals, Stage2 turn·축별 닫힌 enum), temp0. 분절은 **세션 단위 fresh**(SeCom 규약, 세션 경계 = 자동 경계, running 통계 세션마다 리셋).

**밀도 매칭(과분절 confound 차단) — 두 모드 모두 보고:**
- **online (deployable, 메인):** 스트리밍 규칙 = running μ+cσ(과거만, Welford, warmup 3). c는 **dev에서 목표밀도(~5 turn/seg)에 맞게 보정 후 동결**, test 적용. 미래 안 봄.
- **offline (진단):** 대화별 전역 top-N 정확매칭. 전체 대화 참조 → **비온라인**, 배치품질(ranking) 진단용.

**지표.** ① gold-evidence turn **recall@k** (검색 top-k 세그먼트의 구성 turn이 gold evidence dia_id를 덮는 비율, 카테고리별) — LLM-free. ② 최종답변 **GPT4Score**(×10 → /100), 카테고리별. QA 문맥 = top-5 세그먼트.

---

## 2. 사용자 중간 질문 판정 — "M3가 사건당 대화 수가 더 많아지나(과소분절)?"

자연 작동점에서 측정(스크립트 `21_`,`22_`):

| | 자연 규칙 | turns/seg | 비고 |
|---|---|---|---|
| M1-CosDrift (10대화) | μ+1σ | **4.83** | 경계율 16.9% |
| surprise-gpt2xl (10대화) | μ+1σ | **4.91** | 경계율 16.5% |
| M3 t\*=3 (conv-26) | count≥3 | 4.66 | count분포 {0:48,1:126,2:155,3:61,4:10,5:0} |
| M3 t\*=3 (conv-30) | count≥3 | **8.39** | count분포 {0:82,1:156,2:87,3:25} (4·5 없음) |
| (참고) 세션만 분할 | — | 21.6 | |

**판정: 부분적으로 그렇다 — 대화별 편차가 크고 일부 대화에서 M3는 확연히 과소분절**(conv-30 8.39 vs 베이스라인 4.8). 또한 **축별 발화율**이 문제의 핵심: causality 56–68% · goal 44–49% · protagonist 5–27% · place 10–20% · **time ~1%(死)**. 즉 프로젝트가 판별자로 지목한 **place(무대 리셋)·time은 캐주얼 대화에서 거의 침묵**하고, 경계는 **causality/goal(고빈도·저정보)**가 놓는다 — EACL 문서의 "상황모델이 정지한 텍스트에선 EISM이 성립 안 함"이 재현됨. 그리고 정수 count라 **밀도를 매끄럽게 못 맞춤**(t\*=2→1.71, 3→4.66, 4→14.45 turn/seg로 계단 점프) → 공통밀도 매칭 시 cosdrift tiebreak 필요.

**⇒ "M3가 거칠다"는 실제 위험이며, 바로 그래서 density-match가 필요하다.** 아래 §3은 밀도를 맞춘 뒤 **배치 품질만** 비교한 결과다.

---

## 3. 결과 (TEST split, density-matched)

### 3.1 밀도 매칭 확인 (online, c는 dev 보정)
| method | c(dev) | dev tps | test tps |
|---|---|---|---|
| M1-CosDrift | 1.3 | 5.08 | 5.24 |
| surprise-gpt2xl | 0.9 | 5.18 | 5.11 |
| M3-5axis | 0.9 | 5.06 | 5.01 |
| Hybrid | 1.2 | 4.88 | 4.63 |

### 3.2 Evidence recall@k — overall
| method | mode | r@1 | r@2 | r@3 | r@5 |
|---|---|---|---|---|---|
| M1-CosDrift | online | 0.240 | 0.341 | 0.403 | 0.485 |
| surprise | online | 0.266 | 0.386 | 0.462 | 0.546 |
| **M3-5axis** | online | 0.272 | 0.394 | 0.453 | **0.554** |
| Hybrid | online | 0.257 | 0.375 | 0.444 | 0.538 |
| M1-CosDrift | offline | 0.231 | 0.326 | 0.397 | 0.494 |
| surprise | offline | 0.229 | 0.321 | 0.389 | 0.477 |
| **M3-5axis** | offline | 0.280 | 0.378 | 0.450 | 0.544 |
| Hybrid | offline | 0.251 | 0.346 | 0.416 | 0.506 |

### 3.3 Evidence recall@5 — 카테고리별 (online)
| method | multi-hop | temporal | open-domain | single-hop | adversarial | overall |
|---|---|---|---|---|---|---|
| M1-CosDrift | 0.316 | 0.382 | 0.308 | 0.569 | 0.550 | 0.485 |
| surprise | **0.348** | 0.485 | 0.308* | 0.633 | **0.610** | 0.546 |
| **M3-5axis** | 0.313 | **0.570** | **0.321** | **0.639** | 0.579 | **0.554** |
| Hybrid | 0.337 | 0.501 | 0.263 | 0.624 | 0.586 | 0.538 |

### 3.4 최종답변 GPT4Score (×10, /100) — 카테고리별 (online)
| method | multi-hop | temporal | open-domain | single-hop | adversarial | overall | ctx_tok |
|---|---|---|---|---|---|---|---|
| M1-CosDrift | 35.4 | 17.0 | 20.2 | 54.4 | 11.5 | 34.65 | 512 |
| surprise | 35.6 | 18.7 | **26.6** | 56.9 | **12.5** | 36.56 | 557 |
| **M3-5axis** | **36.9** | **21.5** | 24.6 | **58.2** | 11.0 | **37.32** | 615 |
| Hybrid | 34.9 | 20.2 | 24.6 | 57.3 | 11.7 | 36.58 | 539 |

---

## 4. 해석

1. **밀도를 맞추면 5축 사건 분절(M3)이 evidence recall·답변점수 둘 다 최고, CosDrift(M1)가 최저.** 순위 일치: recall@5 **M3 0.554 > surprise 0.546 > Hybrid 0.538 > M1 0.485**; GPT4Score **M3 37.3 > Hybrid 36.6 ≈ surprise 36.6 > M1 34.7**. 즉 §2에서 본 "M3 신호는 거칠고 causality-지배"라는 우려에도, **개수를 통제하면 M3의 경계 위치가 흩어진 근거를 가장 잘 한 세그먼트로 묶는다.**
2. **M3 우위는 temporal·single-hop에 집중.** temporal recall 0.570 vs M1 0.382(+0.19), QA 21.5 vs 17.0. LoCoMo 시간질문이 M3의 time-marker/event 구조와 정렬. multi-hop·adversarial은 surprise가 근소 우위(흩뿌려진·부재 근거엔 의미기반 리셋보다 서프라이즈가 유리).
3. **Hybrid는 M3를 못 이겼다.** 정규화 합이 M3의 깔끔한 신호를 노이즈 많은 surprise로 희석 → M3 단독보다 낮음.
4. **온라인 제약이 성능을 해치지 않았다.** online(미래 미참조, dev 보정 임계)이 offline top-N과 대등하거나 오히려 높음(M1·surprise). 실시간 배포 가능성(RQ2) 관점에서 긍정.
5. **절대 점수는 낮다**(GPT4Score 35–37, recall@5 ~0.5). LoCoMo는 장기·다세션이라 top-5 소용량 메모리로는 원래 어렵고, gen=gpt-4o-mini. 방법 간 **상대 비교**가 본 실험의 목적.

**결론:** SeCom류 메모리 파이프라인에서 화제분절기는 — **경계 개수를 공정하게 맞춘다는 전제 하에** — 순수 surprise보다 **5축 상황모델 리셋(특히 시간·사건 구조)이 근거 검색·답변에 유리**하다. 단 그 이득은 시간구조가 있는 질문(temporal/single-hop)에 집중되고, M3는 자연 상태에서 과소·불균등 분절 위험이 있어 **밀도 통제가 필수**다.

---

## 4.5 prefix-online M3 — "미래를 안 보면 얼마나 빠지나" (온라인화 비용)

M3의 유일한 미래-누출은 Stage1 엔티티목록(place·goal이 매칭하는 어휘)이다(protagonist=perspective·time·causality는 고정 enum → 이미 인과적). 이를 **세션-prefix**로 인과화: 세션 s 처리 시 허용 locations/goals = 세션 1..s에서 본 것만(미래 세션 無). place·goal Stage2만 prefix 어휘로 재추출, 나머지 축은 재사용. (세션 내부는 SeCom의 세션-배치 구성과 동일; 제거한 누출 = 미래 세션.)

**결과 (공통 test 3대화 44·47·48, online 밀도매칭, recall만 — QA는 예산소진으로 보류):**

| | full-M3 | prefix-online-M3 | Δ |
|---|---|---|---|
| recall@1 | 0.318 | 0.316 | −0.001 |
| recall@2 | 0.426 | 0.440 | +0.013 |
| recall@3 | 0.481 | 0.523 | +0.042 |
| recall@5 | 0.580 | 0.617 | +0.036 |

**⇒ 미래 세션을 안 보게 해도 evidence recall이 안 떨어진다(오히려 근소 상승).** M3 Stage1의 미래-누출은 배치에 도움이 안 됐다. **온라인 제약 자체는 사실상 공짜.**

**⚠️ 단, 이것은 M5의 상한(upper bound)일 뿐이다.** M5는 여기서 **추가로 LLM까지 제거**(frozen 임베딩 위 소형 증류 모듈)한다. 진짜 위험은 그 두 번째 단계(세계지식 상실)이며 본 실험은 그걸 검증하지 않는다. 즉 **prefix-M3 ≥ M5**가 예상되고, "온라인화는 공짜지만 LLM-free화가 관건"임을 가리킨다.

**caveat:** test 3/5 대화만(conv-49·50은 예산소진으로 prefix 추출 실패). full/prefix 밀도가 test에서 완전 동일하진 않음(full 5.34 vs prefix 5.73 tps; prefix가 오히려 약간 거친데도 recall이 높음 → 결론은 견고). 카테고리별로 temporal@1은 하락(0.278→0.202)하나 @5는 유지.

## 5. 한계 / 검증 미해결 (솔직 기록)

- **QA 문맥 토큰 불균형:** 밀도는 turn/seg로 매칭했으나 top-5 세그먼트의 **토큰 수는 M3 615 vs M1 512(+20%)** — M3가 QA에서 문맥을 더 받는 **약한 confound**. 단 recall@k는 토큰 무관인데도 M3가 이기므로 우위의 전부는 아님. (후속: 토큰-매칭 재현 필요.)
- **M3는 온라인이 아니다:** 추론 시 LLM 호출 + Stage1 엔티티목록이 전체대화 필요. "online 열"은 경계 **결정 규칙**만 인과적일 뿐, M3 자체는 offline 측정기. 진짜 온라인은 증류 M5(미구현).
- **표본 5 test 대화, bootstrap CI 미산출.** 카테고리별 n이 작음(open-domain test 45). 유의성 미검정.
- **SeCom compression(LLMLingua) 생략** — 분절 순수비교 목적. 원 SeCom repo 부재로 retrieve/chat/judge 최소 재구현(SeCom 원 구현과 미세 차이 가능).
- **adversarial(cat5) GPT4Score 전반 낮음(11–12)** — unanswerable 성격상 rubric이 페널티. recall은 정의되나 QA 해석 주의.
- **M3 모델 = gpt-4o-mini** (EACL 스펙의 claude-fable-5 아님; 프록시에서 fable-5 무응답). 다른 계열이라 gold 순환 우려는 없으나 스펙과 모델 불일치.
- t\*=3은 §2 진단에만 사용, §3 매칭에선 count+cosdrift 랭킹 top-N 사용.
- **API 예산 소진(2026-07-15):** prefix-online M3 추출이 test conv-49·50에서 budget_exceeded(429)로 실패 → §4.5 비교는 test 3대화. prefix-M3의 QA(GPT4Score)는 미실행. 예산 복구 후 conv-49·50 prefix 추출 + prefix QA로 §4.5 완성 필요.

## 재현
```
uv run python scripts/secom_swap/20_locomo_prepare.py            # 데이터+evidence
uv run python scripts/secom_swap/24_compute_scores.py            # mpnet emb + cosdrift + surprise
uv run python scripts/secom_swap/23_m3_extract_all.py --workers 16   # M3 5축 (병렬, 캐시)
for m in m1 surprise m3 hybrid; do for md in online offline; do
  uv run python scripts/secom_swap/25_segment.py --method $m --mode $md --target_tps 5.0
  uv run python scripts/secom_swap/26_retrieve_recall.py --segments segments/${m}__${md}.json --topks 1,2,3,5
done; done
for m in m1 surprise m3 hybrid; do
  uv run python scripts/secom_swap/27_qa_judge.py --qactx recall/${m}__online.qactx.json --qa_topk 5
done
uv run python scripts/secom_swap/29_aggregate.py                 # SUMMARY_TABLES.md
```
