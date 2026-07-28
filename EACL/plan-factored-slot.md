# plan.md — 실행 로드맵 (kill-switch 우선, 실패 분기 포함)

> **현재 진행 위치·우선순위는 이 문서에 적지 않는다** (Phase가 진행될 때마다 낡는다). **유일한 갱신 지점 = `phase0-decisions-and-runs.md` 상단 "📌 현재 유효 상태 스냅숏".** 이 문서는 Phase별 설계·GATE·실패 분기만 담는다.

> 짝 문서: `context/methodology/factored-slot-eism-v0.1.md` (설계 명세). 이 문서는 **무엇을 어떤 순서로 시도하고, 실패하면 어디로 분기하는지**만 다룬다.
> 핵심 원칙: **가장 치명적인 가정을 가장 싸게 먼저 깨본다.** 무거운 모델을 짓기 전에, 틀리면 전부 무너지는 전제부터 헐값 실험으로 검증한다.
> 각 단계: **목표 / 행동 / 통과 기준(GATE) / 실패 시 분기(→)**. 임계값은 잠정치이며 Phase 0에서 보정.

> **⚠️ GRANULARITY 정정 (BREAKING) — 이 로드맵 전체는 "사건 경계(B)" 기준으로 읽는다.**
> 타깃 = 이야기 *안*의 사건 경계이지 "썰과 썰 사이(A)"가 아니다. 상세: ledger의 "GRANULARITY 정정" 섹션.
> **Phase 0 재설정:** ① WritingPrompts concat(=A 경계) 실행 S1~S4는 **타깃 불일치** — 측정 위생(halo/이월/노이즈 제거, closed-enum, carry-forward) 자산만 승계하고 수치는 H1 검정에서 제외. ② **본실험 데이터 = 사람이 붙인 사건 경계**(Pieman/Monkey/Tunnel; 최종 gold는 Michelmann 통일본 — Pieman 18 / Monkey 30(goldA)·40(goldB) / Tunnel 18, ledger "동결 갱신" 참조. 초기 phoebsc 기준 24/43/30은 폐기). ③ **체호프 concat 취소** — 이어붙이기는 A 경계만 생성. "문체 동질로 IDM 무력화"는 **B(단일 이야기 내부)가 자동 제공**(문체 균일 → IDM 약화; mouthguard 마진 0.04가 그 증거). ④ 측정기는 **S5**(사건 경계용 enum 재설계, `phase0-annotation-guide.md`)로 교체.
> **Phase 1 영향:** 병합 코퍼스는 **경계 라벨 소스로는 무효**. 다만 **dynamics 자기지도(next-obs) 사전학습용 대량 서사**로는 여전히 유효. 경계 평가·anchor 라벨은 사건 경계 데이터에서 확보.

---

## ⚠️ 방법론 명명 (필독 — `method-registry.md`가 canonical)

**"IDM"은 M1이 아니다.** IDM = Inverse Dynamics Model = **M2(MonoLAM) 내부 모듈만.** 기존 결과표의 "IDM" 열은 전부 **CosDrift(M1)**로 읽는다. M1~M7 전체 지도·정의·상태는 `method-registry.md` §1을 본다 (여기 중복 게재하지 않음).

**이 로드맵의 Phase 2~5 = M4~M7.** 핵심 대조: **M5 vs M4 = 표현 분리의 순수 효과** / **M4 vs M2 = anchor의 순수 효과.**
M2(anchor 없는 monolithic)는 **이미 실패 확인** — 단 M2는 5축 압력이 없었으므로 **M4의 실패를 의미하지 않는다.** M4는 반드시 별도로 돌린다.

**베이스라인 사다리 → Phase 매핑 (기존 'K=1(단일잠재/IDM)' 표기는 폐기):**

| 조건 | 표현 | anchor | 통제 대상 | 상태 |
|---|---|---|---|---|
| M1-CosDrift | 없음(코사인) | — | 학습이 필요한가 | 완료 |
| M2-MonoLAM | monolithic | ❌ | anchor 없이 되나 | **완료·실패** |
| **M4-MonoAnchor** | monolithic | ✅ | **분리가 필요한가** | **미실행 — 핵심 통제군** |
| **M5-FactoredSlot** | factored 5 | ✅ | (메인) | 미실행 |
| M6-FactoredShared | factored 5, dyn 공유 | ✅ | 축이 교환가능한가 | 미실행 |
| M7-SlotAttnFactored | factored 5, slot-attn | ✅ | 단순 분해로 부족한가 | 미실행 |

**⚠️ Phase 2 GATE 보강:** 공유(M4) vs 분리(M5)의 **probe 정확도·교차혼선**을 Phase 5가 아니라 **Phase 2에서 이미 비교**한다. monolithic은 "어느 축이 리셋됐는지 읽을 수 없어" **H1 검정 자체가 불가능**하므로, 이 비교가 일찍 나와야 한다.


---

## 위험도 순서 요약 (왜 이 순서인가)
1. **[P0]** 핵심 과학 베팅이 참인가? — "다지표 동시 리셋 > 스칼라 드리프트" 신호가 *애초에 존재하나*. 틀리면 논문 죽음. 제일 쌈 → 맨 앞.
2. **[P1]** 데이터 파이프라인이 서나 — 병합 코퍼스 + change-silver. 이게 없으면 하류 전부 굶음.
3. **[P2]** 약지도로 슬롯이 *실제로* EISM 차원에 정렬되나 — 이론상 가능(§5)이지만 실전 붕괴 위험.
4. **[P3]** 학습된 dynamics가 raw 드리프트를 오프라인에서 이기나 — 모델링과 추론을 분리해 검증.
5. **[P4]** 온라인 추론이 감당되나 + 지연커밋이 돕나.
6. **[P5/P6]** 전체가 베이스라인을 이기나 + 논문화.

---

## Phase 0 — Kill-switch 실험 ✅ 완료 (판정: 조건부 Go, scope = narrative fiction)

> **✅ 실행 완료 — 실제 경위와 판정은 ledger(`phase0-decisions-and-runs.md`)가 canonical.** 요지: S1~S4는 타깃 불일치(A 경계)에서의 측정법 개발이었고, B 데이터(Pieman/Tunnel/Monkey)로 옮긴 뒤 M3(S5)가 서사 픽션에서 M1-CosDrift 상회. **단 selection-on-test 오염으로 7개 데이터 전부 dev 강등(D-M5), held-out 승리 0 → 정식 숫자는 Tier 2가 공급.**
> 아래 원문은 **당시 설계 기록**으로 보존한다. GATE도 이후 재정의됨(AUC +0.10 → **Score gain ≥ +0.10 & bootstrap-by-doc CI > 0**, ledger MAIN METRIC/GATE 참조).

**목표(원문).** "coordinated multi-index reset이 scalar drift가 못 가르는 것을 가른다"(H1의 가장 날것 버전)가 **데이터에 존재하는지**를 모델 없이 확인.

**행동.**
- 소량 평가셋 확보: 사람 gold 경계 ~100–200개 + 내부(비경계) 지점. (기존 보유 예시 + 수동 몇 개 + LLM silver 소량.)
- Off-the-shelf 추출기로 5지표를 hand-extract (coref/NER=인물, 공간표현=장소, 시간표현/시제=시간, 담화연결어=인과, LLM 이진=목표).
- 문장 전이마다 **지표별 이진 리셋** 계산 → `#simultaneous_resets` (0~5).
- 대조: 같은 지점의 **IDM 스칼라 드리프트** ‖Δembedding‖.
- 두 신호로 경계 예측 비교: AUC, 그리고 "경계 vs 내부"의 분리 마진.

**GATE.** `#resets` 신호의 AUC가 IDM 대비 **+0.10 이상**, 그리고 경계/내부 마진이 IDM의 0.04(관측된 실패 마진)보다 확연히 큼(예: 리셋수 분포가 경계에서 3+, 내부에서 0–1로 갈림).

**→ 실패 시 분기.**
- **(a) 추출기 탓 의심** (지표는 맞는데 추출이 엉성): LLM으로 지표 추출을 격상해 재측정. 개선되면 GATE 재평가 → 통과 시 "추론 시엔 경량 추출/슬롯으로 증류" 전제로 진행.
- **(b) 지표가 동시에 안 리셋됨** (경계에서도 리셋수가 낮거나, 내부에서도 높음): 핵심 베팅 흔들림. 지표별로 쪼개 *어느 지표가 경계와 상관 있나* 진단(H2 조기 확인).
  - 특정 소수 지표(예: 인물+장소)만 강상관 → **descope**: 5지표 factored-SEM 대신 "인물+장소 리셋 탐지" 축소 모델로 방향 전환(여전히 publishable, 야심 축소).
  - 어떤 지표도 안 통함 → **피벗**: "왜 의미기반 분절이 내러티브에서 실패하는가"의 진단/negative-result 논문으로. (Phase 0 자체가 그 논문의 핵심 그림이 됨.)
- 어느 쪽이든 **무거운 모델을 짓기 전에** 방향이 정해짐 — 이게 P0를 맨 앞에 둔 이유.

---

## Phase 1 — 데이터 파이프라인 (1–2주) — ⚠️ GRANULARITY 정정 반영판

> **정정 반영 (D-B1~B4 cascade):** 원안의 "병합 코퍼스 = gold 경계 무료"는 **A 경계라 무효.** 병합 코퍼스는 **dynamics 자기지도(next-obs) 사전학습 전용**으로 강등된다. **경계 평가 = Tier 2 held-out**(`tier2-heldout-spec.md`, LLM-silver, doc 10+), **anchor 라벨 = M3(S5) enum**(`phase0-annotation-guide.md`) — Phase 0에서 이미 확보된 인프라를 쓴다.

**목표.** ① dynamics 사전학습용 대량 **서사 픽션** 코퍼스(scope 확정의 파생 — 저널리즘·대화 제외), ② M3 anchor 라벨 파이프라인 스케일업, ③ Tier 2 held-out 완공, ④ onset prior 표본(각 사건 세그먼트 첫 문장).

**행동(정정판).**
1. **서사 코퍼스**: 소설·단편 대량 수집(Gutenberg 등) — 이어붙이기 없이 **원문 그대로**, dynamics 자기지도 전용. (원안의 문체 치팅 방지 정규화는 경계 gold 용도가 사라져 불필요.) **⚠️ (D-M8) 코퍼스 = 서사 픽션 단일 — 스크린플레이 등 대사-only 데이터 혼합 금지**(M5 vs M4 인과 대조 보호; 스크린플레이는 본편 불포함 → 확장 Phase X). 단 선정 기준에 **대사 비중 스펙트럼 다양화** 포함 — 대사 밀도 높은 픽션을 섞어 서술문(anchor 공급)·대사 구간(암묵 입력)의 자연 커리큘럼(in-scope LUPI 구조) 확보.
2. **anchor 라벨**: M3(S5) 2-stage 추출을 코퍼스에 적용 — 축별 독립 호출·닫힌 enum·코드 변화판정(Phase 0 측정 위생 그대로). 다중 샘플 다수결 + 신뢰도.
3. **held-out gold**: 신규 사람 주석 대신 **Tier 2 스펙**을 따른다(GPT-4 × B2 프롬프트 × 20회 합의). human gold 300–500은 **Tier 3(한국어 썰)**로 이월.
4. **onset 표본**: 사건 세그먼트 첫 문장 수집 → onset prior 학습셋. (세그먼트 정의는 Tier 2 silver 경계 기준.)

**GATE.**
- silver vs human gold 일치도(경계) κ ≥ 0.6, 지표별 change 일치도 각 ≥ 0.5.
- onset 유효표본 ≥ (목표 하한; = 총문장/평균세그길이로 역산해 충분).
- 지표별 silver의 **상호 상관**이 human gold의 상관을 크게 초과하지 않음(LLM이 "경계면 다 리셋" 편향으로 상관을 부풀리지 않는지 — H1 오염 방지).

**→ 실패 시 분기.**
- silver 일치도 낮음 → 프롬프트 재튜닝(Pangakis식 소표본 검증 루프); 그래도 낮으면 문제 지표는 change-silver 포기하고 **absolute 추출 delta로 대체**(그 지표만).
- 상관 부풀림 감지 → 질의를 더 강하게 분리(지표당 별도 세션·문맥 최소화); 그래도면 H1 검정을 "동시리셋 수" 대신 **상관보정된 통계량**으로 재정의(P5 평가와 연동).
- onset 표본 부족 → 병합 규모 확대 or 세그먼트 길이 짧은 소스 비중↑; 그래도 부족하면 onset prior를 **슬롯별 파라메트릭 사전**(적은 파라미터)으로 단순화.
- 병합-실제 gap 큼(병합 경계가 너무 깨끗) → 연속독백 gold 비중↑ + onset prior를 실제 gold로 few-label 보정(§10 리스크 대응).

---

## Phase 2 — 슬롯 정렬 de-risk (dynamics 없이, 1–2주)

**목표.** typed slot + anchor 손실만으로 **슬롯이 실제로 EISM 차원에 정렬되는가** + 붕괴 안 하는가 확인. (아직 SEM dynamics/스위칭 없음. frozen 인코더 위에 슬롯+anchor만.)

**행동.**
- EntNet식 keyed 슬롯 K개 + change-readout 헤드 $h_k$ + `L_anchor`(pairwise change-supervision 주력, absolute 보조; §2 정밀화안).
- bind-then-release 커리큘럼($\lambda$ 감쇠).
- **probing 평가**: 각 슬롯 상태 → 해당 지표를 선형 probe로 얼마나 복원하나(정렬도). 슬롯간 혼선(인물슬롯이 장소를 예측?) 측정.

**GATE.**
- 슬롯별 probe 정확도: 정박된 3지표(인물/장소/시간) 각 ≥ (무작위 대비 유의, 예: 절대 0.7+).
- 교차혼선 낮음(슬롯 $k$가 지표 $j{\neq}k$를 예측하는 정도 << 자기 지표).
- 슬롯 붕괴 없음(어떤 슬롯도 상수/미사용 아님).

**→ 실패 시 분기 (정렬 실패의 사다리).**
1. change-supervision만으로 정렬 약함 → **absolute 항 $\lambda_{abs}$↑** (슬롯에 절대 grounding 추가).
2. 그래도 약함 → **D3 잠재 → D1 전환**(그 지표만 extractor-defined 공간으로; 누수 안전, 발견 포기하되 정렬 보장).
3. 슬롯 붕괴 → commitment/엔트로피 정규화, EntNet **고정 keyed init**.
4. 인과/목표가 끝내 안 정렬(예상된 난점) → **v1은 인물/장소/시간 3축만 typed**, 인과/목표는 잠재-only 또는 단일 residual 슬롯으로 강등. (proposal §6/§10과 일치.)
- 최악: 3축도 정렬 실패 → 이론(§5)상 가능해야 하므로 anchor 신호 품질(P1) 되짚기; 그래도면 **descope**: 슬롯 학습 포기, P0의 hand-extracted 지표를 그대로 스위칭 모델 입력으로(발견 없는 버전, 여전히 H1 검증 가능).

---

## Phase 3 — Dynamics + 스위칭 (오프라인 먼저, 2–3주)

**목표.** 슬롯별 dynamics + sticky-CRP 스위칭을 얹어 **전체 생성모델**을 완성하되, **오프라인(전체 시퀀스, 온라인 제약 없이)**으로 먼저 돌려 *모델링*과 *추론*을 분리 검증. "학습된 dynamics가 raw 드리프트를 이기나"(H1 완전판) 확인.

**행동.**
- 슬롯별 predictive: 인물=점과정, 장소/시간=소형 MoG, 인과/목표=MoG(먼저). 자기지도 next-obs로 dynamics 학습(병합 코퍼스).
- 스위칭: 우도비 곱 × CRP(§4.5), 분자=amortized onset + reuse(§4.7).
- 오프라인 분절(전체 시퀀스 MAP/스무딩)로 Track A(표준 벤치) + Track B(병합) 평가.
- **핵심 진단**: 스위칭 사후확률이 (a) raw 드리프트가 아니라 (b) dynamics-서프라이즈와 상관하는지. P0의 mouthguard 패턴 재현("인과슬롯 자주 놀라도 경계 안 만듦, 경계는 다슬롯 동시").

**GATE.**
- 오프라인 $P_k$/WindowDiff가 **IDM·TextTiling·Cross-Segment BERT 베이스라인을 유의하게 상회** (Track B).
- 진단 통과: 경계=다슬롯 동시 비율상승, 인과슬롯 단독 스파이크는 비경계.
- Track A에서 최소 경쟁력(참패 아님 — 리뷰어 방어).

**→ 실패 시 분기.**
- 인과슬롯이 다봉이라 MoG 과소서프라이즈 → **VQ-범주형으로 교체**(누수도 같이 차단; proposal §4.2 대안).
- 누수 재발(dynamics-서프라이즈가 결국 raw 드리프트와 상관) → 슬롯-native 공간 강화 / VQ / 예측타깃을 구조화 공간으로.
- **학습 dynamics가 raw 드리프트를 못 이김**(가장 아픈 실패) → 진단: dynamics 학습 부족(데이터/용량)인가, 원리적 부재인가.
  - 데이터 부족 의심 → 병합 코퍼스 확대, dynamics 용량↑, 재시도.
  - 원리적 → **fallback**: dynamics를 버리고 P2 정렬 슬롯의 **change-신호 + 스위칭 사전**만으로 분절(예측오차 대신 anchor change를 경계 신호로). 덜 우아하지만 H1(동시리셋)은 여전히 검증. 논문 축은 "learned dynamics"에서 "typed coordinated-reset detection"으로 이동.
- 슬롯/코드북 붕괴 → 정규화(commitment, 성분 엔트로피).
- CRP reuse 항 폭발 → 활성 윈도우/나이제한.

---

## Phase 4 — 온라인 추론 (1–2주)

**목표.** particle filtering + K′문장 지연커밋으로 **온라인화**. 오프라인 성능에 근접하며 라이브 지연 감당.

**행동.**
- particle filter(연속/재사용/신규 스위칭), 경계 사후=전환 파티클 비율.
- 지연커밋= K′문장 particle smoothing.
- 지연 K′·파티클수 P를 스윕 → **latency–accuracy 곡선**.

**GATE.**
- 온라인 $P_k$가 오프라인 대비 열화 ≤ (허용치, 예: +10% 이내).
- 문장 처리속도가 다이어리 실시간 여유(스텝당 상수시간 확인).

**→ 실패 시 분기.**
- 파티클 고갈(degeneracy) → 리샘플링 개선 + jitter + 파티클수↑.
- 너무 느림 → 파티클수↓, onset amortized proposal 강화, 슬롯수↓(인과/목표 병합), reuse 후보 축소.
- particle filter 불안정 → **경량 대체**: 학습된 슬롯-서프라이즈 스트림 위에 **다변량 BOCPD**(Adams–MacKay)로 온라인 변화점 탐지. 완전 베이지안은 아니지만 안정·저비용, 온라인·LLM-free 유지.
- 지연커밋 이득 없음 → K′=0(즉시 커밋)으로 두고 정밀도 손실 수용 or 대체 lookahead 규칙.

---

## Phase 5 — 전체 평가 · 애블레이션 · 교차언어 (2주)

**목표.** 주장 완성. 베이스라인 사다리(§7)를 **같은 코드의 노브**로 돌려 애블레이션=가설검정. 최종은 연속독백 gold.

**행동.**
- 베이스라인 노브: K=1(단일잠재/IDM), λ_anchor=0(교환가능 슬롯), 인물외 마스킹(EntNet), dynamics제거(IDM), 균질가우시안(분포족 통제).
- **H1**: 스칼라 vs 다변량; **H2**: leave-one-index-out; **H3**: EST-scalar vs full-EISM.
- LLM-프롬프트(LumberChunker류)를 **oracle/topline**으로만(추론 금지, 상한 표시).
- 교차언어: 한국어 썰(faster-whisper large-v3 / Qwen3-ASR 전사)로 "무대 리셋" 언어불변성 검증.
- 메트릭: $P_k$, WindowDiff, B, latency–accuracy 곡선.

**GATE.**
- 연속독백 gold에서 베이스라인 사다리 전부 상회, 특히 **교환가능 슬롯(λ=0) 대비 우위**(=타이핑의 인과적 기여 입증).
- H1–H3 중 최소 H1 유의 지지.
- 한국어에서 유의미한 전이(완전 동등 아니어도 방향 일치).

**→ 실패 시 분기.**
- 교환가능 슬롯과 차이 없음 → "성능향상=용량"이란 반박 성립 → 타이핑 서사 약화. **fallback 논문**: 타이핑 대신 "coordinated-reset 신호(수단 무관)가 semantic cohesion을 이긴다"로 주장 재편(H1 중심, 아키텍처 주장 후퇴).
- 한국어 전이 실패 → scope를 영어/장르로 한정 선언 + 실패조건 정직 보고(오히려 신뢰도↑).
- 특정 H 기각 → 그 자체를 결과로 보고(예: H2 기각=모든 지표가 고루 기여).

---

## Phase 6 — 논문화 · novelty lock (1–2주)

**행동.**
- 관련연구 정밀 서베이(proposal roadmap #5): SEM/latent-cause/EntNet/Slot-Attention/segmentation 계열 원문 확인, novelty 문장 3개 확정(§5).
- motivating 그림 = P0/P3 진단(인과슬롯 vs 경계 분리).
- 타깃: 방법 강조→ACL/EMNLP/NAACL; 인지 정박 강조→CogSci/CMCL 워크숍(1차 저위험 관문).

**전체 fallback 사다리 (프로젝트는 항상 어딘가 착지).**
- **A. 완전 성공** → factored-slot 잠재 상황모델 + 온라인 LLM-free 에피소드 분절 (풀 논문).
- **B. 정렬 O, dynamics 한계**(P3 fallback) → "EISM 지표 coordinated-reset이 semantic cohesion을 이긴다" (typed feature + 경량 탐지기).
- **C. 신호는 존재, 모델 난항**(P0 (b) descope) → 소수 지표 리셋 탐지 or "왜 의미기반이 실패하나" 진단/negative-result.
- 각 단계 GATE 실패가 자동으로 다음 착지점을 지정 → **매몰 방지**.

---

## Phase X — X1 ScreenplayAnchor: 대사-only 전이 프로그램 (post-EACL 확장 — 본편 범위 아님)

> **기원.** 사용자 제안(학습 2원화 — place/time을 슬러그라인으로 채운 스크린플레이 하이브리드)을 본편 학습에서 제외하고 여기로 이월(ledger **D-M8**; 근거 a–d). 경험 기반 = **D-M7**: 대본 gold에서 세션 경계 83% 실제 place 변경(가설 참) vs M3 recall 4%(측정 실패) — "잡을 게 있는데 안 보이는" 도메인임이 확정됨.
> **질문.** 서술 채널이 없는 대사-only 스트림에서, **학습된 슬롯이 place/time 잠재 상태를 얼마나 복원하는가** — 명시 추출(M3)이 원리적으로 굶는 도메인에서 "잠재 추론 > 명시 추출"이 성립하는가.
> **패러다임.** 특권정보 학습(LUPI; Vapnik & Vashist 2009) / generalized distillation(Lopez-Paz et al. 2016): **씬 헤딩은 학습 시 anchor로만 사용, 추론 입력 금지** (추론에 주입하면 gold와 동일 원천의 oracle 순환 — D-M7 oracle 천장 ≈0.85–0.9 추정 참조).

### X1-0 — 정보 상한 게이트 (LLM ceiling probe; 싸다 · 선행 필수)
- **행동.** 강한 LLM에 대사-only(+양방향 문맥 허용)로 place/time **절대 상태**를 추정시켜 슬러그라인 gold로 채점 → 축별 "대사가 실어 나르는 정보량"의 상한 측정.
- 문맥 허용이 정당한 이유: H1 gold 생성이 아니라 **독립된 인간 저작 gold(헤딩)에 대한 상한 진단**이므로 halo 오염 무관.
- **GATE.** 축별 상한 > chance. 미달 축은 프로그램에서 제외 — "그 축의 정보가 대사에 원리적으로 부재"라는 경계조건 확정(그 자체로 보고 가치). 부재 사례 실측 존재: BERADA↔DONNIE 연속 미팅 — "같은 사무실·다른 날 vs 다른 장소·같은 날"은 대사만으로 원리적 구별 불가.
- **기대치 캘리브레이션.** 이 상한은 세계지식 기반 추론("웨이터가 주문 받는 대사 → 식당")을 포함하므로, 작은 증류 모델(M5급)의 현실적 상한은 이보다 아래다 — LLM-free는 세계지식을 속도와 맞바꾸는 거래임을 결과 해석에 반영.

### X1-1 — 원 시나리오 코퍼스 구축 (⚠️ SHARE 파생 금지)
- **SHARE 재사용 금지 이유(D-M7 실측):** 화자쌍 재조립로 protagonist 구조적 死(2인 고정) + 세션≠씬 17%(같은 씬 내 세션 경계 존재).
- **원 대본**(ScriptBase — Gorinski & Lapata 2015 — 등)에서 직접 파싱.
- **라벨 (사용자 제안 구성, 축별 출처):**
  - **place / time = 슬러그라인** (씬 수준; 씬 내 불변 = 지속성 구조 자체가 학습 신호. time은 거친 DAY/NIGHT/CONTINUOUS 토큰).
  - **protagonist = 화자 태그 기반 씬별 발화 cast** (강 anchor 후보 — SHARE와 달리 원 대본에선 살아있음).
  - **causality / goal = 대사 기반 약/잠재 anchor** (스펙 §6 mixed anchoring 정합; 대사 내 신호 약함(AUC ≤0.54)을 전제로 약 anchor만).
- **입력 = 대사만** (헤딩·지문 전부 제거; 화자명 태그 노출 여부는 구축 시점에 결정·ledger 기록 — protagonist anchor와의 정보 중복에 유의).

### X1-2 — 학습
- M5 아키텍처 재사용 + bind-then-release 커리큘럼. 헤딩 anchor는 **학습 손실에만** 등장.

### X1-3 — 평가 (3행 사다리; 바닥·천장 이미 확보)

| 행 | 값 | 지위 |
|---|---|---|
| **바닥** | M3 대사-only: place recall 4% / time 0% / Score 0.335 (SHARE valid 25-doc, doc-CI) | 실측·봉인 (D-M7) |
| **검정 대상** | X1 학습 모델 (대사-only 입력) | 프로그램 산출 |
| **천장** | oracle 헤딩 주입 ≈0.85–0.9 (추정) | **순환 — 성능으로 인용 금지**, 간격(0.335↔0.9)의 동기 부여용 |

- **판정.** 학습 모델이 M3 바닥선을 CI 기준 상회 → **"잠재 추론 > 명시 추출"** = M5 노벨티의 독립 확장(별도 논문감). 실패 → "서술 채널 부재는 학습으로도 미회복"이라는 모달리티 경계조건 확정 보고.
- **⚠️ 평가 gold 재정의.** 세션 경계 ≠ 씬 전환(D-M7: 17% 불일치 — 같은 씬 내 세션 경계 존재). 경계 gold는 세션 인덱스가 아니라 **헤딩 변화 기준**으로 재정의하고, 축 수준 평가(place 슬롯 상태 전이 vs 헤딩 장소 변화)를 병기 — 슬롯 접지의 축 단위 반증 가능성 확보.
- **근거 문헌.** LUPI(Vapnik & Vashist 2009) · generalized distillation(Lopez-Paz et al. 2016) · EntNet(Henaff et al. 2017 — 재언급 없는 잠재 세계상태 유지) · GateL0RD(Gumbsch et al. 2021 — 드물게 변하는 잠재상태 귀납편향) · ProPara(Dalvi et al. 2018 — 명시 안 된 엔티티 상태 추적 학습) · ScriptBase(Gorinski & Lapata 2015 — 코퍼스). 인지 근거: 독자·청자는 명시 진술 없이도 공간·시간 상황모델을 추론으로 구축한다(Zwaan & Radvansky 1998 계열) — 라디오 드라마·지문 없는 희곡 독해가 실례. 즉 구성개념은 대사-only 매체에도 존재하며, 검정 대상은 그 계산적 복원 가능성이다.

---

## 지금 당장

**→ ledger 스냅숏의 "다음" 항목이 canonical.** 여기 복사해 두면 낡으므로 적지 않는다.

> Phase-트리거 미결 사항(날짜와 무관, P1/P2 진입 직전에 닫는다): change-set 시간정렬 지터, 인물 점과정↔change 이중계상, change-silver 상관노이즈(P1 GATE에 반영됨), absolute 항 최소강도.

---

## 🎯 목표 위계에 따른 우선순위 재조정 (CANONICAL — `method-registry.md`와 동기)

> **주목표 2개 + 서브목표 1개.** SOTA는 **비목표가 아니라 서브목표** — 되도록 확보하되, 시간 부족 시 주목표에 자원 집중.

| 목표 | 내용 | 필수 실험 | 우선순위 |
|---|---|---|---|
| **RQ1 (주)** | **EISM의 계산적 검증** — 인간의 5지표 사건분절이 구현했을 때 통하는가, 그리고 **언제 실패하는가** | ① **M5 vs M4** (분리의 필요성 = 논문의 심장) · ② **M5 vs B1** (EISM vs EST 이론 대결) · ③ **probe** (슬롯이 실제로 축을 추적하는가) · ④ **경계조건 진단** (Monkey 4구간) | **최상** |
| **RQ2 (주)** | **실시간 배포 가능성** — 온라인·LLM-free | ⑤ **M5 구현**(문장 스트리밍, 추론 시 LLM 無) · ⑥ latency/throughput · ⑦ 증류 후 성능 유지 | **최상** |
| **S1 (서브)** | 경쟁력 있는 정확도 | ⑧ **M1 상회**(달성) · ⑨ **B1 상회**(필수 — H1 검정이므로) · ⑩ B2/B3 대비(topline, 이기면 보너스) · ⑪ Tier2/3 held-out | **중** — 시간 허용 시 |

**⇒ 시간이 부족하면 ⑩·⑪을 버리고 ①~⑦에 집중한다. ⑨(B1)는 S1이지만 실질적으로 RQ1에 속하므로 버리지 않는다.**

### 즉시 우선순위 (Phase 0 마무리 → Phase 1 진입)

1. **Monkey 4구간 진단** (ledger D-M5-d) — 주석 프로토콜 교란(4그룹이 각 1/4씩 분절) vs 장르 요인 판별. **RQ1의 "경계조건" 주장의 타당성이 여기 달림.** 싸고 결정적.
2. **B1(Kumar 2023) 원문 정독 + 재구현** — 특히 **Monkey 결과**와 Bayesian surprise 계산식. 코드 미공개(`kenmcgarry/BayesSurprise`는 무관한 레포).
3. **Tier 2 held-out 구축** — 새 서사 + B2/B3 프로토콜(`s-michelmann/GPT_event_segmentation`, `ryanapanela/EventRecall`)로 silver gold. **doc 다수 → bootstrap CI 최초 확보.**
4. **M5 구현 → M4 대조** (Phase 2~3). **RQ1·RQ2의 본체.**

### 리뷰어 방어 (미리 준비)

예상 질문 3종("그냥 GPT-4 프롬프팅하면 되잖아" / "SOTA가 아닌데" / "Monkey에서 실패했잖아")과 답변은 **`method-registry.md` "리뷰어 예상 질문과 답"이 canonical.** 로드맵 관점의 추가 조건 하나만: Monkey 방어는 **4구간 진단(ledger D-M5-d) 선행이 필수**다.
