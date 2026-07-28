# Hi-OnTop — EACL 연구 문서

## 📌 먼저 읽을 것 (순서대로) + 문서별 canonical 범위

각 사실은 **한 문서에만** 산다. 아래 범위를 벗어난 중복 게재는 금지하고, 다른 문서에서는 링크로 가리킨다.

1. **`method-registry.md`** — **방법론 명칭·전체 지도(M1~M7, B1~B3)·논문 포지셔닝(RQ1/RQ2/S1)·리뷰어 Q&A·순환 방지 규칙**의 canonical 소스.
2. **`phase0-decisions-and-runs.md`** — Phase 0 **결정·실행 ledger** (append-only). 모든 실행 수치·판정·정정의 canonical 기록. **상단에 "현재 유효 상태 스냅숏"과 "결정 색인"(무엇이 superseded인지)이 있다 — 시간순 본문을 읽기 전에 이것부터 본다.**
3. **`phase0-annotation-guide.md`** — **M3(S5-Extractor) 스펙**의 canonical: 5축 enum·리셋 규칙·동결 설정·평가 프로토콜·기각 변형.
4. **`plan-factored-slot.md`** — 실행 로드맵 (Phase별 GATE·실패 분기·우선순위).
5. ~~**`tier2-heldout-spec.md`**~~ — ⛔ **폐기(D-M13①).** Tier2 held-out 제외. held-out = Tier1 6편(신규 *Born a Crime* 3편). 역사 기록으로만 보존.
6. **`factored-slot-eism-v0_1.md`** — 방법론 설계 명세 (이론 정박·생성모델·M5 아키텍처).

---

## ⚠️ 두 가지 필수 전제 (모르면 문서가 안 읽힘)

1. **"IDM"은 M1이 아니다.** IDM = Inverse Dynamics Model = **M2(MonoLAM) 내부 모듈만.** 기존 코드·결과표의 baseline "IDM"은 전부 **CosDrift(M1)** — 학습·예측 없는 코사인 드리프트 화제경계 탐지기 — 의 오칭이다. 정정표·상세: `method-registry.md` §0.
2. **타깃 = 사건 경계(B), 썰 경계(A)가 아니다.** 이야기 *안*의 사건 단위가 타깃. 정정 이전(S1~S4)의 실행은 A 경계(concat seam)라 **타깃 불일치** — 측정 위생 교훈만 승계, 수치는 H1 검정에서 제외. 상세: ledger "GRANULARITY 정정".

## 방법론 지도 (요약 — 전체 표는 `method-registry.md` §1)

**완료:** M1 CosDrift(scalar 한계) · M2 MonoLAM(실패 — anchor 없는 monolithic은 대규모 학습으로도 붕괴) · M3 S5-Extractor(조건부 Go, **M4~M7의 anchor 라벨 공급원**).
**예정:** M4 MonoAnchor(통제군) · **M5 FactoredSlot(★메인)** · M6 FactoredShared · M7 SlotAttnFactored. 핵심 대조 = **M5 vs M4(분리의 인과적 기여)**.

---

## 현재 상태 (canonical = ledger 상단 "📌 현재 유효 상태 스냅숏" — 아래는 그 요약)

- **Phase 0: 조건부 Go — scope = narrative fiction.** M3가 서사 픽션에서 M1-CosDrift를 상회(dev). ~~판별자는 place(무대) 축~~ → **D-M13⑤ 철회**(표본 부족).
- **🔴 그러나 held-out 승리는 0개다.** 9개 변종의 test 점수를 본 뒤 설정을 골라 **selection-on-test 오염** → **7개 데이터 전부 dev로 강등**. 현재 수치는 **탐색적 결과**이며 held-out 성능이 아니다.
- **fiction held-out doc이 1개뿐 → bootstrap CI 불가.**

## 다음 단계 (canonical = ledger 상단 스냅숏 "🔒 다음" — 아래는 요약)

1. ~~Monkey 4구간 진단~~ ✅ · ~~B1-Kumar 재구현~~ ✅ · ~~gold-swap → D-M12~~ ✅ · ~~held-out 재편 + 라벨러 검증 → D-M13~~ ✅.
2. **⬜ 지금 여기 → 저비용 라벨러용 S5 프롬프트 재설계** (D-M13⑥ — deepseek-v4-pro 기각, fable5 예산밖). dev 3편에서만 튜닝, 시험지 봉인. carry-forward 프롬프트 명시 + goal 축 재정의. 선행: 모델 확정 + goal divergence 진단.
3. **Phase 1 서사 픽션 코퍼스 구축 → M5/M4 학습 → 확인적 평가 = 신규 3편 + Tier3.** 시험지 유효 n≈1(같은 작품) 주의(D-M13③).
