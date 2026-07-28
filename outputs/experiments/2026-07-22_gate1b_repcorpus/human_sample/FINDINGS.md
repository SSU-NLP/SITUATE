# ①B human 표본 — 라벨링 중 발견한 설계 이슈 (질적 기록)

> 대표 코퍼스 100문장 블라인드 라벨링을 **사람이 직접 수행하며** 눈으로 발견한 것들.
> 정량 ✗율·PASS/FAIL 은 `RESULT.md`(채점 자동생성) 참조. 이 문서는 **문면을 봐야만 드러나는
> 설계 결함/가설**(숫자로는 안 나오는 것)을 남긴다 — dominant_actor 대명사 발견과 같은 종류의 산출물.
> **전부 frozen `p0s5-eism-v2` 대상이 아니라 "다음 VERSION 재등록 시" 반영 후보.** 지금 코드/프롬프트/산출물 수정 금지.
> 설계 권고는 프로젝트 규칙대로 **codex 위임**으로 받음(2026-07-23).

---

## A. space 축 — Stage 1 "주요 무대 누락"(어휘 공백)

- **사례:** `obs_77176`(1차대전 항공 pulp). Stage 1 locations = [chateau, enemy_airdrome, headquarters_tent,
  home_airdrome, landing_field, no_mans_land, road_crossing, route_nationale, trenches] — **지상만.**
  주인공이 대부분 있는 **"비행기/조종석/공중"이 목록에 없어** 비행 중 문장들이 전부 unspecified 가 된다.
- **성격:** Stage 2 회피(avoidance)가 **아님**. Stage 1 의 **recall miss(폐쇄어휘 공백)** — 별개 실패유형.
  - 회피 = 무대가 목록에 *있는데* 모델이 unspecified.
  - 어휘공백 = 무대는 명백한데 **목록 자체에 그게 없음**.
- **라벨링 영향:** 이런 문항은 "아니오 + 메모(무대=비행기, 목록 밖)"로 처리(모델 unspecified 와 일치). 회피 아님.
- **codex 권고(요지):** 최소·정합 안 = **옵션 1**(Stage 1 "physical setting" 정의를 "행위자가 위치·행동하는
  공간 무대"로 명시 + vehicle/interior/airspace 예시). 탈것을 무대로 넣는 것은 EISM space 와 충돌 안 함
  (기준 = 장소/사물이 아니라 **행위자가 점유하는 같은 공간 영역**; cockpit/aircraft_interior/ship_deck 정상).
  단 cockpit 과 sky 를 둘 다 평면 location 으로 두면 왕복 → **한 수준 고정**("탑승 중=내부, 외부 비행 묘사만 airspace").
- **시점:** 지금 수정 금지(동결). 다음에 라벨러를 **어차피 재등록할 때** 옵션 1 함께. **전제 = 누락이 흔하다는 증거**
  (한 사례로 무거운 재등록 정당화 불가). 진단 = era/문서별 "주요무대 누락률·unspecified율·목록크기·space reset 왕복률".

---

## B. intentionality 축 — 설계 이슈 3종 (codex 정식 권고, 우선순위 ② → ① → ③보류)

### ② 장면목표 vs 상시(기질)목표 혼합 — **심각도 매우 높음 (구성타당도 결함 후보)**
- **문제:** Stage 1 이 목표를 무차별 열거하여 **장면목표**(meet_the_stage, get_milk, escape; 시작·끝 있음)와
  **상시/기질목표**(control_self, avoid_giving_way_to_fancies, avoid_opposition — Yellow Wallpaper 화자가
  이야기 내내 유지)가 한 목록에 섞임. 상시목표는 **사건이 아님**.
- **왜 해로운가:** 상시목표가 라벨되면 carry-forward 로 **실제 장면전환을 넘어 지속**(경계 억제)하거나 산발적으로
  깜빡(가짜 경계). EISM/Zwaan goal 차원은 **이산·순차 목표**를 가정 → 기질목표가 이를 깸.
- **권고(다음 버전):** Stage 1 goals 지시문을 **"시작·진전·완료·포기가 가능한 bounded commitment"** 로 제한,
  **성격·일반욕망·도덕원칙·상시성향 제외**. (선택: 목표를 `actor+outcome` 정규화.) = 목록 단계에서 후보 자체를 거름.
- **상태:** 구조적 실제 문제로 판단. 재등록 정당화 진단 = 목표별 eventiveness 프로파일(span/N·산발 re-entry),
  leave-one-goal-out(상시목표 제외 시 reset/anchor 변화).
- **⚠️ 회색지대(spectrum, 문항 40 발견):** 장면목표 vs 기질목표는 **binary 아니라 연속체**다.
  `control_self`(전편·영구 성격=명백 상시) — **`enjoy_leave`(휴가/체류 *기간에 묶임* → 시작·끝·포기 가능 = 사건형에
  가깝지만, "즐기기"가 *확산적*이라 긴 carry 로 구간 내 경계를 덮을 위험은 공유)** — `escape`(명확 장면목표).
  ⇒ Stage-1 재설계는 "명백한 성향 제외"만으로 부족, **기간묶임+확산적 목표(enjoy_leave류)를 어떻게 다룰지**도
  규정해야 함(예: 확산적 목표는 reset 대상에서 제외하되 carry 는 허용, 또는 "관찰가능한 진전 행위가 있을 때만 활성").

### ① 대사의 "화제 vs 목적" — **가설 (확정 아님, 검증 대기)**
- **관찰:** 판정선은 "말이냐 몸이냐"가 아니라 **목표가 화제(topic, 언급만)냐 목적(purpose, 행동이 진행)냐**여야 한다.
  발화도 행위 — 명령/요청/약속/계획/질문이 목표를 실행할 수 있다("돈 내놔"=강도, 수사 질문="검시관 결론?"=investigate).
- **🔴 정확한 상태(과장 정정):** frozen 프롬프트는 원칙("only discussed" vs "the action serves it")을 **명시하긴 한다.**
  다만 예시가 **클래스당 1개**이고 하필 none=verbal("they talked about escaping"), goal=physical("he sawed the
  fence bars")이라 **표면특징(말/몸)이 라벨과 confound**돼 있고, **"대사→목표" 예시가 없다.** → 모델이 "대사=none"으로
  **오독할 위험**이 있다(LLM 은 few-shot 표면특징에 민감). **"반드시 오독한다"는 근거 없음 = 가설.** none-rate 0.73 은
  정황증거일 뿐.
- **검증(진단, 이게 있어야 ① 확정):** 명령·약속·계획·질문 등 **수행적 대사**를 모델이 **순수 논의 대사만큼 none** 찍나 측정.
  그렇다 → ① 확정. 아니다 → ① 기각(예시 안 고쳐도 됨). 실증 후보 = **문항 39("검시관 결론이 뭐였죠?")**: 정답=investigate_case,
  모델이 none 이면 ① 증거.
- **권고(확정 시):** 정의 유지, **예시를 "대사 내부 최소대조"로 교체**("discussed whether escape possible"→none /
  "Meet me at midnight so we can escape"→escape / "Hand over the money"→rob) + "명령·요청·협상·약속·기만·계획합의도
  목표 실행"이라 명시. (설령 지금 오독이 없어도 "대사→목표" 예시 **추가**는 모호함 제거라 무해한 보강.)

### ③ carry-forward 방향(goal-first vs goal-last) — **보류(사용자 결정)**
- **논점:** 현재 carry-forward 는 **forward(goal-first)** — 목표 첫 등장에 경계, 빌드업(none)은 직전 목표에 귀속.
  대안 goal-last = 빌드업이 목표 발화로 종료(빌드업은 앞 목표에 귀속, 경계는 목표 뒤). 표면 발화시점 ≠ 목표 활성시점.
- **결정: 안 고침(forward 유지).** 이유: backward 로 뒤집으면 goal-first 서사를 대칭 손상 + **미래정보 사용** + **SEM
  온라인성(forward local-MAP) 위반**. 게다가 ③처럼 보이는 게 **①의 부산물**일 수 있음(대사 none-gap 이 방향문제로 오인).
- codex: 전역 backward 비권고, forward 유지 + ffill/bfill anchor flip-rate 민감도만 측정, 크면 그때 별도 후보.

### 우선순위·상호작용 (순서 중요)
- **② → ① → ③(보류).** ② 안 치우면 상시목표가 아무 문장에나 붙어 ①을 **가림/상쇄**. ①이 none-gap 을 늘려 ③처럼
  보이게 함 → ① 전에 방향 판단하면 오결론. dev-3 에서 **누적 ablation**(②만 → ②+① → …)으로 각 기여 기록.
- **재등록 판정:** ①·② 확증 = 단순 버그 아닌 **측정도구 구성타당도 결함** → 새 VERSION+Gate C(ΔScore≥−0.03)+M4~M7
  재생성 정당. ③만이면 별도 민감도 분석으로 남김. **frozen v2 산출물은 그대로 보존.**
- **gold-free 진단 8종**(codex): 화행별 P(goal≠none), speech-act 최소대조 순서일관성, 목표 eventiveness 프로파일,
  leave-one-goal-out, none-gap 구조(대사/비대사별·run 길이·A–none^k–B), ffill/bfill 방향 민감도, 방향증거 비대칭
  (none 문장이 A/B 중 어디에 임베딩 유사), 다중실행 안정성.

---

## E. time 축 — sequence/jump "opens with" 비대칭 (**확정 결함**, 캐시로 실증)

- **문제(사용자 발견):** frozen 프롬프트가 **sequence 에만** "if THIS sentence **opens with** …(then/next/after
  that)" 위치조건을 걸고, **jump("later, the next morning, years later")·simultaneity("meanwhile")엔 위치조건
  없음.** → 같은 **문중** 위치라도 "then"(sequence)은 absent 여야 하나 "later"(jump)는 쳐질 수 있음.
  같은 문장내부 관계인데 **표지 종류로 판정이 갈리는 비대칭 = 결함.** ("He walked home and **later** ate dinner.")
- **🟢 확정(가설 아님) — 캐시 corpus 진단, 예산 0** (`scripts/eacl_phase0/time_marker_position_diag.py`):
  non-absent time 라벨 83건(absent 2162)에서 표지 **문두시작 여부**:
  - **sequence 50건: 문두 38 / 문두아님 12 (24%)** — 대체로 opens-with 준수(모델 OK). 단 문두아님 12 중 3건은
    "So/And yet"를 sequence 로 단 **별도 오발**, 순수 mid-then 2건.
  - **jump 29건: 문두 15 / 문두아님 14 (48%)** — **절반이 문중 발화.** 진짜 mid 확인: "a few moments later",
    "a minute later", "just two years ago", "for now he sees"(전부 문중). = **비대칭 실증.**
  - simultaneity 4건(소표본) 전부 문두.
  - 캐비엇: jump 48%는 regex 문두 미검출("in the course of the next two hours" 등)로 **약간 과대**, 정확치 ±몇 %.
    방향·크기는 견고(jump ≫ sequence 문중발화).
- **재현율 0.974 의 함정:** 모델이 mid jump 를 **일관되게** 포함해 재현은 높음 = **재현성 ≠ 정확성**(사용자 지적).
- **권고 수정(다음 버전):** 위치조건을 **대칭**으로 — jump/simultaneity 도 "opens with" 요구(문중 표지 = absent).
  boundary 의미상 정합(문두 시간부사=담화 시간전환=경계감 / 문중=한 문장 안 연속=경계 아님). 부차: sequence 가
  "So/And yet"(consequence/역접)를 시간표지로 오발하는 것도 배제 문구 추가.
- **우선순위 낮음:** time base rate 0.037·재현 0.974 로 5축 중 가장 안정 → 경계에 큰 영향 없음. 그러나 **구성 일관성
  결함은 실재**하므로 다음 버전 문구 수정 대상(재등록 별도 사유는 약함, ①②에 묶어 처리).

---

## F. 코퍼스 위생 — 앞뒤 잡물(front/back matter) leakage (**PG boilerplate=확실 / 광고=강한 정황**, 스캔 근사)

- **문제(사용자 발견, 문항 63):** `obs_37511`(46문장)에 **PG 제작자 크레딧**([0-1] "Produced by Larry B.
  Harrison…")·**출판사 광고/도서목록**([41-44] "KIGGINS & KELLOGG, Publishers, Booksellers & Stationers…
  JUST PUBLISHED, REDFIELD'S TOY BOOKS…")·**전사자 노트**([45] "Transcriber's Note")가 본문에 섞여 있음
  (≈7/46 ≈15%). 문항 63 = 그 출판사 광고 문장(본문 아님).
- **🔴 근거 등급(과신 금지):** ① **PG boilerplate = 확실** — "Produced by"·"Transcriber's Note"·"Online
  Distributed Proofreading"은 거의 모든 PG 텍스트의 **고정 삽입 문자열**(해석 아님). ② **출판사 광고 = 문면상
  강한 정황**(자동 확정 아님) — 상호+거리주소("KIGGINS & KELLOGG…123 William St.")·판촉문구("invite country
  merchants before purchasing elsewhere")·가격표("JUST PUBLISHED…Price")·ALL-CAPS 조판·본문 끝 back-matter
  위치. 구체 근거는 있으나 100% 정의는 아님 → **문서별 확정은 문면 확인 필요.**
- **원인:** `strip_pg`·story-slice(`slice_story`/`first_story_block`)가 START/END 바깥만 제거, **안쪽의 제작자
  크레딧·전사자 노트·뒷광고를 못 걸러냄.**
- **스캔(예산 0, regex 근사·오검 많음 = 스크리닝 도구지 증거 아님):** 진짜 leakage 는 **무명 문서에 집중** —
  obs_37511(최악), obs_28922(3, "Produced by Greg Weeks…"), obs_25744(1, "Produced by Al Haines"),
  obs_74162(2, "[Transcriber's Note]"). **유명작은 사실상 clean**(hits 대부분 "end of **the** bridge" 류 오검).
  전체 <1%(≈15~20/2245).
- **영향:** 비서사 문장에 5축 라벨이 붙어 **노이즈**. 양은 작으나 **무명작(=Phase-1 타깃)에 몰려** 스케일에서 커질 위험.
  **①B QC(provenance·coverage·parsing)는 이걸 못 잡음** — "JSON 유효"는 봐도 "서사인가"는 안 봄. **human 표본이 잡은 값.**
- **🔴 표본 오염 실측 = 4/100 (heuristic, 하한선):** human 표본 100 중 비서사 4개 —
  **#56**(time, obs_28922 "Produced by Greg Weeks…" boilerplate) · **#60**(time, twain "THEN CLAWED BACK…"
  대문자 헤딩) · **#63**(protagonist, obs_37511 "Their Assortment of SCHOOL…" 광고) · **#82**(causality,
  obs_77176 "IN MID-AIR By R." 제목/byline). ⚠ #60은 "THEN"으로 시작해 **time sequence 로 오라벨 가능**
  = 비서사가 축 라벨을 오염시킨 실례. **채점 규약:** 이 4개는 추출 ✗율이 아니라 **코퍼스 위생**을 재므로
  **✗율은 비서사 제외 96개로 계산**, 4개는 F 증거로 별도 보고(with/without 투명). 기준 가리기 아님(측정 대상이 다름).
- **🔎 leakage class 는 "광고"보다 넓음(all-caps 스캔, 문항 83 계기):** 대문자비율>0.45 문장 7개 중 **제목/헤딩
  5**("THE STAG-HUNT."·"LITTLE MARY."·"IN MID-AIR By R…"(제목+byline)·Twain "IN ENGLISH/IN FRENCH"(소제목))
  + **광고 1**("JUST PUBLISHED…"). ⇒ 비서사 leakage = **boilerplate + 광고 + 제목/헤딩/byline** 전부.
  **대문자 ≠ 광고**(대부분 제목·헤딩), 강조 대사도 대문자 가능 → all-caps 는 "비서사 구조행" 신호이나 **단독 불충분**.
- **🔧 탐지법 후보(나중에 다룸 — Phase-1 prep 설계):** 신뢰도 순 파이프라인.
  1. **결정적 = PG 구조표지 + boilerplate blocklist:** `*** START/END OF THE PROJECT GUTENBERG EBOOK ***`
     바깥 제거(현행) + 안쪽 고정문자열 제거(`Produced by`·`Transcriber('s)? Note`·`Online Distributed
     Proofreading`·`End of (the )?Project Gutenberg`·저작권/라이선스). 고정밀.
  2. **구조 휴리스틱(앞/뒤 블록만):** 위치(극단) + **상업 토큰**(Publishers/Booksellers/Stationers·거리주소·
     Price·`$`·Just Published) + **ALL-CAPS 밀도** + **낮은 서사성**(과거시제 동사·인칭대명사·대사 따옴표 적음).
  3. **목차·제목(보조 신호, 단독 불충분):** CONTENTS 블록·제목 헤딩으로 본문 시작 경계 추정 가능하나 **짧은
     단편·아동 리더는 목차 없음**, 뒷광고도 ALL-CAPS 헤딩이라 **제목 형태만으론 광고/이야기 구별 불가** → 2번 신호 필요.
  4. **LLM 폴백(애매 문서만):** 문서 전체를 값싼 모델에 주고 "서사 본문의 [시작,끝] 범위만, 판권·목차·광고·노트
     제외" 반환. obs_37511 류 견고 처리. Phase-1 문서당 1콜 감당 가능.
  5. **샘플 눈검증.**
- **조치:** 위 파이프라인은 **Stage-0 데이터 정제**라 라벨러(Gate C)와 무관, frozen 추출 결과는 그대로. Phase-1 prep
  때 결정적/휴리스틱/LLM 경계·임계 확정(원하면 codex 위임). 라벨링 시 이런 문항은 기계적 라벨 + "non-narrative" 메모.

---

## G. 문장분리가 장면 구분선(\* \* \*)에서 안 쪼갬 — 장면 병합 (**확정**, 스캔)

- **문제(사용자 발견, 문항 90):** 소스의 **장면 전환 구분선**(`* * * * *`·`___`·`----`)에서 `split_sentences`가
  분할을 안 해 **두 장면이 한 item 에 병합.** 예 obs_25744#203(문항 90): "…for my own people." **\* \* \* \* \***
  Returning to his assigned apartment, Thompson changed back…" = 대사(장면1)+구분선+행동(장면2)이 한 문장.
- **🟢 확정(스캔, 예산 0):** 구분선 포함 11문장 중 **9개가 병합**(구분선 양쪽에 본문). **obscure 집중**
  (obs_28922#69, obs_25744#125·#203, obs_74140#23·#70·#164 …).
- **이중 손해 + 기회:** ① \* \* \* \* \* = 저자·편집자가 명시한 **장면경계(gold-ish 신호)인데 병합으로 파묻힘**
  = 강한 경계단서 폐기. ② 한 item에 두 장면 → 5축 라벨 모호. ③ **기회:** 구분선을 **강제 split 지점 + 약한 경계
  라벨**로 활용 가능(자산).
- **조치(Phase-1 prep, Stage-0, Gate C 무관):** `split_sentences`가 구분선에서 **강제 분할** + 구분선 위치를 경계
  힌트로 로깅. 진단 = 문서별 divider 병합 수(obscure 우선). 라벨링 시 병합 item은 지배적/후행 사건 라벨 + "scene-merge" 메모.
- **표본 영향:** 문항 90 = 병합 item(비서사 4개(#56·60·63·82)와 별개 — 본문이나 오분할). 채점 시 별도 표기.

---

## H. space 라벨의 맥락 누출(context-leak) — target-only 소프트 위반 (**확정**, 채점 model-wrong 2건)

- **발견(채점 불일치 판정):** 유일한 model-wrong 2건이 둘 다 space — **문장에 장소 단서가 전혀 없는데 모델이
  특정 위치를 부여**:
  - `fam_london_fire#214` "…the rapidity with which his cheeks and nose were freezing." → 장소단서 0인데 **yukon_trail**.
  - `fam_doyle_speckled#166` "…sent for medical aid from the **village**." → village는 *언급*뿐인데 모델은 **julia_bedroom**(맥락상 방).
- **성격:** raw 상태(carry-forward 전)인데도 위치를 부여 = **모델이 target-only 를 소프트 위반(주변 장면 맥락 누출).**
  회피(none 남발)의 **반대 방향** 오류 = **과잉귀속.** dominant_actor 우려와 정반대 모드.
- **영향/해석:** 맥락상 옳은 위치라 boundary 에 **꼭 해롭진 않음**(오히려 정답일 수). 그러나 **space reset 이
  순수 문장근거가 아니라 맥락기억에 일부 의존**한다는 뜻 → 측정 순수성·재현성에 영향 가능. 2/20(=10%)로 소량.
- **A 와 대비:** A(어휘공백)=무대 있는데 목록에 없어 unspecified(under). H(context-leak)=무대 없는데 위치 부여(over).
  space 축이 양방향으로 흔들림.
- **다음 버전:** Stage 2 space 프롬프트에 "이 문장 자체에 무대 근거가 없으면 반드시 unspecified(주변 문장으로
  추론 금지)" 강화 문구. 진단 = 장소단서 없는 문장의 non-unspecified 율. Gate C 대상(측정도구 변경).

---

## C. 라벨링 규약 명료화 (이 표본 내 일관 적용 — frozen 프롬프트 의도의 해석, 새 기준 아님)

이 표본을 블라인드로 매기며 target-only 를 일관 적용하려고 프롬프트 의도를 아래로 정리(합격선 불변):

**space (question = "목록의 장소 중 하나에서 벌어지나?")**
- **예** = 문장이 그린 무대가 **목록의 장소 중 하나**(다리·지휘의자→함교). 장소 단어 없어도 무대 그리면 선택.
- **아니오** = 지시어만("여기") + 무대 내용 없음 / 무대가 **목록 밖**(비행기) / 언급만 된 장소(도움 요청한 마을 = mention≠stage).

**intentionality (question = "목록의 목표 중 하나를 향하나?")**
- **예** = 행동이 목표를 **진행/실행**(목적) — 능동적 결심("기꺼이 죽겠다!"=die_gloriously), 명령·요청·계획·수사 질문 포함.
- **아니오(none)** = 목표가 **화제**로만 언급 / 행동이 목표를 안 밈. **반복된 none 패턴**: 감정표현·감탄, 멍한 상태/가만히
  있기, 지각(보다·바라보다), 배경·주변 묘사, 속생각·회상, 순수 신원 설명(명사구 조각).
- **한 줄 판별:** "이 문장의 행동이 그 목표를 향해 **한 걸음 나아갔나?**" 나아감=목적(예) / 입에만 올림=화제(아니오).

**protagonist (question = "행위의 주체·인칭이 드러나 있나?")**
- 라벨 필드 = **perspective(문법적 인칭)**, **정체(누구)가 아님.** "he/she/they"면 **누군지 몰라도** 3인칭 → **예.**
  none = 사람이 아예 없음(순수 사물·풍경). 가정·소망("~했더라면") 형태여도 인칭(he)은 드러남 → 예.
- ⚠️ 질문 문구 "누가 하는지·인칭"이 **정체 vs 인칭**을 섞어 혼동 유발(문항 66). 기준 = **인칭**. "he가 누구냐"는
  별도 필드 `characters_present` 문제이고, **거기가 대명사 target-only 미해소 지점**(원 계획 "He sat up"→none
  dominant_actor). ⇒ perspective=대명사에 강건 / characters_present=대명사에 취약. 채점 시 protagonist 문항은
  **인칭 기준으로 일괄 확인**(문항 바꾸지 않음 — 기준변경 아니라 의도 명료화).

> ⚠️ **일관성 주의:** 위 "화제 vs 목적" 명료화는 표본 중반(문항 ~38-39)에서 확립됨. 이전 intentionality 문항(21-37)은
> 감정/상태/지각/배경/속생각 사유의 none 이라 이 명료화와 **무관**(답 유효). 단 문항 32("I am making a collection of
> birds")는 **답(아니오) 유지, 이유 정정**(❌"대사라서" → ✅"collect_birds 는 화제 + 발화의 실제 목적 target-only 불명").
> 채점 시 대사형 intentionality 문항은 이 잣대로 **일괄 재확인** 권장.

---

## D. 방법론 원칙 (무엇을 기록할 가치가 있나)

- **체계적 편향**(여러 문장을 같은 방향으로 밈 → base rate·경계밀도 변화) = **중요**. 예: pervasive-goal(②), 대사 과소추출(① 가설).
- **일회성 국소 애매**(한 문장, 사건 중간; 다축 t\*=3 투표+carry-forward+Pk/WD 허용창으로 씻겨나감) = **저임팩트**.
  예: 무대 밖 저격수 사격을 주인공 지각으로 서술한 한 문장(문항 37) → 애매로 찍고 넘어감, 큰 발견 아님.
- 판별 한 줄: **"이 애매가 나 혼자냐, 같은 패턴이 수십 문장에서 반복되냐?"** 후자만 값나감.
- **라벨 ≠ 경계.** 경계 = 라벨(상태)이 **≥3축에서 동시에 바뀌는 전이**(t\*=3). 라벨은 상태고, 그 변화 지점이 경계 신호.

---

## 상태 요약 (②진단 2026-07-23 반영 — `outputs/experiments/2026-07-23_findings_diagnostics/REPORT.md`)
| 이슈 | 상태 | ②진단 결과(스케일) | 조치 |
|---|---|---|---|
| A. space 어휘공백(비행기) | 실재·**genre 국소** | 탈것/공중 목록누락 `obs_77176`=26 압도(항공 pulp), 타 문서 대부분 regex FP | ④ 포함(저비용·정합, '흔해서'가 아니라 '해당서사 치명적'이라서) |
| B-② pervasive goal | **확정·보편** | span≥0.6 상시목표 **10/12 문서**; leave-one-out 시 intentionality reset 대폭↓(예 owlcreek 17.5→8.5%) = reset 상당분 생성 | ④ 1순위 포함 ✅(Stage1 bounded commitment 제한) |
| B-① 대사 화제/목적 | **🔴 기각(스케일 미확증)** | 수행적 대사 P(goal≠none)=41.6% > 논의 24.8%, **gap +16.8%**(기준 ≤10% 위반) → 모델 화행 무시 안 함 | ①은 구성타당도 결함으로 기각. '대사→목표' 예시 추가는 *무해한 명료화*라 ④ **선택적 경보강**만(load-bearing 아님) |
| B-③ carry 방향 | **보류(안 고침)** | (진단 안 함) | forward 유지, 민감도만 |
| E. time opens-with 비대칭 | **🟢 확정(캐시 실증)** | jump 문중 48%(29 중 14) vs sequence 문중 24% | ④ 위치조건 대칭화 포함 ✅ |
| F. 앞뒤 잡물 leakage | boilerplate=확실 / 광고=정황 | (Stage-0, ② 대상 아님) | 탐지법 후보 기록, Phase-1 prep 때 확정(Gate C 무관) |
| G. 장면 구분선 병합 | **🟢 확정(스캔 9/11)** | (Stage-0, ② 대상 아님) | Phase-1 prep split_sentences 구분선 강제분할+경계힌트 |
| H. space context-leak | **🟢 확정 + 규모 중간** | no-cue 문장 non-unspecified율 전체 8.8%(최고 owlcreek 31.5%); 다수는 맥락상 정답이나 순수성 문제 실재 | ④ 포함 ✅(space "근거없으면 unspecified" 강화) |
| C. 라벨링 규약 | 표본 내 적용 | — | 채점 시 대사문항 일괄 재확인 |

**②진단 총평:** A·H·B②·E = ④ 포함 확정. **B①만 스케일에서 기각**(예시 보강은 선택). repcorpus gold 경계=0 →
Score 영향은 여기서 측정 불가, ④ dev-3(gold 有) Gate C 로 이월. 실제 재등록 결정은 ④ 통과 후 → `context/06-decision-log.md` 승격.
