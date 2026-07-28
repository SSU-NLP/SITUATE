# HANDOFF_07 — SHARE 원 대본 씬 헤딩으로 place/time gold 구성

**지정일**: 2026-07-14 · **상태**: 파일럿 성공(1편), 전체 파이프라인 구축 중 · **용도**: SHARE M3(S5) probe의 place/time null이 "가설 반증"인지 "측정 한계"인지 분리

## 0. 문제 정의

- **배경.** SHARE(대사-only 영화대화)에서 frozen M3(S5)를 돌린 결과 place/time 축이 near-dead(place 경계율 0.10·time 0.02, AUC ≤0.53). 그러나 **SHARE엔 장소·시간 ground-truth가 없어**(메타데이터는 사건·관계 요약뿐, 원 대본 씬 헤딩은 파싱 때 소실), place/time null이 **"안 바뀜"인지 "바뀌는데 대사에 안 보임"인지 구분 불가** = 가설에 무정보 (ledger D-M7 재해석).
- **문제.** 원 영화 대본의 씬 슬러그라인(`INT./EXT. 장소 - 시간`)을 외부 확보 → SHARE 세션↔씬 정렬 → **세션 경계별 place/time 진짜 변경 여부 gold** 구성. 그 gold로 M3 추출기의 place/time reset을 대조.
- **목표.** 세션 경계를 두 종류로 분리: (a) **진짜 place/time 변경** 경계 — 여기서 M3가 잡나?(측정 recall) (b) **변경 없음** 경계 — 여기서 M3가 조용한가?(정답). 이로써 place/time null의 성격(측정 실패 vs 가설 성립) 판정.
- **성공기준.** probe 영화 다수에서 대본 확보·정렬 성공 → place/time gold 산출 → "진짜 변경 경계에서의 M3 place/time reset recall"과 "무변경 경계에서의 침묵률"을 수치로 보고. (recall 높으면 축 작동/데이터 부족, recall 낮으면 대사-only 측정 한계 확증.)

## 1. 데이터

- **SHARE.** `benchmarks/SHARE/datasets/{valid,test}.json` (HF `eunwoneunwon/SHARE`). 최상위 dict[화자쌍]→{movie, dialogue:[세션]}. 세션 = {session(로컬 순번 1,2,…), dialogues:[{speaker,text}], persona/temporary-event(스트립됨)}. 대사 텍스트 대체로 clean(영화별 OCR 노이즈 편차 있음 — 예: Donnie Brasco 노이즈, American Gangster clean).
- **원 대본.** IMSDB(`imsdb.com/scripts/<Title>.html`) `<pre>` 안 대본. 슬러그라인 = `(INT.|EXT.|INT/EXT.) 장소 - 시간[- 부가]`, 대문자, scene number 동반 가능.
- **probe 대상.** `outputs/eacl_phase0/share_valid_probe30_master.jsonl` 의 30 영화(25 docs가 M3 5축 완료, reset_matrix 有). 영화명 = doc_id의 `__` 앞부분(slug). 실제 movie 필드는 valid.json에서 조회.

## 2. 방법·설정 (재현)

### ⭐ 2.0 외부에서 place/time 정보를 어떻게 가져왔나 (핵심 절차 — 이 실험의 심장)

**문제.** SHARE(대사-only)에는 장소·시간이 없다. 그래서 **"place/time이 세션 경계에서 실제로 바뀌는가"의 정답(gold)을 SHARE 밖에서 가져와야** 한다. 원천 = **SHARE가 파싱하기 전의 원본 영화 대본**. 영화 시나리오는 각 씬 첫머리에 **슬러그라인(slug line)** `INT./EXT. 장소 - 시간`을 명시하므로, 이것이 곧 각본가가 저작한 장소·시간 ground-truth다. SHARE가 대화로 변환하며 이 슬러그라인을 버렸을 뿐, 원 대본엔 그대로 있다.

**데이터 흐름 (SHARE 대사 → 원 대본 씬 → place/time):**

```
SHARE 세션(대사만)  ──정렬──▶  원 대본의 해당 씬  ──슬러그라인──▶  (장소, 시간)
```

**5단계 절차:**

1. **원 대본 확보 (외부 소스).** 각 영화의 `movie` 필드 → IMSDB(`imsdb.com/scripts/<Title>.html`) `<pre>` 대본을 다운로드. 제목 변형 후보 여러 개 시도(원형 / "The"를 뒤로 이동 ", The" / 하이픈화 등). 캐시 = `data/eacl_phase0/share_scripts/<movie_slug>.txt`. 스크립트 `fetch_share_scripts.py`. **오답 대본 방지 sanity check: 그 영화의 SHARE 대사 중 ≥15%가 실제로 대본 텍스트에 존재해야 채택**(match 비율 `_coverage.json`에 기록). 결과 18/30 확보.
2. **슬러그라인 → (장소, 시간) 파싱.** 대본을 라인 분리 → `INT.|EXT.|INT/EXT.`로 시작하는 라인(길이<75) 추출 → scene number 제거 → `-`로 분할 → **시간 토큰**(day/night/morning/later/continuous/dawn/dusk… 사전 매칭)과 **장소 문자열** 분리·정규화. 씬 = 인접 두 슬러그라인 사이 구간. `build_share_scene_gold.py:load_scenes/parse_slug`.
3. **SHARE 세션 ↔ 대본 씬 정렬.** 세션의 각 대사를 정규화(소문자·공백정리)해 앞 30자를 대본 라인에서 부분문자열 검색 → 매칭된 라인 **위쪽 최근접 슬러그라인 = 그 대사의 씬.** 세션의 씬 = 대사별 매칭 씬들의 **다수결.** (SHARE 대사가 대본에 거의 verbatim 존재해서 가능. OCR 노이즈 영화는 매칭률↓.)
4. **place/time gold 도출.** 세션 k→k+1 경계에서 정렬된 두 씬을 비교: 장소 문자열이 다르면 `gold_place=1`, 시간 토큰이 다르면(둘 다 비어있지 않을 때) `gold_time=1`. 이것이 **"경계에서 장소/시간이 실제 바뀌었나"의 외부 정답.**
5. **M3 대조.** M3의 코드판정 place/time reset(`reset_matrix.jsonl`)을 위 gold와 비교 → 진짜변경 경계 **recall**, 무변경 경계 **침묵률(specificity)**.

**실제 예시 (American Gangster, FRANK↔NATE) — 절차가 산출한 것:**

| SHARE 세션 | 정렬된 대본 씬(슬러그라인) | 파싱된 (장소, 시간) |
|---|---|---|
| 1 | `EXT. CONEY ISLAND - BEACH - DAY` | (coney island beach, day) |
| 2 | `EXT. CONEY ISLAND - BEACH - DAY` | (coney island beach, day) |
| 3 | `EXT. CONEY ISLAND - BEACH - DAY` | (coney island beach, day) |
| 4 | `INT. JIMMY RACINE'S APARTMENT - NIGHT` | (jimmy racines apartment, night) |

→ 경계 1→2, 2→3: 장소·시간 동일 → gold_place=0. 경계 3→4: 장소·시간 변경 → gold_place=1, gold_time=1. **이 gold가 SHARE 대사엔 전혀 없던 정보이며, 순전히 원 대본 슬러그라인에서 왔다.**

### 2.1 기타 설정
- **대조 지표.** 진짜변경 경계 recall(=놓치지 않았나), 무변경 경계 침묵률(=오경보 안 했나). 축별 진단으로 보고(Score 아님 — gold가 축별 이진).
- **M3 설정** = D-M7 frozen 그대로. dev(valid)만, test 봉인.

## 3. 시도 기록

### 시도 1 — 파일럿 (American Gangster, FRANK↔NATE) — ✅ 정렬·gold 구성 성공

- 대본 fetch: `imsdb.com/scripts/American-Gangster.html` 200 OK(322KB). 슬러그라인 파싱 정상(`INT. DINER - HARLEM - DAY` 등).
- 정렬: 세션별 턴 매칭 8/9·2/2·5/5·4/5 (양호).
- **씬 gold:** 세션1·2·3 = `EXT. CONEY ISLAND BEACH - DAY`(동일), 세션4 = `INT. JIMMY RACINE'S APARTMENT - NIGHT`.
- **핵심 발견:** **세션 경계 ≠ 씬 전환.** 3개 경계 중 place/time 진짜 변경은 3→4 **1개뿐**(1→2, 2→3은 무변경). ⇒ "세션경계=장소변경" 가정 오류. place null의 상당 부분이 **진짜 무변경**일 수 있음(추출기 정답). 진짜변경 경계(3→4)에서 M3가 잡았는지는 전체 파이프라인에서 대조 예정.

### 시도 2 — 전체 probe 영화 파이프라인 — ✅ **측정 실패 확증 (가설은 참, 대사-only가 못 봄)**

- **커버리지.** IMSDB fetch `fetch_share_scripts.py` → **18/30 영화 대본 확보**(slug 有 16편). 실패=제목변형(Dark Knight Rises→"...,-The" 등)·junk 제목(never_pdf_crdown 등). gold 구성 `build_share_scene_gold.py`.
- **평가 대상.** 대본 有 ∩ reset_matrix 有 = **12 conv, 92 세션경계**(정렬실패 6경계 제외). 세션↔씬 = 대사 부분문자열 다수결 정렬, 씬 = 최근접 상단 슬러그라인, place=location 문자열 상이·time=time-of-token 상이.
- **진짜 변경 base rate (대본 gold):** **place 83%(76/92)** · time 42%(39/92). → "세션경계=장소변경"이 대부분 참(파일럿 FRANK/NATE의 1/3은 예외적 케이스였음).
- **M3 reset 대조 (핵심):**

  | 축 | 진짜변경 경계 recall | 무변경 침묵(specificity) |
  |---|---|---|
  | **place** | **3/76 = 4%** | 15/16 = 94% |
  | **time** | **0/39 = 0%** | 53/53 = 100% |

- **판정: 측정 실패 확증.** place/time은 세션경계에서 **실제로 바뀌는데(83%/42%) M3(대사-only)는 4%/0%만 잡는다.** ⇒ SHARE place/time null = **"안 바뀜"(가설 성립·추출기 정답)이 아니라 "바뀌는데 대사에 없어서 못 봄"(측정 실패).** 앞선 "장소 명시 6.1%"와 정합. specificity 高(94%/100%)는 M3가 거의 안 켜지는 신호부재의 부산물이지 정답 아님.
- **함의.** ① 가설(사건경계=place/time 리셋)은 SHARE ground-truth에서 참. ② 대사-only 모달리티는 place/time 축을 원리적으로 복원 못 함 = **모달리티 한계**(서술문 채널 필요). ③ threshold 적응화·변화추론 추출로도 없는 신호는 못 만듦(동일 결론). ④ SHARE는 place/time **가설 검정용으로 부적합**(대사-only), 단 **"측정 모달리티 한계"의 정량 근거**로는 강력.
- **caveat.** location 문자열 표면변이가 place-change를 다소 부풀릴 수 있으나(같은 장소 다른 표기), recall 4%가 워낙 낮아 결론 불변. time 토큰(DAY/NIGHT)은 견고한데 recall 0%. 12 conv/92 경계 소표본이나 방향 결정적.

## 4. 한계 / 미해결

- IMSDB 미보유·제목 불일치 영화는 제외(커버리지 < 30 예상). 대본 판본이 SHARE 소스와 다를 수 있어 정렬 불완전 가능.
- 슬러그라인 location 문자열 표기 편차(같은 장소 다른 표기) → place 변경 판정에 표면 노이즈. time_of_day는 DAY/NIGHT 등 소수 범주라 견고.
- SHARE 세션이 한 씬을 여러 세션으로 쪼갠 경우(파일럿에서 확인) 다수결 정렬로 흡수하나, 세션이 여러 씬을 걸치면 모호.
- dev(valid)만. test 봉인 유지.
