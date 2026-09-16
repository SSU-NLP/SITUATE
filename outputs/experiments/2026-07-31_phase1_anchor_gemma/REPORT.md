# Phase-1 학습 앵커 코퍼스 재생성 — Gemma-4-26B-A4B (D-B53)

**산출 = `outputs/eacl_phase0/phase1_gemma4a4b_v31/reset_matrix.jsonl`
— 10,584편 · 2,773,435 전이 · 717MB.**
용도 = **M4~M7 앵커 공급 전용.** 모든 레코드 `boundary = 0` 이며 성능 지표를 내지 않는다(평가셋 아님).
기존 Qwen 앵커(D-B15, `phase1_all_v31/`)는 **삭제하지 않고 그대로 둔다** — 두 코퍼스를 나란히 놓고
학습 비교를 할 수 있어야 한다.

---

## 1. 실험 setup

### 목적
D-B51 에서 dev-3 Score 가 기준선을 넘은 **Gemma-4-26B-A4B** 로 Phase-1 앵커를 새로 만든다.
앵커 교체 자체의 채택 판정이 아니라 **교체안을 학습에 걸어볼 수 있게 재료를 만드는 단계**다.

### 라벨러 선택 근거 (D-B51, dev-3 k=3 · deployable Score · t\*=3)

🔴 **채점 축 집합이 두 가지로 갈린다 — D-B54 참조.** 밤샘 중 쓴 `s5_crossmodel.py:17` 은
protagonist 성분을 `protagonist`(charset OR perspective)로 세는 **유일한 이탈 스크립트**이고,
D-B10 동결본·`build_anchor_dataset.py:31`·나머지 7개 스크립트는 `protagonist_present` 를 쓴다.
아래 두 표는 **같은 산출물을 두 열 집합으로 잰 것**이며 판정이 갈린다.

**(A) D-B10 동결 5축 (`protagonist_present`) — 프로젝트 표준**
| 모델 | pieman | tunnel | monkey | 평균 | Δ vs 기준선 |
|---|---|---|---|---|---|
| **Gemma-4-26B-A4B** | 0.544 | 0.489 | 0.413 | **0.4819** | **+0.0447 · 3/3 무하락 → 기준 통과** |
| Qwen3.6-35B-A3B (기준선) | 0.476 | 0.471 | 0.364 | 0.4372 | — |
| gpt-oss-120b | 0.507 | 0.440 | 0.353 | 0.4334 | −0.0038 |
| Gemma-4-31B dense | 0.449 | 0.491 | 0.289 | 0.4098 | −0.0274 |

**(B) `s5_crossmodel.py` 열 집합 (`protagonist`) — 밤샘 중 실제로 본 표**
| 모델 | pieman | tunnel | monkey | 평균 | Δ vs 기준선 |
|---|---|---|---|---|---|
| **Gemma-4-26B-A4B** | 0.468 ±0.061 | 0.338 ±0.032 | 0.364 ±0.007 | **0.390** | +0.008 → 기준 미달 |
| Qwen3.6-35B-A3B (기준선) | 0.424 ±0.018 | 0.410 ±0.012 | 0.314 ±0.032 | 0.382 | — |
| gpt-oss-120b | 0.515 ±0.056 | 0.327 ±0.023 | 0.296 ±0.036 | 0.380 | −0.002 |
| Gemma-4-31B dense | 0.456 ±0.006 | 0.379 ±0.000 | 0.294 ±0.004 | 0.376 | −0.006 |

**모델 선택 순위는 두 표에서 같다**(26B-A4B > 기준선 > gpt-oss > 31B) — 즉 *이 코퍼스를 어느
라벨러로 만들지*는 어느 열로 재도 같은 답이었다. **갈리는 것은 채택 판정뿐**이다.

🔴 **나는 (B)를 보고 라벨러를 골랐고, (A)는 이 코퍼스를 다 만든 뒤에 발견했다.**
(A)가 채택 기준을 통과한다는 사실이 "내가 만든 것을 정당화하는 방향"이라는 점을 명시해 둔다.
**교체 채택은 이 REPORT 로 선언하지 않는다** — D-B54 를 사용자가 판정한 뒤 별도로 한다.

### 데이터
| 항목 | 값 |
|---|---|
| 출처 | Project Gutenberg 영어 단편, `/shared/hiontop/data/shards_all/` (1,000 샤드) |
| 투입 | **10,645편 · 2,786,352 전이** (D-B15 와 동일한 샤드 집합) |
| 격리 | QC 대표 12편 · dev-3 · Tier1 held-out 제외 (샤드 구성 단계에서 이미 반영, D-B15 승계) |
| 최종 | **10,584편 · 2,773,435 전이** (손실 61편 = 0.57%) |

### 설정 (전 구간 단일 조건, 예외 문서 없음)
- 프롬프트 **v3.1 동결**(D-B14) — `--prompt-version v3.1`, `prompt_sha256` 무변경
- 라벨러 **Gemma-4-26B-A4B (`/shared/models/gemma-4-26B-A4B-it`, `served=gemma4-a4b`)**,
  `--no-reasoning` → `chat_template_kwargs={"enable_thinking": false}`
- 🔴 **`--reasoning-parser` 를 쓰지 않는다.** D-B51 에서 `gemma4` 파서 + `enable_thinking:false`
  조합이 content 를 개행 8,000개로 만들어 9 job 전멸시킨 전례가 있다.
- temp 0 · batch 20 · `--stage1-max-tokens 8000` · `--timeout 900` · seed 20260711+axis_index
- vLLM 0.25.1 · **TP=1 · `--enable-prefix-caching` · `--load-format runai_streamer`**
  · `--max-model-len 32768` · `--gpu-memory-utilization 0.90` · `--max-num-seqs 512`
- **k=1 단일 draw** (D-B15 와 동일 한계)

### 인프라
VESSL **A100-SXM4-80GB × 20 Job** (1 Job = 1 GPU = vLLM 복제 1개, Job 간 통신 0).
`b53-anchor-gemma-s0..19`, 스크립트 `/shared/hiontop/job_shard_v8_gemma.sh`.
**KST 2026-07-31 04:09:09 착수 → 07:32:02 완료 (3시간 22분 53초).** 20/20 성공.

---

## 2. v7 → v8 변경 4가지 (사용자 지시 = "서버 설정을 저번보다 더 잘")

D-B15 실적에서 **가장 큰 낭비는 GPU 가 아니라 회수였다** — 라벨링 3h20m < **회수 4h10m**.
따라서 개선의 핵심을 거기에 뒀다.

| # | 변경 | 근거 | 실측 효과 |
|---|---|---|---|
| ① | **`story_lists.tar` 를 풀지 않는다.** 각 Job 이 자기 몫 Stage-1 을 직접 예열 | 그 tar 는 **Qwen 산출물**이고 `llm_s5_states.py:699` 는 캐시가 있으면 **모델을 확인하지 않고 재사용**한다 → 풀었으면 Stage-1 이 Qwen 것으로 오염 | 예열 평균 **456초** 추가. 오염 0 |
| ② | **샤드 단위 배분** (v7 은 (샤드×축) round-robin) | v7 방식은 한 문서의 5축이 서로 다른 머신에 흩어져 Stage-1 캐시 경합 | 꼬리비 **1.63× → 1.05×** |
| ③ | **결과를 tar 1개로 회수** (v7 은 축파일 개별 복사) | D-B15 §D-52: geesefs 에서 축파일 53,184개 읽기 = 3h11m~4h10m | Job 내 tar 생성+복사 **2~3초**, 로컬 전개 수 분 |
| ④ | 모델·양자화·파서 전부 env 화 (v7 은 `qwen36-35b`/fp8/`qwen3` 하드코딩) | 라벨러 교체 시 조용히 틀린 모델명 전송 | `prewarm_stage1_v2.py`·`resilient_shard_job_v2.sh` 신규 |

### 시간 내역 (20 Job)
| 구간 | 평균 | 최소 | 최대 |
|---|---|---|---|
| 서버 ready | 8분 36초 | 7분 55초 | 9분 25초 |
| Stage-1 예열 | 7분 36초 | 6분 28초 | 10분 34초 |
| Stage-2 라벨링 | **2시간 58분 45초** | 2시간 54분 44초 | 3시간 3분 29초 |
| 회수(tar) | 수 초 | | |
| **전체 (착수→마지막 Job)** | **3시간 22분 53초** | | |

D-B15 = 7시간 34분. **3시간 이상 단축**했고 단축분은 거의 전부 회수 구간에서 나왔다.
비용은 D-B15($95, 7h34m) 에 시간 비례로 환산하면 **약 $42** (별도 청구서 확인 안 함 — 추정치).

---

## 3. 결과 — 라벨 분포 (D-B15 Qwen 앵커와 대조)

상세 = `LABEL_DIST.md` / `LABEL_DIST.json`. **동결 5축 = D-B10 채택본**(protagonist 성분 =
`protagonist_present`; `build_anchor_dataset.py:31` "D-B10 동결 5축" 이 학습 데이터를 이 열로 만든다).
🔴 이 코퍼스의 `boundary` 는 전부 0이다. 아래는 **정확도가 아니라 방법이 찍은 값의 분포**다.

| 축 | 신: Gemma-4-26B-A4B | 구: Qwen3.6-35B-A3B | Δ |
|---|---|---|---|
| protagonist_present | 0.2632 | 0.3049 | −0.0417 |
| space | 0.0917 | 0.1047 | −0.0130 |
| time | 0.0359 | 0.0213 | **+0.0145** |
| causality | 0.7033 | 0.6950 | +0.0084 |
| intentionality | 0.1101 | 0.1140 | −0.0039 |
| **경계율 (t\*=3)** | **0.0600** | 0.0649 | −0.0050 |

count 분포 (0/1/2/3/4/5): 신 0.184 / 0.494 / 0.261 / 0.053 / 0.006 / 0.000,
구 0.176 / 0.480 / 0.279 / 0.058 / 0.007 / 0.000. **D-B15 가 기록한 값과 같은 자리**
(0.176 / 0.480 / 0.279 / 0.058 / 0.007 / 0.000, 경계율 0.0649) — 열 집합이 일치한다는 확인이다.

### 해석
- 다섯 축이 **±0.042 이내**, 경계율 차 −0.005. 라벨러를 통째로 바꿨는데도 집계 분포는 거의 같다.
- **두 축이 반대로 움직였다:** `protagonist_present` −0.042(상대 −14%), `time` +0.0145(상대 **+68%**).
  경계율에는 −0.005 밖에 안 남았지만 **축 단위로는 서로 상쇄된 것**이며, 집계 일치가 전이 단위
  일치를 뜻하지 않는다는 D-B16 §2 의 경고를 그대로 보여준다. **전이 단위 κ 는 미측정.**
- `causality 0.703` 은 새 정보가 아니다 — Qwen 0.695, dev 포화 0.674 와 같은 자리다.
  **약축이라는 성질은 라벨러를 바꿔도 그대로**이며(D-B13), 그래도 축은 제거하지 않는다.

---

## 4. 🔴 실패·손실 (숨기지 않는 기록)

### 손실 10,645 → 10,584 (−61편, 0.57%)
| 사유 | 편수 |
|---|---|
| 축파일 전무 (대부분 protagonist) | 55 |
| 축 불완전 (item 수 < 문장 수) | 6 |

D-B15 의 −102편(0.95%)보다 낫지만 **원인은 새로 밝혀졌고 고쳐지지 않았다.**

### 근본 원인 — resilient 재시도 루프가 못 잡는 실패 경로
`resilient_shard_job_v2.sh:23` 은 로그에서 **`RuntimeError: [doc]`** 만 범인으로 검출한다.
이번 실패 10건은 전부 **`json.decoder.JSONDecodeError`** 였다 —
Stage-2 응답이 32,636자에서 잘려(폭주 후 max-token 도달) `json.loads` 가 그 자리에서 죽는다.
`RuntimeError` 가 아니므로 검출기가 폴백으로 넘어가는데, 폴백은
"`story_lists.json` 이 없는 첫 문서"를 범인으로 찍는다(:26-35).
**그런데 이번 런은 Stage-1 을 전부 예열해 뒀으므로 그런 문서가 하나도 없다** →
범인 특정 실패 → `[FAIL] ... 범인 특정 불가` 로 (샤드×축) 전체가 중단.

즉 **변경 ①(Stage-1 예열)이 변경과 무관한 폴백 검출기를 무력화**시켰다. 상호작용을 예견하지 못했다.

| 축 | JSONDecodeError 로 중단된 (샤드×축) |
|---|---|
| protagonist | 5 |
| space | 2 |
| causality / intentionality / time | 각 1 |
| **계** | **10** (전체 5,000 중 0.2%) |
| 참고: 정상 검출(`RuntimeError`) 경로로 복구된 건 | 1 |

**protagonist 편중은 D-B15 와 같다**(그때는 폭주 47편 중 40편이 protagonist).
`cast`/`characters_present` 배열에 길이 제한이 없어 temp 0 탐욕 디코딩이 같은 값을 무한 반복하는
D-B15 §D-42 의 폭주와 동일 현상이며, 라벨러를 Qwen→Gemma 로 바꿔도 **재현된다**.
⚠️ D-B15 결론대로 **`maxItems` 로 막으면 안 된다** — 폭주를 상한에서 자를 뿐이라 고유값 1~5개인
쓰레기가 조용히 통과한다.

**미수정으로 남긴다.** 고치려면 실행 중 스크립트를 바꿔야 했고(금지), 손실이 0.57% 로
D-B15 보다 작아 재실행 이득이 비용을 넘지 않는다. **다음 런 전에 고칠 것**:
검출기를 `RuntimeError` 뿐 아니라 **로그 마지막에 등장한 `[doc_id]` 태그**로 잡게 바꾼다.

### 그 밖
- Stage-1 예열 실패 **2편**(20 Job 중 2 Job 에서 각 1편). 위 61편에 포함.
- Stage-2 정상 skip **2편** (`_skipped_5.log`, `_skipped_8.log`). 위 61편에 포함.
- **미완료 후보 3건.** `b49-deepseek31-awq-k3`(51분 경과 후) · `b50-minimax-on-v2` ·
  `b50-minimax-off-v2`(각 8분 경과 후)가 **`terminated` 상태로 산출 없이 끝났다**.
  앵커 Job 20개 제출(19:08 UTC) 직전인 18:32~18:39 UTC 에 종료됐고 GPU 16장이 그때 풀렸다.
  타이밍은 앵커 런에 자리를 내주려 정리된 것과 일치하지만 **저널에 근거 기록이 없어
  누가/무엇이 종료했는지 확정할 수 없다.** 무인 런 규율(모든 결정 즉시 기록) 위반이다.
  ⇒ MiniMax ON/OFF(D-B50)·DeepSeek-V3.1-AWQ(D-B49)는 **미측정 상태로 남았다.**

---

## 5. 판정

**앵커 코퍼스 = 생성 완료. 라벨러 교체 채택 = 보류(사용자 판정 대기).**
10,584편 앵커를 M4~M7 학습에 *쓸 수 있는* 상태로 만들었다. 단일 조건이며 예외 문서는 없다.

보류 사유는 §1 의 열 집합 분기다. **(A) D-B10 동결 5축으로는 +0.045 · 3/3 무하락 = 기준 통과,
(B) `s5_crossmodel.py` 열로는 +0.008 = 기준 미달.** 어느 쪽이 유효한지는 코드 정합성 문제이지
성능 문제가 아니며, **결과를 본 뒤에 내가 고를 사안이 아니다**(D-M5 재연 방지).
⇒ D-B54 에 사실만 적었고 판정은 사용자에게 넘긴다.

기존 Qwen 앵커(`phase1_all_v31/`)는 **삭제하지 않고 유지**한다. 어느 판정이 나오든 다음 단계는
두 코퍼스로 각각 학습한 M5 를 같은 dev-3 로 비교하는 것이며, 그 비교의 채택 기준은
**실행 전에** 먼저 적는다.

---

## 6. 한계 / 검증 미해결 (제거 금지)

- **k=1 단일 draw.** vLLM 은 단일 요청에서도 결정적이지 않다. 반복 변동이 무작위인지 계통적인지
  이 코퍼스에서 미측정 (D-B15 와 동일 한계).
- **경계 정답 없음.** `boundary` 전부 0. Score 는 gold 가 있는 dev-3 에서만 낸다.
- **집계 일치 ≠ 전이 일치.** 축 발화율이 ±0.017 이내라는 것은 분포의 일치이며, 같은 전이에서
  같은 판단을 했다는 뜻이 아니다. **전이 단위 일치율(κ)은 측정하지 않았다** — `time` 이 상대
  +68% 인 것이 그 필요를 보여준다.
- **라벨러 선택의 dev-3 성분.** Gemma-4-26B-A4B 는 dev-3 3편에서 골랐다. dev-3 는 연습장이고
  선택압이 누적돼 있다. 이 코퍼스는 그 선택 위에 서 있다.
- **도메인 = 서사 픽션 단일**, 작가 편중(D-B15 승계, 메타데이터 보존돼 사후 층화 가능).
- **손실 61편의 문서 특성 미분석.** 폭주하는 문서가 특정 길이·등장인물 수에 몰리는지 안 봤다.
  D-B15 는 길이 무관·등장인물 무관으로 결론냈으나 라벨러가 다르므로 재확인 안 된 상태다.
- **인프라 재현성.** geesefs 대역폭 의존. 다른 스토리지 구성에서 같은 시간을 기대할 수 없다.

---

## 7. 재현

```bash
# 1) 라벨링 (VESSL A100×20)
#    /shared/hiontop/job_shard_v8_gemma.sh, env:
#    SHARD_ID=0..19 NUM_SHARDS=20 OUT_DIR=/shared/hiontop/out/phase1_gemma4a4b_v31
#    MODEL_PATH=/shared/models/gemma-4-26B-A4B-it SERVED=gemma4-a4b
#    CONC=250 PREWARM_WORKERS=64 MAXLEN=32768 GPUUTIL=0.90 READY_WAIT=720

# 2) 회수 + 무결성 검사 + 조립용 master 생성
tar -xf part_{0..19}.tar -C outputs/eacl_phase0/phase1_gemma4a4b_v31
python scripts/eacl_phase1/assemble_anchor_gemma.py \
  --out-dir outputs/eacl_phase0/phase1_gemma4a4b_v31

# 3) 조립 (전부 캐시 히트 — base-url 을 죽은 포트로 줘서 실호출이 나면 즉시 죽게 한다)
python scripts/eacl_phase0/llm_s5_states.py \
  outputs/eacl_phase0/phase1_gemma4a4b_v31/_assembly_master.jsonl \
  outputs/eacl_phase0/phase1_gemma4a4b_v31 \
  --model gemma4-a4b --engine vllm --no-reasoning \
  --base-url http://127.0.0.1:1/v1 --prompt-version v3.1 --stage1-max-tokens 8000

# 4) 분포 대조
python scripts/eacl_phase1/anchor_label_dist.py \
  --new outputs/eacl_phase0/phase1_gemma4a4b_v31/reset_matrix.jsonl \
  --old outputs/eacl_phase0/phase1_all_v31/reset_matrix.jsonl \
  --out outputs/experiments/2026-07-31_phase1_anchor_gemma/LABEL_DIST.md
```
