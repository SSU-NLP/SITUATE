# HANDOFF 08 — 라벨러 교체 시도 + SEM 스위칭 종결 (2026-07-31)

**ledger 원본 = `EACL/phase0-decisions-and-runs.md` D-B50 ~ D-B56.** 아래는 좌표 색인.

---

## 1. 결론 3줄

1. **M3 dev-3 점수는 M5 성능을 예측하지 못한다** (D-B55: M3 +0.045 → M5 **−0.035**, 부호 역전).
   ⇒ 라벨러를 M3 Score 로 고르지 않는다.
2. **라벨러 교체·앵커 교체 = 전부 기각.** 동결 유지 = Qwen3.6-35B-A3B no-thinking · v3.1.
3. **SEM 스위칭 층 재개 게이트 전 항목 불통과** (D-B56) → D-B28 🔒 종결 유지.

---

## 2. 진행 중 (밤새 도는 것)

| 무엇 | 좌표 | 상태 |
|---|---|---|
| gpt-oss-120b 1만편 앵커 | job `b57-gptoss-s0..39` (a100x1 ×40)<br>out `/shared/hiontop/out/phase1_gptoss120b_v31` | 22:37 KST **23.8%**(축파일 315/1325)<br>**ETA ~06:00 KST** |

**진척 재는 법 (중요):** part tar 크기로 재면 **틀린다** — 112MB 가 `_work_*` 상수다.
```
tar -tf .../parts/part_0.tar | grep -c '_states.jsonl$'    # 목표 1325 (265편×5축)
```
느린 이유 = gpt-oss 는 **추론 강제 ON** → 토큰 바운드, 동시성 배칭 이득 못 받음.

### 끝난 뒤 할 일 (Gemma 때와 동일 절차, D-B53 §7 재현 블록 그대로)
```bash
# 1) 회수
ls /shared/hiontop/out/phase1_gptoss120b_v31/parts/part_*.tar | xargs -P8 -I{} \
  tar -xf {} -C outputs/eacl_phase0/phase1_gptoss120b_v31
# 2) 무결성 + 조립용 master
python scripts/eacl_phase1/assemble_anchor_gemma.py --out-dir outputs/eacl_phase0/phase1_gptoss120b_v31
# 3) 조립 (죽은 포트 = 캐시미스 시 즉사)
python scripts/eacl_phase0/llm_s5_states.py \
  outputs/eacl_phase0/phase1_gptoss120b_v31/_assembly_master.jsonl \
  outputs/eacl_phase0/phase1_gptoss120b_v31 \
  --model gptoss-120b --engine vllm --no-reasoning \
  --base-url http://127.0.0.1:1/v1 --prompt-version v3.1 --stage1-max-tokens 8000
# 4) 분포 대조
python scripts/eacl_phase1/anchor_label_dist.py --new .../phase1_gptoss120b_v31/reset_matrix.jsonl \
  --old outputs/eacl_phase0/phase1_all_v31/reset_matrix.jsonl --out <REPORT>/LABEL_DIST.md
# 5) M5 재학습·평가 (D-B55 경로)
python scripts/eacl_phase0/build_anchor_dataset.py --corpus phase1 --tag gptoss \
  --anchor-dir outputs/eacl_phase0/phase1_gptoss120b_v31
#   → --split phase1_gptoss 로 train_anchor_m4m5.py (db55_launch.sh 복제)
#   → db55_anchor_swap_eval.py 의 CONDS 에 ns `_phase1_gptoss_per_slot_slot_pwone` 추가
```
🔴 **평가 기준은 실행 전에 ledger 에 먼저 적는다** (D-B55 §2 형식: ΔScore ≥ +0.03 AND 3편 중 2편 무하락).

---

## 3. 미결 — 사용자 판정 대기

### (a) D-B54 채점 축 집합 불일치 🔴 **가장 중요**
- **D-B10 동결본 = `protagonist_present`.** `build_anchor_dataset.py:31` 포함 **8곳**이 이걸 쓴다.
- **`s5_crossmodel.py:17` 단 1곳만** `protagonist`(charset OR perspective) — **이탈**.
  그런데 그 스크립트가 **D-B47~B52 라벨러 비교 표를 전부 만들었다.**
- 영향: Gemma-26B-A4B 가 (A)동결축 **+0.045 통과** / (B)s5_crossmodel **+0.008 미달** 로 갈림.
- **정할 것:** ① 축 확정 → `s5_crossmodel.py:17` 수정 + D-B47~B52 재발행 여부
  ② 재발행 시 라벨러 교체 채택 여부 (단 D-B55 로 **실익은 이미 부정됨**)
- ⚠️ **M4/M5 학습·평가 파이프라인은 (A)로 일관** → 가중치·D-B44/B55 수치는 **무영향**.

### (b) MiniMax OFF 1만편 — 미착수
- 제출기 준비됨: `bash /shared/hiontop/submit_phase1_v9.sh mm27off <N>` (a100x4 → N replica = 4N GPU)
- gpt-oss 가 40장 점유 중 → **06:00 이후** 착수 권장.
- **MiniMax ON 1만편은 권하지 않음** (10~12배 느림 + Stage-1 15분 타임아웃으로 문서 통째 결손).

### (c) DeepSeek-V3.1-AWQ (D-B49) — 미실행
가중치 `/shared/models/DeepSeek-V3.1-AWQ` (337GiB) 다운로드만 완료.

---

## 4. 오늘 끝낸 것 (ledger ID → 좌표)

| ID | 결과 | 산출 |
|---|---|---|
| **D-B50** | MiniMax ON−OFF **Δ+0.003**(노이즈), 비용 **10~12배**, ON 은 pieman 3run 전멸 | `outputs/eacl_phase0/{tunnel,monkey}_michelmann_s5_mm27{on,off}_v31*` |
| **D-B52** | Gemma 4 2종 dev-3. 🔴 31B 사전등록 누락 자기신고 | `outputs/experiments/2026-07-31_s5_labeler_v31/` |
| **D-B53** | Gemma 앵커 **10,584편·2,773,435전이** 생성(3h23m, 20job). 손실 61편 원인 규명 | `outputs/eacl_phase0/phase1_gemma4a4b_v31/`<br>`outputs/experiments/2026-07-31_phase1_anchor_gemma/` |
| **D-B54** | 축 집합 불일치 발견 (위 3a) | ledger only |
| **D-B55** | Gemma 앵커 M5 **0.3609 vs Qwen 0.3963 = −0.035 기각** | `outputs/experiments/2026-07-31_db55_anchor_swap/` |
| **D-B56** | 스위칭 재개 게이트 **전 항목 불통과** → D-B28 종결 유지 | `outputs/experiments/2026-07-31_sem_switching_reopen/`<br>(`CODEX_QUESTION/ANSWER/GATE_VERDICT.md`) |
| — | M5 W3 × 임계: **adaptive 0.3963 > EWMA 전 span** | `outputs/experiments/2026-07-31_m5_w3_threshold/` |

무인 런 저널 = `EACL/UNATTENDED_2026-07-31_gemma_phase1.md`

---

## 5. 신규/수정 스크립트

| 경로 | 무엇 |
|---|---|
| `/shared/hiontop/job_shard_v9.sh` | Phase-1 앵커 job. **TP·파서·템플릿·CLIENT_EXTRA 전부 env** (v8 은 TP=1·파서없음 하드코딩) |
| `/shared/hiontop/code/scripts/eacl_phase1/resilient_shard_job_v3.sh` | 클라이언트 플래그 env화 + **D-B53 검출기 결함 수정**(로그 마지막 `[doc_id]` 도 범인 후보) |
| `/shared/hiontop/submit_phase1_v9.sh` | `gptoss\|mm27off\|mm27on` 제출기 |
| `scripts/eacl_phase1/assemble_anchor_gemma.py` | 회수분 무결성 검사 + 조립용 master 생성 |
| `scripts/eacl_phase1/anchor_label_dist.py` | 앵커 라벨 분포 대조 (동결축 + s5_crossmodel 축 병기) |
| `scripts/eacl_phase0/db55_anchor_swap_eval.py` | 앵커 교체 M5 dev-3 평가 |
| `scripts/eacl_phase0/db55_launch.sh` | 데이터빌드 대기 → W3 3seed 학습 → 평가 |
| `scripts/eacl_phase0/db56_repr_audit.py` | 스위칭 표현 감사 (**dev-3 로드 안 함**) |
| `scripts/eacl_phase0/db56_gate_verdict.py` | 게이트 판정 |
| `scripts/eacl_phase0/m5_w3_threshold_dev3.py` | M5 W3 × adaptive/EWMA |
| `scripts/eacl_phase0/build_anchor_dataset.py` | **`--anchor-dir` 추가**(+`--tag` 강제) — 다른 라벨러 앵커용 |

---

## 6. 🔴 반복 실수 방지 (오늘 실제로 밟은 것)

- **part tar 크기 = 진척 아님.** `_work_*` 112MB 상수 포함. 축파일 수를 세라.
- **`switch_scorer.py` 는 dev-3 Score 를 출력한다.** "dev-3 안 본다" 선언한 단계에서 쓰면 위반(오늘 1회 발생, ledger D-B56 에 자기신고).
- **Stage-1 예열이 resilient 폴백 검출기를 무력화한다** (D-B53, 61편 손실) → v3 러너에서 수정됨.
- **`llm_s5_states.py:699` 는 Stage-1 캐시를 모델 확인 없이 재사용** → 라벨러 바꿀 때 남의 `story_lists.tar` 풀지 말 것.
- **실행 중 스크립트 수정 금지** (bash 바이트 오프셋). 새 버전 파일명으로.
- **결과 본 뒤 기준 변경 금지.** 게이트/채택 기준은 실행 **전** ledger 에 append.

---

## 7. 🔴 끝나면 `EACL/pipeline-overview-ko.md` 에 반영할 것 (545줄 기준 좌표)

문서 규약 = **결과 먼저 · 구성요소 표 · 용어 통일 · 신조어 금지 · 내부 코드명 금지**
(W3/D-B55 같은 내부 ID 를 본문에 쓰지 말 것. "양성 가중치 제거 조건" 처럼 뜻이 보이게).

### ① §1.2 라벨링 — L60~93 **전면 갱신**
- **L62 헤드라인 유지**("동결 유지, 후보 교체 없음") — 결론은 안 바뀐다. 근거만 대폭 강해진다.
- **L74~78 후보 표에 4종 추가.** 현재 gpt-oss 한 줄뿐이다. 최종 6종(dev 3편·3회 평균):

  | 라벨러 | Pieman | Tunnel | Monkey | 평균 |
  |---|---|---|---|---|
  | Qwen3.6-35B-A3B (현행) | 0.476 | 0.471 | 0.364 | 0.4372 |
  | Gemma-4-26B-A4B | 0.544 | 0.489 | 0.413 | **0.4819** |
  | gpt-oss-120b (추론 켬) | 0.507 | 0.440 | 0.353 | 0.4334 |
  | Gemma-4-31B 밀집 | 0.449 | 0.491 | 0.289 | 0.4098 |
  | MiniMax-M2.7 추론 끔 | — | 0.535 | 0.360 | (2편) |
  | MiniMax-M2.7 추론 켬 | 실패 | 0.563 | 0.337 | (2편) |

  🔴 **위 수치는 D-B54 판정에 의존한다.** 동결 5축(`protagonist_present`) 기준이며,
  현재 문서의 0.382/0.380 은 **이탈 스크립트(`s5_crossmodel.py`) 기준**이다.
  **D-B54 가 확정되기 전에는 표를 갈아끼우지 말 것.** 확정 후 §1.3.1 L147 의 0.382 도 함께.
- **L83~86 "추론 조건 비대칭" 한계 → 정량화된 근거로 교체.**
  MiniMax 로 같은 모델의 추론 켬/끔을 직접 비교했다: **평균 차 +0.003(표준편차의 1/10),
  2편에서 부호가 갈림.** ⇒ "추론을 켠 채 비교한 것이 gpt-oss 를 크게 불리하게 하지는 않았다"
  로 한계 문장을 **약화**한다. 단 **비대칭 하나가 남는다** — 켬 조건만 추론 파서를 쓴다.
- **L87~92 하드웨어 판정 갱신 2건:**
  ① "MiniMax-M2.7 … 추론 비활성 스위치 없음" → **템플릿 1줄 개조로 끔 조건을 만들어 실제로 비교했다.**
  ② **추론 켬의 비용을 명시**: 같은 문서에서 **10~12배 느리고**, Pieman 은 선택지 생성 단계가
  15분 제한을 넘겨 **3회 모두 산출 실패**.
- **L72 규모 행에 2번째 앵커 실적 추가**(인프라 개선 근거): A100×20, **3시간 23분**, 20/20 성공.
  단축분은 거의 전부 **회수 구간**(개별 파일 복사 4시간 10분 → 묶음 회수 수 초)에서 나왔다.

### ② §1.3 라벨링 평가 — **새 소절 신설**(L184 전문가 검수 앞)
제목 예: `### 1.3.3 라벨러 점수는 분절기 점수를 예측하지 않는다`
- 가장 높은 점수의 라벨러(Gemma-4-26B-A4B, +0.045)로 앵커 1만 편을 다시 만들어 분절기를
  같은 조건으로 학습했더니 **0.3609 대 0.3963 = −0.035 로 오히려 나빠졌다.** 3개 시드, 2편 하락.
- 앵커에는 경계 정답이 없다 ⇒ 라벨러의 dev 우위는 *그 라벨러를 분절기로 직접 쓸 때*의 우위이지
  *그 라벨을 학습한 모델*의 우위가 아니다.
- ⇒ **라벨러 후보를 라벨러 단독 점수로 고르지 않는다**를 방법론으로 명시.
- 두 앵커 라벨 분포는 거의 같다(5축 ±0.042, 경계율 0.0600 대 0.0649) —
  **집계 일치가 전이 단위 일치를 뜻하지 않는다**(주인공 −14% 대 시간 +68% 가 상쇄).

### ③ §2.1.2 손실 — L263 표의 α 행
**`α = [1.0, 0.5, 1.0, 0.4, 0.25]` 는 근거 기록이 없다**를 각주로 명시(오늘 전수 조사 확인).
설계 본체 `factored-slot-eism-v0_1.md:110` 은 *"No ad-hoc per-slot weights; normalization is
emergent"* 로 **정반대**를 규정한다. 현행 α 의 유일한 근거는 **균등화 조건보다 dev 3편에서
0.0094 높았다**는 사후 실측이며, 그 선택에는 dev 선택 성분이 있다.

### ④ §2.2 적응적 임계 — L291 이후에 국소 임계 대조 추가
동일 체크포인트·동일 신호에서 **누적 규칙 0.3963 이 국소(EWMA) 규칙을 전 구간에서 이긴다**:
span 50/30/20/10 = 0.3815 / 0.3683 / 0.3675 / 0.3615, 과분절 2.51~2.95배(누적은 2.21배).
span 이 길수록 누적에 수렴한다(극한이 누적평균이므로 당연). ⇒ **국소 임계가 과분절을 악화시킨다.**

### ⑤ §2.1 말미 L252~254 — 스위칭 종결 문구 **범위 축소**
현재 "스위칭 경로를 종결" 로만 적혀 있다. 다음을 반영:
- 설계대로 4단계를 **전부 붙여 검정**했고(온셋 분포 → 슬롯 고유 예측공간 → 이종 분포족 → 차원 사영)
  **네 번 모두 개선됐으나 매번 리셋 확률 합 아래**였다.
- 양성 가중치를 제거한 새 체크포인트에서 **재개 게이트를 다시 걸었으나 전 항목 불통과**:
  우도비 분산 감소 없음(−16.8%/+10.3%), 지속성 사전은 여전히 무력(경계 수 변동 1.015~1.017배),
  같은 판에서 **우도비 AUC 0.713~0.717 < 리셋 확률 합 0.805~0.808**.
- 🔴 **주장 범위를 좁힌다**: *"SEM 확률적 추론층 전체 기각"* 이 아니라
  **"검정된 전진 local-MAP 스위칭 인스턴스 기각"**. **지연 확정(particle smoothing)은 미검정**이며
  결론을 내리지 않고 후속 범위로 남긴다. (권장 문구 = ledger D-B56 §6 원문 그대로 인용)

### ⑥ §3.3 종합 판정 — L536~545 표에 행 추가/수정
- L543 `확정 패스 … 앵커 재생성` → **"앵커 재생성은 시도했고 이득 없음(−0.035)"** 으로 수정.
  남은 것은 재학습·시험지 개봉뿐임을 분명히.
- 신규 행 3개:
  | 라벨러 교체 | **기각** — 후보 5종 전부 채택 조건 미달, 최고 점수 라벨러로 앵커를 다시 만들어도 분절기는 하락 |
  | SEM 확률적 전환층 | **종결** — 재개 게이트 전 항목 불통과. 지연 확정만 미검정으로 남김 |
  | 추론(thinking) 사용 | **비권장** — 점수 이득 없음(+0.003), 비용 10~12배, 문서 단위 산출 실패 유발 |

### ⑦ 진행 중 항목 반영 (gpt-oss 앵커가 끝난 뒤)
결과가 나오면 §1.2 후보 표와 §1.3.3(신설)에 **"gpt-oss 앵커로 학습한 분절기" 행**을 추가한다.
🔴 그 비교의 채택 기준은 **실행 전에** ledger 에 먼저 적는다(D-B55 §2 형식).

### ⑧ 갱신 후 필수 검사
- **내부 코드명 제거 확인**: 본문에 `W0/W1/W3`, `D-Bxx`, `_per_slot_slot_pwone` 같은 표기가
  남지 않았는지. ledger 참조는 각주로만.
- **용어 일관**: "경계" / "전이" / "리셋" / "동시 리셋 문턱" / "과분절 배수" — §9 용어표와 대조.
- **dev 3편 절대 점수 인용 금지 규약**과 충돌하지 않는지 — 이 문서는 Δ 와 방향 위주로 쓴다.
- **cascade 검사**(CLAUDE.md): `context/methodology/factored-slot.md`(α 근거 부재·스위칭 종결 범위),
  `README.md`, `plan-factored-slot.md` 도 같은 변경이 필요한지 확인 후 사용자에게 일괄 제시.
