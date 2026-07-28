# S5 라벨러 추론 서버 설정 — 외부 인계용

작성 2026-07-27. 대상 = 이 설정을 다른 사람/다른 클러스터에서 그대로 재현해야 하는 경우.
자체 완결 문서 — 이 파일만 보고 서버를 띄우고 확장할 수 있어야 한다.

---

## 1. 한 줄 요약

**Qwen3.6-35B-A3B(bf16 원본)를 vLLM 에서 fp8 로 로드해 A100 80GB 1장당 복제 1개씩 띄우고,
복제 수를 늘려 처리량을 선형으로 키운다. TP(텐서 병렬)를 키우면 오히려 느려진다.**

---

## 2. 서버 기동 명령 (canonical)

```bash
# venv/bin 을 PATH 에 넣어야 한다 — 아래 §6 함정 1 참조
export PATH="/path/to/venv/bin:$PATH"

CUDA_VISIBLE_DEVICES=<GPU번호> python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B \
  --served-model-name qwen36-35b \
  --quantization fp8 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 512 \
  --reasoning-parser qwen3 \
  --no-enable-log-requests \
  --enable-prefix-caching \
  --port <8001+GPU번호>
```

- vLLM **0.25.1**. GPU 1장당 이 명령 1개, 포트만 다르게.
- `--model` 은 HF ID 또는 스냅샷 절대경로. 공유 스토리지에서 읽을 땐 **절대경로 권장**
  (허브 접근 0 → 읽기전용 마운트·오프라인 안전). 리비전 `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- 체크포인트는 **bf16 67GB 하나뿐**. fp8 은 별도 파일이 아니라 로드 시 옵션이다.

### 클라이언트 (`llm_s5_states.py`)

```
--engine vllm --base-url http://127.0.0.1:<port>/v1 --model qwen36-35b --no-reasoning
```

`--no-reasoning` → `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`. **frozen = no-thinking.**

---

## 3. 실측치 (A100-SXM4-80GB, 2026-07-27)

| TP | 가중치/GPU | KV/GPU | KV 토큰 | 비고 |
|---:|---:|---:|---:|---|
| 1 | **33.89 GiB** | 29.99 GiB | 1,283,125 | 80GB 1장에 여유 |
| 2 | 17.06 GiB | 46.29 GiB | 4,302,906 | |
| 4 | 8.67 GiB | 54.73 GiB | 5,396,293 | |

**기동 시간** (TP=1): 가중치 로드+fp8 양자화 60s(캐시 warm) / 105s(cold) → KV 프로파일 + CUDA graph
캡처 → **ready 까지 ~200s**.

**처리량** (배치20 Stage-2, conc 200, 2026-07-26 실측):

| 구성 | 콜/s |
|---|---:|
| TP=1 복제 1개 | 6.01 |
| TP=2 복제 1개 | 7.88 |
| TP=4 복제 1개 | 9.45 |

→ GPU 4장 기준: TP=1×4 ≈ **24.0** (투영) > TP=2×2 = 15.76 (실측) > TP=4×1 = 9.45 (실측).

> ⚠ **24.0 은 6.01×4 투영이며 아직 실측되지 않았다.** 2026-07-27 측정 시도는 두 세션이
> 같은 포트를 동시에 때려 무효화됐다. 인계받는 쪽에서 반드시 재측정할 것.

**동시성 한계** (2026-07-24 실측, §infrastructure 4c):
- no-thinking KV 벽 ≈ **320 동시**. 시퀀스 ~2.5k tok(입력 지배).
- `--max-num-seqs` 가 하드캡. 512 로 두면 KV 벽(320)이 먼저 걸린다.
- 권장 클라이언트 동시성 = **복제당 250~300**. 넘기면 preemption(재계산)으로 손해.

**Stage-2 프롬프트 구조** (12문서×2축 실측):
- 프롬프트 984~1,180 tok. **배치 간 공유 접두 = 평균 27.0%** (지시문+필드설명+개체목록).
- Stage-2 에는 **스토리 본문이 들어가지 않는다** (Stage-1 만 전문 입력).
- → prefix caching 이득 상한 = prefill 의 27%. decode 지배 워크로드라 end-to-end 효과는 한 자릿수 %.

---

## 4. 설계 근거

**왜 TP=1 인가.** 이 모델은 MoE 로 총 35B / **활성 3B**다. 활성 파라미터가 작아 연산량 대비
GPU 간 all-reduce 통신이 지배적이다. TP 를 4배(1→4)로 키워도 1.57배(6.01→9.45)밖에 못 얻는다.
반면 복제를 늘리면 통신이 0이라 **선형**이다. 가중치가 33.89 GiB 라 80GB 1장에 들어가는 것이 전제.

- 40GB 카드에서는 TP=1 불가 → 이 구성 전체가 성립하지 않는다.
- 48GB(L40S)는 가중치 34 + KV ~7 GiB 로 빠듯해 동시성이 급감한다. 권장하지 않음.

**왜 `--max-model-len 32768` 인가.** 16384 로 내리면 Stage-1 컨텍스트 초과로 **278편이 탈락**한다.
내리지 말 것.

**왜 `--reasoning-parser qwen3` 인가.** thinking 을 쓸 경우 이 파서가 없으면 guided `json_schema` 가
첫 토큰부터 JSON 을 강제해 추론이 억제된다. frozen 은 no-thinking 이지만 파서는 붙여둔다.

**왜 fp8 사전 양자화를 하지 않는가.** on-the-fly 양자화 비용이 60초에 불과하고, 사전 양자화본은
스케일 계산 방식이 달라져 **출력이 미세하게 달라질 수 있다**. 기존 결과가 전부 on-the-fly fp8 로
나왔으므로 일관성 위험이 이득보다 크다.

> A100(Ampere)은 FP8 텐서코어가 없다. fp8 은 **메모리 절약만** 주고 연산 속도 이득은 없다.
> H100 이상에서는 연산까지 빨라진다.

---

## 5. 확장 (복제 N개)

복제끼리 통신이 없으므로 **처리량은 복제 수에 선형**이다.

```
GPU 4장  → 복제 4개  → ~24 콜/s (투영)
GPU 20장 → 복제 20개 → ~120 콜/s (투영)
```

작업 분배는 **(문서 × 축) 단위 round-robin**. `{axis}_states.jsonl` 이 축별 파일로 분리돼
같은 출력 디렉토리에 여러 복제가 동시에 써도 충돌하지 않는다 (로컬 2복제로 검증됨).

resume 은 `item_id` 기반이라 프로세스를 죽여도 손실이 없다. **단, 프롬프트를 바꿨으면 해당 축
파일을 삭제 후 재추출**해야 한다(안 그러면 구 프롬프트 산출을 이어받는다).

### VESSL Job 으로 돌리는 경우

`scripts/vessl/` 3종 참조. Job 1개 = A100 1장 = 복제 1개.

```bash
bash scripts/vessl/stage_to_shared.sh <샤드디렉토리>   # 코드·데이터를 /shared 에 스테이징
N=2  RUN=test1  bash scripts/vessl/submit_jobs.sh      # 검증
N=20 RUN=<name> bash scripts/vessl/submit_jobs.sh      # 본run
```

- 모델은 **공유 볼륨에 1회만** 배치한다(`/shared/models/Qwen3.6-35B-A3B`). Job 마다 받으면
  67GB × N 의 다운로드가 발생한다.
- `resourcespec-a100x1` = $1.55/hr. 20개 = $31.00/hr.

---

## 6. 함정 (전부 실제로 밟은 것)

1. **`ninja` / PATH.** FlashInfer 샘플러가 JIT 빌드 때 `ninja` 를 exec 로 찾는다. venv 활성화 없이
   `venv/bin/python` 을 직접 부르면 PATH 에 `ninja` 가 없어 **기동 즉시 전멸**한다
   (`FileNotFoundError: 'ninja'`). `export PATH="$(dirname $(command -v python)):$PATH"` 필수.
2. **`pkill -f <패턴>` 자폭.** 자기 명령줄까지 매치해 스스로 죽는다(exit 144). PID 로 죽일 것.
   프로세스 확인도 bracket 트릭(`grep "api_serve[r]"`)을 쓴다.
3. **포트 충돌.** 여러 사람/세션이 같은 포트를 때리면 측정이 조용히 오염된다. 측정 전
   `ps` 로 다른 클라이언트를 확인하고, 측정 중 복제별 `Running: N reqs` 를 찍어 분산을 눈으로 볼 것.
4. **측정 창 < 지연.** conc 130 에서 콜 지연 median 이 28.5s 인데 측정 창이 30s 면 정상상태 전에
   창이 닫혀 값이 절반으로 나온다. **시간 고정이 아니라 호출 수 고정**으로 재는 편이 안전하다.
5. **서버 재시작 후 잔류 요청.** kill 해도 in-flight 요청이 서버에서 계속 돈다. 다음 측정 전
   `Running: 0` 을 확인할 것(실측 345s 걸린 적 있음).

---

## 7. 일관성 절대 조건

**축이나 코퍼스 중간에 서버 설정을 바꾸지 않는다.**

vLLM 의 연속 배칭은 비트 단위로 결정적이지 않다. 배치 구성·prefix cache 히트 여부·복제 수가
바뀌면 같은 입력에도 출력이 미세하게 달라질 수 있다. 설정 변경은 **새 코퍼스 시작 시점에만**
한다. 이미 뽑은 결과와 새 설정 결과를 한 축 안에 섞으면 비교가 오염된다.

`--enable-prefix-caching` 을 켜는 것도 이 규칙의 적용 대상이다(기존 결과는 전부 캐싱 OFF 로 산출됨).

---

## 8. 검증 미해결

- **TP=1 × 4복제 합계 처리량이 실측되지 않았다.** 24.0 콜/s 는 단일 복제 6.01 의 산술 투영이다.
  클라이언트 CPU 경합·공유 스토리지 I/O 가 선형성을 깎을 수 있다.
- **prefix caching 의 실제 end-to-end 이득이 측정되지 않았다.** 공유 접두 27% 는 정적 분석치고,
  실제 콜/s 개선폭은 미측정.
- **VESSL Job 환경에서 한 번도 실행된 적이 없다.** 이미지 차이, `/shared`(object storage FUSE)의
  67GB 로드 속도, 20 Job 동시 확보 가능성 모두 미확인.
- 권장 동시성 250~300 은 2026-07-24 의 `--max-model-len 16384` 조건에서 나온 KV 벽(320) 기준이다.
  32768 조건에서 재확인 필요.
