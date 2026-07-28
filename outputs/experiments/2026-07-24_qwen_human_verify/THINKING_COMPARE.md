# Qwen3.6-35B **thinking-on** 재라벨 + 동시성 램프 (human-verify 100문장)

> 모델 Qwen/Qwen3.6-35B-A3B(fp8), `--reasoning-parser qwen3`, enable_thinking=True, guided json_schema. off와 동일 Stage-2 프롬프트. n=100/100 완료, 에러 0. wall 82.6s.

## 처리량

- 총 출력 146161 tok / 82.6s = **1770 tok/s**(집계).
- 호출당 지연 median 46.3s / p90 58.6s / max 79.0s. 평균 출력 1462 tok/call.

## 동시성 램프 (offered cap → 서버 실제 running/waiting)

| offered | running mean | running peak | waiting peak | gpu_cache peak |
|---|---|---|---|---|
| 8 | 6.7 | 8 | 0 | 0.00 |
| 16 | 14.7 | 16 | 0 | 0.00 |
| 32 | 29.3 | 32 | 0 | 0.00 |
| 64 | 60.0 | 64 | 24 | 0.00 |
| 100 | 60.4 | 100 | 6 | 0.00 |

- **관측된 최대 동시 실행(running peak) = 100** (offered 100까지 올렸을 때). 그 이상은 waiting 으로 큐잉.

## thinking vs 사람(중립/비중립) — off 대비

| 축 | n | think 일치 | (off 일치) | Δ | think over | think under | flips | →정답 | →오답 |
|---|---|---|---|---|---|---|---|---|---|
| space | 20 | 15 | 14 | +1 | 3 | 2 | 8 | 3 | 2 |
| intentionality | 20 | 19 | 17 | +2 | 0 | 1 | 4 | 2 | 0 |
| time | 20 | 20 | 18 | +2 | 0 | 0 | 2 | 2 | 0 |
| protagonist | 20 | 19 | 17 | +2 | 1 | 0 | 3 | 2 | 0 |
| causality | 20 | 17 | 15 | +2 | 0 | 3 | 2 | 2 | 0 |
| **전체** | 100 | **90** | 81 | +9 | 4 | 6 | 19 | 11 | 2 |

> Δ = think 일치 − off 일치(둘 다 vs 사람). flips=off 대비 라벨 바뀐 수, →정답=off가 틀렸는데 think가 맞춤, →오답=off 맞았는데 think가 틀림.

