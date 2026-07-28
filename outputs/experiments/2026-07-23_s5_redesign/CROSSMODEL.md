# ④ cross-model Score 비교 (프롬프트 고정, dev-3, deployable t\*=3)

> 각 1 run(배치 비결정 ±0.01~0.05 노이즈). v3=전부 no-think(공정). deepseek v2=reasoning(교란).

## v2 프롬프트 (⚠️ deepseek=reasoning, 오픈=no-think)

| 모델 | pieman | tunnel | monkey | 평균 |
|---|---|---|---|---|
| deepseek-v2 (reason) | 0.578 | 0.344 | 0.253 | **0.392** ★ |
| GLM-v2 | 0.365 | 0.383 | 0.291 | **0.346** |
| Qwen32B-v2 | 0.447 | 0.291 | 0.316 | **0.351** |
| Qwen3-Next-v2 | 0.517 | 0.280 | 0.318 | **0.372** |

## v3 프롬프트 — 전부 no-think (공정 비교)

| 모델 | pieman | tunnel | monkey | 평균 |
|---|---|---|---|---|
| deepseek-v3-OFF | 0.490 | 0.250 | 0.351 | **0.364** ★ |
| GLM-v3 | 0.401 | 0.343 | 0.311 | **0.352** |
| Qwen32B-v3 | 0.391 | 0.294 | 0.362 | **0.349** |
| Qwen3-Next-v3 | 0.415 | 0.236 | 0.304 | **0.318** |

> 참고 deepseek-v3-ON(reason) 평균 = 0.327 (v3에서 reasoning은 해로움).

