### Density (turns/segment) — online c calibrated on dev, applied to test

| method | mode | c(dev) | dev tps | test tps |
|---|---|---|---|---|
| M1-CosDrift | online | 1.3 | 5.083 | 5.238 |
| surprise-gpt2xl | online | 0.9 | 5.178 | 5.11 |
| M3-5axis | online | 0.9 | 5.055 | 5.011 |
| Hybrid | online | 1.2 | 4.876 | 4.625 |

### Evidence recall@k — TEST split, overall (density-matched)

| method | mode | r@1 | r@2 | r@3 | r@5 |
|---|---|---|---|---|---|
| M1-CosDrift | online | 0.240 | 0.341 | 0.403 | 0.485 |
| M1-CosDrift | offline | 0.231 | 0.326 | 0.397 | 0.494 |
| surprise-gpt2xl | online | 0.266 | 0.386 | 0.462 | 0.546 |
| surprise-gpt2xl | offline | 0.229 | 0.321 | 0.389 | 0.477 |
| M3-5axis | online | 0.272 | 0.394 | 0.453 | 0.554 |
| M3-5axis | offline | 0.280 | 0.378 | 0.450 | 0.544 |
| Hybrid | online | 0.257 | 0.375 | 0.444 | 0.538 |
| Hybrid | offline | 0.251 | 0.346 | 0.416 | 0.506 |

### Evidence recall@5 by category — TEST, ONLINE (deployable)

| method | multi-hop | temporal | open-domain | single-hop | adversarial | overall |
|---|---|---|---|---|---|---|
| M1-CosDrift | 0.316 | 0.382 | 0.308 | 0.569 | 0.550 | **0.485** |
| surprise-gpt2xl | 0.348 | 0.485 | 0.262 | 0.633 | 0.610 | **0.546** |
| M3-5axis | 0.313 | 0.570 | 0.321 | 0.639 | 0.579 | **0.554** |
| Hybrid | 0.337 | 0.501 | 0.263 | 0.624 | 0.586 | **0.538** |

### Final answer GPT4Score (×10, /100) by category — TEST, ONLINE

| method | multi-hop | temporal | open-domain | single-hop | adversarial | overall | ctx_tok |
|---|---|---|---|---|---|---|---|
| M1-CosDrift | 35.4 | 17.0 | 20.2 | 54.4 | 11.5 | **34.65** | 512 |
| surprise-gpt2xl | 35.6 | 18.7 | 26.6 | 56.9 | 12.5 | **36.56** | 557 |
| M3-5axis | 36.9 | 21.5 | 24.6 | 58.2 | 11.0 | **37.32** | 615 |
| Hybrid | 34.9 | 20.2 | 24.6 | 57.3 | 11.7 | **36.58** | 539 |
