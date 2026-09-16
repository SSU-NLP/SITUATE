# CSM **offline** (SuperDialseg `CSMSegmenter`, whole-dialogue)

backbone=bert-base-uncased max_utt_len=50 cut_rate=1.0 ckpt=external/Dialogue-Topic-Segmenter/checkpoints/cpt_125525.pth device=cuda:0
데이터=Def-DTS 번들 test, metric=segeval Pk/WD + boundary-set F1, Score=0.5F1+0.25(1-Pk)+0.25(1-WD).
online 판 (methods/CSM/online/delay2.py) 과 동일 harness.

| dataset | n_dial | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/dial(ms) | pred bs | gold bs | miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100 | 0.4209 | 0.4476 | 0.4165 | 0.4911 | 12.6 | 307 | 315 | 0 |
| dialseg711 | 711 | 0.4310 | 0.4724 | 0.3955 | 0.4719 | 19.1 | 4434 | 2743 | 0 |
| superseg | 1322 | 0.4756 | 0.4841 | 0.3920 | 0.4561 | 11.2 | 3173 | 4017 | 0 |

## 한계
- offline = 대화 전체(미래 포함). online 판은 `methods/CSM/online/delay2.py` (delay-2, AUXILIARY).
- ckpt 미지정 시 BERT-NSP pretrained 가중치만 사용 → 점수 낮음. 동작 검증용.
- SuperDialseg `CSMSegmenter` 의 `CoherenceScoringModel` 은 lxing532 `CoherenceNet` 와 다른 모듈 구조 — ckpt 호환 주의.
