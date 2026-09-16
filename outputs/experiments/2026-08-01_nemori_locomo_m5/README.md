# 2026-08-01_nemori_locomo_m5 — 재현 방법

Nemori 의 LLM 배치 분절기를 M5-FactoredSlot 으로 교체하고 LoCoMo 에서 대조한 실험.
설계·동결값은 `PRE_REGISTRATION.md`, 결과·해석은 `REPORT.md`.

## 0. 사전 준비 (1회)

```bash
# 외부 저장소 (gitignore 대상 — 커밋 안 됨)
git clone --depth 1 https://github.com/nemori-ai/nemori.git external/nemori

# 파이썬 의존성 (기존 .venv 에 **추가만**, 기존 패키지 버전 변경 없음)
uv pip install --python .venv/bin/python asyncpg "qdrant-client>=1.7.0" rouge-score bert-score

# PostgreSQL 16
apt-get install -y postgresql postgresql-contrib
pg_ctlcluster 16 main start
su postgres -c "psql -c \"CREATE USER nemori WITH PASSWORD 'nemori' SUPERUSER;\""
su postgres -c "createdb -O nemori nemori"

# Qdrant 1.18.3 (docker 없는 서버라 정적 바이너리)
mkdir -p external/_infra && cd external/_infra
curl -sL -o qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.18.3/qdrant-x86_64-unknown-linux-musl.tar.gz
tar xzf qdrant.tar.gz && nohup ./qdrant --disable-telemetry > qdrant.log 2>&1 &
```

`.env` 에 `OPENAI_API_KEY` · `OPENAI_BASE_URL` · `CF_ACCESS_CLIENT_ID` · `CF_ACCESS_CLIENT_SECRET`.
환경변수 이름 변환·헤더 주입은 `scripts/nemori_locomo/nemori_env.py` 가 알아서 한다.

## 1. 사전점검 (LLM 호출 0회)

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/nemori_locomo/segment_dryrun.py
```

## 2. 두 팔 실행 (병렬 가능)

```bash
for arm in baseline m5; do
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
  .venv/bin/python scripts/nemori_locomo/run_arm.py \
      --arm $arm --name 2026-08-01_nemori_locomo_m5 --reset \
      --stages add,search,evals,scores &
done; wait
```

스모크는 `--limit 1 --limit-qa 12` 를 붙인다.

## 3. 집계

```bash
PYTHONPATH=src .venv/bin/python scripts/nemori_locomo/collect_stats.py    --name 2026-08-01_nemori_locomo_m5
PYTHONPATH=src .venv/bin/python scripts/nemori_locomo/evidence_recall.py  --name 2026-08-01_nemori_locomo_m5
PYTHONPATH=src .venv/bin/python scripts/nemori_locomo/make_report.py      --name 2026-08-01_nemori_locomo_m5
```

## 4. 계약 테스트 (LLM·DB 불필요)

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_nemori_segmenter.py -q
```

## 산출 파일

| 경로 | 내용 |
|---|---|
| `<arm>/config.json` | 그 팔이 실제로 쓴 MemoryConfig |
| `<arm>/config_published_reference.json` | 원본 `evaluation/locomo/config.json` (비교용) |
| `<arm>/segmenter_info.json` | 어떤 분절기가 꽂혔는지 |
| `<arm>/run.log` | 전체 로그 (add.py 의 LLM 사용 요약 포함) |
| `<arm>/results.json` | 질문별 검색 결과 + 생성 답변 |
| `<arm>/metrics.json` | BLEU / F1 / LLM judge |
| `m5/m5_segmentation_trace.jsonl` | 배치별 분절 통계 |
| `stats.json` · `evidence_recall.json` · `RESULTS_TABLES.md` | 집계 |
