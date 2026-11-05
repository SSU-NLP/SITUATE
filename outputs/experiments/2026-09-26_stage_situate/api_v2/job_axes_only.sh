#!/usr/bin/env bash
set -euo pipefail
WORK=$(mktemp -d /tmp/stage-axes.XXXXXX)
BASE=/shared/hiontop/stage-situate-20260926
OUT=/shared/hiontop/stage-situate-api-v2
mkdir -p "$OUT"
exec > >(tee -a "$OUT/axes-job.log") 2>&1
printf '%s  %s\n' 577d6d026f121abb3741fd18b5370e291d54d512e14e481f888980a0ebbfc1ea "$BASE/package-v1.tar.gz" | sha256sum -c -
tar -xzf "$BASE/package-v1.tar.gz" -C "$WORK"
cd "$WORK"
cp "$OUT/extract_axes.py" scripts/stage/extract_axes.py
mkdir -p "$OUT/axes"
cp -n "$BASE"/shard{0,1}/axes/*.json "$OUT/axes/"
python -m pip install 'transformers==4.56.2' 'peft==0.17.1' 'tokenizers==0.22.2' 'nltk==3.10.3' sentencepiece
python -c "import nltk; nltk.download('punkt_tab', raise_on_error=True)"
python scripts/stage/verify_checkpoint.py --artifacts artifacts --output "$OUT/checkpoint_verification.json"
timeout 3600 python scripts/stage/extract_axes.py --inputs inputs --artifacts artifacts --output "$OUT/axes" --num-shards 1
python - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);n=len(list((p/'axes').glob('*.json')));assert n==151,n
(p/'axes_complete.json').write_text(json.dumps({'movie_count':n,'status':'complete'}))
PY
