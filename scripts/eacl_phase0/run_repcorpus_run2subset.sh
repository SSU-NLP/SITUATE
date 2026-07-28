#!/bin/bash
# ①B run2 = 결정성 검사 6편 부분(트림, D-B11 REPORT에 사유 명시). -P3로 origin 부하↓.
set -u
cd /root/Hi-OnTop
export PY=/root/Hi-OnTop/.venv/bin/python
export OUT=outputs/eacl_phase0/repcorpus_s5_eismv2_run2
export MAS=outputs/eacl_phase0/repcorpus_masters
mkdir -p "$OUT"
DOCS="fam_gilman_wallpaper fam_london_fire fam_doyle_speckled obs_28922 obs_25744 obs_74140"
AXES="protagonist space time causality intentionality"
echo "=== RUN2 subset start $(date +%H:%M:%S) ==="
for d in $DOCS; do for a in $AXES; do echo "$d $a"; done; done \
  | xargs -P3 -n2 bash -c '
      $PY scripts/eacl_phase0/llm_s5_states.py "$MAS/$0.jsonl" "$OUT" --axis "$1" \
        >> "$OUT/parallel_$0_$1.log" 2>&1 && echo "[ok] $0 $1 $(date +%H:%M:%S)" || echo "[FAIL] $0 $1"'
echo "--- RUN2 full pass (subset master) $(date +%H:%M:%S) ---"
$PY scripts/eacl_phase0/llm_s5_states.py outputs/eacl_phase0/repcorpus_run2subset_master.jsonl "$OUT" >> "$OUT/full.log" 2>&1
echo "=== RUN2 done $(date +%H:%M:%S): reset_matrix rows = $(wc -l < "$OUT/reset_matrix.jsonl" 2>/dev/null) ==="
