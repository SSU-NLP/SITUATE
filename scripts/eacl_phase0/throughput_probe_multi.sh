#!/usr/bin/env bash
# 다복제 합계 처리량 실측 — throughput_probe.py 를 복제 수만큼 동시 실행하고 콜/s 를 합산.
#
#   bash scripts/eacl_phase0/throughput_probe_multi.sh <복제당동시성> <포트...>
#   예: bash scripts/eacl_phase0/throughput_probe_multi.sh 130 8001 8002 8003 8004
#
# 단일 복제 측정(6.01/7.88/9.45, 2026-07-26)과 달리 복제 간 클라이언트 경합까지 포함한
# 실제 합계를 잰다. 각 probe 는 warmup 10s + 측정 30s.
set -u
cd /root/Hi-OnTop
CONC="${1:?usage: throughput_probe_multi.sh <conc_per_replica> <port...>}"; shift
PORTS=("$@")
[ "${#PORTS[@]}" -gt 0 ] || { echo "포트를 하나 이상 지정"; exit 1; }
PY=/root/Hi-OnTop/.venv/bin/python
OUT=$(mktemp -d)

echo "=== multi-probe: 복제 ${#PORTS[@]}대 × conc $CONC (합계 동시성 $(( CONC * ${#PORTS[@]} ))) ==="
pids=()
for p in "${PORTS[@]}"; do
  $PY scripts/eacl_phase0/throughput_probe.py "$CONC" "http://localhost:$p/v1" \
    > "$OUT/p$p.txt" 2>&1 &
  pids+=($!)
done
echo "  probe PID: ${pids[*]}  (기동 확인 중...)"
sleep 25
echo "  살아있는 probe: $(ps -o pid= -p "${pids[@]}" 2>/dev/null | wc -l)/${#PORTS[@]}"
for p in "${PORTS[@]}"; do
  printf '    :%s 부하 → %s\n' "$p" \
    "$(tail -80 /root/vllm_pilot/logs/tp1_rep$p.log 2>/dev/null | grep -oE 'Running: [0-9]+ reqs' | tail -1)"
done
wait "${pids[@]}" 2>/dev/null

tot=0
for p in "${PORTS[@]}"; do
  c=$(grep -oE '\*\*[0-9.]+ 콜/s\*\*' "$OUT/p$p.txt" | grep -oE '[0-9.]+' | head -1)
  o=$(grep -oE '출력 tok/s: [0-9]+' "$OUT/p$p.txt" | grep -oE '[0-9]+' | head -1)
  l=$(grep -oE 'median [0-9.]+s' "$OUT/p$p.txt" | grep -oE '[0-9.]+' | head -1)
  printf '  :%s  %6s 콜/s  출력 %5s tok/s  지연 median %ss\n' "$p" "${c:-FAIL}" "${o:-?}" "${l:-?}"
  [ -n "${c:-}" ] && tot=$(echo "$tot + $c" | bc -l)
done
printf '\n  합계 = %.2f 콜/s\n' "$tot"
echo "  (원시 출력: $OUT)"
