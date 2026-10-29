#!/usr/bin/env bash
# Experiment B: for each prefill chunk budget, restart the server and measure chat users'
# token gaps under three workloads (off, colocated, kvremote), REPEATS times each.
# Run Experiment A first: its results set how often summarization requests arrive.
#
#   bash run_exp_b.sh                                               # rented GPU, Qwen2.5-7B
#   LPT=1 bash run_exp_b.sh                                         # also cap each request's chunk
#   MOCK=1 REPEATS=1 bash run_exp_b.sh --warmup 3 --duration 20      # laptop practice, fake numbers
#
# LPT=1 adds --long-prefill-token-threshold (chunk budget minus 64), so one long prompt can
# never take the whole step and leave chat users without a token. Comparing LPT=0 and LPT=1
# is a result in itself.
# Anything after the script name is passed to exp_b_interference.py.
set -euo pipefail
cd "$(dirname "$0")"
source ./server.sh

CHUNKS=${CHUNKS:-"512 2048 8192"}
REPEATS=${REPEATS:-3}
MODES=${MODES:-"off colocated kvremote"}
mkdir -p results/exp_b logs
record_env results/exp_b/env_b.txt

for CHUNK in $CHUNKS; do
  CONFIG="chunk${CHUNK}"
  EXTRA=(--max-num-batched-tokens "$CHUNK" --enable-prefix-caching)
  if [[ "${LPT:-0}" == "1" ]]; then
    CONFIG="chunk${CHUNK}-lpt"
    EXTRA+=(--long-prefill-token-threshold "$((CHUNK - 64))")
  fi
  echo "=== $CONFIG ==="
  start_server "logs/server_b_$CONFIG.log" "${EXTRA[@]}"
  start_gpu_log "results/exp_b/gpu_log_$CONFIG.csv"
  for REP in $(seq 1 "$REPEATS"); do
    for MODE in $MODES; do
      python exp_b_interference.py --base-url "http://localhost:$PORT" --mode "$MODE" \
        --seed "$REP" --label "${CONFIG}_${MODE}_r${REP}" "$@"
    done
  done
  stop_gpu_log
  stop_server
  save_server_log "logs/server_b_$CONFIG.log"
done
echo "done: results in results/exp_b/"
