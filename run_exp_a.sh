#!/usr/bin/env bash
# Experiment A + A2: start the server, run the prefill/decode sweep, estimate KV transfer costs.
#
#   bash run_exp_a.sh                                               # rented 24 GB GPU, Qwen2.5-7B
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct DTYPE=half bash run_exp_a.sh   # free T4 (Colab/Kaggle)
#   MOCK=1 bash run_exp_a.sh --repeats 2                            # laptop practice, fake numbers
#
# CHUNK sets --max-num-batched-tokens (default 2048, vLLM's own default on a 24 GB GPU).
# Anything after the script name is passed to exp_a_sweep.py.
set -euo pipefail
cd "$(dirname "$0")"
source ./server.sh

CHUNK=${CHUNK:-2048}
mkdir -p results logs

record_env results/env_a.txt
start_server logs/server_a.log --max-num-batched-tokens "$CHUNK" --no-enable-prefix-caching
start_gpu_log results/gpu_log_a.csv

python exp_a_sweep.py --base-url "http://localhost:$PORT" --chunk-budget "$CHUNK" \
  --out results/exp_a.csv "$@"

stop_gpu_log
stop_server
save_server_log logs/server_a.log

KEY_ARGS=()
if [[ "${MOCK:-0}" == "1" ]]; then KEY_ARGS=(--model-key qwen2.5-7b); fi
python exp_a2_kv_model.py --exp-a results/exp_a.csv ${KEY_ARGS[@]+"${KEY_ARGS[@]}"}
