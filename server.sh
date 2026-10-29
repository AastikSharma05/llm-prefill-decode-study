# Shared helpers for run_exp_a.sh and run_exp_b.sh. Not meant to be run on its own.
#
# start_server <log file> [extra vllm flags...]   starts vLLM (or the mock when MOCK=1)
# stop_server                                     stops it and waits until the GPU is free

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DTYPE=${DTYPE:-bfloat16}
PORT=${PORT:-8000}
SERVER=""

start_server() {
  local log=$1; shift
  if [[ "${MOCK:-0}" == "1" ]]; then
    # The mock understands the scheduling flags; drop the ones it does not need.
    local args=() a
    while (($#)); do
      a=$1; shift
      case "$a" in
        --max-num-batched-tokens|--long-prefill-token-threshold) args+=("$a" "$1"); shift ;;
        --enable-prefix-caching) args+=("$a") ;;
        *) ;;
      esac
    done
    setsid python mock_server.py --port "$PORT" ${args[@]+"${args[@]}"} > "$log" 2>&1 &
  else
    # setsid puts vLLM and its engine process in their own group, so stop_server can stop both.
    setsid vllm serve "$MODEL" --dtype "$DTYPE" --port "$PORT" \
      --max-model-len 20480 --max-num-seqs 64 \
      --gpu-memory-utilization 0.90 --generation-config vllm \
      "$@" > "$log" 2>&1 &
  fi
  SERVER=$!
  echo "starting server, log in $log (the very first start downloads the model, ~15 GB)"
  until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    if ! kill -0 "$SERVER" 2>/dev/null; then
      echo "server stopped. Last lines of $log:"; tail -n 30 "$log"; exit 1
    fi
    sleep 5
  done
  echo "server ready"
}

stop_server() {
  [[ -z "$SERVER" ]] && return 0
  kill -- -"$SERVER" 2>/dev/null || kill "$SERVER" 2>/dev/null || true
  # Give vLLM 30 s to shut down cleanly, then force it.
  for _ in $(seq 1 30); do kill -0 "$SERVER" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=""
  # Wait (up to 2 minutes) until GPU memory is released, so the next server start finds it free.
  if [[ "${MOCK:-0}" != "1" ]] && command -v nvidia-smi > /dev/null; then
    for _ in $(seq 1 60); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      (( ${used:-0} < 1000 )) && break
      sleep 2
    done
  fi
  sleep 2
}

start_gpu_log() {
  # Record GPU clock, power and temperature every 0.5 s, to spot throttling later.
  GPULOG_PID=""
  if [[ "${MOCK:-0}" != "1" ]] && command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu \
      --format=csv -lms 500 > "$1" 2>/dev/null &
    GPULOG_PID=$!
  fi
}

stop_gpu_log() {
  [[ -n "${GPULOG_PID:-}" ]] && kill "$GPULOG_PID" 2>/dev/null || true
}

record_env() {
  # Write down exactly what the numbers were measured on (for the README "Setup" section).
  {
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "model: $MODEL  dtype: $DTYPE  mock: ${MOCK:-0}"
    if command -v nvidia-smi > /dev/null; then
      nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
    fi
    python - <<'EOF' 2>/dev/null || true
import platform
print("python:", platform.python_version())
for name in ("vllm", "torch", "transformers"):
    try:
        mod = __import__(name)
        extra = f" (CUDA {mod.version.cuda})" if name == "torch" else ""
        print(f"{name}: {mod.__version__}{extra}")
    except Exception:
        print(f"{name}: not installed")
EOF
  } > "$1"
  echo "environment written to $1"
}

save_server_log() {
  # logs/ is not committed; keep a copy of the server log (it prints vLLM's effective settings).
  mkdir -p results/server_logs
  cp "$1" results/server_logs/ 2>/dev/null || true
}

trap 'stop_gpu_log; stop_server' EXIT
