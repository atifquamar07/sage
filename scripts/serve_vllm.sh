#!/usr/bin/env bash
# Start the vLLM servers SAGE needs on one GPU: the task model, xFinder and xVerify.
#
#   scripts/serve_vllm.sh                       # Qwen2.5-1.5B-Instruct + graders on GPU 0
#   scripts/serve_vllm.sh --task ministral      # Ministral-3-3B-Instruct-2512 (needs vLLM >= 0.13)
#   scripts/serve_vllm.sh --gpu 1 --no-graders  # task model only, on GPU 1
#   scripts/serve_vllm.sh --stop                # stop the servers started from this folder
#
# The ports match configs/models/*.yaml: task 8000, xFinder 8300, xVerify 8400.
# Revisions are pinned to the ones used in the paper. Logs go to logs/vllm_<port>.log.
set -euo pipefail

TASK=qwen
GPU=0
GRADERS=1
LOG_DIR=logs
VLLM=${VLLM:-vllm}
PID_FILE=""

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --task) TASK=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --no-graders) GRADERS=0; shift ;;
    --logs) LOG_DIR=$2; shift 2 ;;
    --stop) STOP=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/vllm.pids"

if [[ "${STOP:-0}" == 1 ]]; then
  [[ -f "$PID_FILE" ]] || { echo "no $PID_FILE"; exit 0; }
  xargs -r kill < "$PID_FILE" || true
  rm -f "$PID_FILE"
  echo "stopped"
  exit 0
fi

case "$TASK" in
  qwen)
    TASK_ARGS=(Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
               --served-model-name qwen2.5-1.5b-instruct --generation-config auto) ;;
  ministral)
    TASK_ARGS=(mistralai/Ministral-3-3B-Instruct-2512-BF16 --revision b6d637bef2393152b3da2b2fde72eecdee30557e
               --served-model-name ministral-3-3b-instruct-2512-bf16 --dtype half
               --config-format mistral --load-format mistral --tokenizer-mode mistral
               --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching) ;;
  *) echo "--task must be qwen or ministral" >&2; exit 1 ;;
esac

start() {  # port, vllm serve arguments...
  local port=$1; shift
  # Single-threaded CPU math, as in the paper runs: with per-request seeds, vLLM samples
  # on the CPU side and thread oversubscription slows generation by several times.
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    nohup "$VLLM" serve "$@" --host 127.0.0.1 --port "$port" \
    > "$LOG_DIR/vllm_${port}.log" 2>&1 &
  echo $! >> "$PID_FILE"
  echo "started $1 on port $port (pid $!, log $LOG_DIR/vllm_${port}.log)"
}

wait_ready() {  # port
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$1/v1/models" > /dev/null && { echo "port $1 ready"; return 0; }
    sleep 5
  done
  echo "port $1 did not become ready; see $LOG_DIR/vllm_$1.log" >&2
  return 1
}

# Servers start one after another: vLLM sizes its memory at startup.
start 8000 "${TASK_ARGS[@]}" --gpu-memory-utilization 0.72 --max-model-len 32768 --max-num-seqs 256
wait_ready 8000
if [[ "$GRADERS" == 1 ]]; then
  start 8300 IAAR-Shanghai/xFinder-qwen1505 --revision 74710b225ed6b7655701d0540d868edc5466e350 \
    --served-model-name xfinder-qwen1505 --generation-config auto \
    --gpu-memory-utilization 0.10 --max-model-len 8192 --max-num-seqs 512
  wait_ready 8300
  start 8400 IAAR-Shanghai/xVerify-0.5B-I --revision 7ddfe002f965f9474c524c17f331263604b8c2da \
    --served-model-name xverify-0.5b-i --generation-config auto \
    --gpu-memory-utilization 0.14 --max-model-len 8192 --max-num-seqs 512
  wait_ready 8400
fi
