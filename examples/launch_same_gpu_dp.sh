#!/bin/bash
# Launch N serving replicas on ONE GPU behind a per-GPU CUDA MPS daemon.
# Companion to docs/basic_usage/same_gpu_dp.md.
#
# Usage:
#   MODEL=bosonai/higgs-tts-3-4b GPU_ID=0 N=2 CORE_BLOCKS="0-15 16-31" \
#     bash examples/launch_same_gpu_dp.sh up
#   bash examples/launch_same_gpu_dp.sh down
#
# Environment knobs (defaults in parentheses):
#   MODEL        model path (bosonai/higgs-tts-3-4b)
#   MODEL_NAME   --model-name value (higgs)
#   GPU_ID       physical GPU index (0)
#   N            replica count (2). DP2 is the recommended starting point.
#   MF           --mem-fraction-static per replica (0.42 for N=2, 0.27 for N=3)
#   BASE_PORT    first replica port (8801)
#   CORE_BLOCKS  space-separated CPU blocks, one per replica, NON-overlapping,
#                on the GPU's NUMA node (required for `up`)
# Keep benchmark/load-generator processes OFF these core blocks; on SMT hosts
# remember logical CPUs N and N+ncores may be the same physical core.
set -u

MODEL=${MODEL:-bosonai/higgs-tts-3-4b}
MODEL_NAME=${MODEL_NAME:-higgs}
GPU_ID=${GPU_ID:-0}
N=${N:-2}
BASE_PORT=${BASE_PORT:-8801}
if [ -z "${MF:-}" ]; then
  case "$N" in 2) MF=0.42 ;; 3) MF=0.27 ;; *) echo "set MF explicitly for N=$N" >&2; exit 1 ;; esac
fi
MPS_DIR=${MPS_DIR:-/tmp/mps-$USER-gpu$GPU_ID}
PID_FILE=${PID_FILE:-/tmp/same_gpu_dp_gpu$GPU_ID.pids}
CMD=${1:?usage: launch_same_gpu_dp.sh up|down}

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU_ID")
NODE=$(cat "/sys/class/drm/card$GPU_ID/device/numa_node" 2>/dev/null || echo 0)
[ "$NODE" -lt 0 ] && NODE=0

mps_env() {
  export CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPS_DIR/log
}

drain_gpu() {
  # Wait for every compute app on OUR GPU to exit. Stage workers are
  # `multiprocessing-fork` children, so drain by GPU compute-apps, not by name.
  local i
  for i in $(seq 1 40); do
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
      | grep -q "$GPU_UUID" || return 0
    sleep 3
  done
  return 1
}

down() {
  # 1) stop your traffic first, then TERM the tracked replica PIDs
  if [ -f "$PID_FILE" ]; then
    local p
    while read -r p; do kill -TERM "$p" 2>/dev/null; done < "$PID_FILE"
  else
    echo "note: $PID_FILE missing, falling back to name-based TERM" >&2
    pkill -TERM -f "sgl-omni serve.*--model-name $MODEL_NAME" 2>/dev/null
  fi
  # 2) wait until every compute app (incl. stage-worker children) left the GPU
  drain_gpu || echo "warning: GPU $GPU_ID did not drain within 120s" >&2
  # 3) only after the GPU drained, quit the daemon
  if [ -d "$MPS_DIR/pipe" ]; then
    mps_env
    echo quit | timeout 10 nvidia-cuda-mps-control 2>/dev/null
    rm -rf "$MPS_DIR"
  fi
  # 4) SIGKILL is the LAST fallback, only for stragglers still holding the GPU
  local q
  for q in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
             | grep "$GPU_UUID" | cut -d, -f1); do
    kill -9 "$q" 2>/dev/null
  done
  rm -f "$PID_FILE"
  echo "down: GPU $GPU_ID clean"
}

up() {
  # one MPS daemon per GPU, private pipe/log dirs
  mps_env
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d 2>/dev/null
  echo get_default_active_thread_percentage | nvidia-cuda-mps-control >/dev/null \
    || { echo "MPS daemon failed to start" >&2; exit 1; }

  if [ -z "${CORE_BLOCKS:-}" ]; then
    echo "CORE_BLOCKS unset; give each replica its own block from node $NODE cores:" >&2
    numactl -H | sed -n "s/^node $NODE cpus: //p" >&2
    exit 1
  fi
  local blocks=($CORE_BLOCKS)
  [ "${#blocks[@]}" = "$N" ] || { echo "CORE_BLOCKS must have $N blocks" >&2; exit 1; }

  : > "$PID_FILE"
  local i port
  for i in $(seq 0 $((N-1))); do
    port=$((BASE_PORT+i))
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    numactl --cpunodebind="$NODE" --membind="$NODE" -C "${blocks[$i]}" \
      sgl-omni serve --model-path "$MODEL" --model-name "$MODEL_NAME" \
        --mem-fraction-static "$MF" \
        --host 127.0.0.1 --port "$port" > "srv_$port.log" 2>&1 &
    echo $! >> "$PID_FILE"
    # sequential launch: wait for health before starting the next replica
    until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 127.0.0.1:$port/health)" = 200 ]; do
      sleep 6
    done
    echo "replica $i healthy on port $port (cores ${blocks[$i]}, mf $MF)"
  done

  # per-replica KV pools: fractions are NOT additive shares; later replicas can
  # receive much smaller pools. Confirm every replica got a workable pool.
  grep -H -m1 -oE '#tokens: [0-9]+' srv_88*.log || true

  # verify every replica actually attached as an MPS client (PID-level)
  local srv
  echo "MPS servers: $(echo get_server_list | nvidia-cuda-mps-control | tr '\n' ' ')"
  for srv in $(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null); do
    echo "MPS clients of $srv: $(echo "get_client_list $srv" | nvidia-cuda-mps-control | tr '\n' ' ')"
  done
  if [ "$(cat srv_88*.log 2>/dev/null | grep -c MpsRpc)" != 0 ]; then
    echo "warning: MpsRpc errors in replica logs; MPS attach is broken, restart from down" >&2
    exit 1
  fi
  echo "up: $N replicas on GPU $GPU_ID behind MPS ($MPS_DIR); PIDs in $PID_FILE"
}

case "$CMD" in
  up) up ;;
  down) down ;;
  *) echo "usage: launch_same_gpu_dp.sh up|down" >&2; exit 1 ;;
esac
