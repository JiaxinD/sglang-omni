#!/bin/bash
# Launch N serving replicas on ONE GPU behind a private CUDA MPS daemon.
# Companion to docs/basic_usage/same_gpu_dp.md.
# This is a tested example, not a production process supervisor.
#
# Usage:
#   MODEL=bosonai/higgs-tts-3-4b GPU_ID=0 N=2 CORE_BLOCKS="0-15 16-31" \
#     bash examples/launch_same_gpu_dp.sh up
#   bash examples/launch_same_gpu_dp.sh list
#   bash examples/launch_same_gpu_dp.sh verify [RUN_ID]
#   bash examples/launch_same_gpu_dp.sh down [RUN_ID]
#
# Environment for `up` (defaults in parentheses):
#   MODEL (bosonai/higgs-tts-3-4b), MODEL_NAME (higgs), GPU_ID (0), N (2),
#   MF (0.42 for N=2, 0.27 for N=3), BASE_PORT (8801),
#   CORE_BLOCKS: N non-overlapping CPU blocks on the GPU's NUMA node, required.
set -euo pipefail

STATE_ROOT=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$USER}
CMD=${1:-}
RUN_ARG=${2:-}
HEALTH_TRIES=${HEALTH_TRIES:-50}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-6}

die() { echo "error: $*" >&2; exit 1; }

mps_env() { # $1 = state dir
  export CUDA_MPS_PIPE_DIRECTORY=$1/mps/pipe CUDA_MPS_LOG_DIRECTORY=$1/mps/log
}

mps_ctl() { # $1 = state dir, stdin = command
  CUDA_MPS_PIPE_DIRECTORY=$1/mps/pipe CUDA_MPS_LOG_DIRECTORY=$1/mps/log \
    timeout 10 nvidia-cuda-mps-control 2>/dev/null || true
}

resolve_numa() { # $1 = GPU_ID -> echoes node
  # Note (jiaxin): /sys/class/drm ordinals are not guaranteed to match nvidia-smi
  # ordinals, so the NUMA node is derived from the GPU's PCI bus id instead.
  local bus node
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1")
  bus=${bus,,}; bus=${bus:4}
  node=$(cat "/sys/bus/pci/devices/$bus/numa_node" 2>/dev/null || echo "")
  [ -n "$node" ] || die "cannot resolve NUMA node for GPU $1 (pci $bus)"
  [ "$node" -ge 0 ] || { echo "warning: NUMA node unknown for GPU $1, using 0" >&2; node=0; }
  echo "$node"
}

find_runs() { ls -d "$STATE_ROOT"/gpu-*/run-* 2>/dev/null || true; }

resolve_state() { # $1 = optional RUN_ID or path -> echoes state dir
  local runs
  if [ -n "$1" ]; then
    if [ -d "$1" ]; then echo "$1"; return 0; fi
    runs=$(find_runs | grep -F "/$1" || true)
    [ -n "$runs" ] || die "no state found for run '$1' under $STATE_ROOT"
    echo "$runs" | head -1
    return 0
  fi
  runs=$(find_runs)
  if [ -z "$runs" ]; then
    echo "No launcher state found under $STATE_ROOT — refusing to guess." >&2
    echo "Inspect manually before signalling anything:" >&2
    echo "  nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv" >&2
    echo "  ps -o pid,pgid,cmd -p <pid>" >&2
    exit 1
  fi
  if [ "$(echo "$runs" | wc -l)" -gt 1 ]; then
    echo "Multiple runs found; pass a RUN_ID:" >&2
    echo "$runs" >&2
    exit 1
  fi
  echo "$runs"
}

tracked_pids() { # $1 = state dir -> echoes live pids of this run's process groups
  local pgid out=""
  while IFS=$'\t' read -r _ _ pgid _ _; do
    out+=" $(pgrep -g "$pgid" 2>/dev/null || true)"
  done < "$1/replicas.tsv"
  echo "$out"
}

verify_attach() { # $1 = state dir -> 0 iff every replica has an attached MPS client
  local state=$1 servers clients all=" " idx pid pgid port log ok=0
  [ -n "$state" ] && [ -f "$state/replicas.tsv" ] || die "invalid or missing run state '$state'"
  servers=$(echo get_server_list | mps_ctl "$state" | grep -E '^[0-9]+$' || true)
  local art="$state/mps_attach.txt"
  : > "$art"
  if [ -z "$servers" ]; then
    echo "attach verification FAILED: no MPS server under $state/mps/pipe" | tee -a "$art" >&2
    return 1
  fi
  for s in $servers; do
    clients=$(echo "get_client_list $s" | mps_ctl "$state" | grep -E '^[0-9]+$' || true)
    echo "server $s clients: $clients" >> "$art"
    all+="$(echo "$clients" | tr '\n' ' ') "
  done
  while IFS=$'\t' read -r idx pid pgid port log; do
    local expected matched=""
    expected=$(pgrep -g "$pgid" 2>/dev/null || true)
    for p in $expected; do
      case "$all" in *" $p "*) matched+="$p ";; esac
    done
    echo "replica $idx port $port pgid $pgid expected: $(echo $expected) matched: ${matched:-NONE}" >> "$art"
    if [ -z "$matched" ]; then
      echo "attach verification FAILED: replica $idx (port $port) has no process in the MPS client list" >&2
      ok=1
    fi
  done < "$state/replicas.tsv"
  echo "attach mapping written to $art"
  return $ok
}

teardown_state() { # $1 = state dir; touches only PIDs recorded in this run's manifest
  # Note (jiaxin): these GPUs are shared; teardown must never signal processes that
  # are not registered in this run's state, and must not treat "GPU empty" as done.
  local state=$1 pgid t live
  [ -n "$state" ] && [ -f "$state/replicas.tsv" ] || die "invalid or missing run state '$state'"
  while IFS=$'\t' read -r _ _ pgid _ _; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done < "$state/replicas.tsv"
  for ((t=1; t<=40; t++)); do
    live=$(tracked_pids "$state")
    [ -z "${live// /}" ] && break
    sleep 3
  done
  # Note (jiaxin): live MPS clients must be gone before the daemon quits; quitting
  # (or SIGKILLing) around live clients can wedge the MPS server with RPC failures
  # that outlast this run.
  echo quit | mps_ctl "$state" > /dev/null
  live=$(tracked_pids "$state")
  if [ -n "${live// /}" ]; then
    echo "warning: tracked processes survived TERM + drain; last-resort SIGKILL on tracked groups only" >&2
    while IFS=$'\t' read -r _ _ pgid _ _; do
      kill -KILL -- "-$pgid" 2>/dev/null || true
    done < "$state/replicas.tsv"
  fi
  live=$(tracked_pids "$state")
  if [ -n "${live// /}" ]; then
    echo "error: tracked pids still alive:$live — inspect manually (ps -o pid,pgid,cmd -p ...)" >&2
    return 1
  fi
  rm -rf "$state"
  echo "down: run state $state cleaned; only this run's processes were touched"
}

up() {
  local model=${MODEL:-bosonai/higgs-tts-3-4b} model_name=${MODEL_NAME:-higgs}
  local gpu=${GPU_ID:-0} n=${N:-2} base_port=${BASE_PORT:-8801} mf=${MF:-}
  if [ -z "$mf" ]; then
    case "$n" in 2) mf=0.42 ;; 3) mf=0.27 ;; *) die "set MF explicitly for N=$n" ;; esac
  fi
  [ -n "${CORE_BLOCKS:-}" ] || {
    echo "CORE_BLOCKS is required: N non-overlapping blocks on the GPU's NUMA node." >&2
    echo "Cores on that node: numactl -H" >&2
    exit 1
  }
  local blocks=($CORE_BLOCKS)
  [ "${#blocks[@]}" = "$n" ] || die "CORE_BLOCKS must contain exactly $n blocks"

  local uuid node run state
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")
  node=$(resolve_numa "$gpu")
  run="run-$(date +%Y%m%d-%H%M%S)-$$"
  state=$STATE_ROOT/gpu-$gpu/$run
  mkdir -p "$state/logs" "$state/mps/pipe" "$state/mps/log"
  {
    echo "run_id=$run"; echo "gpu_id=$gpu"; echo "gpu_uuid=$uuid"; echo "numa_node=$node"
    echo "model=$model"; echo "model_name=$model_name"; echo "n=$n"; echo "mf=$mf"
    echo "base_port=$base_port"; echo "core_blocks=$CORE_BLOCKS"
  } > "$state/manifest"
  : > "$state/replicas.tsv"

  local up_done=0
  trap '[ "$up_done" = 1 ] || { echo "startup failed; cleaning up this run only" >&2; teardown_state "'"$state"'" || true; }' EXIT

  mps_env "$state"
  nvidia-cuda-mps-control -d 2>/dev/null || true
  [ -n "$(echo get_default_active_thread_percentage | mps_ctl "$state")" ] \
    || die "MPS control daemon did not start (pipe $state/mps/pipe)"

  local i port pid log
  for ((i=0; i<n; i++)); do
    port=$((base_port+i))
    log=$state/logs/replica_$i.log
    # Note (jiaxin): concurrent colocated launches raced on CUDA-graph capture and
    # memory profiling in testing, so replicas start sequentially behind a health
    # gate; setsid gives each replica its own process group so teardown can signal
    # exactly this run's process trees.
    CUDA_VISIBLE_DEVICES=$gpu \
    setsid numactl --cpunodebind="$node" --membind="$node" -C "${blocks[$i]}" \
      sgl-omni serve --model-path "$model" --model-name "$model_name" \
        --mem-fraction-static "$mf" \
        --host 127.0.0.1 --port "$port" > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$pid" "$pid" "$port" "$log" >> "$state/replicas.tsv"
    local healthy=0 t code
    for ((t=1; t<=HEALTH_TRIES; t++)); do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "replica $i exited during startup; last log lines:" >&2
        tail -n 8 "$log" >&2
        exit 1
      fi
      code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "127.0.0.1:$port/health" || true)
      [ "$code" = 200 ] && { healthy=1; break; }
      sleep "$HEALTH_INTERVAL"
    done
    if [ "$healthy" != 1 ]; then
      echo "replica $i health timeout after $((HEALTH_TRIES*HEALTH_INTERVAL))s; last log lines:" >&2
      tail -n 8 "$log" >&2
      exit 1
    fi
    echo "replica $i healthy on port $port (cores ${blocks[$i]}, mf $mf)"
    # Note (jiaxin): per-replica KV pools are not additive shares of the device;
    # later replicas can receive much smaller pools, so surface each allocation.
    grep -m1 -oE '#tokens: [0-9]+' "$log" | sed "s/^/replica $i KV /" || true
  done

  verify_attach "$state" || exit 1
  if [ "$(cat "$state"/logs/replica_*.log 2>/dev/null | grep -c MpsRpc)" != 0 ]; then
    echo "warning: MpsRpc errors present in replica logs; restart from 'down'" >&2
    exit 1
  fi
  up_done=1
  trap - EXIT
  echo "up: $n replicas on GPU $gpu; state: $state"
  echo "tear down with: bash $0 down $run"
}

case "$CMD" in
  up) up ;;
  down) st=$(resolve_state "$RUN_ARG") || exit 1; teardown_state "$st" ;;
  verify) st=$(resolve_state "$RUN_ARG") || exit 1; verify_attach "$st" ;;
  list) find_runs ;;
  *) die "usage: launch_same_gpu_dp.sh up|down [RUN_ID]|verify [RUN_ID]|list" ;;
esac
