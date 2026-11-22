#!/bin/bash
# Capture overhead: alternate runs with and without the capture library, clock locked at 2000 MHz.
# Usage (root; VENV, LEDGER and MODEL set): bash run_overhead.sh <rounds>
set -uo pipefail
V=${VENV:?set VENV to the vLLM venv}; L=${LEDGER:?set LEDGER to the ledger checkout}; W=$L/paper/overhead
M=${MODEL:?set MODEL to the Qwen3.8-27B-NVFP4 path}
mkdir -p $W; LOG=$W/overhead.log
export PATH=$V/bin:/usr/local/cuda/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
cd $L
run() {  # $1 tag, $2 preload (empty = none)
  local O=$W/$1; rm -rf $O; mkdir -p $O
  LEDGER_OUT=$O LD_PRELOAD=$2 $V/bin/python examples/vllm_decode.py --steps 40 --batch 1 -- \
    --model $M --trust-remote-code --dtype bfloat16 --max-model-len 131072 --max-num-seqs 4 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --no-async-scheduling \
    --speculative-config "{\"method\": \"qwen3_5_mtp\", \"model\": \"$M\", \"num_speculative_tokens\": 3}" \
    > $O/run.log 2>&1
  echo "$1 exit=$? $(grep LEDGER_DECODE $O/run.log)" >> $LOG
}
echo "lock: $(nvidia-smi -lgc 2000,2000 2>&1 | tail -1)" > $LOG
for r in $(seq 1 ${1:-3}); do
  run off-r$r ""
  run on-r$r $L/capture/libledger.so
done
echo "reset: $(nvidia-smi -rgc 2>&1 | tail -1)" >> $LOG
echo OVERHEAD_DONE >> $LOG
