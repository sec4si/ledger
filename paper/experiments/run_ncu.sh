#!/bin/bash
# ncu on the isolated GEMM only (small process; never the whole engine). Usage (root): bash run_ncu.sh
set -uo pipefail
V=${VENV:?set VENV to the vLLM venv}; L=${LEDGER:?set LEDGER to the ledger checkout}; W=$L/paper/ncu
mkdir -p $W
export PATH=$V/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
echo "lock: $(nvidia-smi -lgc 2000,2000 2>&1 | tail -1)" > $W/ncu.log
timeout 600 /usr/local/cuda/bin/ncu --replay-mode application --cache-control none \
  --kernel-name regex:cutlass --kernel-name-base demangled \
  --metrics gpu__time_duration.sum,lts__t_sectors.sum,lts__d_sectors.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_write.sum \
  --csv --log-file $W/ncu.csv $V/bin/python $L/paper/experiments/ncu_isolation.py > $W/stdout.txt 2> $W/stderr.txt
echo "exit=$?" >> $W/ncu.log
echo "reset: $(nvidia-smi -rgc 2>&1 | tail -1)" >> $W/ncu.log
