# Ledger

Per-launch accounting for LLM inference on GPUs: who read what, who wrote what, and who really paid for it.

Profilers group time by kernel name. In a decode step the same GEMM kernel runs on dozens of
different weights, and the cost of one kernel's writes can surface as the next kernel's time.
Name-level totals average all of that away. Ledger keeps the books per launch instead:

- it records the parameters of every kernel the engine launches, including every kernel node of
  every CUDA graph, and the memory map of the engine's tensors (weights, KV and recurrent-state caches);
- it times each launch inside a steady-state window;
- it resolves each launch's pointer arguments against the memory map, so every launch knows which
  weight it streamed and which caches it touched;
- it compares each launch with the measured bandwidth ceiling and flags launches that should
  behave the same but do not.

Ledger is part of the [sec4si](https://sec4.si) research ecosystem.

## What it found on DGX Spark

On a DGX Spark (GB10) running Qwen3.8-27B NVFP4 with MTP speculative decoding in vLLM, the
output projection of the Gated DeltaNet layers took 186 us per launch while the attention output
projection, the same CUTLASS kernel on a weight of the same shape, took 138 us. In isolation the
two weights run at identical speed. The difference is the GDN decode kernel that runs just
before: with three draft tokens it writes a full 3.15 MB recurrent state for each of four
positions, 12.6 MB per layer, and those dirty lines are flushed from L2 while the projection
streams its weight. Only the state at the accepted position is ever read again.

A name-level profile shows "this GEMM is slow". Ledger shows whose bill it is. The full case
study, with the measurements and the scripts to reproduce them, is in
[`case-studies/gb10-qwen3.8-27b`](case-studies/gb10-qwen3.8-27b).

## Use

Build the capture library against the CUDA toolkit of the target machine:

```sh
make -C capture CUDA=/usr/local/cuda
```

Measure the device's read bandwidth ceiling once:

```sh
nvcc -O3 -arch=native -o bwtest bench/bwtest.cu && ./bwtest
```

Capture a steady-state decode window. For vLLM, `examples/vllm_decode.py` starts an in-process
engine with your arguments, exports the memory map and decodes inside
`cudaProfilerStart`/`cudaProfilerStop`:

```sh
LD_PRELOAD=$PWD/capture/libledger.so LEDGER_OUT=out \
  python examples/vllm_decode.py --steps 40 --batch 1 -- --model <model> [vLLM args...]
```

Any other CUDA program works the same way: preload the library, bracket the window you care about
with `cudaProfilerStart`/`cudaProfilerStop`, and write the memory map as `regions-<pid>.json`
(a list of `{kind, name, addr, size, shape, dtype}`; see `ledger/vllm_regions.py`).

Analyze:

```sh
python -m ledger.analyze out --ceiling-gbps <measured> --out report.md
```

## Limits

- Byte counts take each referenced weight as read once per launch and ignore activations and
  cache traffic. That is accurate for decode with a handful of tokens per step and wrong for
  prefill or large batches.
- Pointers are found by scanning parameter bytes for addresses inside known regions. Kernels that
  take pointers indirectly (pointer arrays in device memory) are invisible to it.
- Runtime-API launches from a library with a statically linked cudart can only be decoded when the
  same kernel also appears in a CUDA graph. In the GB10 case about 22 small eager launches per step
  stay opaque; one of them streams `lm_head`.
- Tested on NVIDIA GB10 (sm_121) with vLLM 0.31.

## Next

- Read and write sets per launch (binary instrumentation, e.g. NVBit) to find writes that are
  overwritten before anyone reads them, without knowing the model.
- Multi-GPU: account for collectives.

## License

Apache-2.0
