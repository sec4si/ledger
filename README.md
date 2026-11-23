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
  behave the same but do not;
- it traces which device memory each launch read and wrote, and follows data from writer to
  reader: bytes written and overwritten unread, hand-offs between neighbours, and, by poisoning
  writes in a replay, which of them the output actually depends on.

Ledger is part of the [sec4si](https://sec4.si) research ecosystem.

## What it found on DGX Spark

On a DGX Spark (GB10) running Qwen3.8-27B NVFP4 with MTP speculative decoding in vLLM, the
output projection of the Gated DeltaNet layers took 195 us per launch while the attention output
projection, the same CUTLASS kernel on a weight of the same shape, took 147 us. In isolation the
two weights run at identical speed. The difference is the GDN decode kernel that runs just
before: with three draft tokens it writes a full 3.15 MB recurrent state for each of four
positions, 12.6 MB per layer, and those dirty lines are flushed from L2 while the projection
streams its weight. Only the state at the accepted position is ever read again.

A name-level profile shows "this GEMM is slow". Ledger shows whose bill it is. Deferring the
stores to the next step is bit-identical and raises serving throughput by 5.2% / 7.5% at
concurrency 4 / 8 on that machine; it is submitted upstream as
[vllm-project/vllm#69160](https://github.com/vllm-project/vllm/pull/69160). The full case study,
with the measurements and the scripts to reproduce them, is in
[`case-studies/gb10-qwen3.8-27b`](case-studies/gb10-qwen3.8-27b).

On the same machine, Qwen3-30B-A3B FP8 (a mixture of experts) writes almost nothing for nothing, and its
expert GEMMs read exactly the eight routed experts per layer, but at 60-66% of the bandwidth ceiling.
vLLM picks DeepGEMM for block FP8 MoE on SM12x; its Triton backend is faster there, by 18-44% in serving
throughput at concurrency 1-8 with unchanged GSM8K accuracy. The change is proposed as
[vllm-project/vllm#69355](https://github.com/vllm-project/vllm/pull/69355); the case study is in
[`case-studies/gb10-qwen3-30b-a3b`](case-studies/gb10-qwen3-30b-a3b).

The access trace of the Qwen3.8 step finds 460 MB of its 712 MB of writes per step dead, but only
3.4 MB of that can simply be dropped: the GDN states are dead in every step, yet which of the four
depends on how many drafts were accepted, which is why the fix defers them rather than dropping them
([`dataflow.md`](case-studies/gb10-qwen3.8-27b/dataflow.md)). The books also put the three draft
reads of `lm_head` at 16.8 ms per step; drafting over the first 32,768 token ids in a replay saved
14.3 ms with unchanged greedy output (section 9 of the case study).

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

Programs that never call `cudaProfilerStart` can open the window by graph launch count instead:
`LEDGER_GRAPH_WINDOW=<skip>:<take>` opens it at the graph launch numbered `<skip>` and closes it
before the one numbered `<skip>+<take>`. Close it while the program is still running: kernels still
in flight when the window closes get bogus end times, so capture one step more than you need and
drop the last. CUPTI's worker threads do not survive `fork()`; for programs that fork one process
per GPU, set `LEDGER_INIT_IN_CHILD=1` so that each child starts CUPTI itself.

A [Sibyl](https://github.com/GeetoRinku/sibyl) replay keeps the recorded addresses, so the
recording's memory map applies to it unchanged:

```sh
python -m ledger.sibyl_regions <recording>/ranges-rank0-<pid>.json out
LD_PRELOAD=$PWD/capture/libledger.so LEDGER_OUT=out LEDGER_GRAPH_WINDOW=8:44 <replay program> ...
```

Analyze:

```sh
python -m ledger.analyze out --ceiling-gbps <measured> --out report.md
```

Memory copies in the window are kept the same way. Engines that hold part of the model off the GPU
(expert or KV offload, weight streaming) pay for it in copies, and a copy costs the step only the
part of it that no kernel runs alongside. `ledger.transfers` names the region each copy fills or
drains, sums bytes, time and exposed time per region and direction, flags pageable host memory and
compares bandwidth with the host-to-device ceilings that `bwtest` also prints:

```sh
python -m ledger.transfers out --h2d-gbps <measured> --d2h-gbps <measured> --out transfers.md
```

## Who read and wrote what

The access tracer (`access/`, Compute Sanitizer API) records, per launch inside the window, the
device memory it read and wrote at 32-byte resolution, with the PC of each access and the byte
ranges of copies and fills. It runs CUDA graphs node by node itself, and reads the tensor maps
among a launch's parameters, so TMA kernels are covered too. See [`access/README.md`](access/README.md)
for the build, the environment variables and the self-tests.

```sh
make -C access CUDA=/usr/local/cuda ARCH=<sm>
LD_PRELOAD=$PWD/access/libledger_access.so LEDGER_ACCESS_PATCHES=$PWD/access/patches.fatbin \
  LEDGER_OUT=acc LEDGER_ACCESS_SKIP=auto <program>          # window: the profiling range, as above
python -m ledger.dataflow acc --anchor <first kernel of a step> [--cubins <dir>] [--masks-out m.npz]
python -m ledger.dead_across acc*/m.npz                     # several traces: droppable or input-dependent
python -m ledger.sass <cubin dir> <function> <offset>       # the instruction behind a dead store
```

`ledger.dataflow` reports dead writes (by kernel, instruction and region), launches whose every
write is dead, hand-offs to the next launch only, constant inputs and independent neighbours.
`ledger.dead_across` splits dead writes over several traces into the same bytes dead every step,
as much dead every step but different bytes, and dead in some steps only.

Value experiments need a replay that can poison memory right after a given launch (Sibyl's
`replay_perturb`): `ledger.perturb_plan` turns a trace into experiments, one per tensor a launch
wrote, and `ledger.perturb_report` reads the results back. `ledger.traffic` joins a trace with an
Nsight Compute profile to compare what each launch needed with what it pulled from memory.

## Limits

- Byte counts take each referenced weight as read once per launch and ignore activations and
  cache traffic. That is accurate for decode with a handful of tokens per step and wrong for
  prefill or large batches.
- Pointers are found by scanning parameter bytes for addresses inside known regions. Kernels that
  take pointers indirectly (pointer arrays in device memory) are invisible to it.
- Some runtime-API launches reach the callback with neither a function handle nor a symbol name,
  and neither `cudaGetFuncBySymbol` nor `dladdr` on the host stub resolves them, so their
  parameters cannot be decoded (we suspect a statically linked cudart; not confirmed). In the GB10
  case 14 of the 45 eager launches per step stay opaque, all small PyTorch kernels.
- Tested on NVIDIA GB10 (sm_121) with vLLM 0.31 and 0.34.
- Copy addresses are decoded for linear copies (`cudaMemcpy[Async]`, `cuMemcpy[Async]`,
  `cuMemcpy{HtoD,DtoH,DtoD}[Async]` and their per-thread-stream variants) and for memcpy nodes of
  graphs. 2D/3D, peer and symbol copies are timed and counted but not attributed to a region.

- The access tracer finds what a tensor-map launch wrote by comparing its tensors before and after
  it (a write of the value already there does not show) and takes it to read every tensor its maps
  span; maps encoded for more rows than their tensor holds (DeepGEMM's) then read too much, which
  can hide dead writes but not invent them. Maps kept in device memory are not found. The tracer
  slows a decode step down 7 to 80 times on GB10 (`access/README.md`); the capture library by at
  most 7%.
- Dead writes are found for the inputs traced; a value experiment's verdict holds for the steps it
  compares (24 steps changed it for 0.3% of the bytes judged unused over 3) and was not reproducible
  for about 1 MB of 476 MB.

## Next

- Multi-GPU: account for collectives.

## License

Apache-2.0
