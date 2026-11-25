# Ledger

Per-launch accounting for LLM inference on GPUs: who read what, who wrote what, and who really paid for it.

Profilers group time by kernel name. In a decode step the same GEMM kernel runs on dozens of
different weights, and the cost of one kernel's writes can surface as the next kernel's time.
Name-level totals average all of that away. Ledger records every kernel launch (including every
kernel node of every CUDA graph), resolves its pointer arguments against the engine's memory map
(weights, KV and recurrent-state caches), compares each launch against measured hardware ceilings,
and can trace byte-level reads, writes, and dead stores across launches.

Ledger is part of the [sec4si](https://sec4.si) research ecosystem.

## Findings on DGX Spark (GB10)

- **Cross-kernel dirty-L2 write-back in GDN speculative decoding:** On Qwen3.8-27B NVFP4 with MTP
  (`k=3`), GDN `out_proj` took 195 us while attention `o_proj` — the same CUTLASS kernel on the
  same weight shape — took 147 us, because the preceding GDN decode kernel wrote 12.6 MB/layer of
  recurrent state that flushed from L2 during `out_proj`. Deferring the stores is bit-identical and
  raises serving throughput by 5.2% / 7.5% at concurrency 4 / 8 (submitted as
  [vllm#69160](https://github.com/vllm-project/vllm/pull/69160);
  [case study](case-studies/gb10-qwen3.8-27b)).
- **Slow MoE backend selection on SM12x:** On Qwen3-30B-A3B FP8, Ledger showed almost no wasted
  writes and exact 8-expert reads per layer, but DeepGEMM ran at only 60–66% of the 252 GB/s
  ceiling. Preferring vLLM's Triton backend raises serving throughput by 18–44% at concurrency 1–8
  with unchanged GSM8K accuracy (submitted as
  [vllm#69355](https://github.com/vllm-project/vllm/pull/69355);
  [case study](case-studies/gb10-qwen3-30b-a3b)).
- **GEMMs that outgrow the L2:** On GB10, FP8 and NVFP4 GEMMs whose A operand is larger than about
  12 MiB run 2-3x slower (an 8192-row FP8 GEMM at 55 against 170 TFLOP/s), which the compute
  ceilings exposed in prefill. Running them as row slices is bit-identical and cuts the latency of an
  8192-token prompt on Qwen3.8-27B by 42% (submitted as
  [vllm#69595](https://github.com/vllm-project/vllm/pull/69595)).
- **Dead writes vs. input-dependent state, and repeated `lm_head` reads:** Access tracing of the
  Qwen3.8 step found 460 MB of 712 MB written per step dead, of which only 3.4 MB is fixed and
  droppable while the rest depends on draft acceptance
  ([dataflow.md](case-studies/gb10-qwen3.8-27b/dataflow.md)); the books also put the three draft
  `lm_head` reads at 16.8 ms/step, where drafting over the first 32,768 token IDs in replay saved
  14.3 ms with unchanged greedy output
  ([section 9](case-studies/gb10-qwen3.8-27b/README.md)).

## Quickstart

In the Python environment with `vllm` and CUDA `torch`:

```sh
pip install -e .                                      # once published: pip install sec4si-ledger
ledger build
ledger ceilings                                       # run with the GPU idle
ledger profile --steps 20 --out run1 -- --model <model> [vLLM args...]
```

Open `run1/report.md` starting at `## Findings`, load `run1/trace.json` in
[ui.perfetto.dev](https://ui.perfetto.dev), and compare two runs with `ledger diff run1 run2`.
See [docs/quickstart.md](docs/quickstart.md) for the full walkthrough (including `ledger report`,
`vllm serve` plugin usage, non-vLLM CUDA programs, and Sibyl replays).

## Tools

| command | what it does |
|---|---|
| `ledger build` | Compile `capture/libledger.so` (and with `--access`, the Compute Sanitizer tracer) against the local CUDA toolkit. |
| `ledger ceilings` | Measure and cache GPU DRAM read bandwidth (`bench/bwtest.cu`) and GEMM TFLOP/s (`bench/flops.py`). |
| `ledger profile` | Capture a vLLM run under CUPTI and write `<out>/report.md` (`analyze` + `steps`) and `<out>/trace.json`. |
| `ledger report` | Re-generate the combined `analyze` + `steps` Markdown report from an existing capture directory. |
| `ledger analyze` | Per-launch bandwidth, compute floor, same-kernel/same-shape splits, repeated reads, underfilled grids, and top `## Findings`. |
| `ledger steps` | Step-to-step cycle split (GPU busy, intra-step idle, host idle), speculative yield, KV-cache attention bandwidth, and SM clocks. |
| `ledger diff` | Compare two captures launch group by launch group past run-to-run clock/power-cap drift, plus token equality and `--fail-above` CI gating ([example workflow](docs/ci-diff-gate.yml)). |
| `ledger perfetto` | Export a capture to Chrome Trace Event JSON (`Steps`, per-CUDA `Stream`, and `Memcpy` tracks) for [ui.perfetto.dev](https://ui.perfetto.dev). |
| `ledger transfers` | Account for H2D/D2H/D2D memory copies by region, direction, exposed time, and pageable host memory. |
| `ledger dataflow` | Follow byte-level reads and writes from an access trace (`access/`) to find dead writes, hand-offs, and constant inputs. |
| `ledger dead-across` | Split dead writes across multiple traces into fixed (droppable), moving (input-dependent), and varying. |
| `ledger traffic` | Join an access trace with an Nsight Compute profile to compare bytes needed against bytes pulled into L2. |
| `ledger sass` | Look up the SASS instruction behind a dead store from a cubin directory, function, and PC offset. |
| `ledger perturb-plan` / `perturb-report` | Generate and summarize replay memory-poisoning experiments (e.g. Sibyl `replay_perturb`). |
| `ledger sibyl-regions` | Convert a Sibyl recording's address ranges into Ledger's `regions-*.json` memory map. |

## Documentation

- [Quickstart](docs/quickstart.md) — install, build, measure ceilings, profile vLLM, Perfetto export, `vllm serve` plugin, and non-vLLM programs.
- [Reading Ledger Reports](docs/reading-reports.md) — section-by-section guide to `analyze`, `steps`, and `diff`, plus how to confirm findings.
- [Access Tracer (`access/README.md`)](access/README.md) — Compute Sanitizer build, environment variables, and self-tests.
- [Case Study: Qwen3.8-27B NVFP4 on GB10](case-studies/gb10-qwen3.8-27b) — GDN dirty-L2 write-back and deferred stores.
- [Case Study: Qwen3-30B-A3B FP8 MoE on GB10](case-studies/gb10-qwen3-30b-a3b) — DeepGEMM vs. Triton backend selection.

## Limits

- Byte counts take each referenced weight as read once per launch, and add the activations of
  linear layers (the M x K input once, in the weight's precision, and the M x N BF16 output) when
  the regions record the layer's widths (the vLLM exporter does). MoE experts, embeddings and
  attention are counted by weight only; `ledger steps` judges attention against its KV reads. The
  count is the least traffic a launch needs: on GB10, ncu measured output writes within 1% of it,
  and reads within 10% for GEMMs whose operands fit in L2 but 2.5 to 7 times it for prefill-sized
  ones, which reread tiles of A and the weight.
- Pointers are found by scanning parameter bytes for addresses inside known regions. Kernels that
  take pointers indirectly (pointer arrays in device memory) are invisible to it.
- A runtime that loads modules context-independently launches through the driver with a `CUkernel`
  in the function slot; the capture library resolves it to the context's function. Launches it
  still cannot name carry no parameters; on GB10 with vLLM 0.34 there were none, in decode or in an
  eager 2048-token prefill.
- Idle time between and inside steps is the engine's as configured: with
  `--no-async-scheduling` (used for the case studies, to keep steps apart) host work shows up as GPU
  idle that the default asynchronous scheduler hides (Qwen3-30B batch 1: 25.3 against 22.9 ms per
  step, the same GPU busy time).
- NVML power readings on GB10 are not usable as GPU power (a few watts under full load); the SM
  clock and temperature are.
- Tested on NVIDIA GB10 (sm_121) with vLLM 0.31, 0.34 and main.
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
