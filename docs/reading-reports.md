# Reading Ledger Reports

This guide explains how to read the markdown reports produced by `ledger.analyze`, `ledger.steps`,
and `ledger.diff`, which numbers signal real waste versus noise, and how to confirm a finding with
controlled experiments.

All concrete numbers below come directly from the DGX Spark (GB10, measured read ceiling 252 GB/s)
case studies in [`case-studies/gb10-qwen3.8-27b`](../case-studies/gb10-qwen3.8-27b) and
[`case-studies/gb10-qwen3-30b-a3b`](../case-studies/gb10-qwen3-30b-a3b).

---

## 1. `ledger.analyze` — Per-launch bandwidth and invariant checks

```sh
python -m ledger.analyze <capture> --ceiling-gbps <measured> [--peak-tflops bf16=...,fp8=...] [--out report.md]
```

### Header line

```
38880 kernel launches over 40 steps, 1779 memory regions, 560 launches without parameters. Bandwidth ceiling 252 GB/s.
```

Check **launches without parameters** first. Launches whose arguments could not be captured carry
no weight or cache attribution. On GB10 with vLLM 0.31, 560 of 38,880 launches (14 per step, all
small PyTorch kernels that stream no weight) had no parameters. Since the capture library's
`CUkernel` fix (`5f221cd`), GB10 with vLLM 0.34 has zero launches without parameters in both decode
and eager prefill. If a large fraction of launches is unresolved on another stack, the weight byte
totals below are under-counted.

### `## Findings`

Lists up to the five largest costs found across the rest of the report, ranked by **ms per step**
(items below 0.5 ms/step are omitted). Each entry is measured against its own reference:

- **Inter-kernel GPU idle** (`Step`), when idle is at least 1 ms/step or 2% of step wall time;
- **Slow weight-streaming groups** (`Weight-streaming launches by weight`), against the bandwidth
  floor — or, when `--peak-tflops` is passed, the larger of the bandwidth and compute floors
  (groups multiplying more than 64 rows without `--peak-tflops` are flagged with a note to pass
  peaks rather than judged against bandwidth alone);
- **Same-kernel, same-shape splits** (`Same kernel, same shape`), excess time of the slower weight
  group over the faster group on the same kernel and grid;
- **Repeated weight reads** (`Weights read more than once`), extra bytes per step priced at the
  bandwidth ceiling;
- **Underfilled launches** (`Launches with fewer blocks than SMs`), the idle SMs' share of
  alone time (`alone_ms * (1 - blocks / SMs)`) for launches not already near the bandwidth ceiling.

Treat each number in `## Findings` as an **upper bound on what fixing that item could save**, not a
guaranteed speedup; jump to the section named in parentheses for the evidence and caveats.

### `## Step`

Summarizes one step across the capture window:

- **wall time / GPU busy / idle between kernels**: GPU busy is the union of kernel intervals; idle
  is the gap between kernels inside the window. On GB10 Qwen3.8-27B (batch 1), wall time is
  120.88 ms, GPU busy is 120.52 ms, and idle between kernels is 0.36 ms (0.3%). When idle is under
  a percent, fusing kernels solely to eliminate launch gaps buys almost nothing.
- **weight bytes streamed / effective bandwidth / floor at ceiling**: `floor = weight bytes / ceiling`.
  In the same capture, 25.47 GB at 252 GB/s gives a 101.08 ms floor against 120.52 ms GPU busy
  (211 GB/s, 84% of ceiling). The gap between GPU busy and the floor is the maximum time you could
  recover without changing the bytes read.

### `## Weight-streaming launches by weight`

Groups launches streaming at least 1 MB by folded weight name (`layers.N`), shape, and dtype,
sorted by **lost ms/step** (time spent above the bandwidth floor).

**What to watch for:**
- Dense FP8 and NVFP4 GEMMs on GB10 typically reach **85–96%** of the 252 GB/s ceiling.
- A weight group sitting well below its peers (for example, `linear_attn.out_proj.weight` at
  **64%** of ceiling, losing **3.33 ms/step**, or MoE grouped GEMMs on Qwen3-30B-A3B at
  **60–66%** of ceiling) is the primary target.
- A low `% ceiling` alone does not tell you *why* a launch is slow. It can be:
  1. **Surrounding traffic** (dirty L2 write-back from the preceding kernel), which shows up in the
     next section;
  2. **The kernel itself** (as in Qwen3-30B-A3B, where DeepGEMM reads the exact 8 routed experts
     per layer at 60–66% of ceiling, and switching to vLLM's Triton backend cuts decode step time
     by 12.5% at batch 1 and 19.3% at batch 4); or
  3. **Compute-bound execution** at larger batches or prefill, which shows up in `## Compute`.

### `## Same kernel, same shape, different speed`

Flags any `(kernel, grid, weight shape, dtype)` tuple where two weight groups differ by **more than
15%** in median launch duration, and prints the most frequent cache-touching kernel among the three
launches immediately preceding each side.

In the GB10 Qwen3.8-27B capture, one CUTLASS FP8 kernel on `[6144, 5120]` splits cleanly:
- `linear_attn.out_proj.weight`: median **194.6 us** (1920 launches), preceded by
  `gdn_decode_post_conv_mtp_kernel` in 1920/1920 launches.
- `self_attn.o_proj.weight`: median **147.3 us** (640 launches), preceded by `kernel_mha` in
  640/640 launches.
- Excess: **47.3 us per launch**, **2.27 ms per step**.

Because the code, grid, shape, and dtype are identical, the slowdown belongs to the surroundings,
not the GEMM.

**How to confirm it:**
1. **Isolate the weights.** Run the exact same kernel on the slow and fast weights in isolation
   after flushing L2 with a read-only sweep (see `case-studies/gb10-qwen3.8-27b/gemm_isolation.py`).
   On GB10, all four tested weights (`out_proj` layers 0–1, `o_proj` layers 3 and 7) ran at an
   identical **147.5 us**.
2. **Inject dirty L2 traffic.** Write $N$ MB immediately before the isolated GEMM. On GB10,
   writing 0 / 2 / 4 / 8 / 12 / 24 MB slowed the GEMM from **147.5** to **158.7 / 169.0 / 187.4 /
   197.6 / 213.2 us**, saturating at the 24 MB L2 size and matching the 12.6 MB per-layer GDN state
   write.
3. **Perturb the suspect in-engine.** Halving the recurrent state size (`--mamba-ssm-cache-dtype
   bfloat16`) in alternating runs at a locked clock cut both GDN decode (41.2 → 21.6 us) and GDN
   `out_proj` (192 → 175 us) while leaving `lm_head` unchanged as a control.

### `## Weights read more than once per step`

Lists weights of at least 1 MB streamed more than 1.5 times per step, along with extra GB/step.

In Qwen3.8-27B with 3 MTP draft tokens, `lm_head` (1271.4 MB) is read once by the main model and
three times by the draft head — 4.0 reads/step (3.81 extra GB/step, 5.1 GB total, about **18%** of
the step). Draft-layer weights (`speculator.*`) are also read 3 times per step.

Beneath the table, **Partial reads** lists tensors referenced by launches too short to have read
the whole weight even at 1.3× the ceiling (such as `embed_tokens`, which gathers a few rows rather
than streaming the matrix). These are excluded from the byte totals.

### `## Launches with fewer blocks than SMs`

Shown when the SM count is known (`run.json` or `--sms`). Any kernel with `prod(grid) < SMs` leaves
SMs idle, but only costs the step when no other kernel runs beside it.

Read **alone ms/step** first (the table drops groups below 0.05 ms/step of alone time). A kernel on
a parallel CUDA graph branch can have a large total `ms/step` while its `alone ms/step` is near
zero. Also check the **GB/s** column: a bandwidth-bound kernel can still approach the memory
ceiling with fewer blocks than SMs, whereas a compute-bound or latency-bound one cannot.

### `## Compute`

Shown when `--peak-tflops` is passed (measured with `bench/flops.py`). For each weight group,
Ledger computes both a **memory floor** (`bytes / ceiling`) and a **compute floor**
(`2 * M * elements / peak`), where `M` is inferred from `run.json` for vLLM's layout. The larger
floor sets the `bound` column (`memory` or `compute`). Low-batch decode is almost entirely
`memory`-bound; prefill and large batches flip to `compute`-bound, where the bandwidth-only
`% ceiling` table above should not be used to judge kernel efficiency.

---

## 2. `ledger.steps` — Step cycle, speculation, KV attention, and clocks

```sh
python -m ledger.steps <capture> --ceiling-gbps <measured> [--out steps.md]
```

Requires step markers (`spin_kernel`) and `steps.jsonl`, both written by `examples/vllm_decode.py`.
Medians are taken over steps followed by another step so every step has a full cycle.

### `## Time per step`

Splits the **cycle** (first kernel of step $i$ to first kernel of step $i+1$) into three parts:
- **GPU busy**: kernel execution time inside the step.
- **idle between kernels in the step**: gaps inside the step's kernel span.
- **idle between steps (host)**: time the GPU sits empty waiting for host work (scheduling,
  sampling, output handling).

Compare **idle between steps (host)** against `host time of engine.step()`. Note that
`--no-async-scheduling` (used in the case studies to keep steps separated) exposes host work as
inter-step GPU idle that vLLM's default async scheduler hides (on Qwen3-30B batch 1: 25.3 ms vs
22.9 ms per step, with identical GPU busy time). If inter-step idle is large with async scheduling
enabled, the bottleneck is on the CPU, not the GPU.

### `## Speculative decoding`

Shown when `steps.jsonl` records gained tokens and the capture contains draft-model launches.
Key lines:
- **tokens per request** vs. `draft_tokens`: acceptance yield per step.
- **main model and sampling, GPU busy** vs. **draft steps, GPU busy**: how much of the step goes to
  drafting.
- **cycle per generated token** vs. **cycle per token without drafts, if the main pass cost the
  same**: the last line estimates non-speculative latency assuming a bandwidth-bound main pass
  costs roughly the same for 1 token as for $1+k$ tokens. If the actual cycle per generated token
  is not clearly below that estimate, drafting overhead or low acceptance is eating the gain.

### `## Attention over the KV cache`

Shown when `attention-*.json` is present. For each kernel touching a layer's KV cache view, Ledger
compares its time against one full read of K and V for the current context length.

Kernels whose implied bandwidth exceeds **1.3× the ceiling** are marked `(not a context read)` —
they write newly appended KV entries or read only a slice — and are excluded from the summary line.
For genuine context-reading kernels, watch `% ceiling` at long contexts (`--prompt-tokens N`): low
efficiency here points to attention kernel overhead or uncoalesced KV layout rather than weight
streaming.

### `## Clock and power`

Reports median, min, and max SM clock, power, and temperature across steps, and flags any step
running **3% or more below the median SM clock** along with how much slower its GPU busy time was.

**Caveat:** On GB10, NVML power readings are not usable as GPU power (they read a few watts under
full load), while SM clock and temperature are reliable. If slow steps correlate with clock drops,
lock clocks or alternate A/B runs before comparing two configurations.

---

## 3. `ledger.diff` — Comparing two captures past run-to-run drift

```sh
python -m ledger.diff <capture A> <capture B> --ceiling-gbps <measured> [--min-ms 0.05] [--tolerance 0.03]
```

On power-capped GPUs such as GB10, whole-step wall time can drift by 1–2 ms between runs — as large
as a real batch-1 optimization. `ledger.diff` separates real kernel changes from run-wide drift.

### Summary and Drift

The top table compares **GPU busy per step**, median **SM clock**, and median **power** between A
and B, followed by two diagnostic lines:
- **Drift**: the time-weighted median of `median_B / median_A` across all `(kernel, grid, weight)`
  groups present in both runs. A drift of, say, `+1.2%` means unchanged kernels ran 1.2% slower in
  B (clock or thermal/power-cap shift). Every kernel change below is measured *against* this drift,
  not against zero.
- **Generated tokens**: when both captures include `tokens.json`, reports `identical` or
  `different`. For a supposed bit-identical optimization (such as deferred GDN stores), `different`
  immediately catches a correctness regression or an unexpected eager fallback (eager and CUDA
  graph execution in vLLM can diverge in greedy tokens).

### `## Changed beyond drift`

Lists shared groups whose per-step time moved by at least `--min-ms` (default 0.05 ms) **and**
whose median ratio departed from the drift by more than `--tolerance` (default 3%).

- Groups whose template arguments changed (for example, switching recurrent state dtype from FP32
  to BF16) are paired automatically by short kernel name, grid, and streamed weight and tagged
  `(other variant)`.
- Read **beyond drift ms/step** and the summary total (`Together X ms per step beyond drift`).
  In an A/B test of a localized change, only the modified kernel and any downstream victim (such as
  GDN decode and GDN `out_proj`) should appear here; unrelated control projections (`o_proj`,
  `gate_up_proj`, `lm_head`) should stay within drift.

### `## In A only` and `## In B only`

Lists launch groups ($\ge$ `--min-ms` per step) that exist in only one capture. Expect entries here
when switching backends (for example, `deep_gemm` vs. `triton` in Qwen3-30B-A3B), changing graph
fusions (as between vLLM 0.31.0 and 0.34.0), or when one run unexpectedly fell back to eager
execution.

### `## Gate` (CI regression gate)

Pass `--fail-above <percent>` (and optionally `--fail-min-ms`, default `0.5` ms/step) to use
`ledger diff` as a CI check: it exits non-zero and lists the offending entries under `## Gate` when
overall GPU busy per step slows down by more than the threshold, or when any shared kernel group
regresses by more than the threshold beyond drift with at least `--fail-min-ms` extra ms/step. See
[`docs/ci-diff-gate.yml`](ci-diff-gate.yml) for a self-hosted GPU GitHub Actions example.
