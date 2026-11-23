# Case study: decode on DGX Spark, Qwen3.8-27B NVFP4 with MTP

Where does a decode step go on a DGX Spark, and which of it can be bought back?

## Setup

| | |
|---|---|
| Machine | NVIDIA DGX Spark: GB10 (sm_121, 48 SMs, 24 MB L2), 121 GB unified LPDDR5x |
| Engine | vLLM 0.31.0 (main measurements) and 0.34.0 (comparison), CUDA graphs, no async scheduling |
| Model | Qwen3.8-27B, NVFP4 checkpoint (MLP NVFP4 in layers 0-55, FP8 elsewhere) |
| Speculation | `qwen3_5_mtp`, 3 draft tokens: 4 tokens through the main model per step |
| Load | batch 1, 40 steady-state steps |

## 1. The ceiling

`bench/bwtest.cu` and [BabelStream](https://github.com/UoB-HPC/BabelStream) agree:

| | GB/s |
|---|---|
| read-only kernel (`bwtest`) | 253.5 |
| BabelStream Dot (double) | 253.0 |
| BabelStream Copy / Mul / Add / Triad | 250.4 / 252.0 / 248.8 / 248.3 |
| `cudaMemcpy` device to device | 243.0 |
| spec sheet | 273 |

All percentages below use 252 GB/s.

## 2. The step

Ledger, batch 1 (`report-031.md` has the full output, `results.md` the raw numbers of every experiment below):

| | per step |
|---|---|
| wall time | 120.9 ms |
| GPU busy | 120.5 ms |
| idle between kernels | 0.36 ms (0.3%) |
| launches | 972 |
| weight bytes streamed | 25.5 GB |
| effective bandwidth | 211 GB/s, 84% of the ceiling |
| floor at the ceiling | 101 ms |

The step is bandwidth-bound end to end, and idle time between kernels is negligible: fusing
kernels to remove launch gaps buys almost nothing on this machine. Batch 2 costs the same as
batch 1 within noise, as expected when every step streams the full weight set.

## 3. Where the bandwidth goes

Most weight-streaming launches run at 85 to 96% of the ceiling. Two items stand out:

- **`lm_head` is read four times per step.** The main model reads it once and each of the three
  MTP draft steps reads the full 248k-vocabulary head again: 5.1 GB, about 18% of the step.
  Section 9.
- **GDN `out_proj` runs at 64%**, far below its neighbours. Section 4.

## 4. Same kernel, same shape, different speed

Ledger groups launches by kernel, grid and weight shape. One group splits:

| weight | launches/step | median |
|---|---|---|
| `linear_attn.out_proj` (Gated DeltaNet layers) | 48 | 194 us |
| `self_attn.o_proj` (attention layers) | 16 | 148 us |

Same CUTLASS FP8 kernel, same grid, weights of the same shape [6144, 5120], both 4 KB aligned.
The last kernel touching caches before every slow launch is the GDN decode kernel; before every
fast launch it is attention.

`gemm_isolation.py` separates the weight from the surroundings:

| experiment | GEMM time |
|---|---|
| out_proj layer 0, isolated, L2 flushed by a read | 147.5 us |
| out_proj layer 1, isolated | 147.5 us |
| o_proj layer 3, isolated | 147.5 us |
| o_proj layer 7, isolated | 147.5 us |
| o_proj after writing 2 MB | 158.7 us |
| ... 4 MB | 169.0 us |
| ... 8 MB | 187.4 us |
| ... 12 MB | 197.6 us |
| ... 24 MB (L2 size) | 213.2 us |

The weights are not the cause; dirty lines left in L2 are. The GDN decode kernel in vLLM
(`csrc/gdn/fused_gdn_decode_kernel.cu`) writes the full recurrent state after every token of
the step into a separate slot, so speculative tokens can be rolled back:
48 heads x 128 x 128 x fp32 = 3.15 MB per position, 12.6 MB per layer for four positions. The
kernel finishes quickly because the stores land in L2; `out_proj` then pays for the write-back.
The next step reads only the slot of the last accepted position.

Halving the state size confirms the mechanism. Alternating runs at a locked 2000 MHz clock,
fp32 state (A) against `--mamba-ssm-cache-dtype bfloat16` (B):

| run | GDN decode kernel | GDN out_proj | lm_head (control) |
|---|---|---|---|
| A1 | 41.2 us | 192 us | 5422 us |
| B1 | 21.6 us | 175 us | 5431 us |
| A2 | 40.6 us | 190 us | 5505 us |
| B2 | 21.7 us | 176 us | 5538 us |

Both kernels move by the predicted amount in every pair, about 1.7 ms per step together. Whole-step
times drift by as much as the effect (the GB10 sits at its software power cap; see the control
column), so the per-kernel books are the measurement. BF16 state is a probe, not a fix: it
changes numerics.

## 5. Removing the stores

To measure what the fix would buy, the vLLM 0.31.0 kernel source was built as a standalone
extension (bitwise identical to the shipped kernel when compiled with `--use_fast_math`) with a
switch that keeps only the last position's store, and swapped into the engine before graph
capture. That writes exactly as much state as a real fix would. It is a performance probe only:
after a rejected draft the state is wrong.

Alternating runs at a locked 2000 MHz clock, 40 steps each, GPU busy time per step:

| batch | all positions stored | last position only | saving |
|---|---|---|---|
| 1 | 120.9 / 121.0 ms | 119.7 ms | 1.2 to 1.4 ms (1%) |
| 4 | 131.6 / 131.9 ms | 122.5 / 123.0 ms | **9.0 ms (6.8%)**, both pairs |

| kernel, median | batch 1 all | batch 1 last | batch 4 all | batch 4 last |
|---|---|---|---|---|
| GDN decode | 40.9 us | 23.3 us | 267.6 us | 83.7 us |
| GDN out_proj | 194.7 us | 162.7 us | 199.8 us | 192.5 us |
| o_proj (control) | 149.0 us | 149.3 us | 150.2 us | 150.1 us |

At batch 1 the stores fit in L2 and the cost lands on the next GEMM; at batch 4 they are 50 MB
per layer, twice the L2, and the GDN kernel itself stalls on DRAM writes. The state traffic grows
with the batch while weight traffic does not, so the saving grows with load. (At batch 1 the
kernel-level saving is 2.4 ms per step but the step only gained 1.2 to 1.4 ms: in that run the
MLP gate_up GEMMs were 4% slower in every layer, GDN or not, consistent with power-cap drift. A
fourth batch-1 run fell back to eager execution and is excluded; the cause is unknown, and an
unmodified baseline run did the same.)

## 6. Deferring the stores

The Triton GDN kernels already have a fix in review upstream (vLLM PR #61434, #64325): store the
state after the first token in slot 0, record the inputs of the later tokens in slot 1, and replay
the accepted ones at the start of the next step. `gdn-probe` implements the same layout in the
fused CUDA kernel (store mode 2). The replay runs the same per-token update, so it is
bit-identical to the shipped kernel: every output and the slot-0 state match over ten steps of
random token and accepted counts (`test_defer.py`), and greedy decoding produces the same 448
tokens per request in every run below.

Alternating runs at 2000 MHz, GPU busy time per step:

| batch | per-token stores | deferred | saving | GDN kernel |
|---|---|---|---|---|
| 1 | 120.90 / 121.04 ms | 118.71 / 119.10 ms | 2.1 ms (1.7%) | 40.7 -> 27.4 us |
| 4 | 132.27 ms | 123.50 / 123.72 ms | 8.7 ms (6.5%) | 270 -> 97 us |

The replay costs about 4 us per GDN launch at batch 1 over the store-last-only probe. One batch-4
baseline run fell back to eager execution and is excluded; its tokens also differ from the graph
runs, as eager and graph execution do in vLLM regardless of this change.

## 7. Upstream

Reported as [vllm-project/vllm#69152](https://github.com/vllm-project/vllm/issues/69152) and a fix submitted in
[vllm-project/vllm#69160](https://github.com/vllm-project/vllm/pull/69160), which ports the
deferred stores to vLLM main on top of #61434. On main the CUDA kernel writes its records in the
Triton kernel's layout (one record per 16-row V block), so a request can move between the fused
CUDA op and the Triton spec path from one step to the next. That costs some speed against the
prototype: the GDN kernel takes 116 us at batch 4 instead of 97 us.

Same machine and model, vLLM main built for sm_121:

| | per-token stores | deferred |
|---|---|---|
| decode step, batch 1 (GPU busy) | 120.00 / 119.85 ms | 117.95 / 117.90 ms (-1.7%) |
| decode step, batch 4 (GPU busy) | 132.22 / 132.94 ms | 125.35 / 125.74 ms (-5.3%) |
| `vllm bench serve`, concurrency 1 | 27.14 tok/s | 27.50 tok/s (+1.3%) |
| `vllm bench serve`, concurrency 4 | 94.48 tok/s | 99.42 tok/s (+5.2%) |
| `vllm bench serve`, concurrency 8 | 164.18 tok/s | 176.56 tok/s (+7.5%) |
| GSM8K, 200 questions, greedy | 0.975 | 0.975 |

Greedy tokens are identical between the two arms in every decode-step run. Kernel tests: 162
passed, including eight new ones (bit-identical outputs and slot-0 state against per-token
stores, and the CUDA and Triton paths handing a request to each other). Commands and raw output
are in the PR.

## 8. vLLM 0.34.0

| | 0.31.0 | 0.34.0 |
|---|---|---|
| step (median) | 118.4 ms | 116.7 ms |
| launches per step | 974 | 957 |

0.34.0 runs the small BF16 `in_proj_ba` GEMM on a parallel graph branch next to `in_proj_qkvz`
and fuses several norm/quant pairs. The GDN out_proj split, the four `lm_head` reads and the
BF16 draft weights are unchanged.

## 9. Smaller draft heads

The drafts only propose tokens; the main model checks them. A draft head over part of the vocabulary
reads that part of `lm_head` instead of all 1.27 GB, and can only cost acceptance, never correctness.
The ledger put the three draft reads at 5.6 ms each, 16.8 ms per step. Tried in Sibyl's replay of
the recording (the same kernels and addresses as the engine): the drafts' `lm_head` GEMM replaced by an
FP8 GEMV over the first 32,768 token ids, every other column -inf ([`draft-head`](draft-head)).
Batch 1, 200 tokens per request, one English prompt:

| | full vocabulary | first 32,768 ids |
|---|---|---|
| step (median) | 120.4 ms | **106.1 ms** |
| greedy: tokens per step | 2.84 | 2.73 |
| greedy: tokens/s | 23.6 | **25.7 (+9%)** |
| greedy: output | | identical |
| sampled (3 seeds, T 0.2 and 0.9): tokens per step | 2.51 | 2.49 |
| sampled: tokens/s | 20.7 | **23.4 (+13%)** |

The step loses 14.3 ms of the 16.8 the ledger predicted; the rest is the GEMV and the -inf fill.
Greedy decoding loses some acceptance where the best draft is outside the first 32,768 ids. Choosing
the rows by frequency in GSM8K, Python sources and English documentation did no better (2.76 and
2.43 tokens per step); a corpus closer to what the model writes may. Not measured: other languages
(the low ids are mostly English pieces), larger batches. vLLM's EAGLE-3 drafts already map a small
draft vocabulary onto the target's; the MTP drafts share the full head.

## Reproduce

```sh
make -C ../../capture CUDA=/usr/local/cuda
LD_PRELOAD=../../capture/libledger.so LEDGER_OUT=out python ../../examples/vllm_decode.py \
  --steps 40 --batch 1 -- --model <Qwen3.8-27B-NVFP4> --trust-remote-code --dtype bfloat16 \
  --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --no-async-scheduling \
  --speculative-config '{"method": "qwen3_5_mtp", "model": "<same>", "num_speculative_tokens": 3}'
python -m ledger.analyze out --ceiling-gbps 252
python gemm_isolation.py <Qwen3.8-27B-NVFP4>/model.safetensors
```
