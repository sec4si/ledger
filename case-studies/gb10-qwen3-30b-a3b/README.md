# Case study: decode on DGX Spark, Qwen3-30B-A3B FP8 (MoE)

The access trace finds almost nothing to remove; the profile says where the time goes instead, and a
one-line change in vLLM's backend choice buys most of it back.

## Setup

| | |
|---|---|
| Machine | NVIDIA DGX Spark: GB10 (sm_121, 48 SMs, 24 MB L2), 121 GB unified LPDDR5x, read ceiling 252 GB/s |
| Engine | vLLM 0.34.0; traces and the Nsight Compute profile with `--enforce-eager`, timings with CUDA graphs |
| Model | Qwen3-30B-A3B-Instruct-2507-FP8: 48 layers, 128 experts, 8 routed per token, block FP8 (128x128) |
| Load | decode, batch 1 and 4 |

## 1. Nothing is written for nothing

The access tracer (`access/`, weights skipped) and `ledger.dataflow` over four decode steps:

| | batch 1 | batch 4 |
|---|---|---|
| non-weight bytes written per step | 47.0 MB | 190.8 MB |
| overwritten before anyone read them | 0.4 MB | 1.5 MB |

What little is dead comes from `_fwd_kernel_ep_scatter_2` (about 40% of what it writes), which lays tokens out
per expert for the grouped GEMM. The rest of the step reads everything it writes. The 192 launches per step of
DeepGEMM's sm120 GEMMs move their operands with tensor-map copies, which the Sanitizer API does not instrument;
`ledger.dataflow` treats them as reading and writing the traced range their pointers start (see `access/README.md`).

The tracer now runs CUDA graphs node by node and reads the tensor maps among a launch's parameters, so the
same step can be traced with CUDA graphs on (1,169 launches per step, every one seen, 96% with parameters):
78.9 MB written per step, 0.3 MB of it dead. What the GEMMs write through tensor maps is measured by
comparing the tensors before and after each launch (about 15 MB per step, none of it dead). Their reads
are not: DeepGEMM encodes its activation maps for 4,096 rows, so each is taken to read up to the end of
the allocator segment it starts in. That can hide dead writes and does here: `_fwd_kernel_ep_scatter_2`
no longer shows its 40%.

## 2. The experts are read exactly, but slowly

The expert weights move through tensor maps too, so the tracer cannot say which experts a step reads.
Nsight Compute can say how much each launch pulled from memory into the L2
(`lts__d_sectors_fill_sysmem`, application replay, caches left as the program leaves them); `ledger.traffic`
joins it with the trace. One eager decode step at batch 1:

| launch | per layer | pulled | for 8 experts | time | of the ceiling |
|---|---|---|---|---|---|
| grouped GEMM, w13 (gate and up) | 1 | 26.8 MB | 25.2 MB + scales | 160 us | 66% |
| grouped GEMM, w2 (down) | 1 | 12.8 MB | 12.6 MB + scales | 84 us | 60% |
| `lm_head` (151,936 x 2,048 BF16) | 1 per step | 633 MB | 622 MB | 4.1 ms | 61% |

Routing costs no extra traffic: each layer reads its eight experts and nothing else. The step pulls 3.49 GB,
13.8 ms at the ceiling, against 30 ms of GPU time; the expert GEMMs lose about 4 ms of it and `lm_head` 1.6 ms.
Dense FP8 GEMMs on the same machine reach 85-92% (see the Qwen3.8 case study), so 60-66% is the kernel, not the
memory.

## 3. Other backends

vLLM picks the MoE backend from a priority list. On Hopper it already moves Triton to the front for block FP8
with TP; SM12x has no such rule and falls through to DeepGEMM. The same model with each backend, CUDA graphs on:

| `--moe-backend` | decode step, batch 1 | decode step, batch 4 |
|---|---|---|
| `deep_gemm` (default) | 23.52 ms | 48.02 ms |
| `triton` | 20.58 ms (-12.5%) | 38.73 ms (-19.3%) |
| `marlin` (weight-only FP8) | 20.74 ms (-11.8%) | 36.34 ms (-24.3%) |
| `cutlass` | not supported for this configuration | |

Marlin dequantizes the weights and keeps the activations in BF16, which changes the numerics; Triton keeps the
W8A8 path.

## 4. Serving and accuracy

`vllm bench serve`, random 512 in / 256 out, one run per point:

| concurrency | DeepGEMM tok/s | Triton tok/s | median TPOT (ms) | median TTFT (ms) |
|---|---|---|---|---|
| 1 | 45.32 | 54.97 (+21.3%) | 20.36 -> 17.74 | 255 -> 138 |
| 4 | 99.63 | 117.61 (+18.0%) | 37.52 -> 32.53 | 431 -> 348 |
| 8 | 136.90 | 196.55 (+43.6%) | 56.36 -> 39.00 | 589 -> 452 |

DeepGEMM's first request at concurrency 1 takes 3.1 s (P99), most likely JIT compilation; the medians leave it out.

`vllm bench latency`, 128 in / 128 out:

| batch | DeepGEMM avg latency (s) | Triton avg latency (s) |
|---|---|---|
| 1 | 2.683 | 2.322 (-13.5%) |
| 4 | 5.918 | 4.745 (-19.8%) |
| 16 | 12.295 | 7.989 (-35.0%) |

`vllm bench throughput`:

| workload | DeepGEMM tok/s | Triton tok/s |
|---|---|---|
| random 1024 in / 128 out, 200 prompts | 3516 | 3879 (+10.3%) |
| random 4096 in / 16 out, 64 prompts (prefill-bound) | 6382 | 6317 (-1.0%) |

GSM8K, 200 questions, 5-shot, greedy (`tests/evals/gsm8k/gsm8k_eval.py`): 0.965 with DeepGEMM, 0.960 with Triton.

## 5. Upstream

Proposed in [vllm-project/vllm#69355](https://github.com/vllm-project/vllm/pull/69355): prefer Triton for
block FP8 MoE with TP on the SM12x family, as on Hopper.
Only GB10 was measured; RTX 50 series cards share the kernels but not the memory system.

## Reproduce

```sh
# access trace with the weights skipped (window: the profiling range of examples/vllm_decode.py)
LD_PRELOAD=$PWD/access/libledger_access.so LEDGER_ACCESS_PATCHES=$PWD/access/patches.fatbin \
  LEDGER_ACCESS_SKIP=auto LEDGER_OUT=acc CUDA_HOME=/usr/local/cuda \
  python examples/vllm_decode.py --steps 4 --batch 1 -- --model <Qwen3-30B-A3B-Instruct-2507-FP8> \
  --enforce-eager --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.7 --no-async-scheduling
python -m ledger.dataflow acc --anchor <first kernel of a step>

# what each launch pulled from memory
ncu --replay-mode application --cache-control none --profile-from-start off --print-kernel-base mangled --csv \
  --metrics gpu__time_duration.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum \
  python examples/vllm_decode.py --steps 4 --batch 1 -- <same vLLM arguments> > profile.csv
python -m ledger.traffic acc - profile.csv --regions acc/regions-<pid>.json --anchor <first kernel of a step>
```
