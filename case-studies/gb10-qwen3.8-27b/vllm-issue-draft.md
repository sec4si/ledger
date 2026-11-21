# [Perf] GDN MTP decode writes a full recurrent state per draft position; the write-back lands on the next GEMM

**Draft, not filed.**

## Summary

With MTP speculative decoding, `gdn_decode_post_conv_mtp_kernel` (csrc/gdn/fused_gdn_decode_kernel.cu)
stores the full recurrent state after every token of the step, one slot per position. For
Qwen3.8-27B (48 value heads, 128x128, fp32) that is 3.15 MB per position, 12.6 MB per layer with
three draft tokens. The next step reads only the slot of the last accepted position.

The kernel itself stays fast because the stores land in L2. The cost shows up in the following
GEMM (`linear_attn.out_proj`), which has to flush those dirty lines while it streams its weight.

## Environment

DGX Spark (GB10, sm_121, 24 MB L2, ~252 GB/s measured read bandwidth), vLLM 0.31.0 and 0.34.0,
Qwen3.8-27B NVFP4, `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}'`,
batch 1, CUDA graphs.

## Evidence

1. Same CUTLASS FP8 kernel, same grid, same weight shape [6144, 5120] (both 4 KB aligned):
   GDN `out_proj` 186 us per launch, attention `o_proj` 138 us. 0.34.0 shows the same split.
2. In isolation (L2 flushed with a read), both weights take 147 us.
3. Writing N MB right before the GEMM: 0 MB 147 us, 4 MB 169 us, 8 MB 188 us, 12 MB 199 us,
   24 MB 214 us.
4. `--mamba-ssm-cache-dtype bfloat16` halves the state writes: the GDN kernel drops from 41 us to
   22 us and `out_proj` from 191 us to 175 us, consistently across alternating runs at a locked
   clock. (Not proposed as a fix; it changes numerics.)

Per step, the per-position states cost roughly 48 x 46 us = 2.2 ms of the 118 ms step on this
machine.

## Possible fix

Keep the state slot of the step's starting state, store the per-token inputs of the update (k, v,
gates, beta: a few KB per head), and let the next step replay the accepted tokens on top of the
starting state before processing its own tokens. Only one full state is written per step. The
arithmetic is the same sequence of rank-1 updates, so outputs can stay bitwise identical.

Expected saving on this setup: about three quarters of the state writes, roughly 2 to 2.5 ms per
step. Larger batches multiply the state traffic.

## How this was found

[Ledger](https://github.com/sec4si/ledger) attributes each kernel launch to the tensors it reads;
the out_proj/o_proj split showed up as "same kernel, same shape, different speed".
