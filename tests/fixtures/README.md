# Test captures

`captures.tar.xz` holds five real captures from a DGX Spark (GB10, sm_121), vLLM main `4a107a28`
built for 12.1, cut to their first three steps with `tools/trim_capture.py`:

| directory | run |
|---|---|
| `decode-q38` | Qwen3.8-27B NVFP4, MTP with 3 draft tokens, batch 1, CUDA graphs |
| `decode-q38-bf16state` | the same with `--mamba-ssm-cache-dtype bfloat16` (for `ledger.diff`: GDN decode and `out_proj` change) |
| `longctx-q38` | Qwen3.8-27B, 32,000-token prompt, eager (no graph nodes), attention views |
| `prefill-q38` | Qwen3.8-27B, one 2048-token prefill step in a CUDA graph |
| `decode-q30-moe` | Qwen3-30B-A3B FP8 (MoE), batch 1, CUDA graphs |

Each directory is what `examples/vllm_decode.py` writes under `LEDGER_OUT`: the capture library's
JSONL files, `regions-*.json`, `attention-*.json` (where exported), `run.json` and `steps.jsonl`.
The captures predate the capture library's CUkernel fix (`5f221cd`), so each keeps 14 launches per step
without parameters (small PyTorch kernels); a current capture of the same run has none. Tests unpack the
archive into a temporary directory. Measured ceilings on that machine: 252 GB/s
read bandwidth, BF16 91 and FP8 181 TFLOP/s.
