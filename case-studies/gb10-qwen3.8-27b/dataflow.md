# Qwen3.8-27B on GB10: what the decode step writes for nothing

The access tracer (`access/`) on Sibyl's replay of the vLLM 0.31.0 recording of this model (the same
kernels and addresses as the engine), batch 1, CUDA graphs launched node by node by the tracer. Every
one of the 974 launches per step is seen, the 269 tensor-map launches included.

## Dead writes, and which can be dropped

Four traces of the same prompt with different seeds and temperatures (0.2, 0.9, 0.9, 1.3), 12 steps
each, 44 resolved steps. `ledger.dead_across` splits each launch's dead bytes into the same bytes dead
in every step (fixed), as many dead every step but different ones (moving), and dead in some steps only
(varies):

| kernel | launches/step | written MB/step | dead MB/step | fixed | moving | varies |
|---|---|---|---|---|---|---|
| `gdn_decode_post_conv_mtp_kernel` | 48 | 606.34 | 452.98 | 0.00 | 452.98 | 0.00 |
| `_causal_conv1d_update_kernel` | 48 | 9.83 | 2.95 | 0.00 | 2.95 | 0.00 |
| `triton_poi_fused_zeros_4` | 47 | 2.31 | 2.31 | 2.31 | 0.00 | 0.00 |
| `_temperature_kernel` | 1 | 3.97 | 0.79 | 0.00 | 0.00 | 0.79 |
| `_compute_slot_mappings_kernel` | 3 | 0.79 | 0.79 | 0.79 | 0.00 | 0.00 |
| fp4 quantization (`triton_red_fused_..._cutlass_scaled_mm...`) | 72 | 6.68 | 0.28 | 0.28 | 0.00 | 0.00 |
| total | 974 | 711.87 | 460.18 | 3.43 | 455.96 | 0.79 |

Almost everything dead is moving. The GDN kernel stores the recurrent state after each of the four
tokens a step verifies, and the next step reads the one after the last accepted token: three of four
are dead every step, but which three depends on how many drafts were accepted, which is not known when
they are written. They cannot be dropped, only deferred (section 6 of the [README](README.md)); the
causal-conv state is the same. `_temperature_kernel` scales the logits of all four positions; the
fourth is read only when every draft is accepted. What can simply be dropped is 3.4 MB per step,
mostly zero fills nobody reads.

## Writes the output does not depend on

`ledger.perturb_plan --by-range 4` cuts each launch's writes into tensors, and Sibyl's
`replay_perturb` poisons each right after its launch (with 0xFF, then 0x00) and compares the tokens of
the following steps with a baseline; the tensor-map GEMM outputs are included (656 of 3,172
experiments). With 3 steps compared, 934 experiments covering 476.3 MB of a step's writes were unused.
Rerunning them with 24 steps compared:

| | experiments | MB |
|---|---|---|
| unused after 24 steps | 853 | 473.8 |
| changed the tokens after step 3 only | 47 | 1.3 |
| changed them within 3 steps, unlike the first run | 34 | 1.2 |

The late ones are mostly `cvt_fp16_to_fp4` (the activations fed to the NVFP4 GEMMs), whose effect
reaches the tokens 4 to 17 steps later. The 34 that disagree with the first run were poisoned with
0xFF; the cause is not known, so a verdict is good to about 1 MB here. The GDN states stay unused over
24 steps: this entry accepts every draft in its first step, so only the last position is read.

## Reproduce

```sh
# trace a request through the replay service (window: anchor launches 5 to 17)
LD_PRELOAD=access/libledger_access.so LEDGER_ACCESS_PATCHES=access/patches.fatbin LEDGER_OUT=out \
  LEDGER_GRAPH_WINDOW=5:12 LEDGER_ACCESS_ANCHOR=<first kernel of a step> LEDGER_ACCESS_SKIP=<weights> \
  LEDGER_ACCESS_EVENTS=33554432 <replay engine> ...
python -m ledger.dataflow out --anchor <first kernel of a step> --layout mem_layout.json --masks-out out/masks.npz
python -m ledger.dead_across out*/masks.npz
```
