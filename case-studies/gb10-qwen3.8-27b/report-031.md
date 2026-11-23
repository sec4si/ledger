# Ledger report: `out-031-base`

38880 kernel launches over 40 steps, 1779 memory regions, 560 launches without parameters. Bandwidth ceiling 252 GB/s.

## Step

| | per step |
|---|---|
| wall time (first launch to last) | 120.88 ms |
| GPU busy | 120.52 ms |
| idle between kernels | 0.36 ms (0.3%) |
| launches | 972 |
| weight bytes streamed | 25.47 GB |
| effective bandwidth | 211 GB/s (84% of ceiling) |
| floor at ceiling | 101.08 ms |

## Weight-streaming launches by weight

| weight | shape | dtype | launches/step | avg us | MB | GB/s | % ceiling | lost ms/step |
|---|---|---|---|---|---|---|---|---|
| `language_model.model.layers.N.linear_attn.out_proj.weight` | [6144, 5120] | float8_e4m3fn | 48.0 | 194.4 | 31.5 | 162 | 64% | 3.33 |
| `language_model.lm_head.weight` | [5120, 248320] | float8_e4m3fn | 4.0 | 5570.8 | 1272.4 | 228 | 91% | 2.09 |
| `language_model.model.layers.N.mlp.gate_up_proj.weight` | [34816, 2560] | uint8 | 56.0 | 434.5 | 100.3 | 231 | 92% | 2.05 |
| `language_model.model.layers.N.linear_attn.in_proj_qkvz.weight` | [5120, 16384] | float8_e4m3fn | 48.0 | 374.6 | 84.0 | 224 | 89% | 1.99 |
| `language_model.model.layers.N.mlp.down_proj.weight` | [5120, 8704] | uint8 | 56.0 | 233.9 | 50.1 | 214 | 85% | 1.96 |
| `language_model.model.layers.N.mlp.gate_up_proj.weight` | [5120, 34816] | float8_e4m3fn | 8.0 | 781.2 | 178.4 | 228 | 91% | 0.59 |
| `language_model.model.layers.N.self_attn.qkv_proj.weight` | [5120, 14336] | float8_e4m3fn | 16.0 | 327.7 | 73.5 | 224 | 89% | 0.58 |
| `language_model.model.layers.N.mlp.down_proj.weight` | [17408, 5120] | float8_e4m3fn | 8.0 | 404.1 | 89.1 | 221 | 88% | 0.40 |
| `language_model.model.layers.N.self_attn.o_proj.weight` | [6144, 5120] | float8_e4m3fn | 16.0 | 147.3 | 31.5 | 214 | 85% | 0.36 |
| `speculator.model.layers.N.mlp.down_proj.weight` | [5120, 17408] | bfloat16 | 3.0 | 777.9 | 178.3 | 229 | 91% | 0.21 |
| `speculator.model.layers.N.mlp.gate_up_proj.weight` | [34816, 5120] | bfloat16 | 3.0 | 1477.4 | 356.5 | 241 | 96% | 0.19 |
| `speculator.model.fc.weight` | [5120, 10240] | bfloat16 | 3.0 | 457.5 | 104.9 | 229 | 91% | 0.12 |
| `speculator.model.layers.N.self_attn.qkv_proj.weight` | [14336, 5120] | bfloat16 | 3.0 | 619.6 | 146.8 | 237 | 94% | 0.11 |
| `speculator.model.layers.N.self_attn.o_proj.weight` | [5120, 6144] | bfloat16 | 3.0 | 265.9 | 62.9 | 237 | 94% | 0.05 |

## Same kernel, same shape, different speed

### `cutlass::device_kernel<…>` grid [1, 80, 1] on [6144, 5120] float8_e4m3fn

- `language_model.model.layers.N.linear_attn.out_proj.weight`: median **194.6 us** over 1920 launches
- `language_model.model.layers.N.self_attn.o_proj.weight`: median 147.3 us over 640 launches
- excess: 47.3 us per launch, **2.27 ms per step**
- last kernel touching caches before `language_model.model.layers.N.linear_attn.out_proj.weight`: `gdn_decode_post_conv_mtp_kernel<…>` (1920/1920 launches)
- last kernel touching caches before `language_model.model.layers.N.self_attn.o_proj.weight`: `kernel_mha` (640/640 launches)

## Weights read more than once per step

| weight | tensors | MB each | reads/step | extra GB/step |
|---|---|---|---|---|
| `language_model.lm_head.weight` | 1 | 1271.4 | 4.0 | 3.81 |
| `speculator.model.layers.N.mlp.gate_up_proj.weight` | 1 | 356.5 | 3.0 | 0.71 |
| `speculator.model.layers.N.mlp.down_proj.weight` | 1 | 178.3 | 3.0 | 0.36 |
| `speculator.model.layers.N.self_attn.qkv_proj.weight` | 1 | 146.8 | 3.0 | 0.29 |
| `speculator.model.fc.weight` | 1 | 104.9 | 3.0 | 0.21 |
| `speculator.model.layers.N.self_attn.o_proj.weight` | 1 | 62.9 | 3.0 | 0.13 |

Partial reads (too short to stream the whole tensor, not counted above): `language_model.model.embed_tokens.weight` ×4.0/step
