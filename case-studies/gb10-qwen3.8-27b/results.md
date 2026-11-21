# Raw results

## gemm_isolation.py (vLLM 0.31.0, GB10)

```
1. same kernel, same shape, different weights
  GDN out_proj, layer 0          147.5 us   213.3 GB/s
  GDN out_proj, layer 1          147.5 us   213.3 GB/s
  attention o_proj, layer 3      147.5 us   213.3 GB/s
  attention o_proj, layer 7      147.5 us   213.3 GB/s

2. dirty L2 before the GEMM
    0 MB written before:   147.5 us
    2 MB written before:   158.7 us
    4 MB written before:   169.0 us
    8 MB written before:   187.4 us
   12 MB written before:   197.6 us
   16 MB written before:   201.7 us
   24 MB written before:   213.2 us
   32 MB written before:   215.0 us
```

## bench/bwtest.cu

```
read  blocks/SM= 4  253.5 GB/s
read  blocks/SM= 8  251.9 GB/s
read  blocks/SM=16  251.5 GB/s
read  blocks/SM=32  250.7 GB/s
d2d copy (read+write) 243.0 GB/s
```

## BabelStream (a992a27, CUDA, -s 268435456 -n 50, MB/s)

```
double: Copy 250387  Mul 251999  Add 248765  Triad 248252  Dot 253036
float:  Copy 245514  Mul 247194  Add 248109  Triad 248340  Dot 246542
```

## FP32 vs BF16 recurrent state, alternating, clock locked at 2000 MHz (batch 1, 40 steps)

| run | step median | GDN decode kernel | GDN out_proj | lm_head |
|---|---|---|---|---|
| A1 fp32 | 121.0 ms | 41.2 us | 192 us | 5422 us |
| B1 bf16 | 120.3 ms | 21.6 us | 175 us | 5431 us |
| A2 fp32 | 122.8 ms | 40.6 us | 190 us | 5505 us |
| B2 bf16 | 121.9 ms | 21.7 us | 176 us | 5538 us |

## vLLM 0.31.0 vs 0.34.0 (batch 1, 40 steps, nsys)

| | 0.31.0 | 0.34.0 |
|---|---|---|
| step median | 118.4 ms | 116.7 ms |
| GPU busy per step | 116.9 ms | 115.1 ms |
| launches per step | 974 | 957 |
| distinct kernels | 68 | 62 |
