# draft-head

`draft_head.cu` computes the draft logits over part of the vocabulary: an FP8 GEMV over the chosen
rows of `lm_head` (copied into one compact block unless they are the first n), scaled by the
activation and per-row weight scales as the CUTLASS epilogue does, written to the chosen columns of the
full-vocabulary BF16 logits; every other column gets -inf, so the argmax after it is unchanged.

```sh
nvcc -cubin -arch=sm_121 -O3 -o draft_head.cubin draft_head.cu
```

It was swapped into Sibyl's replay of the recording, where every launch is a graph node with its
recorded parameter block. In the three draft graphs (a few dozen nodes each; the main graph has
hundreds) the `lm_head` GEMM is the CUTLASS launch whose weight tensor map spans 248,320 x 5,120 FP8
with a one-row activation map. Its 1,664-byte parameter block holds, as `ledger.access` decodes it:

| offset | what |
|---|---|
| 128 | activation tensor map (FP8, 5,120 x 1); address in its first 8 bytes |
| 384 | weight tensor map (`lm_head.weight`) |
| 768 | pointer to the activation scale (FP32) |
| 784 | pointer to `lm_head.weight_scale` (FP32, one per row) |
| 1152 | output tensor map (BF16, 248,320 x 1) |

The node becomes `draft_head` with 192 blocks of 256 threads and 20 KB of shared memory. The replay
patch is not part of Sibyl.
