// Draft logits over part of the vocabulary, for one row of activations (m == 1):
//   d[ids[j]] = a_scale * w_scale[j] * dot(a, w[j])   for the compact rows w (n x k, fp8 e4m3)
//   d[c] = -inf                                       for every column c not among ids (inv[c] < 0)
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

__device__ __forceinline__ float fp8(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}

extern "C" __global__ void __launch_bounds__(256) draft_head(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                                             const float* w_scale, const int* ids, const int* inv,
                                                             int n, int k, int vocab, __nv_bfloat16* d) {
  // The activations as floats, laid out [e][c] for 16-byte chunk c and byte e of it: the lanes of a warp
  // read consecutive chunks, so for a fixed e they hit consecutive words.
  extern __shared__ float sa[];
  const int chunks = k / 16;
  for (int i = threadIdx.x; i < k; i += blockDim.x) sa[(i % 16) * chunks + i / 16] = fp8(a[i]);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, warps = (gridDim.x * blockDim.x) >> 5;
  const float s = *a_scale;
  for (int j = warp; j < n; j += warps) {
    const uint4* row = reinterpret_cast<const uint4*>(w + static_cast<size_t>(j) * k);
    float acc = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
      const uint4 v = __ldg(row + c);
      const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
      for (int e = 0; e < 16; ++e) acc += sa[e * chunks + c] * fp8(b[e]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (!lane) d[ids[j]] = __float2bfloat16(acc * s * w_scale[j]);
  }
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < vocab; c += gridDim.x * blockDim.x)
    if (inv[c] < 0) d[c] = __float2bfloat16(-INFINITY);
}
