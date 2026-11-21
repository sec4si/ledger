// 本机显存读带宽上限：每线程 16 字节向量读、求和防优化；另测 D2D 拷贝。
#include <cstdio>
#include <cuda_runtime.h>

__global__ void read_kernel(const int4* __restrict__ p, size_t n, int* out) {
  int acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x7fffffff) *out = acc;
}

int main() {
  const size_t bytes = 4ull << 30;  // 4 GiB，远大于 L2
  int4* buf; int* out; void* dst;
  cudaMalloc(&buf, bytes); cudaMalloc(&dst, bytes); cudaMalloc(&out, 4);
  cudaMemset(buf, 1, bytes);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  for (int blocks_per_sm : {4, 8, 16, 32}) {
    float best = 1e9;
    for (int r = 0; r < 6; ++r) {
      cudaEventRecord(a);
      read_kernel<<<sms * blocks_per_sm, 512>>>(buf, bytes / 16, out);
      cudaEventRecord(b); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); if (r && ms < best) best = ms;
    }
    printf("read  blocks/SM=%2d  %.1f GB/s\n", blocks_per_sm, bytes / best / 1e6);
  }
  float best = 1e9;
  for (int r = 0; r < 6; ++r) {
    cudaEventRecord(a); cudaMemcpyAsync(dst, buf, bytes, cudaMemcpyDeviceToDevice); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); if (r && ms < best) best = ms;
  }
  printf("d2d copy (read+write) %.1f GB/s\n", 2.0 * bytes / best / 1e6);
  return 0;
}
