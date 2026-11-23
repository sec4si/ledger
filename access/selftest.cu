// Self-test of the access tracer: three kernels with known access patterns, run for six steps.
//   fill_half   writes bytes [0, 512 KiB) of a 1 MiB buffer
//   read_all    reads the whole buffer
//   write_lows  writes bytes [0, 512) of every KiB
// Run with LEDGER_ACCESS_ANCHOR=_Z9fill_halfPf and LEDGER_GRAPH_WINDOW=0:3, or with SELFTEST_PROFILER=1
// (profiling range around steps 1 to 3) and no window; see README.md for the expected analysis. With
// SELFTEST_GRAPH=1 each step is one graph launch, which the tracer must run node by node.
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

__global__ void fill_half(float* b) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (512u << 10) / 4) b[i] = 1.0f;
}
__global__ void read_all(const float* b, float* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (1u << 20) / 4) atomicAdd(out, b[i] * 0.0f);
}
__global__ void write_lows(float* b) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per float of the low halves
  if (i < 1024 * 128) b[(i / 128) * 256 + i % 128] = 2.0f;
}

int main() {
  float *buf, *out;
  cudaMalloc(&buf, 1 << 20);
  cudaMalloc(&out, 4);
  const bool profiler = getenv("SELFTEST_PROFILER") != nullptr;
  cudaStream_t s;
  cudaStreamCreate(&s);
  const auto step_kernels = [&] {
    fill_half<<<512, 256, 0, s>>>(buf);
    read_all<<<1024, 256, 0, s>>>(buf, out);
    write_lows<<<512, 256, 0, s>>>(buf);
  };
  cudaGraphExec_t graph = nullptr;
  if (getenv("SELFTEST_GRAPH")) {
    cudaGraph_t g;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    step_kernels();
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&graph, g, 0);
    cudaGraphDestroy(g);
  }
  for (int step = 0; step < 6; ++step) {
    if (profiler && step == 1) cudaProfilerStart();
    if (profiler && step == 4) cudaProfilerStop();
    if (graph)
      cudaGraphLaunch(graph, s);
    else
      step_kernels();
    cudaStreamSynchronize(s);
  }
  printf("buf=%p out=%p\n", static_cast<void*>(buf), static_cast<void*>(out));
  return cudaGetLastError() != cudaSuccess;
}
