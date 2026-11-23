// Self-test of tensor-map launches: tma_copy moves the top half of a with tensor-map copies into b,
// then clear overwrites all of b, so every byte tma_copy writes is dead.
//   fill      writes a (1 MiB)
//   tma_copy  reads a and writes b[0, 512 KiB) through tensor maps only
//   clear     writes all of b (1 MiB)
// Run like selftest.cu, with LEDGER_ACCESS_ANCHOR=_Z4fillPff; SELFTEST_GRAPH=1 runs each step as one
// graph launch. The tracer must give tma_copy "tma_writes" of exactly b[0, 512 KiB), and the analysis
// 512 KiB of dead writes per step, all from tma_copy.
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

constexpr int kCols = 1024, kRows = 256, kBoxX = 64, kBoxY = 8;

__global__ void fill(float* a, float v) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kCols * kRows) a[i] = v + i * 1e-3f;
}

__global__ void clear(float* b) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kCols * kRows) b[i] = 0.0f;
}

// One box per block, copied from a to b by one thread: global -> shared -> global, all through TMA.
__global__ void tma_copy(const __grid_constant__ CUtensorMap a, const __grid_constant__ CUtensorMap b) {
  __shared__ alignas(128) float tile[kBoxY][kBoxX];
  __shared__ alignas(8) unsigned long long bar;
  if (threadIdx.x) return;
  const int x = blockIdx.x * kBoxX, y = blockIdx.y * kBoxY;
  const unsigned sbar = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  const unsigned stile = static_cast<unsigned>(__cvta_generic_to_shared(tile));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sbar));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sbar), "r"(kBoxX * kBoxY * 4) : "memory");
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(stile),
      "l"(&a), "r"(x), "r"(y), "r"(sbar)
      : "memory");
  asm volatile(
      "{\n .reg .pred p;\n WAIT%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WAIT%=;\n}" ::"r"(sbar)
      : "memory");
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(&b), "r"(x), "r"(y),
               "r"(stile)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

CUtensorMap map_of(float* p) {
  CUtensorMap m;
  const cuuint64_t dim[2] = {kCols, kRows}, stride[1] = {kCols * 4};
  const cuuint32_t box[2] = {kBoxX, kBoxY}, step[2] = {1, 1};
  if (cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, p, dim, stride, box, step,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
    exit(1);
  }
  return m;
}

int main() {
  float *a, *b;
  cudaMalloc(&a, kCols * kRows * 4);
  cudaMalloc(&b, kCols * kRows * 4);
  const CUtensorMap ma = map_of(a), mb = map_of(b);
  const bool profiler = getenv("SELFTEST_PROFILER") != nullptr;
  cudaStream_t s;
  cudaStreamCreate(&s);
  float v = 1.0f;
  const auto step_kernels = [&] {
    fill<<<kCols * kRows / 256, 256, 0, s>>>(a, v);
    tma_copy<<<dim3(kCols / kBoxX, kRows / 2 / kBoxY), 32, 0, s>>>(ma, mb);
    clear<<<kCols * kRows / 256, 256, 0, s>>>(b);
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
  float first;
  cudaMemcpy(&first, b, 4, cudaMemcpyDeviceToHost);
  printf("a=%p b=%p b[0]=%g\n", static_cast<void*>(a), static_cast<void*>(b), first);
  return cudaGetLastError() != cudaSuccess;
}
