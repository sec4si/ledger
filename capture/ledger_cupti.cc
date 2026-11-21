// Ledger capture library. Load with LD_PRELOAD into any CUDA program.
//
// Records, for every kernel node of every CUDA graph the process instantiates and for every
// kernel launched directly inside the profiling range, the function name, launch geometry and
// the raw bytes of each parameter. Inside the profiling range (cudaProfilerStart/Stop) it also
// records kernel timings through the CUPTI activity API. The analysis step joins the two to tell
// which memory each launch touched and how long it took.
//
// Output directory: $LEDGER_OUT (default ./ledger-out). One set of JSONL files per process:
//   nodes-<pid>.jsonl     graph kernel nodes  {"id","name","grid","block","smem","params"}
//   clones-<pid>.jsonl    node clones made at instantiation {"id","orig"}
//   launches-<pid>.jsonl  direct launches in range {"corr","name","grid","block","smem","params"}
//   kernels-<pid>.jsonl   kernel timings in range {"start","end","corr","node","graph","stream","name","grid"}
// "params" is a list of [offset, size, hex].

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/stat.h>

namespace {

// One output file per stream, created on first write so processes that never touch CUDA leave
// nothing behind.
struct Out {
  const char* stem;
  FILE* f = nullptr;
};

std::mutex g_mu;
Out g_nodes{"nodes"}, g_clones{"clones"}, g_launches{"launches"}, g_kernels{"kernels"};
std::atomic<bool> g_in_range{false};
CUpti_SubscriberHandle g_sub;

FILE* open_out(const char* stem) {
  const char* dir = getenv("LEDGER_OUT");
  std::string d = dir ? dir : "ledger-out";
  mkdir(d.c_str(), 0755);
  std::string path = d + "/" + stem + "-" + std::to_string(getpid()) + ".jsonl";
  FILE* f = fopen(path.c_str(), "w");
  if (f) setvbuf(f, nullptr, _IOFBF, 1 << 22);
  return f;
}

void json_escape(std::string& out, const char* s) {
  for (; s && *s; ++s) {
    if (*s == '"' || *s == '\\') out += '\\';
    if (static_cast<unsigned char>(*s) >= 0x20) out += *s;
  }
}

void append_hex(std::string& out, const void* p, size_t n) {
  static const char* digits = "0123456789abcdef";
  const auto* b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) {
    out += digits[b[i] >> 4];
    out += digits[b[i] & 15];
  }
}

// Parameter layouts (offset, size). Exact layouts are cached per CUfunction. Runtime launches from
// libraries with a statically linked cudart cannot be mapped back to a CUfunction, so a second map
// keyed by name serves them, but only for names whose layout is the same everywhere: JIT compilers
// reuse kernel names across modules with different signatures.
using Layout = std::vector<std::pair<size_t, size_t>>;
struct NamedLayout {
  Layout layout;
  bool ambiguous = false;
};
std::mutex g_layout_mu;
std::unordered_map<CUfunction, Layout> g_func_layouts;
std::unordered_map<std::string, NamedLayout> g_name_layouts;

const Layout* layout_of(CUfunction f, const char** name) {
  *name = nullptr;
  if (!f || cuFuncGetName(name, f) != CUDA_SUCCESS || !*name) return nullptr;
  std::lock_guard<std::mutex> lock(g_layout_mu);
  auto it = g_func_layouts.find(f);
  if (it != g_func_layouts.end()) return &it->second;
  Layout layout;
  for (size_t i = 0;; ++i) {
    size_t off = 0, size = 0;
    if (cuFuncGetParamInfo(f, i, &off, &size) != CUDA_SUCCESS) break;
    layout.emplace_back(off, size);
  }
  auto [named, fresh] = g_name_layouts.try_emplace(*name, NamedLayout{layout});
  if (!fresh && named->second.layout != layout) named->second.ambiguous = true;
  return &g_func_layouts.emplace(f, std::move(layout)).first->second;
}

const Layout* layout_by_name(const char* name) {
  if (!name) return nullptr;
  std::lock_guard<std::mutex> lock(g_layout_mu);
  auto it = g_name_layouts.find(name);
  return it == g_name_layouts.end() || it->second.ambiguous ? nullptr : &it->second.layout;
}

// Name, geometry and parameter bytes of one launch as a JSON fragment (no braces).
std::string describe(const char* name, const Layout* layout, unsigned gx, unsigned gy, unsigned gz,
                     unsigned bx, unsigned by, unsigned bz, unsigned smem, void** params, void** extra) {
  std::string s = "\"name\":\"";
  json_escape(s, name);
  char geo[160];
  snprintf(geo, sizeof geo, "\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"smem\":%u,\"params\":[", gx,
           gy, gz, bx, by, bz, smem);
  s += geo;
  if (params && layout) {
    for (size_t i = 0; i < layout->size(); ++i) {
      auto [off, size] = (*layout)[i];
      if (i) s += ',';
      s += "[" + std::to_string(off) + "," + std::to_string(size) + ",\"";
      append_hex(s, params[i], size);
      s += "\"]";
    }
  } else if (extra) {  // CU_LAUNCH_PARAM_BUFFER_POINTER / _SIZE: one packed buffer
    void* buf = nullptr;
    size_t size = 0;
    for (void** e = extra; *e != CU_LAUNCH_PARAM_END; e += 2) {
      if (*e == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = e[1];
      if (*e == CU_LAUNCH_PARAM_BUFFER_SIZE) size = *static_cast<size_t*>(e[1]);
    }
    if (buf) {
      s += "[0," + std::to_string(size) + ",\"";
      append_hex(s, buf, size);
      s += "\"]";
    }
  }
  s += "]";
  return s;
}

std::string describe(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                     unsigned bz, unsigned smem, void** params, void** extra) {
  const char* name = nullptr;
  const Layout* layout = layout_of(f, &name);
  return describe(name, layout, gx, gy, gz, bx, by, bz, smem, params, extra);
}

void write_line(Out& out, const std::string& line) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!out.f) out.f = open_out(out.stem);
  if (!out.f) return;
  fputs(line.c_str(), out.f);
  fputc('\n', out.f);
}

void record_graph_nodes(CUgraph graph) {
  size_t n = 0;
  if (cuGraphGetNodes(graph, nullptr, &n) != CUDA_SUCCESS || n == 0) return;
  auto* nodes = static_cast<CUgraphNode*>(malloc(n * sizeof(CUgraphNode)));
  cuGraphGetNodes(graph, nodes, &n);
  for (size_t i = 0; i < n; ++i) {
    CUgraphNodeType type;
    if (cuGraphNodeGetType(nodes[i], &type) != CUDA_SUCCESS || type != CU_GRAPH_NODE_TYPE_KERNEL) continue;
    CUDA_KERNEL_NODE_PARAMS p{};
    if (cuGraphKernelNodeGetParams(nodes[i], &p) != CUDA_SUCCESS) continue;
    CUfunction f = p.func;
    if (!f && p.kern) cuKernelGetFunction(&f, p.kern);
    uint64_t id = 0;
    cuptiGetGraphNodeId(nodes[i], &id);
    write_line(g_nodes, "{\"id\":" + std::to_string(id) + "," +
                            describe(f, p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX, p.blockDimY,
                                     p.blockDimZ, p.sharedMemBytes, p.kernelParams, p.extra) +
                            "}");
  }
  free(nodes);
}

void set_range(bool on) {
  if (on == g_in_range.exchange(on)) return;
  if (on) {
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  } else {
    cuptiActivityFlushAll(1);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityFlushAll(1);
  }
}

// nvcc gives a kernel's host stub the kernel's mangled name, so the stub address names the kernel.
const char* stub_name(const void* stub) {
  Dl_info info;
  return stub && dladdr(stub, &info) && info.dli_sname ? info.dli_sname : nullptr;
}

void on_direct_launch(const CUpti_CallbackData* cb, CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                      unsigned bx, unsigned by, unsigned bz, unsigned smem, void** params, void** extra,
                      const void* stub = nullptr) {
  if (!g_in_range.load()) return;
  const char* name = cb->symbolName && *cb->symbolName ? cb->symbolName : stub_name(stub);
  std::string body = f ? describe(f, gx, gy, gz, bx, by, bz, smem, params, extra)
                       : describe(name, layout_by_name(name), gx, gy, gz, bx, by, bz, smem, params, extra);
  write_line(g_launches, "{\"corr\":" + std::to_string(cb->correlationId) + "," + body + "}");
}

void on_driver(const CUpti_CallbackData* cb, CUpti_CallbackId id) {
  if (cb->callbackSite != CUPTI_API_ENTER) return;
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz: {
      auto* a = static_cast<const cuLaunchKernel_params*>(cb->functionParams);
      on_direct_launch(cb, a->f, a->gridDimX, a->gridDimY, a->gridDimZ, a->blockDimX, a->blockDimY,
                       a->blockDimZ, a->sharedMemBytes, a->kernelParams, a->extra);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
      auto* a = static_cast<const cuLaunchKernelEx_params*>(cb->functionParams);
      const CUlaunchConfig* c = a->config;
      on_direct_launch(cb, a->f, c->gridDimX, c->gridDimY, c->gridDimZ, c->blockDimX, c->blockDimY,
                       c->blockDimZ, c->sharedMemBytes, a->kernelParams, a->extra);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuProfilerStart: set_range(true); break;
    case CUPTI_DRIVER_TRACE_CBID_cuProfilerStop: set_range(false); break;
    default: break;
  }
}

// Runtime launches name a host stub; only the application's own cudart knows which function it
// registered, so look the symbol up in whatever cudart is already loaded instead of linking one.
// A failed lookup leaves an error in cudart's per-thread last-error slot, which the application
// would then report as its own; clear it.
CUfunction runtime_func(const void* symbol) {
  using GetFn = cudaError_t (*)(cudaFunction_t*, const void*);
  using ErrFn = cudaError_t (*)();
  // Python extensions load cudart with RTLD_LOCAL, so look for an already loaded copy by soname.
  static void* lib = [] {
    for (const char* soname : {"libcudart.so.13", "libcudart.so.12", "libcudart.so"})
      if (void* h = dlopen(soname, RTLD_NOW | RTLD_NOLOAD)) return h;
    return RTLD_DEFAULT;
  }();
  static GetFn get = reinterpret_cast<GetFn>(dlsym(lib, "cudaGetFuncBySymbol"));
  static ErrFn clear = reinterpret_cast<ErrFn>(dlsym(lib, "cudaGetLastError"));
  cudaFunction_t f = nullptr;
  if (!get) return nullptr;
  if (get(&f, symbol) != cudaSuccess) {
    if (clear) clear();
    return nullptr;
  }
  return reinterpret_cast<CUfunction>(f);
}

void on_runtime(const CUpti_CallbackData* cb, CUpti_CallbackId id) {
  if (cb->callbackSite != CUPTI_API_ENTER) return;
  switch (id) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000: {
      if (!g_in_range.load()) return;  // the symbol lookup below is not free; only pay it in range
      auto* a = static_cast<const cudaLaunchKernel_v7000_params*>(cb->functionParams);
      on_direct_launch(cb, runtime_func(a->func), a->gridDim.x, a->gridDim.y, a->gridDim.z,
                       a->blockDim.x, a->blockDim.y, a->blockDim.z, a->sharedMem, a->args, nullptr, a->func);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060: {
      if (!g_in_range.load()) return;
      auto* a = static_cast<const cudaLaunchKernelExC_v11060_params*>(cb->functionParams);
      const cudaLaunchConfig_t* c = a->config;
      on_direct_launch(cb, runtime_func(a->func), c->gridDim.x, c->gridDim.y, c->gridDim.z,
                       c->blockDim.x, c->blockDim.y, c->blockDim.z, c->dynamicSmemBytes, a->args, nullptr,
                       a->func);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000: set_range(true); break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000: set_range(false); break;
    default: break;
  }
}

void on_resource(const CUpti_ResourceData* r, CUpti_CallbackId id) {
  auto* g = static_cast<const CUpti_GraphData*>(r->resourceDescriptor);
  if (id == CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED) {
    record_graph_nodes(g->graph);
  } else if (id == CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED) {
    uint64_t node = 0, orig = 0;
    cuptiGetGraphNodeId(g->node, &node);
    cuptiGetGraphNodeId(g->originalNode, &orig);
    write_line(g_clones, "{\"id\":" + std::to_string(node) + ",\"orig\":" + std::to_string(orig) + "}");
  }
}

void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) on_driver(static_cast<const CUpti_CallbackData*>(data), id);
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) on_runtime(static_cast<const CUpti_CallbackData*>(data), id);
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) on_resource(static_cast<const CUpti_ResourceData*>(data), id);
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
  *size = 16 << 20;
  *buffer = static_cast<uint8_t*>(aligned_alloc(8, *size));
  *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* rec = nullptr;
  std::string line;
  while (cuptiActivityGetNextRecord(buffer, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
    auto* k = reinterpret_cast<CUpti_ActivityKernel10*>(rec);
    line = "{\"start\":" + std::to_string(k->start) + ",\"end\":" + std::to_string(k->end) +
           ",\"corr\":" + std::to_string(k->correlationId) + ",\"node\":" + std::to_string(k->graphNodeId) +
           ",\"graph\":" + std::to_string(k->graphId) + ",\"stream\":" + std::to_string(k->streamId) +
           ",\"grid\":[" + std::to_string(k->gridX) + "," + std::to_string(k->gridY) + "," +
           std::to_string(k->gridZ) + "],\"name\":\"";
    json_escape(line, k->name);
    line += "\"}";
    write_line(g_kernels, line);
  }
  free(buffer);
}

__attribute__((constructor)) void ledger_init() {
  cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
  cuptiSubscribe(&g_sub, callback, nullptr);
  const CUpti_CallbackId driver[] = {
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuProfilerStart,  CUPTI_DRIVER_TRACE_CBID_cuProfilerStop};
  for (auto id : driver) cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API, id);
  const CUpti_CallbackId runtime[] = {
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060,
      CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000,      CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000};
  for (auto id : runtime) cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_RUNTIME_API, id);
  cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED);
  cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED);
}

__attribute__((destructor)) void ledger_fini() {
  set_range(false);
  cuptiActivityFlushAll(1);
  for (Out* o : {&g_nodes, &g_clones, &g_launches, &g_kernels})
    if (o->f) fclose(o->f);
}

}  // namespace
