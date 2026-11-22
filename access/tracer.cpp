// Ledger access tracer. Load with LD_PRELOAD; built on the Compute Sanitizer API.
//
// For every kernel launch inside a window it records which granules of device memory the launch
// read and which it wrote, so that the analysis can follow data from writer to reader across the
// launches of a step. Weights are read-only and already accounted for by the capture library; the
// ranges listed in LEDGER_ACCESS_SKIP are left untracked.
//
// The window is the profiling range (cudaProfilerStart/Stop), as in the capture library, or is
// counted in graph launches with LEDGER_GRAPH_WINDOW=<skip>:<take>.
// Tensor-map copies (cp.async.bulk.tensor, TMA) are not instrumented by the Sanitizer API; a launch
// that moves data only that way shows no accesses here.
//
// Environment:
//   LEDGER_OUT              output directory (default ./ledger-out)
//   LEDGER_GRAPH_WINDOW     <skip>:<take>
//   LEDGER_ACCESS_ANCHOR    count the window in launches of this kernel (exact name) instead of graph
//                           launches; for programs that run graphs node by node
//   LEDGER_ACCESS_PATCHES   path of patches.fatbin
//   LEDGER_ACCESS_SKIP      text file, one "<begin> <end>" decimal byte range per line; or "auto": when the
//                           window opens, skip the weights listed in LEDGER_OUT/regions-<pid>.json
//   LEDGER_ACCESS_GRANULE   log2 of the granule size in bytes (default 10, at least 5)
//   LEDGER_ACCESS_SLOTS     log2 of the granule table size (default 22)
//   LEDGER_ACCESS_EVENTS    event log capacity (default 8388608)
//
// Launches made through the driver API also carry their parameters ("params": [[offset, size, hex], ...],
// as in the capture library), so the analysis can follow the pointers of launches it cannot see into.
//
// Output, per process: access-<pid>.bin (event words, see access.h), access-masks-<pid>.bin (one uint32
// sub-block mask per event), access-pcs-<pid>.bin (one uint64 PC per event; launches carry their
// function's start "pc"), access-launches-<pid>.jsonl
// ({"id","name","grid","block","graph","graph_launch"}; memcpy and memset calls take ids from the same
// sequence and appear as {"id","name":"memcpy"|"memset","reads":[[b,e]...],"writes":[[b,e]...]}) and
// access-meta-<pid>.json.

#include <sanitizer.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "access.h"

using namespace ledger_access;

namespace {

// Parameters of cuGraphLaunch as the driver API callback passes them (generated_cuda_meta.h).
struct GraphLaunchParams {
  CUgraphExec hGraph;
  CUstream hStream;
};
struct LaunchKernelParams {
  CUfunction f;
  unsigned int gx, gy, gz, bx, by, bz, smem;
  CUstream hStream;
  void** kernelParams;
};
struct LaunchKernelExParams {
  const CUlaunchConfig* config;
  CUfunction f;
  void** kernelParams;
};

std::string hex(const void* p, size_t n) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (size_t i = 0; i < n; ++i) {
    const unsigned char c = static_cast<const unsigned char*>(p)[i];
    out += digits[c >> 4];
    out += digits[c & 15];
  }
  return out;
}

// "[[offset,size,hex],...]" of a launch, or empty when the layout or the values are unavailable.
std::string param_json(CUfunction f, void** params) {
  if (!f || !params) return "";
  std::string out = "[";
  for (size_t i = 0;; ++i) {
    size_t off = 0, size = 0;
    if (cuFuncGetParamInfo(f, i, &off, &size) != CUDA_SUCCESS &&
        cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f), i, &off, &size) != CUDA_SUCCESS)
      break;
    if (i) out += ",";
    out += "[" + std::to_string(off) + "," + std::to_string(size) + ",\"" + hex(params[i], size) + "\"]";
  }
  return out + "]";
}

struct State {
  std::recursive_mutex mu;  // synchronizing inside a callback re-enters the synchronize callback
  std::string out = "ledger-out", patches;
  long skip = -1, take = 0;
  long graph_launches = 0;
  long current_graph_launch = -1;
  bool profiler_range = false;  // no LEDGER_GRAPH_WINDOW: follow cudaProfilerStart/Stop
  bool skip_auto = false;       // read the weights from this process's regions file
  std::string pending_params;   // of the driver launch whose launch callback comes next
  std::string anchor;
  long anchor_launches = 0;
  bool dump_pending = false;  // memory copies do not work inside a launch callback; dump at the next API call
  bool active = false, dumped = false, ready = false;
  unsigned int shift = 10, slots_log2 = 22, log_cap = 1u << 23;
  std::vector<unsigned long long> skip_ranges;
  CUcontext ctx = nullptr;
  Tables t{};
  Launch* d_launches = nullptr;  // kMaxLaunches + 1 entries; the last one is disabled
  unsigned int next_id = 0;
  std::vector<std::string> meta;
};

State& S() {
  static State* s = new State;  // never destroyed: callbacks may run during exit
  return *s;
}

// Start PC of a function, so that event PCs can be turned into offsets into its SASS.
uint64_t function_pc(CUmodule module, const char* name) {
  uint64_t pc = 0, size = 0;
  if (module && name && sanitizerGetFunctionPcAndSize(module, name, &pc, &size) == SANITIZER_SUCCESS) return pc;
  return 0;
}

std::string escape(const char* s) {
  std::string out;
  for (; s && *s; ++s) {
    if (*s == '"' || *s == '\\') out += '\\';
    if (static_cast<unsigned char>(*s) >= 0x20) out += *s;
  }
  return out;
}

std::vector<unsigned long long> merge(std::vector<std::pair<unsigned long long, unsigned long long>> r);

std::vector<unsigned long long> read_ranges(const char* path) {
  std::vector<std::pair<unsigned long long, unsigned long long>> r;
  std::ifstream in(path);
  unsigned long long b, e;
  while (in >> b >> e)
    if (e > b) r.emplace_back(b, e);
  return merge(std::move(r));
}

// Weights of a Ledger regions file (a JSON list of flat objects with "kind", "addr" and "size").
std::vector<unsigned long long> read_weight_regions(const std::string& path) {
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)), {});
  std::vector<std::pair<unsigned long long, unsigned long long>> r;
  const auto number = [&](const std::string& obj, const char* key) -> unsigned long long {
    const auto at = obj.find(key);
    return at == std::string::npos ? 0 : std::stoull(obj.substr(at + std::strlen(key)));
  };
  for (size_t start = text.find('{'); start != std::string::npos; start = text.find('{', start + 1)) {
    const auto end = text.find('}', start);
    if (end == std::string::npos) break;
    const std::string obj = text.substr(start, end - start);
    if (obj.find("\"kind\": \"weight\"") == std::string::npos) continue;
    const unsigned long long addr = number(obj, "\"addr\": "), size = number(obj, "\"size\": ");
    if (addr && size) r.emplace_back(addr, addr + size);
  }
  return merge(std::move(r));
}

std::vector<unsigned long long> merge(std::vector<std::pair<unsigned long long, unsigned long long>> r) {
  std::sort(r.begin(), r.end());
  std::vector<unsigned long long> merged;
  for (auto [b2, e2] : r) {
    if (!merged.empty() && b2 <= merged.back())
      merged.back() = std::max(merged.back(), e2);
    else {
      merged.push_back(b2);
      merged.push_back(e2);
    }
  }
  return merged;
}

bool check(SanitizerResult r, const char* what) {
  if (r == SANITIZER_SUCCESS) return true;
  fprintf(stderr, "ledger-access: %s failed (%d)\n", what, static_cast<int>(r));
  return false;
}

// Allocate and clear the tables in the context of the first window launch.
bool prepare(CUcontext ctx, Sanitizer_StreamHandle stream) {
  State& s = S();
  if (s.ready) return true;
  s.ctx = ctx;
  if (s.skip_auto) {
    s.skip_ranges = read_weight_regions(s.out + "/regions-" + std::to_string(getpid()) + ".json");
    fprintf(stderr, "ledger-access: pid %d skips %zu weight ranges from its regions file\n", getpid(),
            s.skip_ranges.size() / 2);
  }
  const size_t slots = size_t{1} << s.slots_log2;
  void *keys, *cur, *log, *masks, *pcs, *counters, *skip, *launches;
  const size_t n_launch = kMaxLaunches + 1;
  if (!check(sanitizerAlloc(ctx, &keys, slots * 8), "alloc keys") ||
      !check(sanitizerAlloc(ctx, &cur, slots * 16), "alloc cur") ||
      !check(sanitizerAlloc(ctx, &log, size_t{s.log_cap} * 8), "alloc log") ||
      !check(sanitizerAlloc(ctx, &masks, size_t{s.log_cap} * 4), "alloc masks") ||
      !check(sanitizerAlloc(ctx, &pcs, size_t{s.log_cap} * 8), "alloc pcs") ||
      !check(sanitizerAlloc(ctx, &counters, 32), "alloc counters") ||
      !check(sanitizerAlloc(ctx, &skip, std::max<size_t>(16, s.skip_ranges.size() * 8)), "alloc skip") ||
      !check(sanitizerAlloc(ctx, &launches, n_launch * sizeof(Launch)), "alloc launches"))
    return false;
  check(sanitizerMemset(keys, 0, slots * 8, stream), "clear keys");
  check(sanitizerMemset(cur, 0, slots * 16, stream), "clear cur");
  check(sanitizerMemset(masks, 0, size_t{s.log_cap} * 4, stream), "clear masks");
  check(sanitizerMemset(counters, 0, 32, stream), "clear counters");
  check(sanitizerMemset(log, 0, size_t{s.log_cap} * 8, stream), "clear log");
  if (!s.skip_ranges.empty())
    check(sanitizerMemcpyHostToDeviceAsync(skip, s.skip_ranges.data(), s.skip_ranges.size() * 8, stream), "copy skip");
  s.t = Tables{static_cast<unsigned long long*>(keys), static_cast<unsigned long long*>(cur),
               static_cast<unsigned long long*>(log), static_cast<unsigned int*>(masks),
               static_cast<unsigned long long*>(pcs),
               static_cast<unsigned int*>(counters),
               static_cast<const unsigned long long*>(skip), static_cast<unsigned int>(s.skip_ranges.size() / 2),
               s.log_cap, static_cast<unsigned int>(slots - 1), s.shift};
  static std::vector<Launch> host;  // kept alive until the async copy has run
  host.assign(n_launch, Launch{s.t, 0, 1});
  for (size_t i = 0; i < n_launch; ++i) host[i].id = static_cast<unsigned int>(i);
  host.back().enabled = 0;
  s.d_launches = static_cast<Launch*>(launches);
  check(sanitizerMemcpyHostToDeviceAsync(launches, host.data(), n_launch * sizeof(Launch), stream), "copy launches");
  check(sanitizerStreamSynchronize(stream), "synchronize");
  s.ready = true;
  return true;
}

void dump(Sanitizer_StreamHandle stream);

// Window by anchor kernel: opens at its launch numbered skip, closes at skip+take, and two anchor
// launches later drains the stream and writes the log.
void count_anchor(const Sanitizer_LaunchData& d) {
  State& s = S();
  if (s.anchor.empty() || !d.functionName || s.anchor != d.functionName) return;
  const long n = s.anchor_launches++;
  s.current_graph_launch = n;
  if (n == s.skip || n == s.skip + s.take)
    fprintf(stderr, "ledger-access: pid %d anchor launch %ld, window %s\n", getpid(), n,
            n == s.skip ? "opens" : "closes");
  if (n == s.skip) s.active = true;
  if (n == s.skip + s.take) s.active = false;
  if (n == s.skip + s.take + 2 && s.ready) s.dump_pending = true;
}

// Copies and fills are not kernels: the patches never see them, so record their byte ranges here.
// 2D/3D shapes are taken as the whole pitched span, which can only over-report.
std::string span(uint64_t base, uint64_t width, uint64_t height, uint64_t depth, uint64_t pitch) {
  const uint64_t rows = std::max<uint64_t>(1, height) * std::max<uint64_t>(1, depth);
  const uint64_t end = base + (rows > 1 ? pitch * (rows - 1) : 0) + width;
  return "[[" + std::to_string(base) + "," + std::to_string(end) + "]]";
}

void on_copy(const char* name, const std::string& reads, const std::string& writes) {
  State& s = S();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!s.active || s.next_id >= kMaxLaunches) return;
  const unsigned int id = s.next_id++;
  s.meta.push_back("{\"id\":" + std::to_string(id) + ",\"name\":\"" + name + "\",\"reads\":" + reads +
                   ",\"writes\":" + writes + ",\"graph_launch\":" + std::to_string(s.current_graph_launch) + "}");
}

void on_launch(const Sanitizer_LaunchData& d, bool graph) {
  State& s = S();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  count_anchor(d);
  Launch* data = nullptr;
  if (s.active && s.next_id < kMaxLaunches && prepare(d.context, d.hStream)) {
    const unsigned int id = s.next_id++;
    data = s.d_launches + id;
    s.meta.push_back("{\"id\":" + std::to_string(id) + ",\"name\":\"" + escape(d.functionName) + "\",\"grid\":[" +
                     std::to_string(d.gridDim_x) + "," + std::to_string(d.gridDim_y) + "," +
                     std::to_string(d.gridDim_z) + "],\"block\":[" + std::to_string(d.blockDim_x) + "," +
                     std::to_string(d.blockDim_y) + "," + std::to_string(d.blockDim_z) + "],\"graph\":" +
                     (graph ? "true" : "false") + ",\"graph_launch\":" + std::to_string(s.current_graph_launch) +
                     ",\"pc\":" + std::to_string(function_pc(d.module, d.functionName)) +
                     (!graph && !s.pending_params.empty() ? ",\"params\":" + s.pending_params : std::string()) + "}");
  } else if (s.ready) {
    data = s.d_launches + kMaxLaunches;
  }
  if (!graph) s.pending_params.clear();
  sanitizerSetLaunchCallbackData(d.hLaunch, d.function, graph ? d.hApiStream : d.hStream, data);
}

void dump(Sanitizer_StreamHandle stream) {
  State& s = S();
  if (s.dumped || !s.ready) return;
  s.dumped = true;
  unsigned int counters[8] = {};
  check(sanitizerMemcpyDeviceToHost(counters, s.t.counters, 32, stream), "read counters");
  const unsigned int n = std::min(counters[0], s.log_cap);
  std::vector<unsigned long long> log(n);
  std::vector<unsigned int> masks(n);
  std::vector<unsigned long long> pcs(n);
  if (n) {
    check(sanitizerMemcpyDeviceToHost(log.data(), s.t.log, size_t{n} * 8, stream), "read log");
    check(sanitizerMemcpyDeviceToHost(masks.data(), s.t.masks, size_t{n} * 4, stream), "read masks");
    check(sanitizerMemcpyDeviceToHost(pcs.data(), s.t.pcs, size_t{n} * 8, stream), "read pcs");
  }
  mkdir(s.out.c_str(), 0755);
  const std::string pid = std::to_string(getpid());
  if (FILE* f = fopen((s.out + "/access-" + pid + ".bin").c_str(), "wb")) {
    fwrite(log.data(), 8, n, f);
    fclose(f);
  }
  if (FILE* f = fopen((s.out + "/access-masks-" + pid + ".bin").c_str(), "wb")) {
    fwrite(masks.data(), 4, n, f);
    fclose(f);
  }
  if (FILE* f = fopen((s.out + "/access-pcs-" + pid + ".bin").c_str(), "wb")) {
    fwrite(pcs.data(), 8, n, f);
    fclose(f);
  }
  if (FILE* f = fopen((s.out + "/access-launches-" + pid + ".jsonl").c_str(), "w")) {
    for (const auto& line : s.meta) fprintf(f, "%s\n", line.c_str());
    fclose(f);
  }
  if (FILE* f = fopen((s.out + "/access-meta-" + pid + ".json").c_str(), "w")) {
    fprintf(f, "{\"granule_shift\":%u,\"events\":%u,\"dropped_events\":%u,\"table_overflow\":%u,\"launches\":%u,"
               "\"skip_ranges\":%zu,\"window\":[%ld,%ld]}\n",
            s.shift, counters[0], counters[1], counters[2], s.next_id, s.skip_ranges.size() / 2, s.skip, s.take);
    fclose(f);
  }
  fprintf(stderr, "ledger-access: pid %d wrote %u events for %u launches (dropped %u, table overflow %u, claim retries %u)\n",
          getpid(), n, s.next_id, counters[1], counters[2], counters[3]);
}

void patch_module(CUmodule module, CUcontext ctx) {
  State& s = S();
  if (!check(sanitizerAddPatchesFromFile(s.patches.c_str(), ctx), "add patches")) return;
  sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "LedgerGlobal");
  sanitizerPatchInstructions(SANITIZER_INSTRUCTION_MEMCPY_ASYNC, module, "LedgerMemcpyAsync");
  sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BULK_COPY_GLOBAL_TO_SHARED, module, "LedgerBulkGlobalToShared");
  sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BULK_COPY_SHARED_TO_GLOBAL, module, "LedgerBulkSharedToGlobal");
  sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BULK_REDUCTION_SHARED_TO_GLOBAL, module,
                             "LedgerBulkReduceSharedToGlobal");
  check(sanitizerPatchModule(module), "patch module");
}

void SANITIZERAPI callback(void*, Sanitizer_CallbackDomain domain, Sanitizer_CallbackId cbid, const void* data) {
  State& s = S();
  switch (domain) {
    case SANITIZER_CB_DOMAIN_RESOURCE:
      if (cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
        auto* m = static_cast<const Sanitizer_ResourceModuleData*>(data);
        patch_module(m->module, m->context);
      }
      break;
    case SANITIZER_CB_DOMAIN_DRIVER_API: {
      auto* cb = static_cast<const Sanitizer_CallbackData*>(data);
      if (cb->callbackSite != SANITIZER_API_ENTER) break;
      std::lock_guard<std::recursive_mutex> lock(s.mu);
      if (s.active && (cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernel || cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernel_ptsz)) {
        auto* p = static_cast<const LaunchKernelParams*>(cb->functionParams);
        s.pending_params = param_json(p->f, p->kernelParams);
      } else if (s.active && (cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx ||
                              cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx_ptsz)) {
        auto* p = static_cast<const LaunchKernelExParams*>(cb->functionParams);
        s.pending_params = param_json(p->f, p->kernelParams);
      }
      if (s.profiler_range) {
        if (cbid == SANITIZER_CBID_DRIVER_API_cuProfilerStart) {
          s.active = true;
          s.current_graph_launch = 0;
          fprintf(stderr, "ledger-access: pid %d profiler range opens\n", getpid());
        } else if (cbid == SANITIZER_CBID_DRIVER_API_cuProfilerStop && s.active) {
          s.active = false;
          fprintf(stderr, "ledger-access: pid %d profiler range closes\n", getpid());
          Sanitizer_StreamHandle h = nullptr;
          cuCtxSynchronize();
          if (s.ready && check(sanitizerGetStreamHandle(cb->context, nullptr, &h), "stream handle")) dump(h);
        }
        break;
      }
      if (!s.anchor.empty()) {
        if (!s.dump_pending || s.dumped) break;
        CUstream stream = nullptr;
        if (cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernel || cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernel_ptsz)
          stream = static_cast<const LaunchKernelParams*>(cb->functionParams)->hStream;
        else if (cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx || cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx_ptsz)
          stream = static_cast<const LaunchKernelExParams*>(cb->functionParams)->config->hStream;
        else
          break;
        Sanitizer_StreamHandle h = nullptr;
        if (check(sanitizerGetStreamHandle(cb->context, stream, &h), "stream handle")) {
          check(sanitizerStreamSynchronize(h), "synchronize");
          dump(h);
        }
        break;
      }
      if (cbid != SANITIZER_CBID_DRIVER_API_cuGraphLaunch && cbid != SANITIZER_CBID_DRIVER_API_cuGraphLaunch_ptsz) break;
      const long n = s.graph_launches++;
      s.current_graph_launch = n;
      if (n == s.skip || n == s.skip + s.take)
        fprintf(stderr, "ledger-access: pid %d graph launch %ld, window %s\n", getpid(), n,
                n == s.skip ? "opens" : "closes");
      if (n == s.skip) s.active = true;
      if (n == s.skip + s.take) s.active = false;
      // The program may wait for its results without a synchronize call (polling mapped memory),
      // so do not count on a synchronize callback: two graph launches after the window, drain the
      // stream and write the log.
      if (n == s.skip + s.take + 2 && s.ready) {
        Sanitizer_StreamHandle h = nullptr;
        auto* p = static_cast<const GraphLaunchParams*>(cb->functionParams);
        if (check(sanitizerGetStreamHandle(cb->context, p->hStream, &h), "stream handle")) {
          sanitizerStreamSynchronize(h);
          dump(h);
        }
      }
      break;
    }
    case SANITIZER_CB_DOMAIN_LAUNCH:
      if (cbid == SANITIZER_CBID_LAUNCH_BEGIN) on_launch(*static_cast<const Sanitizer_LaunchData*>(data), false);
      break;
    case SANITIZER_CB_DOMAIN_GRAPHS:
      if (cbid == SANITIZER_CBID_GRAPHS_NODE_LAUNCH_BEGIN) {
        auto* g = static_cast<const Sanitizer_GraphNodeLaunchData*>(data);
        if (g->nodeType == CU_GRAPH_NODE_TYPE_KERNEL && !g->isGraphUpload) on_launch(g->launchData, true);
      }
      break;
    case SANITIZER_CB_DOMAIN_MEMCPY:
      if (cbid == SANITIZER_CBID_MEMCPY_STARTING) {
        auto* m = static_cast<const Sanitizer_MemcpyData*>(data);
        const uint64_t w = m->width ? m->width : m->size;
        const auto dir = m->direction;  // unknown (unified addressing): take both sides as device memory
        const bool src_dev = dir != SANITIZER_MEMCPY_DIRECTION_HOST_TO_DEVICE && dir != SANITIZER_MEMCPY_DIRECTION_HOST_TO_HOST;
        const bool dst_dev = dir != SANITIZER_MEMCPY_DIRECTION_DEVICE_TO_HOST && dir != SANITIZER_MEMCPY_DIRECTION_HOST_TO_HOST;
        on_copy("memcpy", src_dev ? span(m->srcAddress, w, m->height, m->depth, m->srcPitch) : "[]",
                dst_dev ? span(m->dstAddress, w, m->height, m->depth, m->dstPitch) : "[]");
      }
      break;
    case SANITIZER_CB_DOMAIN_MEMSET:
      if (cbid == SANITIZER_CBID_MEMSET_STARTING) {
        auto* m = static_cast<const Sanitizer_MemsetData*>(data);
        on_copy("memset", "[]", span(m->address, m->width * std::max(1u, m->elementSize), m->height, 1, m->pitch));
      }
      break;
    case SANITIZER_CB_DOMAIN_SYNCHRONIZE: {
      auto* d = static_cast<const Sanitizer_SynchronizeData*>(data);
      std::lock_guard<std::recursive_mutex> lock(s.mu);
      if (s.anchor.empty() && !s.active && s.graph_launches > s.skip + s.take && s.skip >= 0) dump(d->hStream);
      break;
    }
    default:
      break;
  }
}

int init() {
  State& s = S();
  if (const char* v = getenv("LEDGER_OUT")) s.out = v;
  if (const char* v = getenv("LEDGER_ACCESS_PATCHES")) s.patches = v;
  if (const char* v = getenv("LEDGER_GRAPH_WINDOW")) {
    if (sscanf(v, "%ld:%ld", &s.skip, &s.take) != 2 || s.skip < 0 || s.take <= 0) s.skip = -1;
  }
  if (const char* v = getenv("LEDGER_ACCESS_SKIP")) {
    if (std::string(v) == "auto")
      s.skip_auto = true;
    else
      s.skip_ranges = read_ranges(v);
  }
  if (const char* v = getenv("LEDGER_ACCESS_ANCHOR")) s.anchor = v;
  if (const char* v = getenv("LEDGER_ACCESS_GRANULE")) s.shift = static_cast<unsigned int>(atoi(v));
  if (const char* v = getenv("LEDGER_ACCESS_SLOTS")) s.slots_log2 = static_cast<unsigned int>(atoi(v));
  if (const char* v = getenv("LEDGER_ACCESS_EVENTS")) s.log_cap = static_cast<unsigned int>(atoll(v));
  if (s.patches.empty()) {
    fprintf(stderr, "ledger-access: set LEDGER_ACCESS_PATCHES\n");
    return 0;
  }
  s.profiler_range = s.skip < 0;
  Sanitizer_SubscriberHandle handle;
  if (!check(sanitizerSubscribe(&handle, callback, nullptr), "subscribe")) return 0;
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_RESOURCE, SANITIZER_CBID_RESOURCE_MODULE_LOADED);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_DRIVER_API, SANITIZER_CBID_DRIVER_API_cuGraphLaunch);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_DRIVER_API, SANITIZER_CBID_DRIVER_API_cuGraphLaunch_ptsz);
  for (auto id : {SANITIZER_CBID_DRIVER_API_cuLaunchKernel, SANITIZER_CBID_DRIVER_API_cuLaunchKernel_ptsz,
                  SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx, SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx_ptsz})
    sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_DRIVER_API, id);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_DRIVER_API, SANITIZER_CBID_DRIVER_API_cuProfilerStart);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_DRIVER_API, SANITIZER_CBID_DRIVER_API_cuProfilerStop);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_LAUNCH, SANITIZER_CBID_LAUNCH_BEGIN);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_GRAPHS, SANITIZER_CBID_GRAPHS_NODE_LAUNCH_BEGIN);
  sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_SYNCHRONIZE);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_MEMCPY, SANITIZER_CBID_MEMCPY_STARTING);
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_MEMSET, SANITIZER_CBID_MEMSET_STARTING);
  if (s.profiler_range)
    fprintf(stderr, "ledger-access: profiler range window, granule %u B, %zu skipped ranges\n", 1u << s.shift,
            s.skip_ranges.size() / 2);
  else
    fprintf(stderr, "ledger-access: window %ld:%ld, granule %u B, %zu skipped ranges\n", s.skip, s.take,
            1u << s.shift, s.skip_ranges.size() / 2);
  return 0;
}

const int initialized = init();

}  // namespace
