// Ledger access tracer. Load with LD_PRELOAD; built on the Compute Sanitizer API.
//
// For every kernel launch inside a window it records which granules of device memory the launch
// read and which it wrote, so that the analysis can follow data from writer to reader across the
// launches of a step. Weights are read-only and already accounted for by the capture library; the
// ranges listed in LEDGER_ACCESS_SKIP are left untracked.
//
// The window is the profiling range (cudaProfilerStart/Stop), as in the capture library, or is
// counted in graph launches with LEDGER_GRAPH_WINDOW=<skip>:<take>.
//
// The Sanitizer API gives every node of a CUDA graph the callback data of the last one, so accesses
// inside a graph cannot be told apart. The tracer keeps a copy of each graph as it is instantiated and,
// when the program launches it, launches its nodes one by one on the same stream instead, then an empty
// graph in its place.
//
// Tensor-map copies (cp.async.bulk.tensor, TMA) are not instrumented by the Sanitizer API. The tracer
// finds the tensor maps among a launch's parameters, takes the launch to read every tensor they span,
// and finds what it wrote by comparing those tensors before and after it: a lower bound, since a write
// of the value already there does not show.
//
// Environment:
//   LEDGER_OUT              output directory (default ./ledger-out)
//   LEDGER_GRAPH_WINDOW     <skip>:<take>
//   LEDGER_ACCESS_ANCHOR    count the window in launches of this kernel (exact name) instead of graph
//                           launches; for programs that run graphs node by node
//   LEDGER_ACCESS_PATCHES   path of patches.fatbin
//   LEDGER_ACCESS_SKIP      text file, one "<begin> <end>" decimal byte range per line; or "auto": when the
//                           window opens, skip the weights listed in LEDGER_OUT/regions-<pid>.json, except
//                           those whose name contains LEDGER_ACCESS_TRACE_WEIGHTS (e.g. MoE "experts")
//   LEDGER_ACCESS_GRANULE   log2 of the granule size in bytes (default 10, at least 5)
//   LEDGER_ACCESS_SLOTS     log2 of the granule table size (default 22)
//   LEDGER_ACCESS_EVENTS    event log capacity (default 8388608)
//   LEDGER_ACCESS_UNROLL    0: leave graphs alone (default 1, launch their nodes one by one)
//   LEDGER_ACCESS_TMA_DIFF  most bytes of tensors compared around one tensor-map launch (default 1 GiB);
//                           a launch over it, or with 0, gets "tma_partial"
//
// Launches also carry their parameters ("params": [[offset, size, hex], ...], as in the capture
// library), so the analysis can follow the pointers of launches it cannot see into. Launches with
// tensor maps carry "tma_reads" and "tma_writes" ([[begin, end], ...]; weights left out of the reads).
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
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
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

// The tracer's own driver calls (synchronize, copy, graph queries) re-enter the callback; it ignores them.
thread_local int internal = 0;
struct Internal {
  Internal() { ++internal; }
  ~Internal() { --internal; }
  Internal(const Internal&) = delete;
  Internal& operator=(const Internal&) = delete;
};

struct Param {
  size_t off, size;
  const unsigned char* bytes;
};

std::vector<Param> params_of(CUfunction f, void** params) {
  std::vector<Param> out;
  if (!f || !params) return out;
  Internal in;
  for (size_t i = 0;; ++i) {
    size_t off = 0, size = 0;
    if (cuFuncGetParamInfo(f, i, &off, &size) != CUDA_SUCCESS &&
        cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f), i, &off, &size) != CUDA_SUCCESS)
      break;
    out.push_back({off, size, static_cast<const unsigned char*>(params[i])});
  }
  return out;
}

// "[[offset,size,hex],...]" of a launch, or empty when the layout or the values are unavailable.
std::string param_json(const std::vector<Param>& ps) {
  if (ps.empty()) return "";
  std::string out = "[";
  for (const Param& p : ps)
    out += (out.size() > 1 ? ",[" : "[") + std::to_string(p.off) + "," + std::to_string(p.size) + ",\"" +
           hex(p.bytes, p.size) + "\"]";
  return out + "]";
}

struct Span {
  uint64_t b, e;
};

std::string spans_json(const std::vector<Span>& v) {
  std::string out = "[";
  for (const Span& s : v) out += (out.size() > 1 ? ",[" : "[") + std::to_string(s.b) + "," + std::to_string(s.e) + "]";
  return out + "]";
}

std::vector<Span> merge_spans(std::vector<Span> v) {
  std::sort(v.begin(), v.end(), [](const Span& a, const Span& b) { return a.b < b.b; });
  std::vector<Span> out;
  for (const Span& s : v) {
    if (!out.empty() && s.b <= out.back().e)
      out.back().e = std::max(out.back().e, s.e);
    else
      out.push_back(s);
  }
  return out;
}

// The bytes a tensor map spans, if the 128 bytes at m are one. The layout is the driver's for sm_90 to
// sm_121, found by encoding maps with one parameter varied at a time: bytes 0-7 the global address;
// byte 8 bit 0 im2col, bits 4-6 rank - 1; byte 8 bit 7 and byte 9 bits 0-2 a data-type code; u32 at
// 12 + 4i the stride of dimension i + 1 in 16-byte units, its bits 32-35 in nibble i from byte 28;
// u32 at 32 + 4i the size of dimension i, minus one; bytes 80 to 127 are zero.
bool decode_map(const unsigned char* m, Span& out) {
  static const unsigned bits[] = {8, 16, 32, 32, 64, 64, 16, 32, 32, 64, 16, 4, 4, 6};  // per type code
  const auto u32 = [m](int o) {
    uint32_t v;
    memcpy(&v, m + o, 4);
    return uint64_t{v};
  };
  uint64_t addr;
  memcpy(&addr, m, 8);
  const unsigned rank = ((m[8] >> 4) & 7) + 1, code = ((m[9] & 7) << 1) | (m[8] >> 7);
  if (!addr || (addr & 15) || rank > 5 || code >= std::size(bits)) return false;
  for (int i = 80; i < 128; ++i)
    if (m[i]) return false;
  uint64_t end = addr + ((u32(32) + 1) * bits[code] + 7) / 8;
  for (unsigned i = 1; i < rank; ++i) {
    const uint64_t high = (m[28 + (i - 1) / 2] >> ((i - 1) % 2 * 4)) & 15;
    const uint64_t stride = (u32(12 + 4 * (i - 1)) | high << 32) << 4;  // 0 for a broadcast (batch) dimension
    end += u32(32 + 4 * i) * stride;
  }
  out = {addr, end};
  return true;
}

// The allocations covering the span from its start, cut where device memory ends: a map may be encoded
// for more rows than its tensor holds, and a launch cannot touch memory that is not there. Empty when
// the start is not device memory.
std::vector<Span> on_device(const Span& sp) {
  Internal in;
  std::vector<Span> pieces;
  for (uint64_t at = sp.b; at < sp.e;) {
    CUdeviceptr base;
    size_t size;
    if (cuMemGetAddressRange(&base, &size, at) != CUDA_SUCCESS) break;
    const uint64_t end = std::min<uint64_t>(sp.e, base + size);
    pieces.push_back({at, end});
    at = end;
  }
  return pieces;
}

void warn_copy(CUresult r) {
  static bool warned = false;
  if (!warned) fprintf(stderr, "ledger-access: pid %d cannot copy a tensor for comparison (%d)\n", getpid(), static_cast<int>(r));
  warned = true;
}

// A launch with tensor maps: the tensors it may read, and copies of them from before it ran.
struct TmaLaunch {
  long id = -1;
  std::vector<Span> reads, watched;
  std::vector<std::vector<unsigned char>> before;
  bool partial = false;  // a tensor left out of the comparison
};

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
  std::vector<unsigned long long> weights;  // every weight, traced or not: tensor maps over them are only read
  bool weights_loaded = false;
  bool unroll = true;
  std::unordered_map<CUgraphExec, CUgraph> sources;  // a copy of the graph behind each executable graph
  CUgraphExec empty_graph = nullptr;
  uint64_t tma_diff_max = 1ull << 30;
  std::unique_ptr<TmaLaunch> tma_next, tma_wait;  // of the launch being made; of the last one made
  std::map<unsigned int, std::string> extra;      // more fields of a launch record, by id
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
// Weights whose name contains `keep` (LEDGER_ACCESS_TRACE_WEIGHTS, e.g. "experts") stay traced.
std::vector<unsigned long long> read_weight_regions(const std::string& path, const std::string& keep) {
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
    if (!keep.empty() && obj.find(keep) != std::string::npos) continue;
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

// The weights, once the window opens (the capture library writes the regions file before that).
void load_weights() {
  State& s = S();
  if (s.weights_loaded) return;
  s.weights_loaded = true;
  if (!s.skip_auto) {
    s.weights = s.skip_ranges;
    return;
  }
  const std::string path = s.out + "/regions-" + std::to_string(getpid()) + ".json";
  const char* keep = getenv("LEDGER_ACCESS_TRACE_WEIGHTS");
  s.skip_ranges = read_weight_regions(path, keep ? keep : "");
  s.weights = read_weight_regions(path, "");
  fprintf(stderr, "ledger-access: pid %d skips %zu weight ranges from its regions file\n", getpid(),
          s.skip_ranges.size() / 2);
}

bool inside_weight(const Span& sp) {
  const auto& w = S().weights;
  const auto it = std::upper_bound(w.begin(), w.end(), sp.b);  // first bound past sp.b
  return (it - w.begin()) % 2 == 1 && sp.e <= *it;
}

// Allocate and clear the tables in the context of the first window launch.
bool prepare(CUcontext ctx, Sanitizer_StreamHandle stream) {
  State& s = S();
  if (s.ready) return true;
  s.ctx = ctx;
  load_weights();
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

// At the entry of a launch: its tensor maps, and copies of the tensors it may write.
void prepare_tma(const std::vector<Param>& ps, CUstream stream) {
  State& s = S();
  std::vector<Span> maps;
  for (const Param& p : ps)
    for (size_t k = (64 - p.off % 64) % 64; k + 128 <= p.size; k += 64) {
      Span sp;
      if (decode_map(p.bytes + k, sp))
        for (const Span& piece : on_device(sp)) maps.push_back(piece);
    }
  if (maps.empty()) return;
  load_weights();
  auto t = std::make_unique<TmaLaunch>();
  // Compared allocation by allocation: neighbouring allocations merge into a span no single copy may cross.
  std::sort(maps.begin(), maps.end(), [](const Span& x, const Span& y) { return x.b != y.b ? x.b < y.b : x.e < y.e; });
  maps.erase(std::unique(maps.begin(), maps.end(), [](const Span& x, const Span& y) { return x.b == y.b && x.e == y.e; }),
             maps.end());
  maps.erase(std::remove_if(maps.begin(), maps.end(), inside_weight), maps.end());
  uint64_t total = 0;
  for (const Span& m : maps) total += m.e - m.b;
  t->reads = merge_spans(maps);
  Internal in;
  CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
  cuStreamIsCapturing(stream, &capture);
  if (total > s.tma_diff_max || capture != CU_STREAM_CAPTURE_STATUS_NONE) {
    t->partial = !maps.empty();
  } else if (!maps.empty()) {
    cuCtxSynchronize();
    for (const Span& r : maps) {
      std::vector<unsigned char> b(r.e - r.b);
      if (const CUresult e = cuMemcpyDtoH(b.data(), r.b, b.size()); e != CUDA_SUCCESS) {
        warn_copy(e);
        t->partial = true;
        continue;
      }
      t->watched.push_back(r);
      t->before.push_back(std::move(b));
    }
  }
  s.tma_next = std::move(t);
}

// The last tensor-map launch has finished by the time the program makes its next call: what changed.
void resolve_tma() {
  State& s = S();
  if (!s.tma_wait) return;
  const std::unique_ptr<TmaLaunch> t = std::move(s.tma_wait);
  std::vector<Span> writes;
  Internal in;
  cuCtxSynchronize();
  for (size_t i = 0; i < t->watched.size(); ++i) {
    const Span& r = t->watched[i];
    std::vector<unsigned char> after(r.e - r.b);
    if (const CUresult e = cuMemcpyDtoH(after.data(), r.b, after.size()); e != CUDA_SUCCESS) {
      warn_copy(e);
      t->partial = true;
      continue;
    }
    for (size_t o = 0; o < after.size(); o += 32) {
      const size_t n = std::min<size_t>(32, after.size() - o);
      if (!memcmp(after.data() + o, t->before[i].data() + o, n)) continue;
      if (!writes.empty() && writes.back().e == r.b + o)
        writes.back().e += n;
      else
        writes.push_back({r.b + o, r.b + o + n});
    }
  }
  s.extra[t->id] = ",\"tma_reads\":" + spans_json(t->reads) + ",\"tma_writes\":" + spans_json(merge_spans(writes)) +
                   (t->partial ? ",\"tma_partial\":true" : "");
}

// Parameters of the instantiate calls, which all start alike (generated_cuda_meta.h).
struct InstantiateParams {
  CUgraphExec* exec;
  CUgraph graph;
};

bool is_instantiate(Sanitizer_CallbackId cbid) {
  return cbid == SANITIZER_CBID_DRIVER_API_cuGraphInstantiate || cbid == SANITIZER_CBID_DRIVER_API_cuGraphInstantiate_v2 ||
         cbid == SANITIZER_CBID_DRIVER_API_cuGraphInstantiateWithFlags ||
         cbid == SANITIZER_CBID_DRIVER_API_cuGraphInstantiateWithParams ||
         cbid == SANITIZER_CBID_DRIVER_API_cuGraphInstantiateWithParams_ptsz;
}

// Keep a copy of the graph: the program may destroy it once instantiated.
void on_instantiated(const Sanitizer_CallbackData* cb) {
  State& s = S();
  if (!s.unroll || *static_cast<const CUresult*>(cb->functionReturnValue) != CUDA_SUCCESS) return;
  auto* p = static_cast<const InstantiateParams*>(cb->functionParams);
  Internal in;
  CUgraph copy;
  if (cuGraphClone(&copy, p->graph) != CUDA_SUCCESS) return;
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  CUgraph& slot = s.sources[*p->exec];
  if (slot) cuGraphDestroy(slot);
  slot = copy;
}

void forget(CUgraphExec exec) {
  State& s = S();
  const auto it = s.sources.find(exec);
  if (it == s.sources.end()) return;
  Internal in;
  cuGraphDestroy(it->second);
  s.sources.erase(it);
}

// The nodes of a graph in an order its edges allow: of the ready nodes, the first added runs first.
std::vector<CUgraphNode> node_order(CUgraph g) {
  Internal in;
  size_t n = 0, m = 0;
  cuGraphGetNodes(g, nullptr, &n);
  std::vector<CUgraphNode> nodes(n);
  cuGraphGetNodes(g, nodes.data(), &n);
  cuGraphGetEdges(g, nullptr, nullptr, nullptr, &m);
  std::vector<CUgraphNode> from(m), to(m);
  std::vector<CUgraphEdgeData> data(m);
  cuGraphGetEdges(g, from.data(), to.data(), data.data(), &m);
  std::unordered_map<CUgraphNode, size_t> index;
  for (size_t i = 0; i < n; ++i) index[nodes[i]] = i;
  std::vector<size_t> waiting(n);
  std::vector<std::vector<size_t>> next(n);
  for (size_t e = 0; e < m; ++e) {
    next[index[from[e]]].push_back(index[to[e]]);
    ++waiting[index[to[e]]];
  }
  std::set<size_t> ready;
  for (size_t i = 0; i < n; ++i)
    if (!waiting[i]) ready.insert(i);
  std::vector<CUgraphNode> order;
  while (!ready.empty()) {
    const size_t i = *ready.begin();
    ready.erase(ready.begin());
    order.push_back(nodes[i]);
    for (size_t j : next[i])
      if (!--waiting[j]) ready.insert(j);
  }
  return order;
}

bool run_graph(CUgraph g, CUstream stream, bool run);

// Issue one node on the stream (or, with run false, only check that it can be).
bool run_node(CUgraphNode node, CUstream stream, bool run) {
  CUgraphNodeType type;
  {
    Internal in;
    if (cuGraphNodeGetType(node, &type) != CUDA_SUCCESS) return false;
  }
  switch (type) {
    case CU_GRAPH_NODE_TYPE_EMPTY:
      return true;
    case CU_GRAPH_NODE_TYPE_KERNEL: {
      if (!run) return true;
      CUDA_KERNEL_NODE_PARAMS p{};
      CUlaunchAttribute attrs[2];
      unsigned int n_attrs = 0;
      {
        Internal in;
        if (cuGraphKernelNodeGetParams(node, &p) != CUDA_SUCCESS) return false;
        if (!p.func && p.kern && cuKernelGetFunction(&p.func, p.kern) != CUDA_SUCCESS) return false;
        CUlaunchAttributeValue v{};
        if (cuGraphKernelNodeGetAttribute(node, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, &v) == CUDA_SUCCESS &&
            v.clusterDim.x * v.clusterDim.y * v.clusterDim.z > 1)
          attrs[n_attrs++] = {CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, {}, v};
        v = {};
        if (cuGraphKernelNodeGetAttribute(node, CU_LAUNCH_ATTRIBUTE_COOPERATIVE, &v) == CUDA_SUCCESS && v.cooperative)
          attrs[n_attrs++] = {CU_LAUNCH_ATTRIBUTE_COOPERATIVE, {}, v};
      }
      CUlaunchConfig c{p.gridDimX, p.gridDimY,      p.gridDimZ, p.blockDimX, p.blockDimY,
                       p.blockDimZ, p.sharedMemBytes, stream,     attrs,       n_attrs};
      return cuLaunchKernelEx(&c, p.func, p.kernelParams, p.extra) == CUDA_SUCCESS;
    }
    case CU_GRAPH_NODE_TYPE_MEMSET: {
      CUDA_MEMSET_NODE_PARAMS p{};
      {
        Internal in;
        if (cuGraphMemsetNodeGetParams(node, &p) != CUDA_SUCCESS) return false;
      }
      if (!run) return true;
      const size_t pitch = p.pitch ? p.pitch : p.width * p.elementSize;
      if (p.elementSize == 1) return cuMemsetD2D8Async(p.dst, pitch, static_cast<unsigned char>(p.value), p.width, p.height, stream) == CUDA_SUCCESS;
      if (p.elementSize == 2) return cuMemsetD2D16Async(p.dst, pitch, static_cast<unsigned short>(p.value), p.width, p.height, stream) == CUDA_SUCCESS;
      return cuMemsetD2D32Async(p.dst, pitch, p.value, p.width, p.height, stream) == CUDA_SUCCESS;
    }
    case CU_GRAPH_NODE_TYPE_MEMCPY: {
      CUDA_MEMCPY3D p{};
      {
        Internal in;
        if (cuGraphMemcpyNodeGetParams(node, &p) != CUDA_SUCCESS) return false;
      }
      return !run || cuMemcpy3DAsync(&p, stream) == CUDA_SUCCESS;
    }
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: {
      CUevent ev;
      {
        Internal in;
        if ((type == CU_GRAPH_NODE_TYPE_EVENT_RECORD ? cuGraphEventRecordNodeGetEvent(node, &ev)
                                                      : cuGraphEventWaitNodeGetEvent(node, &ev)) != CUDA_SUCCESS)
          return false;
      }
      if (!run) return true;
      return (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD ? cuEventRecord(ev, stream) : cuStreamWaitEvent(stream, ev, 0)) ==
             CUDA_SUCCESS;
    }
    case CU_GRAPH_NODE_TYPE_HOST: {
      CUDA_HOST_NODE_PARAMS p{};
      {
        Internal in;
        if (cuGraphHostNodeGetParams(node, &p) != CUDA_SUCCESS) return false;
      }
      return !run || cuLaunchHostFunc(stream, p.fn, p.userData) == CUDA_SUCCESS;
    }
    case CU_GRAPH_NODE_TYPE_GRAPH: {
      CUgraph child;
      {
        Internal in;
        if (cuGraphChildGraphNodeGetGraph(node, &child) != CUDA_SUCCESS) return false;
      }
      return run_graph(child, stream, run);
    }
    default:
      return false;  // memory allocation, conditional, external semaphore and batch nodes
  }
}

bool run_graph(CUgraph g, CUstream stream, bool run) {
  for (CUgraphNode node : node_order(g))
    if (!run_node(node, stream, run)) return false;
  return true;
}

// At the entry of cuGraphLaunch: launch the graph's nodes one by one, then let the call launch an
// empty graph.
void unroll(const Sanitizer_CallbackData* cb) {
  State& s = S();
  auto* p = static_cast<GraphLaunchParams*>(const_cast<void*>(cb->functionParams));
  const auto it = s.sources.find(p->hGraph);
  if (!s.unroll || s.dumped || it == s.sources.end()) return;
  if (!run_graph(it->second, p->hStream, false)) {
    static bool warned = false;
    if (!warned) fprintf(stderr, "ledger-access: pid %d a graph holds nodes that cannot run one by one; left whole\n", getpid());
    warned = true;
    return;
  }
  if (!s.empty_graph) {
    Internal in;
    CUgraph g;
    cuGraphCreate(&g, 0);
    cuGraphInstantiateWithFlags(&s.empty_graph, g, 0);
    cuGraphDestroy(g);
  }
  if (!run_graph(it->second, p->hStream, true))
    fprintf(stderr, "ledger-access: pid %d a graph node failed to launch on its own\n", getpid());
  p->hGraph = s.empty_graph;
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
    if (!graph && s.tma_next) {
      s.tma_wait = std::move(s.tma_next);
      s.tma_wait->id = id;
    }
  } else if (s.ready) {
    data = s.d_launches + kMaxLaunches;
  }
  if (!graph) s.pending_params.clear();
  sanitizerSetLaunchCallbackData(d.hLaunch, d.function, graph ? d.hApiStream : d.hStream, data);
}

void dump(Sanitizer_StreamHandle stream) {
  State& s = S();
  if (s.dumped || !s.ready) return;
  resolve_tma();
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
    for (size_t i = 0; i < s.meta.size(); ++i) {  // meta[i] is the record of id i
      const auto x = s.extra.find(static_cast<unsigned int>(i));
      const std::string& line = s.meta[i];
      if (x == s.extra.end())
        fprintf(f, "%s\n", line.c_str());
      else
        fprintf(f, "%s%s}\n", line.substr(0, line.size() - 1).c_str(), x->second.c_str());
    }
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
  if (internal && domain != SANITIZER_CB_DOMAIN_RESOURCE && domain != SANITIZER_CB_DOMAIN_LAUNCH) return;
  switch (domain) {
    case SANITIZER_CB_DOMAIN_RESOURCE:
      if (cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
        auto* m = static_cast<const Sanitizer_ResourceModuleData*>(data);
        patch_module(m->module, m->context);
      }
      break;
    case SANITIZER_CB_DOMAIN_DRIVER_API: {
      auto* cb = static_cast<const Sanitizer_CallbackData*>(data);
      if (cb->callbackSite != SANITIZER_API_ENTER) {
        if (is_instantiate(cbid)) on_instantiated(cb);
        break;
      }
      std::lock_guard<std::recursive_mutex> lock(s.mu);
      resolve_tma();
      s.tma_next.reset();
      const bool launch = cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernel || cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernel_ptsz;
      const bool launch_ex = cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx || cbid == SANITIZER_CBID_DRIVER_API_cuLaunchKernelEx_ptsz;
      const bool graph_launch = cbid == SANITIZER_CBID_DRIVER_API_cuGraphLaunch || cbid == SANITIZER_CBID_DRIVER_API_cuGraphLaunch_ptsz;
      CUstream stream = nullptr;
      if (launch) {
        auto* p = static_cast<const LaunchKernelParams*>(cb->functionParams);
        stream = p->hStream;
        if (s.active) {
          const auto ps = params_of(p->f, p->kernelParams);
          s.pending_params = param_json(ps);
          prepare_tma(ps, stream);
        }
      } else if (launch_ex) {
        auto* p = static_cast<const LaunchKernelExParams*>(cb->functionParams);
        stream = p->config->hStream;
        if (s.active) {
          const auto ps = params_of(p->f, p->kernelParams);
          s.pending_params = param_json(ps);
          prepare_tma(ps, stream);
        }
      } else if (cbid == SANITIZER_CBID_DRIVER_API_cuGraphExecDestroy) {
        forget(*static_cast<const CUgraphExec*>(cb->functionParams));
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
          {
            Internal in;
            cuCtxSynchronize();
          }
          if (s.ready && check(sanitizerGetStreamHandle(cb->context, nullptr, &h), "stream handle")) dump(h);
        }
      } else if (!s.anchor.empty()) {
        Sanitizer_StreamHandle h = nullptr;
        if (s.dump_pending && !s.dumped && (launch || launch_ex) &&
            check(sanitizerGetStreamHandle(cb->context, stream, &h), "stream handle")) {
          check(sanitizerStreamSynchronize(h), "synchronize");
          dump(h);
        }
      } else if (graph_launch) {
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
      }
      if (graph_launch) unroll(cb);
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
  if (const char* v = getenv("LEDGER_ACCESS_UNROLL")) s.unroll = atoi(v) != 0;
  if (const char* v = getenv("LEDGER_ACCESS_TMA_DIFF")) s.tma_diff_max = strtoull(v, nullptr, 10);
  if (s.patches.empty()) {
    fprintf(stderr, "ledger-access: set LEDGER_ACCESS_PATCHES\n");
    return 0;
  }
  s.profiler_range = s.skip < 0;
  Sanitizer_SubscriberHandle handle;
  if (!check(sanitizerSubscribe(&handle, callback, nullptr), "subscribe")) return 0;
  sanitizerEnableCallback(1, handle, SANITIZER_CB_DOMAIN_RESOURCE, SANITIZER_CBID_RESOURCE_MODULE_LOADED);
  // Every driver call: the next call after a tensor-map launch is where its writes are found.
  sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_DRIVER_API);
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
