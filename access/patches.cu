// Device patches of the Ledger access tracer: record, per launch, which granules of tracked memory
// it read and wrote. Each (launch, granule, kind) is logged once; weights are skipped.
#include <sanitizer_patching.h>

#include "access.h"

using namespace ledger_access;

namespace {

__device__ bool skipped(const Tables& t, unsigned long long a) {
  unsigned int lo = 0, hi = t.n_skip;  // first pair whose begin > a
  while (lo < hi) {
    const unsigned int mid = (lo + hi) / 2;
    if (t.skip[2 * mid] <= a) lo = mid + 1; else hi = mid;
  }
  return lo > 0 && a < t.skip[2 * (lo - 1) + 1];
}

// Claim the (granule, kind) slot for this launch and return the index of its event, or ~0u when the
// event cannot be logged. Called by one lane per group of lanes touching the same granule.
__device__ unsigned int claim(const Launch& l, unsigned long long g, unsigned int k, unsigned long long pc) {
  const Tables& t = l.t;
  unsigned int h = static_cast<unsigned int>((g * 0x9E3779B97F4A7C15ull) >> 40) & t.mask;
  const unsigned long long key = g + 1;
  for (int probe = 0;; ++probe) {
    const unsigned long long old = atomicCAS(&t.keys[h], 0ull, key);
    if (old == 0 || old == key) break;
    if (probe == 128) {
      atomicAdd(&t.counters[2], 1u);
      return ~0u;
    }
    h = (h + 1) & t.mask;
  }
  // cur holds (launch id + 1) << 32 | event index; kPending while the owner is still logging, kNone
  // when it could not log. Claiming first and logging second spends one log entry per claim.
  constexpr unsigned int kPending = 0xffffffffu, kNone = 0xfffffffeu;
  unsigned long long* cur = &t.cur[2 * h + k];
  const unsigned long long me = static_cast<unsigned long long>(l.id + 1) << 32;
  unsigned long long old = atomicAdd(cur, 0ull);  // a coherent read
  for (;;) {
    if ((old >> 32) == (me >> 32)) {
      const unsigned int idx = static_cast<unsigned int>(old);
      if (idx == kPending) {  // another warp of this launch is logging this slot
        old = atomicAdd(cur, 0ull);
        continue;
      }
      return idx == kNone ? ~0u : idx;
    }
    if ((old >> 32) > (me >> 32)) return ~0u;  // a later launch already owns the slot
    const unsigned long long prev = atomicCAS(cur, old, me | kPending);
    if (prev != old) {
      atomicAdd(&t.counters[3], 1u);
      old = prev;
      continue;
    }
    const unsigned int n = atomicAdd(&t.counters[0], 1u);
    if (n >= t.log_cap) {
      atomicAdd(&t.counters[1], 1u);
      atomicExch(cur, me | kNone);
      return ~0u;
    }
    t.log[n] = (static_cast<unsigned long long>(l.id) << kLaunchShift) |
               (static_cast<unsigned long long>(k) << kKindShift) | g;
    t.pcs[n] = pc;
    atomicAdd(&t.counters[5], 1u);
    atomicExch(cur, me | n);
    return n;
  }
}

// kinds: bit 0 read, bit 1 write.
__device__ void touch(void* userdata, unsigned long long pc, unsigned long long addr, unsigned long long bytes,
                      unsigned int kinds) {
  if (!userdata || !bytes) return;
  const Launch& l = *static_cast<const Launch*>(userdata);
  if (!l.enabled || skipped(l.t, addr)) return;
  const unsigned long long end = addr + bytes;
  const unsigned int sub = l.t.shift - 5;  // log2 of the sub-block size
  const unsigned int lane = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) % 32;
  for (unsigned long long g = addr >> l.t.shift; g <= (end - 1) >> l.t.shift; ++g) {
    const unsigned long long base = g << l.t.shift;
    const unsigned long long lo = addr > base ? addr : base;
    const unsigned long long hi = end < base + (1ull << l.t.shift) ? end : base + (1ull << l.t.shift);
    const unsigned int first = static_cast<unsigned int>((lo - base) >> sub);
    const unsigned int last = static_cast<unsigned int>((hi - 1 - base) >> sub);
    const unsigned int bits = (last == 31 ? 0xffffffffu : ((1u << (last + 1)) - 1)) & ~((1u << first) - 1);
    for (unsigned int k = 0; k < 2; ++k) {
      if (!(kinds & (1u << k))) continue;
      // The lanes of a warp usually hit the same granule together: one of them claims for the group,
      // then every lane adds its own sub-blocks.
      const unsigned int group = __match_any_sync(__activemask(), (g << 1) | k);
      const int leader = __ffs(group) - 1;
      unsigned int n = 0;
      if (static_cast<int>(lane) == leader) n = claim(l, g, k, pc);
      n = __shfl_sync(group, n, leader);
      if (n != ~0u) atomicOr(&l.t.masks[n], bits);
    }
  }
}

}  // namespace

extern "C" __device__ __noinline__ SanitizerPatchResult LedgerGlobal(void* u, uint64_t pc, void* ptr, uint32_t size,
                                                                     uint32_t flags, const void*) {
  unsigned int kinds = 0;
  if (flags & SANITIZER_MEMORY_DEVICE_FLAG_READ) kinds |= 1;
  if (flags & SANITIZER_MEMORY_DEVICE_FLAG_WRITE) kinds |= 2;
  if (flags & SANITIZER_MEMORY_DEVICE_FLAG_PREFETCH) kinds = 0;
  if (kinds) touch(u, pc, reinterpret_cast<uintptr_t>(ptr), size, kinds);
  return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__ SanitizerPatchResult LedgerMemcpyAsync(void* u, uint64_t pc, void* src, uint32_t,
                                                                          uint32_t size) {
  if (src) touch(u, pc, reinterpret_cast<uintptr_t>(src), size, 1);
  return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__ SanitizerPatchResult LedgerBulkGlobalToShared(void* u, uint64_t pc, uint64_t src,
                                                                                 uint32_t, uint32_t, uint32_t data,
                                                                                 uint32_t) {
  touch(u, pc, src, 16ull * (data & 0xffff), 1);
  return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__ SanitizerPatchResult LedgerBulkSharedToGlobal(void* u, uint64_t pc, uint64_t dst,
                                                                                 uint32_t, uint32_t blocks) {
  touch(u, pc, dst, 16ull * blocks, 2);
  return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__ SanitizerPatchResult LedgerBulkReduceSharedToGlobal(void* u, uint64_t pc, uint64_t dst,
                                                                                       uint32_t, uint32_t blocks) {
  touch(u, pc, dst, 16ull * blocks, 3);
  return SANITIZER_PATCH_SUCCESS;
}
