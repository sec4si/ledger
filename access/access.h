// Shared between the Ledger access tracer (host) and its device patches.
#pragma once
#include <cstdint>

namespace ledger_access {

// Event word: launch id (20 bits) | kind (1 bit, 0 read, 1 write) | granule (43 bits); all ones marks
// a slot lost to a race. Each event has a mask word: bit i set = sub-block i of the granule touched,
// a granule being split into 32 sub-blocks.
constexpr int kKindShift = 43;
constexpr int kLaunchShift = 44;
constexpr uint32_t kMaxLaunches = 1u << 16;  // launches recorded per window

struct Tables {
  unsigned long long* keys;        // granule + 1 per slot, 0 = empty
  unsigned long long* cur;         // per slot and kind: (launch id + 1) << 32 | its event index
  unsigned long long* log;         // first touch of a granule by a launch, per kind
  unsigned int* masks;             // sub-blocks touched, per event
  unsigned int* counters;          // [0] events, [1] dropped events, [2] table overflow, [3] claim retries,
                                   // [5] claims
  const unsigned long long* skip;  // sorted, disjoint [begin, end) pairs left untracked (weights)
  unsigned int n_skip;
  unsigned int log_cap;
  unsigned int mask;               // slots - 1
  unsigned int shift;              // log2 granule bytes
};

// The callback data of one launch: its id, or enabled == 0 outside the window.
struct Launch {
  Tables t;
  unsigned int id;
  unsigned int enabled;
};

}  // namespace ledger_access
