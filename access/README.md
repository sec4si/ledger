# Access tracer

Records, for every kernel launch inside a window, which device memory it read and which it wrote: 1 KiB
granules with a 32-byte sub-block mask, plus the byte ranges of memcpy and memset calls. Built on the
Compute Sanitizer API; `python -m ledger.dataflow` turns the trace into dead writes, dead launches,
hand-offs between neighbours, constant inputs and independent neighbours.

```sh
make -C access CUDA=/usr/local/cuda ARCH=121
LD_PRELOAD=$PWD/access/libledger_access.so LEDGER_ACCESS_PATCHES=$PWD/access/patches.fatbin \
  LEDGER_OUT=out LEDGER_GRAPH_WINDOW=<skip>:<take> [LEDGER_ACCESS_ANCHOR=<kernel>] \
  LEDGER_ACCESS_SKIP=weights.ranges <program>
python -m ledger.dataflow out --anchor <kernel> [--capture <capture dir>] [--layout mem_layout.json]
```

- `LEDGER_ACCESS_SKIP` lists byte ranges left untracked, one `<begin> <end>` per line; pass the weights.
- The Sanitizer API gives every node of a CUDA graph the callback data of the last one, so accesses
  inside a graph cannot be told apart. The tracer keeps a copy of every graph the program instantiates
  and, when the program launches it, launches its nodes one by one on the same stream (in an order its
  edges allow), then an empty graph in its place. Programs run unchanged, CUDA graphs and all; count the
  window in launches of one kernel that runs once per step (`LEDGER_ACCESS_ANCHOR`). A graph holding
  nodes that cannot run on their own (memory allocation, conditional) is left whole, with a warning.
  `LEDGER_ACCESS_UNROLL=0` leaves every graph whole.
- Tensor-map copies (TMA) are not instrumented. The tracer finds tensor maps among a launch's
  parameters (the driver's 128-byte encoding, decoded in `decode_map`), records the tensors they span
  as read (`tma_reads`, weights left out) and finds what the launch wrote by comparing those tensors
  before and after it (`tma_writes`, 32-byte blocks). The comparison misses a write of the value already
  there, so it is a lower bound. A launch whose tensors exceed `LEDGER_ACCESS_TMA_DIFF` bytes, or one
  with a 128-byte parameter that is not a tensor map the tracer can read (maps kept in device memory),
  stays opaque: the analysis takes its pointers and assumes it read and wrote everything they land in,
  so its blind spot can hide waste but not invent it.
- Without `LEDGER_GRAPH_WINDOW` the window is the profiling range (`cudaProfilerStart`/`Stop`), as with
  the capture library; the log is written when the range closes. With a launch-count window it must
  close at least two anchor launches before the program ends.

`make selftest selftest_tma` builds the self-tests. `selftest.cu` has three kernels with known patterns. Under the tracer (`LEDGER_ACCESS_ANCHOR=_Z9fill_halfPf` and
either `LEDGER_GRAPH_WINDOW=0:3` or `SELFTEST_PROFILER=1`) the analysis must report 256 KiB of dead writes per step, all from
`write_lows`, and a hand-off from `fill_half` to `read_all`.
Each must give the same analysis with `SELFTEST_GRAPH=1`, which runs each step as one graph launch.
`selftest_tma.cu` copies the top half of one buffer into another through tensor maps only, then clears
the second: with `LEDGER_ACCESS_ANCHOR=_Z4fillPff`, `tma_copy` must carry `tma_writes` of exactly the
first 512 KiB of the second buffer and the analysis must report all 512 KiB per step dead.
