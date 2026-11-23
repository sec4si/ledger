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

## Overhead

GB10 at a locked 2,000 MHz, vLLM 0.34 with CUDA graphs and `examples/vllm_decode.py`, two rounds per
setting (both shown). Time per decode step inside the window, the tracer with `LEDGER_ACCESS_SKIP=auto`;
Qwen3.8 with MTP, three draft tokens:

| model | batch | none | capture library | access tracer | tracer, graphs left whole | tracer, no tensor-map comparison |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B FP8 | 1 | 23.2 / 23.4 ms | 24.9 / 25.1 ms (1.07x) | 1904 / 1772 ms (79x) | 470 / 470 ms (20x) | 491 / 489 ms (21x) |
| Qwen3-30B-A3B FP8 | 4 | 48.7 / 49.3 ms | 50.1 / 50.0 ms (1.02x) | 1339 / 1575 ms (30x) | 336 / 335 ms (6.8x) | 345 / 343 ms (7.0x) |
| Qwen3.8-27B NVFP4 | 1 | 121 / 125 ms | 121 / 121 ms (1.0x) | 1606 / 1566 ms (13x) | 1498 / 1500 ms (12x) | 1549 / 1548 ms (13x) |
| Qwen3.8-27B NVFP4 | 4 | 134 / 134 ms | 137 / 134 ms (1.0x) | 953 / 926 ms (7.0x) | 859 / 857 ms (6.4x) | 878 / 874 ms (6.5x) |

- The capture library costs at most 7%, and nothing measurable on Qwen3.8.
- The tracer slows a step down 7 to 80 times. Where most of it goes depends on the model. On
  Qwen3-30B it is the tensor-map comparison: DeepGEMM's maps are encoded for 4,096 rows, so each
  launch copies the rest of its allocator segment twice (`LEDGER_ACCESS_TMA_DIFF=0` takes batch 1
  from 79x to 21x). On Qwen3.8 neither the comparison nor the graph unrolling adds more than 10%;
  the instrumented accesses themselves are the cost.
- "Graphs left whole" (`LEDGER_ACCESS_UNROLL=0`) is not a fair measure of what unrolling costs: the
  accesses inside graphs then go to the wrong launch or are lost (on Qwen3-30B the trace keeps 4% of
  its events).
- Qwen3.8 at batch 1 is slower under the tracer than at batch 4; we do not know why.
- At batch 4, Qwen3.8 writes 20 million events in six steps, more than the default log holds
  (`LEDGER_ACCESS_EVENTS`, 8,388,608); a full log drops the rest and runs faster. The batch-4
  rows above were run with 33,554,432; check `dropped_events` in `access-meta-<pid>.json`.

Trace size per step: 6 MB (Qwen3-30B, batch 1) to 70 MB (Qwen3.8, batch 4), against 0.6 to 1.5 MB for
a capture.

