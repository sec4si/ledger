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
  inside a graph cannot be told apart. Run the program with its graphs launched node by node (for a
  Sibyl replay, `SIBYL_REPLAY_EAGER_GRAPHS=1`) and count the window in launches of one kernel that runs
  once per step (`LEDGER_ACCESS_ANCHOR`).
- Tensor-map copies (TMA) are not instrumented. A launch that moves data only that way shows no
  accesses; the analysis then takes its pointers from a capture of the same program (`--capture`) and
  assumes it read and wrote everything they land in, so its blind spot can hide waste but not invent it.
- The window must close at least two anchor launches before the program ends; the log is written then.

`selftest.cu` has three kernels with known patterns. Under the tracer (`LEDGER_ACCESS_ANCHOR=_Z9fill_halfPf
LEDGER_GRAPH_WINDOW=0:3`) the analysis must report 256 KiB of dead writes per step, all from
`write_lows`, and a hand-off from `fill_half` to `read_all`.
