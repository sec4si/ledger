# Quickstart

The shortest path from a clean checkout to a per-launch report, a Perfetto timeline, and an A/B
diff on vLLM.

## 1. Install

Run inside the Python environment where `vllm` and CUDA-enabled `torch` are installed (with `nvcc`
on `PATH` or `CUDA_HOME` set):

```sh
pip install -e .
```

If you also plan to use the access-trace tools (`dataflow`, `dead-across`, `traffic`,
`perturb-plan`, `perturb-report`), install the optional `numpy` extra:

```sh
pip install -e '.[dataflow]'
```

## 2. Build the capture library

```sh
ledger build
```

This compiles `capture/libledger.so` against your CUDA toolkit and caches it under
`~/.cache/ledger/build/` (printing the full path). Pass `--access` (and optionally `--arch <sm>`)
only if you also need the Compute Sanitizer access tracer.

## 3. Measure the GPU ceilings once

Make sure **no other process is using the GPU** — `ledger ceilings` checks `nvidia-smi` and refuses
to measure if another CUDA context is active, so noisy numbers do not get cached.

```sh
ledger ceilings
```

This runs `bench/bwtest.cu` and `bench/flops.py`, prints the measured DRAM read bandwidth and GEMM
peaks, and caches them in `~/.cache/ledger/ceilings-<device>.json`. Subsequent `ledger analyze`,
`ledger steps`, `ledger diff`, `ledger report`, and `ledger perfetto` calls load this cache
automatically. Use `ledger ceilings --refresh` to re-measure.

## 4. Profile a vLLM run

```sh
ledger profile --steps 20 --out run1 -- --model <model> [vLLM args...]
```

`--out` must be a new or empty directory. Optional flags before `--`: `--steps` (default 40),
`--batch` (default 1), `--prompt-tokens` (default 0), and `--prefill` (default 0). Everything after
`--` is passed straight to vLLM.

When the run finishes, `run1/` contains the raw CUPTI capture, `run1/vllm.log`, the combined
`analyze` + `steps` report in `run1/report.md`, and the Perfetto timeline in `run1/trace.json`
(printing the URL to open it).

## 5. Read the report

Open `run1/report.md` and start with **`## Findings`** at the top: it ranks up to the five largest
costs in ms per step, each as an upper bound on what fixing it could save. Follow the section name
in parentheses for the full table and evidence. See [Reading Ledger Reports](reading-reports.md)
for how to interpret every section and confirm a finding.

## 6. Inspect the timeline in Perfetto

`ledger profile` already wrote `run1/trace.json`. Open [ui.perfetto.dev](https://ui.perfetto.dev)
and load `run1/trace.json` to inspect the `Steps`, per-CUDA `Stream`, and `Memcpy` tracks.

To export a timeline from an existing capture later (for example, one captured with `vllm serve` or
another engine):

```sh
ledger perfetto run1 --out run1.json
```

`ledger perfetto` fills in `--ceiling-gbps` from the GPU cache automatically when available (pass
`--ceiling-gbps <measured>` explicitly on a machine without a cache to include per-launch `gbps`,
`pct_ceiling`, and partial-read filtering).

## 7. Compare before and after

After changing a kernel, backend, or flag, capture a second run and diff the two:

```sh
ledger profile --steps 20 --out run2 -- --model <model> [updated vLLM args...]
ledger diff run1 run2
```

`ledger diff` uses the cached ceilings, estimates run-to-run clock/power-cap drift from shared
launch groups, and lists only the groups that moved beyond that drift.

## 8. Re-generate a report from an existing capture

After upgrading or modifying Ledger, you do not need to re-run the GPU to refresh the report:

```sh
ledger report run1 [--out path/to/report.md]
```

`ledger report` re-runs `analyze` and `steps` over the saved capture using the cached ceilings
(or explicit `--ceiling-gbps` / `--peak-tflops`) and writes `<capture>/report.md` by default.

---

## Profiling `vllm serve` with the plugin


`pip install -e .` registers `ledger.vllm_plugin:register` under the `vllm.general_plugins` entry
point. When `LEDGER_OUT` is unset it does nothing. To capture a window from a live `vllm serve`
instance:

```sh
LEDGER_OUT=out LEDGER_GRAPH_WINDOW=<skip>:<take> LD_PRELOAD=$(ledger build) \
  vllm serve <model> [vLLM args...]
```

Send requests while the graph-launch window (`<skip>` skipped, `<take>` captured) is open, and keep
the server running until the window closes. Kernels still running when it closes come back without
timestamps; the analysis drops them and says how many. Each GPU worker exports
`regions-<pid>.json`, `attention-<pid>.json`, and `run.json` after warm-up and emits a step marker
at every `execute_model` call.

Then write the report with `ledger report out` (or `ledger analyze out`): it counts steps from the
markers automatically. On a GB10 with vLLM main, Qwen3.8-27B NVFP4 with MTP,
`LEDGER_GRAPH_WINDOW=300:200` and `vllm bench serve` sending 32 requests at concurrency 4 gave a
window of 49 steps.
Because serving batch size varies from step to step, `run.json` has no fixed `batch` or `steps`, so
the `Compute` section (which needs `M`) applies only when steps have a known shape.

---

## Profiling SGLang serve

Install Ledger in the same environment as SGLang. The plugin uses the SGLang Runtime
general plugin entry point (`sglang.srt.plugins`, verified with SGLang 0.5.23) and only
activates when `LEDGER_OUT` is set. For a single-GPU server with CUDA graphs enabled:

```sh
ledger build
capture_lib=$(ledger build)
LEDGER_OUT=sglang-out LEDGER_GRAPH_WINDOW=16:32 LD_PRELOAD="$capture_lib" \
  sglang serve --model-path <model>
```

Use a fresh output directory. Send requests with enough generated tokens to reach the end
of the graph-launch window while the server is still running. Adjust the skip and length
for your workload. Each CUDA worker writes a parameter and KV pool map after initialization.
Token-major KV views also include per-layer addresses and bytes per token. The target
model emits one step marker per forward, including when it replays a CUDA graph.

Run `ledger report sglang-out` and `ledger perfetto sglang-out --out sglang-out/trace.json` to
inspect the capture. The report uses the measured bandwidth cache or `--ceiling-gbps`.

---

## Other CUDA programs and Sibyl replays

Any CUDA program works with the capture library directly:

1. Preload `libledger.so` (`LD_PRELOAD=$(ledger build) LEDGER_OUT=out`).
2. Bracket the window with `cudaProfilerStart` / `cudaProfilerStop`, or set
   `LEDGER_GRAPH_WINDOW=<skip>:<take>` if the program never calls the profiler API. For programs
   that `fork()` one process per GPU, set `LEDGER_INIT_IN_CHILD=1` so each child starts CUPTI
   itself (CUPTI worker threads do not survive `fork()`).
3. Export the memory map as `out/regions-<pid>.json` (a list of
   `{kind, name, addr, size, shape, dtype}`; see `ledger/vllm_regions.py`).

For a [Sibyl](https://github.com/GeetoRinku/sibyl) replay, which preserves recorded addresses:

```sh
ledger sibyl-regions <recording>/ranges-rank0-<pid>.json out
LD_PRELOAD=$(ledger build) LEDGER_OUT=out LEDGER_GRAPH_WINDOW=8:44 <replay program> ...
```

---

## Huawei Ascend CANN (`msprof` + `aclGraph`)

To analyze an Ascend NPU run exported with `msprof --export=on`, import its `op_summary_*.csv`
(and optional `task_time_*.csv`), `aclmdlRIDebugPrint` graph dumps (`--graph`), and weight / KV
descriptor maps (`--weights`, `--draft-weights`, `--owners`, or `--regions`):

```sh
ledger ascend-import <msprof PROF_* dir> out \
  --graph capture.json --graph draft-graph.json \
  --weights target_weights.json --draft-weights draft_weights.json --owners owners.json \
  --device "Ascend 910B1" --sms 24 --steps 1 --batch 8 --draft-tokens 7

ledger analyze out --ceiling-gbps 1350 --peak-tflops bf16=320,int8=640 --out out/report.md
ledger transfers out --steps 1 --out out/transfers.md
ledger perfetto out --ceiling-gbps 1350 --out out/trace.json
```

msprof does not record the size or direction of `MEMCPY_ASYNC` tasks: `ledger transfers` reports
their times, but the copies are imported with 0 bytes and labelled DtoD.
