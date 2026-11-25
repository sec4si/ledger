# Changelog

## 0.1.0

Initial release of Ledger: per-launch accounting for LLM inference on GPUs.

### Capture and profiling
- CUPTI `LD_PRELOAD` capture library (`capture/libledger.so`) recording per-launch parameters,
  CUDA graph kernel nodes, kernel timings, and linear memory copies without modifying the engine.
- `ledger build` to compile and cache the capture library (and `--access` for the Compute Sanitizer
  access tracer) against the local CUDA toolkit.
- `ledger ceilings` to measure and cache GPU DRAM read bandwidth (`bench/bwtest.cu`) and GEMM
  TFLOP/s (`bench/flops.py`), refusing to measure while other processes use the GPU.
- `ledger profile` to capture a vLLM run and write both `<out>/report.md` and `<out>/trace.json`.
- vLLM `vllm.general_plugins` entry point (`ledger.vllm_plugin`) for profiling live `vllm serve`
  instances.

### Analysis and visualization
- `ledger analyze`: top `## Findings` summary, step bandwidth floor, per-weight efficiency,
  same-kernel/same-shape splits with preceding cache-touching kernel attribution, repeated weight
  reads, underfilled SM launches, and compute-vs-memory floors (`--peak-tflops`).
- `ledger steps`: step-to-step cycle breakdown (GPU busy, intra-step idle, host idle), speculative
  decoding yield, KV-cache attention bandwidth, and SM clock/temperature tracking.
- `ledger diff`: launch-group comparison past run-to-run clock/power-cap drift, template-variant
  pairing, and generated token equality checks.
- `ledger report`: re-generate the combined `analyze` + `steps` report from an existing capture.
- `ledger perfetto`: Chrome Trace Event JSON export (`Steps`, per-CUDA `Stream` with automatic
  concurrency lanes, and `Memcpy` tracks) for Perfetto UI.
- `ledger transfers`: H2D/D2H/D2D memory copy accounting by region, direction, exposed time, and
  pageable host memory.

### Byte-level access tracing and dataflow
- Compute Sanitizer access tracer (`access/libledger_access.so`) with per-node CUDA graph
  execution and TMA tensor-map decoding.
- `ledger dataflow`, `ledger dead-across`, `ledger traffic`, `ledger sass`, `ledger perturb-plan`,
  `ledger perturb-report`, and `ledger sibyl-regions`.
