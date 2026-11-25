"""vLLM plugin: lets `vllm serve` (or any vLLM engine) run under Ledger's capture library.

With LEDGER_OUT set, every GPU worker writes its memory map (regions-<pid>.json, attention-<pid>.json)
and a run.json once warm-up and CUDA graph capture are done, and launches a marker kernel at the start
of each step so the analysis can cut the window into steps. Without LEDGER_OUT it does nothing.

    pip install -e <ledger>     # registers the plugin (entry point group vllm.general_plugins)
    LEDGER_OUT=out LEDGER_GRAPH_WINDOW=<skip>:<take> LD_PRELOAD=<ledger>/capture/libledger.so \\
        vllm serve <model> ...

The window is counted in graph launches (LEDGER_GRAPH_WINDOW); send load while it is open. run.json
has no fixed batch or step count here: the analysis counts steps from the markers, and the batch of a
serving step varies, so the compute section (which needs M) only fits steps of a known shape.
"""

import functools
import json
import os


def register():
    out = os.environ.get("LEDGER_OUT")
    if not out:
        return
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_ledger_patched", False):  # plugins can load more than once per process
        return
    Worker._ledger_patched = True
    warm_up, execute = Worker.compile_or_warm_up_model, Worker.execute_model

    @functools.wraps(warm_up)
    def compile_or_warm_up_model(self, *args, **kwargs):
        result = warm_up(self, *args, **kwargs)
        import torch

        from ledger.vllm_regions import dump_regions

        os.makedirs(out, exist_ok=True)
        dump_regions(self, out)
        spec = getattr(self.vllm_config, "speculative_config", None)
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        run = {"draft_tokens": getattr(spec, "num_speculative_tokens", 0) or 0,
               "sms": props.multi_processor_count, "device": props.name, "source": "vllm_plugin"}
        with open(os.path.join(out, "run.json"), "w") as f:
            json.dump(run, f)
        return result

    @functools.wraps(execute)
    def execute_model(self, *args, **kwargs):
        import torch

        torch.cuda._sleep(0)  # step marker (spin_kernel), as examples/vllm_decode.py launches
        return execute(self, *args, **kwargs)

    Worker.compile_or_warm_up_model = compile_or_warm_up_model
    Worker.execute_model = execute_model
