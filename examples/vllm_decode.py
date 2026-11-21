"""Run steady-state decode on vLLM under Ledger.

Starts an in-process engine with the given vLLM CLI arguments, warms up, exports the memory map,
then decodes STEPS engine steps with BATCH requests inside cudaProfilerStart/Stop so the capture
library only times the steady state.

    LD_PRELOAD=capture/libledger.so LEDGER_OUT=out \\
        python examples/vllm_decode.py --steps 40 --batch 1 -- --model <path> [vLLM args...]
"""

import argparse
import dataclasses
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # worker in this process
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc with a function

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

from ledger.vllm_regions import dump_regions  # noqa: E402

PROMPT = (
    "Write a detailed technical explanation of how a GPU executes a matrix multiplication, "
    "covering tiling, shared memory, tensor cores and memory bandwidth. "
)


def main():
    own = argparse.ArgumentParser()
    own.add_argument("--steps", type=int, default=40)
    own.add_argument("--batch", type=int, default=1)
    own.add_argument("--warmup", type=int, default=8)
    split = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    opts = own.parse_args(sys.argv[1:split])
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args(sys.argv[split + 1 :]))
    llm = LLM(**{f.name: getattr(args, f.name) for f in dataclasses.fields(args)})

    params = SamplingParams(max_tokens=8 * (opts.steps + opts.warmup) + 64, ignore_eos=True, temperature=0)
    llm.generate([PROMPT], SamplingParams(max_tokens=16, ignore_eos=True))  # compile / capture warm-up
    out_dir = os.environ.get("LEDGER_OUT", "ledger-out")
    os.makedirs(out_dir, exist_ok=True)
    print("regions:", llm.collective_rpc(dump_regions, args=(out_dir,)), flush=True)

    eng = llm.llm_engine
    for i in range(opts.batch):
        eng.add_request(f"ledger-{i}", PROMPT * (i + 1), params)
    tokens = {}

    def step():
        for out in eng.step():
            tokens[out.request_id] = list(out.outputs[0].token_ids)

    for _ in range(opts.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    for _ in range(opts.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    torch.cuda.profiler.stop()
    print(f"LEDGER_DECODE steps={opts.steps} batch={opts.batch} ms_per_step={elapsed / opts.steps * 1e3:.3f}")
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        f.write(f'{{"steps": {opts.steps}, "batch": {opts.batch}, "ms_per_step": {elapsed / opts.steps * 1e3:.3f}}}')
    while eng.has_unfinished_requests():
        step()
    # Greedy outputs, so two runs that should compute the same thing can be compared token for token.
    import json

    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump(tokens, f)


if __name__ == "__main__":
    main()
