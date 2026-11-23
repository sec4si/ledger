"""Run steady-state decode on vLLM under Ledger.

Starts an in-process engine with the given vLLM CLI arguments, warms up, exports the memory map,
then decodes STEPS engine steps with BATCH requests inside cudaProfilerStart/Stop so the capture
library only times the steady state.

    LD_PRELOAD=capture/libledger.so LEDGER_OUT=out \\
        python examples/vllm_decode.py --steps 40 --batch 1 -- --model <path> [vLLM args...]

Each step in the window starts with a marker kernel (`torch.cuda._sleep(0)`, a `spin_kernel`), so
the analysis can cut the window into steps. Next to the capture it writes

- run.json: steps, batch, draft tokens, SM count and device;
- steps.jsonl: per step, the host time of `engine.step()`, the tokens each request had before and
  gained, and the SM clock, power and temperature sampled during the step (NVML, if available);
- tokens.json: the generated tokens, to compare two runs.

--prompt-tokens N gives every request a prompt of N tokens (default: a short sentence), so attention
reads a long KV cache. --prefill N instead times one step that prefills a single N-token prompt.
"""

import argparse
import dataclasses
import json
import os
import sys
import threading
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


class Sampler:
    """SM clock, power and temperature every `period` seconds, on a thread (NVML releases the GIL)."""

    def __init__(self, period):
        self.samples, self.stop = [], threading.Event()
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml, self.dev = pynvml, pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception:
            self.nvml = None
        if self.nvml and period > 0:
            self.thread = threading.Thread(target=self.run, args=(period,), daemon=True)
            self.thread.start()

    def run(self, period):
        n, d = self.nvml, self.dev
        while not self.stop.wait(period):
            self.samples.append((time.perf_counter(), n.nvmlDeviceGetClockInfo(d, n.NVML_CLOCK_SM),
                                 n.nvmlDeviceGetPowerUsage(d) / 1e3, n.nvmlDeviceGetTemperature(d, 0)))

    def between(self, t0, t1):
        xs = [s for s in self.samples if t0 <= s[0] <= t1]
        if not xs:
            return {}
        mean = lambda i: round(sum(s[i] for s in xs) / len(xs), 2)  # noqa: E731
        return {"sm_mhz": mean(1), "power_w": mean(2), "temp_c": mean(3), "samples": len(xs)}


def token_prompt(llm, n):
    ids = llm.get_tokenizer().encode(PROMPT, add_special_tokens=False)
    return {"prompt_token_ids": (ids * (n // len(ids) + 1))[:n]}


def main():
    own = argparse.ArgumentParser()
    own.add_argument("--steps", type=int, default=40)
    own.add_argument("--batch", type=int, default=1)
    own.add_argument("--warmup", type=int, default=8)
    own.add_argument("--prompt-tokens", type=int, default=0, help="prompt length of every request")
    own.add_argument("--prefill", type=int, default=0, help="time one step prefilling an N-token prompt")
    own.add_argument("--nvml-ms", type=float, default=10, help="NVML sampling period; 0 turns it off")
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
    prompt = lambda i: token_prompt(llm, opts.prompt_tokens) if opts.prompt_tokens else PROMPT * (i + 1)  # noqa: E731
    if opts.prefill:
        opts.steps, opts.batch = 1, 1
    else:
        for i in range(opts.batch):
            eng.add_request(f"ledger-{i}", prompt(i), params)
    tokens, context = {}, {}

    def step():
        for out in eng.step():
            tokens[out.request_id] = list(out.outputs[0].token_ids)
            context[out.request_id] = len(out.prompt_token_ids or []) + len(tokens[out.request_id])

    for _ in range(opts.warmup):
        step()
    if opts.prefill:
        eng.add_request("ledger-prefill", token_prompt(llm, opts.prefill), SamplingParams(max_tokens=1, temperature=0))
    sampler = Sampler(opts.nvml_ms / 1e3)
    records = []
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    for i in range(opts.steps):
        before = dict(context)
        h0 = time.perf_counter()
        torch.cuda._sleep(0)  # step marker
        step()
        h1 = time.perf_counter()
        records.append({"step": i, "host_ms": round((h1 - h0) * 1e3, 3), "context": before,
                        "gained": {r: context[r] - before.get(r, 0) for r in context}, "h0": h0, "h1": h1})
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    torch.cuda.profiler.stop()
    sampler.stop.set()
    ms = elapsed / opts.steps * 1e3
    print(f"LEDGER_DECODE steps={opts.steps} batch={opts.batch} ms_per_step={ms:.3f}")

    spec = getattr(llm.llm_engine.vllm_config, "speculative_config", None)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    run = {"steps": opts.steps, "batch": opts.batch, "ms_per_step": round(ms, 3),
           "draft_tokens": getattr(spec, "num_speculative_tokens", 0) or 0, "prefill": opts.prefill,
           "prompt_tokens": opts.prompt_tokens, "sms": props.multi_processor_count, "device": props.name}
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f)
    with open(os.path.join(out_dir, "steps.jsonl"), "w") as f:
        for r in records:
            r.update(sampler.between(r.pop("h0"), r.pop("h1")))
            f.write(json.dumps(r) + "\n")
    while eng.has_unfinished_requests():
        step()
    # Greedy outputs, so two runs that should compute the same thing can be compared token for token.
    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump(tokens, f)


if __name__ == "__main__":
    main()
