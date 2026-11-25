"""Export SGLang worker memory maps and mark target-model forward steps."""

from collections import defaultdict
import json
import math
import os
from pathlib import Path

RUNNER = "sglang.srt.model_executor.model_runner.ModelRunner"
SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"


def tensor_region(kind, name, tensor, view=False):
    if not getattr(tensor, "is_cuda", False) or not tensor.numel():
        return None
    shape = list(tensor.shape)
    if view:
        span = (sum((n - 1) * stride for n, stride in zip(shape, tensor.stride())) + 1) * tensor.element_size()
        addr, size = tensor.data_ptr(), span
    else:
        storage = tensor.untyped_storage()
        addr, size = storage.data_ptr(), storage.nbytes()
    return {"kind": kind, "name": name, "addr": addr, "size": size, "shape": shape,
            "dtype": str(tensor.dtype).removeprefix("torch.")}


def cache_layers(pool):
    """Yield persistent per-layer K/V views, including pools nested in a hybrid pool."""
    children = [child for name in ("full_kv_pool", "swa_kv_pool")
                if (child := getattr(pool, name, None)) is not None]
    if children:
        for child in children:
            yield from cache_layers(child)
        return
    layers = defaultdict(list)
    layer_ids = getattr(pool, "model_layer_ids", None)
    start = getattr(pool, "start_layer", 0)
    for name in ("k", "v", "kv"):
        for i, tensor in enumerate(getattr(pool, f"{name}_buffer", None) or ()):
            layer = layer_ids[i] if layer_ids and i < len(layer_ids) else start + i
            if getattr(tensor, "is_cuda", False) and tensor.numel():
                layers[layer].append((name, tensor))
    yield from layers.items()


def write_regions(path, records):
    """Keep the target and draft models in one PID map without duplicate addresses."""
    previous = json.loads(path.read_text()) if path.exists() else []
    merged = {}
    for record in (*previous, *records):
        merged.setdefault(record["addr"], record)
    path.write_text(json.dumps(list(merged.values())))


def dump_runner(runner, out, properties, draft_tokens):
    """Write Ledger metadata for a model runner after its weights and KV pool are ready."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "drafter." if runner.is_draft_worker else ""  # analyze.DRAFT tells draft launches by it
    regions, attention = [], []
    for kind, tensors in (("weight", runner.model.named_parameters()), ("buffer", runner.model.named_buffers())):
        for name, tensor in tensors:
            if region := tensor_region(kind, prefix + name, tensor):
                regions.append(region)

    for layer, buffers in cache_layers(getattr(runner, "token_to_kv_pool", None)):
        name = f"{prefix}layers.{layer}.kv_cache"
        token_major = all(len(tensor.shape) == 3 for _, tensor in buffers)
        token_bytes = sum(math.prod(tensor.shape[1:]) * tensor.element_size() for _, tensor in buffers)
        for suffix, tensor in buffers:
            regions.append(tensor_region("cache", f"{name}.{suffix}", tensor))
            if token_major:
                region = tensor_region("attention", name, tensor, view=True)
                attention.append({**region, "kv_bytes_per_token": token_bytes})

    pid = os.getpid()
    write_regions(out / f"regions-{pid}.json", regions)
    write_regions(out / f"attention-{pid}.json", attention)
    if not runner.is_draft_worker:
        run = {"draft_tokens": draft_tokens, "sms": properties.multi_processor_count,
               "device": properties.name, "source": "sglang_plugin"}
        (out / "run.json").write_text(json.dumps(run))


def on_initialized(_result, scheduler, *args, **kwargs):
    # SGLang can allocate the runner's KV pool and resize it after graph capture.
    runner = scheduler.tp_worker.model_runner
    if str(runner.device).split(":", 1)[0] != "cuda":
        return
    import torch
    from sglang.srt.runtime_context import get_spec

    properties = torch.cuda.get_device_properties(runner.gpu_id)
    out = os.environ["LEDGER_OUT"]
    draft_tokens = get_spec().speculative_num_draft_tokens or 0
    dump_runner(runner, out, properties, draft_tokens)
    draft_runners = getattr(getattr(scheduler, "draft_worker", None), "_draft_model_runners", None)
    if draft_runners is not None:
        for draft in draft_runners():
            dump_runner(draft, out, properties, draft_tokens)


def mark_step(runner, *args, **kwargs):
    if runner.is_draft_worker or str(runner.device).split(":", 1)[0] != "cuda":
        return
    import torch

    torch.cuda._sleep(0)


def register():
    """Register worker-local hooks through SGLang's general plugin entry point."""
    if not os.environ.get("LEDGER_OUT"):
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(f"{SCHEDULER}.__init__", on_initialized, HookType.AFTER)
    HookRegistry.register(f"{RUNNER}.forward", mark_step, HookType.BEFORE)
