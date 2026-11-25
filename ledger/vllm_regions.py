"""Export the device memory map of a running vLLM worker: every parameter, buffer, KV/state cache
and draft-model tensor with its address range. Ledger uses it to tell what each kernel touched.

Call through collective_rpc from the driver process (in-process engine), e.g.
    llm.collective_rpc(dump_regions, args=(out_dir,))
"""

import json
import os
from pathlib import Path


def dump_regions(worker, out_dir):
    import torch

    regions, seen, attention = [], set(), []

    def add_attention(layer_name, layer, t):
        """Attention layers: where the layer's view of the cache starts and the bytes of K and V per
        cached token. Hybrid models carve every layer's cache out of one shared storage, so the
        storage cannot tell layers apart; the view's first byte can (attention kernels get it)."""
        heads, size = getattr(layer, "num_kv_heads", None), getattr(layer, "head_size", None)
        dtype = getattr(layer, "kv_cache_torch_dtype", None)
        if not (heads and size and isinstance(dtype, torch.dtype) and isinstance(t, torch.Tensor) and t.is_cuda):
            return
        span = (sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1) * t.element_size()
        attention.append({"kind": "attention", "name": layer_name, "addr": t.data_ptr(), "size": span,
                          "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                          "kv_bytes_per_token": 2 * heads * size * dtype.itemsize})

    def add(kind, name, t):
        if not (isinstance(t, torch.Tensor) and t.is_cuda):
            return
        storage = t.untyped_storage()
        if not storage.nbytes() or (storage.data_ptr(), kind) in seen:
            return
        seen.add((storage.data_ptr(), kind))
        regions.append(
            {
                "kind": kind,
                "name": name,
                "addr": storage.data_ptr(),
                "size": storage.nbytes(),
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
            }
        )

    runner = worker.model_runner
    model = getattr(runner, "model", None) or runner.get_model()
    for name, t in model.named_parameters():
        add("weight", name, t)
    # Linear layers: the input and output width of this rank's slice, so the analysis can count the
    # activations a GEMM reads and writes besides its weight.
    from vllm.model_executor.layers.linear import LinearBase

    widths = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, LinearBase):
            k, n = mod.input_size_per_partition, mod.output_size_per_partition
            for p_name, _ in mod.named_parameters(recurse=False):
                widths[f"{mod_name}.{p_name}" if mod_name else p_name] = (k, n)
    for r in regions:
        if r["name"] in widths:
            r["in_features"], r["out_features"] = widths[r["name"]]
    for name, t in model.named_buffers():
        add("buffer", name, t)

    # Modules hanging off the runner that are not the main model: drafters / speculative heads.
    for attr in ("drafter", "speculator", "spec_decode_model"):
        module = getattr(runner, attr, None)
        sub = getattr(module, "model", module)
        if isinstance(sub, torch.nn.Module):
            for name, t in sub.named_parameters():
                add("weight", f"{attr}.{name}", t)

    # Attention KV caches and recurrent (Mamba / GDN) state caches, one entry per layer.
    context = worker.vllm_config.compilation_config.static_forward_context
    for layer_name, layer in context.items():
        caches = getattr(layer, "kv_cache", None)
        items = caches if isinstance(caches, (list, tuple)) else [caches]
        for i, t in enumerate(items):
            tensors = t if isinstance(t, (list, tuple)) else [t]
            for j, tensor in enumerate(tensors):
                add("cache", f"{layer_name}[{i}][{j}]", tensor)
                add_attention(layer_name, layer, tensor)

    # Allocation segments of the caching allocator: where an unnamed pointer's tensor can lie.
    for seg in torch.cuda.memory_snapshot():
        regions.append({"kind": "segment", "name": "segment", "addr": seg["address"], "size": seg["total_size"],
                        "shape": [], "dtype": ""})

    path = Path(out_dir) / f"regions-{os.getpid()}.json"
    path.write_text(json.dumps(regions))
    (Path(out_dir) / f"attention-{os.getpid()}.json").write_text(json.dumps(attention))
    return len(regions)
