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

    regions, seen = [], set()

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

    path = Path(out_dir) / f"regions-{os.getpid()}.json"
    path.write_text(json.dumps(regions))
    return len(regions)
