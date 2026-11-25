"""Exercise SGLang metadata and hooks with no SGLang, PyTorch, or GPU."""

import json
import math
import os
import sys
from types import ModuleType, SimpleNamespace

from ledger import sglang_plugin as plugin


class Tensor:
    is_cuda = True
    dtype = "torch.bfloat16"

    def __init__(self, address, shape, strides, storage=None):
        self.address = address
        self.shape = shape
        self.strides = strides
        self.storage = storage or SimpleNamespace(data_ptr=lambda: address, nbytes=lambda: self.numel() * 2)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 2

    def data_ptr(self):
        return self.address

    def stride(self):
        return self.strides

    def untyped_storage(self):
        return self.storage


def test_initializer_exports_named_weights_kv_views_and_run(tmp_path, monkeypatch):
    weight = Tensor(1000, (2, 4), (4, 1))
    host = Tensor(2000, (2, 4), (4, 1))
    host.is_cuda = False
    storage = SimpleNamespace(data_ptr=lambda: 3000, nbytes=lambda: 256)
    key = Tensor(3000, (3, 2, 4), (16, 4, 1), storage)
    value = Tensor(3128, (3, 2, 2), (8, 2, 1), storage)
    pool = SimpleNamespace(model_layer_ids=[7], k_buffer=[key], v_buffer=[value])
    model = SimpleNamespace(
        named_parameters=lambda: [("layers.7.proj.weight", weight), ("host", host)],
        named_buffers=lambda: [("shared.weight_view", weight)],
    )
    runner = SimpleNamespace(model=model, token_to_kv_pool=pool, is_draft_worker=False, device="cuda", gpu_id=0)
    props = SimpleNamespace(name="NVIDIA GB10", multi_processor_count=48)
    devices = []

    def get_properties(gpu_id):
        devices.append(gpu_id)
        return props

    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=get_properties))
    runtime = SimpleNamespace(get_spec=lambda: SimpleNamespace(speculative_num_draft_tokens=3))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    monkeypatch.setenv("LEDGER_OUT", str(tmp_path))
    scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
    plugin.on_initialized(None, scheduler)
    assert devices == [0]

    regions = json.loads((tmp_path / f"regions-{os.getpid()}.json").read_text())
    assert [(r["kind"], r["name"], r["addr"], r["size"]) for r in regions] == [
        ("weight", "layers.7.proj.weight", 1000, 16),
        ("cache", "layers.7.kv_cache.k", 3000, 256),
    ]
    attention = json.loads((tmp_path / f"attention-{os.getpid()}.json").read_text())
    assert [(r["name"], r["addr"], r["size"], r["kv_bytes_per_token"]) for r in attention] == [
        ("layers.7.kv_cache", 3000, 80, 24),
        ("layers.7.kv_cache", 3128, 40, 24),
    ]
    assert json.loads((tmp_path / "run.json").read_text()) == {
        "draft_tokens": 3, "sms": 48, "device": "NVIDIA GB10", "source": "sglang_plugin",
    }

    draft = Tensor(4000, (2, 4), (4, 1))
    runner.is_draft_worker = True
    runner.model = SimpleNamespace(named_parameters=lambda: [("proj.weight", draft)], named_buffers=lambda: [])
    plugin.dump_runner(runner, tmp_path, props, draft_tokens=1)
    merged = json.loads((tmp_path / f"regions-{os.getpid()}.json").read_text())
    assert merged[:-1] == regions
    assert merged[-1]["name"] == "draft.proj.weight"
    assert json.loads((tmp_path / "run.json").read_text())["draft_tokens"] == 3


def test_register_does_nothing_without_output(monkeypatch):
    monkeypatch.delenv("LEDGER_OUT", raising=False)
    monkeypatch.setitem(sys.modules, "sglang.srt.plugins.hook_registry", None)
    plugin.register()


def test_register_and_mark_only_target_cuda_steps(monkeypatch, tmp_path):
    hooks, markers = [], []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookRegistry = SimpleNamespace(register=lambda *args: hooks.append(args))
    registry.HookType = SimpleNamespace(AFTER="after", BEFORE="before")
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(_sleep=markers.append)))
    monkeypatch.setenv("LEDGER_OUT", str(tmp_path))

    plugin.register()
    assert hooks == [
        (f"{plugin.SCHEDULER}.__init__", plugin.on_initialized, "after"),
        (f"{plugin.RUNNER}.forward", plugin.mark_step, "before"),
    ]
    for device, draft in [("cpu", False), ("cuda", True), ("cuda:0", False)]:
        plugin.mark_step(SimpleNamespace(device=device, is_draft_worker=draft))
    assert markers == [0]
