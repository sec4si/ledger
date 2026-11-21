"""Swap vLLM's GDN MTP decode op for the extension build (KEI_GDN_MODE: 0 per-token stores, 1 last
position only, 2 deferred), then run Ledger's vllm_decode.py.
Usage: python gdn_patch_run.py <vllm_decode.py> [args...]"""

import atexit
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("KEI_GDN_TAG", "fastmath")
os.environ.setdefault("KEI_GDN_FLAGS", "--use_fast_math")

import torch  # noqa: E402

from kei_gdn_build import build  # noqa: E402

kei = build()
mode = int(os.environ.get("KEI_GDN_MODE", "0"))
kei.set_store_mode(mode)

import vllm._custom_ops as ops  # noqa: E402

calls = [0]


def patched(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state,
            output_gate, norm_weight, out=None, scale=128**-0.5, norm_eps=1e-5,
            output_gate_activation="silu"):
    if out is None:
        out = torch.empty_like(output_gate)
    calls[0] += 1
    kei.fused_gdn_decode_post_conv_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
                                       num_accepted_tokens, state, output_gate, norm_weight, out,
                                       scale, norm_eps, output_gate_activation)
    return out


ops.fused_gdn_decode_post_conv_mtp = patched
atexit.register(lambda: print(f"KEI_GDN mode={mode} python_calls={calls[0]}", flush=True))

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
