"""Two experiments behind the GDN out_proj finding, run in the vLLM environment.

1. The GDN out_proj and the attention o_proj weights (same shape, FP8) on the same
   cutlass_scaled_mm at M=4, L2 flushed with a read before each launch. If the weights themselves
   were the cause, their timings would differ here.
2. The same GEMM after writing N MB right before it, which leaves N MB of dirty lines in L2 that
   must be written back while the GEMM streams its weight.

    python gemm_isolation.py /path/to/Qwen3.8-27B-NVFP4/model.safetensors
"""

import sys

import torch
from safetensors import safe_open
from vllm import _custom_ops as ops

M = 4  # tokens per decode step with 3 draft tokens
PREFIX = "model.language_model.layers.{}.{}"


def load(f, layer, proj):
    w = f.get_tensor(PREFIX.format(layer, proj) + ".weight").cuda()  # [N, K] fp8
    s = f.get_tensor(PREFIX.format(layer, proj) + ".weight_scale").cuda().float()  # [N, 1]
    return w.t(), s  # column-major [K, N], as vLLM holds it after loading


def median_us(b, sb, flush, dirty=None, iters=50):
    a = (torch.randn(M, b.shape[0], device="cuda") * 0.1).to(torch.float8_e4m3fn)
    sa = torch.ones(M, 1, device="cuda")
    times = []
    for i in range(iters + 5):
        flush.sum()  # read-only sweep: evicts L2 without leaving dirty lines
        if dirty is not None:
            dirty.fill_(i)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        ops.cutlass_scaled_mm(a, b, sa, sb, torch.bfloat16)
        e1.record()
        torch.cuda.synchronize()
        if i >= 5:
            times.append(e0.elapsed_time(e1) * 1e3)
    return sorted(times)[len(times) // 2]


def main():
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    with safe_open(sys.argv[1], "pt") as f:
        cases = {
            "GDN out_proj, layer 0": load(f, 0, "linear_attn.out_proj"),
            "GDN out_proj, layer 1": load(f, 1, "linear_attn.out_proj"),
            "attention o_proj, layer 3": load(f, 3, "self_attn.o_proj"),
            "attention o_proj, layer 7": load(f, 7, "self_attn.o_proj"),
        }
    print(f"L2: {torch.cuda.get_device_properties(0).L2_cache_size >> 20} MB")
    print("\n1. same kernel, same shape, different weights")
    for name, (b, s) in cases.items():
        t = median_us(b, s, flush)
        print(f"  {name:28s} {t:7.1f} us  {b.numel() / t / 1e3:6.1f} GB/s")
    print("\n2. dirty L2 before the GEMM")
    b, s = cases["attention o_proj, layer 3"]
    for mb in (0, 2, 4, 8, 12, 16, 24, 32):
        dirty = torch.empty(mb << 20, dtype=torch.uint8, device="cuda") if mb else None
        print(f"  {mb:3d} MB written before: {median_us(b, s, flush, dirty):7.1f} us")


if __name__ == "__main__":
    main()
