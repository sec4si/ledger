"""DRAM traffic of the out_proj-shaped GEMM after a clean or a dirty predecessor, for ncu.

    ncu --replay-mode application --cache-control none --kernel-name regex:cutlass \
        --metrics gpu__time_duration.sum,lts__t_sectors.sum,lts__d_sectors.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_write.sum \
        --csv python ncu_isolation.py

Before each GEMM: a read-only sweep of 256 MB (empties L2 without dirty lines), then a predecessor
touching N MB, either by reading it (clean) or by writing it (dirty). The GEMM is vLLM's
cutlass_scaled_mm at M=4 on a [6144, 5120] FP8 weight, the shape of GDN out_proj and attention
o_proj. Launch order is fixed and printed, so ncu rows map back to cases by index.
"""

import torch
from vllm import _custom_ops as ops

M, K, N = 4, 6144, 5120
SIZES_MB = [0, 4, 12, 24]
REPS = 3


def main():
    torch.manual_seed(0)
    b = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn).t()  # column-major [K, N]
    sb = torch.ones(N, 1, device="cuda")
    a = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    sa = torch.ones(M, 1, device="cuda")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    buf = torch.empty(max(SIZES_MB) << 20, dtype=torch.uint8, device="cuda")
    ops.cutlass_scaled_mm(a, b, sa, sb, torch.bfloat16)  # warm-up, gemm index 0
    torch.cuda.synchronize()
    idx = 1
    for mb in SIZES_MB:
        for mode in ("clean", "dirty"):
            for _ in range(REPS):
                flush.sum()
                if mb:
                    view = buf[: mb << 20]
                    view.sum() if mode == "clean" else view.fill_(idx & 0xFF)
                ops.cutlass_scaled_mm(a, b, sa, sb, torch.bfloat16)
                torch.cuda.synchronize()
                print(f"CASE {idx} {mode} {mb}", flush=True)
                idx += 1


if __name__ == "__main__":
    main()
