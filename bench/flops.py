"""Measured GEMM throughput per precision, for `ledger.analyze --peak-tflops`.

    python bench/flops.py [--n 2048,4096,8192] [--iters 20]

BF16 through torch.matmul, FP8 (e4m3, per-tensor scales) through torch._scaled_mm, and, when vLLM
is installed, NVFP4 (16-element block scales) through vLLM's cutlass_scaled_fp4_mm, all on square
N x N x N problems, keeping the best size per precision (on GB10, GEMMs with per-row scales and
NVFP4 GEMMs slow down beyond M = 4096, so the largest size is not always the fastest); prints `bf16=<TFLOP/s>,fp8=<TFLOP/s>[,fp4=<TFLOP/s>]` last.
"""

import argparse

import torch


def best(fn, flops, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    return flops / min(times) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="2048,4096,8192")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    peaks = {}
    for n in (int(x) for x in a.n.split(",")):
        for k, v in measure(n, a.iters).items():
            print(f"{k:5s} {v:7.1f} TFLOP/s  ({n} x {n} x {n})")
            peaks[k] = max(peaks.get(k, 0), v)
    print(",".join(f"{k}={v:.0f}" for k, v in peaks.items()))


def measure(n, iters):
    flops = 2 * n**3
    x = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    peaks = {"bf16": best(lambda: x @ y, flops, iters)}
    x8, y8 = x.to(torch.float8_e4m3fn), y.to(torch.float8_e4m3fn).t().contiguous().t()
    one = torch.ones((), device="cuda")
    peaks["fp8"] = best(lambda: torch._scaled_mm(x8, y8, one, one, out_dtype=torch.bfloat16), flops, iters)
    try:
        from vllm import _custom_ops as ops

        g = torch.ones((), device="cuda")
        x4, xs = ops.scaled_fp4_quant(x, g)
        y4, ys = ops.scaled_fp4_quant(y, g)  # [N, K], as vLLM holds the weight
        peaks["fp4"] = best(lambda: ops.cutlass_scaled_fp4_mm(x4, y4, xs, ys, g, torch.bfloat16), flops, iters)
    except Exception as e:  # no vLLM, or no FP4 GEMM for this GPU
        print(f"fp4: not measured ({type(e).__name__}: {e})")
    return peaks


if __name__ == "__main__":
    main()
