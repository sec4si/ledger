"""Measured GEMM throughput per precision, for `ledger.analyze --peak-tflops`.

    python bench/flops.py [--n 8192] [--iters 20]

BF16 through torch.matmul, FP8 (e4m3, per-tensor scales) through torch._scaled_mm, both on square
N x N x N problems; prints `bf16=<TFLOP/s>,fp8=<TFLOP/s>` last. FP4 has no portable PyTorch entry
point; measure it with the kernel the engine uses and add `fp4=<TFLOP/s>` by hand.
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
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    n, flops = a.n, 2 * a.n**3
    x = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    peaks = {"bf16": best(lambda: x @ y, flops, a.iters)}
    x8, y8 = x.to(torch.float8_e4m3fn), y.to(torch.float8_e4m3fn).t().contiguous().t()
    one = torch.ones((), device="cuda")
    peaks["fp8"] = best(lambda: torch._scaled_mm(x8, y8, one, one, out_dtype=torch.bfloat16), flops, a.iters)
    for k, v in peaks.items():
        print(f"{k:5s} {v:7.1f} TFLOP/s  ({n} x {n} x {n})")
    print(",".join(f"{k}={v:.0f}" for k, v in peaks.items()))


if __name__ == "__main__":
    main()
