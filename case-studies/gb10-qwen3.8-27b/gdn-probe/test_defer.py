"""Multi-step check: deferred stores (mode 2) against per-token stores (mode 0, bit-identical to the
shipped vLLM op). STEPS consecutive steps with random token counts (1..T) and accepted counts; every
step's output and the slot-0 state must match bit for bit. N requests, H=16, HV=48 (Qwen3.8-27B)."""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KEI_GDN_TAG", "fastmath")
os.environ.setdefault("KEI_GDN_FLAGS", "--use_fast_math")
from kei_gdn_build import build  # noqa: E402

H, HV, MAXT, STEPS, N = 16, 48, 4, 10, 3


def main():
    kei = build()
    rng = random.Random(0)
    g = torch.Generator(device="cuda").manual_seed(0)
    slots = 1 + N * MAXT
    base = (torch.randn(slots, HV, 128, 128, generator=g, device="cuda") * 0.05).float()
    state_ref, state_def = base.clone(), base.clone()
    idx = torch.tensor([[1 + n * MAXT + t for t in range(MAXT)] for n in range(N)], dtype=torch.int32, device="cuda")
    A_log = (torch.randn(HV, generator=g, device="cuda") * 0.5).float()
    dt_bias = (torch.randn(HV, generator=g, device="cuda") * 0.5).float()
    norm_w = (torch.randn(128, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    accepted = [1] * N
    ok = True
    for step in range(STEPS):
        T = [rng.randint(1, MAXT) for _ in range(N)]
        cu = torch.tensor([0] + list(torch.tensor(T).cumsum(0)), dtype=torch.int32, device="cuda")
        L = sum(T)
        mixed = (torch.randn(L, 2 * H * 128 + HV * 128, generator=g, device="cuda")).to(torch.bfloat16)
        a = torch.randn(L, HV, generator=g, device="cuda").to(torch.bfloat16)
        b = torch.randn(L, HV, generator=g, device="cuda").to(torch.bfloat16)
        gate = torch.randn(L, HV, 128, generator=g, device="cuda").to(torch.bfloat16)
        acc = torch.tensor(accepted, dtype=torch.int32, device="cuda")
        outs = []
        for mode, st in ((0, state_ref), (2, state_def)):
            kei.set_store_mode(mode)
            out = torch.zeros(L, HV, 128, dtype=torch.bfloat16, device="cuda")
            kei.fused_gdn_decode_post_conv_mtp(mixed, a, b, A_log, dt_bias, idx, cu, acc, st, gate, norm_w, out,
                                               128**-0.5, 1e-6, "sigmoid")
            outs.append(out)
        torch.cuda.synchronize()
        same_out = torch.equal(outs[0], outs[1])
        # Accepted counts for the next step (1..T).
        accepted = [rng.randint(1, t) for t in T]
        same_slot0 = all(torch.equal(state_ref[idx[n, 0]], state_def[idx[n, 0]]) for n in range(N))
        print(f"step {step}: T={T} next_accepted={accepted} out_bitwise={same_out} slot0_bitwise={same_slot0}")
        ok &= same_out and same_slot0
    print("ALL_OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
