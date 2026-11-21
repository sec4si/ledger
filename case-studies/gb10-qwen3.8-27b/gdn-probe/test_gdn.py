"""Compare the extension build of the GDN MTP decode kernel with the op shipped in vLLM 0.31.0.
Mode 0 must match outputs and every state slot bit for bit; mode 1 must match outputs and store only
the last position's slot. Shapes follow Qwen3.8-27B: H=16, HV=48, 4 tokens (1 + 3 drafts)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm._custom_ops  # noqa: F401,E402  loads torch.ops._C

from kei_gdn_build import build  # noqa: E402

H, HV, T, SLOTS = 16, 48, 4, 8


def inputs(seed, accepted):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s, dt=torch.bfloat16, k=1.0: (torch.randn(*s, generator=g, device="cuda") * k).to(dt)  # noqa: E731
    return dict(
        mixed_qkv=r(T, 2 * H * 128 + HV * 128),
        a=r(T, HV),
        b=r(T, HV),
        A_log=r(HV, dt=torch.float32, k=0.5),
        dt_bias=r(HV, dt=torch.float32, k=0.5),
        state_indices=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device="cuda"),
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device="cuda"),
        state=r(SLOTS, HV, 128, 128, dt=torch.float32, k=0.05),
        output_gate=r(T, HV, 128),
        norm_weight=r(128, k=0.5),
        out=torch.zeros(T, HV, 128, dtype=torch.bfloat16, device="cuda"),
    )


def run(op, x):
    x = {k: v.clone() for k, v in x.items()}
    op(x["mixed_qkv"], x["a"], x["b"], x["A_log"], x["dt_bias"], x["state_indices"], x["cu_seqlens"],
       x["num_accepted_tokens"], x["state"], x["output_gate"], x["norm_weight"], x["out"],
       128**-0.5, 1e-6, "sigmoid")
    torch.cuda.synchronize()
    return x["out"], x["state"]


def main():
    kei = build()
    ok = True
    for seed, accepted in [(1, 1), (2, 2), (3, 4)]:
        x = inputs(seed, accepted)
        ref_out, ref_state = run(torch.ops._C.fused_gdn_decode_post_conv_mtp, x)
        kei.set_write_last_only(0)
        out0, state0 = run(kei.fused_gdn_decode_post_conv_mtp, x)
        kei.set_write_last_only(1)
        out1, state1 = run(kei.fused_gdn_decode_post_conv_mtp, x)
        m0 = torch.equal(out0, ref_out) and torch.equal(state0, ref_state)
        m1_out = torch.equal(out1, ref_out)
        m1_last = torch.equal(state1[4], ref_state[4])
        m1_skip = all(torch.equal(state1[s], x["state"][s]) for s in (1, 2, 3))
        print(f"seed={seed} accepted={accepted}  mode0 bitwise={m0}  mode1 out={m1_out} "
              f"last_slot={m1_last} other_slots_untouched={m1_skip}")
        ok &= m0 and m1_out and m1_last and m1_skip
    print("ALL_OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
