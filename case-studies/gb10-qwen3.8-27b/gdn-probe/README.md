# GDN store probe

`gdn_mtp.cu` is vLLM v0.31.0 `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu` (Apache-2.0),
ported from the stable libtorch API to ATen so it builds as a standalone extension, with a store
mode switch, `set_store_mode(m)`:

- `0` per-token stores, as shipped (bit-identical to the vLLM op with `--use_fast_math`);
- `1` last position only, a performance probe whose state is wrong after a rejection;
- `2` deferred stores, the layout of vLLM PR #61434 in the CUDA kernel: the state after the first
  token goes to slot 0, the inputs of the later tokens are recorded in slot 1, and the next step
  replays the accepted ones with the same arithmetic. Bit-identical to mode 0.

```sh
KEI_GDN_TAG=fastmath KEI_GDN_FLAGS=--use_fast_math python test_gdn.py   # modes 0/1 vs torch.ops._C
python test_defer.py                                                    # mode 2 vs mode 0, 10 steps
KEI_GDN_MODE=1 LD_PRELOAD=../../../capture/libledger.so LEDGER_OUT=out \
  python gdn_patch_run.py ../../../examples/vllm_decode.py --steps 40 --batch 4 -- <vLLM args>
```

Mode 1 is a performance probe: after a rejected draft the stored state is wrong. Mode 2 is correct.
