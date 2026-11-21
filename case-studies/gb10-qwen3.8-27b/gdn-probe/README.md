# GDN store probe

`gdn_mtp.cu` is vLLM v0.31.0 `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu` (Apache-2.0),
ported from the stable libtorch API to ATen so it builds as a standalone extension, plus one switch:
`set_write_last_only(1)` keeps only the last position's state store.

```sh
KEI_GDN_TAG=fastmath KEI_GDN_FLAGS=--use_fast_math python test_gdn.py   # bitwise check vs torch.ops._C
KEI_GDN_MODE=1 LD_PRELOAD=../../../capture/libledger.so LEDGER_OUT=out \
  python gdn_patch_run.py ../../../examples/vllm_decode.py --steps 40 --batch 4 -- <vLLM args>
```

Mode 1 is a performance probe: after a rejected draft the stored state is wrong.
