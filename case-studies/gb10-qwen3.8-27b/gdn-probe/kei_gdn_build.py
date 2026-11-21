"""Build gdn_mtp.cu as a torch extension registering torch.ops.kei_gdn.*. KEI_GDN_FLAGS adds nvcc flags
(use --use_fast_math to match the shipped vLLM op bit for bit), KEI_GDN_ARCH sets the target."""

import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))


def build():
    extra = os.environ.get("KEI_GDN_FLAGS", "").split()
    arch = os.environ.get("KEI_GDN_ARCH", "compute_121,code=sm_121")
    tag = os.environ.get("KEI_GDN_TAG", "default")
    os.makedirs(os.path.join(HERE, "build-" + tag), exist_ok=True)
    load(
        name="kei_gdn_" + tag.replace("-", "_"),
        sources=[os.path.join(HERE, "gdn_mtp.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=" + arch] + extra,
        build_directory=os.path.join(HERE, "build-" + tag),
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.kei_gdn
