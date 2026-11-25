"""Measure and cache bandwidth and GEMM throughput ceilings by GPU model."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from ledger.analyze import parse_peaks


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base).expanduser().resolve() / "ledger"


def source_dir(name):
    """Locate build and benchmark inputs in a checkout or in an installed wheel."""
    package = Path(__file__).resolve().parent
    installed = package / f"_{name}"
    return installed if installed.is_dir() else package.parent / name


def cuda_nvcc():
    """Find the compiler and resolve symlinks to locate its toolkit."""
    cuda_home = os.environ.get("CUDA_HOME")
    compiler = Path(cuda_home).expanduser().resolve() / "bin" / "nvcc" if cuda_home else shutil.which("nvcc")
    if compiler and Path(compiler).is_file() and os.access(compiler, os.X_OK):
        return Path(compiler) if cuda_home else Path(compiler).resolve()
    raise SystemExit("ledger: cannot find nvcc. Set CUDA_HOME or put nvcc on PATH.")


def cuda_device():
    """Use CUDA device zero, the device selected by the bandwidth benchmark."""
    try:
        import torch
    except ImportError as error:
        raise SystemExit("ledger: a CUDA-enabled PyTorch installation is required.") from error
    if not torch.cuda.is_available():
        raise SystemExit("ledger: no CUDA device is visible to PyTorch.")
    return torch.cuda.get_device_properties(0)


def cache_path(device):
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", device).strip("-")
    return cache_dir() / f"ceilings-{name}.json"


def positive(value):
    return type(value) in (float, int) and 0 < value < math.inf


def valid_ceiling(data, device):
    if not isinstance(data, dict) or data.get("device") != device:
        return False
    peaks = data.get("peak_tflops")
    return (
        positive(data.get("ceiling_gbps"))
        and isinstance(peaks, dict)
        and bool(peaks)
        and all(key in {"bf16", "fp8", "fp4"} and positive(value) for key, value in peaks.items())
    )


def load(device):
    """Return the measured ceilings for this model, or None if no cache exists."""
    path = cache_path(device)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise SystemExit(f"ledger: cannot read {path}: {error}. Run 'ledger ceilings --refresh'.") from error
    if not valid_ceiling(data, device):
        raise SystemExit(f"ledger: invalid ceilings in {path}. Run 'ledger ceilings --refresh'.")
    return data


def save(data):
    """Publish a complete result so concurrent readers never see a partial JSON file."""
    path = cache_path(data["device"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)
    return path


def parse_measurements(device, bandwidth, compute):
    reads = re.findall(r"(?m)^read\s+blocks/SM=\s*\d+\s+(\S+)\s+GB/s\s*$", bandwidth)
    try:
        peaks = parse_peaks(compute.strip().splitlines()[-1])
        result = {"device": device, "ceiling_gbps": max(map(float, reads)), "peak_tflops": peaks}
        if reads and all(positive(float(value)) for value in reads) and valid_ceiling(result, device):
            return result
    except (ValueError, IndexError):
        pass
    raise SystemExit("ledger: could not read valid bandwidth and compute ceilings from the benchmarks.")


def measure(device):
    """Run the checked-in benchmarks and cache their measured peaks."""
    compiler = cuda_nvcc()
    with tempfile.TemporaryDirectory(prefix="ledger-ceilings-") as directory:
        binary = Path(directory) / "bwtest"
        source = source_dir("bench")
        subprocess.run([compiler, "-O3", "-std=c++17", "-arch=native", "-o", binary, source / "bwtest.cu"],
                       check=True)
        bandwidth = subprocess.check_output([binary], text=True)
        print(bandwidth, end="")
        compute = subprocess.check_output([sys.executable, source / "flops.py"], text=True)
        print(compute, end="")
    result = parse_measurements(device, bandwidth, compute)
    save(result)
    return result


def peak_argument(peaks):
    return ",".join(f"{key}={value}" for key, value in peaks.items())


def main(args):
    parser = argparse.ArgumentParser(prog="ledger ceilings", description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="measure again even if a cache exists")
    opts = parser.parse_args(args)
    device = cuda_device().name
    result = (None if opts.refresh else load(device)) or measure(device)
    print(f"{device}: {result['ceiling_gbps']:g} GB/s, {peak_argument(result['peak_tflops'])} TFLOP/s")
    print(cache_path(device))
