"""Build Ledger's CUDA libraries and capture a vLLM run in a fresh directory."""

import argparse
from collections import deque
from contextlib import redirect_stdout
import fcntl
from hashlib import sha256
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys

from ledger import ceilings
from ledger.cli import call_main

CAPTURE_SOURCES = ("Makefile", "ledger_cupti.cc")
ACCESS_SOURCES = ("Makefile", "tracer.cpp", "access.h", "patches.cu")
BUILD_VARIABLES = ("CXX", "CUPTI_INC", "CUPTI_LIB", "SANITIZER", "LIBRARY_PATH")


def build_library(group, sources, targets, compiler, *flags):
    """Build once for these sources and CUDA toolkit, publishing only complete targets."""
    source = ceilings.source_dir(group)
    inputs = [(name, (source / name).read_bytes()) for name in sources]
    config = (str(compiler), compiler.stat().st_mtime_ns, flags, [os.getenv(name) for name in BUILD_VARIABLES])
    key = sha256(repr(config).encode())
    for name, content in inputs:
        key.update(name.encode())
        key.update(content)
    directory = ceilings.cache_dir() / "build" / f"{group}-{key.hexdigest()[:16]}"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = tuple(directory / target for target in targets)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stamp = directory / ".built"
        if not stamp.is_file() or not all(path.is_file() for path in outputs):
            stamp.unlink(missing_ok=True)
            for path in outputs:
                path.unlink(missing_ok=True)
            for name, content in inputs:
                (directory / name).write_bytes(content)
            subprocess.run(["make", "-C", directory, f"CUDA={compiler.parents[1]}", *flags, *targets], check=True)
            if not all(path.is_file() for path in outputs):
                raise SystemExit(f"ledger: make did not produce {', '.join(targets)} in {directory}.")
            stamp.touch()
    return outputs


def build_capture(compiler):
    return build_library("capture", CAPTURE_SOURCES, ("libledger.so",), compiler)[0]


def build_main(args):
    parser = argparse.ArgumentParser(prog="ledger build", description="Build Ledger against the CUDA toolkit.")
    parser.add_argument("--access", action="store_true", help="also build the access tracer and its device patches")
    parser.add_argument("--arch", type=int, help="access tracer SM version, e.g. 121 (default: visible GPU)")
    opts = parser.parse_args(args)
    if opts.arch is not None and (not opts.access or opts.arch <= 0):
        parser.error("--arch must be a positive SM version and requires --access")
    compiler = ceilings.cuda_nvcc()
    print(build_capture(compiler))
    if opts.access:
        arch = opts.arch
        if arch is None:
            device = ceilings.cuda_device()
            arch = 10 * device.major + device.minor
        for path in build_library("access", ACCESS_SOURCES, ("libledger_access.so", "patches.fatbin"),
                                  compiler, f"ARCH={arch}"):
            print(path)


def profile_options(args):
    parser = argparse.ArgumentParser(prog="ledger profile", description=__doc__,
                                     usage="%(prog)s [options] -- <vLLM arguments>")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=0)
    parser.add_argument("--prefill", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("ledger-out"),
                        help="new or empty capture directory (default: %(default)s)")
    split = args.index("--") if "--" in args else len(args)
    opts = parser.parse_args(args[:split])
    vllm_args = args[split + 1:]
    if not vllm_args:
        parser.error("pass vLLM arguments after --")
    if opts.steps < 1 or opts.batch < 1 or opts.prompt_tokens < 0 or opts.prefill < 0:
        parser.error("--steps and --batch must be positive, and token counts must be nonnegative")
    return opts, vllm_args


def capture(opts, vllm_args, library, out):
    """Run the example under CUPTI and retain combined stdout and stderr."""
    env = dict(os.environ, LEDGER_OUT=str(out))
    env["LD_PRELOAD"] = ":".join(filter(None, (str(library), env.get("LD_PRELOAD"))))
    runner = ceilings.source_dir("examples") / "vllm_decode.py"
    command = [sys.executable, "-u", runner, "--steps", str(opts.steps), "--batch", str(opts.batch),
               "--prompt-tokens", str(opts.prompt_tokens), "--prefill", str(opts.prefill), "--", *vllm_args]
    log = out / "vllm.log"
    print(f"Capturing vLLM output in {log}", flush=True)
    try:
        with log.open("x") as file:
            result = subprocess.run(command, env=env, stdout=file, stderr=subprocess.STDOUT)
    except FileExistsError as error:
        raise SystemExit(f"ledger: {log} already exists. Choose an empty --out directory.") from error
    if result.returncode:
        with log.open(errors="replace") as file:
            tail = "".join(deque(file, maxlen=15)).rstrip()
        raise SystemExit(f"ledger: vLLM exited with status {result.returncode}. Log: {log}\n{tail}")


def write_report(out, measured):
    common = [str(out), "--ceiling-gbps", str(measured["ceiling_gbps"])]
    output = StringIO()
    with redirect_stdout(output):
        call_main("analyze", "analyze", [*common, "--peak-tflops", ceilings.peak_argument(measured["peak_tflops"])])
        print()
        call_main("steps", "steps", common)
    report = out / "report.md"
    report.write_text(output.getvalue())
    return report


def main(args):
    opts, vllm_args = profile_options(args)
    out = opts.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"ledger: {out} is not an empty directory. Choose another --out.")
    device = ceilings.cuda_device().name
    library = build_capture(ceilings.cuda_nvcc())
    measured = ceilings.load(device) or ceilings.measure(device)
    out.mkdir(parents=True, exist_ok=True)
    capture(opts, vllm_args, library, out)
    print(write_report(out, measured))
