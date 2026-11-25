"""Build Ledger's CUDA libraries and capture a vLLM run in a fresh directory."""

import argparse
from collections import deque
from contextlib import redirect_stdout
import fcntl
from hashlib import sha256
from io import StringIO
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ledger import ceilings, perfetto
from ledger.cli import call_main, report_ceilings

CAPTURE_SOURCES = ("Makefile", "ledger_cupti.cc")
ACCESS_SOURCES = ("Makefile", "tracer.cpp", "access.h", "patches.cu")
BUILD_VARIABLES = ("CXX", "CUPTI_INC", "CUPTI_LIB", "SANITIZER", "LIBRARY_PATH")


def _find_first(env_name, candidates, filename_glob):
    """Return the first existing directory matching `filename_glob`, honoring an env override."""
    override = os.getenv(env_name)
    dirs = [Path(override).expanduser()] if override else list(candidates)
    for directory in dirs:
        if directory.is_dir() and any(directory.glob(filename_glob)):
            return directory
    return None


def _check_build_prereqs(group, cuda_root):
    """Fail early with an actionable message if make, C++ compiler, CUPTI, or Sanitizer is missing."""
    if not shutil.which("make"):
        raise SystemExit("ledger: cannot find 'make' on PATH. Install GNU make to build Ledger.")
    cxx = os.getenv("CXX") or "c++"
    if not shutil.which(cxx):
        raise SystemExit(f"ledger: cannot find C++ compiler '{cxx}'. Install g++ or set CXX=<compiler>.")

    if group == "capture":
        inc_candidates = [cuda_root / "extras/CUPTI/include", *sorted(cuda_root.glob("targets/*/include"))]
        if not _find_first("CUPTI_INC", inc_candidates, "cupti.h"):
            raise SystemExit(
                f"ledger: cannot find CUPTI headers (cupti.h) under {cuda_root}. "
                "Install the CUDA CUPTI development package or set CUPTI_INC=<dir>."
            )
        lib_candidates = [cuda_root / "extras/CUPTI/lib64", *sorted(cuda_root.glob("targets/*/lib"))]
        if not _find_first("CUPTI_LIB", lib_candidates, "libcupti.so*"):
            raise SystemExit(
                f"ledger: cannot find CUPTI libraries (libcupti.so) under {cuda_root}. "
                "Install the CUDA CUPTI package or set CUPTI_LIB=<dir>."
            )
    elif group == "access":
        san = Path(os.getenv("SANITIZER") or (cuda_root / "compute-sanitizer")).expanduser()
        if not (san / "include/sanitizer.h").is_file() or not any(san.glob("libsanitizer-public.so*")):
            raise SystemExit(
                f"ledger: cannot find Compute Sanitizer headers/libraries under {san}. "
                "Install CUDA Compute Sanitizer or set SANITIZER=<dir>."
            )


def _format_build_failure(group, cuda_root, result):
    """Turn a failed make invocation into a concise diagnostic instead of raw compiler spam."""
    combined = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
    hints = [
        ("cupti.h", "missing CUPTI headers; set CUPTI_INC=<dir> or CUDA_HOME=<cuda>"),
        ("-lcupti", "missing libcupti; set CUPTI_LIB=<dir> or CUDA_HOME=<cuda>"),
        ("-lcuda", "missing CUDA driver library (libcuda); check LIBRARY_PATH or driver stubs"),
        ("sanitizer.h", "missing Compute Sanitizer headers; set SANITIZER=<dir>"),
        ("-lsanitizer-public", "missing Compute Sanitizer library; set SANITIZER=<dir>"),
    ]
    hint = next((msg for needle, msg in hints if needle in combined), None)
    tail = "\n".join(line for line in combined.splitlines()[-6:] if line.strip())
    summary = f"ledger: failed to build {group} library (CUDA={cuda_root})"
    if hint:
        summary += f": {hint}."
    else:
        summary += f" (make exited with status {result.returncode})."
    return f"{summary}\n{tail}" if tail else summary


def build_library(group, sources, targets, compiler, *flags):
    """Build once for these sources and CUDA toolkit, publishing only complete targets."""
    cuda_root = compiler.parents[1]
    _check_build_prereqs(group, cuda_root)
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
            result = subprocess.run(
                ["make", "-C", directory, f"CUDA={cuda_root}", *flags, *targets],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise SystemExit(_format_build_failure(group, cuda_root, result))
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


def write_report(capture_dir, report, bandwidth, peaks):
    """Combine the launch analysis and step breakdown in one Markdown report."""
    common = [str(capture_dir), "--ceiling-gbps", str(bandwidth)]
    analyze_args = [*common, "--peak-tflops", peaks] if peaks else common
    output = StringIO()
    with redirect_stdout(output):
        call_main("analyze", "analyze", analyze_args)
        print()
        call_main("steps", "steps", common)
    report.write_text(output.getvalue())
    return report


def report_main(args):
    parser = argparse.ArgumentParser(prog="ledger report", description="Write launch and step reports together.")
    parser.add_argument("capture", type=Path, help="existing capture directory")
    parser.add_argument("--out", type=Path, help="output Markdown path (default: <capture>/report.md)")
    parser.add_argument("--ceiling-gbps", type=float, help="measured read bandwidth (default: GPU cache)")
    parser.add_argument("--peak-tflops", help="measured peaks per precision, e.g. bf16=91,fp8=181 (default: GPU cache)")
    opts = parser.parse_args(args)
    capture_dir = opts.capture.resolve()
    report = (opts.out or capture_dir / "report.md").resolve()
    bandwidth, peaks = report_ceilings("analyze", [capture_dir], opts.ceiling_gbps, opts.peak_tflops)
    print(write_report(capture_dir, report, bandwidth, peaks))


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
    bandwidth = measured["ceiling_gbps"]
    report = write_report(out, out / "report.md", bandwidth, ceilings.peak_argument(measured["peak_tflops"]))
    trace = out / "trace.json"
    perfetto.export(out, bandwidth, trace)
    print(report)
    print(f"{trace} (open at https://ui.perfetto.dev)")
