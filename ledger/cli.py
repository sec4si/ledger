"""Command line entry point for Ledger's capture and analysis tools."""

import argparse
from importlib import import_module
from pathlib import Path
import subprocess
import sys

from ledger import ceilings
from ledger.analyze import load_run

COMMANDS = {
    "analyze": "Summarize launches, tensors, and bandwidth",
    "steps": "Explain where time goes in each step",
    "diff": "Compare two captures",
    "report": "Write a combined launch and step report from a capture",
    "perfetto": "Export a capture as a Perfetto timeline",
    "transfers": "Account for memory copies",
    "dataflow": "Find dead writes and hand-offs in an access trace",
    "dead-across": "Compare dead writes across traces",
    "traffic": "Compare needed bytes with measured memory traffic",
    "sass": "Look up a GPU instruction",
    "perturb-plan": "Generate replay perturbation experiments",
    "perturb-report": "Report results of perturbation experiments",
    "sibyl-regions": "Convert a Sibyl recording's memory map",
    "build": "Build the CUDA capture libraries",
    "ceilings": "Measure and cache a GPU's bandwidth and compute ceilings",
    "profile": "Capture a vLLM run and write its report and timeline",
}

REPORT_OPTIONS = {
    "analyze": ("--out", "--steps", "--sms", "--peak-tflops"),
    "steps": ("--out",),
    "diff": ("--out", "--min-ms", "--tolerance", "--fail-above", "--fail-min-ms"),
    "perfetto": ("--out",),
}


def call_main(module, command, args):
    """Pass CLI arguments to an existing module without changing its entry point."""
    argv = sys.argv
    try:
        sys.argv = [f"ledger {command}", *args]
        return import_module(f"ledger.{module}").main()
    except ModuleNotFoundError as error:
        if error.name != "numpy":
            raise
        raise SystemExit("ledger: install the dataflow extra with pip install 'ledger[dataflow]'.") from error
    finally:
        sys.argv = argv


def capture_device(captures, required):
    """Read the GPU model from the captures so offline analysis needs no GPU."""
    names = []
    for capture in captures:
        try:
            device = load_run(capture).get("device")
        except (OSError, ValueError, AttributeError) as error:
            if not required:
                return None
            raise SystemExit(f"ledger: could not read {Path(capture) / 'run.json'}: {error}") from error
        if not isinstance(device, str) or not device.strip():
            if not required:
                return None
            raise SystemExit(f"ledger: {Path(capture) / 'run.json'} does not identify the GPU. "
                             "Pass --ceiling-gbps explicitly.")
        names.append(device)
    if len(set(names)) != 1:
        raise SystemExit("ledger: the captures name different GPU models. Pass --ceiling-gbps explicitly.")
    return names[0]


def report_ceilings(command, captures, bandwidth, peaks=None):
    """Fill in missing ceilings using the GPU model recorded in the capture."""
    need_compute = command == "analyze" and peaks is None
    if bandwidth is not None and not need_compute:
        return bandwidth, peaks

    required = bandwidth is None and command != "perfetto"
    device = capture_device(captures, required)
    cached = ceilings.load(device) if device else None
    if cached is None:
        if required:
            raise SystemExit(f"ledger: no cached ceilings for {device} ({ceilings.cache_path(device)}). "
                             "Run 'ledger ceilings' on that GPU or pass --ceiling-gbps explicitly.")
        return bandwidth, peaks
    if bandwidth is None:
        bandwidth = str(cached["ceiling_gbps"])
    if need_compute:
        peaks = ceilings.peak_argument(cached["peak_tflops"])
    return bandwidth, peaks


def report_args(command, args):
    """Prepend only missing options, leaving user arguments with the module that owns them."""
    options = args[:args.index("--")] if "--" in args else args
    if any(option == "-h" or (option.startswith("--h") and "--help".startswith(option)) for option in options):
        return args
    parser = argparse.ArgumentParser(prog=f"ledger {command}", add_help=False)
    parser.add_argument("capture")
    if command == "diff":
        parser.add_argument("other")
    parser.add_argument("--ceiling-gbps")
    for option in REPORT_OPTIONS[command]:
        parser.add_argument(option)
    opts, unknown = parser.parse_known_args(args)
    if unknown:
        return args
    captures = [opts.capture, opts.other] if command == "diff" else [opts.capture]
    explicit_peaks = getattr(opts, "peak_tflops", None)
    bandwidth, peaks = report_ceilings(command, captures, opts.ceiling_gbps, explicit_peaks)
    defaults = []
    if opts.ceiling_gbps is None and bandwidth is not None:
        defaults += ["--ceiling-gbps", bandwidth]
    if explicit_peaks is None and peaks is not None:
        defaults += ["--peak-tflops", peaks]
    return [*defaults, *args]


def main():
    parser = argparse.ArgumentParser(prog="ledger", description=__doc__,
                                     epilog="Reports and timelines can use the cache from 'ledger ceilings'.")
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command, help_text in COMMANDS.items():
        subcommands.add_parser(command, help=help_text, add_help=False)

    args = sys.argv[1:]
    selected = parser.parse_args(args[:1]).command
    if selected is None:
        parser.print_help()
        return
    rest = args[1:]
    try:
        if selected == "ceilings":
            return ceilings.main(rest)
        if selected in ("build", "profile", "report"):
            from ledger import profile

            handlers = {"build": profile.build_main, "profile": profile.main, "report": profile.report_main}
            return handlers[selected](rest)
        if selected in REPORT_OPTIONS:
            rest = report_args(selected, rest)
        return call_main(selected.replace("-", "_"), selected, rest)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ledger: {error}") from error


if __name__ == "__main__":
    main()
