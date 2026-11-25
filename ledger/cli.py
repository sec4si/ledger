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
    "profile": "Capture a vLLM run and write its report",
}

REPORT_OPTIONS = {
    "analyze": ("--out", "--steps", "--sms", "--peak-tflops"),
    "steps": ("--out",),
    "diff": ("--out", "--min-ms", "--tolerance"),
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


def capture_device(captures, missing):
    """Read the GPU model from the captures so offline analysis needs no GPU."""
    names = []
    for capture in captures:
        try:
            device = load_run(capture).get("device")
        except (OSError, ValueError, AttributeError) as error:
            raise SystemExit(f"ledger: could not read {Path(capture) / 'run.json'}: {error}") from error
        if not isinstance(device, str) or not device.strip():
            raise SystemExit(f"ledger: {Path(capture) / 'run.json'} does not identify the GPU. "
                             f"Pass {missing} explicitly.")
        names.append(device)
    if len(set(names)) != 1:
        raise SystemExit("ledger: the captures name different GPU models. Pass --ceiling-gbps explicitly.")
    return names[0]


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
    captures = [opts.capture, opts.other] if command == "diff" else [opts.capture]
    need_bandwidth = opts.ceiling_gbps is None
    need_compute = command == "analyze" and opts.peak_tflops is None
    if unknown or not (need_bandwidth or need_compute):
        return args

    missing = " and ".join(option for required, option in
                           ((need_bandwidth, "--ceiling-gbps"), (need_compute, "--peak-tflops")) if required)
    device = capture_device(captures, missing)
    cached = ceilings.load(device)
    if cached is None:
        raise SystemExit(f"ledger: no cached ceilings for {device} ({ceilings.cache_path(device)}). "
                         f"Run 'ledger ceilings' on that GPU or pass {missing} explicitly.")

    defaults = []
    if need_bandwidth:
        defaults += ["--ceiling-gbps", str(cached["ceiling_gbps"])]
    if need_compute:
        defaults += ["--peak-tflops", ceilings.peak_argument(cached["peak_tflops"])]
    return [*defaults, *args]


def main():
    parser = argparse.ArgumentParser(prog="ledger", description=__doc__,
                                     epilog="analyze, steps, and diff can use the cache from 'ledger ceilings'.")
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
        if selected in ("build", "profile"):
            from ledger import profile

            return profile.build_main(rest) if selected == "build" else profile.main(rest)
        if selected in REPORT_OPTIONS:
            rest = report_args(selected, rest)
        return call_main(selected.replace("-", "_"), selected, rest)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ledger: {error}") from error


if __name__ == "__main__":
    main()
