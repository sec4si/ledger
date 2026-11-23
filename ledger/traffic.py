"""Compare what each launch had to read with what it pulled from memory.

Needed bytes are what the launch's computation touches: the weight it streams (as in the main report:
its largest referenced weight and the tensors of the same module) plus the distinct non-weight bytes the
access tracer saw it read. Pulled bytes are what the L2 filled from memory while it ran, from an Nsight
Compute profile of the same step (application replay, caches left as the program leaves them):

    ncu --replay-mode application --cache-control none --profile-from-start off --print-kernel-base mangled \
        --metrics gpu__time_duration.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum --csv <program> > profile.csv

A launch pulling far more than it needs re-reads data that fell out of the L2, or fetches it uncoalesced;
one pulling about what it needs but running slow is limited by something other than memory; one pulling
less than it needs found its inputs still in the L2, written there by the launch before it.

    python -m ledger.traffic <access dir> <launches file | -> <profile.csv> --regions <regions json>
        [--anchor <kernel>] [--traced-weights <name part>] [--out report.md]

With "-" for the launches file the step is the second full anchor-to-anchor span of the trace. Weights
whose name contains --traced-weights were traced (LEDGER_ACCESS_TRACE_WEIGHTS) and count as what was
actually read, not as the whole tensor.
"""

import argparse
import csv
import json
from collections import defaultdict

from ledger import analyze as A
from ledger.dataflow import SUB, load_access


def read_profile(path):
    """Per launch, in order: name and metric values."""
    rows = defaultdict(dict)
    with open(path) as f:
        lines = [line for line in f if line.startswith('"')]
    for r in csv.DictReader(lines):
        rows[int(r["ID"])]["name"] = r["Kernel Name"]
        rows[int(r["ID"])][r["Metric Name"]] = float(r["Metric Value"].replace(",", "") or 0)
    return [rows[i] for i in sorted(rows)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("access")
    ap.add_argument("launches")
    ap.add_argument("profile")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--ceiling-gbps", type=float, default=252, help="read bandwidth ceiling, as in the main report")
    ap.add_argument("--anchor", help="first kernel of a step, when no launches file is given")
    ap.add_argument("--traced-weights", default="", help="name part of weights the trace covers")
    ap.add_argument("--out")
    args = ap.parse_args()

    meta, shift, lid, kind, gran, mask, _pcs = load_access(args.access)
    kernels = [m for m in meta if m["name"] not in ("memcpy", "memset")]
    names = [m["name"] for m in kernels]
    if args.launches == "-":
        at = [i for i, n in enumerate(names) if n == args.anchor]
        if len(at) < 3:
            raise SystemExit("need three anchor launches in the trace")
        step = names[at[1] : at[2]]
    else:
        step = [line.strip() for line in open(args.launches) if line.strip()]
    start = next((i for i in range(len(names) - len(step) + 1) if names[i : i + len(step)] == step), None)
    if start is None:
        raise SystemExit("the step does not appear in the trace")
    ids = [kernels[start + p]["id"] for p in range(len(step))]
    prof = read_profile(args.profile)
    pnames = [p["name"] for p in prof]
    at = next((i for i in range(len(pnames) - len(step) + 1) if pnames[i : i + len(step)] == step), None)
    if at is None:
        raise SystemExit(f"the step ({len(step)} launches) does not appear in the profile ({len(prof)} launches)")
    prof = prof[at : at + len(step)]

    # Distinct non-weight bytes each launch read.
    sub = (1 << shift) // SUB
    read_bits = defaultdict(lambda: defaultdict(int))
    wanted = set(ids)
    for L, k, g, m in zip(lid.tolist(), kind.tolist(), gran.tolist(), mask.tolist()):
        if k == 0 and L in wanted:
            read_bits[L][g] |= m
    traced = {L: sum(bin(m).count("1") for m in gs.values()) * sub for L, gs in read_bits.items()}

    mem = A.Memory(json.load(open(args.regions)))
    by_id = {m["id"]: m for m in kernels}

    def weight_bytes(i):
        params = by_id[i].get("params") or []
        hits = mem.touched(params)
        weights = [j for j in hits if mem.regions[j]["kind"] == "weight"
                   and not (args.traced_weights and args.traced_weights in mem.regions[j]["name"])]
        exact = [j for j in weights if hits[j]] or weights
        if not exact:
            return 0
        top = max(exact, key=lambda j: mem.regions[j]["size"])
        mod = A.module(mem.regions[top]["name"])
        return sum(mem.regions[j]["size"] for j in exact if A.module(mem.regions[j]["name"]) == mod)

    pretty = A.demangle(list(set(step)))
    rows = []
    for p, i in enumerate(ids):
        pr = prof[p]
        pulled = pr.get("lts__d_sectors_fill_sysmem.sum", 0) * 32
        us = pr.get("gpu__time_duration.sum", 0) / 1e3
        w = weight_bytes(i)
        if w > us * 1e3 * args.ceiling_gbps * 1.3:
            w = 0  # too short to stream the weight: a gather (embedding lookup), as in the main report
        need = w + traced.get(i, 0)
        rows.append((p, A.short(pretty[step[p]], 60), us, need, pulled))

    out = []
    emit = out.append
    total_us = sum(r[2] for r in rows)
    emit("# Ledger traffic: needed vs pulled from memory\n")
    emit(f"{len(rows)} launches, {total_us / 1e3:.2f} ms GPU time, needed {sum(r[3] for r in rows) / 1e6:.1f} MB, "
         f"pulled {sum(r[4] for r in rows) / 1e6:.1f} MB from memory into the L2.\n")
    groups = defaultdict(lambda: [0, 0.0, 0, 0])
    for _p, name, us, need, pulled in rows:
        g = groups[name]
        g[0] += 1
        g[1] += us
        g[2] += need
        g[3] += pulled
    emit("## By kernel, slowest first\n")
    emit("| kernel | launches | ms | needed MB | pulled MB | pulled / needed |")
    emit("|---|---|---|---|---|---|")
    for name, (n, us, need, pulled) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:20]:
        ratio = f"{pulled / need:.2f}" if need else "-"
        emit(f"| `{name}` | {n} | {us / 1e3:.2f} | {need / 1e6:.1f} | {pulled / 1e6:.1f} | {ratio} |")
    emit("\n## Launches pulling at least twice what they need (over 1 MB)\n")
    emit("| position | kernel | us | needed MB | pulled MB | ratio |\n|---|---|---|---|---|---|")
    over = sorted((r for r in rows if r[3] and r[4] > 2 * r[3] and r[4] > 1e6), key=lambda r: -(r[4] - r[3]))
    for p, name, us, need, pulled in over[:15]:
        emit(f"| {p} | `{name}` | {us:.1f} | {need / 1e6:.2f} | {pulled / 1e6:.2f} | {pulled / need:.1f} |")
    if not over:
        emit("None.")
    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
