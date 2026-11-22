"""Plan value-level experiments from an access trace: poison what a launch wrote, see if the output moves.

A replay harness that can run one step, poison a byte range right after the k-th launch of a kernel,
and compare the generated tokens with a baseline (Sibyl's replay_perturb does) turns this plan into a
per-launch answer to "was this computed for nothing?". Reads the trace of the harness itself and the
list of kernels it launches in its baseline step, finds that step in the trace, and writes

- <out>.snapshot: "<addr> <bytes>" per line, every device range a kernel wrote in the window, which the
  harness saves once and restores before each experiment;
- <out>.experiments: "<kernel> <occurrence> <n> <addr> <bytes> ..." per line (n ranges), each launch's
  writes cut into at most --chunks pieces;
- <out>.json: the same experiments with their position in the step, for the report.

    python -m ledger.perturb_plan <access dir> <launches file> <out prefix> [--chunks 4] [--min-bytes 4096]
"""

import argparse
import json

from ledger.dataflow import SUB, load_access


def merge(ranges):
    out = []
    for b, e in sorted(ranges):
        if out and b <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([b, e])
    return out


def chunks(ranges, n):
    """Cut merged ranges into at most n pieces of about equal size."""
    total = sum(e - b for b, e in ranges)
    target = max(1, -(-total // n))
    out, cur, size = [], [], 0
    for b, e in ranges:
        while b < e:
            take = min(e - b, target - size)
            cur.append((b, b + take))
            size += take
            b += take
            if size >= target:
                out.append(cur)
                cur, size = [], 0
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("access")
    ap.add_argument("launches")
    ap.add_argument("out")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--min-bytes", type=int, default=4096)
    args = ap.parse_args()

    meta, shift, lid, kind, gran, mask, _pcs = load_access(args.access)
    kernels = [m for m in meta if m["name"] not in ("memcpy", "memset")]
    names = [m["name"] for m in kernels]
    step = [line.strip() for line in open(args.launches) if line.strip()]
    start = next((i for i in range(len(names) - len(step) + 1) if names[i : i + len(step)] == step), None)
    if start is None:
        raise SystemExit("the harness's step does not appear in the trace")
    ids = [kernels[start + p]["id"] for p in range(len(step))]

    sub = (1 << shift) // SUB
    writes = {}
    union = []
    kernel_ids = {m["id"] for m in kernels}
    for L, k, g, m in zip(lid.tolist(), kind.tolist(), gran.tolist(), mask.tolist()):
        if k != 1 or L not in kernel_ids:
            continue
        union.append((g << shift, (g + 1) << shift))
        for bit in range(SUB):
            if m >> bit & 1:
                writes.setdefault(L, []).append(((g << shift) + bit * sub, (g << shift) + (bit + 1) * sub))

    snapshot = merge(union)
    with open(args.out + ".snapshot", "w") as f:
        for b, e in snapshot:
            f.write(f"{b} {e - b}\n")

    seen, plan = {}, []
    for p, (name, i) in enumerate(zip(step, ids)):
        occ = seen.get(name, 0)
        seen[name] = occ + 1
        ranges = merge(writes.get(i, []))
        if sum(e - b for b, e in ranges) < args.min_bytes:
            continue
        for c, piece in enumerate(chunks(ranges, args.chunks)):
            plan.append({"position": p, "id": i, "kernel": name, "occurrence": occ, "chunk": c,
                         "ranges": [[b, e - b] for b, e in piece], "bytes": sum(e - b for b, e in piece)})
    with open(args.out + ".experiments", "w") as f:
        for x in plan:
            rs = " ".join(f"{b} {n}" for b, n in x["ranges"])
            f.write(f"{x['kernel']} {x['occurrence']} {len(x['ranges'])} {rs}\n")
    json.dump(plan, open(args.out + ".json", "w"))
    print(f"step at trace launch {start}: {len(step)} launches, snapshot {len(snapshot)} ranges "
          f"{sum(e - b for b, e in snapshot) / 1e6:.1f} MB, {len(plan)} experiments")


if __name__ == "__main__":
    main()
