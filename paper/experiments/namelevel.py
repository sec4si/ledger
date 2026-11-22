"""Name-level view vs. Ledger's per-weight view of the same launches.

    python namelevel.py <capture dir> --ceiling-gbps 252

Groups every launch in the capture by kernel name only (what a name-level profile reports), then
splits the group that holds the most time by the weight each launch streamed.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ledger.analyze import build, group_name  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summary(us):
    return {"n": len(us), "total_ms": sum(us) / 1e3, "median": statistics.median(us),
            "p10": pct(us, 0.1), "p90": pct(us, 0.9), "cv": statistics.pstdev(us) / statistics.mean(us)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--ceiling-gbps", type=float, default=252)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    steps = json.load(open(os.path.join(a.capture, "run.json")))["steps"]
    rows, _, _ = build(a.capture, a.ceiling_gbps)
    by_name = defaultdict(list)
    for r in rows:
        by_name[r["name"]].append(r)
    ranked = sorted(by_name.items(), key=lambda kv: -sum(r["us"] for r in kv[1]))
    out = {"steps": steps, "names": [], "splits": []}
    for name, rs in ranked[: a.top]:
        s = summary([r["us"] for r in rs])
        s["name"] = name
        s["weights"] = len({group_name(r["weights"][0]["name"]) for r in rs if r["weights"]})
        out["names"].append(s)
    for name, rs in ranked[: a.top]:
        groups = defaultdict(list)
        for r in rs:
            if r["weights"]:
                w = r["weights"][0]
                groups[(group_name(w["name"]), tuple(w["shape"]), r["grid"])].append(r["us"])
        for (w, shape, grid), us in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
            s = summary(us)
            s.update(name=name, weight=w, shape=list(shape), grid=list(grid))
            out["splits"].append(s)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
