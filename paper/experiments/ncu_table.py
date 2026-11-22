"""Summarize ncu_isolation output: per case (predecessor, MB), median of each metric over reps.

    python ncu_table.py ncu.csv stdout.txt
"""

import csv
import statistics
import sys
from collections import defaultdict


def num(s):
    return float(s.replace(",", "")) if s not in ("n/a", "") else float("nan")


def main():
    metrics = defaultdict(dict)
    with open(sys.argv[1]) as f:
        for r in csv.DictReader(line for line in f if line.startswith('"')):
            metrics[int(r["ID"])][r["Metric Name"]] = num(r["Metric Value"])
    cases = {}
    for line in open(sys.argv[2]):
        if line.startswith("CASE"):
            _, i, mode, mb = line.split()
            cases[int(i)] = (mode, int(mb))
    groups = defaultdict(list)
    for i, case in cases.items():
        groups[case].append(metrics[i])
    names = sorted({k for m in metrics.values() for k in m})
    print("case\t" + "\t".join(names))
    for (mode, mb), ms in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        meds = [statistics.median(m[n] for m in ms) for n in names]
        print(f"{mode} {mb}MB\t" + "\t".join(f"{v:,.0f}" for v in meds))


if __name__ == "__main__":
    main()
