"""Convert a Sibyl recording's memory map into Ledger's regions file.

Sibyl records the engine's named tensors as `ranges-rank<N>-pid<P>.json`, a list of
[kind, name, addr, size, shape, dtype]. A replay that maps the recording at its original addresses
uses the same pointers, so the map applies to a capture of the replay unchanged. Ledger's analysis
only distinguishes weights and caches: draft weights count as weights, the KV pool as a cache.

    python -m ledger.sibyl_regions <ranges-rank0-...json> <capture dir>
"""

import json
import os
import sys

KINDS = {"weight": "weight", "draft_weight": "weight", "kv": "cache"}


def convert(ranges):
    return [
        {"kind": KINDS.get(kind, kind), "name": name, "addr": addr, "size": size, "shape": shape,
         "dtype": dtype.removeprefix("torch.")}
        for kind, name, addr, size, shape, dtype in ranges
    ]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with open(src) as f:
        regions = convert(json.load(f))
    with open(os.path.join(out, "regions-sibyl.json"), "w") as f:
        json.dump(regions, f)
    print(f"{len(regions)} regions -> {out}/regions-sibyl.json")


if __name__ == "__main__":
    main()
