"""Turn (function, PC offset) into the SASS instruction and, when the cubin has line info, the source line.

    python -m ledger.sass <cubin dir> <function> <offset> [<offset> ...]
"""

import os
import re
import shutil
import struct
import subprocess
import sys
from functools import lru_cache


def text_sections(path):
    """Function names of an ELF cubin, from its .text.<name> section headers."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return []
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    def section(i):
        name, _type, _flags, _addr, offset, size = struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize)
        return name, offset, size
    _n, stroff, strsize = section(shstrndx)
    names = data[stroff : stroff + strsize]
    out = []
    for i in range(shnum):
        n = section(i)[0]
        s = names[n : names.index(b"\0", n)].decode(errors="replace")
        if s.startswith(".text."):
            out.append(s[len(".text."):])
    return out


NVDISASM = shutil.which("nvdisasm") or os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvdisasm")


class Cubins:
    def __init__(self, directory):
        self.where = {}
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            for fn in text_sections(path):
                self.where.setdefault(fn, path)

    @lru_cache(maxsize=None)
    def disassemble(self, path):
        """{function: {offset: (instruction, source)}} of one cubin (nvdisasm -fun wants indices)."""
        text = subprocess.run([NVDISASM, "-g", "-c", path], capture_output=True, text=True).stdout
        out, current, source = {}, None, ""
        for line in text.splitlines():
            m = re.match(r"//-+ \.text\.(\S+)", line)
            if m:
                current, source = out.setdefault(m.group(1), {}), ""
                continue
            m = re.search(r'//## File "([^"]+)", line (\d+)', line)
            if m:
                source = f"{os.path.basename(m.group(1))}:{m.group(2)}"
                continue
            m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
            if m and current is not None:
                current[int(m.group(1), 16)] = (re.sub(r"\s+", " ", m.group(2)), source)
        return out

    def listing(self, function):
        path = self.where.get(function)
        return self.disassemble(path).get(function, {}) if path else {}

    def at(self, function, offset):
        return self.listing(function).get(offset, ("?", ""))


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    cubins = Cubins(sys.argv[1])
    for off in sys.argv[3:]:
        ins, src = cubins.at(sys.argv[2], int(off, 0))
        print(f"{off}: {ins}  {src}")


if __name__ == "__main__":
    main()
