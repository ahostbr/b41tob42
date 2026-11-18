"""Fails (exit 1) when a folder holds map data or game files: .lotheader, .lotpack, chunkdata_*.bin, .pzby, .pack, .tiles,
worldmap*.bin, pyramid zips, .jar or .class, anywhere except tests/fixtures/. The b41tob42 package must contain our code only.

  python check_no_map_data.py <folder>
"""
import os
import re
import sys

BANNED = re.compile(r"(\.lotheader|\.lotpack|\.pzby|\.pack|\.tiles|\.jar|\.class)$|^chunkdata_.*\.bin$|^worldmap.*\.bin$|pyramid.*\.zip$",
                    re.I)


def offenders(root):
    out = []
    for dp, dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root).replace("\\", "/")
        if rel == "tests/fixtures" or rel.startswith("tests/fixtures/"):
            continue
        out += [os.path.join(rel, f).replace("\\", "/") for f in fn if BANNED.search(f)]
    return sorted(out)


def main(argv):
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    bad = offenders(argv[0])
    for b in bad:
        print("map data or game file:", b)
    print("FAIL" if bad else "OK: no map data or game files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
