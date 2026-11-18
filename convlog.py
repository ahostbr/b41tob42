"""WS0: structured run log + receipt for the B41 -> B42 map converter.

One JSONL line per stage boundary (read | regrid | tiles | write | stamp | offset | calibrate) with numbers,
and one receipt per run: receipts/<run>.json. Stdlib only. Plan: docs/plans/b41-to-b42-converter (247c244).
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("read", "regrid", "tiles", "write", "stamp", "offset", "calibrate")
# the stages every cell must log in a run, by run kind and grid (R1.5: an offline run has no write/stamp
# stage). Source cells are B41 cells ("b41:46_6"), output cells are B42 cells ("b42:54_7").
REQUIRED = {
    "offline": {"b41": ("read",), "b42": ("regrid", "tiles")},
    "calibrate": {"b41": ("read",), "b42": ("regrid", "tiles")},
    "inmod": {"b41": ("read",), "b42": ("regrid", "tiles", "write")},
}
FIELDS = ("squares", "levels", "tiles_total", "tiles_mapped", "tiles_unmapped",
          "buildings_in", "buildings_out", "rooms_in", "rooms_out")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Run:
    """A converter run: Run(kind).log(...) per stage, then .receipt() / .write_receipt()."""

    def __init__(self, kind="offline", run_id=None, out_dir=None, writes=False):
        assert kind in REQUIRED, kind
        self.kind = kind
        self.required = {g: list(s) for g, s in REQUIRED[kind].items()}
        if writes and "write" not in self.required["b42"]:
            self.required["b42"].append("write")
        self.id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.out = out_dir or HERE
        os.makedirs(os.path.join(self.out, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "receipts"), exist_ok=True)
        self.log_path = os.path.join(self.out, "logs", self.id + ".jsonl")
        self._f = open(self.log_path, "w", encoding="utf-8")
        self.maps = {}  # map -> dict(source, cells{key: {stage: line}}, unmapped{}, calibration?, roundtrip?)

    def map(self, name, **source):
        return self.maps.setdefault(name, dict(source=source, cells={}, unmapped={}, calibration=None, roundtrip=None))

    def roundtrip(self, map_name, **block):
        """WS2: the written B42 files read back and compared with what should have been written."""
        self.map(map_name)["roundtrip"] = block

    def log(self, map_name, stage, cell=None, grid=None, ms=0, **nums):
        assert stage in STAGES, stage
        assert (cell is None) == (grid is None) and grid in (None, "b41", "b42"), (cell, grid)
        bad = set(nums) - set(FIELDS) - {"note", "rooms_split"}
        assert not bad, bad
        line = dict(ts=now(), run=self.id, map=map_name, stage=stage, grid=grid,
                    cell=list(cell) if cell is not None else None, ms=int(ms), **nums)
        self._f.write(json.dumps(line, sort_keys=True) + "\n")
        self._f.flush()
        if cell is not None:  # run-wide lines (map "*", or no cell) are in the log only
            self.map(map_name)["cells"].setdefault("%s:%d_%d" % ((grid,) + tuple(cell)), {})[stage] = line
        return line

    def unmapped(self, map_name, b41_name, count, suggestion, author_b42_choice=None, how=None):
        u = self.map(map_name)["unmapped"]
        e = u.setdefault(b41_name, dict(b41_name=b41_name, count=0, suggestion=suggestion,
                                        author_b42_choice=author_b42_choice, how=how))
        e["count"] += count

    def calibration(self, map_name, **block):
        """block: geometry_match, match_pct_by_level, squares_diff, levels_diff, tile_match_pct, offset, ..."""
        self.map(map_name)["calibration"] = block

    def receipt(self):
        maps = {}
        for name, m in self.maps.items():
            totals = {k: 0 for k in FIELDS if k != "levels"}
            totals["rooms_split"] = 0
            totals["cells_b41"] = sum(1 for k in m["cells"] if k.startswith("b41:"))
            totals["cells_b42"] = sum(1 for k in m["cells"] if k.startswith("b42:"))
            for c in m["cells"].values():
                for stage, line in c.items():
                    if stage == "read":
                        for k in ("buildings_in", "rooms_in"):
                            totals[k] += line.get(k, 0) or 0
                    if stage == "regrid":
                        for k in ("squares", "buildings_out", "rooms_out", "rooms_split"):
                            totals[k] += line.get(k, 0) or 0
                    if stage == "tiles":
                        for k in ("tiles_total", "tiles_mapped", "tiles_unmapped"):
                            totals[k] += line.get(k, 0) or 0
            t = totals["tiles_total"]
            totals["tiles_mapped_pct"] = round(100.0 * totals["tiles_mapped"] / t, 3) if t else None
            maps[name] = dict(
                source=m["source"], totals=totals,
                cells={k: {s: {f: v for f, v in line.items() if f not in ("ts", "run", "map", "stage", "grid", "cell")}
                           for s, line in c.items()} for k, c in sorted(m["cells"].items())},
                unmapped=sorted(m["unmapped"].values(), key=lambda e: -e["count"]),
                calibration=m["calibration"], roundtrip=m["roundtrip"])
        return dict(run=self.id, kind=self.kind, written=now(),
                    required_stages={g: list(s) for g, s in self.required.items()},
                    log=os.path.relpath(self.log_path, self.out).replace("\\", "/"), maps=maps)

    def write_receipt(self):
        self._f.close()
        path = os.path.join(self.out, "receipts", self.id + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.receipt(), f, indent=1, sort_keys=True)
            f.write("\n")
        return path
