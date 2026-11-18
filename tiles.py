"""WS1 T4-T6: B41 -> B42 tile table.

Step zero (Revision 1.3): a B41 tile name that already exists in B42 maps to itself. "Exists in B42" =
named in the installed game's tile definitions (media/tile_definitions/*.tiles, moved there in 42.22), in
the tile table of a vanilla B42 lotheader, or in a .tiles file the map ships for B42.
Second: positional votes from the calibration pairs (same square, same stack height, B41 name -> author's
B42 name), majority wins. Last (T5): closest match, reported, never capped.
"""
import collections
import json
import os
import re
import struct

import b41

GAME = None  # the Project Zomboid install folder: the CLI passes --game; the repo's dev tools set it from pzlib.GAME_DIR
NAME_RE = re.compile(r"^(.*)_(\d+)$")
FAMILIES = ("wall", "floor", "door", "window", "roof", "stair", "fence", "furniture", "fixture", "light",
            "street", "vegetation", "tree", "trim", "carpentry", "appliance", "sign")


def read_tdef(path):
    """[(name, property count)] from a binary TileZed tile definition file ("tdef"), name = <tileset>_<index>.
    Every slot of every tileset is listed; the .tiles.txt the game ships beside it lists exactly the slots
    with >= 1 property (measured on tiledefinitions_erosion.tiles: 909 + 797 zero-property slots)."""
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == b"tdef", path
    p = 4
    version, n = struct.unpack_from("<ii", d, p); p += 8
    names = []

    def line():
        nonlocal p
        e = d.index(b"\n", p)
        s = d[p:e].decode("utf-8", "replace").strip()
        p = e + 1
        return s
    for _ in range(n):
        ts = line(); line()
        cols, rows = struct.unpack_from("<ii", d, p); p += 8
        if version >= 1:
            p += 4  # tileset id
        count, = struct.unpack_from("<i", d, p); p += 4
        for i in range(count):
            k, = struct.unpack_from("<i", d, p); p += 4
            names.append(("%s_%d" % (ts, i), k))
            for _ in range(k):
                line(); line()
    return names


def known_b42(game=None, map_tiles=()):
    """-> {name: source} of tile names that exist in B42."""
    game = game or GAME
    if not game:
        raise ValueError("known_b42: no Project Zomboid install folder (pass game=, the CLI's --game)")
    known = {}
    td = os.path.join(game, "media", "tile_definitions")
    later = {}
    for fn in sorted(os.listdir(td)):
        if fn.endswith(".tiles"):
            for n, k in read_tdef(os.path.join(td, fn)):
                (known if k else later).setdefault(n, "tiledefs:" + fn if k else "tiledefs_noprops:" + fn)
    van = os.path.join(game, "media", "maps", "Muldraugh, KY")
    for cell, (h, _) in sorted(b41.map_cells(van).items()):
        for n in b41.read_header(h)["tiles"]:
            known.setdefault(n, "vanilla_lots")
    for item in map_tiles:  # path, or (path, label)
        path, label = item if isinstance(item, tuple) else (item, "map_tiles")
        for n, k in read_tdef(path):
            known.setdefault(n, "%s:%s" % (label, os.path.basename(path)))
    for n, src in later.items():  # a slot with no properties: the sheet has it, nothing defines it
        known.setdefault(n, src)
    return known


def family(name):
    low = name.lower()
    for f in FAMILIES:
        if f in low:
            return f
    return None


def load_tsv(path):
    """tables/b41_to_b42_tiles.tsv -> {b41: b42 or "" (= drop)} (save_tsv's format; the mod reads the same file)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            out[p[0]] = p[1] if len(p) > 1 else ""
    return out


class Table:
    def __init__(self, known, preset=None):
        self.known = known
        self.preset = preset or {}  # a shipped table (load_tsv): used after "exists in B42", before votes
        self.votes = collections.defaultdict(collections.Counter)  # b41 -> Counter(b42 name)
        self.vote_maps = collections.defaultdict(set)
        by_set = collections.defaultdict(list)
        for n in known:
            m = NAME_RE.match(n)
            if m:
                by_set[m.group(1)].append(int(m.group(2)))
        self.by_set = {k: sorted(v) for k, v in sorted(by_set.items())}  # sorted: the Java port iterates a TreeSet
        # last-resort pick per family: the alphabetically first known name of it (a rule the in-mod Java port can
        # apply to the live sprite names: B41Convert.TileTable)
        self.family_pick = {}
        for n in sorted(known):
            f = family(n)
            if f:
                self.family_pick.setdefault(f, n)

    def add_pairs(self, pairs, map_name):
        """pairs: {(b41 name, b42 name or None = the author removed it there): squares}"""
        for (a, b), n in pairs.items():
            self.votes[a][b] += n
            self.vote_maps[a].add(map_name)

    def resolve(self, name):
        """-> (b42 name or None, how, votes). how: exists_in_b42 | exists_slot_noprops | exists_via:<label>
        | calibration_vote | author_dropped (b42 None: the author's B42 version removed it) | None"""
        v = self.votes.get(name)
        if name in self.known:
            src = self.known[name]
            how = ("exists_slot_noprops" if src.startswith("tiledefs_noprops") else
                   "exists_in_b42" if src.startswith(("tiledefs:", "vanilla_lots", "known_file")) else "exists_via:" + src.split(":")[0])
            return name, how, v[name] if v else 0
        if name in self.preset:
            b = self.preset[name] or None
            return b, "table" if b else "author_dropped", 0
        if v:  # what the author did on the same squares in their own B42 version
            b, n = v.most_common(1)[0]
            return b, "calibration_vote" if b is not None else "author_dropped", n
        return None, None, 0

    def suggest(self, name):
        """Closest match for an unmapped name. -> (suggestion or None, how)"""
        m = NAME_RE.match(name)
        if m and m.group(1) in self.by_set:
            idx = int(m.group(2))
            near = min(self.by_set[m.group(1)], key=lambda i: (abs(i - idx), i))
            return "%s_%d" % (m.group(1), near), "same_tileset_nearest_index"
        if m:  # the B42 tileset whose name shares the most leading words, holding the same index if possible
            idx, words = int(m.group(2)), m.group(1).split("_")
            best, best_k = None, 0
            for ts, ids in self.by_set.items():
                tw = ts.split("_")
                k = 0
                while k < min(len(tw), len(words)) and tw[k] == words[k]:
                    k += 1
                if k == 0:
                    continue
                key = (k, idx in ids, -abs(len(tw) - len(words)))
                if best is None or key > best_k:
                    best, best_k = ts, key
            if best and best_k[0] >= 1 and family(best) == family(name):
                ids = self.by_set[best]
                near = min(ids, key=lambda i: (abs(i - idx), i))
                return "%s_%d" % (best, near), "closest_tileset_name:%s" % best
        f = family(name)
        if f and f in self.family_pick:
            return self.family_pick[f], "same_family:" + f
        return None, "no_suggestion"

    def to_json(self, names):
        out = {}
        for n in sorted(names):
            b, how, votes = self.resolve(n)
            e = dict(b42=b, how=how, votes=votes, source_maps=sorted(self.vote_maps.get(n, ())))
            if b is None:
                e["suggestion"], e["how"] = self.suggest(n)
            if n in self.votes:
                e["author_choices"] = self.votes[n].most_common(3)
            out[n] = e
        return out


def write_review(path, map_name, unmapped, total_instances):
    """review/unmapped_<map>.md: every unmapped tile with count, suggestion, the author's own B42 choice."""
    lines = ["# Unmapped tiles: %s" % map_name, "",
             "%d tile names, %d placements, of %d placements in the map. The agent review pass edits "
             "tables/b41_to_b42_tiles.json and re-runs; see docs/plans/b41-to-b42-converter/sub-ws1." % (
                 len(unmapped), sum(u["count"] for u in unmapped), total_instances), "",
             "| B41 tile | placements | suggestion | how | author's B42 choice |", "|---|---|---|---|---|"]
    for u in unmapped:
        lines.append("| %s | %d | %s | %s | %s |" % (u["b41_name"], u["count"], u["suggestion"] or "",
                                                    u.get("how") or "", u.get("author_b42_choice") or ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_tsv(path, table, names):
    """tables/b41_to_b42_tiles.tsv for the in-mod converter (B41Convert.readTable): b41 TAB final b42 ("" = drop)
    TAB how. Names that exist in B42 are left out (the mod checks the live sprite names first)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# b41\tb42 (empty = drop)\thow  -- tools/pz/convert/tiles.py save_tsv\n")
        for n in sorted(names):
            b, how, _ = table.resolve(n)
            if how and how.startswith(("exists_in_b42", "exists_slot_noprops", "exists_via")):
                continue
            if b is None and how != "author_dropped":
                b, how = table.suggest(n)
            f.write("%s\t%s\t%s\n" % (n, b or "", how))


def save_known(path, known):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for n in sorted(known):
            f.write(n + "\n")


def load_known(path):
    with open(path, encoding="utf-8") as f:
        return {line.strip(): "known_file" for line in f if line.strip()}


def save_table(path, table_json):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")  # one tile per line: diffs between table versions stay readable
        items = sorted(table_json.items())
        for i, (k, v) in enumerate(items):
            f.write(" %s: %s%s\n" % (json.dumps(k), json.dumps(v, sort_keys=True), "," if i < len(items) - 1 else ""))
        f.write("}\n")
