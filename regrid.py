"""WS1 T2 + T3: load a map folder into absolute squares, re-grid to B42 cells, and diff against an author's
own B42 version (calibration, with offset detection for relocated ports, Revision 1.2 / 1.5).

World SQUARE coordinates are kept between builds (research b672d60): a B41 square at absolute (x, y, z) is the
B42 square (x, y, z); only the cell (300 -> 256) and chunk (10 -> 8) grids change. Read-only on its inputs.
"""
import collections
import itertools
import os
import time

import numpy as np

import b41

K = 1 << 16  # key packing: x, y < 65536; z in -32..32

# B41 maps paint zombie intensity 0..255 per chunk, B42 maps about 0..10 (2026-11-16, last 1024 bytes of the lotheaders: Raven
# Creek B41 nonzero mean 106.7 max 255; vanilla B42 Louisville cells 47-58 x 3-19 nonzero mean 2.2 max 10; the author's own B42
# port of Raven Creek nonzero 2.0 max 7), and B42 spawns from the byte as is (IsoMetaChunk.getUnadjustedZombieIntensity = b & 255,
# times the seed's voronoi). zombie.pot.POT copies the bytes unscaled, so a B41 map converted 1:1 spawned ~13,000 zombies within
# 80 squares in boot 5. Dividing by 51 lands Raven Creek on Louisville and the author's port; a chunk with any B41 zombies keeps 1.
# Must equal B41Convert.B41_ZOMBIE_INTENSITY_DIVISOR (ConvertParityTest).
B41_ZOMBIE_INTENSITY_DIVISOR = 51


class _View:
    """The squares of one output cell: parallel arrays x, y, z, room, start, count and the tile ids they point into."""


class _Lru:
    """The few B41 cells (or chunkdata arrays) an output cell reads, kept while its neighbours need them: `capacity` entries."""

    def __init__(self, capacity):
        self.capacity, self.d, self.loads = capacity, collections.OrderedDict(), 0

    def get(self, key, load):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = self.d[key] = load()
        self.loads += 1
        while len(self.d) > self.capacity:
            self.d.popitem(last=False)
        return v


class _Chunkdata:
    """chunkdata_X_Y.bin of a lazy Map: the paths are kept, an array is read when an output cell needs it (b42write.cell_chunkdata)."""

    def __init__(self, paths, version, capacity):
        self.paths, self.version, self.cache = paths, version, _Lru(capacity)

    def __len__(self):
        return len(self.paths)

    def get(self, key):
        return self.cache.get(key, lambda: b41.read_chunkdata(self.paths[key], self.version))


class Map:
    """All squares of one map folder in absolute coordinates, tile names in one map-wide vocabulary.

    lazy=True (the converter's way, tools/pz/convert/cli.py): the squares of the B41 cells are NOT concatenated. Only headers, rooms,
    buildings and zombie density are kept (they are small), and a B41 cell's squares are read when an output cell needs them
    (view). Memory then follows one output cell plus a small cache of source cells, not the map (Project Russia, 3,600 cells,
    died allocating 1.22 GiB for one of the whole-map arrays). The calibration tools keep the default: they compare whole maps."""

    def __init__(self, folder, run=None, name=None, want_tiles=True, lazy=False):
        self.folder, self.name = folder, name or folder
        self.lazy = lazy
        self.cells = {}
        self._src = {}  # lazy: (cx, cy) -> what read_pack and the remap need
        self.vocab, self.gid = [], {}
        xs, ys, zs, rooms, starts, counts, tiles = [], [], [], [], [], [], []
        self.rooms, self.buildings = [], []  # rooms: dict(name, level, rects abs, cell, objects); buildings: [room idx]
        self.zpop = {}  # (cx, cy) -> array
        self.chunkdata = {}  # (cx, cy) -> uint8 [x][y] collision bits of the cell (chunkdata_X_Y.bin), when present
        ntiles = 0
        cd_paths = {}
        for (cx, cy), (hp, pp) in sorted(b41.map_cells(folder).items()):
            t0 = time.time()
            h = b41.read_header(hp)
            self.version, self.cell, self.chunk = h["version"], h["cell"], h["chunk"]
            remap = np.array([self._gid(n) for n in h["tiles"]] or [0], dtype=np.int32)
            room_base = len(self.rooms)
            for r in h["rooms"]:
                self.rooms.append(dict(name=r["name"], level=r["level"], cell=(cx, cy),
                                       rects=[(x + cx * self.cell, y + cy * self.cell, w, hh) for x, y, w, hh in r["rects"]],
                                       objects=r["objects"]))
            for b in h["buildings"]:
                self.buildings.append([room_base + i for i in b])
            self.zpop[(cx, cy)] = h["zpop"]
            cd = os.path.join(folder, "chunkdata_%d_%d.bin" % (cx, cy))
            if os.path.exists(cd):
                if lazy:
                    cd_paths[(cx, cy)] = cd
                else:
                    self.chunkdata[(cx, cy)] = b41.read_chunkdata(cd, h["version"])
            n = 0
            if pp and lazy:
                n = None  # counted when the cell is first read (squares_total)
                self._src[(cx, cy)] = dict(pp=pp, remap=remap, room_base=room_base,
                                           header={k: h[k] for k in ("version", "chunks", "chunk", "minlevel", "maxlevel")})
            elif pp:
                p = b41.read_pack(pp, h, want_tiles)
                n = len(p["x"])
                xs.append(p["x"] + cx * self.cell); ys.append(p["y"] + cy * self.cell); zs.append(p["z"])
                rr = p["room"].copy()
                rr[rr >= 0] += room_base
                rooms.append(rr)
                starts.append(p["start"] + ntiles); counts.append(p["count"])
                tiles.append(remap[p["tiles"]] if len(p["tiles"]) else p["tiles"])
                ntiles += len(p["tiles"])
            self.cells[(cx, cy)] = dict(squares=n, buildings=len(h["buildings"]), rooms=len(h["rooms"]),
                                        levels=[h["minlevel"], h["maxlevel"] - 1], tiles=len(h["tiles"]))
            if run:
                run.log(self.name, "read", cell=(cx, cy), grid="b41" if self.version == 0 else "b42",
                        ms=(time.time() - t0) * 1000, squares=n, levels=[h["minlevel"], h["maxlevel"] - 1],
                        buildings_in=len(h["buildings"]), rooms_in=len(h["rooms"]))
        if lazy:
            # one B41 column of cells is read by the output cells of two neighbouring columns: two B41 rows of cells stay cached
            ys_ = [k[1] for k in self._src] or [0]
            cap = 2 * (max(ys_) - min(ys_) + 1) + 4
            self._cache = _Lru(cap)
            self.chunkdata = _Chunkdata(cd_paths, getattr(self, "version", 0), cap) if cd_paths else {}
            return
        cat = lambda v: np.concatenate(v) if v else np.zeros(0, np.int32)
        self.x, self.y, self.z = cat(xs), cat(ys), cat(zs)
        self.room, self.start, self.count, self.tiles = cat(rooms), cat(starts), cat(counts), cat(tiles)

    def squares_total(self):
        """Squares holding tiles in the whole map (lazy: those of the cells read so far; after write_map, all of them)."""
        if not self.lazy:
            return int(len(self.x))
        return sum(c["squares"] or 0 for c in self.cells.values())

    def source(self, key):
        """lazy: one B41 cell's squares in absolute coordinates, room indexes and tile ids already the map's own."""
        def load():
            s = self._src[key]
            p = b41.read_pack(s["pp"], s["header"])
            rr = p["room"].copy()
            rr[rr >= 0] += s["room_base"]
            self.cells[key]["squares"] = len(p["x"])
            return dict(x=p["x"] + key[0] * self.cell, y=p["y"] + key[1] * self.cell, z=p["z"], room=rr, start=p["start"],
                        count=p["count"], tiles=s["remap"][p["tiles"]] if len(p["tiles"]) else p["tiles"])
        return self._cache.get(key, load)

    def view(self, key, cell=256):
        """lazy: the squares of output cell `key` as an object with x, y, z, room, start, count, tiles (the map's arrays, restricted),
        in the order the eager map lists them: source cells by (cx, cy), each in file order."""
        ox, oy = key[0] * cell, key[1] * cell
        parts = []
        for sx in range(ox // self.cell, (ox + cell - 1) // self.cell + 1):
            for sy in range(oy // self.cell, (oy + cell - 1) // self.cell + 1):
                if (sx, sy) in self._src:
                    s = self.source((sx, sy))
                    sel = (s["x"] >= ox) & (s["x"] < ox + cell) & (s["y"] >= oy) & (s["y"] < oy + cell)
                    if sel.any():
                        parts.append((s, sel))
        v = _View()
        names = ("x", "y", "z", "room")
        for n in names:
            setattr(v, n, np.concatenate([s[n][sel] for s, sel in parts]) if parts else np.zeros(0, np.int32))
        starts, counts, tiles, base = [], [], [], 0
        for s, sel in parts:
            cnt, st = s["count"][sel].astype(np.int64), s["start"][sel].astype(np.int64)
            new = np.cumsum(cnt) - cnt
            total = int(cnt.sum())
            tiles.append(s["tiles"][np.repeat(st - new, cnt) + np.arange(total)] if total else np.zeros(0, np.int32))
            starts.append((new + base).astype(np.int32))
            counts.append(cnt.astype(np.int32))
            base += total
        v.start = np.concatenate(starts) if starts else np.zeros(0, np.int32)
        v.count = np.concatenate(counts) if counts else np.zeros(0, np.int32)
        v.tiles = np.concatenate(tiles) if tiles else np.zeros(0, np.int32)
        return v

    def _gid(self, name):
        g = self.gid.get(name)
        if g is None:
            g = self.gid[name] = len(self.vocab)
            self.vocab.append(name)
        return g

    def keys(self, dx=0, dy=0):
        x, y = self.x.astype(np.int64) + dx, self.y.astype(np.int64) + dy
        k = ((self.z.astype(np.int64) + 32) * K + y) * K + x
        bad = (x < 0) | (x >= K) | (y < 0) | (y >= K)
        k[bad] = -1 - np.flatnonzero(bad)  # off the key grid: unique, never equal to a real square
        return k

    def stack(self, i):
        s = self.start[i]
        return self.tiles[s:s + self.count[i]]

    def building_sigs(self):
        """-> list of (signature, x0, y0) per building: bbox w/h, room count and level set of non-empty rooms."""
        out = []
        for b in self.buildings:
            x0 = y0 = 1 << 30
            x1 = y1 = -(1 << 30)
            n, lv = 0, set()
            for i in b:
                r = self.rooms[i]
                if r["name"].lower() != "emptyoutside":
                    n += 1
                    lv.add(r["level"])
                for x, y, w, h in r["rects"]:
                    x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h)
            if x1 > x0:
                out.append(((x1 - x0, y1 - y0, n, tuple(sorted(lv))), x0, y0))
        return out


def regrid(m, run=None, cell=256, chunk=8):
    """Re-grid a map's absolute squares into B42 cells. -> {(cx, cy): dict(index array, rooms, buildings,
    zpop 32x32)}. Buildings go to the cell holding their bbox min corner (rooms keep absolute rects;
    rooms whose rects cross that cell's edge are counted as rooms_split for the WS2 writer)."""
    t0 = time.time()
    out = {}
    empty = lambda: dict(idx=None if m.lazy else np.zeros(0, np.int64), buildings=[], rooms=set(), rooms_split=0)  # lazy: idx None = m.view(key)
    if not m.lazy:  # the cells that hold squares; a lazy map has no square list, and every such cell is also one a source cell overlaps (below)
        cx, cy = m.x // cell, m.y // cell
        order = np.lexsort((cy, cx))
        if len(order):
            ck = (cx[order].astype(np.int64) << 32) | (cy[order].astype(np.int64) & 0xFFFFFFFF)
            cuts = np.flatnonzero(np.diff(ck)) + 1
            for part in np.split(order, cuts):
                key = (int(cx[part[0]]), int(cy[part[0]]))
                out[key] = dict(idx=part, buildings=[], rooms=set(), rooms_split=0)
    for bi, b in enumerate(m.buildings):
        rects = [rc for i in b for rc in m.rooms[i]["rects"]]
        if not rects:
            continue
        x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
        key = (x0 // cell, y0 // cell)
        c = out.setdefault(key, empty())
        c["buildings"].append(bi)
        ox, oy = key[0] * cell, key[1] * cell
        for i in b:
            c["rooms"].add(i)
            if any(x < ox or y < oy or x + w > ox + cell or y + h > oy + cell for x, y, w, h in m.rooms[i]["rects"]):
                c["rooms_split"] += 1
    # every B42 cell the source's cells overlap gets files, empty ones too (header, an all-empty lotpack, chunkdata), as the game's own
    # converter zombie.pot.POT does (potdiff: POT writes Taylorsville's empty cell 35_24); POT additionally drops partial cells at the
    # east and south edge, which we keep (their squares hold buildings: 50427 squares on levels 0-2 in Taylorsville)
    for (sx, sy) in m.cells:
        x0, y0 = sx * m.cell, sy * m.cell
        for ncx in range(x0 // cell, (x0 + m.cell - 1) // cell + 1):
            for ncy in range(y0 // cell, (y0 + m.cell - 1) // cell + 1):
                out.setdefault((ncx, ncy), empty())
    # zombie density: B41 grid entry = one 10x10 chunk; B42 entry = one 8x8 chunk; area-weighted mean on the
    # 2-square common grid (gcd(10, 8) = 2), absent source coverage counts as 0
    src_cell, src_chunk = m.cell, m.chunk
    for key, c in out.items():
        z = np.zeros((cell // chunk, cell // chunk), dtype=np.float64)
        ox, oy = key[0] * cell, key[1] * cell
        sx = np.arange(ox, ox + cell, 2) + 1
        sy = np.arange(oy, oy + cell, 2) + 1
        gx, gy = np.meshgrid(sx, sy, indexing="ij")
        scx, scy = gx // src_cell, gy // src_cell
        val = np.zeros(gx.shape)
        # only the source cells this output cell overlaps (was: every cell of the map, 3,600 x 3,600 numpy masks on Project Russia)
        for a, b2 in itertools.product(range(int(scx.min()), int(scx.max()) + 1), range(int(scy.min()), int(scy.max()) + 1)):
            zp = m.zpop.get((a, b2))
            if zp is None:
                continue
            sel = (scx == a) & (scy == b2)
            if sel.any():
                lx = (gx[sel] - a * src_cell) // src_chunk
                ly = (gy[sel] - b2 * src_cell) // src_chunk
                val[sel] = zp[lx, ly]
        z = val.reshape(cell // chunk, chunk // 2, cell // chunk, chunk // 2).mean(axis=(1, 3))
        if src_cell == 300:  # B41 source: rescale to B42's intensity range (B41_ZOMBIE_INTENSITY_DIVISOR)
            q = np.rint(z / B41_ZOMBIE_INTENSITY_DIVISOR)
            z = np.where((z > 0) & (q == 0), 1, q)
        c["zpop"] = np.rint(z).astype(np.uint8)
        if run:
            lv = None if m.lazy else m.z[c["idx"]]  # lazy: the squares are read when the cell is written (its "write" line has them)
            run.log(m.name, "regrid", cell=key, grid="b42", ms=(time.time() - t0) * 1000,
                    squares=None if m.lazy else int(len(c["idx"])),
                    levels=[int(lv.min()), int(lv.max())] if lv is not None and len(lv) else None,
                    buildings_out=len(c["buildings"]), rooms_out=len(c["rooms"]), rooms_split=c["rooms_split"])
            t0 = time.time()
    return out


def detect_offset(a, b, ntop=6):
    """Offset (dx, dy) that moves map a onto map b. Candidates: (0, 0) plus the most-voted offsets between
    buildings of equal bbox size (authors change room lists between builds, so rooms are not in the vote).
    Winner: the candidate that lands the most upper-level squares (z >= 1) of a on occupied squares of b;
    upper floors are sparse, so a wrong offset scores near zero while ground (z 0) is filled almost everywhere.
    -> dict(dx, dy, upper_matched, votes, candidates)"""
    bsig = collections.defaultdict(list)
    for sig, x, y in b.building_sigs():
        bsig[sig[:2]].append((x, y))
    votes = collections.Counter()
    for sig, x, y in a.building_sigs():
        for (x2, y2) in bsig.get(sig[:2], ()):
            votes[(x2 - x, y2 - y)] += 1
    cands = [(0, 0)] + [k for k, _ in votes.most_common(ntop) if k != (0, 0)]
    up_a, up_b = a.z >= 1, b.z >= 1
    kb = b.keys()[up_b]
    scored = []
    for dx, dy in cands:
        ka = a.keys(dx, dy)[up_a]
        scored.append((int(np.isin(ka, kb, assume_unique=True).sum()), dx, dy))
    scored.sort(key=lambda t: (-t[0], abs(t[1]) + abs(t[2])))
    best = scored[0]
    return dict(dx=best[1], dy=best[2], upper_matched=best[0], upper_b41=int(up_a.sum()),
                votes=votes.get((best[1], best[2]), 0), buildings_a=len(a.buildings), buildings_b=len(b.buildings),
                candidates=[[dx, dy, s, votes.get((dx, dy), 0)] for s, dx, dy in scored],
                how="(0,0) + building-size votes, scored by upper-level squares matched")


def compare(a, b, dx=0, dy=0, bar=None):
    """Geometry + tile diff of map a (moved by dx, dy) against map b.
    Match % per level = squares of a, inside b's footprint, that land on an occupied square of b / squares of a
    inside b's footprint on that level. Footprint = the B42 cells b has files for: squares the author dropped
    from their B42 version (whole cells) are counted separately (outside_b42), not as mismatches."""
    ka, kb = a.keys(dx, dy), b.keys()
    common, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    fp = np.array(sorted(b.cells), dtype=np.int64).reshape(-1, 2)
    acx, acy = (a.x + dx) // b.cell, (a.y + dy) // b.cell
    inside = np.isin(acx.astype(np.int64) * K + acy, fp[:, 0] * K + fp[:, 1])
    by_level = {}
    for lv in sorted(set(np.unique(a.z).tolist()) | set(np.unique(b.z).tolist())):
        on = a.z == lv
        na, nin, nb = int(on.sum()), int((on & inside).sum()), int((b.z == lv).sum())
        nm = int((a.z[ia] == lv).sum())
        by_level[lv] = dict(b41=na, b41_inside=nin, outside_b42=na - nin, b42=nb, matched=nm,
                            pct=round(100.0 * nm / nin, 3) if nin else None,
                            pct_of_b42=round(100.0 * nm / nb, 3) if nb else None)
    pcts = {str(lv): v["pct"] for lv, v in by_level.items() if v["pct"] is not None}
    # tile comparison on matched squares, in one id space (a's vocabulary, then b-only names)
    uni = list(a.vocab)
    gid = dict(a.gid)
    for n in b.vocab:
        if n not in gid:
            gid[n] = len(uni)
            uni.append(n)
    bmap = np.array([gid[n] for n in b.vocab] or [0], dtype=np.int32)
    ca, cb = a.count[ia], b.count[ib]
    same_mask = np.zeros(len(ia), bool)
    ident = np.zeros(len(uni), np.int64)
    for c in np.unique(ca[ca == cb]).tolist():
        sel = np.flatnonzero((ca == c) & (cb == c))
        A = np.sort(a.tiles[a.start[ia[sel]][:, None] + np.arange(c)], axis=1)
        B = np.sort(bmap[b.tiles[b.start[ib[sel]][:, None] + np.arange(c)]], axis=1)
        eq = (A == B).all(axis=1)
        same_mask[sel[eq]] = True
        ident += np.bincount(A[eq].ravel(), minlength=len(uni))
    same = int(same_mask.sum())
    # votes: identical squares vote name -> same name; on the other squares, the names only one side has are
    # paired in stack order when both sides changed the same number of tiles (a rename/replacement); otherwise
    # each B41-only name votes "removed" (None)
    pairs = collections.Counter({(uni[g], uni[g]): int(n) for g in np.flatnonzero(ident).tolist() for n in [ident[g]]})
    for k in np.flatnonzero(~same_mask).tolist():
        i, j = int(ia[k]), int(ib[k])
        la = a.stack(i).tolist()
        lb = bmap[b.stack(j)].tolist()
        ra, rb = list(la), list(lb)
        for t in la:
            if t in rb:
                rb.remove(t)
                ra.remove(t)
        if ra and len(ra) == len(rb):
            for p, q in zip(ra, rb):
                pairs[(uni[p], uni[q])] += 1
        else:  # the author's B42 square lacks these B41 tiles: a vote for "removed"
            for p in ra:
                pairs[(uni[p], None)] += 1
    # bar: None (measure only) or dict(pct=X, min_level, min_squares): every level >= min_level holding >=
    # min_squares B41 squares inside the footprint must match >= X % (ground and tiny floors are reported only)
    gated = {} if bar is None else {str(lv): v["pct"] for lv, v in by_level.items()
                                    if lv >= bar["min_level"] and v["b41_inside"] >= bar["min_squares"]}
    geometry = None if bar is None else bool(gated) and all(p >= bar["pct"] for p in gated.values())
    lv_a, lv_b = set(np.unique(a.z).tolist()), set(np.unique(b.z).tolist())

    def bset(m, ddx, ddy):  # building bbox min corner -> sorted room list (name, level, absolute rects)
        out = collections.defaultdict(list)
        for bl in m.buildings:
            rects = [rc for i in bl for rc in m.rooms[i]["rects"]]
            if rects:
                rooms = sorted((m.rooms[i]["name"], m.rooms[i]["level"],
                                tuple(sorted((x + ddx, y + ddy, w, h) for x, y, w, h in m.rooms[i]["rects"]))) for i in bl)
                out[(min(r[0] for r in rects) + ddx, min(r[1] for r in rects) + ddy)].append(rooms)
        return out
    ba, bb = bset(a, dx, dy), bset(b, 0, 0)
    n_a = sum(len(v) for v in ba.values())
    same_corner = sum(len(v) for k, v in ba.items() if k in bb)
    same_rooms = sum(1 for k, v in ba.items() if k in bb for r in v if r in bb[k])
    buildings = dict(b41=n_a, b42=sum(len(v) for v in bb.values()), same_corner=same_corner, same_rooms=same_rooms,
                     same_corner_pct=round(100.0 * same_corner / n_a, 3) if n_a else None,
                     same_rooms_pct=round(100.0 * same_rooms / n_a, 3) if n_a else None)
    return dict(buildings=buildings,
        geometry_match=geometry, bar=bar, offset=[dx, dy], gated_levels=gated,
        match_pct_by_level=pcts, by_level={str(k): v for k, v in by_level.items()},
        squares_b41=int(len(ka)), squares_b42=int(len(kb)), squares_matched=int(len(common)),
        squares_diff=int(len(ka) - len(common)),
        levels_diff=dict(only_b41=sorted(lv_a - lv_b), only_b42=sorted(lv_b - lv_a)),
        tile_match_pct=round(100.0 * same / len(common), 3) if len(common) else None,
    ), pairs
