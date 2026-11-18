"""WS2 (Revision 1.1): write a converted map as real B42 lot files (.lotheader + world_X_Y.lotpack).

Ground truth for the format, read in the game's own loader (src_decompiled/vanilla/zombie/iso/IsoLot.java):
- lotpack: "LOTP", int version 1, int 1024, then 1024 x 8-byte entries; the loader reads an int at each
  entry (IsoLot.java:143-145). Chunk index = local chunk x * 32 + local chunk y (IsoLot.java:140-142).
- chunk stream: for z in minLevel..maxLevel, for x 0..7, for y 0..7 (IsoLot.java:151-153): int count;
  -1 then int n = skip n squares; count > 1 = int room, then count-1 tile indices.
- the room int is READ AND DISCARDED in B42 (IsoLot.java:170): rooms come from the header's room rects.
  We still write the cell-local room index (or -1) so offline readers can use it.
- header: as lothdr.py / b41.read_header version 1. Buildings belong to the cell holding their bbox min
  corner; rects are cell-relative and may run past the cell's +x/+y edge, never below 0 (measured on the
  Taylorsville, Mount Crow and Raven Creek B42 files: 3464 buildings, 0 with a negative corner, 301 over).
Writes only under the given output folder: conversion is local only.
"""
import os
import struct
import time

import numpy as np

CELL, CHUNK, CHUNKS = 256, 8, 32


TYPE_OF = {0: 0, 1: 1, 8: 3, 16: 4}  # POTChunkData.getTypeOf: a chunk whose squares all hold one of these is stored as its type


def chunkdata_bytes(bits):
    """uint8 [256][256] ([x][y]) collision bits -> chunkdata_X_Y.bin bytes, as zombie.pot.POTChunkData.save writes them."""
    out = bytearray(b"\x00\x01")
    for cy in range(CHUNKS):
        for cx in range(CHUNKS):
            c = bits[cx * CHUNK:(cx + 1) * CHUNK, cy * CHUNK:(cy + 1) * CHUNK]
            v = int(c[0, 0])
            if v in TYPE_OF and (c == v).all():
                out.append(TYPE_OF[v])
            else:
                out.append(2)
                out += c.T.astype(np.uint8).tobytes()  # index lx + ly*8
    return bytes(out)


def cell_chunkdata(m, key):
    """The output cell's collision bits from the source map's chunkdata at the same absolute squares (0 where there is none),
    as zombie.pot.POT.convertChunkData does (getOldChunkDataBits by absolute square)."""
    ox, oy = key[0] * CELL, key[1] * CELL
    out = np.zeros((CELL, CELL), dtype=np.uint8)
    cd = m.chunkdata
    if hasattr(cd, "paths"):  # a lazy map: read only the source cells this output cell overlaps
        src = [(k, cd.get(k)) for k in ((sx, sy) for sx in range(ox // m.cell, (ox + CELL - 1) // m.cell + 1)
                                        for sy in range(oy // m.cell, (oy + CELL - 1) // m.cell + 1)) if k in cd.paths]
    else:
        src = cd.items()
    for (sx, sy), arr in src:
        x0, y0 = sx * m.cell, sy * m.cell
        ax0, ay0 = max(ox, x0), max(oy, y0)
        ax1, ay1 = min(ox + CELL, x0 + m.cell), min(oy + CELL, y0 + m.cell)
        if ax0 < ax1 and ay0 < ay1:
            out[ax0 - ox:ax1 - ox, ay0 - oy:ay1 - oy] = arr[ax0 - x0:ax1 - x0, ay0 - y0:ay1 - y0]
    return out


def map_tiles(m, table):
    """B41 vocabulary -> (B42 names, gid -> B42 name or None). None = drop the tile (author removed it,
    or no B42 name and no suggestion)."""
    out = []
    for n in m.vocab:
        b, how, _ = table.resolve(n)
        if b is None and how != "author_dropped":
            b, _ = table.suggest(n)
        out.append(b)
    return out


def _line(s):
    return s.encode("utf-8") + b"\n"


def write_cell(folder, key, m, cell, names_of_gid, zpop):
    """Write one B42 cell. -> dict of numbers for the log/receipt."""
    cx, cy = key
    ox, oy = cx * CELL, cy * CELL
    idx = cell["idx"]
    src = m
    if idx is None:  # a lazy map (regrid.Map(lazy=True)): this output cell's squares are read now, nothing else is held
        src = m.view(key)
        idx = np.arange(len(src.x))
    lx = (src.x[idx] - ox).astype(np.int64)
    ly = (src.y[idx] - oy).astype(np.int64)
    lz = src.z[idx].astype(np.int64)
    # rooms of the buildings this cell owns, renumbered; old room index -> new
    rooms_new, room_map, blds = [], {}, []
    for bi in cell["buildings"]:
        ids = []
        for ri in m.buildings[bi]:
            if ri not in room_map:
                room_map[ri] = len(rooms_new)
                rooms_new.append(m.rooms[ri])
            ids.append(room_map[ri])
        blds.append(ids)
    # tiles: per square, its B42 names (dropped tiles removed); squares left empty are not written
    vocab, vid = [], {}
    stacks = []
    keep = np.zeros(len(idx), bool)
    dropped = 0
    for k, i in enumerate(idx.tolist()):
        s = src.start[i]
        st = []
        for g in src.tiles[s:s + src.count[i]].tolist():
            n = names_of_gid[g]
            if n is None:
                dropped += 1
                continue
            v = vid.get(n)
            if v is None:
                v = vid[n] = len(vocab)
                vocab.append(n)
            st.append(v)
        stacks.append(st)
        keep[k] = bool(st)
    levels = [int(lz[keep].min()), int(lz[keep].max())] if keep.any() else [0, 0]
    if rooms_new:
        levels = [min(levels[0], min(r["level"] for r in rooms_new)), max(levels[1], max(r["level"] for r in rooms_new))]
    minl, maxl = min(levels[0], 0), levels[1]
    # --- header
    h = bytearray(b"LOTH")
    h += struct.pack("<ii", 1, len(vocab))
    for n in vocab:
        h += _line(n)
    h += struct.pack("<iiii", CHUNK, CHUNK, minl, maxl)
    h += struct.pack("<i", len(rooms_new))
    for r in rooms_new:
        sx, sy = r["cell"][0] * m.cell, r["cell"][1] * m.cell  # B41 objects are relative to their source cell
        h += _line(r["name"]) + struct.pack("<ii", r["level"], len(r["rects"]))
        for x, y, w, hh in r["rects"]:
            h += struct.pack("<iiii", x - ox, y - oy, w, hh)
        h += struct.pack("<i", len(r["objects"]))
        for t, x, y in r["objects"]:
            h += struct.pack("<iii", t, x + sx - ox, y + sy - oy)
    h += struct.pack("<i", len(blds))
    for b in blds:
        h += struct.pack("<i%di" % len(b), len(b), *b)
    h += zpop.astype(np.uint8).tobytes()
    # --- pack: squares sorted by (chunk index, z, x, y) = the loader's read order
    order = [k for k in range(len(idx)) if keep[k]]
    ci = (lx // CHUNK) * CHUNKS + (ly // CHUNK)
    per = (maxl - minl + 1) * CHUNK * CHUNK
    pos_in = (lz - minl) * CHUNK * CHUNK + (lx % CHUNK) * CHUNK + (ly % CHUNK)
    order.sort(key=lambda k: (int(ci[k]), int(pos_in[k])))
    room_of = src.room[idx]
    head = b"LOTP" + struct.pack("<ii", 1, CHUNKS * CHUNKS)
    table_at = len(head)
    body = bytearray(head + b"\x00" * (8 * CHUNKS * CHUNKS))
    j = 0
    for c in range(CHUNKS * CHUNKS):
        struct.pack_into("<q", body, table_at + 8 * c, len(body))
        at = 0
        while j < len(order) and int(ci[order[j]]) == c:
            k = order[j]
            p = int(pos_in[k])
            if p > at:
                body += struct.pack("<ii", -1, p - at)
            st = stacks[k]
            r = room_map.get(int(room_of[k]), -1) if room_of[k] >= 0 else -1
            body += struct.pack("<ii%di" % len(st), len(st) + 1, r, *st)
            at = p + 1
            j += 1
        if at < per:
            body += struct.pack("<ii", -1, per - at)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "%d_%d.lotheader" % key), "wb") as f:
        f.write(h)
    with open(os.path.join(folder, "world_%d_%d.lotpack" % key), "wb") as f:
        f.write(body)
    if m.chunkdata:  # navigation / collision bits (MapCollisionData), only when the source map has them
        with open(os.path.join(folder, "chunkdata_%d_%d.bin" % key), "wb") as f:
            f.write(chunkdata_bytes(cell_chunkdata(m, key)))
    return dict(squares=len(order), levels=[minl, maxl], tiles_total=sum(len(stacks[k]) for k in order),
                tiles_unmapped=dropped, buildings_out=len(blds), rooms_out=len(rooms_new))


def roundtrip(m, cells, table, folder):
    """Read the written files back (b41.read_header/read_pack, the same reader the calibration trusts) and
    compare with what should have been written. -> dict(ok, squares_expected, squares_read, squares_equal,
    stacks_equal, rooms_equal, buildings_equal, zpop_equal, trailing_bytes, first_diff)"""
    import b41
    names = map_tiles(m, table)
    rooms_ok = blds_ok = zpop_ok = chunk_ok = sq_equal = stacks_equal = True
    n_exp = n_got = 0
    trailing = 0
    first = None
    for key in sorted(cells):  # one output cell at a time: what should be in it, then what is in its files
        src, idx = m, cells[key]["idx"]
        if idx is None:  # a lazy map
            src = m.view(key)
            idx = np.arange(len(src.x))
        exp = {}
        for i in idx.tolist():
            s = src.start[i]
            st = tuple(n for n in (names[g] for g in src.tiles[s:s + src.count[i]].tolist()) if n is not None)
            if st:
                exp[(int(src.x[i]), int(src.y[i]), int(src.z[i]))] = st
        got = {}
        hp = os.path.join(folder, "%d_%d.lotheader" % key)
        h = b41.read_header(hp)
        trailing += h["trailing"]
        p = b41.read_pack(os.path.join(folder, "world_%d_%d.lotpack" % key), h)
        ox, oy = key[0] * CELL, key[1] * CELL
        for i in range(len(p["x"])):
            s = p["start"][i]
            got[(int(p["x"][i]) + ox, int(p["y"][i]) + oy, int(p["z"][i]))] = tuple(
                h["tiles"][t] for t in p["tiles"][s:s + p["count"][i]].tolist())
        want_rooms = [m.rooms[ri] for bi in cells[key]["buildings"] for ri in m.buildings[bi]]
        seen, uniq = set(), []
        for r in want_rooms:
            if id(r) not in seen:
                seen.add(id(r))
                uniq.append(r)
        got_rooms = [(r["name"], r["level"], [(x + ox, y + oy, w, hh) for x, y, w, hh in r["rects"]]) for r in h["rooms"]]
        if got_rooms != [(r["name"], r["level"], r["rects"]) for r in uniq]:
            rooms_ok = False
            first = first or ("rooms", key)
        if len(h["buildings"]) != len(cells[key]["buildings"]):
            blds_ok = False
            first = first or ("buildings", key)
        if not np.array_equal(h["zpop"], cells[key]["zpop"]):
            zpop_ok = False
            first = first or ("zpop", key)
        if m.chunkdata:
            cd = os.path.join(folder, "chunkdata_%d_%d.bin" % key)
            if not (os.path.exists(cd) and np.array_equal(b41.read_chunkdata(cd, 1), cell_chunkdata(m, key))):
                chunk_ok = False
                first = first or ("chunkdata", key)
        n_exp += len(exp)
        n_got += len(got)
        same = set(exp) == set(got)
        sq_equal = sq_equal and same
        same_stacks = same and all(exp[k] == got[k] for k in exp)
        stacks_equal = stacks_equal and same_stacks
        if not same_stacks and first is None:
            first = ("squares", sorted(set(exp) ^ set(got))[:1] or [k for k in exp if exp[k] != got.get(k)][:1])
    ok = sq_equal and stacks_equal and rooms_ok and blds_ok and zpop_ok and chunk_ok and trailing == 0
    return dict(ok=ok, squares_expected=n_exp, squares_read=n_got, squares_equal=sq_equal,
                stacks_equal=stacks_equal, rooms_equal=rooms_ok, buildings_equal=blds_ok, zpop_equal=zpop_ok,
                chunkdata_equal=chunk_ok if m.chunkdata else None,
                trailing_bytes=trailing, first_diff=repr(first) if first else None)


def write_map(m, cells, table, folder, run=None):
    """Write every output cell of a re-gridded map into folder. -> {cell: numbers}"""
    names = map_tiles(m, table)
    out = {}
    for key in sorted(cells):
        t0 = time.time()
        n = write_cell(folder, key, m, cells[key], names, cells[key]["zpop"])
        out[key] = n
        if run:
            run.log(m.name, "write", cell=key, grid="b42", ms=(time.time() - t0) * 1000,
                    squares=n["squares"], levels=n["levels"], buildings_out=n["buildings_out"], rooms_out=n["rooms_out"],
                    note="%d tile placements written, %d dropped" % (n["tiles_total"], n["tiles_unmapped"]))
    return out
