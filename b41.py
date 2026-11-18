"""WS1 T1: reader for Project Zomboid lot files, B41 (version 0) and B42 (version 1).

Reading logic ported from pzmap2dzi (https://github.com/cff29546/pzmap2dzi, pzmap2dzi/lotheader.py,
binfile.py, cell.py), MIT License, Copyright (c) 2022 Min Xiang. Changes: one module, numpy output,
layers indexed from minlayer (pzmap2dzi indexes B42's negative layers from the end of the list).

  B41 .lotheader: int version (0), tile names, one 0x00 byte, chunk w, h (10), level count, rooms,
                  buildings, 30x30 zombie bytes.      lotpack: int chunk count, table, chunks.
  B42 .lotheader: "LOTH", int version (1), tile names, chunk w, h (8), min level, max level (inclusive),
                  rooms, buildings, 32x32 zombie bytes. lotpack: "LOTP", int version, chunk count, ...
  A chunk is a run-length stream over (level, x, y): int count; -1 = then int n, skip n squares;
  count > 1 = int room id, then count-1 tile indices into the header's tile names.

Stdlib + numpy. Read-only on its inputs.
"""
import os
import re
import struct
from array import array

import numpy as np

HEADER_RE = re.compile(r"^(-?\d+)_(-?\d+)\.lotheader$")
FORMATS = {0: dict(build="B41", chunks=30, chunk=10), 1: dict(build="B42", chunks=32, chunk=8)}


def _line(d, p):
    e = d.index(b"\n", p)
    return d[p:e].rstrip(b"\r").decode("utf-8", "replace").strip(), e + 1


def read_header(path):
    with open(path, "rb") as f:
        d = f.read()
    p = 0
    if d[:4] == b"LOTH":
        p = 4
    version, = struct.unpack_from("<i", d, p); p += 4
    fmt = FORMATS[version]
    n, = struct.unpack_from("<i", d, p); p += 4
    tiles = []
    for _ in range(n):
        s, p = _line(d, p)
        tiles.append(s)
    if version == 0:
        p += 1
    cw, ch = struct.unpack_from("<ii", d, p); p += 8
    assert cw == ch == fmt["chunk"], (path, cw, ch, version)
    if version == 0:
        lv, = struct.unpack_from("<i", d, p); p += 4
        minl, maxl = 0, lv  # max exclusive
    else:
        minl, maxl = struct.unpack_from("<ii", d, p); p += 8
        maxl += 1
    rooms = []
    nr, = struct.unpack_from("<i", d, p); p += 4
    for _ in range(nr):
        name, p = _line(d, p)
        level, nrect = struct.unpack_from("<ii", d, p); p += 8
        rects = [struct.unpack_from("<iiii", d, p + 16 * i) for i in range(nrect)]
        p += 16 * nrect
        nobj, = struct.unpack_from("<i", d, p); p += 4
        objs = [struct.unpack_from("<iii", d, p + 12 * i) for i in range(nobj)]
        p += 12 * nobj
        rooms.append(dict(name=name, level=level, rects=rects, objects=objs))
    blds = []
    nb, = struct.unpack_from("<i", d, p); p += 4
    for _ in range(nb):
        k, = struct.unpack_from("<i", d, p); p += 4
        blds.append(list(struct.unpack_from("<%di" % k, d, p))); p += 4 * k
    zn = fmt["chunks"]
    zpop = np.frombuffer(d, dtype=np.uint8, count=zn * zn, offset=p).reshape(zn, zn).copy()
    p += zn * zn
    return dict(path=path, version=version, build=fmt["build"], chunks=fmt["chunks"], chunk=cw,
                cell=fmt["chunks"] * cw, minlevel=minl, maxlevel=maxl, tiles=tiles, rooms=rooms,
                buildings=blds, zpop=zpop, trailing=len(d) - p)


def read_pack(path, header, want_tiles=True):
    """-> dict of numpy arrays, one entry per square holding tiles:
    x, y (cell-local squares), z (level), room, start/count into `tiles` (tile-name indices)."""
    with open(path, "rb") as f:
        d = f.read()
    p = 0
    if d[:4] == b"LOTP":
        v, = struct.unpack_from("<i", d, 4)
        assert v == header["version"], (path, v, header["version"])
        p = 8
    else:
        assert header["version"] == 0, path
    count_field, = struct.unpack_from("<i", d, p); p += 4
    bpc, bs = header["chunks"], header["chunk"]
    # the game never reads this field (IsoLot.java:143 seeks straight to the table); vanilla B42 writes 1024, the game's own
    # converter zombie.pot.POT writes 8 with a full 1024-entry table. The table is always chunks x chunks entries.
    nchunks = bpc * bpc
    assert count_field in (nchunks, 8), (path, count_field)
    offs = np.frombuffer(d, dtype="<i8", count=nchunks, offset=p)
    a = memoryview(d).cast("i") if len(d) % 4 == 0 else memoryview(d[: len(d) - len(d) % 4]).cast("i")
    minl, maxl = header["minlevel"], header["maxlevel"]
    per = (maxl - minl) * bs * bs
    xs, ys, zs, rooms, starts, counts, tiles = (array("i") for _ in range(7))
    for ci in range(nchunks):
        off = int(offs[ci])
        assert off % 4 == 0, (path, ci, off)
        q = off // 4
        bx, by = divmod(ci, bpc)
        s = 0
        while s < per:
            c = a[q]; q += 1
            if c == -1:
                k = a[q]; q += 1
                s += k if k > 0 else 1
                continue
            if c > 1:
                z, r = divmod(s, bs * bs)
                lx, ly = divmod(r, bs)
                xs.append(bx * bs + lx); ys.append(by * bs + ly); zs.append(z + minl)
                rooms.append(a[q])
                starts.append(len(tiles)); counts.append(c - 1)
                if want_tiles:
                    tiles.frombytes(d[(q + 1) * 4:(q + c) * 4])
                q += c
            s += 1
    i32 = lambda v: np.frombuffer(v, dtype=np.int32).copy() if len(v) else np.zeros(0, np.int32)
    return dict(x=i32(xs), y=i32(ys), z=i32(zs), room=i32(rooms), start=i32(starts), count=i32(counts),
                tiles=i32(tiles))


def read_chunkdata(path, version):
    """chunkdata_X_Y.bin -> uint8 array [x][y] of the cell's squares (one level): collision bits SOLID 1, WALLN 2, WALLW 4,
    WATER 8, ROOM 16. Format from the game's own converter (zombie.pot.POTChunkData load/save, javap): big-endian short 1,
    then per chunk (cy outer, cx inner) a type byte 0 empty / 1 solid / 3 water / 4 room (every square that value: 0, 1, 8,
    16) or 2 = regular, followed by chunk*chunk bytes in index lx + ly*chunk."""
    fmt = FORMATS[version]
    cpc, dim = fmt["chunks"], fmt["chunk"]
    with open(path, "rb") as f:
        d = f.read()
    v, = struct.unpack_from(">h", d, 0)
    assert v == 1, (path, v)
    out = np.zeros((cpc * dim, cpc * dim), dtype=np.uint8)
    p = 2
    # type 5: no payload, meaning UNKNOWN. POT (this build's own converter) never writes it; vanilla Muldraugh files do not use it;
    # newer author exports do (Fort Boonesborough's B42 chunkdata_54_7: 673 of 1024 chunks). Read as 255 = "unknown", never written.
    fill = {0: 0, 1: 1, 3: 8, 4: 16, 5: 255}
    for cy in range(cpc):
        for cx in range(cpc):
            t = d[p]; p += 1
            if t == 2:
                bits = np.frombuffer(d, dtype=np.uint8, count=dim * dim, offset=p).reshape(dim, dim)  # [ly][lx]
                p += dim * dim
                out[cx * dim:(cx + 1) * dim, cy * dim:(cy + 1) * dim] = bits.T
            else:
                out[cx * dim:(cx + 1) * dim, cy * dim:(cy + 1) * dim] = fill[t]
    assert p == len(d), (path, p, len(d))
    return out


def map_cells(folder):
    """-> {(cx, cy): (lotheader path, lotpack path or None)} for one map folder."""
    out = {}
    for fn in os.listdir(folder):
        m = HEADER_RE.match(fn)
        if not m:
            continue
        cx, cy = int(m.group(1)), int(m.group(2))
        lp = os.path.join(folder, "world_%d_%d.lotpack" % (cx, cy))
        out[(cx, cy)] = (os.path.join(folder, fn), lp if os.path.exists(lp) else None)
    return out


def folder_version(folder):
    """0 (B41), 1 (B42) or None from the first lotheader in the folder."""
    for cell, (h, _) in sorted(map_cells(folder).items()):
        return read_header(h)["version"]
    return None
