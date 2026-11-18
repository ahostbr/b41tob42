"""b41tob42: convert a Project Zomboid Build 41 map folder into Build 42 map files, on your own PC.

  python cli.py convert "<B41 map folder>" --out "<folder>" --game "<Project Zomboid install folder>"
                [--mods "<Workshop content folder or mods folder>" ...] [--table tables/b41_to_b42_tiles.tsv] [--name NAME]
  python cli.py info "<map folder>"

convert writes <out>/<name>/: N_M.lotheader, world_N_M.lotpack and chunkdata_N_M.bin for every 256-square Build 42 cell the map
covers, plus the map's own map.info, objects.lua and spawnpoints.lua copied as they are (both use coordinates Build 42 reads the
same way). It then reads every written file back and compares it with what should have been written (exit 1 if anything
differs), and writes <out>/<name>.receipt.json and <out>/<name>.unmapped.md (tile names with no Build 42 match and the closest
match it used).

What it does not do: the world map (worldmap.xml / .bin), spawn regions, foraging biome maps, and the map's own tile packs. A map
whose tile packs were made for Build 41 may show missing tiles in Build 42. See README.md.

--game is only read: the tile names your installed Build 42 defines (media/tile_definitions, vanilla lot headers). Nothing of the
game is copied into the output. The output is yours to use locally; see README.md before you share a converted map.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41  # noqa: E402
import b42write  # noqa: E402
import convlog  # noqa: E402
import deps  # noqa: E402
import regrid  # noqa: E402
import tiles  # noqa: E402

COPIED = ("map.info", "objects.lua", "spawnpoints.lua")


def convert(src, out, game, mod_roots=None, table_tsv=None, name=None):
    if b41.folder_version(src) != 0:
        raise SystemExit("not a Build 41 map folder (no B41 .lotheader files): %s" % src)
    name = name or os.path.basename(os.path.normpath(src))
    dest = os.path.join(out, name)
    for banned in [game] + list(mod_roots or deps.workshop_roots(src)):
        if banned and os.path.normcase(os.path.abspath(dest)).startswith(os.path.normcase(os.path.abspath(banned))):
            raise SystemExit("refusing to write into the game or mods folder: %s" % dest)
    if os.path.exists(dest) and os.listdir(dest):
        raise SystemExit("output folder is not empty: %s" % dest)
    run = convlog.Run("offline", out_dir=os.path.join(out, ".b41tob42"), writes=True)
    t0 = time.time()
    dep_tiles, notes = deps.dependency_tiles(src, mod_roots)
    known = tiles.known_b42(game=game, map_tiles=dep_tiles)
    table = tiles.Table(known, tiles.load_tsv(table_tsv) if table_tsv else None)
    run.map(name, source=src, dependencies=notes, known_tiles=len(known))
    m = regrid.Map(src, run, name)
    cells = regrid.regrid(m, run)
    b42write.write_map(m, cells, table, dest, run)
    for f in COPIED:
        if os.path.isfile(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(dest, f))
    rt = b42write.roundtrip(m, cells, table, dest)
    run.roundtrip(name, **rt)
    names = b42write.map_tiles(m, table)
    counts = {}
    for n in m.vocab:
        how = table.resolve(n)[1] or table.suggest(n)[1]
        counts[how.split(":")[0]] = counts.get(how.split(":")[0], 0) + 1
    unmapped = [(n, table.suggest(n)) for n, b in zip(m.vocab, names) if table.resolve(n)[0] is None and table.resolve(n)[1] != "author_dropped"]
    rec = run.receipt()
    rec["summary"] = dict(source=src, output=dest, b41_cells=len(m.cells), b42_cells=len(cells), squares=int(len(m.x)),
                          buildings=len(m.buildings), rooms=len(m.rooms), tile_names=counts, chunkdata=bool(m.chunkdata),
                          dependencies=notes, roundtrip_ok=rt["ok"], seconds=round(time.time() - t0, 1))
    with open(os.path.join(out, name + ".receipt.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(out, name + ".unmapped.md"), "w", encoding="utf-8") as f:
        f.write("# %s: tile names with no Build 42 match\n\n%d names. Each was replaced by the closest match shown (or left out).\n\n"
                "| Build 41 tile | used instead | how |\n|---|---|---|\n" % (name, len(unmapped)))
        for n, (s, how) in sorted(unmapped):
            f.write("| %s | %s | %s |\n" % (n, s or "(left out)", how))
    run._f.close()
    return rec["summary"]


def info(folder):
    v = b41.folder_version(folder)
    m = regrid.Map(folder, want_tiles=False)
    lv = sorted(set(m.z.tolist()))
    return dict(folder=folder, build={0: "B41", 1: "B42", None: "?"}[v], cells=len(m.cells), squares=int(len(m.x)),
                buildings=len(m.buildings), rooms=len(m.rooms), levels=[lv[0], lv[-1]] if lv else None,
                chunkdata=len(m.chunkdata))


def main(argv):
    ap = argparse.ArgumentParser(prog="b41tob42", description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("source")
    c.add_argument("--out", required=True)
    c.add_argument("--game", required=True, help="your Project Zomboid (Build 42) install folder; only read")
    c.add_argument("--mods", action="append", default=None, help="where installed mods are (default: the Workshop folder above the map)")
    c.add_argument("--table", default=os.path.join(HERE, "tables", "b41_to_b42_tiles.tsv"))
    c.add_argument("--name", default=None)
    i = sub.add_parser("info")
    i.add_argument("folder")
    a = ap.parse_args(argv)
    if a.cmd == "info":
        print(json.dumps(info(a.folder), indent=1))
        return 0
    s = convert(a.source, a.out, a.game, a.mods, a.table if a.table and os.path.isfile(a.table) else None, a.name)
    print(json.dumps(s, indent=1))
    return 0 if s["roundtrip_ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
