# b41tob42

Convert a Project Zomboid **Build 41** map folder into **Build 42** map files, on your own PC.

> **We do not provide, host or endorse converted maps.** This tool ships no map data. A converted map is a port of somebody
> else's work: **do not share or upload one without the original author's permission.** What you do with your own copy is up to
> you; you can copy the converted files into your own local mod by hand (see below).

## What it does

Build 41 maps use 300x300-square cells made of 10x10 chunks; Build 42 uses 256x256 cells made of 8x8 chunks. The world square
coordinates are the same in both, so every square, room and building keeps its place; only the files are cut differently.

`convert` writes, for every Build 42 cell the map covers:

- `N_M.lotheader`: tile names, rooms (name, level, rectangles, room objects), buildings, zombie density (32x32 per cell)
- `world_N_M.lotpack`: the tiles of every square, every level
- `chunkdata_N_M.bin`: the collision bits (solid, walls, water, rooms) when the source map has them

and copies the map's own `map.info`, `objects.lua` and `spawnpoints.lua` as they are. Then it reads every file it wrote back
and compares it with what it meant to write; any difference makes it exit with code 1.

Tile names: a name that exists in your installed Build 42 (its tile definitions, or a tile mod the map requires) is kept. A name
from the shipped table (`tables/b41_to_b42_tiles.tsv`, learned from maps whose authors made their own Build 42 versions) is
replaced. Any other name gets the closest Build 42 tile and is listed in `<name>.unmapped.md`.

## Use

Python 3.10+ and numpy.

```
python cli.py convert "<B41 map folder>" --out "<output folder>" --game "<Project Zomboid install folder>"
python cli.py info "<map folder>"
```

- `<B41 map folder>` is the folder with the `.lotheader` files, e.g. `...\steamapps\workshop\content\108600\<item>\mods\<mod>\media\maps\<map>`.
- `--game` is only read, to learn which tile names Build 42 has. Nothing of the game is copied into the output.
- `--mods` (optional, repeatable): where installed mods are, to find the tile mods the map requires (`require=` in its
  `mod.info`). Default: the Workshop content folder the map is in.
- Output: `<output folder>/<map>/`, `<map>.receipt.json` (what was done, with numbers) and `<map>.unmapped.md`.
- It refuses to write into the game folder or the mods folder.

## Using the result yourself

Make a local Build 42 mod in `Zomboid/mods/<YourMod>/` with `mod.info` (and `42/mod.info`), and copy the converted folder to
`common/media/maps/<map>/`. If the original map requires a tile mod, require it too. Keep the original author's name and credit
in your mod.info, and keep it on your own PC unless the author agrees.

## What it does not do

- The in-game world map (`worldmap.xml` / `.bin`), spawn regions and foraging biome maps are not converted.
- A map's own tile packs made for Build 41 are not converted; whether they load in Build 42 is not known. Missing tiles show as
  gaps. Build 42 versions of shared tile packs (for example Erika's tiles) are used when installed.
- Chunk type 5 (seen in some newer author exports) is read as "unknown" and never written.

## How we checked it

- Calibration: four maps whose authors (or a porter) made Build 42 versions: Taylorsville, Fort Boonesborough, Mount Crow City,
  Raven Creek. On every floor at level 1 and above with at least 1000 squares, 84 to 100% of the converted squares land on an
  occupied square of the author's version; a copy shifted by one chunk drops below 71%.
- The game's own converter: Project Zomboid contains an internal old-to-new map converter. Run on the same four maps (a
  development check with your own installed game, not part of this tool), every square, room, building and chunkdata file it
  writes is identical to ours. It skips partial cells at a map's east and south edge, which we keep, and places zombie density
  with an offset, which we do not.

## Developer tools (not part of this package)

In the Seeded Knox-Reroll repository, `potdiff.py` with `PotRun.java` run the game's internal converter from **your own
installed game** to compare with this tool. They ship no game code, and neither does this package.

## Credits

- Reading the lot formats started from [pzmap2dzi](https://github.com/cff29546/pzmap2dzi) by Min Xiang (MIT). See `NOTICE`.
- Made for Seeded Knox-Reroll, a Project Zomboid mod.

## Licence

MIT. See `LICENSE`. The pzmap2dzi notice for the ported lot-file reading is in `NOTICE`.
