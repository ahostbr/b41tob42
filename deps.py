"""The tile mods a B41 map relies on (its mod.info require=), found among the installed mods. No fixed paths: the caller names where
installed mods live (a Steam Workshop content folder holds <item>/mods/<mod>/mod.info; a plain mods folder holds <mod>/mod.info).
"""
import glob
import os

_INDEX = {}


def mod_index(mod_roots):
    """{mod id: [mod folder]} for every mod.info under the given roots (Workshop content folders or plain mods folders)."""
    key = tuple(sorted(os.path.abspath(r) for r in mod_roots))
    if key not in _INDEX:
        idx = {}
        for r in key:
            for info in glob.glob(os.path.join(r, "*", "mods", "*", "mod.info")) + glob.glob(os.path.join(r, "*", "mod.info")):
                with open(info, encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
                for line in lines:
                    if line.strip().lower().startswith("id="):
                        idx.setdefault(line.split("=", 1)[1].strip(), []).append(os.path.dirname(info))
        _INDEX[key] = idx
    return _INDEX[key]


def mod_root(folder):
    """The mod folder (the one holding mod.info) above a map folder, or None."""
    d = os.path.abspath(folder)
    while d and os.path.dirname(d) != d:
        if os.path.exists(os.path.join(d, "mod.info")) and d != os.path.abspath(folder):
            return d
        d = os.path.dirname(d)
    return None


def workshop_roots(map_folder):
    """Default search roots for a map folder: the Workshop content folder (an ancestor named 108600), else the folder
    two levels above its mod (a plain mods folder)."""
    d = os.path.abspath(map_folder)
    while d and os.path.dirname(d) != d:
        if os.path.basename(d) == "108600":
            return [d]
        d = os.path.dirname(d)
    root = mod_root(map_folder)
    return [os.path.dirname(root)] if root else []


def requires(root):
    """Mod ids from require= in the mod's mod.info files (root, 42/, common/)."""
    ids = []
    for info in (os.path.join(root, "mod.info"), os.path.join(root, "42", "mod.info"), os.path.join(root, "common", "mod.info")):
        if os.path.exists(info):
            with open(info, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            for line in lines:
                if line.strip().lower().startswith("require="):
                    for i in line.split("=", 1)[1].split(","):
                        i = i.strip().lstrip("\\").strip()
                        if i and i not in ids:
                            ids.append(i)
    return ids


def b42_tiles(root):
    """.tiles a mod ships for B42 (its 42/ and common/ trees); [] if it has no B42 tree."""
    return sorted(glob.glob(os.path.join(root, "42", "**", "*.tiles"), recursive=True) +
                  glob.glob(os.path.join(root, "common", "**", "*.tiles"), recursive=True))


def dependency_tiles(b41_folder, mod_roots=None):
    """-> ([(path, label)], notes): tile definitions a B41 map can rely on in B42: its required tile mods' B42 .tiles (label
    require), else their B41 .tiles (label require_b41only: whether they load in B42 is UNPROVEN), and the map's own B41 .tiles
    (label map_own_b41: UNPROVEN)."""
    out, notes = [], []
    root = mod_root(b41_folder)
    if not root:
        return out, ["no mod.info above %s" % b41_folder]
    roots = mod_roots if mod_roots is not None else workshop_roots(b41_folder)
    for mid in requires(root):
        dirs = mod_index(roots).get(mid)
        if not dirs:
            notes.append("require %s: not installed" % mid)
            continue
        t = b42_tiles(dirs[0])
        if t:
            out += [(p, "require") for p in t]
            notes.append("require %s: %d B42 .tiles from %s" % (mid, len(t), dirs[0]))
        else:
            t = sorted(glob.glob(os.path.join(dirs[0], "media", "**", "*.tiles"), recursive=True))
            out += [(p, "require_b41only") for p in t]
            notes.append("require %s: no B42 tree, %d B41 .tiles (B42 load unproven)" % (mid, len(t)))
    own = sorted(glob.glob(os.path.join(root, "media", "**", "*.tiles"), recursive=True))
    out += [(p, "map_own_b41") for p in own]
    if own:
        notes.append("map's own B41 .tiles: %d (B42 load unproven)" % len(own))
    return out, notes
