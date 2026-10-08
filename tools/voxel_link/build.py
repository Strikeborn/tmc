#!/usr/bin/env python3
"""Voxel Link prototype: carve 3D models from Link's own sprite frames.

Input: folders written by the game's frame recorder (TMC_LINK_FRAMES=<dir>,
see port/port_voxel.cpp RecordLinkFrame): link_NNNN.png on a 64x64 canvas
with his feet at (32, 52), plus frames.jsonl with each frame's animation.

Base model: the standing pose, from its front, back and side pictures, around
one straight body axis. Every front/back pixel gets a depth between the side
profile's back and front edge on its row, rounded off toward the outline (a
pillow, not a slab) and toward each body part's own edges (hair, skin,
cloth, boots, metal), so parts read as separate shapes with a crease between
them. The side profile trims the result (nose, hair spikes).

Animated poses (walk, sword): each frame is carved from the one picture of
its own direction, with the base model's depth profile and the base colours
for the sides that picture can't see. Nothing mixes directions, so a swing
keeps its sword where that direction drew it. Metal (the blade) becomes a
thin plate at the hand instead of a pillow.

The sprites' black outline is recoloured from inside so 3D lighting does the
shading. Hand edits from the viewer (edits.json) are applied last.

Output: a self-contained HTML viewer (three.js from cdnjs). The models are
derived from the game's art, so keep the output local; don't commit or
publish it.

Usage:
  python tools/voxel_link/build.py <frames dir>... -o <out.html> [--edits edits.json] [--scale N]
"""
from __future__ import annotations

import argparse
import base64
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image

FOOT_X, FOOT_Y, SIZE = 32, 52, 64
ROUND = 4.0        # sprite px from the outline over which the body rounds to full depth
PART_ROUND = 2.0   # sprite px from a part's edge over which it rounds (the crease)
PART_DEPTH = 0.2   # how deep the crease between parts goes (share of depth)
DEPTH = 0.85       # the side profile is drawn for a top-down camera; slim it a little
TRIM_SLACK = 2     # sprite px a pose may reach past the standing side profile
# Link's sprites: sprite 1 wears Ezlo, sprite 4 is capless (before Ezlo).
# Animations: base + direction, direction 0 up 1 right 2 down 3 left.
POSES = {1: {"walk": 4, "sword": 8}, 4: {"walk": 4}}
IDLE = 0
SPRITE_NAMES = {1: "cap", 4: "no cap"}
UP, RIGHT, DOWN = 0, 1, 2
ARROW = {DOWN: "down (front)", UP: "up (back)", RIGHT: "right (side)"}
PARTS = ["other", "hair", "skin", "cloth", "boots", "metal", "eye", "dark"]
METAL = PARTS.index("metal")


def part_of(rgb) -> int:
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.2:
        return PARTS.index("dark")
    if s < 0.2 and v > 0.7:
        return METAL                     # sword blade, whites
    if 0.5 < h < 0.75 and s > 0.4:
        return PARTS.index("eye")        # blue
    if (h < 0.03 or h > 0.95) and s > 0.5:
        return PARTS.index("boots")      # red
    if 0.2 < h < 0.45:
        return PARTS.index("cloth")      # greens: cap, tunic
    if 0.11 < h < 0.2 and s > 0.5:
        return PARTS.index("hair")       # yellow
    if h < 0.11 and s < 0.5:
        return PARTS.index("skin")       # peach
    if h < 0.11:
        return PARTS.index("hair")       # browns: hair shade, belt
    return PARTS.index("other")


def parts_of(img: np.ndarray) -> np.ndarray:
    out = np.zeros(img.shape[:2], int)
    for y, x in zip(*np.nonzero(img[..., 3] > 0)):
        out[y, x] = part_of(img[y, x, :3])
    return out


def deoutline(img: np.ndarray) -> np.ndarray:
    """Recolour the sprite's dark outline with the colour just inside it; the
    pixels stay part of the shape. Dark pixels away from the edge (pupils) keep
    their colour."""
    out = img.copy()
    a = img[..., 3] > 0
    dark = a & (img[..., :3].max(axis=2) < 48)
    edge = np.zeros_like(a)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= ~np.roll(np.roll(a, dy, 0), dx, 1)
    todo = dark & edge
    for _ in range(4):
        if not todo.any():
            break
        done = a & ~todo
        nxt = todo.copy()
        for y, x in zip(*np.nonzero(todo)):
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < SIZE and 0 <= xx < SIZE and done[yy, xx] and not dark[yy, xx]:
                    out[y, x, :3] = out[yy, xx, :3]
                    nxt[y, x] = False
                    break
        dark &= nxt | ~todo
        todo = nxt
    return out


def distance_inside(mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance (sprite px) from each set pixel to the
    nearest unset one (two-pass chamfer); 0 outside."""
    h, w = mask.shape
    d = np.where(mask, 1e6, 0.0)
    for y in range(h):
        for x in range(w):
            if d[y, x]:
                d[y, x] = min(d[y, x], (d[y - 1, x] + 1) if y else 1, (d[y, x - 1] + 1) if x else 1,
                              (d[y - 1, x - 1] + 1.4) if x and y else 1.4,
                              (d[y - 1, x + 1] + 1.4) if y and x + 1 < w else 1.4)
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if d[y, x]:
                d[y, x] = min(d[y, x], (d[y + 1, x] + 1) if y + 1 < h else 1,
                              (d[y, x + 1] + 1) if x + 1 < w else 1,
                              (d[y + 1, x + 1] + 1.4) if x + 1 < w and y + 1 < h else 1.4,
                              (d[y + 1, x - 1] + 1.4) if y + 1 < h and x else 1.4)
    return d


def smooth_up(a: np.ndarray, scale: int) -> np.ndarray:
    """Sprite-resolution field to voxel resolution, bilinear (smooth rounding)."""
    im = Image.fromarray(a.astype(np.float32))
    return np.asarray(im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.BILINEAR))


def up(a: np.ndarray, scale: int) -> np.ndarray:
    return a.repeat(scale, 0).repeat(scale, 1)


def shape_field(img: np.ndarray, scale: int) -> np.ndarray:
    """0..1 per voxel column: how far toward full depth (pillow + part creases)."""
    sil = img[..., 3] > 0
    part = parts_of(img)
    body = np.sqrt(np.clip(distance_inside(sil) / ROUND, 0, 1))
    crease = np.ones(sil.shape)
    for p in range(len(PARTS)):
        m = sil & (part == p)
        if m.any() and PARTS[p] not in ("eye", "dark"):
            crease[m] = np.sqrt(np.clip(distance_inside(m)[m] / PART_ROUND, 0, 1))
    f = smooth_up(body * (1 - PART_DEPTH + PART_DEPTH * crease), scale)
    return np.where(up(sil, scale), f, 0)


class Profile:
    """The standing pose's side profile: per voxel row, back and front edge."""

    def __init__(self, side: np.ndarray, front: np.ndarray, scale: int):
        n = SIZE * scale
        sa = up(side[..., 3] > 0, scale)
        fa = up(front[..., 3] > 0, scale)
        self.zb, self.zf = np.full(n, np.nan), np.full(n, np.nan)
        self.xl, self.xr = np.full(n, np.nan), np.full(n, np.nan)
        for y in range(n):
            z = np.nonzero(sa[y])[0]
            if len(z):
                self.zb[y], self.zf[y] = z.min(), z.max() + 1
            x = np.nonzero(fa[y])[0]
            if len(x):
                self.xl[y], self.xr[y] = x.min(), x.max() + 1
        for a in (self.zb, self.zf, self.xl, self.xr):
            rows = np.nonzero(~np.isnan(a))[0]
            a[:] = np.interp(np.arange(n), rows, a[rows])
        mid = (self.zb + self.zf) / 2
        self.zc = float(np.median(mid))  # one straight body axis
        self.xc = float(FOOT_X * scale)
        # where the standing pose reaches, with slack for limbs in motion
        slack = TRIM_SLACK * scale
        self.reach = np.zeros_like(sa)
        for d in range(-slack, slack + 1):
            self.reach |= np.roll(sa, d, axis=1)


def emit(occ: np.ndarray, color, scale: int):
    """Surface voxels of occ[y, x, z] -> packed (x, y, z, r, g, b) rows."""
    n = occ.shape[0]
    inner = occ.copy()
    for ax in range(3):
        for d in (1, -1):
            inner &= np.roll(occ, d, axis=ax)
    inner[[0, -1], :, :] = False
    inner[:, [0, -1], :] = False
    inner[:, :, [0, -1]] = False
    ys, xs, zs = np.nonzero(occ & ~inner)
    rgb = color(ys, xs, zs)
    out = np.empty((len(ys), 6), np.int16)
    out[:, 0] = xs - FOOT_X * scale
    out[:, 1] = FOOT_Y * scale - ys
    out[:, 2] = zs - int(round(FOOT_X * scale))
    out[:, 3:] = rgb
    return out


def neighbourhood_normals(occ: np.ndarray, ys, xs, zs, r: int):
    """Which way each surface voxel faces, from where its neighbourhood is empty."""
    n = occ.shape[0]
    cx = np.cumsum(np.pad(occ, ((0, 0), (1, 0), (0, 0))), axis=1, dtype=np.int32)
    cz = np.cumsum(np.pad(occ, ((0, 0), (0, 0), (1, 0))), axis=2, dtype=np.int32)
    lo_x, hi_x = np.clip(xs - r, 0, n), np.clip(xs + r + 1, 0, n)
    lo_z, hi_z = np.clip(zs - r, 0, n), np.clip(zs + r + 1, 0, n)
    nx = (cx[ys, xs, zs] - cx[ys, lo_x, zs]) - (cx[ys, hi_x, zs] - cx[ys, xs + 1, zs])  # >0: open toward +x
    nz = (cz[ys, xs, zs] - cz[ys, xs, lo_z]) - (cz[ys, xs, hi_z] - cz[ys, xs, zs + 1])  # >0: open toward +z
    return nx, nz


def carve_base(front, back, side, scale: int, prof: Profile):
    """Standing pose from all three pictures, around the profile's straight axis."""
    n = SIZE * scale
    back = back[:, ::-1]
    F, B, S = up(front, scale), up(back, scale), up(side, scale)
    fa, ba, sa = F[..., 3] > 0, B[..., 3] > 0, S[..., 3] > 0
    shape = np.maximum(shape_field(front, scale), shape_field(back, scale))
    zi = np.arange(n)
    lo = prof.zc - (prof.zc - prof.zb)[:, None] * shape * DEPTH
    hi = prof.zc + (prof.zf - prof.zc)[:, None] * shape * DEPTH
    lo, hi = np.minimum(lo, prof.zc - scale / 2), np.maximum(hi, prof.zc + scale / 2)  # >= 1 sprite px thick
    occ = (zi[None, None, :] >= np.floor(lo)[..., None]) & (zi[None, None, :] < np.ceil(hi)[..., None])
    occ &= (fa | ba)[..., None]
    trimmed = occ & sa[:, None, :]
    keep = trimmed.sum(axis=2) >= scale  # never trim a column away (thin hair tips)
    occ = np.where(keep[..., None], trimmed, occ)

    def color(ys, xs, zs):
        nx, nz = neighbourhood_normals(occ, ys, xs, zs, scale)
        flank = (np.abs(nx) > np.abs(nz) + 1) & sa[ys, zs]
        front_side = zs >= prof.zc
        c = np.where(front_side[:, None],
                     np.where(fa[ys, xs][:, None], F[ys, xs, :3], B[ys, xs, :3]),
                     np.where(ba[ys, xs][:, None], B[ys, xs, :3], F[ys, xs, :3]))
        return np.where(flank[:, None], S[ys, zs, :3], c)

    return emit(occ, color, scale)


HAND_ROW = FOOT_Y - 10  # sprite row of the sword hand when standing


def blade_voxels(img, direction: int, scale: int, prof: Profile) -> np.ndarray:
    """The sword, undoing the GBA's top-down view: on screen, 'lower' means
    'nearer the camera', so blade pixels below the hand lie flat at hand height
    reaching toward the camera, and those above it reach away; pixels level
    with or above the hand in a raised swing stay upright at the hand plane.
    Returns packed (x, y, z, r, g, b) rows, a plate one sprite pixel thick."""
    part = parts_of(img)
    rows = []
    for y, x in zip(*np.nonzero(part == METAL)):
        c = img[y, x, :3]
        reach = (y - HAND_ROW) if direction == DOWN else (HAND_ROW - y) if direction == UP else (y - HAND_ROW)
        if direction in (DOWN, UP):
            mx = (SIZE - 1 - x) if direction == UP else x
            if reach > 0:  # flat at hand height, out in front (Link faces +z in both views)
                py, pz = HAND_ROW, prof.zc / scale + 2 + reach
            else:
                py, pz = y, prof.zc / scale + 2
            for dy in range(scale):
                for dx in range(scale):
                    for dz in range(max(1, scale // 2)):
                        rows.append((mx * scale + dx, py * scale + dy, int(pz * scale) + dz, *c))
        else:
            # side picture: image x is forward (z); below the hand reaches toward the camera (-x)
            if reach > 0:
                py, px = HAND_ROW, prof.xc / scale - 2 - reach
            else:
                py, px = y, prof.xc / scale - 2
            for dy in range(scale):
                for dz in range(scale):
                    for dx in range(max(1, scale // 2)):
                        rows.append((int(px * scale) + dx, py * scale + dy, x * scale + dz, *c))
    if not rows:
        return np.zeros((0, 6), np.int16)
    a = np.array(rows, np.int64)
    out = np.empty_like(a)
    out[:, 0] = a[:, 0] - FOOT_X * scale
    out[:, 1] = FOOT_Y * scale - a[:, 1]
    out[:, 2] = a[:, 2] - FOOT_X * scale
    out[:, 3:] = a[:, 3:]
    return out.astype(np.int16)


def carve_pose(img, direction: int, scale: int, prof: Profile, base: dict):
    """One frame from its own picture; the base pictures colour what it can't see.
    The blade is built separately (blade_voxels)."""
    src = img
    n = SIZE * scale
    zi = np.arange(n)
    I = up(img, scale)
    ia = I[..., 3] > 0
    part = up(parts_of(img), scale)
    metal = part == METAL
    if direction in (DOWN, UP):
        if direction == UP:
            img, I, ia, part, metal = img[:, ::-1], I[:, ::-1], ia[:, ::-1], part[:, ::-1], metal[:, ::-1]
        shape = shape_field(img, scale)
        lo = prof.zc - (prof.zc - prof.zb)[:, None] * shape * DEPTH
        hi = prof.zc + (prof.zf - prof.zc)[:, None] * shape * DEPTH
        lo, hi = np.minimum(lo, prof.zc - scale / 2), np.maximum(hi, prof.zc + scale / 2)
        occ = (zi[None, None, :] >= np.floor(lo)[..., None]) & (zi[None, None, :] < np.ceil(hi)[..., None])
        occ &= (ia & ~metal)[..., None]
        trimmed = occ & prof.reach[:, None, :]
        keep = trimmed.sum(axis=2) >= scale
        occ = np.where(keep[..., None], trimmed, occ)
        seen, other = I, base["front"] if direction == UP else base["back"]
        seen_is_front = direction == DOWN

        def color(ys, xs, zs):
            nx, nz = neighbourhood_normals(occ, ys, xs, zs, scale)
            flank = (np.abs(nx) > np.abs(nz) + 1) & (base["side"][ys, zs, 3] > 0) & ~metal[ys, xs]
            toward = (zs >= prof.zc) == seen_is_front  # facing the picture's camera
            oa = other[ys, xs, 3] > 0
            c = np.where((toward | ~oa)[:, None], seen[ys, xs, :3], other[ys, xs, :3])
            return np.where(flank[:, None], base["side"][ys, zs, :3], c)
    else:
        # side picture: image x is model z; width comes from the base front outline
        shape = shape_field(img, scale)  # [y, z]
        half_l = (prof.xc - prof.xl)[:, None] * shape * DEPTH
        half_r = (prof.xr - prof.xc)[:, None] * shape * DEPTH
        half_l, half_r = np.maximum(half_l, scale / 2), np.maximum(half_r, scale / 2)
        lo, hi = prof.xc - half_l, prof.xc + half_r
        xi = zi
        occ = (xi[None, :, None] >= np.floor(lo)[:, None, :]) & (xi[None, :, None] < np.ceil(hi)[:, None, :])
        occ &= (ia & ~metal)[:, None, :]  # [y, x, z]

        def color(ys, xs, zs):
            nx, nz = neighbourhood_normals(occ, ys, xs, zs, scale)
            facing_z = (np.abs(nz) > np.abs(nx) + 1) & ~metal[ys, zs]
            fa = base["front"][ys, xs, 3] > 0
            ba = base["back"][ys, xs, 3] > 0
            c = I[ys, zs, :3]
            c = np.where((facing_z & (nz > 0) & fa)[:, None], base["front"][ys, xs, :3], c)
            c = np.where((facing_z & (nz < 0) & ba)[:, None], base["back"][ys, xs, :3], c)
            return c

    vox = np.vstack([emit(occ, color, scale), blade_voxels(src, direction, scale, prof)])
    # Art below the feet is nearer the camera, not underground (the GBA's
    # top-down view): fold it forward along the ground, toward that camera.
    below = vox[:, 1] < 0
    depth = -vox[below, 1]
    if direction == DOWN:
        vox[below, 2] += depth
    elif direction == UP:
        vox[below, 2] -= depth
    else:
        vox[below, 0] -= depth
    vox[below, 1] = 0
    return vox


def load(frames_dirs: list[Path]):
    rows = [(d, json.loads(l)) for d in frames_dirs
            for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]
    pics = {}
    for frames_dir, r in rows:
        if r["sprite"] not in POSES:
            continue
        img = np.asarray(Image.open(frames_dir / r["file"]).convert("RGBA"))
        key = (r["sprite"], r["anim"], max(r["step"], 0))
        # the most complete capture of each step (one taken while OAM was
        # still being built can miss pieces)
        if key not in pics or (img[..., 3] > 0).sum() > (pics[key][..., 3] > 0).sum():
            pics[key] = img
    return {k: deoutline(v) for k, v in pics.items()}


def pack(vox: np.ndarray) -> str:
    """int16 x,y,z then uint8 r,g,b per voxel, base64 (keeps the page small)."""
    xyz = vox[:, :3].astype("<i2").tobytes()
    rgb = vox[:, 3:].astype(np.uint8).tobytes()
    return base64.b64encode(len(vox).to_bytes(4, "little") + xyz + rgb).decode()


def apply_edits(models: dict, edits: dict) -> None:
    """edits: {model: {step: {"del": [[x,y,z],...], "set": [[x,y,z,r,g,b],...]}}} from the viewer."""
    for name, steps in edits.items():
        seq = models.get(name)
        if not seq:
            continue
        for step, e in steps.items():
            s = int(step)
            if s >= len(seq):
                continue
            v = seq[s]
            keys = {tuple(p): i for i, p in enumerate(v[:, :3].tolist())}
            gone = [keys[tuple(p[:3])] for p in e.get("del", []) if tuple(p[:3]) in keys]
            keep = np.ones(len(v), bool)
            keep[gone] = False
            added = []
            for p in e.get("set", []):
                k = tuple(p[:3])
                if k in keys:
                    v[keys[k], 3:] = p[3:6]
                else:
                    added.append(p[:6])
            seq[s] = np.vstack([v[keep]] + ([np.array(added, np.int16)] if added else []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, nargs="+", help="recorder folders (with and without the cap)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--edits", type=Path, help="hand edits saved from the viewer")
    ap.add_argument("--scale", type=int, default=4, help="voxels per sprite pixel")
    a = ap.parse_args()
    scale = a.scale
    pics = load(a.frames)
    models = {}
    for sprite, poses in POSES.items():
        idle = {d: pics.get((sprite, IDLE + d, 0)) for d in (UP, RIGHT, DOWN)}
        if any(v is None for v in idle.values()):
            continue
        prof = Profile(idle[RIGHT], idle[DOWN], scale)
        base = {"front": up(idle[DOWN], scale), "back": up(idle[UP][:, ::-1], scale), "side": up(idle[RIGHT], scale)}
        name = SPRITE_NAMES[sprite]
        models[f"{name}: standing"] = [carve_base(idle[DOWN], idle[UP], idle[RIGHT], scale, prof)]
        for pose, anim in poses.items():
            for d in (DOWN, RIGHT, UP):
                steps = sorted(s for (sp, an, s) in pics if sp == sprite and an == anim + d)
                seq = [carve_pose(pics[(sprite, anim + d, s)], d, scale, prof, base) for s in steps]
                if seq:
                    models[f"{name}: {pose}, {ARROW[d]}"] = seq
        print(f"{name}: {len([k for k in models if k.startswith(name)])} models")
    if a.edits and a.edits.exists():
        apply_edits(models, json.loads(a.edits.read_text()))
    data = {k: [pack(v) for v in seq] for k, seq in models.items()}
    print(f"{sum(len(v) for seq in models.values() for v in seq) // max(1, sum(map(len, models.values())))} "
          "surface voxels per model avg")
    tpl = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")
    page = tpl.replace("/*MODELS*/null", json.dumps(data, separators=(",", ":")))
    page = page.replace("/*SCALE*/1", str(scale))
    a.out.write_text(page, encoding="utf-8")
    print(f"-> {a.out} ({a.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
