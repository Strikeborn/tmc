#!/usr/bin/env python3
"""Voxel Link prototype: carve 3D models from Link's own sprite frames.

Input: a folder written by the game's frame recorder (TMC_LINK_FRAMES=<dir>,
see port/port_voxel.cpp RecordLinkFrame): link_NNNN.png on a 64x64 canvas
with his feet at (32, 52), plus frames.jsonl with each frame's animation.

A pose is an animation step that exists facing down, up and right. Each one
is carved at SCALE voxels per sprite pixel:
  - every front/back pixel gets a depth between the side profile's front and
    back edge on its row, rounded off toward the outline (a pillow, not a slab)
  - each body part (by colour: hair, skin, cloth, boots, metal, eyes) also
    rounds toward its own edges, so parts read as separate shapes with a
    crease between them; nothing gets thinner than MIN_HALF voxels, so ears
    and hair tips survive
  - the side profile trims the result (nose, hair spikes, sword)
  - eyes sit a voxel into the face
The front/back pictures paint the faces they look at, the side picture the
flanks. The sprites' black outline is recoloured from inside so 3D lighting
does the shading. Hand edits from the viewer (edits.json) are applied last.

Output: a self-contained HTML viewer (three.js from cdnjs) with the models.
The models are derived from the game's art, so keep the output local; don't
commit or publish it.

Usage:
  python tools/voxel_link/build.py <frames dir>... -o <out.html> [--edits edits.json]
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image

FOOT_X, FOOT_Y, SIZE = 32, 52, 64
SCALE = 2          # voxels per sprite pixel
ROUND = 4.0        # sprite px from the outline over which the body rounds to full depth
PART_ROUND = 2.0   # sprite px from a part's edge over which it rounds (the crease)
PART_DEPTH = 0.2   # how deep the crease between parts goes (share of depth)
DEPTH = 0.85       # the side profile is drawn for a top-down camera; slim it a little
MIN_HALF = 1       # voxels: thinnest half-thickness anywhere
# Link's sprites: sprite 1 wears Ezlo, sprite 4 is capless (before Ezlo).
# Animations: base + direction, direction 0 up 1 right 2 down 3 left.
POSES = {1: {"idle": 0, "walk": 4, "sword": 8}, 4: {"idle": 0, "walk": 4}}
SPRITE_NAMES = {1: "cap", 4: "no cap"}
UP, RIGHT, DOWN = 0, 1, 2

PARTS = ["other", "hair", "skin", "cloth", "boots", "metal", "eye", "dark"]


def part_of(rgb) -> int:
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.2:
        return PARTS.index("dark")
    if s < 0.2 and v > 0.7:
        return PARTS.index("metal")      # sword blade, whites
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


def deoutline(img: np.ndarray) -> np.ndarray:
    """Recolour the sprite's dark outline (the pixel-art stand-in for shading)
    with the colour just inside it; the pixels stay part of the shape. Dark
    pixels away from the edge (pupils) keep their colour."""
    out = img.copy()
    a = img[..., 3] > 0
    dark = a & (img[..., :3].max(axis=2) < 48)
    edge = np.zeros_like(a)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= ~np.roll(np.roll(a, dy, 0), dx, 1)
    todo = dark & edge
    for _ in range(4):  # outlines are 1-2 px; grow inward colours outward
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
    """Approximate Euclidean distance from each set pixel to the nearest unset one
    (two-pass chamfer, 1 / 1.4 weights); 0 outside."""
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


def load(frames_dirs: list[Path]):
    rows = [(d, json.loads(l)) for d in frames_dirs
            for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]
    pics = {}
    for frames_dir, r in rows:
        if r["sprite"] not in POSES:
            continue
        img = np.asarray(Image.open(frames_dir / r["file"]).convert("RGBA"))
        key = (r["sprite"], r["anim"], max(r["step"], 0))
        # The most complete capture of each step: one taken while the game was
        # still building OAM can miss pieces.
        if key not in pics or (img[..., 3] > 0).sum() > (pics[key][..., 3] > 0).sum():
            pics[key] = img
    return pics


def up(img: np.ndarray) -> np.ndarray:
    return img.repeat(SCALE, 0).repeat(SCALE, 1)


def carve(front: np.ndarray, back: np.ndarray, side: np.ndarray):
    """front: seen from +z (Link faces +z). back: from -z. side: from -x, image x = world +z.
    Images are already SCALEd. Returns [(x, y, z, r, g, b, part)] in voxels, x/z centred
    on the feet, y up from the ground."""
    N = SIZE * SCALE
    back = back[:, ::-1]
    fa, ba, sa = front[..., 3] > 0, back[..., 3] > 0, side[..., 3] > 0
    if not (fa.any() and sa.any()):
        return []
    # side profile per row: back (min z) and front (max z) edges
    zb = np.full(N, np.nan)
    zf = np.full(N, np.nan)
    for y in range(N):
        zs = np.nonzero(sa[y])[0]
        if len(zs):
            zb[y], zf[y] = zs.min(), zs.max() + 1
    rows = np.nonzero(~np.isnan(zf))[0]
    zb = np.interp(np.arange(N), rows, zb[rows])
    zf = np.interp(np.arange(N), rows, zf[rows])
    zc = (zf + zb) / 2
    half_f = (zf - zc) * DEPTH
    half_b = (zc - zb) * DEPTH

    sil = fa | ba
    rgb = np.where(fa[..., None], front[..., :3], back[..., :3])
    part = np.zeros(sil.shape, int)
    for y, x in zip(*np.nonzero(sil)):
        part[y, x] = part_of(rgb[y, x])
    body = np.sqrt(np.clip(distance_inside(sil) / (ROUND * SCALE), 0, 1))
    crease = np.ones(sil.shape)
    for p in range(len(PARTS)):
        m = sil & (part == p)
        if m.any() and PARTS[p] not in ("eye", "dark"):  # eyes and pupils sit in the face
            crease[m] = np.sqrt(np.clip(distance_inside(m)[m] / (PART_ROUND * SCALE), 0, 1))
    shape = body * (1 - PART_DEPTH + PART_DEPTH * crease)
    eye = sil & ((part == PARTS.index("eye")) | (part == PARTS.index("dark")))

    occ = np.zeros((N, N, N), bool)  # [y, x, z]
    zi = np.arange(N)
    for y in range(N):
        for x in np.nonzero(sil[y])[0]:
            r = shape[y, x]
            lo = zc[y] - max(half_b[y] * r, MIN_HALF)
            hi = zc[y] + max(half_f[y] * r, MIN_HALF)
            if eye[y, x]:
                hi -= 1  # set into the face
            col = (zi >= np.floor(lo)) & (zi < np.ceil(hi))
            trimmed = col & sa[y]
            occ[y, x] = trimmed if trimmed.sum() >= 2 * MIN_HALF else col  # keep thin tips

    vox = []
    ys, xs, zs = np.nonzero(occ)
    for y, x, z in zip(ys, xs, zs):
        exposed = (z + 1 >= N or not occ[y, x, z + 1]) or (z == 0 or not occ[y, x, z - 1]) or \
                  (x == 0 or not occ[y, x - 1, z]) or (x + 1 >= N or not occ[y, x + 1, z]) or \
                  (y == 0 or not occ[y - 1, x, z]) or (y + 1 >= N or not occ[y + 1, x, z])
        if not exposed:
            continue
        x0, x1, z0, z1 = max(x - 3, 0), min(x + 4, N), max(z - 3, 0), min(z + 4, N)
        nx = occ[y, x0:x, z].sum() - occ[y, x + 1:x1, z].sum()  # >0: open toward +x
        nz = occ[y, x, z0:z].sum() - occ[y, x, z + 1:z1].sum()  # >0: open toward +z (front)
        if abs(nx) > abs(nz) + 1 and side[y, z, 3]:
            c = side[y, z, :3]
        elif z >= zc[y]:
            c = front[y, x, :3] if fa[y, x] else back[y, x, :3]
        else:
            c = back[y, x, :3] if ba[y, x] else front[y, x, :3]
        vox.append((int(x) - FOOT_X * SCALE, FOOT_Y * SCALE - int(y), int(z) - FOOT_X * SCALE,
                    *(int(v) for v in c), int(part[y, x])))
    return vox


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
            gone = {tuple(p[:3]) for p in e.get("del", [])}
            setc = {tuple(p[:3]): p[3:6] for p in e.get("set", [])}
            out = []
            for v in seq[s]:
                k = tuple(v[:3])
                if k in gone:
                    continue
                if k in setc:
                    v = (*k, *setc.pop(k), v[6])
                out.append(v)
            out += [(*k, *c, 0) for k, c in setc.items()]  # added voxels
            seq[s] = out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, nargs="+", help="recorder folders (with and without the cap)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--edits", type=Path, help="hand edits saved from the viewer")
    a = ap.parse_args()
    pics = {k: up(deoutline(v)) for k, v in load(a.frames).items()}
    models = {}
    for sprite, poses in POSES.items():
        for pose, base in poses.items():
            steps = sorted({s for (sp, anim, s) in pics if sp == sprite and anim == base + DOWN})
            seq = []
            for s in steps:
                f = pics.get((sprite, base + DOWN, s))
                b = pics.get((sprite, base + UP, s))
                r = pics.get((sprite, base + RIGHT, s))
                if f is None or b is None or r is None:
                    continue
                seq.append(carve(f, b, r))
            if seq:
                name = f"{SPRITE_NAMES[sprite]}: {pose}"
                models[name] = seq
                print(f"{name}: {len(seq)} steps, {sum(map(len, seq)) // len(seq)} surface voxels avg")
    if a.edits and a.edits.exists():
        apply_edits(models, json.loads(a.edits.read_text()))
    tpl = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")
    page = tpl.replace("/*MODELS*/null", json.dumps(models, separators=(",", ":")))
    page = page.replace("/*SCALE*/1", str(SCALE))
    a.out.write_text(page, encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
