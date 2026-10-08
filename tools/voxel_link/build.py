#!/usr/bin/env python3
"""Voxel Link prototype: carve 3D models from Link's own sprite frames.

Input: a folder written by the game's frame recorder (TMC_LINK_FRAMES=<dir>,
see port/port_voxel.cpp RecordLinkFrame): link_NNNN.png on a 64x64 canvas
with his feet at (32, 52), plus frames.jsonl with each frame's animation.

For every pose (an animation step that exists facing down, up and right) the
front, back and side pictures are intersected as silhouettes (a "visual
hull"): a voxel is kept when all three views show Link there. Each surface
voxel takes its colour from the view that sees it first.

Output: a self-contained HTML viewer (three.js from cdnjs) with the models.
The models are derived from the game's art, so keep the output local; don't
commit or publish it.

Usage:
  python tools/voxel_link/build.py <frames dir> -o <out.html>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

FOOT_X, FOOT_Y, SIZE = 32, 52, 64
# Link's animations (sprite 4): base + direction, direction 0 up 1 right 2 down 3 left.
POSES = {"idle": 0, "walk": 4}
UP, RIGHT, DOWN = 0, 1, 2


def deoutline(img: np.ndarray) -> np.ndarray:
    """Recolour the sprite's dark outline (the pixel-art stand-in for shading)
    with the colour just inside it; the pixels stay part of the shape. Dark
    pixels away from the edge (pupils, belt) keep their colour."""
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


def load(frames_dir: Path):
    rows = [json.loads(l) for l in (frames_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    pics = {}
    for r in rows:
        if r["sprite"] != 4:
            continue
        img = np.asarray(Image.open(frames_dir / r["file"]).convert("RGBA"))
        key = (r["anim"], max(r["step"], 0))
        # The most complete capture of each step: one taken while the game was
        # still building OAM can miss pieces.
        if key not in pics or (img[..., 3] > 0).sum() > (pics[key][..., 3] > 0).sum():
            pics[key] = img
    return pics


ROUND = 5.0  # px from the outline over which the body rounds off to full depth


def distance_inside(mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance from each set pixel to the nearest unset one
    (two-pass chamfer, 1 / 1.4 weights); 0 outside."""
    big = 1e6
    d = np.where(mask, big, 0.0)
    h, w = d.shape
    for y in range(h):
        for x in range(w):
            if d[y, x]:
                d[y, x] = min(d[y, x],
                              (d[y - 1, x] + 1) if y else 1, (d[y, x - 1] + 1) if x else 1,
                              (d[y - 1, x - 1] + 1.4) if x and y else 1.4,
                              (d[y - 1, x + 1] + 1.4) if y and x + 1 < w else 1.4)
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if d[y, x]:
                d[y, x] = min(d[y, x],
                              (d[y + 1, x] + 1) if y + 1 < h else 1, (d[y, x + 1] + 1) if x + 1 < w else 1,
                              (d[y + 1, x + 1] + 1.4) if x + 1 < w and y + 1 < h else 1.4,
                              (d[y + 1, x - 1] + 1.4) if y + 1 < h and x else 1.4)
    return d


def carve(front: np.ndarray, back: np.ndarray, side: np.ndarray):
    """front: seen from +z (Link faces +z). back: from -z. side: from -x, image x = world +z.
    Returns list of (x, y, z, r, g, b) with x, z centred on the feet and y up from the ground.

    Pillow model: each front/back pixel gets a depth that rounds off toward the
    outline (full depth ROUND px in), between the side profile's front and back
    edge on that row, and the side profile trims the result. The front and back
    pictures paint the faces they look at; the side picture paints the flanks."""
    fa, ba, sa = front[..., 3] > 0, back[:, ::-1, 3] > 0, side[..., 3] > 0
    back_rgb = back[:, ::-1, :3]
    if not (fa.any() and sa.any()):
        return []
    # side profile per row: front (max z) and back (min z) edges and centre
    zf = np.full(SIZE, np.nan)
    zb = np.full(SIZE, np.nan)
    for y in range(SIZE):
        zs = np.nonzero(sa[y])[0]
        if len(zs):
            zb[y], zf[y] = zs.min(), zs.max() + 1
    rows = np.nonzero(~np.isnan(zf))[0]
    zf = np.interp(np.arange(SIZE), rows, zf[rows])  # rows the side picture misses
    zb = np.interp(np.arange(SIZE), rows, zb[rows])
    zc = (zf + zb) / 2
    sil = fa | ba
    roundf = np.sqrt(np.clip(distance_inside(fa | ba) / ROUND, 0, 1))
    occ = np.zeros((SIZE, SIZE, SIZE), bool)  # [y, x, z]
    zi = np.arange(SIZE)
    for y in range(SIZE):
        for x in np.nonzero(sil[y])[0]:
            r = roundf[y, x]
            lo = zc[y] - (zc[y] - zb[y]) * r
            hi = zc[y] + (zf[y] - zc[y]) * r
            col = (zi >= np.floor(lo)) & (zi < np.ceil(hi))
            # the side profile trims, but never to nothing (thin hair tips)
            trimmed = col & sa[y]
            occ[y, x] = trimmed if trimmed.any() else col
    vox = []
    ys, xs, zs = np.nonzero(occ)
    for y, x, z in zip(ys, xs, zs):
        exposed = (z + 1 >= SIZE or not occ[y, x, z + 1]) or (z == 0 or not occ[y, x, z - 1]) or \
                  (x == 0 or not occ[y, x - 1, z]) or (x + 1 >= SIZE or not occ[y, x + 1, z]) or \
                  (y == 0 or not occ[y - 1, x, z]) or (y + 1 >= SIZE or not occ[y + 1, x, z])
        if not exposed:
            continue
        # Surface direction from the neighbourhood: which way is mostly empty.
        x0, x1, z0, z1 = max(x - 2, 0), min(x + 3, SIZE), max(z - 2, 0), min(z + 3, SIZE)
        nx = occ[y, x0:x, z].sum() - occ[y, x + 1:x1, z].sum()   # >0: open toward +x
        nz = occ[y, x, z0:z].sum() - occ[y, x, z + 1:z1].sum()   # >0: open toward +z (front)
        if abs(nx) > abs(nz) + 1 and side[y, z, 3]:
            c = side[y, z, :3]
        elif z >= zc[y]:
            c = front[y, x, :3] if fa[y, x] else back_rgb[y, x]
        else:
            c = back_rgb[y, x] if ba[y, x] else front[y, x, :3]
        vox.append((int(x) - FOOT_X, FOOT_Y - int(y), int(z) - FOOT_X, *(int(v) for v in c)))
    return vox


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    pics = load(a.frames)
    models = {}
    for name, base in POSES.items():
        steps = sorted({s for (anim, s) in pics if anim == base + DOWN})
        seq = []
        for s in steps:
            f, b, r = pics.get((base + DOWN, s)), pics.get((base + UP, s)), pics.get((base + RIGHT, s))
            if f is None or b is None or r is None:
                continue
            seq.append(carve(deoutline(f), deoutline(b), deoutline(r)))
        if seq:
            models[name] = seq
            print(f"{name}: {len(seq)} steps, {sum(map(len, seq)) // len(seq)} voxels avg")
    tpl = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")
    a.out.write_text(tpl.replace("/*MODELS*/null", json.dumps(models, separators=(",", ":"))), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
