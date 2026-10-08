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


def carve(front: np.ndarray, back: np.ndarray, side: np.ndarray):
    """front: seen from +z (Link faces +z). back: from -z. side: from -x, image x = world +z.
    Returns list of (x, y, z, r, g, b) with x, z centred on the feet and y up from the ground."""
    fa, ba, sa = front[..., 3] > 0, back[:, ::-1, 3] > 0, side[..., 3] > 0
    back_rgb = back[:, ::-1, :3]
    # occupancy[y, x, z]; image rows go down, x/z columns centre on the feet
    occ = fa[:, :, None] & ba[:, :, None] & sa[:, None, :]
    if not occ.any():
        return []
    vox = []
    ys, xs, zs = np.nonzero(occ)
    for y, x, z in zip(ys, xs, zs):
        # visible faces, in priority order: front (+z), back (-z), right side (-x), left side (+x)
        col = None
        if z + 1 >= SIZE or not occ[y, x, z + 1]:
            col = front[y, x, :3]
        elif z == 0 or not occ[y, x, z - 1]:
            col = back_rgb[y, x]
        elif x == 0 or not occ[y, x - 1, z]:
            col = side[y, z, :3]
        elif x + 1 >= SIZE or not occ[y, x + 1, z]:
            col = side[y, z, :3]  # left side: mirror of the right
        elif y == 0 or not occ[y - 1, x, z]:
            col = front[y, x, :3]  # top: take the front picture's colour
        else:
            continue  # inside, never seen
        vox.append((int(x) - FOOT_X, FOOT_Y - int(y), int(z) - FOOT_X, *(int(c) for c in col)))
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
            seq.append(carve(f, b, r))
        if seq:
            models[name] = seq
            print(f"{name}: {len(seq)} steps, {sum(map(len, seq)) // len(seq)} voxels avg")
    tpl = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")
    a.out.write_text(tpl.replace("/*MODELS*/null", json.dumps(models, separators=(",", ":"))), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
