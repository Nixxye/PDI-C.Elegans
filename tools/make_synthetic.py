"""Gera uma sequencia sintetica com vermes que se cruzam, no MESMO formato do dataset Zenodo
(root/images/<seq>/k.png, root/masks/<seq>/k_i.png), para testar o pipeline sem os dados reais.

  python tools/make_synthetic.py --out data/synthetic --seq seq0 --frames 60 --worms 7
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


class Worm:
    def __init__(self, rng, size):
        self.rng, self.size = rng, size
        self.pos = rng.uniform(0.2, 0.8, 2) * size
        self.heading = rng.uniform(0, 2 * np.pi)
        self.speed = rng.uniform(1.5, 3.0)
        self.length, self.n = rng.uniform(45, 70), 14
        self.phase = rng.uniform(0, 2 * np.pi)
        self.amp = rng.uniform(4, 8)
        self.t = 0

    def step(self):
        self.t += 1
        self.heading += self.rng.normal(0, 0.12)
        d = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.pos = self.pos + self.speed * d
        for k in (0, 1):  # rebate nas bordas
            if not 25 < self.pos[k] < self.size - 25:
                self.heading = np.arctan2(*(self.size / 2 - self.pos)[::-1]) + self.rng.normal(0, 0.3)
                self.pos = np.clip(self.pos, 25, self.size - 25)

    def body(self):
        d = np.array([np.cos(self.heading), np.sin(self.heading)])
        nrm = np.array([-d[1], d[0]])
        s = np.linspace(0, 1, self.n)
        wave = self.amp * np.sin(2 * np.pi * (1.2 * s) - 0.5 * self.t + self.phase)
        return [tuple(self.pos - d * self.length * si + nrm * w) for si, w in zip(s, wave)]

    def mask(self):
        im = Image.new("L", (self.size, self.size), 0)
        pts = self.body()
        dr = ImageDraw.Draw(im)
        dr.line(pts, fill=255, width=6, joint="curve")
        for x, y in (pts[0], pts[-1]):
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=255)
        return np.asarray(im) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seq", default="seq0")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--worms", type=int, default=7)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    img_dir = Path(a.out) / "images" / a.seq
    msk_dir = Path(a.out) / "masks" / a.seq
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[: a.size, : a.size]
    bg = 0.80 - 0.10 * (xx / a.size) + 0.05 * np.sin(yy / 40.0)  # iluminacao nao uniforme
    bg = bg + ndi.gaussian_filter(rng.normal(0, 1, bg.shape), 6) * 0.15  # manchas do agar
    worms = [Worm(rng, a.size) for _ in range(a.worms)]
    for k in range(a.frames):
        masks = []
        for w in worms:
            w.step()
            masks.append(w.mask())
        alpha = ndi.gaussian_filter(np.any(masks, axis=0).astype(np.float32), 0.8)
        frame = bg * (1 - 0.55 * alpha) + rng.normal(0, 0.012, bg.shape)
        Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).save(img_dir / f"{k}.png")
        for i, m in enumerate(masks):
            Image.fromarray((m * 255).astype(np.uint8)).save(msk_dir / f"{k}_{i}.png")
    print(f"{a.frames} quadros em {Path(a.out) / 'images' / a.seq}")


if __name__ == "__main__":
    main()
