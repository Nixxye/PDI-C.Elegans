"""Visualizacao: sobreposicao com contorno + ID + rastro, painel de etapas e GIF.

Cores por ID usam uma pseudocolorizacao de rotulos, como o label2rgb da [Aula 10, pag. 29].
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..core import Stage


def id_color(i: int) -> tuple:
    h = (i * 0.61803398875) % 1.0  # razao aurea: cores vizinhas bem distintas
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


class Visualizer(Stage):
    name = "visualize"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.dir = env.out_dir / "overlay"
        self.gif_frames = []
        self.panels = 0

    def process(self, ctx):
        v = self.cfg.viz
        tracker = self.env.shared.get("tracker")
        self.dir.mkdir(parents=True, exist_ok=True)
        rgb = np.stack([ctx.raw] * 3, axis=-1).copy()
        for inst in ctx.instances:
            m = ctx.inst_labels == inst.id
            edge = m & ~ndi.binary_erosion(m)
            tid = inst.track_id or 0
            if inst.ambiguous:  # veio de amontoado: contorno mais grosso
                edge = ndi.binary_dilation(edge) & m
            rgb[edge] = id_color(tid)
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img)
        if tracker is not None:
            for t in tracker.tracks.values():
                if not tracker.confirmed(t):
                    continue
                pts = [(x, y) for f, y, x, st, a, amb in t.history[-v.trail_length:]]
                if len(pts) > 1:
                    d.line(pts, fill=id_color(t.id), width=1)
        for inst in ctx.instances:
            trk = tracker.tracks.get(inst.track_id) if tracker else None
            if inst.track_id and (tracker is None or (trk is not None and tracker.confirmed(trk))):
                y, x = inst.centroid
                d.text((x + 4, y - 4), str(inst.track_id), fill=id_color(inst.track_id))
        if ctx.index % max(v.save_every, 1) == 0:
            img.save(self.dir / f"{ctx.index:05d}.png")
        if v.gif and ctx.index % max(v.gif_step, 1) == 0:
            self.gif_frames.append(img.copy())
        self._panel(ctx)

    # painel com o resultado de cada etapa (para o relatorio)
    def _panel(self, ctx):
        v = self.cfg.viz
        first = ctx.index == 0
        has_cluster = any(c.is_cluster for c in ctx.components)
        if not (first or (has_cluster and self.panels < v.debug_max_panels)):
            return
        if not first:
            self.panels += 1
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = ctx.artifacts
        items = [
            ("1. original", ctx.raw, "gray"),
            ("2. suavizado [Aula 04]", a.get("denoised"), "gray"),
            ("3. sinal pos-fundo", a.get("signal"), "gray"),
            ("4. limiar [Aula 08 p.10]", a.get("mask_raw"), "gray"),
            ("5. morfologia [Aula 08 p.24-27]", a.get("mask_clean"), "gray"),
            ("6. rotulos [Aula 08 p.50-60]", a.get("labels"), "nipy_spectral"),
            ("7. distancia [Aula 01c p.11]", a.get("dist"), "magma"),
            ("8. watershed [Aula 11 p.31-39]", a.get("inst_labels"), "nipy_spectral"),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, (title, im, cmap) in zip(axes.ravel(), items):
            if im is not None:
                shown = np.ma.masked_equal(im, 0) if cmap == "nipy_spectral" else im
                ax.imshow(shown, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.env.out_dir / f"painel_etapas_{ctx.index:05d}.png", dpi=110)
        plt.close(fig)

    def finish(self):
        if self.cfg.viz.gif and self.gif_frames:
            self.gif_frames[0].save(
                self.env.out_dir / "tracking.gif", save_all=True, append_images=self.gif_frames[1:],
                duration=100, loop=0,
            )

