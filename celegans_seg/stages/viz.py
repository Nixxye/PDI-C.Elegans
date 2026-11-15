"""Visualizacao: contorno de cada verme sobre o quadro, painel de etapas e GIF.

A cor de cada instancia usa uma pseudocolorizacao de rotulos, como o label2rgb da [Aula 10, pag. 29].
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..core import Stage


CLUSTER_COLOR = (255, 40, 40)  # amontoado nao separado (vai para a CNN)


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
        self.dir.mkdir(parents=True, exist_ok=True)
        rgb = np.stack([ctx.raw] * 3, axis=-1).copy()
        for inst in ctx.instances:
            m = ctx.inst_labels == inst.id
            edge = m & ~ndi.binary_erosion(m)
            if inst.ambiguous:  # veio de amontoado: contorno mais grosso
                edge = ndi.binary_dilation(edge) & m
            rgb[edge] = CLUSTER_COLOR if inst.source == "cluster" else id_color(inst.id)
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img)
        for inst in ctx.instances:
            if inst.source == "cluster":  # amontoado que a CNN nao separou
                y, x = inst.centroid
                d.text((x - 20, y - 5), "amontoado", fill=CLUSTER_COLOR)
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
            ("7. amontoados (mascara p/ CNN)", a.get("cluster_mask"), "gray"),
            ("8. instancias: vermes + amontoados", a.get("inst_labels"), "nipy_spectral"),
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
                self.env.out_dir / "overlay.gif", save_all=True, append_images=self.gif_frames[1:],
                duration=100, loop=0,
            )

