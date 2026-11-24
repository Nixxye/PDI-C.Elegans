"""Pre-processamento: Normalize -> Denoise -> BackgroundSubtraction. Termina em `ctx.signal` (verme claro, fundo ~0)."""
from __future__ import annotations

import numpy as np

from .. import ops
from ..core import Stage


class Normalize(Stage):
    """Escala para [0,1] e garante verme escuro sobre fundo claro."""

    name = "normalize"

    def process(self, ctx):
        img = ctx.raw.astype(np.float32) / 255.0
        ctx.image = img if self.cfg.input.worms_dark else 1.0 - img
        ctx.save_artifact("image", ctx.image)


class Denoise(Stage):
    """Filtro gaussiano [Aula 04, pag. 16-19]."""

    name = "denoise"

    def process(self, ctx):
        ctx.image = ops.gaussian(ctx.image, self.cfg.denoise.sigma)
        ctx.save_artifact("denoised", ctx.image)


class BackgroundSubtraction(Stage):
    """Fundo por percentil temporal: `signal = fundo - quadro`. Top-hat [Aula 08, pag. 62-71] em `signal_spatial`."""

    name = "background"
    needs_fit = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.bg = None

    def fit(self, samples):
        self.bg = np.percentile(np.stack([s.image for s in samples]), self.cfg.background.percentile,
                                axis=0).astype(np.float32)
        self.env.shared["background"] = self.bg

    def process(self, ctx):
        ctx.signal = np.clip(self.bg - ctx.image, 0, 1).astype(np.float32)
        ctx.save_artifact("signal", ctx.signal)
        ctx.signal_spatial = ops.black_tophat(ctx.image, self.cfg.background.tophat_size).astype(np.float32)
        ctx.save_artifact("signal_spatial", ctx.signal_spatial)
