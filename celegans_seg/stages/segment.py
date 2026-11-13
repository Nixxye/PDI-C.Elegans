"""PARTE 1 / Etapa B - Segmentacao binaria (Aula 08): limiar -> morfologia -> rotulagem."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .. import ops
from ..core import Stage


class Threshold(Stage):
    """Binariza `ctx.signal`.

    - otsu: [Aula 08, pag. 10-13].  Aplicamos um ganho antes (realce de contraste, Aula 03) e um
      PISO no limiar: em quadros sem vermes o Otsu separaria apenas ruido.
    - adaptive: limiar pela media local [Aula 08, pag. 14-15], para iluminacao nao uniforme.
    """

    name = "threshold"

    def process(self, ctx):
        c = self.cfg.threshold
        s8 = np.clip(ctx.signal * 255.0 * c.gain, 0, 255).astype(np.uint8)
        if c.method == "otsu":
            t = max(ops.otsu_threshold(s8), c.min_threshold)
            mask = s8 > t
            ctx.metrics["otsu_t"] = t
        elif c.method == "adaptive":
            mask = ops.local_mean_threshold(s8, c.adaptive_window, c.adaptive_offset) & (s8 > c.min_threshold)
        else:
            raise ValueError(f"threshold.method desconhecido: {c.method}")
        if ctx.valid is not None:
            mask = mask & ctx.valid
        ctx.mask = mask
        ctx.save_artifact("mask_raw", mask)


class Morphology(Stage):
    """Limpeza da mascara: abertura (tira ruido) -> fechamento (liga pedacos) -> preenche buracos.
    [Aula 08, pag. 19-35: dilatacao 20-21, erosao 22-23, abertura 24-25, fechamento 26-27,
    comparacao 28 e 33]."""

    name = "morphology"

    def process(self, ctx):
        c = self.cfg.morphology
        m = ops.opening(ctx.mask, c.opening_radius)
        m = ops.closing(m, c.closing_radius)
        if c.fill_holes:
            m = ndi.binary_fill_holes(m)
        ctx.mask = m
        ctx.save_artifact("mask_clean", m)


class Labeling(Stage):
    """Um rotulo por componente conexo (vizinhanca-8), descartando os muito pequenos.
    [Aula 08, pag. 50-60 (rotulagem em duas passadas); Aula 01c, pag. 9 (vizinhanca)].
    A pseudocolorizacao dos rotulos para visualizacao e a da [Aula 10, pag. 29]."""

    name = "labeling"

    def process(self, ctx):
        lab, n = ops.label_components(ctx.mask)
        lab, n = ops.remove_small(lab, n, self.cfg.labeling.min_area)
        ctx.labels = lab
        ctx.mask = lab > 0
        ctx.metrics["n_components"] = n
        ctx.save_artifact("labels", lab)
