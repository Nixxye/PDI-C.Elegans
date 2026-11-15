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
        ctx.save_artifact("mask_temporal", mask)  # so o que o Otsu pegou no sinal da etapa 4
        if c.grow_dist > 0 and ctx.signal_spatial is not None:
            # O fundo temporal "absorve" a parte do verme que fica parada (sinal fraco no corpo/cauda), mas
            # o verme continua escuro na imagem. Entao: sementes = mascara temporal; completa com o
            # top-hat (escuro em relacao a vizinhanca, sem depender do fundo) so perto das sementes
            # e so o que esta ligado a elas. Nao esta nos slides (reconstrucao geodesica).
            s8s = np.clip(ctx.signal_spatial * 255.0 * c.gain, 0, 255).astype(np.uint8)
            weak = s8s > t * c.grow_thr  # mesmo rigor do limiar principal: limiar proprio (Otsu do top-hat) pegava a sujeira do agar
            lab, n = ops.label_components(mask)
            lab, n = ops.remove_small(lab, n, c.grow_seed_area)
            seeds = lab > 0
            near = ndi.distance_transform_edt(~seeds) <= c.grow_dist
            added = ops.opening((weak & near) | seeds, c.grow_open)  # espinhos finos nao passam; cauda (largura de verme) passa
            cand, nc = ops.label_components(added | seeds)
            mask = np.isin(cand, np.unique(cand[seeds]))
            ctx.metrics["grown_px"] = int(mask.sum() - seeds.sum())
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
        c = self.cfg.labeling
        lab, n = ops.remove_small(lab, n, c.min_area)
        if c.bridge_gap > 0 and n > 1:
            # verme partido em dois (meio parado/claro): liga os pedacos grandes e rotula de novo.
            # Feito DEPOIS de tirar os pequenos: ligar tambem os pequenos junta sujeira do agar
            # (testado: bridge_min_area 60 gerou amontoados falsos)
            lab, n = ops.label_components(ops.bridge_components(lab, n, c.bridge_gap, c.bridge_min_area))
        ctx.labels = lab
        ctx.mask = lab > 0
        ctx.metrics["n_components"] = n
        ctx.save_artifact("labels", lab)
