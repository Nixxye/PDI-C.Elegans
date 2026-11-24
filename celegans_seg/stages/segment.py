"""Segmentacao binaria [Aula 08]: limiar -> morfologia -> rotulagem."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .. import ops
from ..core import Stage


class Threshold(Stage):
    """Otsu [Aula 08, pag. 10-13], mais o crescimento pelo top-hat."""

    name = "threshold"

    def process(self, ctx):
        c = self.cfg.threshold
        s8 = np.clip(ctx.signal * 255.0 * c.gain, 0, 255).astype(np.uint8)
        t = max(ops.otsu_threshold(s8), c.min_threshold)
        mask = s8 > t
        ctx.metrics["otsu_t"] = t
        ctx.save_artifact("mask_temporal", mask)
        if c.grow_dist > 0 and ctx.signal_spatial is not None:
            # completa a cauda: sementes = mascara temporal; acrescenta o top-hat perto delas
            s8s = np.clip(ctx.signal_spatial * 255.0 * c.gain, 0, 255).astype(np.uint8)
            weak = s8s > t * c.grow_thr
            lab, n = ops.label_components(mask)
            lab, n = ops.remove_small(lab, n, c.grow_seed_area)
            seeds = lab > 0
            near = ndi.distance_transform_edt(~seeds) <= c.grow_dist
            added = ops.opening((weak & near) | seeds, c.grow_open)
            cand, nc = ops.label_components(added | seeds)
            mask = np.isin(cand, np.unique(cand[seeds]))
            ctx.metrics["grown_px"] = int(mask.sum() - seeds.sum())
            if c.spatial_min_area > 0:
                # verme parado: entra pelo top-hat se o componente for grande
                cs, ncs = ops.label_components(ops.opening(weak, c.grow_open))
                big = np.bincount(cs.ravel(), minlength=ncs + 1) >= c.spatial_min_area
                big[0] = False
                mask = mask | big[cs]
        ctx.mask = mask
        ctx.save_artifact("mask_raw", mask)


class Morphology(Stage):
    """Abertura -> fechamento -> preenche buracos [Aula 08, pag. 19-35]."""

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
    """Componentes conexos, vizinhanca-8 [Aula 08, pag. 50-60; Aula 01c, pag. 9]."""

    name = "labeling"

    def process(self, ctx):
        lab, n = ops.label_components(ctx.mask)
        c = self.cfg.labeling
        lab, n = ops.remove_small(lab, n, c.min_area)
        if c.bridge_gap > 0 and n > 1:
            # liga pedacos de verme partido (so os grandes)
            lab, n = ops.label_components(ops.bridge_components(lab, n, c.bridge_gap, c.bridge_min_area))
        ctx.labels = lab
        ctx.mask = lab > 0
        ctx.metrics["n_components"] = n
        ctx.save_artifact("labels", lab)
