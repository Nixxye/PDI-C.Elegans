"""Analise dos componentes: isolado ou amontoado."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .. import ops
from ..core import Component, Stage


class ComponentAnalysis(Stage):
    """Atributos de cada componente; amontoado = area >= cluster_ratio x referencia."""

    name = "analysis"
    needs_fit = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ref_area = cfg.analysis.ref_area or 0.0

    def fit(self, samples):
        if self.ref_area > 0:
            return
        areas = []
        for s in samples:
            n = int(s.labels.max())
            if n:
                areas += list(np.bincount(s.labels.ravel(), minlength=n + 1)[1:])
        self.ref_area = float(np.median(areas)) if areas else self.cfg.analysis.fallback_ref_area
        self.env.shared["ref_area"] = self.ref_area

    def process(self, ctx):
        a = self.cfg.analysis
        comps = []
        n = int(ctx.labels.max())
        slices = ndi.find_objects(ctx.labels)
        for j in range(1, n + 1):
            sl = slices[j - 1]
            if sl is None:
                continue
            m = ctx.labels[sl] == j
            area = int(m.sum())
            cy, cx = ndi.center_of_mass(m)
            r0, c0 = sl[0].start, sl[1].start
            padded = np.pad(m, 1)
            skel = ops.thin(padded)
            length, ends, junc = ops.skeleton_stats(skel)
            thick = 2.0 * float(ops.distance_transform(padded).max())
            ratio = area / max(self.ref_area, 1.0)
            is_cluster = ratio >= a.cluster_ratio
            comps.append(Component(
                id=j, bbox=(r0, c0, sl[0].stop, sl[1].stop), area=area,
                centroid=(cy + r0, cx + c0), skel_len=length, n_ends=ends, n_junctions=junc,
                thickness=thick, mean_width=area / max(length, 1), area_ratio=ratio,
                n_est=int(np.clip(round(ratio), 2, a.max_worms)) if is_cluster else 1,
                is_cluster=is_cluster,
            ))
        comps = self._drop_bad_shapes(ctx, comps)
        ctx.components = comps
        cl_ids = [c.id for c in comps if c.is_cluster]
        ctx.cluster_mask = np.isin(ctx.labels, cl_ids) if cl_ids else np.zeros(ctx.labels.shape, bool)
        ctx.save_artifact("cluster_mask", ctx.cluster_mask)
        ctx.metrics["n_clusters"] = len(cl_ids)

    def _drop_bad_shapes(self, ctx, comps):
        """Descarta componentes finos ou largos demais (largura media = area / esqueleto)."""
        a = self.cfg.analysis
        bad = {c.id for c in comps if (0 < a.min_mean_width > c.mean_width) or (0 < a.max_mean_width < c.mean_width)}
        ctx.metrics["n_rejected_shape"] = len(bad)
        ctx.save_artifact("rejected_mask", np.isin(ctx.labels, list(bad)) if bad else np.zeros(ctx.labels.shape, bool))
        if not bad:
            return comps
        keep = [c for c in comps if c.id not in bad]
        lut = np.zeros(int(ctx.labels.max()) + 1, ctx.labels.dtype)
        for new_id, c in enumerate(keep, start=1):
            lut[c.id] = new_id
            c.id = new_id
        ctx.labels = lut[ctx.labels]
        ctx.mask = ctx.labels > 0
        ctx.save_artifact("labels", ctx.labels)
        return keep
