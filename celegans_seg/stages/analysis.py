"""PARTE 1 / Etapa C - Analise dos componentes: quem e verme isolado e quem e amontoado.

E o ponto de decisao do pipeline: componentes "normais" seguem direto; os "amontoados" sao
separados (split.py) e, mais tarde, entregues a CNN (refine.py / export.py).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .. import ops
from ..core import Component, Stage


class ComponentAnalysis(Stage):
    """Extrai atributos de cada componente e marca os amontoados.

    Atributos (nivel "medio" da [Aula 06]): area, centroide, bounding box; esqueleto por
    afinamento [Aula 08, pag. 37-48] -> comprimento, pontas e juncoes; espessura pela
    transformada de distancia [Aula 01c, pag. 11].

    Criterio de amontoado: area >= `cluster_ratio` x area de referencia de um verme (mediana das
    areas dos componentes nos quadros de amostra). Limitacao: sobreposicao REDUZ a area visivel
    (dois vermes cruzados em X somam menos que 2x), por isso `n_junctions` tambem e registrado,
    para a CNN/heuristicas futuras.
    """

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
        self.ref_area = float(np.median(areas)) if areas else 400.0
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
                thickness=thick, area_ratio=ratio,
                n_est=int(np.clip(round(ratio), 2, a.max_worms)) if is_cluster else 1,
                is_cluster=is_cluster,
            ))
        ctx.components = comps
        cl_ids = [c.id for c in comps if c.is_cluster]
        ctx.cluster_mask = np.isin(ctx.labels, cl_ids) if cl_ids else np.zeros(ctx.labels.shape, bool)
        ctx.save_artifact("cluster_mask", ctx.cluster_mask)
        ctx.metrics["n_clusters"] = len(cl_ids)
