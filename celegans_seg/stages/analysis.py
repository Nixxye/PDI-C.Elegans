"""PARTE 1 / Etapa C - Analise dos componentes: quem e verme isolado e quem e amontoado.

E o ponto de decisao do pipeline: componentes "normais" seguem direto; os "amontoados" sao
virar instancias (instances.py) e, mais tarde, entregues a CNN (refine.py / export.py).
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
        """Descarta o que nao tem forma de verme: finos demais (sujeira do agar) ou largos demais (mancha, bolha).

        Criterio: largura media = area / comprimento do esqueleto. Medido no CSB-1 (4.722 componentes, 7
        sequencias): `< 5 px` ou `> 40 px` removeu 179 componentes sem verme no gabarito e nenhum com verme.
        Vermes enrolados em bola tem largura media ate ~33 px, por isso o limite de cima e 40.
        Os ids dos componentes que ficam sao renumerados de 1..n; o que saiu fica em `artifacts["rejected_mask"]`.
        """
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
