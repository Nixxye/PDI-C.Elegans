"""PARTE 1 / Etapa D - Separacao de amontoados por transformada de distancia + watershed.

Esta e a melhor separacao "classica" que o conteudo da disciplina permite, e e de proposito
tratada como HIPOTESE (`Instance.ambiguous = True`): vermes cruzados ou colados em paralelo
nao tem pescoco na mascara, entao o watershed erra. Esses casos sao justamente o que a CNN
(parte 2) deve resolver, recebendo os recortes gerados por `stages/export.py`.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .. import ops
from ..core import Instance, Stage


class WatershedSplitter(Stage):
    """Gera `ctx.inst_labels` / `ctx.instances` (um rotulo por verme).

    Para cada amontoado:
      1. D = transformada de distancia da mascara  [Aula 01c, pag. 11]
      2. suaviza D (gaussiano, [Aula 04]) e pega maximos locais como marcadores, no maximo
         `n_est` deles e afastados de `marker_min_dist` px
      3. watershed sobre -D  [Aula 11, pag. 31-39: conceito de inundacao a partir dos minimos
         regionais]. Usamos -D (inverte "montes" de distancia em "bacias") em vez do gradiente
         da imagem, porque queremos dividir a FORMA, nao seguir bordas de intensidade.
    Componentes isolados passam direto como uma instancia.
    """

    name = "split"

    def process(self, ctx):
        c = self.cfg.split
        inst = np.zeros_like(ctx.labels)
        instances, nxt = [], 1
        dist_full = np.zeros(ctx.labels.shape, np.float32)
        markers_full = np.zeros(ctx.labels.shape, bool)
        for comp in ctx.components:
            r0, c0, r1, c1 = comp.bbox
            m = ctx.labels[r0:r1, c0:c1] == comp.id
            pieces = None
            if comp.is_cluster and c.enabled:
                pieces, dist, mk = self._split(m, comp.n_est)
                dist_full[r0:r1, c0:c1][m] = dist[m]
                if mk is not None:
                    markers_full[r0:r1, c0:c1] |= mk
            if pieces is None:
                pieces = m.astype(np.int32)  # tudo vira uma instancia so
            for p in range(1, int(pieces.max()) + 1):
                pm = pieces == p
                if not pm.any():
                    continue
                inst[r0:r1, c0:c1][pm] = nxt
                cy, cx = ndi.center_of_mass(pm)
                if not comp.is_cluster:
                    source = "single"
                elif int(pieces.max()) > 1:
                    source = "watershed"
                else:
                    source = "cluster"  # amontoado nao separado: so a mascara "aqui ha varios vermes"
                instances.append(Instance(
                    id=nxt, component_id=comp.id,
                    source=source,
                    area=int(pm.sum()), centroid=(cy + r0, cx + c0),
                    ambiguous=comp.is_cluster,
                ))
                nxt += 1
        ctx.inst_labels, ctx.instances = inst, instances
        ctx.save_artifact("dist", dist_full)
        ctx.save_artifact("markers", markers_full)
        ctx.save_artifact("inst_labels", inst)

    def _split(self, m: np.ndarray, n_est: int):
        c = self.cfg.split
        pad = 2
        mp = np.pad(m, pad)
        dist = ndi.gaussian_filter(ops.distance_transform(mp), c.smooth_sigma)
        peak = ndi.maximum_filter(dist, footprint=ops.disk(max(c.marker_min_dist // 2, 1)))
        cand = np.argwhere((dist == peak) & mp & (dist >= c.min_dt))
        cand = cand[np.argsort(-dist[tuple(cand.T)])] if len(cand) else cand
        kept = []
        for y, x in cand:  # guloso: maiores primeiro, respeitando a distancia minima
            if all(np.hypot(y - ky, x - kx) >= c.marker_min_dist for ky, kx in kept):
                kept.append((y, x))
            if len(kept) >= n_est:
                break
        crop = (slice(pad, -pad), slice(pad, -pad))
        if len(kept) < 2:
            return None, dist[crop], None
        cost = 255 - (dist / max(dist.max(), 1e-6) * 254).astype(np.uint8)
        cost[~mp] = 255
        while len(kept) >= 2:
            markers = np.zeros(mp.shape, np.int32)
            for i, (y, x) in enumerate(kept, start=1):
                markers[y, x] = i
            ws = ndi.watershed_ift(cost, markers)
            ws[~mp] = 0
            areas = np.bincount(ws.ravel(), minlength=len(kept) + 1)[1:]
            if areas.min() >= c.min_piece_area:
                mk = np.zeros(mp.shape, bool)
                for y, x in kept:
                    mk[y, x] = True
                return ws[crop].astype(np.int32), dist[crop], mk[crop]
            kept.pop(int(np.argmin(areas)))  # descarta o marcador de menor bacia e repete
        return None, dist[crop], None
