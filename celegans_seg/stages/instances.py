"""Cada componente vira uma instancia (single ou cluster)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..core import Instance, Stage


class InstanceBuilder(Stage):
    """Gera `ctx.inst_labels` / `ctx.instances` a partir de `ctx.components` (um rotulo por componente)."""

    name = "instances"

    def process(self, ctx):
        inst = np.zeros_like(ctx.labels)
        instances = []
        for nxt, comp in enumerate(ctx.components, start=1):
            r0, c0, r1, c1 = comp.bbox
            m = ctx.labels[r0:r1, c0:c1] == comp.id
            if not m.any():
                continue
            inst[r0:r1, c0:c1][m] = nxt
            cy, cx = ndi.center_of_mass(m)
            instances.append(Instance(
                id=nxt, component_id=comp.id,
                source="cluster" if comp.is_cluster else "single",
                area=int(m.sum()), centroid=(cy + r0, cx + c0),
                ambiguous=comp.is_cluster,
            ))
        ctx.inst_labels, ctx.instances = inst, instances
        ctx.save_artifact("inst_labels", inst)
