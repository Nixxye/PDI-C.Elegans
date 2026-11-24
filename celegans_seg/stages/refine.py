"""PARTE 2 (gancho) - Onde a CNN entra no pipeline.

O pipeline classico termina com instancias (`ctx.instances`); cada amontoado e UMA instancia `source="cluster"`.
`CNNRefiner` troca essas instancias pelos vermes que `cnn.ClusterSeparator` separa dentro delas. O resto (avaliacao,
visualizacao) nao muda. O que a rede recebe e devolve esta descrito no topo de celegans_seg/cnn.py.

`refine.kind` no config.yaml escolhe o estagio: `none` (padrao, so PDI) ou `cnn`.
"""
from __future__ import annotations

import numpy as np

from ..core import Instance, Stage
from .export import crop_center


class NullRefiner(Stage):
    """Padrao (`refine.kind: none`): nao faz nada; os amontoados continuam como uma instancia `cluster`."""

    name = "refine"

    def process(self, ctx):
        return


def paste_center(patch: np.ndarray, shape: tuple, cy: int, cx: int) -> np.ndarray:
    """Inverso de `crop_center`: devolve um array do tamanho do quadro com `patch` colado na janela."""
    S = patch.shape[0]
    out = np.zeros(shape, patch.dtype)
    y0, x0 = cy - S // 2, cx - S // 2
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + S, shape[0]), min(x0 + S, shape[1])
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = patch[ys - y0 : ye - y0, xs - x0 : xe - x0]
    return out


class CNNRefiner(Stage):
    """Troca cada amontoado (`source="cluster"`) pelos vermes que a CNN separa dentro dele.

    Monta o mesmo recorte que `export.py` grava (imagem, sinal, mascara do amontoado), pergunta a
    `cnn.ClusterSeparator.separate` e devolve as instancias `source="cnn"` (marcadas `ambiguous`: a separacao dentro de
    um amontoado e menos confiavel que a de um verme isolado). Se a CNN nao achar >= 2 vermes, ou o amontoado precisar de
    uma janela maior que `refine.max_window`, ele fica como estava. Amontoados que nao cabem em `export.patch_size`
    usam uma janela maior (ate `max_window`).
    """

    name = "refine"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from ..cnn import ClusterSeparator  # esqueleto em celegans_seg/cnn.py (a rede e implementada la)

        r = cfg.refine
        self.sep = ClusterSeparator(r.weights, threshold=r.threshold, min_area=r.min_area, score_thr=r.score_thr)

    def process(self, ctx):
        clusters = [i for i in ctx.instances if i.source == "cluster"]
        if not clusters:
            return
        S = self.cfg.export.patch_size
        s8 = np.clip(ctx.signal * 255 * self.cfg.threshold.gain, 0, 255).astype(np.uint8)
        nxt = max(i.id for i in ctx.instances) + 1
        for inst in clusters:
            m_full = ctx.inst_labels == inst.id
            cy, cx = (int(round(v)) for v in inst.centroid)
            # janela = patch_size, ou maior se o amontoado nao cabe (a rede ajusta a escala pelo tamanho da janela,
            # entao o verme continua com o tamanho que ela viu no treino)
            ys0, xs0 = np.nonzero(m_full)
            reach = max(cy - ys0.min(), ys0.max() - cy, cx - xs0.min(), xs0.max() - cx) + 1
            W = max(S, 2 * int(reach) + self.cfg.refine.window_margin)
            if W > self.cfg.refine.max_window:
                continue
            mask = crop_center(m_full, cy, cx, W)
            if mask.sum() != m_full.sum():  # parte do amontoado ficou fora da janela
                continue
            lab = self.sep.separate(dict(image=crop_center(ctx.raw, cy, cx, W), signal=crop_center(s8, cy, cx, W),
                                         mask=mask))
            if int(lab.max()) < 2:
                continue
            full = paste_center(lab, m_full.shape, cy, cx)
            ctx.inst_labels[m_full] = 0
            ctx.instances.remove(inst)
            for p in range(1, int(lab.max()) + 1):
                pm = (full == p) & m_full
                if not pm.any():
                    continue
                ctx.inst_labels[pm] = nxt
                ys, xs = np.nonzero(pm)
                ctx.instances.append(Instance(id=nxt, component_id=inst.component_id, source="cnn",
                                              area=int(pm.sum()), centroid=(float(ys.mean()), float(xs.mean())),
                                              ambiguous=True))
                nxt += 1
        ctx.save_artifact("inst_labels", ctx.inst_labels)
