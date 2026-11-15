"""PARTE 2 (gancho) - Onde a CNN entra.

O pipeline classico termina com hipoteses de instancias (`ctx.instances`). Um `InstanceRefiner`
pode substituir a separacao dos amontoados pela predicao de uma CNN, sem tocar no resto.

Contrato (o que a CNN recebe e devolve) - os mesmos arrays que `export.py` grava no dataset:
  entrada : recorte quadrado (patch_size x patch_size) centrado no amontoado, com
            `image` (cinza), `signal` (sinal pos-fundo), `mask` (mascara binaria do amontoado),
            `dist` (transformada de distancia), `skeleton`, `ws_labels` (proposta do watershed)
  saida   : mapa de rotulos de instancia (patch_size x patch_size, 0 = fundo, 1..k = vermes),
            COM sobreposicao permitida se a rede for do tipo "uma mascara por verme".
Ao implementar, sobrescreva `refine_patch` e troque `refine.kind` no config.yaml.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..core import Instance, Stage
from .export import crop_center


class InstanceRefiner(Stage):
    """Base: percorre os amontoados e pede a `refine_patch` um novo mapa de instancias."""

    name = "refine"

    def refine_patch(self, patch: dict) -> np.ndarray | None:
        """Devolve mapa de rotulos (S x S) ou None para manter a hipotese do watershed."""
        raise NotImplementedError

    def process(self, ctx):  # pragma: no cover - sera usado pela CNN
        raise NotImplementedError("Implemente a insercao dos novos rotulos em ctx.inst_labels/ctx.instances")


class NullRefiner(InstanceRefiner):
    """Padrao: nao faz nada (mantem o resultado do watershed)."""

    def refine_patch(self, patch):
        return None

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


class CNNRefiner(InstanceRefiner):
    """Troca cada amontoado (`source="cluster"`) pelos vermes que a CNN separa dentro dele.

    Monta o mesmo recorte que `export.py` grava (imagem, sinal, mascara do amontoado), pergunta a
    `cnn.ClusterSeparator` e devolve as instancias `source="cnn"` (marcadas `ambiguous`: a
    identidade dentro de um amontoado e menos confiavel). Se a CNN nao achar >= 2 vermes, ou o
    amontoado nao couber no recorte, ele fica como estava.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from ..cnn import ClusterSeparator  # so precisa de torch com refine.kind=cnn

        r = cfg.refine
        self.sep = ClusterSeparator(r.weights, threshold=r.threshold, min_area=r.min_area,
                                    use_prev_count=r.use_prev_count)
        self.prev = None  # (indice do quadro, inst_labels, {id da instancia: quantos vermes ela vale}) do quadro anterior

    def _remember(self, ctx):
        """Guarda a saida deste quadro para o proximo (canais de entrada dos pesos `--temporal` e/ou `use_prev_count`)."""
        comps = {c.id: c for c in ctx.components}
        w = {i.id: (comps[i.component_id].n_est if i.source == "cluster" and i.component_id in comps else 1)
             for i in ctx.instances}
        self.prev = (ctx.index, ctx.inst_labels.copy(), w)

    def _prev_info(self, ctx, mask_win, cy, cx, S):
        """(uniao dos vermes do quadro anterior perto do amontoado, quantos sao), recortados na janela.
        Instancia que ja era um amontoado nao separado vale `n_est` vermes."""
        from ..cnn import PREV_DILATE

        if self.prev is None or self.prev[0] != ctx.index - 1:
            return None, 0
        _, lab, weight = self.prev
        lab_win = crop_center(lab, cy, cx, S)
        ids = [int(i) for i in np.unique(lab_win[ndi.binary_dilation(mask_win, iterations=PREV_DILATE)]) if i > 0]
        if not ids:
            return None, 0
        return np.isin(lab_win, ids), int(round(sum(weight.get(i, 1) for i in ids)))

    def process(self, ctx):
        try:
            self._separate(ctx)
        finally:
            self._remember(ctx)

    def _separate(self, ctx):
        clusters = [i for i in ctx.instances if i.source == "cluster"]
        if not clusters:
            return
        S = self.cfg.export.patch_size
        s8 = np.clip(ctx.signal * 255 * self.cfg.threshold.gain, 0, 255).astype(np.uint8)
        nxt = max(i.id for i in ctx.instances) + 1
        for inst in clusters:
            m_full = ctx.inst_labels == inst.id
            cy, cx = (int(round(v)) for v in inst.centroid)
            mask = crop_center(m_full, cy, cx, S)
            if mask.sum() != m_full.sum():  # parte do amontoado ficou fora da janela
                continue
            union, n_prev = self._prev_info(ctx, mask, cy, cx, S)
            lab = self.sep.separate(dict(image=crop_center(ctx.raw, cy, cx, S), signal=crop_center(s8, cy, cx, S),
                                         mask=mask, prev_union=union, prev_n=n_prev))
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
