"""PARTE 1 -> PARTE 2: entrega de dados para a CNN.

Para cada amontoado detectado, grava um recorte quadrado (`patch_size`) centrado nele, em
`<out>/cnn_dataset/<sequencia>/f<quadro>_c<componente>.npz`, mais um `index.csv`.

Conteudo de cada .npz (todos S x S, S = patch_size; fora da imagem = 0):
  image      uint8   quadro original em cinza
  signal     uint8   sinal pos-fundo (verme claro), ganho aplicado
  mask       bool    mascara binaria do amontoado (so este componente)
  dist       float16 transformada de distancia [Aula 01c, pag. 11]
  skeleton   bool    esqueleto por afinamento [Aula 08, pag. 37-48]
  gt         bool    (N, S, S) uma mascara por verme do gabarito -- COM sobreposicao -- se houver
  meta       json    frame, componente, bbox, n_est, area, fits (False se o amontoado nao cabe em S)
O canal `gt` e o alvo de treino; `dist` e `skeleton` sao entradas auxiliares que a
CNN pode ou nao usar (ablacao: com/sem informacao classica).
"""
from __future__ import annotations

import csv
import json

import numpy as np
from PIL import Image

from .. import ops
from ..core import Stage


def crop_center(arr: np.ndarray, cy: int, cx: int, size: int, fill=0) -> np.ndarray:
    """Janela size x size centrada em (cy, cx); o que cai fora da imagem vira `fill`."""
    out = np.full((size, size) + arr.shape[2:], fill, dtype=arr.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, arr.shape[0]), min(x0 + size, arr.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0 : ye - y0, xs - x0 : xe - x0] = arr[ys:ye, xs:xe]
    return out


class ClusterPatchExporter(Stage):
    name = "export"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.rows = []
        seq = cfg.input.sequence or (getattr(env.source, "folder", None) and env.source.folder.name) or "seq"
        self.dir = env.out_dir / "cnn_dataset" / str(seq)

    def process(self, ctx):
        e = self.cfg.export
        if ctx.index % max(e.save_every, 1):
            return
        if ctx.cluster_mask is not None:  # a mascara binaria "aqui ha amontoado", um PNG por quadro
            mdir = self.env.out_dir / "cluster_masks"
            mdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ctx.cluster_mask.astype(np.uint8) * 255).save(mdir / f"{ctx.name}.png")
        targets = [c for c in ctx.components if c.is_cluster or not e.only_clusters]
        if not targets:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        gt = self.env.masks.instances(ctx.name) if self.env.masks else []
        s8 = np.clip(ctx.signal * 255 * self.cfg.threshold.gain, 0, 255).astype(np.uint8)
        for comp in targets:
            cy, cx = (int(round(v)) for v in comp.centroid)
            S = e.patch_size
            m_full = ctx.labels == comp.id
            m = crop_center(m_full, cy, cx, S)
            r0, c0, r1, c1 = comp.bbox
            fits = (r1 - r0) <= S and (c1 - c0) <= S
            data = dict(
                image=crop_center(ctx.raw, cy, cx, S),
                signal=crop_center(s8, cy, cx, S),
                mask=m,
                dist=ops.distance_transform(m).astype(np.float16),
                skeleton=ops.thin(m),
                meta=json.dumps(dict(frame=ctx.index, name=ctx.name, component=comp.id, bbox=comp.bbox,
                                     n_est=comp.n_est, area=comp.area, fits=bool(fits))),
            )
            if gt:
                g = [crop_center(x, cy, cx, S) for x in gt]
                g = [x for x in g if x.any() and (x & m).any()]  # vermes deste amontoado
                data["gt"] = np.stack(g) if g else np.zeros((0, S, S), bool)
            fname = f"f{ctx.index:05d}_c{comp.id:02d}.npz"
            np.savez_compressed(self.dir / fname, **data)
            self.rows.append(dict(file=fname, frame=ctx.index, component=comp.id, n_est=comp.n_est,
                                  area=comp.area, fits=int(fits), n_gt=(len(data["gt"]) if gt else -1)))

    def finish(self):
        if not self.rows:
            return
        with open(self.dir / "index.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            wr.writeheader()
            wr.writerows(self.rows)
        print(f"\nCNN: {len(self.rows)} recortes de amontoados em {self.dir}")
