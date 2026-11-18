"""Avaliacao contra o gabarito (so roda se `input.masks` existir).

Metricas por quadro:
  - dice_semantic : Dice entre a uniao das mascaras previstas e a uniao do gabarito
  - n_pred / n_gt : numero de vermes previstos x reais
  - precision/recall/F1 a IoU >= 0.5 por instancia (casamento hungaro previsto x gabarito)
  - cluster_recall: dos vermes do gabarito que estao em amontoados, quantos foram recuperados
(Dice, MSE/PSNR sao as metricas de comparacao vistas em [Aula 05, pag. 18-22]; aqui usamos
Dice/IoU, que e o padrao para segmentacao.)
"""
from __future__ import annotations

import csv

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from ..core import Stage


def iou_matrix(pred: list[np.ndarray], gt: list[np.ndarray]) -> np.ndarray:
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)))
    p = np.stack([m.ravel() for m in pred]).astype(np.float32)
    g = np.stack([m.ravel() for m in gt]).astype(np.float32)
    inter = p @ g.T
    union = p.sum(1)[:, None] + g.sum(1)[None] - inter
    return inter / np.maximum(union, 1)


def gt_fraction(labels: np.ndarray, components: list, gt: list) -> dict:
    """{id do componente: fracao da sua area que esta sobre algum verme do gabarito}.

    Menos de 0,3 = "sem verme": sujeira, mancha, ou verme que o gabarito nao anotou (falso positivo)."""
    if not components:
        return {}
    gtu = np.any(gt, axis=0) if gt else np.zeros(labels.shape, bool)
    area = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    on_gt = np.bincount(labels[gtu].ravel(), minlength=len(area))
    return {c.id: float(on_gt[c.id] / area[c.id]) if area[c.id] else 0.0 for c in components}


def touches_border(bbox: tuple, shape: tuple, margin: int = 2) -> bool:
    """True se a caixa do componente encosta na borda do quadro. O gabarito do CSB-1 NAO anota os vermes cortados
    pela borda, entao um componente sem gabarito que encosta nela e, em geral, um verme real nao anotado."""
    r0, c0, r1, c1 = bbox
    return r0 <= margin or c0 <= margin or r1 >= shape[0] - margin or c1 >= shape[1] - margin


# Forma tipica de um verme, para separar "verme que o gabarito nao anotou" de sujeira entre os componentes sem gabarito.
# Medido no CSB-1: 95,8 % dos isolados que TEM verme no gabarito caem nesta faixa; dos 187 componentes sem gabarito fora
# da borda, 28 tem forma de verme (12 de 12 amostrados eram vermes reais) e 159 nao (rabiscos e sujeira do agar).
WORM_AREA_RATIO = (0.7, 1.4)  # area / area de referencia de um verme
WORM_MEAN_WIDTH = (6.5, 10.5)  # px: area / comprimento do esqueleto
WORM_MIN_CONTRAST = 0.2  # sinal espacial (top-hat) medio
# "Verme pequeno": o que sobra sem gabarito, fora da borda e sem forma de verme normal, mas com tamanho e contraste minimos.
# Os vermes do gabarito sao todos grandes (area p10-p90 = 514-673 px), entao esta faixa NAO vem do gabarito: e uma DECISAO
# (os falsos positivos tem area mediana 255 px, comprimento de esqueleto 37 px e contraste 0,14; os vermes anotados, 580 px,
# 81 px e 0,22-0,31). Sem prova de que sejam vermes; aperte os limites para ver os falsos positivos voltarem.
SMALL_WORM_MIN_AREA = 150  # px (= labeling.min_area: tudo que sobrou da rotulagem passa)
SMALL_WORM_MIN_CONTRAST = 0.1


def classify_no_gt(ctx, gt: list, thr: float = 0.3, small_min_area: int = SMALL_WORM_MIN_AREA,
                   small_min_contrast: float = SMALL_WORM_MIN_CONTRAST) -> dict:
    """{id: categoria} dos componentes com < `thr` da area sobre o gabarito. Categorias, nesta ordem:

    - "borda": encosta na borda do quadro (o gabarito nao anota vermes cortados pela borda);
    - "verme": tem forma, largura e contraste de verme normal (provavel verme que o gabarito nao anotou);
    - "pequeno": area >= small_min_area e contraste >= small_min_contrast (provavel verme PEQUENO nao anotado; e uma
      decisao, nao uma medicao: ver SMALL_WORM_MIN_AREA);
    - "fp": o resto (falso positivo).
    """
    fractions = gt_fraction(ctx.labels, ctx.components, gt)
    sig = ctx.signal_spatial if ctx.signal_spatial is not None else ctx.signal
    area = np.bincount(ctx.labels.ravel(), minlength=int(ctx.labels.max()) + 1)
    contrast = np.bincount(ctx.labels.ravel(), weights=sig.ravel(), minlength=len(area)) / np.maximum(area, 1)
    out = {}
    for c in ctx.components:
        if fractions[c.id] >= thr:
            continue
        if touches_border(c.bbox, ctx.labels.shape):
            out[c.id] = "borda"
        elif (WORM_AREA_RATIO[0] <= c.area_ratio <= WORM_AREA_RATIO[1] and WORM_MEAN_WIDTH[0] <= c.mean_width <= WORM_MEAN_WIDTH[1]
              and contrast[c.id] >= WORM_MIN_CONTRAST):
            out[c.id] = "verme"
        elif c.area >= small_min_area and contrast[c.id] >= small_min_contrast:
            out[c.id] = "pequeno"
        else:
            out[c.id] = "fp"
    return out


class Evaluate(Stage):
    name = "evaluate"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.rows = []

    def process(self, ctx):
        if self.env.masks is None:
            return
        gt = self.env.masks.instances(ctx.name)
        pred = [ctx.inst_labels == i.id for i in ctx.instances]
        gt_union = np.any(gt, axis=0) if gt else np.zeros_like(ctx.mask)
        pr_union = ctx.inst_labels > 0
        inter = (gt_union & pr_union).sum()
        dice = 2 * inter / max(gt_union.sum() + pr_union.sum(), 1)

        iou = iou_matrix(pred, gt)
        tp = 0
        matched_gt = set()
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            for a, b in zip(r, c):
                if iou[a, b] >= 0.5:
                    tp += 1
                    matched_gt.add(b)
        prec = tp / max(len(pred), 1)
        rec = tp / max(len(gt), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        # vermes do gabarito que caem dentro de componentes marcados como amontoado
        cl_ids = {c.id for c in ctx.components if c.is_cluster}
        in_cluster = [
            j for j, g in enumerate(gt)
            if cl_ids & set(np.unique(ctx.labels[g]).tolist()) - {0}
        ]
        cl_rec = (len([j for j in in_cluster if j in matched_gt]) / len(in_cluster)) if in_cluster else float("nan")

        # sem exigir a separacao dentro do amontoado:
        #  - gabarito: amontoado = componente conexo da uniao do gabarito com >= 2 vermes
        #  - f1_single: vermes isolados previstos x vermes do gabarito fora de amontoados
        #  - cluster_dice: mascara de amontoados prevista x do gabarito (nan se nenhuma tem)
        gt_lab, _ = ndi.label(gt_union, structure=np.ones((3, 3)))
        owner = [int(np.bincount(gt_lab[g].ravel(), minlength=1)[1:].argmax()) + 1 if g.any() else 0 for g in gt]
        count = {o: owner.count(o) for o in set(owner)}
        gt_single = [g for g, o in zip(gt, owner) if count[o] < 2]
        gt_cluster = np.isin(gt_lab, [o for o, n in count.items() if n >= 2]) if gt else np.zeros_like(ctx.mask)
        pred_single = [ctx.inst_labels == i.id for i in ctx.instances if i.source == "single"]
        # separados dentro de amontoado (CNN) x vermes do gabarito que estao em amontoado
        gt_in_cl = [g for g, o in zip(gt, owner) if count[o] >= 2]
        pred_in_cl = [ctx.inst_labels == i.id for i in ctx.instances if i.source in ("cnn",)]
        iou_c = iou_matrix(pred_in_cl, gt_in_cl)
        tp_c = 0
        if iou_c.size:
            r, c = linear_sum_assignment(-iou_c)
            tp_c = int(sum(iou_c[a, b] >= 0.5 for a, b in zip(r, c)))
        f1_c = 2 * tp_c / (len(pred_in_cl) + len(gt_in_cl)) if (pred_in_cl or gt_in_cl) else float("nan")
        iou_s = iou_matrix(pred_single, gt_single)
        tp_s = 0
        if iou_s.size:
            r, c = linear_sum_assignment(-iou_s)
            tp_s = int(sum(iou_s[a, b] >= 0.5 for a, b in zip(r, c)))
        p_s, r_s = tp_s / max(len(pred_single), 1), tp_s / max(len(gt_single), 1)
        f1_s = 2 * p_s * r_s / max(p_s + r_s, 1e-9) if (pred_single or gt_single) else float("nan")
        pc = ctx.cluster_mask if ctx.cluster_mask is not None else np.zeros_like(gt_cluster)
        cl_dice = 2 * (pc & gt_cluster).sum() / (pc.sum() + gt_cluster.sum()) if (pc.any() or gt_cluster.any()) else float("nan")

        row = dict(frame=ctx.index, name=ctx.name, dice_semantic=dice, n_pred=len(pred), n_gt=len(gt),
                   precision=prec, recall=rec, f1_iou50=f1, n_gt_in_clusters=len(in_cluster), cluster_recall=cl_rec,
                   f1_single=f1_s, cluster_dice=cl_dice, f1_in_cluster=f1_c)
        ctx.metrics.update(row)
        self.rows.append(row)

    def finish(self):
        if not self.rows:
            return
        out = self.env.out_dir
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            wr.writeheader()
            wr.writerows(self.rows)
        mean = lambda k: float(np.nanmean([r[k] for r in self.rows]))
        summary = {k: mean(k) for k in ("dice_semantic", "precision", "recall", "f1_iou50", "cluster_recall",
                                         "f1_single", "cluster_dice", "f1_in_cluster")}
        self.env.shared["summary"] = summary
        print("\n== Avaliacao (media dos quadros) ==")
        for k, v in summary.items():
            print(f"  {k:15s} {v:.3f}")
