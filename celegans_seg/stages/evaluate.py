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

        row = dict(frame=ctx.index, name=ctx.name, dice_semantic=dice, n_pred=len(pred), n_gt=len(gt),
                   precision=prec, recall=rec, f1_iou50=f1, n_gt_in_clusters=len(in_cluster), cluster_recall=cl_rec)
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
        summary = {k: mean(k) for k in ("dice_semantic", "precision", "recall", "f1_iou50", "cluster_recall")}
        self.env.shared["summary"] = summary
        print("\n== Avaliacao (media dos quadros) ==")
        for k, v in summary.items():
            print(f"  {k:15s} {v:.3f}")
