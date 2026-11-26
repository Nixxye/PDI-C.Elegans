"""Utilidades compartilhadas para avaliar a Mask R-CNN do colega (cnn/) com e sem informacao temporal.

- carga do modelo (1 canal de cinza replicado em 3, ou 3 canais com a mascara do quadro anterior no 3o canal);
- predicao sequencial de uma sequencia (RLE para caber na memoria);
- rastreador por IoU (pos-processamento, sem retreinar);
- metricas: F1 por verme (IoU >= 0,5 / 0,75), contagem e trocas de ID.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "csb-1_extracted" / "csb-1_dataset"
SCORE_LOW = 0.05  # guarda tambem deteccoes fracas: o rastreador pode aproveita-las
SCORE_THR = 0.30  # limiar do colega (run_model.py)
MASK_THR = 0.50


# ---------------------------------------------------------------- dados
def frames_of(seq: str) -> list[int]:
    return sorted(int(p.stem) for p in (DATASET / "images" / seq).glob("*.png"))


def load_gray(seq: str, k: int) -> np.ndarray:
    return np.array(Image.open(DATASET / "images" / seq / f"{k}.png").convert("L"))


def load_gt(seq: str, k: int) -> list[np.ndarray]:
    out = []
    for f in sorted((DATASET / "masks" / seq).glob(f"{k}_*.png")):
        m = np.array(Image.open(f))
        m = m.any(axis=2) if m.ndim == 3 else m > 0
        if m.any():
            out.append(m)
    return out


def rle(m: np.ndarray):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return r


def rles(masks) -> list:
    return [rle(m) for m in masks]


def dedup_gt(rles_list: list, thr: float = 0.5) -> list:
    """Funde mascaras duplicadas do gabarito (IoU > thr no mesmo quadro): fica a maior. O CSB-1 anota alguns videos duas vezes."""
    keep = []
    for r in sorted(rles_list, key=lambda r: -float(mu.area(r))):
        if not keep or mu.iou(keep, [r], [0]).max() <= thr:
            keep.append(r)
    return keep


def union_mask(masks, shape) -> np.ndarray:
    u = np.zeros(shape, bool)
    for m in masks:
        u |= m
    return u


# ---------------------------------------------------------------- modelo
def make_input(gray: np.ndarray, prev_mask: np.ndarray | None) -> torch.Tensor:
    """Entrada (3,H,W) em [0,1]. Sem `prev_mask`: cinza replicado (modelo base).
    Com `prev_mask` (modelo temporal): [cinza, cinza, mascara dos vermes do quadro anterior]."""
    g = torch.from_numpy(gray).float() / 255.0
    third = torch.from_numpy(prev_mask.astype(np.float32)) if prev_mask is not None else g
    return torch.stack([g, g, third])


def load_model(ckpt: str | Path, device) -> torch.nn.Module:
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=2, box_score_thresh=SCORE_LOW)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def detect(model, x: torch.Tensor, device) -> list[dict]:
    """Uma imagem -> lista de deteccoes {score, rle, box}, ordenadas por score (mascara em RLE: cabe na memoria)."""
    p = model([x.to(device)])[0]
    masks = (p["masks"][:, 0] >= MASK_THR).cpu().numpy()
    scores = p["scores"].cpu().numpy()
    boxes = p["boxes"].cpu().numpy()
    return [dict(score=float(s), rle=rle(m), box=b) for s, m, b in zip(scores, masks, boxes) if m.any()]


def union_rle(rs, shape) -> np.ndarray:
    if not rs:
        return np.zeros(shape, bool)
    return mu.decode(mu.merge(rs, intersect=False)).astype(bool)


def predict_sequence(model, seq: str, device, temporal: bool, limit: int | None = None, log=print):
    """Percorre a sequencia na ordem. Modelo temporal: o 3o canal e a uniao das deteccoes (score >= SCORE_THR)
    do quadro anterior, isto e, o proprio "chute" da rede."""
    ks = frames_of(seq)[:limit]
    out, prev = [], None
    for n, k in enumerate(ks):
        gray = load_gray(seq, k)
        if temporal and prev is None:
            prev = np.zeros(gray.shape, bool)
        dets = detect(model, make_input(gray, prev if temporal else None), device)
        out.append(dets)
        if temporal:
            prev = union_rle([d["rle"] for d in dets if d["score"] >= SCORE_THR], gray.shape)
        if n % 50 == 0:
            log(f"  quadro {n}/{len(ks)}")
    return ks, out


# ---------------------------------------------------------------- rastreador
class Tracker:
    """Pos-processamento temporal por IoU (sem retreinar), causal (so olha o passado).

    - associa as deteccoes do quadro as trilhas ativas (hungaro sobre IoU, minimo `iou_min`);
    - umuma trilha nova so nasce com score >= `score_new` (alto: evita ruido que aparece em 1 quadro), enquanto
      uma trilha existente continua com score >= `score_track`;
    - trilha sem deteccao e "mantida" (usa a ultima mascara) por ate `coast` quadros (verme perdido por 1-2 quadros);
    - deteccoes de score baixo sem trilha sao descartadas (ruido que aparece em 1 quadro).
    """

    # padroes escolhidos numa varredura na sequencia de validacao 24_0_0_1 (a de teste nao foi usada)
    def __init__(self, score_new=0.70, score_track=SCORE_THR, iou_min=0.30, coast=1):
        self.score_new, self.score_track, self.iou_min, self.coast = score_new, score_track, iou_min, coast
        self.tracks, self.next_id = [], 1

    def step(self, dets: list[dict]) -> list[dict]:
        cand = [d for d in dets if d["score"] >= self.score_track]
        iou = np.zeros((len(self.tracks), len(cand)))
        if self.tracks and cand:
            iou = mu.iou([t["rle"] for t in self.tracks], [d["rle"] for d in cand], [0] * len(cand))
        used_t, used_d = set(), set()
        if iou.size:
            for i, j in zip(*linear_sum_assignment(-iou)):
                if iou[i, j] >= self.iou_min:
                    t, d = self.tracks[i], cand[j]
                    t.update(rle=d["rle"], score=d["score"], missed=0, hits=t["hits"] + 1)
                    used_t.add(i)
                    used_d.add(j)
        out = []
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t["missed"] += 1
        for j, d in enumerate(cand):
            if j in used_d:
                continue
            if d["score"] >= self.score_new:  # nasce trilha nova
                self.tracks.append(dict(id=self.next_id, rle=d["rle"], score=d["score"], missed=0, hits=1))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.coast]
        for t in self.tracks:
            out.append(dict(id=t["id"], rle=t["rle"], score=t["score"], coasted=t["missed"] > 0))
        return out


def run_tracker(dets_per_frame: list[list[dict]], **kw) -> list[list[dict]]:
    tr = Tracker(**kw)
    return [tr.step(d) for d in dets_per_frame]


def plain(dets_per_frame: list[list[dict]], thr=SCORE_THR) -> list[list[dict]]:
    """Sem rastreador: so o limiar de score (ids ausentes)."""
    return [[dict(id=None, rle=d["rle"], score=d["score"]) for d in dets if d["score"] >= thr]
            for dets in dets_per_frame]


# ---------------------------------------------------------------- metricas
def _iou_matrix(pred_rles, gt_rles):
    if not pred_rles or not gt_rles:
        return np.zeros((len(pred_rles), len(gt_rles)))
    return mu.iou(pred_rles, gt_rles, [0] * len(gt_rles))


def gt_tracks(gt_frames: list[list]) -> list[list[int]]:
    """Identidade dos vermes do gabarito, encadeando quadros consecutivos por IoU (>= 0,3)."""
    ids, nxt, prev = [], 1, []
    for masks in gt_frames:
        cur = [0] * len(masks)
        if prev and masks:
            iou = _iou_matrix([m for _, m in prev], masks)  # masks: RLEs do gabarito
            for i, j in zip(*linear_sum_assignment(-iou)):
                if iou[i, j] >= 0.3:
                    cur[j] = prev[i][0]
        for j in range(len(cur)):
            if cur[j] == 0:
                cur[j], nxt = nxt, nxt + 1
        ids.append(cur)
        prev = list(zip(cur, masks))
    return ids


def evaluate(pred_frames: list[list[dict]], gt_frames: list[list]) -> dict:
    """`gt_frames`: RLEs do gabarito (use `rles(load_gt(...))`). F1 por verme a IoU 0,5 e 0,75, contagem e (se houver ids) trocas de ID / fragmentacao."""
    gtid = gt_tracks(gt_frames)
    tp5 = tp75 = fp = fn = 0
    ious, cnt_err = [], []
    last_pred_of_gt: dict[int, int] = {}
    switches = 0
    seen_gt: set[int] = set()
    matched_frames = {}
    for t, (preds, gts) in enumerate(zip(pred_frames, gt_frames)):
        iou = _iou_matrix([p["rle"] for p in preds], gts)
        m5 = 0
        if iou.size:
            for i, j in zip(*linear_sum_assignment(-iou)):
                if iou[i, j] >= 0.5:
                    m5 += 1
                    ious.append(float(iou[i, j]))
                    tp75 += iou[i, j] >= 0.75
                    pid = preds[i].get("id")
                    if pid is not None:
                        g = gtid[t][j]
                        if g in last_pred_of_gt and last_pred_of_gt[g] != pid:
                            switches += 1
                        last_pred_of_gt[g] = pid
                    matched_frames.setdefault(gtid[t][j], set()).add(t)
        tp5 += m5
        fp += len(preds) - m5
        fn += len(gts) - m5
        cnt_err.append(abs(len(preds) - len(gts)))
        seen_gt.update(gtid[t])
    # fragmentacao: quantas vezes um verme do gabarito deixa de ser encontrado e volta
    frag = 0
    for g, fr in matched_frames.items():
        fr = sorted(fr)
        frag += sum(1 for a, b in zip(fr, fr[1:]) if b - a > 1)
    f1 = lambda tp: 2 * tp / max(2 * tp + fp + fn, 1)  # noqa: E731
    n_gt = tp5 + fn
    return dict(
        quadros=len(gt_frames), vermes_gt=int(n_gt), previstos=int(tp5 + fp), tp=int(tp5), fp=int(fp), fn=int(fn),
        precisao=tp5 / max(tp5 + fp, 1), recall=tp5 / max(n_gt, 1), f1=f1(tp5),
        f1_075=2 * tp75 / max(2 * tp75 + (n_gt - tp75) + (tp5 + fp - tp75), 1),
        iou_medio=float(np.mean(ious)) if ious else 0.0, erro_contagem=float(np.mean(cnt_err)),
        trocas_id=int(switches), fragmentacoes=int(frag),
    )
