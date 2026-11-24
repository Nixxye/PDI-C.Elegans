"""PARTE 2 - CNN que separa os vermes de um amontoado: ESQUELETO (a implementacao fica por conta de quem fizer a rede).

Este arquivo define o CONTRATO com o resto do projeto e liga a Mask R-CNN do colega (submodule `cnn/`): para usar,
rode com `--set refine.kind=cnn --set refine.weights=<checkpoint.pth>` (torch e torchvision so sao importados nesse caso).

Como o PDI entrega os dados (codigo pronto, nao precisa mexer):
  1. stages/analysis.py  decide quais componentes sao AMONTOADO (area >= analysis.cluster_ratio x a de um verme)
  2. stages/instances.py cada amontoado vira UMA instancia `source="cluster"`
  3. stages/refine.py    `CNNRefiner` recorta uma janela `export.patch_size` x `export.patch_size` (192) centrada no
                         amontoado, chama `ClusterSeparator.separate(patch)` e cola o resultado de volta no quadro,
                         trocando a instancia `cluster` por instancias `source="cnn"`

O recorte (`patch`) e um dict de arrays, todos S x S:
  image    uint8  quadro original em cinza
  signal   uint8  sinal pos-fundo (verme CLARO sobre fundo ~0), com o ganho do limiar (`threshold.gain`) aplicado
  mask     bool   mascara so daquele amontoado (o que a rede deve dividir)
O treino usa os mesmos campos, mais `gt` (N, S, S): uma mascara por verme do gabarito, sobreposicao preservada.
Os arquivos de treino sao gerados por `tools/build_cnn_dataset.py` (stages/export.py) e lidos por `load_patches`.

O que a rede deve devolver: uma mascara booleana por verme, dentro de `mask` (podem se sobrepor).

Atencao ao gabarito (ver docs/DADOS.md): ha mascaras DUPLICADAS (duas para o mesmo verme) e ele nao anota vermes cortados
pela borda nem alguns vermes inteiros; funda as duplicadas (`tools/pdi_report.py` tem `dedup`) antes de treinar/avaliar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

MIN_TARGET_AREA = 12  # vermes do gabarito menores que isso (dentro da mascara) sao ignorados


# --------------------------------------------------------------------------------------
# Dados (implementado: so le o que stages/export.py grava)
# --------------------------------------------------------------------------------------

def load_patches(root: str | Path, seqs: list[str] | None = None, require_gt: bool = True) -> list[dict]:
    """Le os .npz de `<root>/<sequencia>/*.npz`. Cada item: dict com image/signal/mask/gt/meta/seq/file."""
    root = Path(root)
    out = []
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if seqs is not None and sdir.name not in seqs:
            continue
        for f in sorted(sdir.glob("*.npz")):
            d = np.load(f, allow_pickle=False)
            meta = json.loads(str(d["meta"]))
            if "gt" not in d.files:
                if require_gt:
                    continue
                gt = np.zeros((0,) + d["mask"].shape, bool)
            else:
                gt = d["gt"]
            out.append(dict(image=d["image"], signal=d["signal"], mask=d["mask"], gt=gt, meta=meta,
                            seq=sdir.name, file=f.name))
    return out


def targets_in_mask(patch: dict) -> np.ndarray:
    """Alvos (N, S, S): cada verme do gabarito restrito a mascara do amontoado (so os maiores que MIN_TARGET_AREA)."""
    m = patch["mask"]
    g = patch["gt"] & m[None] if len(patch["gt"]) else np.zeros((0,) + m.shape, bool)
    return g[[i for i in range(len(g)) if g[i].sum() >= MIN_TARGET_AREA]]


# --------------------------------------------------------------------------------------
# Avaliacao (implementado, independe do modelo)
# --------------------------------------------------------------------------------------

def match_scores(pred: list[np.ndarray], gt: list[np.ndarray], thr: float = 0.5) -> dict:
    """Casa mascaras previstas e do gabarito (algoritmo hungaro sobre o IoU). Devolve tp/fp/fn dos pares com IoU >= thr."""
    if not pred or not gt:
        return dict(tp=0, fp=len(pred), fn=len(gt))
    p = np.stack([m.ravel() for m in pred]).astype(np.float32)
    g = np.stack([m.ravel() for m in gt]).astype(np.float32)
    inter = p @ g.T
    iou = inter / np.maximum(p.sum(1)[:, None] + g.sum(1)[None] - inter, 1)
    r, c = linear_sum_assignment(-iou)
    tp = int(sum(iou[a, b] >= thr for a, b in zip(r, c)))
    return dict(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp)


def f1_from(counts: dict) -> float:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return 2 * tp / max(2 * tp + fp + fn, 1)


# --------------------------------------------------------------------------------------
# A rede (Mask R-CNN do colega; `train` continua a cargo de tools/train_maskrcnn.ps1)
# --------------------------------------------------------------------------------------

class ClusterSeparator:
    """Separa um recorte de amontoado em mascaras por verme usando a Mask R-CNN do colega (submodule `cnn/`, treinada
    por `tools/train_maskrcnn.ps1`). `refine.kind=cnn` instancia esta classe com `refine.weights`, `refine.threshold`,
    `refine.min_area` e `refine.score_thr` (ver celegans_seg/config.py, RefineCfg).

    A rede foi treinada em quadros inteiros (912x736, ampliados ~1,09x pelo torchvision). Para o recorte (192x192) ter a
    mesma escala dos vermes, o lado menor e levado a `round(S * TRAIN_SCALE)` px em vez dos 800 px padrao."""

    TRAIN_SCALE = 800 / 736

    def __init__(self, weights: str | Path | None = None, threshold: float = 0.5, min_area: int = 20,
                 score_thr: float = 0.3):
        self.weights, self.threshold, self.min_area, self.score_thr = weights, threshold, min_area, score_thr
        self._model = None
        self._device = None
        self._mask_channel = False  # checkpoints de tools/cluster_train.py recebem a mascara do amontoado no 3o canal
        self._scale = self.TRAIN_SCALE

    def _load(self, size: int):
        import torch
        import torchvision

        if self.weights is None or not Path(self.weights).is_file():
            raise NotImplementedError(f"sem pesos da CNN (refine.weights={self.weights!r}); treine com "
                                      "tools/train_maskrcnn.ps1 e aponte refine.weights para o checkpoint.pth")
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=2, box_score_thresh=self.score_thr)
            ck = torch.load(self.weights, map_location="cpu", weights_only=True)
            m.load_state_dict(ck["model"])
            self._mask_channel = bool(ck.get("mask_channel", False))
            self._scale = float(ck.get("scale", self.TRAIN_SCALE))
            self._model = m.to(self._device).eval()
        self._model.transform.min_size = (int(round(size * self._scale)),)
        self._model.transform.max_size = 1333
        return self._model

    def predict_masks(self, patch: dict) -> list[np.ndarray]:
        """Roda a Mask R-CNN no recorte (cinza replicado em 3 canais; ou [cinza, cinza, mascara do amontoado]) e devolve uma mascara booleana (S, S) por verme
        detectado (score >= `score_thr`), ja restrita a `patch["mask"]` (podem se sobrepor). Lista vazia ou com 1
        mascara = nao separou (o amontoado fica como estava)."""
        import torch

        img = patch["image"]
        model = self._load(img.shape[0])
        g = torch.from_numpy(img).float().div(255.0)
        third = torch.from_numpy(patch["mask"].astype(np.float32)) if self._mask_channel else g
        x = torch.stack([g, g, third]).to(self._device)
        with torch.no_grad():
            out = model([x])[0]
        masks = (out["masks"][:, 0] >= self.threshold).cpu().numpy()
        return [m & patch["mask"] for m in masks]

    def separate(self, patch: dict) -> np.ndarray:
        """Mapa de rotulos (S, S) int32: 0 = fora do amontoado, 1..n = verme. E o que `CNNRefiner` usa (implementado
        em cima de `predict_masks`): pixel em varias mascaras fica com a primeira; pixel do amontoado sem mascara vai
        para o rotulo mais proximo, para nao perder area."""
        lab = np.zeros(patch["mask"].shape, np.int32)
        masks = [m & patch["mask"] for m in self.predict_masks(patch) if (m & patch["mask"]).sum() >= self.min_area]
        for i, m in enumerate(masks, start=1):
            lab[m & (lab == 0)] = i
        if masks:
            idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
            lab = np.where(patch["mask"], lab[tuple(idx)], 0).astype(np.int32)
        return lab


def train(patches: list[dict], out_path: str | Path, val_patches: list[dict] | None = None, **kwargs) -> Path:
    """A IMPLEMENTAR. `patches`: saida de `load_patches` (cada um com `gt`; use `targets_in_mask`). Grava os pesos em
    `out_path` (o caminho que `refine.weights` aponta) e o devolve. `val_patches`: validacao (escolher a melhor epoca)."""
    raise NotImplementedError("CNN ainda nao implementada: implemente train() em celegans_seg/cnn.py")
