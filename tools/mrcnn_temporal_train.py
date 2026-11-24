"""Treina a Mask R-CNN com entrada temporal: [cinza, cinza, mascara dos vermes do quadro anterior].

Mesma receita do treino do colega (cnn/detection/train.py): COCO pretreinado, 10 epocas, SGD, lr em degraus, AMP,
mesmo split (grupo 24_0_0_1 fora do treino). A unica diferenca e o 3o canal da entrada. Como a rede tem 3 canais de
entrada, nenhuma camada muda: o 3o canal (que era uma copia do cinza) passa a carregar o "chute" anterior.

No treino o chute anterior e o gabarito do quadro t-1 *estragado* para parecer uma predicao real: cada verme some com
prob. 10 %, a mascara toda desloca ate 3 px, e em 10 % das amostras o canal vem vazio (primeiro quadro da sequencia).
Na inferencia (tools/mrcnn_eval.py) o canal e a uniao das deteccoes da propria rede no quadro anterior.

  python tools/mrcnn_temporal_train.py                 # treino completo (~50 min na RTX 5070 Ti)
  python tools/mrcnn_temporal_train.py --smoke         # teste de 1 minuto
Saida: runs/cnn_temporal/{checkpoint.pth, model_<epoca>.pth}; retoma sozinho de checkpoint.pth.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cnn" / "detection"))
import utils  # noqa: E402  (do colega)
from celegans_dataset import CelegansDataset  # noqa: E402
from engine import train_one_epoch  # noqa: E402


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, base: CelegansDataset, indices, augment=True):
        self.base, self.indices, self.augment = base, list(indices), augment
        by_name = {}
        for i, iid in enumerate(base.image_ids):
            fn = str(base.coco.loadImgs([iid])[0]["file_name"]).replace("\\", "/")
            by_name[(Path(fn).parent.name, int(Path(fn).stem))] = iid
        self.key = {iid: k for k, iid in by_name.items()}
        self.by_name = by_name

    def __len__(self):
        return len(self.indices)

    def _prev_mask(self, iid, shape):
        group, num = self.key[iid]
        prev = self.by_name.get((group, num - 1))
        m = np.zeros(shape, np.float32)
        if prev is None or random.random() < 0.10:  # primeiro quadro / "sem informacao"
            return m
        for ann in self.base.coco.loadAnns(self.base.coco.getAnnIds(imgIds=[prev], iscrowd=None)):
            if random.random() < 0.10:  # a rede deixou este verme passar
                continue
            m = np.maximum(m, self.base.coco.annToMask(ann).astype(np.float32))
        dy, dx = (random.randint(-3, 3) for _ in range(2))
        return np.roll(m, (dy, dx), axis=(0, 1))

    def __getitem__(self, i):
        idx = self.indices[i]
        img, target = self.base[idx]  # transforms=None: imagem PIL RGB
        g = torch.from_numpy(np.array(img.convert("L"))).float() / 255.0
        prev = torch.from_numpy(self._prev_mask(self.base.image_ids[idx], tuple(g.shape)))
        x = torch.stack([g, g, prev])
        if self.augment and random.random() < 0.5:  # espelha imagem, canal anterior, mascaras e caixas juntos
            x = x.flip(-1)
            target["masks"] = target["masks"].flip(-1)
            w = x.shape[-1]
            b = target["boxes"].clone()
            b[:, [0, 2]] = w - target["boxes"][:, [2, 0]]
            target["boxes"] = b
        return x, target


def build_model(coco_weights=True):
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if coco_weights else None)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default=str(ROOT / "data" / "csb-1_extracted" / "csb-1_dataset"))
    ap.add_argument("--output-dir", default=str(ROOT / "runs" / "cnn_temporal"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--validation-group", default="24_0_0_1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    root = Path(args.data_path)
    base = CelegansDataset(root, root / "coco_annotations" / "train_annotations.json", transforms=None)
    idx = [i for i, iid in enumerate(base.image_ids) if base.image_groups[iid] != args.validation_group]
    if args.smoke:
        idx, args.epochs = idx[:8], 1
    ds = TemporalDataset(base, idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                         collate_fn=utils.collate_fn, drop_last=True, persistent_workers=args.workers > 0)
    print(f"{len(ds)} imagens de treino (grupo {args.validation_group} fora)")

    dev = torch.device("cuda")
    model = build_model().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6, 8], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = 0
    ck = out / "checkpoint.pth"
    if ck.exists() and not args.smoke:
        st = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["lr_scheduler"])
        start = st["epoch"] + 1
        print(f"retomando na epoca {start}")

    for epoch in range(start, args.epochs):
        train_one_epoch(model, opt, loader, dev, epoch, 20, scaler)
        sched.step()
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), lr_scheduler=sched.state_dict(),
                     epoch=epoch)
        torch.save(state, out / f"model_{epoch}.pth")
        torch.save(state, ck)
    print("pronto:", ck)


if __name__ == "__main__":
    main()
