"""Treina a Mask R-CNN para separar amontoados usando SO o Mating Dataset (md_dataset, WormSwin, Zenodo 7456803).

Por que: as mascaras do CSB-1 tem erros; o MD vem de outro video e outros anotadores. Aqui nenhum dado do CSB-1 e usado
(com `--init coco` nem os pesos dele: parte do COCO do torchvision; com `--init base` parte do modelo treinado no CSB-1).

Amostras = recortes de 192 px centrados em cada componente conexo do gabarito (uniao das mascaras, dilatada 1 px):
componentes com 2+ vermes (amontoados) pesam 4x mais que os de 1 verme. A cada passo o centro do recorte e deslocado
ate 30 px, e ha espelhamentos/rotacoes de 90 graus. Alvo = cada verme do gabarito restrito ao componente.
Selecao da melhor epoca: 15 % das imagens de treino (por imagem) ficam para validacao; o teste do MD nunca e usado.

  python tools/md_train.py --init coco --out runs/cnn_md_coco
  python tools/md_train.py --init base --out runs/cnn_md_base
Avaliar: python tools/md_eval.py --patch-model md_coco=runs/cnn_md_coco/checkpoint.pth ...
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from scipy import ndimage as ndi
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "cnn" / "detection"))
import cluster_train as CT  # noqa: E402  (dedup, box_of, validate, SCALE)
import utils  # noqa: E402
from celegans_seg import cnn  # noqa: E402
from engine import train_one_epoch  # noqa: E402

MD = ROOT / "data" / "md_extracted" / "md_dataset"
S = 192


def crop(arr, cy, cx, size=S):
    out = np.zeros((size, size) + arr.shape[2:], arr.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, arr.shape[0]), min(x0 + size, arr.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = arr[ys:ye, xs:xe]
    return out


def components(split):
    """Por imagem: (gray, [(mascara do componente, [vermes dentro dele])])."""
    coco = COCO(str(MD / f"{split}_annotations.json"))
    out = []
    for iid in sorted(coco.getImgIds()):
        info = coco.loadImgs([iid])[0]
        gray = np.array(Image.open(MD / "images" / info["file_name"]).convert("L"))
        gts = CT.dedup([coco.annToMask(a).astype(bool) for a in coco.loadAnns(coco.getAnnIds(imgIds=[iid]))])
        if not gts:
            continue
        union = np.zeros(gray.shape, bool)
        for g in gts:
            union |= g
        lab, nl = ndi.label(ndi.binary_dilation(union, iterations=1))
        comps = []
        for c in range(1, nl + 1):
            comp = (lab == c) & union
            inside = [g & comp for g in gts if (g & comp).sum() >= cnn.MIN_TARGET_AREA]
            ys, xs = np.nonzero(comp)
            if inside and (ys.max() - ys.min() < S) and (xs.max() - xs.min() < S):
                comps.append((comp, inside, (int(ys.mean()), int(xs.mean()))))
        out.append((gray, comps))
    return out


class MDDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.items = [(g, c, ins, ctr) for g, comps in images for c, ins, ctr in comps]

    def __len__(self):
        return len(self.items)

    def weight(self, i):
        return 4.0 if len(self.items[i][2]) >= 2 else 1.0

    def __getitem__(self, i):
        gray, comp, inside, (cy, cx) = self.items[i]
        cy += random.randint(-30, 30)
        cx += random.randint(-30, 30)
        g = torch.from_numpy(crop(gray, cy, cx)).float() / 255.0
        m = torch.from_numpy(crop(comp, cy, cx).astype(np.float32))
        masks = torch.from_numpy(np.stack([crop(x, cy, cx) for x in inside]).astype(np.uint8))
        keep = masks.flatten(1).sum(1) >= cnn.MIN_TARGET_AREA  # o deslocamento pode cortar um verme
        masks = masks[keep]
        x = torch.stack([g, g, g])  # 3o canal = cinza (como o modelo base)
        k, flip = random.randint(0, 3), random.random() < 0.5
        x, masks = torch.rot90(x, k, (1, 2)), torch.rot90(masks, k, (1, 2))
        if flip:
            x, masks = x.flip(-1), masks.flip(-1)
        boxes = torch.tensor([CT.box_of(mk.numpy() > 0) for mk in masks], dtype=torch.float32).reshape(-1, 4)
        target = dict(boxes=boxes, labels=torch.ones(len(masks), dtype=torch.int64), masks=masks, image_id=i,
                      area=masks.flatten(1).sum(1).float(), iscrowd=torch.zeros(len(masks), dtype=torch.int64))
        return x, target


def val_patches(images):
    out = []
    for gray, comps in images:
        for comp, inside, (cy, cx) in comps:
            out.append(dict(image=crop(gray, cy, cx), mask=crop(comp, cy, cx),
                            gt=np.stack([crop(x, cy, cx) for x in inside])))
    return out


def build_model(init, base_ckpt):
    if init == "coco":
        m = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
        return m
    return CT.build_model(ROOT / base_ckpt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", choices=["coco", "base"], default="coco")
    ap.add_argument("--base", default="runs/cnn_train/checkpoint.pth")
    ap.add_argument("--out", default="runs/cnn_md_coco")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    imgs = components("train")
    order = list(range(len(imgs)))
    random.Random(1).shuffle(order)
    nv = max(1, int(0.15 * len(imgs)))
    val_i = {order[k] for k in range(nv)}
    train = [imgs[i] for i in range(len(imgs)) if i not in val_i]
    val = val_patches([imgs[i] for i in sorted(val_i)])
    ds = MDDataset(train)
    ncl = sum(len(it[2]) >= 2 for it in ds.items)
    print(f"treino: {len(train)} imagens, {len(ds)} recortes ({ncl} amontoados); validacao: {len(val)} recortes")
    sampler = torch.utils.data.WeightedRandomSampler([ds.weight(i) for i in range(len(ds))], num_samples=len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                                         collate_fn=utils.collate_fn, drop_last=True,
                                         persistent_workers=args.workers > 0)
    dev = torch.device("cuda")
    model = build_model(args.init, args.base).to(dev)
    model.transform.min_size = (int(round(S * CT.SCALE)),)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)],
                                                 gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, [("epoca", "f1_val")]
    for epoch in range(args.epochs):
        model.train()
        model.transform.min_size = (int(round(S * CT.SCALE)),)
        train_one_epoch(model, opt, loader, dev, epoch, 50, scaler)
        sched.step()
        f1 = CT.validate(model, val, dev, S, mask_channel=False)
        print(f"epoca {epoch}: F1 validacao {f1:.4f}", flush=True)
        log.append((epoch, round(f1, 4)))
        state = dict(model=model.state_dict(), mask_channel=False, scale=CT.SCALE, epoch=epoch, f1_val=f1)
        torch.save(state, out / "last.pth")
        if f1 > best:
            best = f1
            torch.save(state, out / "checkpoint.pth")
    with open(out / "log.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(log)
    print(f"pronto: melhor F1 de validacao {best:.4f} -> {out / 'checkpoint.pth'}")


if __name__ == "__main__":
    main()
