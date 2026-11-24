"""Ajusta a Mask R-CNN SO nos amontoados escolhidos pelo PDI (recortes de 192 px de tools/build_cnn_dataset.py).

Parte do checkpoint treinado em quadros inteiros (tools/train_maskrcnn.ps1) e continua o treino nos recortes, com:
  - entrada [cinza, cinza, mascara do amontoado]: a rede sabe onde esta o amontoado (o 3o canal era copia do cinza);
    com `--no-mask-channel` o 3o canal continua sendo o cinza (so o conjunto de treino muda);
  - alvo = cada verme do gabarito (duplicatas fundidas) restrito a mascara do amontoado;
  - amostragem que da mais peso a amontoados com 3 vermes (x2) e 4+ (x3), que sao raros;
  - aumento: espelhamentos e rotacoes de 90 graus (o verme nao tem "cima");
  - escolhe a melhor epoca pelo F1 de validacao (sequencia 24_0_0_1). A de teste (24_2_1_1) nunca e usada.

  python tools/build_cnn_dataset.py --seqs <as de treino> --step 2 --out data/cnn_patches_cur   # uma vez
  python tools/cluster_train.py
Saida (`--out`, padrao runs/cnn_cluster): {checkpoint.pth (melhor), last.pth, log.csv}. Nao mexe nos outros treinos.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cnn" / "detection"))
from celegans_seg import cnn  # noqa: E402
import utils  # noqa: E402  (do colega)
from engine import train_one_epoch  # noqa: E402

TEST_SEQ = "24_2_1_1"
SCALE = 800 / 736  # mesma escala dos vermes do treino em quadro inteiro


def dedup(gt, thr=0.5):
    order = sorted(range(len(gt)), key=lambda i: -gt[i].sum())
    kept = []
    for i in order:
        if all((gt[i] & gt[j]).sum() / max((gt[i] | gt[j]).sum(), 1) <= thr for j in kept):
            kept.append(i)
    return [gt[i] for i in sorted(kept)]


def targets(p):
    m = p["mask"]
    g = [x & m for x in dedup(list(p["gt"]))] if len(p["gt"]) else []
    return [x for x in g if x.sum() >= cnn.MIN_TARGET_AREA]


def box_of(m):
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class ClusterDS(torch.utils.data.Dataset):
    def __init__(self, patches, augment, mask_channel=True):
        self.items = [(p, targets(p)) for p in patches]
        self.augment, self.mask_channel = augment, mask_channel

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, tg = self.items[i]
        g = torch.from_numpy(p["image"]).float() / 255.0
        x = torch.stack([g, g, torch.from_numpy(p["mask"].astype(np.float32)) if self.mask_channel else g])
        masks = torch.from_numpy(np.stack(tg)).to(torch.uint8) if tg else torch.zeros((0, *g.shape), dtype=torch.uint8)
        if self.augment:
            k, flip = random.randint(0, 3), random.random() < 0.5
            x, masks = torch.rot90(x, k, (1, 2)), torch.rot90(masks, k, (1, 2))
            if flip:
                x, masks = x.flip(-1), masks.flip(-1)
        boxes = torch.tensor([box_of(m.numpy() > 0) for m in masks], dtype=torch.float32).reshape(-1, 4)
        target = dict(boxes=boxes, labels=torch.ones(len(masks), dtype=torch.int64), masks=masks,
                      image_id=i, area=masks.flatten(1).sum(1).float(), iscrowd=torch.zeros(len(masks), dtype=torch.int64))
        return x, target

    def weight(self, i):
        n = len(self.items[i][1])
        return 1.0 if n <= 2 else 2.0 if n == 3 else 3.0


def build_model(base_ckpt):
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    m.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=True)["model"])
    return m


@torch.no_grad()
def validate(model, patches, dev, size, mask_channel=True):
    sep = cnn.ClusterSeparator(None)
    model.eval()
    model.transform.min_size = (int(round(size * SCALE)),)
    tot = dict(tp=0, fp=0, fn=0)
    for p in patches:
        g = torch.from_numpy(p["image"]).float() / 255.0
        x = torch.stack([g, g, torch.from_numpy(p["mask"].astype(np.float32)) if mask_channel else g]).to(dev)
        out = model([x])[0]
        preds = [m & p["mask"] for m in (out["masks"][:, 0] >= 0.5).cpu().numpy() if (m & p["mask"]).sum() >= sep.min_area]
        r = cnn.match_scores(preds, targets(p))
        for k in tot:
            tot[k] += r[k]
    return cnn.f1_from(tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default="data/cnn_patches_cur/cnn_dataset")
    ap.add_argument("--base", default="runs/cnn_train/checkpoint.pth", help="ponto de partida (modelo de quadro inteiro)")
    ap.add_argument("--out", default="runs/cnn_cluster")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--val-seq", default="24_0_0_1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-mask-channel", action="store_true",
                    help="3o canal = cinza (como no modelo base); so muda o conjunto de treino para os amontoados")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    allp = [p for p in cnn.load_patches(ROOT / args.patches, require_gt=True) if p["meta"].get("fits", True)]
    train = [p for p in allp if p["seq"] not in (TEST_SEQ, args.val_seq)]
    val = [p for p in allp if p["seq"] == args.val_seq]
    size = train[0]["image"].shape[0]
    mc = not args.no_mask_channel
    ds = ClusterDS(train, augment=True, mask_channel=mc)
    cnt = np.bincount([min(len(t), 5) for _, t in ds.items])
    print(f"treino: {len(train)} recortes (vermes por recorte 0..5+: {cnt.tolist()}); validacao {args.val_seq}: {len(val)}")
    sampler = torch.utils.data.WeightedRandomSampler([ds.weight(i) for i in range(len(ds))], num_samples=len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                                         collate_fn=utils.collate_fn, drop_last=True,
                                         persistent_workers=args.workers > 0)

    dev = torch.device("cuda")
    model = build_model(ROOT / args.base).to(dev)
    model.transform.min_size = (int(round(size * SCALE)),)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)
    steps = [int(args.epochs * 0.6), int(args.epochs * 0.85)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=steps, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    f1_0 = validate(model, val, dev, size, mc)
    print(f"F1 de validacao antes do ajuste (modelo base, 3o canal trocado se mask_channel): {f1_0:.3f}")
    best, log = -1.0, [("epoca", "f1_val")]
    for epoch in range(args.epochs):
        model.train()
        model.transform.min_size = (int(round(size * SCALE)),)
        train_one_epoch(model, opt, loader, dev, epoch, 20, scaler)
        sched.step()
        f1 = validate(model, val, dev, size, mc)
        print(f"epoca {epoch}: F1 validacao {f1:.4f}", flush=True)
        log.append((epoch, round(f1, 4)))
        state = dict(model=model.state_dict(), mask_channel=mc, scale=SCALE, epoch=epoch, f1_val=f1)
        torch.save(state, out / "last.pth")
        if f1 > best:
            best = f1
            torch.save(state, out / "checkpoint.pth")
    with open(out / "log.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(log)
    print(f"pronto: melhor F1 de validacao {best:.4f} -> {out / 'checkpoint.pth'}")


if __name__ == "__main__":
    main()
