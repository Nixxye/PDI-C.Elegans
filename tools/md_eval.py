"""Testa as redes treinadas no CSB-1 no Mating Dataset (WormSwin, Zenodo 7456803, md_dataset.zip): outro video, outro
microscopio e outros anotadores, com bastante sobreposicao. Serve para ver se o resultado vale fora do CSB-1.

  python tools/md_eval.py [--split test] [--limit N]

Gabarito = `<split>_annotations.json` (COCO) de data/md_extracted/md_dataset (o json omite algumas mascaras que existem em
masks/*.png; usamos o json, que e o que o treino de referencia usa).

"Amontoado" aqui = componente conexo da uniao das mascaras do gabarito (dilatada 1 px) com 2+ vermes. Isto NAO e a escolha
do PDI (o PDI precisa de video para estimar o fundo; os quadros do MD nao sao consecutivos): e o amontoado ideal.
Tabelas: (1) quadro inteiro, todos os vermes; (2) so nos amontoados (versoes aplicadas no quadro inteiro e as
aplicadas no recorte de 192 px centrado no amontoado).
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mrcnn_common as C  # noqa: E402
from celegans_seg import cnn  # noqa: E402

MD = ROOT / "data" / "md_extracted" / "md_dataset"
S = 192
MIN_AREA = 20


def crop(arr, cy, cx, size=S):
    out = np.zeros((size, size) + arr.shape[2:], arr.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, arr.shape[0]), min(x0 + size, arr.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = arr[ys:ye, xs:xe]
    return out


def dedup(gt, thr=0.5):
    order = sorted(range(len(gt)), key=lambda i: -gt[i].sum())
    kept = []
    for i in order:
        if all((gt[i] & gt[j]).sum() / max((gt[i] | gt[j]).sum(), 1) <= thr for j in kept):
            kept.append(i)
    return [gt[i] for i in sorted(kept)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base", default="runs/cnn_train/checkpoint.pth")
    ap.add_argument("--patch-model", action="append", default=[], metavar="NOME=CKPT")
    ap.add_argument("--out", default="runs/md_eval")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(MD / f"{args.split}_annotations.json"))
    ids = sorted(coco.getImgIds())[: args.limit]
    dev = torch.device("cuda")
    model = C.load_model(args.base, dev)
    pm = dict(a.split("=", 1) for a in args.patch_model)
    if not pm:
        for nm, d in (("cnn_base", args.base), ("cnn_cluster", "runs/cnn_cluster/checkpoint.pth"),
                      ("cnn_cluster_gray", "runs/cnn_cluster_gray/checkpoint.pth")):
            if Path(d).is_file():
                pm[nm] = d
    seps = {n: cnn.ClusterSeparator(ck) for n, ck in pm.items()}

    gt_rles, pred_frames = [], []
    clusters = []  # (image, mask do amontoado, gts, preds_quadro)
    t0 = time.time()
    for n, iid in enumerate(ids):
        info = coco.loadImgs([iid])[0]
        gray = np.array(Image.open(MD / "images" / info["file_name"]).convert("L"))
        gts = dedup([coco.annToMask(a).astype(bool) for a in coco.loadAnns(coco.getAnnIds(imgIds=[iid]))])
        dets = [d for d in C.detect(model, C.make_input(gray, None), dev) if d["score"] >= C.SCORE_THR]
        gt_rles.append(C.rles(gts))
        pred_frames.append(C.plain([dets])[0])
        # amontoados do gabarito: componentes da uniao (dilatada) com 2+ vermes
        if gts:
            union = np.zeros(gray.shape, bool)
            for g in gts:
                union |= g
            lab, nl = ndi.label(ndi.binary_dilation(union, iterations=1))
            for c in range(1, nl + 1):
                comp = (lab == c) & union
                inside = [g & comp for g in gts if (g & comp).sum() >= cnn.MIN_TARGET_AREA]
                if len(inside) >= 2:
                    ys, xs = np.nonzero(comp)
                    fits = (ys.max() - ys.min() < S) and (xs.max() - xs.min() < S)
                    clusters.append(dict(gray=gray, comp=comp, gts=inside, dets=dets, fits=fits,
                                         c=(int(ys.mean()), int(xs.mean()))))
        if n % 40 == 0:
            print(f"  {n}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)

    r = C.evaluate(pred_frames, gt_rles)
    L = [f"# Mating Dataset ({args.split}): {len(ids)} imagens, {r['vermes_gt']} vermes no gabarito\n",
         "## 1) Quadro inteiro, todos os vermes (rede base, treinada so no CSB-1)\n",
         "| F1@0,5 | precisao | recall | F1@0,75 | IoU medio | FP | FN |", "|---|---|---|---|---|---|---|",
         f"| {r['f1']:.3f} | {r['precisao']:.3f} | {r['recall']:.3f} | {r['f1_075']:.3f} | {r['iou_medio']:.3f} | "
         f"{r['fp']} | {r['fn']} |\n"]

    # ---- 2) amontoados
    fit = [c for c in clusters if c["fits"]]
    by = defaultdict(int)
    for c in fit:
        by[min(len(c["gts"]), 4)] += 1
    L += [f"## 2) Amontoados do gabarito com 2+ vermes: {len(clusters)} ({len(clusters) - len(fit)} maiores que {S} px, fora)\n",
          "Vermes por amontoado: 2: %d, 3: %d, 4+: %d\n" % (by[2], by[3], by[4])]

    def scores(fn, windowed):
        res = []
        for c in fit:
            preds = [m for m in fn(c) if m.sum() >= MIN_AREA]
            gts = [crop(g, *c["c"]) for g in c["gts"]] if windowed else c["gts"]  # recorte: gabarito na mesma janela
            res.append((len(c["gts"]), len(preds), cnn.match_scores(preds, gts)))
        return res

    versions = {}
    versions["quadro_base"] = lambda c: [m & c["comp"] for m in
                                         (C.mu.decode(d["rle"]).astype(bool) for d in c["dets"]) if (m & c["comp"]).any()]
    for n, sep in seps.items():
        def f(c, sep=sep):
            cy, cx = c["c"]
            patch = dict(image=crop(c["gray"], cy, cx), mask=crop(c["comp"], cy, cx))
            return sep.predict_masks(patch)
        versions[n + "_recorte"] = f
    names = list(versions)
    allres = {n: scores(versions[n], n.endswith("_recorte")) for n in names}
    one = [cnn.match_scores([c["comp"]], c["gts"]) for c in fit]

    def f1_of(items):
        return cnn.f1_from(dict(tp=sum(i[2]["tp"] for i in items), fp=sum(i[2]["fp"] for i in items),
                                fn=sum(i[2]["fn"] for i in items)))

    L += ["### F1 por verme (IoU >= 0,5) / contagem exata\n",
          "| vermes | amontoados | sem rede | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for gname, cond in (("2", lambda k: k == 2), ("3", lambda k: k == 3), ("4+", lambda k: k >= 4), ("**2+**", lambda k: True)):
        idx = [i for i, c in enumerate(fit) if cond(len(c["gts"]))]
        if not idx:
            continue
        f1o = cnn.f1_from(dict(tp=sum(one[i]["tp"] for i in idx), fp=sum(one[i]["fp"] for i in idx),
                               fn=sum(one[i]["fn"] for i in idx)))
        cells = []
        for n in names:
            sel = [allres[n][i] for i in idx]
            cells.append(f"{f1_of(sel):.3f} / {100 * sum(a == b for a, b, _ in sel) / len(sel):.0f} %")
        L.append(f"| {gname} | {len(idx)} | {f1o:.3f} | " + " | ".join(cells) + " |")
    L += ["", "### Excesso e falta de vermes (todos os amontoados com 2+)\n", "| versao | recall | precisao | verme a mais | verme a menos |",
          "|---|---|---|---|---|"]
    for n in names:
        tp = sum(i[2]["tp"] for i in allres[n]); fp = sum(i[2]["fp"] for i in allres[n]); fn = sum(i[2]["fn"] for i in allres[n])
        L.append(f"| {n} | {tp / max(tp + fn, 1):.3f} | {tp / max(tp + fp, 1):.3f} | {fp} | {fn} |")
    md = "\n".join(L) + "\n"
    (out / f"relatorio_{args.split}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
