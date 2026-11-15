"""Confere a CNN: compara com o gabarito e com o watershed, recorte a recorte, e salva figuras.

  python tools/cnn_check.py                          # sequencia de teste, 10 recortes, figura em runs/cnn_check/
  python tools/cnn_check.py --seq 24_1_1_2 --n 16 --worst
Colunas: imagem | mascara do amontoado | gabarito | watershed | CNN   (cada cor = um verme)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from celegans_seg import cnn, load_config  # noqa: E402
from celegans_seg.stages.split import WatershedSplitter  # noqa: E402
from celegans_seg.stages.viz import id_color  # noqa: E402


def colored(img_u8, masks):
    """Imagem em cinza com a borda de cada mascara em uma cor."""
    rgb = np.stack([img_u8] * 3, axis=-1).copy()
    for i, m in enumerate(masks, start=1):
        rgb[m & ~ndi.binary_erosion(m)] = id_color(i)
    return rgb


def labels_to_masks(lab):
    return [lab == i for i in range(1, int(lab.max()) + 1) if (lab == i).any()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default="data/cnn_patches/cnn_dataset")
    ap.add_argument("--weights", default="models/cluster_cnn.pt")
    ap.add_argument("--seq", default="24_2_1_1")
    ap.add_argument("--n", type=int, default=10, help="quantos recortes desenhar")
    ap.add_argument("--worst", action="store_true", help="desenha os de menor F1 da CNN em vez de uma amostra regular")
    ap.add_argument("--out", default="runs/cnn_check")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    patches = [p for p in cnn.load_patches(args.patches, [args.seq]) if len(cnn.targets_in_mask(p))]
    if not patches:
        sys.exit(f"nenhum recorte com gabarito para {args.seq} em {args.patches}")
    sep = cnn.ClusterSeparator(args.weights)
    ws = WatershedSplitter(load_config(args.config), None)

    rows, tot_cnn, tot_ws = [], dict(tp=0, fp=0, fn=0), dict(tp=0, fp=0, fn=0)
    acc_cnn = acc_ws = 0
    for p in patches:
        gt = list(cnn.targets_in_mask(p))
        pc = sep.predict_masks(p)
        pieces = ws._split(p["mask"], max(int(p["meta"].get("n_est", 2)), 2))[0]
        pw = labels_to_masks(pieces) if pieces is not None else [p["mask"]]
        sc, sw = cnn.match_scores(pc, gt), cnn.match_scores(pw, gt)
        for tot, s in ((tot_cnn, sc), (tot_ws, sw)):
            for k in tot:
                tot[k] += s[k]
        acc_cnn += len(pc) == len(gt)
        acc_ws += len(pw) == len(gt)
        rows.append((cnn.f1_from(sc), p, gt, pw, pc))

    n = len(patches)
    print(f"\n{args.seq}: {n} recortes de amontoado com gabarito | vermes no gabarito: {sum(len(r[2]) for r in rows)}")
    print(f"{'metodo':10s} {'F1 (IoU>=0,5)':>14s} {'contagem certa':>15s}")
    print(f"{'watershed':10s} {cnn.f1_from(tot_ws):14.3f} {acc_ws / n:15.2f}")
    print(f"{'CNN':10s} {cnn.f1_from(tot_cnn):14.3f} {acc_cnn / n:15.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.worst:
        order = sorted(range(n), key=lambda i: rows[i][0])[: args.n]
    else:
        order = list(np.linspace(0, n - 1, min(args.n, n)).astype(int))
    fig, axes = plt.subplots(len(order), 5, figsize=(13, 2.7 * len(order)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(order):
        f1, p, gt, pw, pc = rows[i]
        img, ax = p["image"], axes[r]
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(p["mask"], cmap="gray")
        ax[2].imshow(colored(img, gt))
        ax[3].imshow(colored(img, pw))
        ax[4].imshow(colored(img, pc))
        names = ["imagem " + p["file"][:-4], "amontoado", f"gabarito ({len(gt)})", f"watershed ({len(pw)})",
                 f"CNN ({len(pc)})  F1 {f1:.2f}"]
        for a, t in zip(ax, names):
            a.set_title(t, fontsize=9)
            a.axis("off")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.seq}{'_piores' if args.worst else ''}.png"
    fig.savefig(path, dpi=90)
    print("figura:", path)


if __name__ == "__main__":
    main()
