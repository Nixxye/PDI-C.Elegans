"""Gera uma interface (pagina HTML) para escolher um verme e acompanhar a mascara dele pelo video.

Roda o pipeline (incluindo o estagio `timeline`) e grava em `--out` (padrao runs/viewer):
  index.html  a interface (abra com duplo clique; nao precisa de servidor)
  data.js     mascaras (uma imagem por quadro, com o id da trilha em cada pixel), instancias e metricas
  frames/     os quadros do video em JPEG

Na interface: clique num verme (ou escolha na lista) para seguir a trilha dele; setas / espaco / barra percorrem o video.
Painel: metricas do quadro atual (Dice, precisao, recall, F1... se houver gabarito) e do verme escolhido (area, posicao,
SSIM/MSE/PSNR com o quadro anterior, IoU com o gabarito) mais graficos dele ao longo dos quadros.

  python tools/timeline_viewer.py --dataset-root data/csb-1_extracted/csb-1_dataset --seq 24_2_1_1 --max-frames 150
  python tools/timeline_viewer.py ... --set refine.kind=cnn --set refine.weights=runs/cnn_cluster/checkpoint.pth
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from celegans_seg import build_pipeline, load_config  # noqa: E402
from celegans_seg.stages.evaluate import iou_matrix  # noqa: E402

SRC = {"single": 0, "cluster": 1, "cnn": 2}
STAGES = ["normalize", "denoise", "background", "threshold", "morphology", "labeling", "analysis", "instances",
          "refine", "timeline", "evaluate"]


def label_png(ctx) -> str:
    """PNG sem perdas: R,G = id da trilha (16 bits), B = origem (0 isolado, 1 amontoado, 2 CNN)."""
    h, w = ctx.inst_labels.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for i in ctx.instances:
        m = ctx.inst_labels == i.id
        rgb[m, 0] = i.track & 255
        rgb[m, 1] = i.track >> 8
        rgb[m, 2] = SRC.get(i.source, 0)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset-root")
    ap.add_argument("--seq")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", default="runs/viewer")
    ap.add_argument("--set", action="append", default=[], metavar="secao.campo=valor")
    args = ap.parse_args()

    ov = list(args.set)
    for key, val in (("input.images", args.images), ("input.masks", args.masks),
                     ("input.dataset_root", args.dataset_root), ("input.sequence", args.seq)):
        if val:
            ov.append(f"{key}={val}")
    if args.max_frames:
        ov.append(f"input.end={args.max_frames}")
    out = Path(args.out)
    ov.append(f"output.dir={out}")
    cfg = load_config(ROOT / args.config, ov)
    cfg.stages = list(STAGES)
    pipe = build_pipeline(cfg)
    n = len(pipe.env.source)
    print(f"{n} quadros | gabarito: {'sim' if pipe.env.masks else 'nao'} | saida: {out}")
    (out / "frames").mkdir(parents=True, exist_ok=True)
    pipe.fit(cfg.background.n_samples)

    frames, t0 = [], time.time()
    for ctx in pipe.run():
        gt = pipe.env.masks.instances(ctx.name) if pipe.env.masks else []
        masks = {i.id: ctx.inst_labels == i.id for i in ctx.instances}
        ious = {}
        if gt and masks:
            M = iou_matrix(list(masks.values()), gt)
            ious = {iid: float(M[k].max()) for k, iid in enumerate(masks)}
        Image.fromarray(ctx.raw).convert("RGB").save(out / "frames" / f"{ctx.index:05d}.jpg", quality=88)
        trow = {}
        insts = []
        for i in ctx.instances:
            ys, xs = np.nonzero(masks[i.id])
            if not len(ys):
                continue
            insts.append(dict(track=i.track, source=i.source, group=int(i.component_id), area=int(i.area), cy=round(float(i.centroid[0]), 1),
                              cx=round(float(i.centroid[1]), 1), bbox=[int(ys.min()), int(xs.min()), int(ys.max()),
                                                                       int(xs.max())],
                              iou_gt=clean(ious.get(i.id)) if gt else None))
        frames.append(dict(index=ctx.index, name=ctx.name, label=label_png(ctx), instances=insts,
                           metrics={k: clean(v) for k, v in ctx.metrics.items()
                                    if k not in ("frame", "name") and not isinstance(v, str)}))
        print(f"\rquadro {ctx.index + 1}/{n}", end="", file=sys.stderr)

    # SSIM/MSE/PSNR com o quadro anterior: vem do timeline.csv gravado no fim da execucao
    import csv
    sims = {}
    with open(out / "timeline.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sims[(int(r["frame"]), int(r["track"]))] = {k: (float(r[k]) if r[k] != "" else None)
                                                         for k in ("ssim", "mse", "psnr")}
    for fr in frames:
        for i in fr["instances"]:
            i.update(sims.get((fr["index"], i["track"]), {}))
    h, w = Image.open(out / "frames" / "00000.jpg").size[::-1]
    data = dict(seq=cfg.input.sequence or "", refine=cfg.refine.kind,
                weights=Path(cfg.refine.weights).parent.name if cfg.refine.kind == "cnn" else "", width=w, height=h, has_gt=pipe.env.masks is not None, frames=frames)
    (out / "data.js").write_text("const DATA = " + json.dumps(data, ensure_ascii=False) + ";", encoding="utf-8")
    (out / "index.html").write_text((Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8"),
                                    encoding="utf-8")
    print(f"\npronto em {time.time() - t0:.0f}s -> abra {out / 'index.html'}")


if __name__ == "__main__":
    main()
