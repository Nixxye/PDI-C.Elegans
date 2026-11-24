"""Processa uma pasta de quadros com o pipeline (PDI + estagio `timeline`, e a CNN se pedida) e grava, em `out_dir`,
o que o player (player/index.html) le:

  data.json        quadros, instancias (trilha, origem, caixa, SSIM) e metricas
  frames/NNNNN.jpg o video
  labels/NNNNN.png mascaras: R,G = id da trilha (16 bits), B = origem (0 isolado, 1 amontoado, 2 CNN)

Usado por player/server.py; tambem roda sozinho:  python player/build.py --images <pasta> --out <saida> [--cnn ckpt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from celegans_seg import build_pipeline, load_config  # noqa: E402
from celegans_seg.stages.viz import id_color  # noqa: E402

SRC = {"single": 0, "cluster": 1, "cnn": 2}
VERSION = 2  # 2 = com as imagens de cada etapa (stages/)
STAGES = ["normalize", "denoise", "background", "threshold", "morphology", "labeling", "analysis", "instances",
          "refine", "timeline", "evaluate"]


def find_masks(images: Path):
    """Gabarito, se a pasta seguir o formato do CSB-1 (.../images/<seq> com .../masks/<seq> ao lado)."""
    m = images.parent.parent / "masks" / images.name
    return str(m) if images.parent.name == "images" and m.is_dir() else ""


def clean(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4) if np.isfinite(v) else None
    return int(v) if isinstance(v, np.integer) else v


def stage_defs(cfg):
    """(id, rotulo, extensao, artefato do pipeline). A etapa "raw" e o proprio quadro (frames/)."""
    g = cfg.threshold.gain
    return [
        ("raw", "Original", "jpg", None),
        ("denoised", f"Suavizada (gaussiano, σ = {cfg.denoise.sigma:g})", "jpg", "denoised"),
        ("signal", f"Sinal = fundo − quadro, ×{g:g}", "jpg", "signal"),
        ("signal_spatial", f"Top-hat, ×{g:g}", "jpg", "signal_spatial"),
        ("mask_temporal", "Limiar de Otsu sobre o sinal", "png", "mask_temporal"),
        ("mask_raw", "+ cauda e verme parado", "png", "mask_raw"),
        ("mask_clean", "Morfologia", "png", "mask_clean"),
        ("labels", "Componentes rotulados", "png", "labels"),
        ("cluster_mask", f"Amontoados", "png", "cluster_mask"),
    ]


def stage_image(kind, art, gain):
    """Artefato do pipeline -> imagem pronta para mostrar (cinza, preto e branco ou colorida)."""
    if kind in ("signal", "signal_spatial"):
        return np.clip(art * 255.0 * gain, 0, 255).astype(np.uint8)
    if kind == "denoised":
        return np.clip(art * 255.0, 0, 255).astype(np.uint8)
    if kind == "labels":
        rgb = np.zeros((*art.shape, 3), np.uint8)
        for k in range(1, int(art.max()) + 1):
            rgb[art == k] = id_color(k)
        return rgb
    return (art > 0).astype(np.uint8) * 255


def build(images, out_dir, cnn="", max_frames=None, progress=lambda done, total: None):
    images, out = Path(images), Path(out_dir)
    ov = [f"input.images={images}", f"output.dir={out}", f"refine.kind={'cnn' if cnn else 'none'}"]
    masks = find_masks(images)
    if masks:
        ov.append(f"input.masks={masks}")
    if cnn:
        ov.append(f"refine.weights={cnn}")
    if max_frames:
        ov.append(f"input.end={max_frames}")
    cfg = load_config(ROOT / "config.yaml", ov)
    cfg.stages = list(STAGES)
    pipe = build_pipeline(cfg)
    total = len(pipe.env.source)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    stages = stage_defs(cfg)
    for sid, _, _, _ in stages[1:]:
        (out / "stages" / sid).mkdir(parents=True, exist_ok=True)
    progress(0, total)
    pipe.fit(cfg.background.n_samples)

    frames = []
    for ctx in pipe.run():
        Image.fromarray(ctx.raw).convert("RGB").save(out / "frames" / f"{ctx.index:05d}.jpg", quality=88)
        rgb = np.zeros((*ctx.inst_labels.shape, 3), np.uint8)
        insts = []
        for i in ctx.instances:
            m = ctx.inst_labels == i.id
            ys, xs = np.nonzero(m)
            if not len(ys):
                continue
            rgb[m] = (i.track & 255, i.track >> 8, SRC.get(i.source, 0))
            insts.append(dict(track=i.track, source=i.source, group=int(i.component_id), area=int(i.area),
                              cy=round(float(i.centroid[0]), 1), cx=round(float(i.centroid[1]), 1),
                              bbox=[int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]))
        Image.fromarray(rgb).save(out / "labels" / f"{ctx.index:05d}.png", optimize=True)
        for sid, _, ext, art in stages[1:]:
            img = Image.fromarray(stage_image(sid, ctx.artifacts[art], cfg.threshold.gain))
            img.save(out / "stages" / sid / f"{ctx.index:05d}.{ext}", **({"quality": 85} if ext == "jpg" else {"optimize": True}))
        frames.append(dict(index=ctx.index, name=ctx.name, instances=insts,
                           metrics={k: clean(ctx.metrics.get(k)) for k in ("n_pred", "n_gt", "precision")}))
        progress(ctx.index + 1, total)

    # SSIM com o quadro anterior (estagio timeline grava timeline.csv ao fim da execucao)
    import csv
    ssim = {}
    with open(out / "timeline.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ssim[(int(r["frame"]), int(r["track"]))] = float(r["ssim"]) if r["ssim"] != "" else None
    for fr in frames:
        for i in fr["instances"]:
            i["ssim"] = ssim.get((fr["index"], i["track"]))
    w, h = Image.open(out / "frames" / "00000.jpg").size
    data = dict(images=str(images), width=w, height=h, has_gt=bool(masks), refine="cnn" if cnn else "none",
                weights=Path(cnn).parent.name if cnn else "", frames=frames,
                stages=[dict(id=s[0], label=s[1], ext=s[2]) for s in stages])
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (out / "ready.json").write_text(json.dumps(dict(version=VERSION)), encoding="utf-8")  # por ultimo: marca "pronto"
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cnn", default="")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    build(a.images, a.out, a.cnn, a.max_frames, lambda d, t: print(f"\rquadro {d}/{t}", end="", file=sys.stderr))
    print("\npronto:", a.out)
