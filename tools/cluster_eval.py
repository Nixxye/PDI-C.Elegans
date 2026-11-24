"""Qualidade da CNN usada SO nos amontoados escolhidos pelo PDI (e comparacao entre versoes da rede).

1. roda o PDI (ate o estagio `export`) e grava um recorte por amontoado, com o gabarito (stages/export.py);
2. cada versao da rede separa os amontoados;
3. compara com o gabarito (duplicatas fundidas) e grava um relatorio + imagens dos piores casos.

Versoes comparaveis (todas avaliadas so dentro da mascara de cada amontoado, com o MESMO gabarito):
  - "recorte": modelos aplicados no recorte de 192 px (`--patch-model nome=checkpoint`), como em `refine.kind=cnn`;
      cnn_base     modelo treinado em quadros inteiros (tools/train_maskrcnn.ps1)
      cnn_cluster  ajustado so nos amontoados, com a mascara do amontoado no 3o canal (tools/cluster_train.py)
      cnn_cluster_gray  idem, mas 3o canal = cinza (--no-mask-channel)
  - "quadro inteiro": predicoes de tools/mrcnn_eval.py (runs/mrcnn_compare/*_<seq>.pkl), recortadas na mascara do
      amontoado:  quadro_base, quadro_base+track, quadro_temporal, quadro_temporal+track.
As versoes que nao existirem sao ignoradas.

Duas perguntas:
  A) a ESCOLHA do PDI: dos componentes marcados como amontoado, quantos tem mesmo 2+ vermes?
  B) a SEPARACAO: nos amontoados com 2+ vermes, cada versao acerta a contagem e as mascaras (F1 a IoU >= 0,5)?

  python tools/cluster_eval.py --seq 24_2_1_1
"""
import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from celegans_seg import build_pipeline, cnn, load_config  # noqa: E402

STAGES = ["normalize", "denoise", "background", "threshold", "morphology", "labeling",
          "analysis", "instances", "export"]
PAL = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
       (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]
MIN_AREA = 20


def dedup(gt, thr=0.5):
    """Funde mascaras quase duplicadas do gabarito (IoU > thr): fica a maior."""
    order = sorted(range(len(gt)), key=lambda i: -gt[i].sum())
    kept = []
    for i in order:
        if all((gt[i] & gt[j]).sum() / max((gt[i] | gt[j]).sum(), 1) <= thr for j in kept):
            kept.append(i)
    return [gt[i] for i in sorted(kept)]


def targets(patch):
    """Vermes do gabarito dentro da mascara do amontoado (duplicatas fundidas, >= MIN_TARGET_AREA px)."""
    m = patch["mask"]
    g = [x & m for x in dedup(list(patch["gt"]))] if len(patch["gt"]) else []
    return [x for x in g if x.sum() >= cnn.MIN_TARGET_AREA]


def window_origin(patch):
    """(y0, x0): posicao do recorte no quadro. O recorte e centrado no amontoado; a caixa dele esta em meta['bbox']."""
    r0, c0, _, _ = patch["meta"]["bbox"]
    ys, xs = np.nonzero(patch["mask"])
    return int(r0 - ys.min()), int(c0 - xs.min())


def crop_rle(rle, y0, x0, S, shape):
    """Mascara (RLE de quadro inteiro) -> recorte S x S na janela (y0, x0); None se nao toca a janela."""
    x, y, w, h = mu.toBbox(rle)
    if x + w <= x0 or x >= x0 + S or y + h <= y0 or y >= y0 + S:
        return None
    full = mu.decode(rle).astype(bool)
    out = np.zeros((S, S), bool)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + S, shape[0]), min(x0 + S, shape[1])
    out[ys - y0:ye - y0, xs - x0:xe - x0] = full[ys:ye, xs:xe]
    return out


def overlay(patch, gts, preds, title):
    """Painel: amontoado | gabarito | rede (contornos coloridos sobre o recorte, ampliado 2x)."""
    img = np.stack([patch["image"]] * 3, -1)
    panels = []
    for masks in ([patch["mask"]], gts, preds):
        im = img.copy()
        for i, m in enumerate(masks):
            col = np.array(PAL[i % len(PAL)])
            pad = np.pad(m, 1)
            edge = m & ~(pad[2:, 1:-1] & pad[:-2, 1:-1] & pad[1:-1, 2:] & pad[1:-1, :-2])
            im[m] = (0.6 * im[m] + 0.4 * col).astype(np.uint8)
            im[edge] = col
        panels.append(Image.fromarray(im).resize((im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST))
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height + 14), (255, 255, 255))
    x = 0
    for p, t in zip(panels, ["amontoado do PDI", f"gabarito: {len(gts)}", f"{title}: {len(preds)}"]):
        canvas.paste(p, (x, 14))
        ImageDraw.Draw(canvas).text((x + 3, 1), t, fill=(0, 0, 0))
        x += p.width
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-model", action="append", default=[], metavar="NOME=CHECKPOINT",
                    help="modelo aplicado no recorte (repetivel). Padrao: cnn_base e, se existir, cnn_cluster")
    ap.add_argument("--no-frame-models", action="store_true", help="ignora as predicoes de quadro inteiro")
    ap.add_argument("--dataset-root", default="data/csb-1_extracted/csb-1_dataset")
    ap.add_argument("--seq", default="24_2_1_1")
    ap.add_argument("--step", type=int, default=2, help="1 a cada N quadros")
    ap.add_argument("--out", default="runs/cluster_eval")
    ap.add_argument("--worst", type=int, default=16, help="quantos piores casos desenhar (por versao)")
    ap.add_argument("--reuse", action="store_true", help="reusa os recortes ja gerados")
    ap.add_argument("--set", nargs="*", default=[], help="overrides do config (ex.: analysis.cluster_ratio=2)")
    args = ap.parse_args()
    out = Path(args.out) / args.seq
    out.mkdir(parents=True, exist_ok=True)
    patches_dir = out / "patches"

    if not (args.reuse and (patches_dir / "cnn_dataset" / args.seq).exists()):
        t0 = time.time()
        cfg = load_config("config.yaml", [f"input.dataset_root={args.dataset_root}", f"input.sequence={args.seq}",
                                          f"input.step={args.step}", f"output.dir={patches_dir}", *args.set])
        cfg.stages = STAGES
        pipe = build_pipeline(cfg)
        pipe.fit(cfg.background.n_samples)
        n = sum(1 for _ in pipe.run())
        print(f"PDI: {n} quadros em {time.time() - t0:.0f}s", flush=True)
    patches = cnn.load_patches(patches_dir / "cnn_dataset", seqs=[args.seq], require_gt=True)
    fits = [p for p in patches if p["meta"].get("fits", True)]
    print(f"{len(patches)} amontoados do PDI ({len(patches) - len(fits)} maiores que o recorte, fora da avaliacao)")
    S = fits[0]["image"].shape[0]
    gts_all = [targets(p) for p in fits]

    # ---- versoes: nome -> funcao(patch) -> lista de mascaras (S,S) dentro do amontoado
    versions = {}
    pm = dict(a.split("=", 1) for a in args.patch_model)
    if not pm:
        pm = {"cnn_base": "runs/cnn_train/checkpoint.pth"}
        for nm, d in (("cnn_cluster", "runs/cnn_cluster"), ("cnn_cluster_gray", "runs/cnn_cluster_gray")):
            if (ROOT / d / "checkpoint.pth").exists():
                pm[nm] = f"{d}/checkpoint.pth"
    for name, ck in pm.items():
        if not Path(ck).is_file():
            print(f"(ignorado {name}: {ck} nao existe)")
            continue
        sep = cnn.ClusterSeparator(ck)
        versions[name] = lambda p, sep=sep: [m for m in sep.predict_masks(p) if m.sum() >= MIN_AREA]
    if not args.no_frame_models:
        import mrcnn_common as C
        ks = C.frames_of(args.seq)
        shape = C.load_gray(args.seq, ks[0]).shape
        for src in ("base", "temporal"):
            pkl = ROOT / "runs" / "mrcnn_compare" / f"{src}_{args.seq}.pkl"
            if not pkl.exists():
                continue
            raw = pickle.load(open(pkl, "rb"))
            for suffix, frames in (("", C.plain(raw)), ("+track", C.run_tracker(raw))):
                def run(p, frames=frames):
                    k = int(p["meta"]["name"])
                    y0, x0 = window_origin(p)
                    res = []
                    for d in frames[ks.index(k)]:
                        c = crop_rle(d["rle"], y0, x0, S, shape)
                        if c is not None:
                            c &= p["mask"]
                            if c.sum() >= MIN_AREA:
                                res.append(c)
                    return res
                versions[f"quadro_{src}{suffix}"] = run
    print("versoes:", ", ".join(versions))

    rows = defaultdict(list)
    for name, fn in versions.items():
        t0 = time.time()
        for k, p in enumerate(fits):
            preds = fn(p)
            gts = gts_all[k]
            sc = cnn.match_scores(preds, gts) if gts else dict(tp=0, fp=len(preds), fn=0)
            rows[name].append(dict(patch=p, gts=gts, preds=preds, n_gt=len(gts), n_pred=len(preds), sc=sc))
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    one = [cnn.match_scores([p["mask"]], g) if g else dict(tp=0, fp=1, fn=0) for p, g in zip(fits, gts_all)]
    N = len(fits)

    # ---- A) escolha do PDI
    by = defaultdict(int)
    for g in gts_all:
        by[min(len(g), 4)] += 1
    pct = lambda n: f"{100 * n / max(N, 1):.1f}"  # noqa: E731
    L = [f"# Amontoados do PDI separados pela rede ({args.seq}, 1 quadro a cada {args.step})\n",
         f"Amontoados avaliados: **{N}** (+ {len(patches) - len(fits)} maiores que o recorte de {S} px, nao avaliados).\n",
         "## A) A escolha do PDI: quantos vermes tem cada amontoado?\n",
         "| vermes no gabarito dentro do amontoado | amontoados | % |", "|---|---|---|",
         f"| 0 (falso amontoado) | {by[0]} | {pct(by[0])} |", f"| 1 (desnecessario) | {by[1]} | {pct(by[1])} |",
         f"| 2 | {by[2]} | {pct(by[2])} |", f"| 3 | {by[3]} | {pct(by[3])} |", f"| 4 ou mais | {by[4]} | {pct(by[4])} |",
         f"\n**Escolha certa (2+ vermes): {pct(by[2] + by[3] + by[4])} %.**\n"]

    # ---- B) comparacao
    def agg(sel):
        tp = sum(r["sc"]["tp"] for r in sel); fp = sum(r["sc"]["fp"] for r in sel); fn = sum(r["sc"]["fn"] for r in sel)
        return tp, fp, fn, cnn.f1_from(dict(tp=tp, fp=fp, fn=fn))

    groups = [("2", lambda r: r["n_gt"] == 2), ("3", lambda r: r["n_gt"] == 3), ("4+", lambda r: r["n_gt"] >= 4),
              ("**2+**", lambda r: r["n_gt"] >= 2), ("1 (desnecessario)", lambda r: r["n_gt"] == 1)]
    names = list(versions)
    L += ["## B) Comparacao das versoes (so dentro dos amontoados; F1 por verme a IoU >= 0,5 / contagem exata)\n",
          "| vermes no gabarito | amontoados | sem rede (1 verme) | " + " | ".join(names) + " |",
          "|---|---|---|" + "---|" * len(names)]
    for gname, cond in groups:
        idx = [i for i in range(N) if cond(rows[names[0]][i])]
        if not idx:
            continue
        f1o = cnn.f1_from(dict(tp=sum(one[i]["tp"] for i in idx), fp=sum(one[i]["fp"] for i in idx),
                               fn=sum(one[i]["fn"] for i in idx)))
        cells = []
        for n in names:
            sel = [rows[n][i] for i in idx]
            ex = 100 * sum(r["n_gt"] == r["n_pred"] for r in sel) / len(sel)
            cells.append(f"{agg(sel)[3]:.3f} / {ex:.0f} %")
        L.append(f"| {gname} | {len(idx)} | {f1o:.3f} | " + " | ".join(cells) + " |")
    L += ["", "### Recall e precisao nos amontoados com 2+ vermes\n", "| versao | recall | precisao | F1 | verme a mais (FP) | verme a menos (FN) |",
          "|---|---|---|---|---|---|"]
    for n in names:
        tp, fp, fn, f1 = agg([r for r in rows[n] if r["n_gt"] >= 2])
        L.append(f"| {n} | {tp / max(tp + fn, 1):.3f} | {tp / max(tp + fp, 1):.3f} | {f1:.3f} | {fp} | {fn} |")

    L += ["", "### Contagem: gabarito (linhas) x versao (colunas), amontoados com 2+ vermes\n"]
    for n in names:
        L += [f"**{n}**\n", "| gabarito \\ prevista | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |", "|" + "---|" * 8]
        for g in range(2, 6):
            sel = [r for r in rows[n] if min(r["n_gt"], 5) == g and r["n_gt"] >= 2]
            if sel:
                L.append(f"| {g if g < 5 else '5+'} | " + " | ".join(
                    str(sum(min(r["n_pred"], 6) == c for r in sel)) for c in range(7)) + " |")
        L.append("")

    # ---- piores casos de cada versao
    for n in names:
        wdir = out / f"piores_{n}"
        wdir.mkdir(exist_ok=True)
        for f in wdir.glob("*.png"):
            f.unlink()
        bad = sorted((r for r in rows[n] if r["n_gt"] >= 2),
                     key=lambda r: (-(r["sc"]["fp"] + r["sc"]["fn"]), -r["n_gt"]))
        for i, r in enumerate(bad[: args.worst]):
            overlay(r["patch"], r["gts"], r["preds"], n).save(
                wdir / f"{i:02d}_{r['patch']['file'][:-4]}_gt{r['n_gt']}_pred{r['n_pred']}.png")
    L += ["Os piores casos de cada versao (mais verme a mais + a menos) estao em `piores_<versao>/`: "
          "amontoado | gabarito | versao.", ""]
    md = "\n".join(L)
    (out / "relatorio.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print("salvo em", out / "relatorio.md")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
