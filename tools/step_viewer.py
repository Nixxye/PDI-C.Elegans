"""Visualizador passo a passo do pipeline de PDI (so para entender o que cada etapa faz).

Abre uma janela com a saida de cada estagio lado a lado para um quadro. Todos os paineis
compartilham os eixos: use a lupa/mao da barra do matplotlib para dar zoom num verme e ver a
mesma regiao em todas as etapas. Os estagios evaluate/export/visualize ficam de fora (gravam arquivos).

Exemplos:
  python tools/step_viewer.py --dataset-root data/csb-1_extracted/csb-1_dataset --seq 24_2_1_1
  python tools/step_viewer.py --dataset-root data/csb-1_extracted/csb-1_dataset --seq 24_2_1_1 --frame 120 --zoom 60,260,300,560
  python tools/step_viewer.py --dataset-root data/synthetic --seq seq0 --save etapas.png      # sem janela

Teclas: setas esq/dir = quadro -1/+1 | cima/baixo = quadro +10/-10 | q = sair
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from celegans_seg import build_pipeline, load_config  # noqa: E402
from celegans_seg.stages.viz import CLUSTER_COLOR, id_color  # noqa: E402

# estagios com estado entre quadros ou efeitos colaterais (gravam arquivos): fora do visualizador
SKIP = {"evaluate", "export", "visualize"}


def contours(rgb, label_img, color_fn=None):
    """Pinta a borda de cada rotulo sobre uma copia da imagem RGB."""
    out = rgb.copy()
    for i in np.unique(label_img):
        if i == 0:
            continue
        m = label_img == i
        out[m & ~ndi.binary_erosion(m)] = color_fn(int(i)) if color_fn else (255, 255, 0)
    return out


def build_panels(pipe, ctx):
    """Lista de (titulo, imagem, cmap) com a saida de cada etapa."""
    a, raw = ctx.artifacts, ctx.raw
    rgb = np.stack([raw] * 3, axis=-1)
    final = rgb
    if ctx.inst_labels is not None:
        final = rgb.copy()
        for inst in ctx.instances:  # amontoado nao separado em vermelho; vermes isolados com cor propria
            m = ctx.inst_labels == inst.id
            final[m & ~ndi.binary_erosion(m)] = CLUSTER_COLOR if inst.source == "cluster" else id_color(inst.id)
    thr_img = None
    if a.get("mask_raw") is not None:
        base = a.get("mask_temporal") if a.get("mask_temporal") is not None else a["mask_raw"]
        thr_img = np.zeros(raw.shape + (3,), np.uint8)
        thr_img[a["mask_raw"]] = (255, 150, 30)  # laranja = acrescentado pelo top-hat (grow_dist)
        thr_img[base] = (255, 255, 255)
    split = rgb.copy()  # verde = verme isolado, vermelho = amontoado, cores = vermes que a CNN/watershed separou
    if ctx.labels is not None:
        split[(ctx.labels > 0) & ~ctx.cluster_mask] = (60, 220, 60)
        split[ctx.cluster_mask] = CLUSTER_COLOR
        for inst in ctx.instances:
            if inst.source in ("cnn", "watershed"):
                split[ctx.inst_labels == inst.id] = id_color(inst.id)
    n_sep = sum(i.source in ("cnn", "watershed") for i in ctx.instances)
    gt_img = None
    if pipe.env.masks is not None:
        gt = np.zeros(raw.shape, np.int32)
        edges = rgb.copy()
        for i, m in enumerate(pipe.env.masks.instances(ctx.name), start=1):
            edges[m & ~ndi.binary_erosion(m)] = id_color(i)  # sobreposicao preservada: cada verme tem a sua borda
        gt_img = edges
    n_cl = sum(c.is_cluster for c in ctx.components)
    return [
        ("1. original", raw, "gray"),
        ("2. normalizado [Aula 03]", a.get("image"), "gray"),
        ("3. suavizado [Aula 04]", a.get("denoised"), "gray"),
        ("4. sinal apos o fundo (claro = verme)", a.get("signal"), "gray"),
        ("5. limiar: Otsu (branco) + completado pelo top-hat (laranja)", thr_img, None),
        ("6. morfologia [Aula 08]", a.get("mask_clean"), "gray"),
        (f"7. rotulos: {len(ctx.components)} comp., {n_cl} amontoado(s)", a.get("labels"), "nipy_spectral"),
        ("8. mascara de amontoados (vai para a CNN)", a.get("cluster_mask"), "gray"),
        ("9. isolados (verde), amontoados (vermelho)" + (", separados (cores)" if n_sep else ""), split, None),
        (f"10. instancias: {n_sep} separados pela CNN/watershed" if n_sep else "10. instancias: vermes + amontoados", a.get("inst_labels"), "nipy_spectral"),
        (f"11. resultado: {sum(i.source != 'cluster' for i in ctx.instances)} vermes + {n_cl} amontoado(s)", final, None),
        ("12. gabarito (se houver)", gt_img, None),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--dataset-root")
    ap.add_argument("--seq")
    ap.add_argument("--frame", type=int, default=0, help="indice do quadro inicial")
    ap.add_argument("--zoom", help="regiao inicial: y0,y1,x0,x1 (pixels)")
    ap.add_argument("--save", help="salva o painel do quadro --frame neste PNG e sai (sem janela)")
    ap.add_argument("--set", action="append", default=[], metavar="secao.campo=valor")
    args = ap.parse_args()

    ov = list(args.set)
    for key, val in (("input.images", args.images), ("input.masks", args.masks),
                     ("input.dataset_root", args.dataset_root), ("input.sequence", args.seq)):
        if val:
            ov.append(f"{key}={val}")
    cfg = load_config(args.config, ov)
    cfg.stages = [s for s in cfg.stages if s not in SKIP]

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    pipe = build_pipeline(cfg)
    n = len(pipe.env.source)
    print(f"{n} quadros. Ajustando fundo/area de referencia...", flush=True)
    pipe.fit(cfg.background.n_samples)
    k0 = int(np.clip(args.frame, 0, n - 1))

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.08, wspace=0.02, hspace=0.12)
    ims = []

    def show(k, first=False):
        if k > 0 and cfg.refine.kind == "cnn":
            pipe.process_frame(k - 1)  # o refine temporal usa a saida do quadro anterior
        ctx = pipe.process_frame(k)
        for idx, (ax, (title, img, cmap)) in enumerate(zip(axes.ravel(), build_panels(pipe, ctx))):
            if img is None:
                img = np.zeros(ctx.raw.shape, np.uint8)
            if cmap == "nipy_spectral":
                img = np.ma.masked_equal(img, 0)
                cmap = plt.get_cmap(cmap).copy()
                cmap.set_bad("black")  # fundo (rotulo 0) preto, como as outras etapas
            if first:
                kw = {"cmap": cmap} if cmap else {}
                if cmap == "gray":  # escala fixa: sem isso o imshow estica o contraste de cada painel
                    kw.update(vmin=0, vmax=255 if img.dtype == np.uint8 else (0.5 if "sinal" in title else 1))
                ims.append(ax.imshow(img, interpolation="nearest", **kw))
                ax.axis("off")
            else:
                ims[idx].set_data(img)
                if cmap == "magma" or isinstance(cmap, matplotlib.colors.Colormap):
                    ims[idx].autoscale()
            ax.set_title(title, fontsize=9)
        fig.suptitle(f"{ctx.name}  (quadro {k + 1}/{n})", fontsize=11)
        fig.canvas.draw_idle()

    if args.zoom:
        y0, y1, x0, x1 = (float(v) for v in args.zoom.split(","))
    show(k0, first=True)
    if args.zoom:
        axes[0, 0].set_xlim(x0, x1)
        axes[0, 0].set_ylim(y1, y0)

    if args.save:
        fig.savefig(args.save, dpi=100)
        print("salvo em", args.save)
        return

    state = {"k": k0, "busy": False}
    ax_s = fig.add_axes([0.2, 0.02, 0.6, 0.025])
    slider = Slider(ax_s, "quadro", 0, n - 1, valinit=k0, valstep=1, valfmt="%d")

    def go(k):
        k = int(np.clip(k, 0, n - 1))
        if state["busy"] or k == state["k"]:
            return
        state["busy"], state["k"] = True, k
        show(k)
        if int(slider.val) != k:
            slider.set_val(k)
        state["busy"] = False

    slider.on_changed(lambda v: go(int(v)))

    def on_key(e):
        step = {"right": 1, "left": -1, "up": 10, "down": -10}.get(e.key)
        if step:
            go(state["k"] + step)
        elif e.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
