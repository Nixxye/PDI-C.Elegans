"""Visualizador passo a passo do pipeline de PDI (so para entender o que cada etapa faz).

Abre uma janela com a saida de cada estagio lado a lado para um quadro. Todos os paineis
compartilham os eixos: use a lupa/mao da barra do matplotlib para dar zoom num verme e ver a
mesma regiao em todas as etapas. Os estagios evaluate/export/visualize ficam de fora (gravam arquivos).

Exemplos:
  python tools/step_viewer.py --dataset-root data/csb-1_extracted/csb-1_dataset --seq 24_2_1_1
  python tools/step_viewer.py --dataset-root data/csb-1_extracted/csb-1_dataset --seq 24_2_1_1 --frame 120 --zoom 60,260,300,560
  python tools/step_viewer.py --dataset-root data/synthetic --seq seq0 --save etapas.png      # sem janela

Falsos positivos (precisa do gabarito): componentes com menos de 30 % da area sobre um verme do gabarito E QUE NAO ENCOSTAM NA BORDA
do quadro e SEM FORMA DE VERME (o gabarito nao anota vermes cortados pela borda nem alguns vermes inteiros; esses ganham so o rotulo
ciano "sem gabarito (borda)" ou "sem gabarito (verme?)" e NAO sao falso positivo). Tipo 1, ISOLADO:
MAGENTA com o rotulo FP (paineis 9 e 11). Tipo 2, AMONTOADO (vai para a CNN): fica VERMELHO e ganha o rotulo "FP amontoado". Podem ser sujeira ou verme que o gabarito nao anotou.
  --fp : comeca no primeiro quadro (a partir de --frame) que tem falso positivo e da zoom nele

Teclas: setas esq/dir = quadro -1/+1 | cima/baixo = quadro +10/-10 | f = proximo quadro com falso positivo | q = sair
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from celegans_seg import build_pipeline, load_config  # noqa: E402
from celegans_seg.stages.evaluate import classify_no_gt  # noqa: E402
from celegans_seg.stages.viz import CLUSTER_COLOR, id_color  # noqa: E402

FP_COLOR = (255, 0, 255)  # falso positivo: componente sem verme no gabarito (magenta)
REJECT_COLOR = (90, 130, 255)  # descartado pelo filtro de forma (azul)

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


def no_gt_components(pipe, ctx):
    """(falsos positivos, sem gabarito na borda, sem gabarito com forma de verme): componentes com < 30 % da area sobre o
    gabarito, separados por `evaluate.classify_no_gt`. So o primeiro grupo e falso positivo. Vazios se nao ha gabarito."""
    if pipe.env.masks is None or not ctx.components:
        return [], [], [], []
    cat = classify_no_gt(ctx, pipe.env.masks.instances(ctx.name))
    by = {k: [c for c in ctx.components if cat.get(c.id) == k] for k in ("fp", "borda", "verme", "pequeno")}
    return by["fp"], by["borda"], by["verme"], by["pequeno"]


def fp_components(pipe, ctx):
    """Falsos positivos (os dois tipos: isolado e amontoado)."""
    return no_gt_components(pipe, ctx)[0]


def build_panels(pipe, ctx):
    """Lista de (titulo, imagem, cmap) com a saida de cada etapa."""
    a, raw = ctx.artifacts, ctx.raw
    fps, border_no_gt, worm_no_gt, small_no_gt = no_gt_components(pipe, ctx)
    fp_ids = {c.id for c in fps}
    rgb = np.stack([raw] * 3, axis=-1)
    final = rgb
    if ctx.inst_labels is not None:
        final = rgb.copy()
        for inst in ctx.instances:  # amontoado nao separado em vermelho; vermes isolados com cor propria
            m = ctx.inst_labels == inst.id
            edge = m & ~ndi.binary_erosion(m)
            if inst.component_id in fp_ids and inst.source != "cluster":  # FP isolado: contorno grosso magenta
                final[ndi.binary_dilation(edge, iterations=1) & m] = FP_COLOR
                continue
            final[edge] = CLUSTER_COLOR if inst.source == "cluster" else id_color(inst.id)
        if fps or border_no_gt or worm_no_gt or small_no_gt:
            from PIL import Image, ImageDraw
            im = Image.fromarray(final)
            d = ImageDraw.Draw(im)
            for c in border_no_gt:  # sem gabarito, encostado na borda: provavel verme cortado, nao e falso positivo
                d.text((c.centroid[1] + 8, c.centroid[0] - 14), "sem gabarito (borda)", fill=(0, 220, 255))
            for c in worm_no_gt:  # sem gabarito, mas com forma de verme: provavel verme nao anotado, nao e falso positivo
                d.text((c.centroid[1] + 8, c.centroid[0] - 14), "sem gabarito (verme?)", fill=(0, 220, 255))
            for c in small_no_gt:  # sem gabarito, pequeno: "verme pequeno" por definicao (decisao, nao medicao)
                d.text((c.centroid[1] + 8, c.centroid[0] - 14), "sem gabarito (pequeno?)", fill=(0, 220, 255))
            for c in fps:
                d.text((c.centroid[1] + 8, c.centroid[0] - 14), "FP amontoado" if c.is_cluster else "FP", fill=FP_COLOR)
            final = np.asarray(im)
    thr_img = None
    if a.get("mask_raw") is not None:
        base = a.get("mask_temporal") if a.get("mask_temporal") is not None else a["mask_raw"]
        thr_img = np.zeros(raw.shape + (3,), np.uint8)
        thr_img[a["mask_raw"]] = (255, 150, 30)  # laranja = acrescentado pelo top-hat (grow_dist)
        thr_img[base] = (255, 255, 255)
    split = rgb.copy()  # verde = verme isolado, vermelho = amontoado, cores = vermes que a CNN separou
    if ctx.labels is not None:
        split[(ctx.labels > 0) & ~ctx.cluster_mask] = (60, 220, 60)
        split[ctx.cluster_mask] = CLUSTER_COLOR
        for inst in ctx.instances:
            if inst.source in ("cnn",):
                split[ctx.inst_labels == inst.id] = id_color(inst.id)
        for c in fps:
            if not c.is_cluster:  # FP amontoado fica vermelho (so ganha o rotulo no painel 11)
                split[ctx.labels == c.id] = FP_COLOR
    if a.get("rejected_mask") is not None:
        split[a["rejected_mask"]] = REJECT_COLOR
    n_sep = sum(i.source in ("cnn",) for i in ctx.instances)
    gt_img = None
    if pipe.env.masks is not None:
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
        ("9. verde=isolado verm.=amontoado magenta=FP azul=forma", split, None),
        (f"10. instancias: {n_sep} separados pela CNN" if n_sep else "10. instancias: vermes + amontoados", a.get("inst_labels"), "nipy_spectral"),
        (f"11. resultado: {sum(i.source != 'cluster' for i in ctx.instances)} vermes + {n_cl} amontoado(s)"
         + (f" | {sum(not c.is_cluster for c in fps)} FP isolado(s) + {sum(c.is_cluster for c in fps)} FP amontoado(s)" if fps else "")
         + (f" | {len(border_no_gt) + len(worm_no_gt) + len(small_no_gt)} sem gabarito, nao FP (ciano)" if border_no_gt or worm_no_gt or small_no_gt else ""), final, None),
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
    ap.add_argument("--fp", action="store_true", help="comeca no primeiro quadro (a partir de --frame) com falso positivo e da zoom nele")
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

    def next_fp(start):
        """(quadro, componente) do proximo quadro, a partir de `start`, que tem falso positivo."""
        for k in range(start, n):
            fps = fp_components(pipe, pipe.process_frame(k))
            if fps:
                return k, max(fps, key=lambda c: c.area)
        return None, None

    def fp_zoom(c, pad=40):
        r0, c0, r1, c1 = c.bbox
        return max(c0 - pad, 0), c1 + pad, r1 + pad, max(r0 - pad, 0)  # xlim e ylim (y invertido)

    fp_view = None
    if args.fp:
        if pipe.env.masks is None:
            sys.exit("--fp precisa do gabarito (mascaras)")
        k0, comp = next_fp(k0)
        if k0 is None:
            sys.exit("nenhum quadro com falso positivo a partir de --frame")
        fp_view = fp_zoom(comp)
        print(f"falso positivo no quadro {k0}: componente com area {comp.area}, centro {tuple(round(v) for v in comp.centroid)}")

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.08, wspace=0.02, hspace=0.12)
    ims = []

    def show(k, first=False):
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
        n_fp = len(fp_components(pipe, ctx))
        fig.suptitle(f"{ctx.name}  (quadro {k + 1}/{n})" + (f"  |  {n_fp} falso(s) positivo(s)" if n_fp else ""), fontsize=11)
        fig.canvas.draw_idle()

    if args.zoom:
        y0, y1, x0, x1 = (float(v) for v in args.zoom.split(","))
    show(k0, first=True)
    if args.zoom:
        axes[0, 0].set_xlim(x0, x1)
        axes[0, 0].set_ylim(y1, y0)
    elif fp_view:
        axes[0, 0].set_xlim(*fp_view[:2])
        axes[0, 0].set_ylim(*fp_view[2:])

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
        elif e.key == "f":  # proximo quadro com falso positivo, com zoom nele
            k, comp = next_fp(state["k"] + 1)
            if k is not None:
                go(k)
                xl, yl = fp_zoom(comp)[:2], fp_zoom(comp)[2:]
                axes[0, 0].set_xlim(*xl)
                axes[0, 0].set_ylim(*yl)
                fig.canvas.draw_idle()
        elif e.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
