"""Relatorio de acuracia das mascaras do PDI (sem CNN), contra o gabarito.

Roda so os estagios de PDI (normalize ... analysis) em cada sequencia e compara com as mascaras por verme.

  python tools/pdi_report.py                                   # todas as sequencias, 1 quadro a cada 10
  python tools/pdi_report.py --seq 24_2_1_1 --step 2           # uma sequencia, mais quadros
  python tools/pdi_report.py --set threshold.spatial_min_area=0    # avalia outra configuracao

Metricas (por verme do gabarito):
  - ENCONTRADO: >= --found-thr (50 %) da area do verme cai dentro da mascara do PDI (qualquer componente).
  - Cada verme encontrado pertence ao componente que mais o cobre. Se esse componente foi marcado como
    AMONTOADO, o verme e "em amontoado"; senao e "isolado".
  - ACERTO DA MASCARA (so vermes isolados): IoU entre a mascara do verme e a do componente >= --hit-iou (50 %).
    Se o componente tem outro verme dentro (amontoado que o PDI nao detectou), o IoU cai e conta como erro.
Tambem conta componentes isolados x amontoados e os FALSOS POSITIVOS (componente com < 30 % da area sobre um verme
do gabarito), em dois tipos: tipo 1 = marcado como isolado; tipo 2 = marcado como amontoado (vai para a CNN, mas quase
nao tem verme). NAO sao falso positivo (linhas a parte): componente que ENCOSTA NA BORDA do quadro (o gabarito nao anota
os vermes cortados pela borda) e componente COM FORMA DE VERME (provavel verme que o gabarito nao anotou; ver
evaluate.classify_no_gt).

O CSB-1 tem mascaras de gabarito DUPLICADAS (duas mascaras quase iguais para o mesmo verme). Por padrao elas
sao fundidas (IoU > 0,5 entre duas mascaras: fica a maior); use --no-dedup para ver os numeros crus.
"""
import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from celegans_seg import build_pipeline, load_config  # noqa: E402
from celegans_seg.config import EvaluateCfg  # noqa: E402
from celegans_seg.stages.evaluate import classify_no_gt, gt_fraction  # noqa: E402

SMALL_WORM_MIN_AREA, SMALL_WORM_MIN_CONTRAST = EvaluateCfg().small_min_area, EvaluateCfg().small_min_contrast

STAGES = ["normalize", "denoise", "background", "threshold", "morphology", "labeling", "analysis"]
MIN_GT_AREA = 30  # mascaras de gabarito menores que isso sao ignoradas


def dedup(gt, thr=0.5):
    """Funde mascaras quase duplicadas (IoU > thr): fica a maior. Devolve (lista, n removidas)."""
    order = sorted(range(len(gt)), key=lambda i: -gt[i].sum())
    kept = []
    for i in order:
        if all((gt[i] & gt[j]).sum() / max((gt[i] | gt[j]).sum(), 1) <= thr for j in kept):
            kept.append(i)
    return [gt[i] for i in sorted(kept)], len(gt) - len(kept)


class Counts(collections.Counter):
    """Acumulador: soma de contagens, listas de IoU/Dice e lista de componentes sem verme no gabarito."""

    def __init__(self):
        super().__init__()
        self.iou, self.dice = [], []
        self.fp_list = []  # componentes sem verme no gabarito (para o CSV e para abrir no visualizador)
        self.seq = ""
        self.gallery = []  # recortes dos componentes das categorias pedidas em --galeria-categorias
        self.gallery_cats = ()
        self.small_min_area, self.small_min_contrast = SMALL_WORM_MIN_AREA, SMALL_WORM_MIN_CONTRAST


def crop_box(bbox, shape, pad=40, min_size=150):
    """Janela (y0, y1, x0, x1) em volta da caixa do componente, com folga, dentro do quadro."""
    r0, c0, r1, c1 = bbox
    h = min(max(r1 - r0 + 2 * pad, min_size), shape[0])
    w = min(max(c1 - c0 + 2 * pad, min_size), shape[1])
    y0 = int(max(0, min((r0 + r1) / 2 - h / 2, shape[0] - h)))
    x0 = int(max(0, min((c0 + c1) / 2 - w / 2, shape[1] - w)))
    return y0, y0 + int(h), x0, x0 + int(w)


def evaluate_frame(ctx, gt, a: Counts, found_thr, hit_iou):
    comp_area = np.bincount(ctx.labels.ravel(), minlength=int(ctx.labels.max()) + 1)
    is_cl = {c.id: c.is_cluster for c in ctx.components}
    a["frames"] += 1
    a["comp_isolado"] += sum(not c.is_cluster for c in ctx.components)
    a["comp_amontoado"] += sum(c.is_cluster for c in ctx.components)

    owner_of = []  # (componente dono, area do verme, sobreposicao) por verme encontrado
    gt_in_comp = collections.defaultdict(list)
    for g in gt:
        n = int(g.sum())
        if n < MIN_GT_AREA:
            continue
        a["gt"] += 1
        ov = np.bincount(ctx.labels[g].ravel(), minlength=len(comp_area))
        cov = 1.0 - ov[0] / n  # fracao do verme dentro de qualquer componente
        if cov < 0.2:
            a["nao_marcado"] += 1
        if cov < found_thr:
            a["nao_encontrado"] += 1
            continue
        a["encontrado"] += 1
        j = int(np.argmax(ov[1:])) + 1
        gt_in_comp[j].append((n, int(ov[j])))
        owner_of.append((j, n, int(ov[j])))

    for j, n, inter in owner_of:
        if is_cl.get(j, False):
            a["encontrado_em_amontoado"] += 1
            continue
        a["encontrado_isolado"] += 1
        iou = inter / (n + comp_area[j] - inter)
        dice = 2 * inter / (n + comp_area[j])
        sole = len(gt_in_comp[j]) == 1
        a["isolado_so_1"] += sole
        a["isolado_2mais"] += not sole  # amontoado que o PDI nao detectou
        a["acerto"] += iou >= hit_iou
        if sole:
            a["acerto_so_1"] += iou >= hit_iou
        a.iou.append(iou)
        a.dice.append(dice)

    # Componente com < 30 % da area sobre um verme do gabarito: tres categorias (ver evaluate.classify_no_gt):
    #   "fp"    : FALSO POSITIVO (sujeira, mancha, verme com muita sujeira colada), em dois tipos: isolado (nada o trata
    #             depois) ou amontoado (vai para a CNN, mas quase nao tem verme);
    #   "borda" : encosta na borda do quadro: o gabarito nao anota vermes cortados pela borda (nao e falso positivo);
    #   "verme" : forma e contraste de verme: provavel verme que o gabarito nao anotou (nao e falso positivo).
    a["rejeitados_forma"] += int(ctx.metrics.get("n_rejected_shape", 0))
    fractions = gt_fraction(ctx.labels, ctx.components, gt)
    category = classify_no_gt(ctx, gt, small_min_area=a.small_min_area, small_min_contrast=a.small_min_contrast)
    gtu = None
    for c in ctx.components:
        cat = category.get(c.id)
        if cat is None:
            continue
        if cat == "borda":
            a["borda_sem_gabarito"] += 1
        elif cat == "verme":
            a["verme_sem_gabarito"] += 1
        elif cat == "pequeno":
            a["pequeno_sem_gabarito"] += 1
        else:
            a["comp_sem_verme_amontoado" if c.is_cluster else "comp_sem_verme_isolado"] += 1
        r0, c0, r1, c1 = c.bbox
        row = dict(sequencia=a.seq, quadro=ctx.name, categoria={"fp": "falso_positivo", "borda": "borda", "verme": "provavel_verme", "pequeno": "provavel_verme_pequeno"}[cat],
                   tipo="amontoado" if c.is_cluster else "isolado", componente=c.id, area=c.area,
                   centro_y=round(c.centroid[0]), centro_x=round(c.centroid[1]),
                   bbox=f"{r0} {c0} {r1} {c1}", fracao_sobre_gabarito=round(fractions[c.id], 2),
                   largura_media=round(c.mean_width, 1))
        a.fp_list.append(row)
        if cat in a.gallery_cats:  # guarda so o recorte (imagem, mascara do componente e gabarito) para a galeria
            if gtu is None:
                gtu = np.any(gt, axis=0) if gt else np.zeros(ctx.labels.shape, bool)
            y0, y1, x0, x1 = crop_box(c.bbox, ctx.labels.shape)
            a.gallery.append(dict(row=row, raw=ctx.raw[y0:y1, x0:x1].copy(), mask=(ctx.labels == c.id)[y0:y1, x0:x1],
                                  gt=gtu[y0:y1, x0:x1].copy()))


def render_gallery_item(item, number, scale=3):
    """Uma imagem por componente: esquerda = recorte limpo; direita = contorno do componente (magenta) e do gabarito (ciano)."""
    raw, mask, gtm, r = item["raw"], item["mask"], item["gt"], item["row"]
    h, w = raw.shape
    clean = np.asarray(Image.fromarray(raw).convert("RGB").resize((w * scale, h * scale), Image.LANCZOS))
    over = clean.copy()
    for m, color in ((gtm, (0, 220, 255)), (mask, (255, 0, 255))):
        big = np.kron(m, np.ones((scale, scale), bool))
        over[big & ~ndi.binary_erosion(big, iterations=2)] = color
    cap_h = 48
    canvas = Image.new("RGB", (w * scale * 2 + 6, h * scale + cap_h), (255, 255, 255))
    canvas.paste(Image.fromarray(clean), (0, cap_h))
    canvas.paste(Image.fromarray(over), (w * scale + 6, cap_h))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)  # a fonte padrao nao desenha acentos: legenda sem acento
    d.text((6, 4), f"#{number:03d}   {r['sequencia']}   quadro {r['quadro']}   componente {r['componente']}   ({r['tipo']}, {r['categoria']})", fill=(0, 0, 0), font=font)
    d.text((6, 25), f"area {r['area']} px | largura media {r['largura_media']} px | fracao sobre o gabarito {r['fracao_sobre_gabarito']}"
                    f"   |   direita: magenta = componente, ciano = gabarito", fill=(60, 60, 60), font=font)
    return canvas


def save_gallery(items, out: Path):
    """Grava uma imagem por componente em <out>/galeria/ e um galeria.html para folhear todas."""
    items = sorted(items, key=lambda it: (it["row"]["sequencia"], int(it["row"]["quadro"]), it["row"]["componente"]))
    folder = out / "galeria"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    names = []
    for k, it in enumerate(items, start=1):
        r = it["row"]
        name = f"{k:03d}_{r['sequencia']}_q{r['quadro']}_c{r['componente']}_{r['categoria']}.png"
        render_gallery_item(it, k).save(folder / name)
        names.append((name, it["row"]))
    html = ["<!doctype html><meta charset='utf-8'><title>Galeria de componentes sem gabarito</title>",
            "<style>body{font-family:sans-serif;margin:16px;background:#fafafa}img{max-width:100%;border:1px solid #ccc;margin-bottom:18px}</style>",
            f"<h2>{len(names)} componentes (ordem: sequência, quadro)</h2>",
            "<p>Esquerda: recorte limpo. Direita: magenta = componente do PDI, ciano = gabarito (quando existe).</p>"]
    html += [f"<img src='galeria/{n}' alt='{n}'><br>" for n, _ in names]
    (out / "galeria.html").write_text("\n".join(html), encoding="utf-8")
    return len(names)


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def summarize(name, a: Counts):
    gt = a["gt"]
    iso = a["encontrado_isolado"]
    return {
        "sequencia": name,
        "quadros": a["frames"],
        "vermes_gabarito": gt,
        "encontrados": a["encontrado"],
        "encontrados_pct": pct(a["encontrado"], gt),
        "nao_marcados": a["nao_marcado"],
        "nao_marcados_pct": pct(a["nao_marcado"], gt),
        "vermes_em_isolado": iso,
        "vermes_em_amontoado": a["encontrado_em_amontoado"],
        "acerto_mascara_isolados_pct": pct(a["acerto"], iso),
        "acerto_mascara_so_1_verme_pct": pct(a["acerto_so_1"], a["isolado_so_1"]),
        "iou_medio_isolados": float(np.mean(a.iou)) if a.iou else float("nan"),
        "dice_medio_isolados": float(np.mean(a.dice)) if a.dice else float("nan"),
        "amontoados_nao_detectados": a["isolado_2mais"],
        "componentes_isolados": a["comp_isolado"],
        "componentes_amontoados": a["comp_amontoado"],
        "isolados_por_quadro": a["comp_isolado"] / max(a["frames"], 1),
        "amontoados_por_quadro": a["comp_amontoado"] / max(a["frames"], 1),
        "fp_isolado": a["comp_sem_verme_isolado"],
        "fp_amontoado": a["comp_sem_verme_amontoado"],
        "fp_total": a["comp_sem_verme_isolado"] + a["comp_sem_verme_amontoado"],
        "sem_gabarito_na_borda": a["borda_sem_gabarito"],
        "sem_gabarito_provavel_verme": a["verme_sem_gabarito"],
        "sem_gabarito_verme_pequeno": a["pequeno_sem_gabarito"],
        "descartados_forma": a["rejeitados_forma"],
    }


def render(rows, total, args, removed):
    t = total
    L = []
    L.append("# Relatório de acurácia do PDI (sem CNN)\n")
    L.append(f"- sequências: {len(rows)} | quadros avaliados: {t['quadros']} (1 a cada {args.step}) | vermes no gabarito: {t['vermes_gabarito']}")
    L.append("- configuração: `config.yaml`" + (f" + `--set {' --set '.join(args.set)}`" if args.set else ""))
    L.append(f"- gabarito: {'máscaras duplicadas fundidas (IoU > 0,5): ' + str(removed) + ' removidas' if not args.no_dedup else 'cru (sem fundir duplicadas)'}")
    L.append(f"- critérios: verme **encontrado** = ≥ {int(args.found_thr * 100)} % da área dentro da máscara do PDI; **acerto da máscara** = IoU ≥ {args.hit_iou}\n")
    L.append("## Resultado geral\n")
    L.append("| Métrica | Valor |")
    L.append("|---|---|")
    L.append(f"| **% de vermes encontrados** | **{t['encontrados_pct']:.1f} %** ({t['encontrados']} de {t['vermes_gabarito']}) |")
    L.append(f"| vermes não marcados (cobertura < 20 %) | {t['nao_marcados']} ({t['nao_marcados_pct']:.1f} %) |")
    L.append(f"| **taxa de acerto da máscara dos vermes isolados** (IoU ≥ {args.hit_iou}) | **{t['acerto_mascara_isolados_pct']:.1f} %** de {t['vermes_em_isolado']} vermes |")
    L.append(f"| ... só os que estão sozinhos no componente | {t['acerto_mascara_so_1_verme_pct']:.1f} % |")
    L.append(f"| IoU / Dice médios dos vermes isolados | {t['iou_medio_isolados']:.3f} / {t['dice_medio_isolados']:.3f} |")
    L.append(f"| **componentes isolados × amontoados** | **{t['componentes_isolados']} × {t['componentes_amontoados']}** ({t['isolados_por_quadro']:.1f} × {t['amontoados_por_quadro']:.1f} por quadro) |")
    L.append(f"| vermes dentro de amontoados | {t['vermes_em_amontoado']} |")
    L.append(f"| amontoados que o PDI **não** detectou (componente \"isolado\" com 2+ vermes) | {t['amontoados_nao_detectados']} vermes |")
    n_comp = t['componentes_isolados'] + t['componentes_amontoados']
    L.append(f"| **falsos positivos tipo 1: isolados** (componente **não** marcado como amontoado, < 30 % da área sobre um verme do gabarito, **sem encostar na borda, sem forma de verme e sem ser \"verme pequeno\"**) | **{t['fp_isolado']}** ({100 * t['fp_isolado'] / max(t['componentes_isolados'], 1):.1f} % dos isolados) |")
    L.append(f"| **falsos positivos tipo 2: amontoados** (marcado como amontoado, mas < 30 % da área sobre verme, sem encostar na borda, sem forma de verme e sem ser \"verme pequeno\"; vai para a CNN) | **{t['fp_amontoado']}** ({100 * t['fp_amontoado'] / max(t['componentes_amontoados'], 1):.1f} % dos amontoados) |")
    L.append(f"| **falsos positivos, os dois tipos** | **{t['fp_total']}** ({100 * t['fp_total'] / max(n_comp, 1):.1f} % de todos os componentes) |")
    L.append(f"| componentes sem verme no gabarito **que encostam na borda** (o gabarito não anota vermes cortados pela borda: **não** contam como falso positivo) | {t['sem_gabarito_na_borda']} |")
    L.append(f"| componentes sem verme no gabarito **com forma de verme** (provável verme que o gabarito não anotou: **não** contam como falso positivo) | {t['sem_gabarito_provavel_verme']} |")
    L.append(f"| componentes sem verme no gabarito que são **\"verme pequeno\"** (por definição: ≥ {args.small_min_area} px e contraste ≥ {args.small_min_contrast}; **decisão, não medição**; **não** contam como falso positivo) | {t['sem_gabarito_verme_pequeno']} |")
    L.append(f"| componentes descartados pelo filtro de forma (finos < {args.min_w} px = sujeira; largos > {args.max_w} px = mancha) | {t['descartados_forma']} |\n")
    L.append("## Por sequência\n")
    L.append("| Sequência | vermes | encontrados | não marcados | acerto máscara (isolados) | IoU médio | isolados × amontoados | amontoados não detectados | FP isolados | FP amontoados | sem gabarito: borda | sem gabarito: forma de verme | sem gabarito: verme pequeno |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['sequencia']} | {r['vermes_gabarito']} | {r['encontrados_pct']:.1f} % | {r['nao_marcados']} | "
                 f"{r['acerto_mascara_isolados_pct']:.1f} % | {r['iou_medio_isolados']:.3f} | "
                 f"{r['componentes_isolados']} × {r['componentes_amontoados']} | {r['amontoados_nao_detectados']} | "
                 f"{r['fp_isolado']} | {r['fp_amontoado']} | {r['sem_gabarito_na_borda']} | {r['sem_gabarito_provavel_verme']} | {r['sem_gabarito_verme_pequeno']} |")
    L.append("")
    L.append("Leitura: *amontoados não detectados* são vermes cujo componente tem outro verme dentro mas ficou abaixo do limiar de área "
             "(`analysis.cluster_ratio`); eles derrubam a taxa de acerto dos isolados. *Falsos positivos* são componentes em que menos de 30 % da área "
             "está sobre algum verme do gabarito, **não encostam na borda do quadro** (o gabarito não anota os vermes cortados pela borda) e **não têm "
             "forma de verme** (área, largura média e contraste típicos; os que têm são prováveis vermes que o gabarito não anotou) e **não são \"verme pequeno\"** (qualquer componente sem gabarito com pelo menos `--small-min-area` px e `--small-min-contrast` de contraste; é uma **decisão**, não uma medição: os vermes do gabarito são todos grandes). Esses grupos ficam em linhas à parte). **Tipo 1** = o PDI o chamou de isolado, e nada o trata depois. **Tipo 2** = o PDI o chamou de "
             "amontoado: vai para a CNN, mas quase não tem verme (em geral um verme com muita sujeira colada).")
    return "\n".join(L) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # o console do Windows (cp1252) nao imprime alguns simbolos
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="data/csb-1_extracted/csb-1_dataset")
    ap.add_argument("--seq", nargs="*", help="sequencias (padrao: todas)")
    ap.add_argument("--step", type=int, default=10, help="usa 1 quadro a cada N")
    ap.add_argument("--found-thr", type=float, default=0.5, help="fracao do verme dentro da mascara para contar como encontrado")
    ap.add_argument("--hit-iou", type=float, default=0.5, help="IoU minimo para a mascara de um verme isolado contar como acerto")
    ap.add_argument("--no-dedup", action="store_true", help="nao funde mascaras duplicadas do gabarito")
    ap.add_argument("--out", default="runs/pdi_report")
    ap.add_argument("--no-galeria", action="store_true", help="nao grava as imagens dos componentes sem gabarito")
    ap.add_argument("--small-min-area", type=int, default=SMALL_WORM_MIN_AREA,
                    help="area minima (px) para um componente sem gabarito contar como verme pequeno, e nao como falso positivo")
    ap.add_argument("--small-min-contrast", type=float, default=SMALL_WORM_MIN_CONTRAST, help="contraste minimo do verme pequeno")
    ap.add_argument("--galeria-categorias", nargs="*", default=["fp"], choices=["fp", "borda", "verme", "pequeno"],
                    help="quais componentes entram na galeria: fp = falsos positivos (padrao), borda, verme (provaveis vermes nao anotados)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="secao.campo=valor")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    seqs = args.seq or sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    rows, total, removed = [], Counts(), 0
    for seq in seqs:
        t0 = time.time()
        cfg = load_config(args.config, [f"input.dataset_root={args.dataset_root}", f"input.sequence={seq}",
                                        f"input.step={args.step}", "refine.kind=none", *args.set])
        cfg.stages = STAGES
        pipe = build_pipeline(cfg)
        pipe.fit(cfg.background.n_samples)
        acc = Counts()
        acc.seq = seq
        acc.gallery_cats = () if args.no_galeria else tuple(args.galeria_categorias)
        acc.small_min_area, acc.small_min_contrast = args.small_min_area, args.small_min_contrast
        args.min_w, args.max_w = cfg.analysis.min_mean_width, cfg.analysis.max_mean_width
        for ctx in pipe.run():
            gt = pipe.env.masks.instances(ctx.name)
            if not args.no_dedup:
                gt, r = dedup(gt)
                removed += r
            evaluate_frame(ctx, gt, acc, args.found_thr, args.hit_iou)
        for k, v in acc.items():
            total[k] += v
        total.iou += acc.iou
        total.dice += acc.dice
        total.fp_list += acc.fp_list
        total.gallery += acc.gallery
        rows.append(summarize(seq, acc))
        print(f"{seq}: {acc['frames']} quadros, {acc['gt']} vermes, {pct(acc['encontrado'], acc['gt']):.1f} % encontrados "
              f"({time.time() - t0:.0f}s)", flush=True)

    tot = summarize("TOTAL", total)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = render(rows, tot, args, removed)
    (out / "relatorio.md").write_text(md, encoding="utf-8")
    with open(out / "por_sequencia.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(tot))
        wr.writeheader()
        wr.writerows(rows + [tot])
    if total.fp_list:
        with open(out / "falsos_positivos.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(total.fp_list[0]))
            wr.writeheader()
            wr.writerows(total.fp_list)
    if total.gallery:
        n_img = save_gallery(total.gallery, out)
        print(f"galeria: {n_img} imagens em {out / 'galeria'}  (abra {out / 'galeria.html'} para folhear)")
    print("\n" + md)
    print(f"salvo em {out / 'relatorio.md'}, {out / 'por_sequencia.csv'} e {out / 'falsos_positivos.csv'}")
    for tipo in ("isolado", "amontoado"):
        big = sorted((r for r in total.fp_list if r["tipo"] == tipo and r["categoria"] == "falso_positivo"), key=lambda r: -r["area"])[:2]
        if big:
            print(f"\nPara ver falsos positivos do tipo {tipo} no visualizador, por exemplo:")
        for r in big:
            r0, c0, r1, c1 = (int(v) for v in r["bbox"].split())
            print(f"  python tools/step_viewer.py --dataset-root {args.dataset_root} --seq {r['sequencia']} --frame {r['quadro']} "
                  f"--zoom {max(r0 - 40, 0)},{r1 + 40},{max(c0 - 40, 0)},{c1 + 40}")


if __name__ == "__main__":
    main()
