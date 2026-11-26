"""Mede a linha do tempo (estagio `timeline`) contra o gabarito: trocas de ID e fragmentacoes (mesma metrica de
tools/mrcnn_eval.py). Compara a ligacao por SSIM (PDI e PDI + CNN) com o rastreador por IoU sobre a CNN no quadro inteiro.

  python tools/timeline_eval.py [--seq 24_2_1_1] [--limit N] [--weights runs/cnn_cluster/checkpoint.pth]
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mrcnn_common as C  # noqa: E402
from celegans_seg import build_pipeline, load_config  # noqa: E402


def run(seq, limit, kind, weights, out, extra=()):
    ov = [*extra, f"input.dataset_root={C.DATASET}", f"input.sequence={seq}", f"refine.kind={kind}", f"output.dir={out}"]
    if weights:
        ov.append(f"refine.weights={weights}")
    if limit:
        ov.append(f"input.end={limit}")
    cfg = load_config(ROOT / "config.yaml", ov)
    cfg.stages = ["normalize", "denoise", "background", "threshold", "morphology", "labeling", "analysis",
                  "instances", "refine", "timeline"]
    pipe = build_pipeline(cfg)
    pipe.fit(cfg.background.n_samples)
    fr = {}
    for ctx in pipe.run():
        fr[int(Path(ctx.name).stem)] = [dict(id=i.track, rle=C.rle(ctx.inst_labels == i.id), score=1.0)
                                        for i in ctx.instances if (ctx.inst_labels == i.id).any()]
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="24_2_1_1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--set", action="append", default=[], help="sobrescreve a config, ex.: timeline.complete_partial=true")
    ap.add_argument("--only-cnn", action="store_true", help="nao roda a versao so PDI")
    ap.add_argument("--raw-gt", action="store_true", help="nao funde mascaras duplicadas do gabarito")
    ap.add_argument("--weights", default="runs/cnn_cluster/checkpoint.pth")
    args = ap.parse_args()
    ks = C.frames_of(args.seq)[: args.limit]
    gt = [C.rles(C.load_gt(args.seq, k)) for k in ks]
    if not args.raw_gt:
        gt = [C.dedup_gt(g) for g in gt]
    rows = {}
    pkl = ROOT / "runs" / "mrcnn_compare" / f"base_{args.seq}.pkl"
    if pkl.exists():
        raw = pickle.load(open(pkl, "rb"))[: len(ks)]
        rows["CNN quadro inteiro + rastreador IoU"] = C.evaluate(C.run_tracker(raw), gt)
    for label, kind, w in (("PDI + linha do tempo SSIM", "none", None), ("PDI + CNN + linha do tempo SSIM", "cnn", args.weights)):
        if args.only_cnn and kind == "none":
            continue
        fr = run(args.seq, args.limit, kind, w, f"runs/timeline_eval/{kind}", args.set)
        rows[label] = C.evaluate([fr.get(k, []) for k in ks], gt)
    L = ["| versao | F1@0,5 | FP | FN | trocas de ID | fragmentacoes |", "|---|---|---|---|---|---|"]
    for n, r in rows.items():
        L.append(f"| {n} | {r['f1']:.3f} | {r['fp']} | {r['fn']} | {r['trocas_id']} | {r['fragmentacoes']} |")
    md = f"# Linha do tempo ({args.seq}, {len(ks)} quadros, {r['vermes_gt']} vermes)\n\n" + "\n".join(L) + "\n"
    Path(ROOT / "runs/timeline_eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/timeline_eval/relatorio.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
