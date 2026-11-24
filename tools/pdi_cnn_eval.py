"""PDI x PDI+CNN x so CNN, ponta a ponta, na mesma sequencia e com a MESMA metrica (mrcnn_common.evaluate).

  python tools/pdi_cnn_eval.py [--seq 24_2_1_1] [--limit N] [--cnn NOME=CKPT ...]

PDI+CNN = pipeline completo com `refine.kind=cnn` (a CNN so atua nos amontoados que o PDI escolhe).
"So CNN" = rede base no quadro inteiro (cache de tools/mrcnn_eval.py em runs/mrcnn_compare).
Saida: runs/pdi_cnn_eval/relatorio_<seq>.md
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mrcnn_common as C  # noqa: E402
from celegans_seg import build_pipeline, load_config  # noqa: E402

STAGES = ["normalize", "denoise", "background", "threshold", "morphology", "labeling", "analysis", "instances",
          "refine"]


def run_pdi(seq, limit, kind, weights):
    ov = [f"input.dataset_root={C.DATASET}", f"input.sequence={seq}", f"refine.kind={kind}"]
    if weights:
        ov.append(f"refine.weights={weights}")
    if limit:
        ov.append(f"input.end={limit}")
    cfg = load_config(ROOT / "config.yaml", ov)
    cfg.stages = STAGES
    pipe = build_pipeline(cfg)
    pipe.fit(cfg.background.n_samples)
    frames, srcs = {}, {"single": 0, "cluster": 0, "cnn": 0}
    for ctx in pipe.run():
        k = int(Path(ctx.name).stem)
        dets = []
        for inst in ctx.instances:
            m = ctx.inst_labels == inst.id
            if m.any():
                dets.append(dict(id=None, rle=C.rle(m), score=1.0))
            srcs[inst.source] = srcs.get(inst.source, 0) + 1
        frames[k] = dets
    return frames, srcs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="24_2_1_1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cnn", action="append", default=[], metavar="NOME=CKPT")
    ap.add_argument("--out", default="runs/pdi_cnn_eval")
    args = ap.parse_args()
    cnns = dict(a.split("=", 1) for a in args.cnn) or {
        "cnn_base": "runs/cnn_train/checkpoint.pth", "cnn_cluster": "runs/cnn_cluster/checkpoint.pth",
        "cnn_cluster_gray": "runs/cnn_cluster_gray/checkpoint.pth", "md_coco": "runs/cnn_md_coco/checkpoint.pth"}
    ks = C.frames_of(args.seq)[: args.limit]
    gt = [C.rles(C.load_gt(args.seq, k)) for k in ks]

    rows = {}
    base_pkl = ROOT / "runs" / "mrcnn_compare" / f"base_{args.seq}.pkl"
    if base_pkl.exists() and not args.limit:
        raw = pickle.load(open(base_pkl, "rb"))
        rows["so CNN (rede base, quadro inteiro)"] = (C.evaluate(C.plain(raw), gt), None)
    for label, kind, w in [("so PDI", "none", None)] + [(f"PDI + {n}", "cnn", p) for n, p in cnns.items()]:
        t0 = time.time()
        fr, srcs = run_pdi(args.seq, args.limit, kind, w)
        pred = [fr.get(k, []) for k in ks]
        rows[label] = (C.evaluate(pred, gt), srcs)
        print(f"{label}: F1 {rows[label][0]['f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    L = ["| versao | F1@0,5 | precisao | recall | F1@0,75 | IoU medio | erro de contagem | FP | FN | instancias de amontoado separadas pela CNN |",
         "|" + "---|" * 10]
    for n, (r, s) in rows.items():
        L.append(f"| {n} | {r['f1']:.3f} | {r['precisao']:.3f} | {r['recall']:.3f} | {r['f1_075']:.3f} | "
                 f"{r['iou_medio']:.3f} | {r['erro_contagem']:.2f} | {r['fp']} | {r['fn']} | "
                 f"{s['cnn'] if s else '-'} |")
    first = next(iter(rows.values()))[0]
    md = f"# PDI x PDI+CNN x so CNN ({args.seq}, {first['quadros']} quadros, {first['vermes_gt']} vermes)\n\n" + "\n".join(L) + "\n"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"relatorio_{args.seq}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
