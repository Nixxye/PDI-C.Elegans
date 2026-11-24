"""Compara a Mask R-CNN do colega em quatro versoes, numa sequencia de teste (padrao 24_2_1_1, nunca vista no treino):

  base        modelo original, cada quadro isolado
  base+track  modelo original + rastreador por IoU (informacao de quadros passados, sem retreinar)
  temporal    rede que recebe o "chute" do quadro anterior como 3o canal (tools/mrcnn_temporal_train.py)
  temporal+track  as duas ideias juntas

  python tools/mrcnn_eval.py --base runs/cnn_train/checkpoint.pth [--temporal runs/cnn_temporal/checkpoint.pth]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mrcnn_common as C  # noqa: E402

OUT = C.ROOT / "runs" / "mrcnn_compare"


def cached(path: Path, fn):
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    v = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(v, fh)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="checkpoint do modelo original")
    ap.add_argument("--temporal", help="checkpoint da rede com entrada temporal")
    ap.add_argument("--seq", default="24_2_1_1")
    ap.add_argument("--limit", type=int, help="so os N primeiros quadros (teste rapido)")
    ap.add_argument("--tag", default="", help="sufixo do cache/relatorio")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    tag = f"{args.seq}{('_' + args.tag) if args.tag else ''}{'_n' + str(args.limit) if args.limit else ''}"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.no_cache:
        for f in OUT.glob(f"*_{tag}.pkl"):
            f.unlink()

    ks = C.frames_of(args.seq)[: args.limit]
    gt = [C.rles(C.load_gt(args.seq, k)) for k in ks]

    results, raw = {}, {}
    for name, ckpt, temporal in (("base", args.base, False), ("temporal", args.temporal, True)):
        if not ckpt:
            continue
        t0 = time.time()

        def run(ckpt=ckpt, temporal=temporal):
            print(f"[{name}] predizendo {len(ks)} quadros ...")
            model = C.load_model(ckpt, dev)
            return C.predict_sequence(model, args.seq, dev, temporal=temporal, limit=args.limit)[1]

        raw[name] = cached(OUT / f"{name}_{tag}.pkl", run)
        print(f"[{name}] pronto ({time.time() - t0:.0f}s)")
        results[name] = C.evaluate(C.plain(raw[name]), gt)
        results[name + "+track"] = C.evaluate(C.run_tracker(raw[name]), gt)

    head = "| versao | F1@0,5 | precisao | recall | F1@0,75 | IoU medio | erro de contagem | FP | FN | trocas de ID | fragmentacoes |"
    lines = [head, "|" + "---|" * 11]
    for name, r in results.items():
        has_id = name.endswith("+track")
        vals = [f"{r['f1']:.3f}", f"{r['precisao']:.3f}", f"{r['recall']:.3f}", f"{r['f1_075']:.3f}",
                f"{r['iou_medio']:.3f}", f"{r['erro_contagem']:.2f}", r["fp"], r["fn"],
                r["trocas_id"] if has_id else "-", r["fragmentacoes"]]
        lines.append(f"| {name} | " + " | ".join(str(v) for v in vals) + " |")
    first = next(iter(results.values()))
    md = (f"# Mask R-CNN: sem x com informacao temporal ({args.seq}, {first['quadros']} quadros, "
          f"{first['vermes_gt']} vermes no gabarito)\n\n" + "\n".join(lines) + "\n\n"
          "F1/precisao/recall: pareamento hungaro por IoU >= 0,5; erro de contagem = |previstos - gabarito| medio por quadro; "
          "trocas de ID: vezes em que o mesmo verme do gabarito muda de ID entre quadros (so versoes com rastreador); "
          "fragmentacoes: vezes em que um verme deixa de ser encontrado por 1+ quadros e volta.\n")
    print("\n" + md)
    (OUT / f"relatorio_{tag}.md").write_text(md, encoding="utf-8")
    (OUT / f"relatorio_{tag}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("salvo em", OUT / f"relatorio_{tag}.md")


if __name__ == "__main__":
    main()
