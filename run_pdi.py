"""Executa a PARTE 1 (PDI): segmenta, separa amontoados, rastreia e exporta dados para a CNN.

Exemplos:
  python run_pdi.py --images D:/zenodo/csb-1_dataset/images/24_2_1_1
  python run_pdi.py --dataset-root D:/zenodo/csb-1_dataset --seq 24_2_1_1 --out runs/teste
  python run_pdi.py --config config.yaml --set threshold.min_threshold=30 --set input.step=2
"""
import argparse
import sys
import time

from celegans_seg import build_pipeline, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--images", help="pasta com os quadros de uma sequencia")
    ap.add_argument("--masks", help="pasta com as mascaras de gabarito (opcional)")
    ap.add_argument("--dataset-root", help="raiz no formato CSB-1: images/<seq>, masks/<seq>")
    ap.add_argument("--seq", help="nome da sequencia dentro de --dataset-root")
    ap.add_argument("--out", help="pasta de saida")
    ap.add_argument("--max-frames", type=int, help="processa apenas os N primeiros quadros")
    ap.add_argument("--set", action="append", default=[], metavar="secao.campo=valor")
    args = ap.parse_args()

    ov = list(args.set)
    for key, val in (("input.images", args.images), ("input.masks", args.masks),
                     ("input.dataset_root", args.dataset_root), ("input.sequence", args.seq),
                     ("output.dir", args.out)):
        if val:
            ov.append(f"{key}={val}")
    if args.max_frames:
        ov.append(f"input.end={args.max_frames}")
    cfg = load_config(args.config, ov)

    pipe = build_pipeline(cfg)
    n = len(pipe.env.source)
    print(f"{n} quadros | gabarito: {'sim' if pipe.env.masks else 'nao'} | saida: {cfg.output.dir}")
    t0 = time.time()
    print("ajustando (fundo, area de referencia)...")
    pipe.fit(cfg.background.n_samples)
    for ctx in pipe.run():
        n_cl = ctx.metrics.get("n_clusters", 0)
        print(f"\rquadro {ctx.index + 1}/{n}  vermes={len(ctx.instances)}  amontoados={n_cl}", end="", file=sys.stderr)
    print(f"\nconcluido em {time.time() - t0:.1f}s -> {cfg.output.dir}")


if __name__ == "__main__":
    main()
