"""Gera os recortes de amontoados (com gabarito) para treinar/avaliar a CNN.

Roda o pipeline de PDI (sem avaliacao/visualizacao) em cada sequencia do CSB-1 e grava,
para cada amontoado, um .npz em  <out>/cnn_dataset/<sequencia>/  (ver stages/export.py).

  python tools/build_cnn_dataset.py                       # todas as sequencias, passo 2
  python tools/build_cnn_dataset.py --seqs 24_0_1_1 --step 1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from celegans_seg import build_pipeline, load_config  # noqa: E402

STAGES = ["normalize", "denoise", "background", "threshold", "morphology", "labeling",
          "analysis", "split", "export"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="data/csb-1_extracted/csb-1_dataset")
    ap.add_argument("--seqs", nargs="*", help="padrao: todas as sequencias de <dataset-root>/images")
    ap.add_argument("--step", type=int, default=2, help="usa 1 a cada N quadros (vizinhos sao quase iguais)")
    ap.add_argument("--patch-size", type=int, default=128, help="lado do recorte; amontoados maiores que isso nao entram")
    ap.add_argument("--out", default="data/cnn_patches")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    seqs = args.seqs or sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    for seq in seqs:
        t0 = time.time()
        cfg = load_config(args.config, [f"input.dataset_root={args.dataset_root}", f"input.sequence={seq}",
                                        f"input.step={args.step}", f"output.dir={args.out}",
                                        f"export.patch_size={args.patch_size}"])
        cfg.stages = STAGES
        pipe = build_pipeline(cfg)
        pipe.fit(cfg.background.n_samples)
        n = sum(1 for _ in pipe.run())
        print(f"{seq}: {n} quadros em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
