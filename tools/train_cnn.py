"""Treina a CNN que separa vermes dentro dos amontoados (celegans_seg/cnn.py: `train`, A IMPLEMENTAR).

  python tools/build_cnn_dataset.py            # 1) gera os recortes (uma vez)
  python tools/train_cnn.py                    # 2) treina; pesos em models/cluster_cnn.pt

A sequencia de teste (24_2_1_1) nunca entra no treino nem na validacao.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from celegans_seg import cnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default="data/cnn_patches/cnn_dataset")
    ap.add_argument("--out", default="models/cluster_cnn.pt")
    ap.add_argument("--test-seq", default="24_2_1_1", help="reservada: fora do treino e da validacao")
    ap.add_argument("--val-seqs", nargs="*", default=["24_1_1_2"], help="validacao (escolhe a melhor epoca)")
    args = ap.parse_args()

    allp = cnn.load_patches(args.patches)
    train = [p for p in allp if p["seq"] != args.test_seq and p["seq"] not in args.val_seqs]
    val = [p for p in allp if p["seq"] in args.val_seqs]
    cnn.train(train, args.out, val_patches=val or None)
    print("pesos em", args.out)


if __name__ == "__main__":
    main()
