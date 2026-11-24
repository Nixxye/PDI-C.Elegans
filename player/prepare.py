"""Pre-processa as opcoes do player (video x rede) para elas abrirem na hora. Pula o que ja esta pronto.

  python player/prepare.py                        # tudo: todos os videos x todas as redes (horas: ~1,3 s por quadro)
  python player/prepare.py --videos 24_2_1_1 --models pdi cnn_cluster
  python player/prepare.py --list                 # mostra as opcoes e o que ja esta pronto

Sem isso o player tambem funciona: ele processa na hora a opcao que voce escolher (e guarda).
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import catalog  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", help="so estes (parte do nome, ex.: 24_2_1_1); padrao: todos")
    ap.add_argument("--models", nargs="*", help="so estas redes (ids; ex.: pdi cnn_cluster); padrao: todas")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    vids = [v for v in catalog.videos() if not a.videos or any(x in v["id"] for x in a.videos)]
    mods = [m for m in catalog.models() if not a.models or m["id"] in a.models]
    todo = [(v, m) for v in vids for m in mods]
    for v, m in todo:
        ok = catalog.is_ready(catalog.cache_name(v["images"], m["path"]))
        if a.list:
            print(f"{'pronto  ' if ok else 'a fazer '} {v['label']:28s} {m['id']}")
    if a.list:
        return
    t0 = time.time()
    for k, (v, m) in enumerate(todo, 1):
        name = catalog.cache_name(v["images"], m["path"])
        if catalog.is_ready(name):
            continue
        print(f"[{k}/{len(todo)}] {v['label']} + {m['id']} ({v['frames']} quadros)", flush=True)
        build.build(v["images"], catalog.CACHE / name, str(catalog.ROOT / m["path"]) if m["path"] else "", None,
                    lambda d, t: print(f"\r  quadro {d}/{t}", end="", file=sys.stderr))
        print(f"\n  pronto em {time.time() - t0:.0f}s (acumulado)", flush=True)


if __name__ == "__main__":
    main()
