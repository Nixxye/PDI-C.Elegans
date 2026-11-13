"""Converte o `Annotated_images.zip` do Zenodo 15526615 (Weheliye et al., 2025) para o layout do projeto:

    <out>/images/<sequencia>/k.png         recorte em cinza
    <out>/masks/<sequencia>/k_i.png        uma mascara por verme (contorno preenchido; sobreposicao preservada)
    <out>/skeletons/<sequencia>.npz        (opcional) esqueletos e larguras originais

Formato original (inspecionado): cada `<fonte>_<split>.p.zip` e na verdade um GZIP de um PICKLE com
uma lista de amostras; cada amostra e uma tupla de 7 itens:
    0 imagem uint8 (H,W) | 1 copia da imagem | 2 larguras (N,49) | 3 esqueleto (N,49,2)
    4 contorno (N,98,2) int | 5 caixas por verme | 6 caixa do recorte
(`from_NNv1_*` traz so imagem, larguras e esqueleto: itens 1, 4, 5 e 6 sao None -> mascaras aproximadas)
As amostras sao recortes independentes (NAO e video): nao ha continuidade temporal entre elas.

SEGURANCA: arquivos .p sao pickles; `pickle.load` comum executa codigo arbitrario. Aqui o
desserializador aceita SOMENTE as classes do NumPy que aparecem nesses arquivos e recusa o resto.

Uso:
  python tools/convert_annotated_images.py --src data/annotated_raw/Annotated_images \
      --out data/annotated_converted --only manual-v2_test manual_test
"""
import argparse
import gzip
import io
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("_codecs", "encode"): __import__("_codecs").encode,
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"classe bloqueada: {module}.{name}") from None


def load_samples(path: Path) -> list:
    raw = gzip.open(path).read()
    return SafeUnpickler(io.BytesIO(raw), encoding="latin1").load()


def rasterize(outline: np.ndarray, shape: tuple, swap: bool) -> np.ndarray:
    pts = [tuple(map(float, (p[1], p[0]) if swap else (p[0], p[1]))) for p in outline]
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255, outline=255)
    return np.asarray(im) > 0


def from_skeleton(skel: np.ndarray, widths: np.ndarray, shape: tuple, swap: bool) -> np.ndarray:
    """Mascara aproximada a partir do esqueleto (49 pontos) e da largura em cada ponto.

    Usada quando a amostra nao traz contorno (`from_NNv1_*`). Calibrada nas amostras `manual-v2_test`,
    que tem os dois: tratando a largura como largura TOTAL, o IoU medio contra o contorno anotado e ~0,88
    (com metade da largura cai para 0,57; com 1,5x para 0,70).
    """
    w = np.clip(widths, 0, None)
    p = skel[:, ::-1] if swap else skel
    t = np.gradient(p, axis=0)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    h = (w / 2)[:, None]
    poly = np.concatenate([p + n * h, (p - n * h)[::-1]])
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    d.polygon([tuple(map(float, q)) for q in poly], fill=255, outline=255)
    for q, r in ((p[0], h[0, 0]), (p[-1], h[-1, 0])):  # pontas arredondadas
        d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=255)
    return np.asarray(im) > 0


def worm_masks(sample, swap: bool) -> tuple[list, bool]:
    """(mascaras por verme, reconstruidas_do_esqueleto?)"""
    img, widths, skeleton, outline = sample[0], sample[2], sample[3], sample[4]
    if outline is not None:
        return [rasterize(outline[i], img.shape, swap) for i in range(len(outline))], False
    return [from_skeleton(skeleton[i], widths[i], img.shape, swap) for i in range(len(skeleton))], True


def detect_swap(samples: list, n: int = 80) -> bool:
    """Descobre a ordem dos eixos (x,y) x (linha,coluna). O verme e ESCURO, entao a ordem certa e a
    que deixa o interior das mascaras mais escuro que um anel logo ao redor delas (contraste local;
    comparar com o recorte inteiro falha nas imagens do Tierpsy, que tem um halo cinza)."""
    from scipy import ndimage as ndi

    score = {False: [], True: []}
    for s in samples[:n]:
        img = s[0].astype(float)
        for swap in (False, True):
            m = worm_masks(s, swap)[0][0]
            ring = ndi.binary_dilation(m, iterations=5) & ~m
            if m.sum() > 20 and ring.sum() > 20:
                score[swap].append(img[ring].mean() - img[m].mean())
    return float(np.mean(score[True])) > float(np.mean(score[False]))


def convert(src: Path, out: Path, name: str, save_skeletons: bool):
    samples = load_samples(src / f"{name}.p.zip")
    if not samples or not isinstance(samples[0], tuple) or len(samples[0]) != 7 and samples[0][4] is not None:
        raise SystemExit(f"{name}: formato diferente do esperado (ex.: from_tierpsy_negative, sem vermes); nao convertido")
    swap = detect_swap(samples)
    img_dir, msk_dir = out / "images" / name, out / "masks" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    skel, n_rebuilt = [], 0
    for k, s in enumerate(samples):
        img, widths, skeleton = s[0], s[2], s[3]
        Image.fromarray(img).save(img_dir / f"{k}.png")
        masks, rebuilt = worm_masks(s, swap)
        n_rebuilt += rebuilt
        for i, m in enumerate(masks):
            Image.fromarray((m * 255).astype(np.uint8)).save(msk_dir / f"{k}_{i}.png")
        skel.append((skeleton, widths))
    if save_skeletons:
        (out / "skeletons").mkdir(exist_ok=True)
        np.savez_compressed(out / "skeletons" / f"{name}.npz",
                            **{f"s{k}": a for k, (a, _) in enumerate(skel)},
                            **{f"w{k}": w for k, (_, w) in enumerate(skel)})
    n_worms = [len(s[3]) for s in samples]
    note = f"; mascaras RECONSTRUIDAS do esqueleto em {n_rebuilt} recortes" if n_rebuilt else ""
    print(f"{name}: {len(samples)} recortes, {sum(n_worms)} vermes (eixos {'trocados' if swap else 'diretos'}){note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pasta com os <fonte>_<split>.p.zip")
    ap.add_argument("--out", default="data/annotated_converted")
    ap.add_argument("--only", nargs="+", required=True, help="nomes sem a extensao, ex.: manual-v2_test")
    ap.add_argument("--skeletons", action="store_true", help="salva tambem esqueletos e larguras")
    a = ap.parse_args()
    for name in a.only:
        convert(Path(a.src), Path(a.out), name, a.skeletons)


if __name__ == "__main__":
    main()
