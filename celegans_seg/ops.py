"""Operacoes de PDI (NumPy/SciPy); cada funcao cita a aula."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------------------
# Filtro de vizinhanca  (Aula 04)
# --------------------------------------------------------------------------------------


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Passa-baixas gaussiano [Aula 04, pag. 16-19]."""
    return ndi.gaussian_filter(img, sigma, truncate=3.0)


# --------------------------------------------------------------------------------------
# Segmentacao por limiar  (Aula 08)
# --------------------------------------------------------------------------------------


def otsu_threshold(img_u8: np.ndarray) -> int:
    """Limiar de Otsu [Aula 08, pag. 10-13]."""
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sigma_b = (mu[-1] * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return int(np.argmax(sigma_b))


def tophat(img: np.ndarray, size: int = 31) -> np.ndarray:
    """Top-hat: G - abertura(G) [Aula 08, pag. 66]. SE quadrado."""
    return img - ndi.grey_opening(img, size=(size, size))


def black_tophat(img: np.ndarray, size: int = 31) -> np.ndarray:
    """Top-hat para objeto escuro (imagem invertida) [Aula 08, pag. 62-71]."""
    return tophat(1.0 - img, size)


# --------------------------------------------------------------------------------------
# Morfologia binaria  (Aula 08)
# --------------------------------------------------------------------------------------


def disk(radius: int) -> np.ndarray:
    """Elemento estruturante em forma de disco."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def opening(mask: np.ndarray, radius: int) -> np.ndarray:
    """Abertura [Aula 08, pag. 24-25]."""
    return mask if radius <= 0 else ndi.binary_opening(mask, structure=disk(radius))


def closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fechamento [Aula 08, pag. 26-27]."""
    return mask if radius <= 0 else ndi.binary_closing(mask, structure=disk(radius))


# --------------------------------------------------------------------------------------
# Vizinhanca, rotulagem e distancia  (Aulas 01c e 08)
# --------------------------------------------------------------------------------------

STRUCT_8 = np.ones((3, 3), dtype=bool)  # vizinhanca-8  [Aula 01c, pag. 9]


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Rotulagem de componentes conexos com vizinhanca-8 [Aula 08, pag. 50-60; Aula 01c, pag. 9]."""
    lab, n = ndi.label(mask, structure=STRUCT_8)
    return lab.astype(np.int32), n


def two_pass_labeling(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Rotulagem em duas passadas, union-find [Aula 08, pag. 56-60]. Usada nos testes."""
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            neigh = [
                lab[y + dy, x + dx]
                for dy, dx in ((0, -1), (-1, -1), (-1, 0), (-1, 1))
                if 0 <= y + dy < h and 0 <= x + dx < w and lab[y + dy, x + dx] > 0
            ]
            if not neigh:
                parent.append(len(parent))
                lab[y, x] = len(parent) - 1
            else:
                roots = [find(n) for n in neigh]
                m = min(roots)
                lab[y, x] = m
                for r in roots:
                    parent[r] = m
    remap = {}
    for y in range(h):
        for x in range(w):
            if lab[y, x]:
                r = find(lab[y, x])
                lab[y, x] = remap.setdefault(r, len(remap) + 1)
    return lab, len(remap)


def bridge_components(lab: np.ndarray, n: int, gap: int, min_area: int) -> np.ndarray:
    """Liga componentes grandes separados por ate `gap` px."""
    if n < 2 or gap <= 0:
        return lab > 0
    out = lab > 0
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    big = [i for i in range(1, n + 1) if areas[i] >= min_area]
    slices = ndi.find_objects(lab)
    H, W = lab.shape
    for i in big:
        r0, c0, r1, c1 = slices[i - 1][0].start, slices[i - 1][1].start, slices[i - 1][0].stop, slices[i - 1][1].stop
        r0, c0, r1, c1 = max(r0 - gap, 0), max(c0 - gap, 0), min(r1 + gap, H), min(c1 + gap, W)
        win = lab[r0:r1, c0:c1]
        dt, inds = ndi.distance_transform_edt(win != i, return_indices=True)
        for j in big:
            if j <= i:
                continue
            q = (win == j)
            if not q.any():
                continue
            d = np.where(q, dt, np.inf)
            k = np.unravel_index(np.argmin(d), d.shape)
            if d[k] <= gap:
                py, px = inds[0][k], inds[1][k]
                for t in np.linspace(0, 1, int(d[k]) * 2 + 2):  # traco reto entre os dois pontos mais proximos
                    out[r0 + int(round(py + t * (k[0] - py))), c0 + int(round(px + t * (k[1] - px)))] = True
    return ndi.binary_dilation(out & ~(lab > 0), structure=np.ones((3, 3))) | out if (out != (lab > 0)).any() else out


def remove_small(lab: np.ndarray, n: int, min_area: int) -> tuple[np.ndarray, int]:
    """Descarta componentes com area < min_area e renumera."""
    if n == 0:
        return lab, 0
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    keep = areas >= min_area
    keep[0] = False
    new_ids = np.cumsum(keep) * keep
    return new_ids[lab].astype(np.int32), int(keep.sum())


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Transformada de distancia [Aula 01c, pag. 11]."""
    return ndi.distance_transform_edt(mask).astype(np.float32)


# --------------------------------------------------------------------------------------
# Esqueletizacao por afinamento  (Aula 08)
# --------------------------------------------------------------------------------------


def thin(mask: np.ndarray) -> np.ndarray:
    """Afinamento (Zhang-Suen) [Aula 08, pag. 37-48]."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2, p3 = img[:-2, 1:-1], img[:-2, 2:]
            p4, p5 = img[1:-1, 2:], img[2:, 2:]
            p6, p7 = img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            c = img[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(np.int16) for n in nb)
            a = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            rm = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def skeleton_stats(skel: np.ndarray) -> tuple[int, int, int]:
    """Comprimento, pontas e juncoes do esqueleto."""
    if not skel.any():
        return 0, 0, 0
    nb = ndi.convolve(skel.astype(np.int8), np.ones((3, 3), np.int8), mode="constant") - 1
    nb = nb * skel
    return int(skel.sum()), int((nb == 1).sum()), int((nb >= 3).sum())


# --------------------------------------------------------------------------------------
# Similaridade entre imagens  (Aula 05, pag. 17-22)
# --------------------------------------------------------------------------------------


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Erro quadratico medio entre duas imagens de mesmo tamanho.  [Aula 05, pag. 17]"""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    """PSNR = 10 log10(peak^2 / MSE), em dB (inf se as imagens forem iguais).  [Aula 05, pag. 18]"""
    e = mse(a, b)
    return float("inf") if e == 0 else float(10 * np.log10(peak**2 / e))


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5, roi: np.ndarray | None = None, peak: float = 1.0) -> float:
    """SSIM medio, janela gaussiana [Aula 05, pag. 19-22]; `roi` limita a media."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    g = lambda x: ndi.gaussian_filter(x, sigma, truncate=3.5)  # noqa: E731
    ma, mb = g(a), g(b)
    va, vb, cov = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    smap = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    if roi is not None and roi.any():
        smap = smap[roi]
    return float(smap.mean())
