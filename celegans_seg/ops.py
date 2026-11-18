"""Operacoes de PDI de baixo nivel, implementadas a mao (NumPy/SciPy) seguindo os slides.

O objetivo e que cada funcao aponte para a aula de onde veio a ideia. Onde o SciPy ja traz a
operacao (convolucao, morfologia binaria, transformada de distancia), usamos o SciPy para
ficar rapido, e deixamos a versao "didatica" ao lado quando ela existe (rotulagem em duas passadas).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------------------
# Transformacoes pontuais e histograma  (Aula 03)
# --------------------------------------------------------------------------------------


def gamma_correct(img: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """s = c * r^gamma, com r normalizado em [0,1].  [Aula 03, pag. 14]"""
    return np.clip(c * np.power(np.clip(img, 0, 1), gamma), 0, 1).astype(np.float32)


def autocontrast(img: np.ndarray, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    """Alargamento de contraste com saturacao nos percentis (stretchlim + imadjust).
    [Aula 03, pag. 17 e 23-24]"""
    lo, hi = np.quantile(img, [low, high])
    if hi - lo < 1e-6:
        return img
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Filtros de vizinhanca  (Aulas 04 e 05)
# --------------------------------------------------------------------------------------


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Passa-baixas gaussiano h = exp(-(x^2+y^2)/2s^2)/(2 pi s^2), normalizado; lado do kernel
    2*ceil(3 sigma)+1 (truncate=3).  [Aula 04, pag. 13-20]"""
    return ndi.gaussian_filter(img, sigma, truncate=3.0)


def median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Filtro de mediana (ruido sal-e-pimenta).  [Aula 04, pag. 27-28]"""
    return ndi.median_filter(img, size=size)


def bilateral(img: np.ndarray, sigma_s: float = 2.0, sigma_r: float = 0.1) -> np.ndarray:
    """Filtro bilateral: peso = gaussiana espacial x gaussiana de intensidade; preserva bordas.
    [Aula 05, pag. 7-15].  Implementacao por deslocamentos (lenta, mas e a definicao)."""
    r = int(np.ceil(3 * sigma_s))
    h, w = img.shape
    pad = np.pad(img, r, mode="reflect")
    num = np.zeros_like(img, dtype=np.float32)
    den = np.zeros_like(img, dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ws = np.exp(-(dx * dx + dy * dy) / (2 * sigma_s**2))
            nb = pad[r + dy : r + dy + h, r + dx : r + dx + w]
            wgt = ws * np.exp(-((nb - img) ** 2) / (2 * sigma_r**2))
            num += wgt * nb
            den += wgt
    return (num / den).astype(np.float32)


# --------------------------------------------------------------------------------------
# Segmentacao por limiar  (Aula 08)
# --------------------------------------------------------------------------------------


def otsu_threshold(img_u8: np.ndarray) -> int:
    """Otsu: o limiar que maximiza a variancia entre classes (equivalente a minimizar a
    variancia dentro delas).  [Aula 08, pag. 10-13]"""
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sigma_b = (mu[-1] * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return int(np.argmax(sigma_b))


def local_mean_threshold(img_u8: np.ndarray, window: int = 51, offset: int = 20) -> np.ndarray:
    """Limiar adaptativo pela media local (como o 'adaptthresh' padrao do MATLAB).
    [Aula 08, pag. 14-15].  Aqui o objeto e CLARO no sinal, entao T = media_local + offset."""
    t = ndi.uniform_filter(img_u8.astype(np.float32), size=window)
    return img_u8.astype(np.float32) > (t + offset)


def black_tophat(img: np.ndarray, size: int = 31) -> np.ndarray:
    """Top-hat 'preto' = fechamento - imagem: realca objetos ESCUROS menores que o elemento
    estruturante e uniformiza o fundo.  [Aula 08, pag. 62-71]  (SE quadrado, por velocidade)."""
    return ndi.grey_closing(img, size=(size, size)) - img


# --------------------------------------------------------------------------------------
# Morfologia binaria  (Aula 08)
# --------------------------------------------------------------------------------------


def disk(radius: int) -> np.ndarray:
    """Elemento estruturante em forma de disco."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def opening(mask: np.ndarray, radius: int) -> np.ndarray:
    """Abertura (erosao seguida de dilatacao): limpa sem crescer os objetos.
    [Aula 08, pag. 24-25]"""
    return mask if radius <= 0 else ndi.binary_opening(mask, structure=disk(radius))


def closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fechamento (dilatacao seguida de erosao): preenche buracos sem reduzir os objetos.
    [Aula 08, pag. 26-27]"""
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
    """Versao didatica da rotulagem em DUAS PASSADAS com union-find, como nos slides:
    1a passada propaga o menor rotulo e registra colisoes; 2a resolve (find) os rotulos.
    [Aula 08, pag. 56-60].  Lenta (Python puro): usada nos testes para validar `label_components`."""
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
    """Liga com um traco reto os pares de componentes GRANDES separados por no maximo `gap` px.

    Um verme com o meio claro (ou parado) sai em dois pedacos; cada pedaco e grande (>= min_area),
    ao contrario da sujeira do agar, que e pequena. Ligar so componentes grandes evita unir
    ruido (o que o fechamento em tons de cinza faz). Dois vermes distintos proximos tambem sao
    ligados, mas isso e aceitavel: viram um amontoado, que vai para a CNN.
    Nao esta nos slides (usa a transformada de distancia, Aula 01c, pag. 11).
    """
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
    """Distancia euclidiana de cada pixel de objeto ao fundo mais proximo (como o 'bwdist' do
    complemento).  [Aula 01c, pag. 11]"""
    return ndi.distance_transform_edt(mask).astype(np.float32)


# --------------------------------------------------------------------------------------
# Esqueletizacao por afinamento  (Aula 08)
# --------------------------------------------------------------------------------------


def thin(mask: np.ndarray) -> np.ndarray:
    """Esqueleto por AFINAMENTO (algoritmo de Zhang-Suen): remove pixels de borda iterativamente
    preservando a conectividade, ao contrario das erosoes sucessivas que podem romper o objeto.
    [Aula 08, pag. 37-48]  (equivalente ao bwmorph(A,'skel',Inf), pag. 45)."""
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
    """(comprimento em pixels, numero de pontas, numero de juncoes) de um esqueleto.
    Ponta = pixel com 1 vizinho; juncao = pixel com >= 3 vizinhos (vizinhanca-8)."""
    if not skel.any():
        return 0, 0, 0
    nb = ndi.convolve(skel.astype(np.int8), np.ones((3, 3), np.int8), mode="constant") - 1
    nb = nb * skel
    return int(skel.sum()), int((nb == 1).sum()), int((nb >= 3).sum())
