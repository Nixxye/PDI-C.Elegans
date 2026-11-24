"""Entrada de dados: quadros de uma pasta e mascaras do gabarito."""
from __future__ import annotations

import glob
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


def _natural_key(path: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]


def to_gray_u8(arr: np.ndarray) -> np.ndarray:
    """Qualquer imagem -> cinza uint8 (0,299 R + 0,587 G + 0,114 B) [Aula 10, pag. 2]."""
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    if arr.ndim == 3:
        arr = np.asarray(Image.fromarray(arr[..., :3].astype(np.uint8)).convert("L"))
    return arr.astype(np.uint8)


class FrameSource(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def frame(self, i: int) -> np.ndarray: ...

    @abstractmethod
    def name(self, i: int) -> str: ...


class ImageFolderSource(FrameSource):
    def __init__(self, folder: str | Path, pattern: str = "*.png", start: int = 0, end: int = -1, step: int = 1):
        files = sorted(glob.glob(str(Path(folder) / pattern)), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"nenhum arquivo '{pattern}' em {folder}")
        self.files = files[start : (None if end < 0 else end) : max(step, 1)]
        self.folder = Path(folder)

    def __len__(self):
        return len(self.files)

    def frame(self, i):
        return to_gray_u8(np.asarray(Image.open(self.files[i])))

    def name(self, i):
        return Path(self.files[i]).stem


class MaskSource:
    """Gabarito: para o quadro `name`, devolve a lista de mascaras booleanas (uma por verme)."""

    def __init__(self, folder: str | Path, pattern: str = "{stem}_*.png"):
        self.folder, self.pattern = Path(folder), pattern

    def instances(self, name: str) -> list[np.ndarray]:
        files = sorted(glob.glob(str(self.folder / self.pattern.format(stem=name))), key=_natural_key)
        return [np.asarray(Image.open(f)) > 0 for f in files]


def build_inputs(icfg):
    """Cria (FrameSource, MaskSource|None) a partir de InputCfg."""
    images, masks = icfg.images, icfg.masks
    if icfg.dataset_root:
        root = Path(icfg.dataset_root)
        images = images or str(root / "images" / icfg.sequence)
        masks = masks or str(root / "masks" / icfg.sequence)
    if not images:
        raise ValueError("defina input.images (ou input.dataset_root + input.sequence)")
    source = ImageFolderSource(images, icfg.pattern, icfg.start, icfg.end, icfg.step)
    mask_src = MaskSource(masks, icfg.mask_pattern) if masks and Path(masks).is_dir() else None
    return source, mask_src
