"""O que o player oferece: os videos (sequencias de quadros) e as redes. O player nao deixa escolher pasta qualquer: so
estas opcoes. Para oferecer outro video, acrescente uma raiz em DATA_ROOTS; para outra rede, uma linha em MODELS.

Cada combinacao (video, rede) e processada uma vez (python player/prepare.py ou, na hora, pelo proprio player) e fica em
player/cache/<nome>/.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / "cache"
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# (rotulo, raiz com images/<sequencia>/ e, opcional, masks/<sequencia>/)
DATA_ROOTS = [
    ("CSB-1", ROOT / "data" / "csb-1_extracted" / "csb-1_dataset"),
    ("Sintético", ROOT / "data" / "synthetic"),
]

# (id, rotulo, checkpoint relativo a raiz do projeto; "" = so PDI). So uma opcao: PDI + a CNN do colega nos amontoados
# (Mask R-CNN de cnn/, treinada com o codigo dele: runs/cnn_train). Com mais de uma, o player mostra o seletor.
MODELS = [
    ("cnn", "PDI + CNN do colega nos amontoados", "runs/cnn_train/checkpoint.pth"),
]


def videos():
    out = []
    for label, root in DATA_ROOTS:
        imgs = root / "images"
        for d in sorted(imgs.iterdir()) if imgs.is_dir() else []:
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT) if d.is_dir() else 0
            if n:
                out.append(dict(id=f"{label}/{d.name}", label=f"{label} · {d.name}", images=str(d), frames=n))
    return out


def models():
    return [dict(id=i, label=l, path=p) for i, l, p in MODELS if not p or (ROOT / p).is_file()]


def cache_name(images: str, cnn: str, max_frames=None) -> str:
    h = hashlib.sha1(f"{images}|{cnn}|{max_frames}".encode()).hexdigest()[:8]
    return f"{Path(images).name}_{Path(cnn).parent.name if cnn else 'pdi'}_{h}"


def is_ready(name: str) -> bool:
    """Pronto = o build terminou (ready.json e gravado por ultimo; caches antigos, sem as etapas, nao contam)."""
    return (CACHE / name / "ready.json").is_file()
