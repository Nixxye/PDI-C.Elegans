"""PARTE 2 - CNN que separa os vermes de um amontoado.

Arquivo independente do pipeline classico: so precisa de torch. Recebe o recorte que o
`stages/export.py` gera para cada amontoado (imagem, sinal e a mascara do amontoado) e devolve
UMA MASCARA POR VERME (as mascaras podem se sobrepor).

Modelo ("slots"):
  - U-Net de 4 niveis com contexto global: 3 canais de entrada (imagem, sinal pos-fundo, mascara do
    amontoado) -> K = 6 canais de saida, cada um a mascara de UM verme, mais uma probabilidade de
    existencia por canal (se aquele canal tem verme).
  - A ordem dos canais nao importa: no treino, cada verme do gabarito e casado com o canal que melhor o
    descreve (algoritmo hungaro sobre o Dice). So os canais casados levam a perda de mascara; a existencia
    e aprendida a parte (como no DETR).
  - So a regiao dentro da mascara do amontoado conta (perda e saida).

Treino: so ~13 % dos amontoados reais tem 3+ vermes e a rede aprendia a devolver sempre 2. Metade das
amostras de treino sao amontoados SINTETICOS (vermes de outros recortes colados com a mascara exata).

Uso:
  treino:   python tools/train_cnn.py
  conferir: python tools/cnn_check.py
  pipeline: python run_pdi.py ... --set refine.kind=cnn
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

K_SLOTS = 6
MIN_TARGET_AREA = 12  # vermes do gabarito menores que isso (dentro da mascara) sao ignorados


# --------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------

def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class ClusterNet(nn.Module):
    """U-Net com 4 niveis + contexto global. Entrada (B, 3, S, S), S multiplo de 16.

    Saida: (logits das mascaras (B, K, S, S), logits de existencia (B, K)).
    O contexto global (media do fundo da rede, repetida em todos os pixels) deixa cada canal saber o que os
    outros estao fazendo no amontoado inteiro. A existencia de cada canal vem dos atributos da U-Net
    medios DENTRO da mascara daquele canal, mais a area dela.
    """

    def __init__(self, k_slots: int = K_SLOTS, ch=(16, 32, 64, 128, 256)):
        super().__init__()
        self.enc = nn.ModuleList([_block(3, ch[0]), _block(ch[0], ch[1]), _block(ch[1], ch[2]), _block(ch[2], ch[3])])
        self.mid = _block(ch[3], ch[4])
        self.glob = nn.Sequential(nn.Linear(ch[4], ch[4]), nn.ReLU(inplace=True))
        self.fuse = nn.Conv2d(ch[4] * 2, ch[4], 1)
        rev = list(reversed(ch))
        self.up = nn.ModuleList([nn.ConvTranspose2d(rev[i], rev[i + 1], 2, 2) for i in range(4)])
        self.dec = nn.ModuleList([_block(rev[i + 1] * 2, rev[i + 1]) for i in range(4)])
        self.out = nn.Conv2d(ch[0], k_slots, 1)
        self.exist = nn.Sequential(nn.Linear(ch[0] + 1, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))

    def forward(self, x):
        inp = x
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        g = self.glob(x.mean((2, 3)))[:, :, None, None].expand_as(x)
        x = self.fuse(torch.cat([x, g], 1))
        for up, dec, sk in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), sk], 1))
        masks = self.out(x)  # (B, K, S, S)
        p = torch.sigmoid(masks.float()) * inp[:, 2:3].float()  # so dentro da mascara do amontoado
        mass = p.sum((2, 3))  # (B, K)
        v = torch.einsum("bkhw,bchw->bkc", p, x.float()) / (mass[..., None] + 1e-3)
        ex = self.exist(torch.cat([v, torch.log1p(mass)[..., None]], -1)).squeeze(-1)  # (B, K)
        return masks, ex


# --------------------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------------------

def patch_to_input(patch: dict) -> np.ndarray:
    """(3, S, S) float32: imagem, sinal pos-fundo, mascara do amontoado. Mesmo preparo no treino e na inferencia."""
    img = (patch["image"].astype(np.float32) / 255.0 - 0.58) / 0.2
    sig = patch["signal"].astype(np.float32) / 255.0
    msk = patch["mask"].astype(np.float32)
    return np.stack([img, sig, msk])


def load_patches(root: str | Path, seqs: list[str] | None = None, require_gt: bool = True) -> list[dict]:
    """Le os .npz de `<root>/<sequencia>/*.npz`. Cada item: dict com image/signal/mask/gt/meta/seq/file."""
    root = Path(root)
    out = []
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if seqs is not None and sdir.name not in seqs:
            continue
        for f in sorted(sdir.glob("*.npz")):
            d = np.load(f, allow_pickle=False)
            meta = json.loads(str(d["meta"]))
            if "gt" not in d.files:
                if require_gt:
                    continue
                gt = np.zeros((0,) + d["mask"].shape, bool)
            else:
                gt = d["gt"]
            out.append(dict(image=d["image"], signal=d["signal"], mask=d["mask"], gt=gt, meta=meta,
                            seq=sdir.name, file=f.name))
    return out


def targets_in_mask(patch: dict) -> np.ndarray:
    """Mascaras-alvo (N, S, S): cada verme do gabarito restrito a mascara do amontoado."""
    m = patch["mask"]
    g = patch["gt"] & m[None] if len(patch["gt"]) else np.zeros((0,) + m.shape, bool)
    keep = [i for i in range(len(g)) if g[i].sum() >= MIN_TARGET_AREA]
    g = g[keep]
    if len(g) > K_SLOTS:  # mais vermes que canais: fica com os maiores
        g = g[np.argsort(-g.reshape(len(g), -1).sum(1))[:K_SLOTS]]
    return g


class PatchDataset:
    """Amostras em RAM + aumento de dados (rotacao de 90 graus, espelho, brilho/contraste, amontoados sinteticos)."""

    def __init__(self, patches: list[dict], augment: bool = True, synth_p: float = 0.0):
        self.items = []
        for p in patches:
            t = targets_in_mask(p)
            if len(t):
                self.items.append((patch_to_input(p), t))
        self.augment = augment
        self.synth_p = synth_p  # chance de fabricar um amontoado maior colando vermes de outros recortes
        self.rng = np.random.default_rng()

    def _composite(self, x: np.ndarray, t: np.ndarray):
        """Amontoado sintetico: cola 1 a 3 vermes de outros recortes (com a mascara exata) sobre este.

        A imagem do verme colado vem do recorte de origem (verme escuro: vale o menor valor); o sinal e a
        mascara do amontoado sao a uniao. O verme colado precisa encostar no que ja existe, porque um
        amontoado real e um bloco so.
        """
        S = x.shape[1]
        x, t = x.copy(), list(t)
        for _ in range(int(self.rng.integers(1, 4))):
            if len(t) >= K_SLOTS:
                break
            xb, tb = self.items[int(self.rng.integers(len(self.items)))]
            w = tb[int(self.rng.integers(len(tb)))]
            ys, xs = np.nonzero(w)
            if len(ys) < MIN_TARGET_AREA:
                continue
            near = ndi.binary_dilation(x[2] > 0, iterations=2)
            wm = None
            for _try in range(40):
                dy = int(self.rng.integers(-ys.min(), S - ys.max()))
                dx = int(self.rng.integers(-xs.min(), S - xs.max()))
                cand = np.zeros_like(w)
                cand[ys + dy, xs + dx] = True
                if (cand & near).any():
                    wm = cand
                    break
            if wm is None:
                continue

            def shifted(a):  # o verme nao atravessa a borda do recorte
                return np.roll(np.roll(a, dy, axis=-2), dx, axis=-1)

            x[0] = np.where(wm, np.minimum(x[0], shifted(xb[0])), x[0])
            x[1] = np.where(wm, np.maximum(x[1], shifted(xb[1])), x[1])
            x[2] = np.maximum(x[2], wm.astype(np.float32))
            t = [a & (x[2] > 0) for a in t] + [wm]
        return x, np.stack(t)

    def __len__(self):
        return len(self.items)

    def get(self, i: int):
        x, t = self.items[i]
        if self.augment and self.synth_p > 0 and self.rng.random() < self.synth_p:
            x, t = self._composite(x, t)
        if self.augment:
            k = np.random.randint(4)
            x, t = np.rot90(x, k, axes=(1, 2)), np.rot90(t, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                x, t = x[:, :, ::-1], t[:, :, ::-1]
            x = x.copy()
            x[0] = x[0] * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.15, 0.15)
        return np.ascontiguousarray(x), np.ascontiguousarray(t)


# --------------------------------------------------------------------------------------
# Perda (casamento hungaro entre canais e vermes)
# --------------------------------------------------------------------------------------

def _soft_dice_matrix(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """p: (K, HW) probabilidades; g: (N, HW) alvo -> Dice (K, N)."""
    inter = p @ g.T
    return (2 * inter + 1) / (p.sum(1)[:, None] + g.sum(1)[None] + 1)


def slot_loss(logits: torch.Tensor, exist: torch.Tensor, targets: list[np.ndarray], masks: torch.Tensor) -> torch.Tensor:
    """logits (B,K,S,S); exist (B,K); targets: lista de (N_i,S,S); masks (B,S,S) float. Perda media do lote.

    Mascara (BCE + Dice): so nos canais casados com um verme. Existencia (BCE): 1 nos canais casados, 0 nos demais.
    """
    B, K, S, _ = logits.shape
    total = logits.new_zeros(())
    for b in range(B):
        m = masks[b].reshape(-1)
        w = m / m.sum().clamp(min=1)
        p = torch.sigmoid(logits[b]).reshape(K, -1) * m[None]
        g = torch.from_numpy(targets[b].reshape(len(targets[b]), -1).astype(np.float32)).to(logits.device)
        cost = (1 - _soft_dice_matrix(p, g)).detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        rows_t = torch.as_tensor(rows, device=logits.device)
        bce = (F.binary_cross_entropy_with_logits(logits[b].reshape(K, -1)[rows_t], g[cols], reduction="none")
               * w[None]).sum(1)
        dice_loss = 1 - _soft_dice_matrix(p[rows_t], g[cols]).diag()
        e_tgt = torch.zeros(K, device=logits.device)
        e_tgt[rows_t] = 1.0
        total = total + bce.sum() + dice_loss.mean() + F.binary_cross_entropy_with_logits(exist[b], e_tgt)
    return total / B


# --------------------------------------------------------------------------------------
# Treino e avaliacao
# --------------------------------------------------------------------------------------

def match_scores(pred: list[np.ndarray], gt: list[np.ndarray], thr: float = 0.5) -> dict:
    """Casa mascaras previstas e do gabarito (hungaro sobre o IoU). Devolve tp/fp/fn e IoU medio dos casados."""
    if not pred or not gt:
        return dict(tp=0, fp=len(pred), fn=len(gt), iou_sum=0.0, n_matched=0)
    p = np.stack([m.ravel() for m in pred]).astype(np.float32)
    g = np.stack([m.ravel() for m in gt]).astype(np.float32)
    inter = p @ g.T
    iou = inter / np.maximum(p.sum(1)[:, None] + g.sum(1)[None] - inter, 1)
    r, c = linear_sum_assignment(-iou)
    ok = [iou[a, b] for a, b in zip(r, c) if iou[a, b] >= thr]
    return dict(tp=len(ok), fp=len(pred) - len(ok), fn=len(gt) - len(ok), iou_sum=float(sum(ok)), n_matched=len(ok))


def f1_from(counts: dict) -> float:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return 2 * tp / max(2 * tp + fp + fn, 1)


def evaluate(separator: "ClusterSeparator", patches: list[dict]) -> dict:
    """F1 por verme (IoU >= 0.5), acuracia da contagem e IoU medio, sobre recortes com gabarito."""
    tot = dict(tp=0, fp=0, fn=0)
    n_ok = n = 0
    for p in patches:
        gt = list(targets_in_mask(p))
        if not gt:
            continue
        pred = separator.predict_masks(p)
        s = match_scores(pred, gt)
        for k in tot:
            tot[k] += s[k]
        n += 1
        n_ok += int(len(pred) == len(gt))
    return dict(f1=f1_from(tot), count_acc=n_ok / max(n, 1), n_patches=n, **tot)


def train(patches: list[dict], out_path: str | Path, val_patches: list[dict] | None = None, epochs: int = 100,
          synth_p: float = 0.5, batch: int = 16, lr: float = 2e-3, device: str | None = None, seed: int = 0,
          log=print) -> Path:
    """Treina e grava os pesos em `out_path` (guarda a melhor epoca pelo F1 de validacao, se houver)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = PatchDataset(patches, augment=True, synth_p=synth_p)
    log(f"treino: {len(ds)} recortes | validacao: {len(val_patches or [])} | dispositivo: {device}")
    net = ClusterNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    steps = max(len(ds) // batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps)
    best, out_path = -1.0, Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        order = np.random.choice(len(ds), size=steps * batch)
        run = 0.0
        for s in range(steps):
            items = [ds.get(i) for i in order[s * batch:(s + 1) * batch]]
            x = torch.from_numpy(np.stack([a for a, _ in items])).to(device)
            masks = x[:, 2]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, ex = net(x)
            loss = slot_loss(logits.float(), ex.float(), [t for _, t in items], masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() / steps
        msg = f"epoca {ep:3d}/{epochs}  perda {run:.4f}  ({time.time() - t0:.0f}s)"
        if val_patches and (ep % 5 == 0 or ep == epochs):
            ev = evaluate(ClusterSeparator(net=net, device=device), val_patches)
            msg += f"  | val F1 {ev['f1']:.3f}  contagem {ev['count_acc']:.2f}"
            if ev["f1"] > best:
                best = ev["f1"]
                torch.save(net.state_dict(), out_path)
        elif not val_patches:
            torch.save(net.state_dict(), out_path)
        log(msg)
    return out_path


# --------------------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------------------

class ClusterSeparator:
    """Carrega os pesos e separa um recorte de amontoado em mascaras por verme."""

    def __init__(self, weights: str | Path | None = None, net: ClusterNet | None = None, device: str | None = None,
                 threshold: float = 0.5, min_area: int = 20, dup_iou: float = 0.7, exist_threshold: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net is None:
            net = ClusterNet().to(self.device)
            net.load_state_dict(torch.load(weights, map_location=self.device))
        self.net = net
        self.net.eval()
        self.threshold, self.min_area, self.dup_iou = threshold, min_area, dup_iou
        self.exist_threshold = exist_threshold

    @torch.no_grad()
    def slot_probs(self, patch: dict) -> tuple[np.ndarray, np.ndarray]:
        """(probabilidade de cada pixel em cada canal (K, S, S), probabilidade de o canal existir (K,))."""
        x = torch.from_numpy(patch_to_input(patch))[None].to(self.device)
        masks, ex = self.net(x)
        return torch.sigmoid(masks.float())[0].cpu().numpy(), torch.sigmoid(ex.float())[0].cpu().numpy()

    def _kept_slots(self, patch: dict):
        """(probs, [(canal, mascara)]): canais que existem (prob >= exist_threshold), com area minima, sem duplicata."""
        probs, exist = self.slot_probs(patch)
        mask = patch["mask"]
        cand = []
        for k in range(len(probs)):
            if exist[k] < self.exist_threshold:
                continue
            m = (probs[k] > self.threshold) & mask
            if m.sum() >= self.min_area:
                cand.append((k, m, float(exist[k]) * m.sum()))
        cand.sort(key=lambda t: -t[2])
        kept: list[tuple[int, np.ndarray]] = []
        for k, m, _ in cand:  # tira canais duplicados (mesmo verme em dois canais)
            if all((m & o).sum() / max((m | o).sum(), 1) < self.dup_iou for _, o in kept):
                kept.append((k, m))
        return probs, kept

    def predict_masks(self, patch: dict) -> list[np.ndarray]:
        """Uma mascara booleana por verme previsto (podem se sobrepor), dentro da mascara do amontoado."""
        return [m for _, m in self._kept_slots(patch)[1]]

    def separate(self, patch: dict) -> np.ndarray:
        """Mapa de rotulos (S, S): 0 = fora do amontoado, 1..n = verme. Pixel disputado vai para o canal
        mais confiante; pixel do amontoado sem canal vai para o canal mais provavel."""
        probs, kept = self._kept_slots(patch)
        lab = np.zeros(patch["mask"].shape, np.int32)
        if not kept:
            return lab
        sub = probs[[k for k, _ in kept]]  # (n, S, S)
        lab[patch["mask"]] = np.argmax(sub[:, patch["mask"]], axis=0) + 1
        return lab
