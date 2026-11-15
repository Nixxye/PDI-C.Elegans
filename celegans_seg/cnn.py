"""PARTE 2 - CNN que separa os vermes de um amontoado.

Arquivo independente do pipeline classico: so precisa de torch. Recebe o recorte que o
`stages/export.py` gera para cada amontoado (imagem, sinal e a mascara do amontoado) e devolve
UMA MASCARA POR VERME (as mascaras podem se sobrepor).

Modelo ("slots", sem pos-processamento classico):
  - U-Net de 4 niveis com contexto global: 3 canais de entrada (imagem, sinal pos-fundo, mascara do amontoado) -> K canais
    de saida (K = 6, o maximo de vermes por amontoado). Cada canal e a mascara de UM verme.
  - A ordem dos canais nao importa: no treino, cada verme do gabarito e casado com o canal que
    melhor o descreve (algoritmo hungaro sobre o Dice). So os canais casados sao cobrados na MASCARA;
    quem diz se um canal "existe" (tem verme) e uma saida a parte, uma probabilidade de existencia por
    canal (como o DETR). Isso evita o defeito da versao anterior: canais sem verme eram forcados a sair
    vazios, a maioria dos amontoados tem so 2 vermes, e a rede aprendia a devolver sempre 2.
  - Saidas sobrepostas sao permitidas (vermes cruzados).
  - So a regiao dentro da mascara do amontoado conta (perda e saida).

Uso:
  treino:   python tools/train_cnn.py
  conferir: python tools/cnn_check.py
  pipeline: python run_pdi.py ... --set refine.kind=cnn
"""
from __future__ import annotations

import csv
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
PREV_DILATE = 10  # px: vermes do quadro anterior que contam para este amontoado (eles andam entre os quadros)
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

    def __init__(self, k_slots: int = K_SLOTS, ch=(16, 32, 64, 128, 256), in_ch: int = 3):
        super().__init__()
        self.enc = nn.ModuleList([_block(in_ch, ch[0]), _block(ch[0], ch[1]), _block(ch[1], ch[2]), _block(ch[2], ch[3])])
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

def prev_info(patch: dict) -> tuple[np.ndarray | None, int]:
    """(uniao das mascaras dos vermes do quadro anterior perto deste amontoado, quantos sao) ou (None, 0).

    Execucao: o `refine` ja entrega `prev_union` e `prev_n` (saida do sistema no quadro anterior).
    Treino/conferencia: vem do gabarito do quadro anterior (`prev_gt`): so os vermes que encostam no
    amontoado dilatado de PREV_DILATE px.
    """
    if patch.get("prev_union") is not None:
        return patch["prev_union"], int(patch.get("prev_n", 0))
    pg = patch.get("prev_gt")
    if pg is None or not len(pg):
        return None, 0
    near = ndi.binary_dilation(patch["mask"], iterations=PREV_DILATE)
    keep = [w for w in pg if (w & near).any()]
    if not keep:
        return None, 0
    return np.any(keep, axis=0), len(keep)


def patch_to_input(patch: dict, in_ch: int = 3) -> np.ndarray:
    """(in_ch, S, S) float32: imagem, sinal, mascara [+ uniao dos vermes do quadro anterior, contagem/6].
    Mesmo preparo no treino e na inferencia; sem quadro anterior os dois canais extras ficam zerados."""
    img = (patch["image"].astype(np.float32) / 255.0 - 0.58) / 0.2
    sig = patch["signal"].astype(np.float32) / 255.0
    msk = patch["mask"].astype(np.float32)
    chans = [img, sig, msk]
    if in_ch >= 5:
        union, n = prev_info(patch)
        chans.append(np.zeros_like(msk) if union is None else union.astype(np.float32))
        chans.append(np.full_like(msk, min(n, K_SLOTS) / K_SLOTS))
    return np.stack(chans)


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
            item = dict(image=d["image"], signal=d["signal"], mask=d["mask"], gt=gt, meta=meta,
                        seq=sdir.name, file=f.name)
            for k in ("dist", "skeleton", "ws_labels", "prev_gt"):
                if k in d.files:
                    item[k] = d[k]
            out.append(item)
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
    """Amostras em RAM + aumento de dados (rotacao de 90 graus, espelho, brilho/contraste)."""

    def __init__(self, patches: list[dict], augment: bool = True, synth_p: float = 0.0, in_ch: int = 3):
        self.items = []
        self.in_ch = in_ch
        for p in patches:
            t = targets_in_mask(p)
            if len(t):
                self.items.append((patch_to_input(p, in_ch), t))
        self.augment = augment
        self.synth_p = synth_p  # chance de fabricar um amontoado maior colando vermes de outros recortes
        self.rng = np.random.default_rng()

    def _composite(self, x: np.ndarray, t: np.ndarray):
        """Amontoado sintetico: cola 1 a 3 vermes de outros recortes (com a mascara exata) sobre este.

        Pouco dado tem 3+ vermes (~13 %), e a rede aprendia a devolver 2. A imagem do verme colado vem do
        recorte de origem (verme escuro: vale o menor valor), o sinal e a mascara do amontoado sao a uniao.
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
            for _try in range(40):  # um amontoado real e um bloco so: o verme colado precisa encostar no que ja existe
                dy = int(self.rng.integers(-ys.min(), S - ys.max()))
                dx = int(self.rng.integers(-xs.min(), S - xs.max()))
                cand = np.zeros_like(w)
                cand[ys + dy, xs + dx] = True
                if (cand & near).any():
                    wm = cand
                    break
            if wm is None:
                continue
            src = np.zeros_like(w)
            src[ys, xs] = True
            sh = lambda a: np.roll(np.roll(a, dy, axis=-2), dx, axis=-1)  # noqa: E731  (o verme nao atravessa a borda)
            x[0] = np.where(wm, np.minimum(x[0], sh(xb[0])), x[0])
            x[1] = np.where(wm, np.maximum(x[1], sh(xb[1])), x[1])
            x[2] = np.maximum(x[2], wm.astype(np.float32))
            t = [a & (x[2] > 0) for a in t] + [wm]
            if self.in_ch >= 5 and x[4].max() > 0:  # no quadro anterior o verme colado estava um pouco deslocado
                jit = np.roll(np.roll(wm, int(self.rng.integers(-3, 4)), axis=0), int(self.rng.integers(-3, 4)), axis=1)
                x[3] = np.maximum(x[3], jit.astype(np.float32))
                x[4] = np.minimum(x[4] + 1.0 / K_SLOTS, 1.0)
        return x, np.stack(t)

    def __len__(self):
        return len(self.items)

    def get(self, i: int):
        x, t = self.items[i]
        if self.augment and self.synth_p > 0 and self.rng.random() < self.synth_p:
            x, t = self._composite(x, t)
        if self.augment and self.in_ch >= 5:  # o quadro anterior nem sempre existe nem e perfeito
            x = x.copy()
            r = self.rng.random()
            if r < 0.2:
                x[3:] = 0.0
            elif r < 0.4 and x[4].max() > 0:
                x[4] = np.clip(x[4] + self.rng.choice([-1, 1]) / K_SLOTS, 0.0, 1.0)
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
# Treino
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


def train(patches: list[dict], out_path: str | Path, val_patches: list[dict] | None = None, epochs: int = 60,
          balance: float = 0.0, synth_p: float = 0.0, temporal: bool = False, batch: int = 16, lr: float = 2e-3, device: str | None = None, seed: int = 0, log=print) -> Path:
    """Treina e grava os pesos em `out_path` (guarda a melhor epoca pelo F1 de validacao, se houver)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    in_ch = 5 if temporal else 3
    ds = PatchDataset(patches, augment=True, synth_p=synth_p, in_ch=in_ch)
    log(f"treino: {len(ds)} recortes | validacao: {len(val_patches or [])} | dispositivo: {device}")
    net = ClusterNet(in_ch=in_ch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    steps = max(len(ds) // batch, 1)
    # amontoados com 1 verme ou 3+ sao raros (2 vermes = ~74 %); `balance` reamostra pelo numero de vermes
    n_per = np.array([len(t) for _, t in ds.items])
    freq = {n: (n_per == n).sum() for n in np.unique(n_per)}
    weights = np.array([float(freq[n]) ** -balance for n in n_per])  # balance 0 = uniforme, 0.5 = 1/raiz, 1 = igual por classe
    weights = weights / weights.sum()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps)
    best, out_path = -1.0, Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        order = np.random.choice(len(ds), size=steps * batch, p=weights)
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
            sep = ClusterSeparator(net=net, device=device)
            ev = evaluate(sep, val_patches)
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
                 threshold: float = 0.5, min_area: int = 20, dup_iou: float = 0.7, exist_threshold: float = 0.5,
                 use_prev_count: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net is None:
            sd = torch.load(weights, map_location=self.device)
            net = ClusterNet(in_ch=sd["enc.0.0.weight"].shape[1]).to(self.device)  # 3 canais (so quadro atual) ou 5 (com o anterior)
            net.load_state_dict(sd)
        self.net = net
        self.in_ch = net.enc[0][0].in_channels
        self.net.eval()
        self.threshold, self.min_area, self.dup_iou = threshold, min_area, dup_iou
        self.exist_threshold = exist_threshold
        self.use_prev_count = use_prev_count

    @torch.no_grad()
    def slot_probs(self, patch: dict) -> tuple[np.ndarray, np.ndarray]:
        """(probabilidade de cada pixel em cada canal (K, S, S), probabilidade de o canal existir (K,))."""
        x = torch.from_numpy(patch_to_input(patch, self.in_ch))[None].to(self.device)
        masks, ex = self.net(x)
        return torch.sigmoid(masks.float())[0].cpu().numpy(), torch.sigmoid(ex.float())[0].cpu().numpy()

    def _kept_slots(self, patch: dict):
        """(probs, [(canal, mascara)]): canais que existem (prob >= exist_threshold), com area minima, sem duplicata."""
        probs, exist = self.slot_probs(patch)
        mask = patch["mask"]
        # quantos vermes havia aqui no quadro anterior (0 = nao se sabe): a rede sozinha conta mal; com essa
        # dica mantem-se os n canais mais provaveis em vez de cortar por um limiar de existencia
        n_prev = min(prev_info(patch)[1], len(probs)) if self.use_prev_count else 0
        cand = []
        for k in range(len(probs)):
            m = (probs[k] > self.threshold) & mask
            if m.sum() >= self.min_area and (n_prev > 0 or exist[k] >= self.exist_threshold):
                cand.append((k, m, float(exist[k]) * m.sum(), float(exist[k])))
        if n_prev > 0:
            cand = sorted(cand, key=lambda t: -t[3])[:n_prev]
        cand = [t[:3] for t in cand]
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
