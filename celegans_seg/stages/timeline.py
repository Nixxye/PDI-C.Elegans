"""Linha do tempo de cada verme: liga as mascaras de quadros vizinhos pela SIMILARIDADE entre elas.

Usa as medidas de comparacao de imagens da Aula 05 (pag. 18-22): SSIM (principal), MSE e PSNR. A ideia: o mesmo verme,
em quadros seguidos, tem mascaras de forma parecida; vermes diferentes, nao.

Para cada verme do quadro atual e cada trilha ativa (verme visto nos ultimos `max_gap`+1 quadros):
  1. so compara se os centroides estao a ate `max_dist` px (x quadros de ausencia);
  2. recorta uma janela `window` x `window` centrada no centroide de CADA mascara (alinha pela posicao, assim o verme
     que andou continua comparavel) e calcula o SSIM entre as duas mascaras suavizadas, medido so em volta delas;
  3. custo = -(SSIM - dist_weight * distancia / limite); a atribuicao otima (hungaro, como no pareamento por IoU)
     liga cada verme a no maximo uma trilha, e so se SSIM >= `min_ssim`.
Quem nao liga a ninguem abre uma trilha nova. Trilha que some por mais de `max_gap` quadros e encerrada.
Um amontoado nao separado (`source="cluster"`) e uma instancia so: ele segura uma trilha, as outras ficam "ausentes"
ate a CNN (ou o fim do contato) separar os vermes e elas serem religadas pelo SSIM.

Saidas em `<out>/`: timeline.csv (uma linha por verme por quadro: trilha, posicao, area, SSIM/MSE/PSNR com o quadro
anterior da trilha), tracks.csv (resumo por trilha) e timeline.png (cada trilha como uma barra ao longo dos quadros).
`Instance.track` recebe o id da trilha (a visualizacao passa a colorir por trilha).
"""
from __future__ import annotations

import csv

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .. import ops
from ..core import Stage
from .export import crop_center


class Timeline(Stage):
    name = "timeline"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.active = {}  # id da trilha -> dict(mask, centroid, last)
        self.next_id = 1
        self.rows = []

    def similarity(self, ma: np.ndarray, ca: tuple, mb: np.ndarray, cb: tuple) -> dict:
        """SSIM/MSE/PSNR entre duas mascaras. Duas leituras do SSIM:
        `ssim_forma`: cada mascara recortada ao redor do proprio centroide (compara a FORMA, mesmo que o verme tenha
        andado); `ssim_pos`: as duas na mesma janela, sem alinhar (alto so se as mascaras se sobrepoem, que e o
        que acontece quando o verme anda pouco entre quadros). `ssim` = a maior das duas."""
        W = self.cfg.timeline.window
        ia = tuple(int(round(v)) for v in ca)
        ib = tuple(int(round(v)) for v in cb)
        mid = ((ia[0] + ib[0]) // 2, (ia[1] + ib[1]) // 2)
        blur = lambda x: ndi.gaussian_filter(x.astype(np.float32), 1.0)  # noqa: E731
        a, b = crop_center(ma, *ia, W), crop_center(mb, *ib, W)
        pa, pb = crop_center(ma, *mid, W), crop_center(mb, *mid, W)
        shape = ops.ssim(blur(a), blur(b), roi=ndi.binary_dilation(a | b, iterations=4))
        pos = ops.ssim(blur(pa), blur(pb), roi=ndi.binary_dilation(pa | pb, iterations=4))
        return dict(ssim=max(shape, pos), ssim_forma=shape, ssim_pos=pos, mse=ops.mse(pa, pb), psnr=ops.psnr(pa, pb))

    def process(self, ctx):
        t = self.cfg.timeline
        insts = [i for i in ctx.instances if (ctx.inst_labels == i.id).any()]
        masks = [ctx.inst_labels == i.id for i in insts]
        for tid in [k for k, v in self.active.items() if ctx.index - v["last"] > t.max_gap + 1]:
            del self.active[tid]  # sumiu por tempo demais: trilha encerrada
        tids = list(self.active)
        score = np.full((len(insts), len(tids)), -np.inf)
        sims = {}
        for i, (inst, m) in enumerate(zip(insts, masks)):
            for j, tid in enumerate(tids):
                tr = self.active[tid]
                lim = t.max_dist * (ctx.index - tr["last"])
                d = float(np.hypot(inst.centroid[0] - tr["centroid"][0], inst.centroid[1] - tr["centroid"][1]))
                if d > lim:
                    continue
                s = self.similarity(m, inst.centroid, tr["mask"], tr["centroid"])
                if s["ssim"] >= t.min_ssim:
                    score[i, j] = s["ssim_forma"] + s["ssim_pos"] - t.dist_weight * d / lim
                    sims[i, j] = s
        assigned = {}
        if insts and tids:
            ri, ci = linear_sum_assignment(-np.where(np.isfinite(score), score, -1e6))
            assigned = {int(i): int(j) for i, j in zip(ri, ci) if np.isfinite(score[i, j])}
        for i, (inst, m) in enumerate(zip(insts, masks)):
            if i in assigned:
                tid, s = tids[assigned[i]], sims[i, assigned[i]]
            else:
                tid, s = self.next_id, None
                self.next_id += 1
            inst.track = tid
            self.active[tid] = dict(mask=m, centroid=inst.centroid, last=ctx.index)
            self.rows.append(dict(
                frame=ctx.index, name=ctx.name, track=tid, instance=inst.id, source=inst.source,
                cy=round(inst.centroid[0], 1), cx=round(inst.centroid[1], 1), area=inst.area,
                ssim="" if s is None else round(s["ssim"], 4), mse="" if s is None else round(s["mse"], 5),
                psnr="" if s is None else round(min(s["psnr"], 99.0), 2)))
        ctx.metrics["n_tracks_active"] = len(self.active)

    def finish(self):
        if not self.rows:
            return
        out = self.env.out_dir
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "timeline.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            wr.writeheader()
            wr.writerows(self.rows)
        tracks = {}
        for r in self.rows:
            tracks.setdefault(r["track"], []).append(r)
        summary = []
        for tid, rs in sorted(tracks.items()):
            pts = np.array([[r["cy"], r["cx"]] for r in rs])
            ss = [r["ssim"] for r in rs if r["ssim"] != ""]
            summary.append(dict(
                track=tid, first_frame=rs[0]["frame"], last_frame=rs[-1]["frame"], n_frames=len(rs),
                n_missing=rs[-1]["frame"] - rs[0]["frame"] + 1 - len(rs),
                path_px=round(float(np.hypot(*np.diff(pts, axis=0).T).sum()), 1) if len(pts) > 1 else 0.0,
                mean_ssim=round(float(np.mean(ss)), 4) if ss else "",
                frames_as_cluster=sum(r["source"] == "cluster" for r in rs)))
        with open(out / "tracks.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(summary[0]))
            wr.writeheader()
            wr.writerows(summary)
        self._plot(out / "timeline.png", summary)

    def _plot(self, path, summary):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .viz import id_color

        keep = [s for s in summary if s["n_frames"] >= self.cfg.timeline.min_track_len]
        keep.sort(key=lambda s: (s["first_frame"], s["track"]))
        fig, ax = plt.subplots(figsize=(11, max(3, 0.12 * len(keep) + 1.5)))
        for row, s in enumerate(keep):
            c = tuple(v / 255 for v in id_color(s["track"]))
            ax.barh(row, s["last_frame"] - s["first_frame"] + 1, left=s["first_frame"], height=0.8, color=c)
        ax.set_xlabel("quadro")
        ax.set_ylabel(f"verme (trilha), {len(keep)} com >= {self.cfg.timeline.min_track_len} quadros")
        ax.set_yticks([])
        ax.set_title("Linha do tempo de cada verme (ligacao por SSIM entre mascaras)")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
