"""Linha do tempo: liga mascaras de quadros vizinhos pelo SSIM [Aula 05, pag. 19-22].
Saidas: timeline.csv, tracks.csv, timeline.png."""
from __future__ import annotations

import csv

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .. import ops
from ..core import Instance, Stage
from .export import crop_center


class Timeline(Stage):
    name = "timeline"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.active = {}  # id da trilha -> dict(mask, centroid, last)
        self.next_id = 1
        self.rows = []
        self.back = {}

    def similarity(self, ma: np.ndarray, ca: tuple, mb: np.ndarray, cb: tuple) -> dict:
        """SSIM/MSE/PSNR entre duas mascaras (forma e posicao)."""
        t = self.cfg.timeline
        W = t.window
        ia = tuple(int(round(v)) for v in ca)
        ib = tuple(int(round(v)) for v in cb)
        mid = ((ia[0] + ib[0]) // 2, (ia[1] + ib[1]) // 2)
        blur = lambda x: ndi.gaussian_filter(x.astype(np.float32), t.blur_sigma)  # noqa: E731
        a, b = crop_center(ma, *ia, W), crop_center(mb, *ib, W)
        pa, pb = crop_center(ma, *mid, W), crop_center(mb, *mid, W)
        shape = ops.ssim(blur(a), blur(b), t.ssim_sigma, ndi.binary_dilation(a | b, iterations=t.roi_margin))
        pos = ops.ssim(blur(pa), blur(pb), t.ssim_sigma, ndi.binary_dilation(pa | pb, iterations=t.roi_margin))
        return dict(ssim=max(shape, pos), ssim_forma=shape, ssim_pos=pos, mse=ops.mse(pa, pb), psnr=ops.psnr(pa, pb))

    def split_by_tracks(self, ctx):
        """Mascara que cobre 2+ trilhas e maior que elas (verme fundido; vira source="cnn"): cada pixel vai para a trilha mais proxima
        [distancia ate a mascara anterior, Aula 01c, pag. 11]."""
        t = self.cfg.timeline
        lab, nxt, new = ctx.inst_labels, int(ctx.inst_labels.max()) + 1, []
        recent = {k: v for k, v in self.active.items() if ctx.index - v["last"] <= t.max_gap}
        for inst in list(ctx.instances):
            m = lab == inst.id
            cov = [k for k, v in recent.items() if v["mask"].any() and (m & v["mask"]).sum() >= t.split_cover * v["mask"].sum()]
            if len(cov) < 2 or inst.area < t.split_ratio * max(recent[k]["mask"].sum() for k in cov):
                continue
            owner = np.stack([ndi.distance_transform_edt(~recent[k]["mask"]) for k in cov]).argmin(0)
            parts = [m & (owner == n) for n in range(len(cov))]
            parts = [ndi.binary_opening(p) if p.sum() > 60 else p for p in parts]
            if sum(p.sum() >= t.split_min_part for p in parts) < 2:
                continue
            lab[m] = 0
            ctx.instances.remove(inst)
            for p in parts:
                if p.sum() < t.split_min_part:
                    continue
                ys, xs = np.nonzero(p)
                lab[p] = nxt
                new.append(Instance(id=nxt, component_id=inst.component_id, source="cnn", area=int(p.sum()),
                                    centroid=(ys.mean(), xs.mean()), ambiguous=True))
                nxt += 1
        ctx.instances.extend(new)

    def hide_and_return(self, ctx, insts, masks, tids, assigned):
        """Trilha sem mascara propria que estava dentro da mascara de outra (amontoado) fica oculta; mascara nova perto
        do amontoado, com area parecida, volta com o id da trilha oculta."""
        t = self.cfg.timeline
        used = {tids[j] for j in assigned.values()}
        for tid in tids:
            tr = self.active[tid]
            if tid in used or ctx.index - tr["last"] > 1 or not tr["mask"].any():
                continue
            for i, m in enumerate(masks):
                if i in assigned and (m & tr["mask"]).sum() >= t.hide_cover * tr["mask"].sum():
                    tr["hidden"] = ctx.index
                    tr["host"] = tids[assigned[i]]
                    break
        hid = [k for k, v in self.active.items() if "hidden" in v and v["hidden"] < ctx.index]
        free = [i for i in range(len(insts)) if i not in assigned]
        self.back = {}
        if not hid or not free:
            return
        cost = np.full((len(free), len(hid)), 1e6)
        for a, i in enumerate(free):
            for b, k in enumerate(hid):
                tr = self.active[k]
                ratio = insts[i].area / max(tr["ref_area"], 1)
                if not t.hide_min_ratio <= ratio <= t.hide_max_ratio:
                    continue
                host = self.active.get(tr["host"])
                ref = host["mask"] if host is not None else tr["mask"]
                near = float(ndi.distance_transform_edt(~ref)[masks[i]].min())
                if near <= t.hide_near:
                    cost[a, b] = abs(np.log(ratio)) + 0.5 * near / t.hide_near
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 1e6:
                assigned[free[a]] = -1
                self.back[free[a]] = hid[b]
                self.active[hid[b]].pop("hidden")
                self.active[hid[b]].pop("host", None)

    def process(self, ctx):
        t = self.cfg.timeline
        if t.split_by_tracks:
            self.split_by_tracks(ctx)
        insts = [i for i in ctx.instances if (ctx.inst_labels == i.id).any()]
        masks = [ctx.inst_labels == i.id for i in insts]
        for tid in [k for k, v in self.active.items() if "hidden" not in v and ctx.index - v["last"] > t.max_gap + 1
                    or "hidden" in v and ctx.index - v["hidden"] > t.hide_max_frames]:
            del self.active[tid]  # sumiu por tempo demais: trilha encerrada
        tids = [k for k, v in self.active.items() if "hidden" not in v]
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
        if t.hide_merged:
            self.hide_and_return(ctx, insts, masks, tids, assigned)
        for i, (inst, m) in enumerate(zip(insts, masks)):
            if i in assigned and assigned[i] >= 0:
                tid, s = tids[assigned[i]], sims[i, assigned[i]]
            elif i in assigned:
                tid, s = self.back[i], None
            else:
                tid, s = self.next_id, None
                self.next_id += 1
            inst.track = tid
            old = self.active.get(tid)
            ref = inst.area if old is None else (0.8 * old["ref_area"] + 0.2 * inst.area
                                                 if inst.area <= 1.3 * old["ref_area"] else old["ref_area"])
            self.active[tid] = dict(mask=m, centroid=inst.centroid, last=ctx.index, ref_area=ref)
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
