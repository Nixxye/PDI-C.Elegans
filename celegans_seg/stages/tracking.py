"""PARTE 1 / Etapa E - Rastreamento: um ID e uma trajetoria por verme ao longo dos quadros.

Nao esta nos slides (a disciplina termina na segmentacao); e a ponte entre a segmentacao por
quadro e o objetivo do projeto ("traçar cada verme individualmente"). Usa:
  - associacao quadro a quadro pelo algoritmo hungaro (scipy.optimize.linear_sum_assignment)
  - predicao de posicao por velocidade constante (amortecida)
  - estado "occluded": se um verme some DENTRO de um amontoado, a trilha continua "voando"
    pela predicao em vez de morrer, e e retomada quando o amontoado se desfaz.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core import Stage

BIG = 1e6


@dataclass
class Track:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    area: float
    hits: int = 1
    missed: int = 0
    occluded_for: int = 0
    state: str = "visible"  # visible | occluded
    history: list = field(default_factory=list)  # (frame, y, x, state, area, ambiguous)


class Tracker(Stage):
    name = "tracking"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.tracks: dict[int, Track] = {}
        self.dead: list[Track] = []
        self.next_id = 1
        env.shared["tracker"] = self

    # ------------------------------------------------------------------ por quadro
    def process(self, ctx):
        c = self.cfg.tracking
        ids = list(self.tracks)
        dets = ctx.instances
        preds = {i: self.tracks[i].pos + self.tracks[i].vel for i in ids}
        assigned: dict[int, int] = {}  # indice da deteccao -> id da trilha
        if ids and dets:
            cost = np.full((len(ids), len(dets)), BIG)
            for a, i in enumerate(ids):
                t = self.tracks[i]
                for b, d in enumerate(dets):
                    dist = float(np.hypot(*(preds[i] - np.array(d.centroid))))
                    if dist > c.max_dist:
                        continue
                    dlog = min(abs(np.log(max(d.area, 1) / max(t.area, 1))), 1.0)
                    cost[a, b] = dist + c.area_weight * c.max_dist * dlog
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < BIG:
                    assigned[b] = ids[a]
        used = set(assigned.values())
        comp_cluster = {k.id: k.is_cluster for k in ctx.components}

        # trilhas com deteccao
        for b, tid in assigned.items():
            d, t = dets[b], self.tracks[tid]
            new = np.array(d.centroid, float)
            t.vel = c.velocity_decay * (new - t.pos) + (1 - c.velocity_decay) * t.vel
            t.pos, t.hits, t.missed, t.occluded_for, t.state = new, t.hits + 1, 0, 0, "visible"
            if not d.ambiguous:
                t.area = 0.8 * t.area + 0.2 * d.area
            d.track_id = tid
            t.history.append((ctx.index, new[0], new[1], "visible", d.area, d.ambiguous))

        # trilhas sem deteccao: ocluidas (dentro de amontoado) ou perdidas
        h, w = ctx.labels.shape
        for tid in ids:
            if tid in used:
                continue
            t = self.tracks[tid]
            p = preds[tid]
            py, px = int(round(p[0])), int(round(p[1]))
            inside = 0 <= py < h and 0 <= px < w and comp_cluster.get(int(ctx.labels[py, px]), False)
            if inside and t.occluded_for < c.max_occluded:
                t.state, t.occluded_for = "occluded", t.occluded_for + 1
                t.pos, t.vel = p, t.vel * 0.9
                t.history.append((ctx.index, p[0], p[1], "occluded", t.area, True))
            else:
                t.missed += 1
                if t.missed > c.max_missed:
                    self.dead.append(self.tracks.pop(tid))

        # deteccoes sem trilha -> trilha nova
        for b, d in enumerate(dets):
            if b in assigned:
                continue
            t = Track(self.next_id, np.array(d.centroid, float), np.zeros(2), float(d.area))
            t.history.append((ctx.index, d.centroid[0], d.centroid[1], "visible", d.area, d.ambiguous))
            self.tracks[t.id] = t
            d.track_id = t.id
            self.next_id += 1
        ctx.metrics["n_tracks_active"] = len(self.tracks)

    def confirmed(self, t: Track) -> bool:
        return t.hits >= self.cfg.tracking.min_hits

    # ------------------------------------------------------------------ fim
    def finish(self):
        out = self.env.out_dir
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "tracks.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame", "track_id", "y", "x", "state", "area", "ambiguous"])
            for t in sorted(self.dead + list(self.tracks.values()), key=lambda t: t.id):
                if not self.confirmed(t):
                    continue
                for f, y, x, st, a, amb in t.history:
                    wr.writerow([f, t.id, f"{y:.1f}", f"{x:.1f}", st, f"{a:.0f}", int(amb)])
