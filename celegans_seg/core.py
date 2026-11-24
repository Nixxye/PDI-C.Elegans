"""Nucleo do pipeline: contexto de quadro, estagio base e orquestrador.

Estilo de pipeline: cada operacao e um `Stage` com `process(ctx)`. O `FrameContext` carrega o
quadro e tudo que os estagios vao produzindo. Estagios que precisam olhar a sequencia inteira
antes (fundo, area de referencia) implementam `fit(samples)`.

Esta organizacao reflete o diagrama de tres niveis da [Aula 06, pag. 1-3 de Parte 1]:
baixo nivel (imagem -> imagem), medio nivel (extracao de atributos) e alto nivel (interpretacao).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Component:
    """Regiao conexa da mascara (saida da rotulagem + analise)."""
    id: int
    bbox: tuple  # (r0, c0, r1, c1) exclusivo
    area: int
    centroid: tuple  # (y, x)
    skel_len: int = 0
    n_ends: int = 0
    n_junctions: int = 0
    thickness: float = 0.0  # 2 * maximo da transformada de distancia
    mean_width: float = 0.0  # area / comprimento do esqueleto: largura media do componente (px)
    area_ratio: float = 1.0
    n_est: int = 1  # numero estimado de vermes
    is_cluster: bool = False


@dataclass
class Instance:
    """Um verme (ou hipotese de verme) depois da separacao."""
    id: int  # valor em ctx.inst_labels
    component_id: int
    source: str  # single | cluster (amontoado nao separado) | cnn
    area: int
    centroid: tuple
    ambiguous: bool = False  # veio de um amontoado: separacao menos confiavel
    track: int = 0  # id do verme na linha do tempo (estagio `timeline`); 0 = nao rastreado


@dataclass
class FrameContext:
    index: int
    name: str
    raw: np.ndarray  # uint8 (H, W)
    keep_artifacts: bool = True
    image: np.ndarray | None = None  # float32 [0,1], verme escuro
    signal: np.ndarray | None = None  # float32 [0,1], verme CLARO (alto = verme)
    signal_spatial: np.ndarray | None = None  # idem, mas por top-hat (sem fundo temporal): completa vermes parados
    mask: np.ndarray | None = None  # bool
    labels: np.ndarray | None = None  # int32, um rotulo por componente conexo
    components: list = field(default_factory=list)
    cluster_mask: np.ndarray | None = None  # bool: pixels dos amontoados (a mascara que vai para a CNN)
    inst_labels: np.ndarray | None = None  # int32, um rotulo por verme
    instances: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)  # imagens intermediarias (para figuras)
    metrics: dict = field(default_factory=dict)

    def save_artifact(self, name: str, arr: np.ndarray):
        if self.keep_artifacts:
            self.artifacts[name] = arr


@dataclass
class Env:
    """O que os estagios compartilham: fonte de dados, config, pasta de saida."""
    source: object
    masks: object | None
    cfg: object
    out_dir: Path
    shared: dict = field(default_factory=dict)


class Stage:
    """Contrato de um estagio."""
    name = "stage"
    needs_fit = False

    def __init__(self, cfg, env: Env):
        self.cfg, self.env = cfg, env

    def fit(self, samples: list):
        """Opcional: aprende algo da sequencia a partir de quadros de amostra ja processados."""

    def process(self, ctx: FrameContext):
        raise NotImplementedError

    def finish(self):
        """Opcional: chamado uma vez ao fim da sequencia (gravar CSVs etc.)."""


class Pipeline:
    def __init__(self, stages: list[Stage], env: Env):
        self.stages, self.env = stages, env

    def make_context(self, k: int, keep_artifacts: bool = True) -> FrameContext:
        src = self.env.source
        return FrameContext(index=k, name=src.name(k), raw=src.frame(k), keep_artifacts=keep_artifacts)

    def fit(self, n_samples: int = 25):
        """Passa quadros de amostra pelos estagios, chamando `fit` quando necessario.

        As amostras avancam estagio a estagio: o `fit` de um estagio ve amostras ja processadas
        por todos os anteriores (ex.: a area de referencia e medida sobre a mascara rotulada).
        """
        fit_idx = [i for i, s in enumerate(self.stages) if s.needs_fit]
        if not fit_idx:
            return
        n = len(self.env.source)
        ks = np.unique(np.linspace(0, n - 1, min(n_samples, n)).astype(int))
        samples = [self.make_context(int(k), keep_artifacts=False) for k in ks]
        for i, stage in enumerate(self.stages[: max(fit_idx) + 1]):
            if stage.needs_fit:
                stage.fit(samples)
            for ctx in samples:
                stage.process(ctx)

    def process_frame(self, k: int) -> FrameContext:
        ctx = self.make_context(k)
        for stage in self.stages:
            stage.process(ctx)
        return ctx

    def run(self) -> Iterator[FrameContext]:
        for k in range(len(self.env.source)):
            yield self.process_frame(k)
        for stage in self.stages:
            stage.finish()
