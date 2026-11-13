"""Segmentacao e rastreamento de C. elegans: parte 1 (PDI) + gancho para a parte 2 (CNN)."""
from pathlib import Path

from .config import Config, load_config
from .core import Env, Pipeline
from .io import build_inputs
from .stages import REGISTRY


def build_pipeline(cfg: Config) -> Pipeline:
    """Monta o pipeline na ordem de `cfg.stages`."""
    source, masks = build_inputs(cfg.input)
    env = Env(source=source, masks=masks, cfg=cfg, out_dir=Path(cfg.output.dir))
    stages = []
    for name in cfg.stages:
        if name not in REGISTRY:
            raise KeyError(f"estagio desconhecido: {name!r}. Disponiveis: {sorted(REGISTRY)}")
        stages.append(REGISTRY[name](cfg, env))
    return Pipeline(stages, env)
