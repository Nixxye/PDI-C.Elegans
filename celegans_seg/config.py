"""Configuracao (dataclasses + config.yaml). Qualquer campo muda na linha de comando com `--set secao.campo=valor`.
Citacao [Aula NN, pag. P] = pagina do PDF da aula. Origem e escolha de cada parametro: docs/ESTUDO_PDI.md."""
from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

import yaml


@dataclass
class InputCfg:
    images: str = ""  # pasta com os quadros de uma sequencia
    masks: str = ""  # gabarito (opcional)
    dataset_root: str = ""  # alternativa: raiz com images/<seq> e masks/<seq>
    sequence: str = ""
    pattern: str = "*.png"
    mask_pattern: str = "{stem}_*.png"
    start: int = 0
    end: int = -1  # -1 = ate o fim
    step: int = 1
    worms_dark: bool = True  # verme escuro sobre fundo claro


@dataclass
class DenoiseCfg:
    sigma: float = 1.0  # gaussiano [Aula 04, pag. 16-19]


@dataclass
class BackgroundCfg:
    n_samples: int = 25  # quadros usados no fundo
    percentile: float = 70.0  # 50 = mediana
    tophat_size: int = 31  # SE do top-hat [Aula 08, pag. 66]


@dataclass
class ThresholdCfg:
    gain: float = 4.0  # ganho antes do Otsu
    min_threshold: int = 40  # piso do limiar (0-255) [Aula 08, pag. 10-13]
    grow_dist: int = 15  # px; completa a cauda com o top-hat (0 = desliga)
    grow_seed_area: int = 150  # area minima da semente
    grow_open: int = 3  # raio da abertura do trecho acrescentado
    grow_thr: float = 1.0  # limiar do top-hat = grow_thr x Otsu
    spatial_min_area: int = 300  # verme parado entra pelo top-hat (0 = desliga)


@dataclass
class MorphCfg:
    opening_radius: int = 1  # [Aula 08, pag. 24-25]
    closing_radius: int = 1  # [Aula 08, pag. 26-27]
    fill_holes: bool = True


@dataclass
class LabelCfg:
    min_area: int = 150  # px [Aula 08, pag. 50-60]
    bridge_gap: int = 8  # px; liga pedacos de verme partido (0 = desliga)
    bridge_min_area: int = 150


@dataclass
class AnalysisCfg:
    cluster_ratio: float = 1.5  # amontoado = area >= ratio x referencia
    ref_area: float = 0.0  # 0 = mediana dos quadros de amostra
    max_worms: int = 6
    min_mean_width: float = 5.0  # px (0 = desliga)
    max_mean_width: float = 40.0  # px (0 = desliga)
    fallback_ref_area: float = 400.0  # px


@dataclass
class RefineCfg:
    kind: str = "none"  # none | cnn
    weights: str = "runs/cnn_train/checkpoint.pth"
    score_thr: float = 0.3
    threshold: float = 0.5
    min_area: int = 20
    window_margin: int = 16
    max_window: int = 448


@dataclass
class TimelineCfg:
    window: int = 128
    max_dist: float = 60.0
    min_ssim: float = 0.35
    dist_weight: float = 0.3
    max_gap: int = 2
    blur_sigma: float = 1.0
    roi_margin: int = 4
    ssim_sigma: float = 1.5  # [Aula 05, pag. 19-22]
    min_track_len: int = 3


@dataclass
class EvaluateCfg:
    iou_match: float = 0.5
    no_gt_fraction: float = 0.3
    border_margin: int = 2
    worm_area_ratio: list = field(default_factory=lambda: [0.7, 1.4])
    worm_mean_width: list = field(default_factory=lambda: [6.5, 10.5])
    worm_min_contrast: float = 0.2
    small_min_area: int = 150
    small_min_contrast: float = 0.1


@dataclass
class ExportCfg:
    patch_size: int = 128
    only_clusters: bool = True
    save_every: int = 1


@dataclass
class VizCfg:
    save_every: int = 1
    gif: bool = True
    gif_step: int = 2
    debug_max_panels: int = 3


@dataclass
class OutputCfg:
    dir: str = "runs/last"


DEFAULT_STAGES = [
    "normalize", "denoise", "background", "threshold", "morphology", "labeling",
    "analysis", "instances", "refine", "timeline", "evaluate", "export", "visualize",
]


@dataclass
class Config:
    input: InputCfg = field(default_factory=InputCfg)
    denoise: DenoiseCfg = field(default_factory=DenoiseCfg)
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    threshold: ThresholdCfg = field(default_factory=ThresholdCfg)
    morphology: MorphCfg = field(default_factory=MorphCfg)
    labeling: LabelCfg = field(default_factory=LabelCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    refine: RefineCfg = field(default_factory=RefineCfg)
    timeline: TimelineCfg = field(default_factory=TimelineCfg)
    evaluate: EvaluateCfg = field(default_factory=EvaluateCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
    viz: VizCfg = field(default_factory=VizCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    stages: list = field(default_factory=lambda: list(DEFAULT_STAGES))


def _update(obj, data: dict):
    for key, val in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"campo desconhecido na configuracao: {key!r} em {type(obj).__name__}")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            _update(cur, val or {})
        else:
            setattr(obj, key, val)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Le o YAML (se houver) e aplica sobrescritas `secao.campo=valor`."""
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            _update(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        section, _, name = key.partition(".")
        target = getattr(cfg, section)
        if name:
            current = getattr(target, name)
            setattr(target, name, yaml.safe_load(raw) if not isinstance(current, str) else raw)
        else:
            setattr(cfg, section, yaml.safe_load(raw))
    return cfg
