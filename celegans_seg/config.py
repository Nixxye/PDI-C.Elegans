"""Configuracao do projeto (dataclasses + YAML).

Tudo que se quer mudar sem mexer no codigo (pasta de entrada, limiares, ordem dos estagios)
fica aqui e em `config.yaml`. Qualquer campo pode ser sobrescrito na linha de comando com
`--set secao.campo=valor` (ex.: `--set threshold.min_threshold=30`).

Convencao de citacao dos slides em todo o projeto:
    [Aula NN, pag. P]  ->  pagina P do PDF da aula NN em Estudos/PDI/Slides
(a pagina e o indice da pagina no arquivo PDF, que pode diferir do numero impresso no slide).
"""
from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

import yaml


@dataclass
class InputCfg:
    # Opcao A: pasta com os quadros de UMA sequencia.
    images: str = ""
    masks: str = ""  # opcional (gabarito, so para avaliar e para gerar dados da CNN)
    # Opcao B: raiz do dataset no formato CSB-1 (root/images/<seq>/k.png, root/masks/<seq>/k_i.png).
    dataset_root: str = ""
    sequence: str = ""
    pattern: str = "*.png"
    mask_pattern: str = "{stem}_*.png"  # mascaras do quadro "12.png" -> "12_*.png"
    start: int = 0
    end: int = -1  # exclusivo; -1 = ate o fim
    step: int = 1
    worms_dark: bool = True  # campo claro: verme escuro sobre fundo claro
    # Recortes com moldura preta de preenchimento (ex.: Zenodo 15526615): regioes de valor 0 que tocam
    # a borda sao tratadas como "sem dado" (senao parecem vermes: sao mais escuras que eles).


@dataclass
class NormalizeCfg:
    gamma: float = 1.0  # [Aula 03, pag. 14] s = c * r^gamma
    autocontrast: bool = False  # [Aula 03, pag. 17] stretchlim/imadjust (por quadro: cuidado com o fundo)
    low: float = 0.01
    high: float = 0.99


@dataclass
class DenoiseCfg:
    method: str = "gaussian"  # gaussian | median | bilateral | none
    sigma: float = 1.0  # [Aula 04, pag. 13-20]
    median_size: int = 3  # [Aula 04, pag. 27]
    bilateral_sigma_s: float = 2.0  # [Aula 05, pag. 7-8]
    bilateral_sigma_r: float = 0.1


@dataclass
class BackgroundCfg:
    method: str = "median"  # median (fundo por mediana temporal) | tophat | none
    n_samples: int = 25
    percentile: float = 50.0  # 50 = mediana. Maior (ex.: 80) = fundo mais claro: nao absorve verme que fica parado
    tophat_size: int = 31  # [Aula 08, pag. 62-71]; deve ser maior que a largura do verme


@dataclass
class ThresholdCfg:
    method: str = "otsu"  # otsu | adaptive
    gain: float = 4.0  # realce de contraste antes do Otsu
    min_threshold: int = 40  # piso (0-255) para quadros sem vermes
    adaptive_window: int = 51  # [Aula 08, pag. 14]
    adaptive_offset: int = 20  # (0-255) acima da media local
    grow_dist: int = 0  # 0 = desligado. Completa o verme com o sinal espacial (top-hat) ate N px das sementes
    grow_seed_area: int = 150  # so componentes com pelo menos esta area servem de semente
    spatial_min_area: int = 0  # 0 = desligado. Componentes do top-hat com pelo menos esta area entram mesmo sem semente temporal (verme parado)
    grow_open: int = 0  # raio da abertura do trecho acrescentado: tira espinhos finos (textura do agar junto do verme)
    grow_thr: float = 1.0  # limiar do sinal espacial = grow_thr x limiar do sinal temporal


@dataclass
class MorphCfg:
    opening_radius: int = 1  # [Aula 08, pag. 24-25]
    closing_radius: int = 1  # [Aula 08, pag. 26-27]
    fill_holes: bool = True


@dataclass
class LabelCfg:
    min_area: int = 80  # componentes menores sao ruido
    bridge_gap: int = 0  # 0 = desligado. Liga componentes grandes separados por ate N px (verme partido em dois)
    bridge_min_area: int = 150  # so liga pedacos com pelo menos esta area (evita unir sujeira)


@dataclass
class AnalysisCfg:
    cluster_ratio: float = 1.5  # area/area_de_referencia a partir da qual e "amontoado"
    ref_area: float = 0.0  # 0 = estimar pela mediana das areas nos quadros de amostra
    max_worms: int = 6
    min_mean_width: float = 5.0  # componente mais fino que isso (px) e sujeira do agar, nao verme (0 = desliga)
    max_mean_width: float = 40.0  # mais largo que isso e mancha/bolha (0 = desliga); verme enrolado em bola fica < 33


@dataclass
class RefineCfg:
    kind: str = "none"  # none | cnn (separa os amontoados com a CNN de celegans_seg/cnn.py)
    weights: str = "runs/cnn_train/checkpoint.pth"  # checkpoint da Mask R-CNN (tools/train_maskrcnn.ps1)
    score_thr: float = 0.3  # confianca minima da deteccao do verme
    threshold: float = 0.5  # limiar de pixel da mascara prevista
    min_area: int = 20  # canais com menos pixels que isso sao descartados
    max_window: int = 448  # maior janela (px) para amontoados que nao cabem em export.patch_size; acima disso fica sem separar


@dataclass
class TimelineCfg:
    window: int = 128  # lado (px) do recorte centrado no centroide de cada mascara, onde o SSIM e calculado
    max_dist: float = 60.0  # px: so compara mascaras com centroides a ate isto (x quadros sem aparecer + 1)
    min_ssim: float = 0.35  # SSIM minimo para ligar duas mascaras ao mesmo verme
    dist_weight: float = 0.3  # peso da distancia no custo: custo = -(SSIM - dist_weight * d / max_dist)
    max_gap: int = 2  # quadros seguidos que um verme pode sumir (oclusao, falha) e ainda manter o id
    min_track_len: int = 3  # trilhas mais curtas que isso nao entram na figura da linha do tempo


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
    normalize: NormalizeCfg = field(default_factory=NormalizeCfg)
    denoise: DenoiseCfg = field(default_factory=DenoiseCfg)
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    threshold: ThresholdCfg = field(default_factory=ThresholdCfg)
    morphology: MorphCfg = field(default_factory=MorphCfg)
    labeling: LabelCfg = field(default_factory=LabelCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    refine: RefineCfg = field(default_factory=RefineCfg)
    timeline: TimelineCfg = field(default_factory=TimelineCfg)
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
