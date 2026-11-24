"""Registro dos estagios: nome (usado em `stages:` do config.yaml) -> classe.

Para reordenar, remover ou trocar estagios basta editar a lista `stages` do config.
Para criar um estagio novo: herde de `core.Stage`, implemente `process(ctx)` e registre aqui.
"""
from .analysis import ComponentAnalysis
from .evaluate import Evaluate
from .export import ClusterPatchExporter
from .instances import InstanceBuilder
from .preprocess import BackgroundSubtraction, Denoise, Normalize
from .refine import CNNRefiner, NullRefiner
from .segment import Labeling, Morphology, Threshold
from .timeline import Timeline
from .viz import Visualizer

REGISTRY = {
    "normalize": Normalize,
    "denoise": Denoise,
    "background": BackgroundSubtraction,
    "threshold": Threshold,
    "morphology": Morphology,
    "labeling": Labeling,
    "analysis": ComponentAnalysis,
    "instances": InstanceBuilder,
    "refine": lambda cfg, env: CNNRefiner(cfg, env) if cfg.refine.kind == "cnn" else NullRefiner(cfg, env),
    "timeline": Timeline,
    "evaluate": Evaluate,
    "export": ClusterPatchExporter,
    "visualize": Visualizer,
}
