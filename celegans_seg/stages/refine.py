"""PARTE 2 (gancho) - Onde a CNN entra.

O pipeline classico termina com hipoteses de instancias (`ctx.instances`). Um `InstanceRefiner`
pode substituir a separacao dos amontoados pela predicao de uma CNN, sem tocar no resto.

Contrato (o que a CNN recebe e devolve) - os mesmos arrays que `export.py` grava no dataset:
  entrada : recorte quadrado (patch_size x patch_size) centrado no amontoado, com
            `image` (cinza), `signal` (sinal pos-fundo), `mask` (mascara binaria do amontoado),
            `dist` (transformada de distancia), `skeleton`, `ws_labels` (proposta do watershed)
  saida   : mapa de rotulos de instancia (patch_size x patch_size, 0 = fundo, 1..k = vermes),
            COM sobreposicao permitida se a rede for do tipo "uma mascara por verme".
Ao implementar, sobrescreva `refine_patch` e troque `refine.kind` no config.yaml.
"""
from __future__ import annotations

import numpy as np

from ..core import Stage


class InstanceRefiner(Stage):
    """Base: percorre os amontoados e pede a `refine_patch` um novo mapa de instancias."""

    name = "refine"

    def refine_patch(self, patch: dict) -> np.ndarray | None:
        """Devolve mapa de rotulos (S x S) ou None para manter a hipotese do watershed."""
        raise NotImplementedError

    def process(self, ctx):  # pragma: no cover - sera usado pela CNN
        raise NotImplementedError("Implemente a insercao dos novos rotulos em ctx.inst_labels/ctx.instances")


class NullRefiner(InstanceRefiner):
    """Padrao: nao faz nada (mantem o resultado do watershed)."""

    def refine_patch(self, patch):
        return None

    def process(self, ctx):
        return
