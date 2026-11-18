"""PARTE 1 / Etapa A - Pre-processamento (imagem -> imagem, "baixo nivel" da Aula 06).

Normalize -> Denoise -> BackgroundSubtraction.  Ao fim, `ctx.signal` e uma imagem em que o
verme e CLARO (valor alto) e o fundo e ~0, pronta para limiarizar.
"""
from __future__ import annotations

import numpy as np

from .. import ops
from ..core import Stage


class Normalize(Stage):
    """Escala para [0,1], garante 'verme escuro' e aplica gamma/autocontraste opcionais.

    - Gamma s = c r^gamma: [Aula 03, pag. 14]; comparacao das transformacoes: [Aula 03, pag. 17].
    - Autocontraste com saturacao: [Aula 03, pag. 17 e 23-24].  Por padrao DESLIGADO: aplicado
      quadro a quadro ele muda o brilho do fundo e atrapalha a subtracao de fundo.
    """

    name = "normalize"

    def process(self, ctx):
        c = self.cfg.normalize
        img = ctx.raw.astype(np.float32) / 255.0
        if not self.cfg.input.worms_dark:
            img = 1.0 - img
        if c.autocontrast:
            img = ops.autocontrast(img, c.low, c.high)
        if c.gamma != 1.0:
            img = ops.gamma_correct(img, c.gamma)
        ctx.image = img
        ctx.save_artifact("image", img)


class Denoise(Stage):
    """Suavizacao para reduzir granulacao do agar e ruido do sensor.

    - gaussian: passa-baixas [Aula 04, pag. 13-20]
    - median: ruido sal-e-pimenta [Aula 04, pag. 27-28]
    - bilateral: suaviza o fundo sem borrar o contorno do verme [Aula 05, pag. 7-15]
    """

    name = "denoise"

    def process(self, ctx):
        c = self.cfg.denoise
        if c.method == "gaussian":
            ctx.image = ops.gaussian(ctx.image, c.sigma)
        elif c.method == "median":
            ctx.image = ops.median(ctx.image, c.median_size)
        elif c.method == "bilateral":
            ctx.image = ops.bilateral(ctx.image, c.bilateral_sigma_s, c.bilateral_sigma_r)
        elif c.method != "none":
            raise ValueError(f"denoise.method desconhecido: {c.method}")
        ctx.save_artifact("denoised", ctx.image)


class BackgroundSubtraction(Stage):
    """Estima o fundo e produz `ctx.signal` (verme claro sobre ~0).

    - median: o fundo da placa e estatico e os vermes se movem, entao a MEDIANA temporal de varios
      quadros remove os vermes (ideia do projeto; nao aparece nos slides). Limite: verme parado
      por toda a amostra entra no fundo.
    - tophat: transformada top-hat para uniformizar a iluminacao, sem precisar de varios quadros
      [Aula 08, pag. 62-71; pipeline completo Imagem->Top-hat->Limiarizacao->Morfologia->Labeling,
      pag. 71].  Alternativa em frequencia: filtro homomorfico [Aula 07, pag. 56-64] (nao implementado).
    - none: signal = 1 - imagem.
    """

    name = "background"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.bg = None

    @property
    def needs_fit(self):  # type: ignore[override]
        return self.cfg.background.method == "median"

    def fit(self, samples):
        # percentil 50 = mediana; vermes sao escuros, entao um percentil mais alto escolhe, em cada pixel,
        # o valor de quando NAO havia verme (menos chance de absorver um verme que fica parado)
        self.bg = np.percentile(np.stack([s.image for s in samples]), self.cfg.background.percentile, axis=0).astype(np.float32)
        self.env.shared["background"] = self.bg

    def process(self, ctx):
        m = self.cfg.background.method
        if m == "median":
            sig = np.clip(self.bg - ctx.image, 0, 1)  # so o que ficou mais ESCURO que o fundo
        elif m == "tophat":
            sig = ops.black_tophat(ctx.image, self.cfg.background.tophat_size)
        elif m == "none":
            sig = 1.0 - ctx.image
        else:
            raise ValueError(f"background.method desconhecido: {m}")
        ctx.signal = sig.astype(np.float32)
        ctx.save_artifact("signal", ctx.signal)
        if self.cfg.threshold.grow_dist > 0 and m == "median":
            ctx.signal_spatial = ops.black_tophat(ctx.image, self.cfg.background.tophat_size).astype(np.float32)
            ctx.save_artifact("signal_spatial", ctx.signal_spatial)
