"""Testa a ligacao PDI <-> CNN sem precisar de rede: um separador falso divide o amontoado em duas metades."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from celegans_seg import load_config  # noqa: E402
from celegans_seg.cnn import ClusterSeparator  # noqa: E402
from celegans_seg.core import Env, FrameContext, Instance  # noqa: E402
from celegans_seg.stages.export import crop_center  # noqa: E402
from celegans_seg.stages.refine import CNNRefiner, paste_center  # noqa: E402


class FakeSeparator(ClusterSeparator):
    """Divide a mascara do amontoado em esquerda e direita (so para testar a cola)."""

    def predict_masks(self, patch):
        m = patch["mask"]
        cols = np.nonzero(m.any(axis=0))[0]
        mid = (cols.min() + cols.max()) // 2
        left, right = m.copy(), m.copy()
        left[:, mid:] = False
        right[:, :mid] = False
        return [left, right]


def cluster_context(shape=(260, 300)):
    """Quadro com um amontoado (retangulo) no meio, como o PDI o entrega ao estagio refine."""
    labels = np.zeros(shape, np.int32)
    labels[100:130, 90:210] = 1
    inst = Instance(id=1, component_id=1, source="cluster", area=int((labels == 1).sum()), centroid=(114.5, 149.5),
                    ambiguous=True)
    ctx = FrameContext(index=0, name="0", raw=np.full(shape, 120, np.uint8))
    ctx.signal = np.where(labels > 0, 0.2, 0.0).astype(np.float32)
    ctx.labels, ctx.inst_labels, ctx.instances = labels, labels.copy(), [inst]
    return ctx


class TestRefine(unittest.TestCase):
    def test_crop_e_paste_sao_inversos_dentro_da_imagem(self):
        img = np.random.default_rng(0).integers(0, 255, (100, 120), dtype=np.uint8)
        back = paste_center(crop_center(img, 40, 60, 32), img.shape, 40, 60)
        self.assertTrue((back[24:56, 44:76] == img[24:56, 44:76]).all())

    def test_refiner_troca_o_amontoado_pelas_instancias_da_rede(self):
        cfg = load_config(None)
        refiner = CNNRefiner(cfg, Env(source=None, masks=None, cfg=cfg, out_dir=Path(".")))
        refiner.sep = FakeSeparator(min_area=1)
        ctx = cluster_context()
        refiner.process(ctx)
        self.assertEqual([i.source for i in ctx.instances], ["cnn", "cnn"])
        self.assertEqual(len(np.unique(ctx.inst_labels[ctx.inst_labels > 0])), 2)
        self.assertEqual(int((ctx.inst_labels > 0).sum()), 30 * 120)  # nenhum pixel do amontoado se perde

    def test_esqueleto_da_rede_avisa_que_falta_implementar(self):
        cfg = load_config(None)
        refiner = CNNRefiner(cfg, Env(source=None, masks=None, cfg=cfg, out_dir=Path(".")))
        with self.assertRaises(NotImplementedError):
            refiner.process(cluster_context())


if __name__ == "__main__":
    unittest.main()
