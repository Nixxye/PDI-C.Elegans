"""Testes da CNN (so rodam se o torch estiver instalado):  python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from celegans_seg import cnn
except ImportError:  # pragma: no cover
    torch = None

from celegans_seg.stages.export import crop_center  # noqa: E402
from celegans_seg.stages.refine import paste_center  # noqa: E402


def fake_patch(S=64):
    m = np.zeros((S, S), bool)
    m[20:30, 8:56] = True
    m[10:50, 28:36] = True
    return dict(image=np.full((S, S), 120, np.uint8), signal=(m * 200).astype(np.uint8), mask=m)


class TestPaste(unittest.TestCase):
    def test_crop_e_paste_sao_inversos_dentro_da_imagem(self):
        img = np.random.default_rng(0).integers(0, 255, (100, 120), dtype=np.uint8)
        back = paste_center(crop_center(img, 40, 60, 32), img.shape, 40, 60)
        self.assertTrue((back[24:56, 44:76] == img[24:56, 44:76]).all())


@unittest.skipIf(torch is None, "torch nao instalado")
class TestCNN(unittest.TestCase):
    def test_forma_da_saida(self):
        masks, ex = cnn.ClusterNet()(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(masks.shape), (2, cnn.K_SLOTS, 64, 64))
        self.assertEqual(tuple(ex.shape), (2, cnn.K_SLOTS))

    def test_separador_devolve_rotulos_so_dentro_da_mascara(self):
        torch.manual_seed(0)
        sep = cnn.ClusterSeparator(net=cnn.ClusterNet(), device="cpu", threshold=0.0, min_area=1, exist_threshold=0.0)
        p = fake_patch()
        lab = sep.separate(p)
        self.assertEqual(lab.shape, p["mask"].shape)
        self.assertTrue((lab[~p["mask"]] == 0).all())
        self.assertTrue((lab[p["mask"]] > 0).all())

    def test_perda_e_finita_e_tem_gradiente(self):
        net = cnn.ClusterNet()
        p = fake_patch()
        x = torch.from_numpy(cnn.patch_to_input(p))[None]
        gt = np.stack([p["mask"] & (np.arange(64)[None] < 32), p["mask"] & (np.arange(64)[None] >= 32)])
        loss = cnn.slot_loss(*net(x), [gt], x[:, 2])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in net.parameters()))


if __name__ == "__main__":
    unittest.main()
