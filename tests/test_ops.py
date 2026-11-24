"""Operacoes de PDI que seguem os slides: top-hat (Aula 08) e SSIM (Aula 05)."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from celegans_seg import ops  # noqa: E402


class OpsTest(unittest.TestCase):
    def test_tophat_do_slide_na_imagem_invertida_e_fechamento_menos_imagem(self):
        rng = np.random.default_rng(0)
        img = rng.random((60, 80)).astype(np.float32)
        esperado = ndi.grey_closing(img, size=(9, 9)) - img  # forma antiga: fechamento - imagem
        np.testing.assert_allclose(ops.black_tophat(img, 9), esperado, atol=1e-6)

    def test_tophat_realca_objeto_claro_pequeno(self):
        img = np.zeros((40, 40), np.float32)
        img[20, 20] = 1.0  # ponto claro menor que o SE
        self.assertAlmostEqual(float(ops.tophat(img, 9)[20, 20]), 1.0, places=6)

    def test_ssim_mse_psnr(self):
        a = np.zeros((32, 32))
        a[8:24, 14:18] = 1
        self.assertAlmostEqual(ops.ssim(a, a), 1.0, places=6)
        self.assertEqual(ops.mse(a, a), 0.0)
        self.assertEqual(ops.psnr(a, a), float("inf"))


if __name__ == "__main__":
    unittest.main()
