"""Testes rapidos:  python -m unittest discover -s tests"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from celegans_seg import build_pipeline, load_config, ops  # noqa: E402


class TestOps(unittest.TestCase):
    def test_otsu_separa_duas_classes(self):
        img = np.concatenate([np.full(500, 40), np.full(500, 200)]).astype(np.uint8)
        self.assertTrue(40 <= ops.otsu_threshold(img) < 200)

    def test_rotulagem_duas_passadas_igual_ao_scipy(self):
        rng = np.random.default_rng(1)
        m = rng.random((40, 40)) > 0.6
        a, na = ops.label_components(m)
        b, nb = ops.two_pass_labeling(m)
        self.assertEqual(na, nb)
        self.assertEqual(len(set(zip(a.ravel().tolist(), b.ravel().tolist()))), na + 1)

    def test_esqueleto_de_barra_e_uma_linha(self):
        m = np.zeros((20, 60), bool)
        m[8:13, 5:55] = True
        sk = ops.thin(m)
        length, ends, junc = ops.skeleton_stats(sk)
        self.assertEqual(ends, 2)
        self.assertEqual(junc, 0)
        self.assertGreater(length, 40)

    def test_remove_small(self):
        m = np.zeros((20, 20), bool)
        m[2:4, 2:4] = True
        m[10:18, 10:18] = True
        lab, n = ops.label_components(m)
        lab, n = ops.remove_small(lab, n, 10)
        self.assertEqual(n, 1)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_sintetico(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic.py"), "--out", tmp,
                            "--frames", "15", "--worms", "4", "--size", "256"], check=True, capture_output=True)
            cfg = load_config(None, [f"input.dataset_root={tmp}", "input.sequence=seq0",
                                     f"output.dir={tmp}/out", "viz.gif=false"])
            cfg.stages = [s for s in cfg.stages if s != "visualize"]
            pipe = build_pipeline(cfg)
            pipe.fit(10)
            ctxs = list(pipe.run())
            self.assertEqual(len(ctxs), 15)
            self.assertGreater(np.mean([c.metrics["dice_semantic"] for c in ctxs]), 0.7)
            self.assertTrue((Path(tmp) / "out" / "metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
