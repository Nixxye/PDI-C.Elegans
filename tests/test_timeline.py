"""Linha do tempo: dois vermes sinteticos que andam mantem o id; um verme novo abre uma trilha nova."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from celegans_seg import load_config, ops  # noqa: E402
from celegans_seg.core import Env, FrameContext, Instance  # noqa: E402
from celegans_seg.stages.timeline import Timeline  # noqa: E402


def frame(k, worms, shape=(200, 300)):
    lab = np.zeros(shape, np.int32)
    insts = []
    for n, (y, x, horiz) in enumerate(worms, 1):
        if horiz:
            lab[y - 4:y + 4, x - 30:x + 30] = n
        else:
            lab[y - 30:y + 30, x - 4:x + 4] = n
        ys, xs = np.nonzero(lab == n)
        insts.append(Instance(id=n, component_id=n, source="single", area=len(ys), centroid=(ys.mean(), xs.mean())))
    ctx = FrameContext(index=k, name=str(k), raw=np.zeros(shape, np.uint8))
    ctx.inst_labels, ctx.instances = lab, insts
    return ctx


class TimelineTest(unittest.TestCase):
    def test_ssim_identico_e_diferente(self):
        a = np.zeros((40, 40))
        a[10:30, 18:22] = 1
        b = np.zeros((40, 40))
        b[18:22, 5:35] = 1
        roi = ndi.binary_dilation((a + b) > 0, iterations=4)  # so em volta das mascaras, como no estagio
        self.assertAlmostEqual(ops.ssim(a, a, roi=roi), 1.0, places=6)
        self.assertLess(ops.ssim(a, b, roi=roi), 0.3)
        self.assertEqual(ops.mse(a, a), 0.0)

    def test_ids_estaveis(self):
        cfg = load_config(None)
        st = Timeline(cfg, Env(source=None, masks=None, cfg=cfg, out_dir=ROOT / "runs" / "_tl_test"))
        ids = []
        for k in range(5):
            worms = [(50 + 2 * k, 80 + 3 * k, True), (140, 200 - 2 * k, False)]
            if k == 4:
                worms.append((100, 20, True))  # entra um terceiro verme
            ctx = frame(k, worms)
            st.process(ctx)
            ids.append([i.track for i in ctx.instances])
        self.assertEqual(ids[0], ids[3][:2])
        self.assertEqual(ids[3], ids[4][:2])
        self.assertNotIn(ids[4][2], ids[0])


if __name__ == "__main__":
    unittest.main()
