"""Fast unit tests for binary_step_experiment.py (no dataset downloads)."""
import unittest
import numpy as np

from binary_step_experiment import (
    assign_binary_colors, binary_metrics, derive, stratified_split,
)


class BinaryStepMetricsTests(unittest.TestCase):
    def test_known_binary_color_metrics(self):
        # Each row is [prediction under color3, prediction under color8], 0=3, 1=8.
        y = np.array([3, 8, 3, 8])
        pred = np.array([[0,0], [1,1], [0,1], [0,1]], dtype=np.uint8)
        m = binary_metrics(pred, y)
        self.assertAlmostEqual(m["aligned_accuracy"], 1.)
        self.assertAlmostEqual(m["conflict_accuracy"], .5)
        self.assertAlmostEqual(m["mean_color_accuracy"], .75)
        self.assertAlmostEqual(m["all_colors_accuracy"], .5)
        self.assertAlmostEqual(m["flip_rate"], .5)
        self.assertAlmostEqual(m["shortcut_gap"], .5)
        self.assertAlmostEqual(m["recovery_score"], .1875)
        self.assertAlmostEqual(m["digit3_q"], .5)
        self.assertAlmostEqual(m["digit8_q"], .5)

    def test_all_correct_and_color_invariant(self):
        y = np.array([3, 8] * 30)
        pred = (y == 8).astype(np.uint8)[:,None].repeat(2, axis=1)
        m = binary_metrics(pred, y)
        self.assertAlmostEqual(m["all_colors_accuracy"], 1.)
        self.assertAlmostEqual(m["flip_rate"], 0.)
        self.assertAlmostEqual(m["shortcut_gap"], 0.)
        self.assertAlmostEqual(m["recovery_score"], 1.)

    def test_neutral_allocation_is_balanced_and_reproducible(self):
        labels = np.array([3]*37 + [8]*39)
        x = assign_binary_colors(labels, .5, seed=42)
        self.assertTrue(np.array_equal(x, assign_binary_colors(labels, .5, seed=42)))
        for d in (3,8):
            n = (labels==d).sum()
            self.assertLessEqual(abs(int((x[labels==d]==d).sum())-.5*n),.5)
        y = assign_binary_colors(labels, .99, seed=42)
        for d in (3,8):
            self.assertLessEqual(abs(int((y[labels==d]==d).sum())-.99*(labels==d).sum()),.5)

    def test_three_way_split_is_disjoint(self):
        labels = np.array([3]*101 + [8]*107)
        ids = np.arange(len(labels))
        a = stratified_split(ids, labels, seed=17)
        b = stratified_split(ids, labels, seed=17)
        for key in ("exposure","recovery","validation"):
            self.assertTrue(np.array_equal(a[key],b[key]))
            self.assertEqual(set(labels[a[key]]), {3,8})
        joined = np.concatenate(list(a.values()))
        self.assertEqual(len(np.unique(joined)),len(ids))
        self.assertEqual(set(joined),set(ids))

    def test_reference_is_condition_specific(self):
        # Changing dataset/exposure length must not allow borrowing another control.
        rows = []
        for dataset, ref in (("mnist_38",.9),("emnist_digits_38",.8)):
            for p, initial, final in ((.5,.5,ref),(.99,.2,.6)):
                for step, q in ((0,initial),(1,final)):
                    rows.append({"dataset":dataset,"split":"validation","seed":42,
                        "exposure_epochs":1,"p":p,"recovery_step":step,
                        "all_colors_accuracy":q})
        out = derive(rows,{})
        for name,expected in (("mnist_38",100*(.6-.2)/(.9-.2)),
                              ("emnist_digits_38",100*(.6-.2)/(.8-.2))):
            measured = next(r["q_recovery_percent"] for r in out
                if r["dataset"]==name and r["p"]==.99 and r["recovery_step"]==1)
            self.assertAlmostEqual(measured,expected)
        self.assertTrue(all(r["q_recovery_percent"] is None for r in out if r["p"]==.5))


if __name__ == "__main__":
    unittest.main()
