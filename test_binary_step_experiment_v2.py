"""Quick tests for binary_step_experiment_v2.py; no dataset downloads needed.

Run from GitHub repository root:
    python -m unittest test_binary_step_experiment_v2 -v
"""
import math
import unittest

import numpy as np

from binary_step_experiment_v2 import (
    assign_binary_colors,
    binary_metrics,
    derive,
    parse_args,
    scheduled_recovery_steps,
    should_evaluate_recovery,
    stratified_split,
)


class ExperimentV2Tests(unittest.TestCase):
    def test_new_formal_defaults(self):
        args = parse_args([])
        self.assertEqual(args.p_values, [.5, .8, .9, .99])
        self.assertEqual(args.exposure_epochs, [4, 8, 16, 24, 32])
        self.assertEqual(args.recovery_epochs, 5)
        self.assertEqual(args.late_eval_interval, 5)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.seeds, [42, 43, 44])
        self.assertEqual(args.datasets, ["mnist", "emnist_digits"])
        self.assertEqual(args.output.name, "binary_step_v2_full")

    def test_epoch1_every_step_and_epoch2_every_five(self):
        self.assertEqual(scheduled_recovery_steps(8, 2),
                         list(range(9)) + [13, 16])
        self.assertEqual(scheduled_recovery_steps(7, 3),
                         list(range(8)) + [12, 14, 19, 21])
        self.assertTrue(should_evaluate_recovery(1, 3, 8))
        self.assertFalse(should_evaluate_recovery(2, 3, 8))
        self.assertTrue(should_evaluate_recovery(2, 5, 8))
        self.assertTrue(should_evaluate_recovery(2, 8, 8))

    def test_seventy_five_step_epoch_schedule(self):
        # Original MNIST 3/8 before optional exact-duplicate audit.
        self.assertEqual(math.ceil(4792 / 64), 75)
        steps = scheduled_recovery_steps(75, 5)
        self.assertEqual(steps[:76], list(range(76)))
        self.assertEqual(steps[-1], 375)
        self.assertEqual(len(steps), 136)
        for end in [75, 150, 225, 300, 375]:
            self.assertIn(end, steps)
        self.assertNotIn(76, steps)
        self.assertIn(80, steps)

    def test_reproducible_color_assignment_for_all_probabilities(self):
        y = np.array([3] * 301 + [8] * 297, dtype=np.int64)
        for p in [.5, .8, .9, .99]:
            a = assign_binary_colors(y, p, seed=2026)
            b = assign_binary_colors(y, p, seed=2026)
            self.assertTrue(np.array_equal(a, b))
            for d in (3, 8):
                n = int((y == d).sum())
                aligned = int((a[y == d] == d).sum())
                self.assertLessEqual(abs(aligned - n*p), .500001)

    def test_split_disjoint_and_seed_stable(self):
        labels = np.array([3] * 102 + [8] * 99)
        ids = np.arange(len(labels))
        a = stratified_split(ids, labels, seed=42)
        b = stratified_split(ids, labels, seed=42)
        for k in ("exposure", "recovery", "validation"):
            self.assertTrue(np.array_equal(a[k], b[k]))
            self.assertEqual(set(labels[a[k]]), {3, 8})
        joined = np.concatenate(list(a.values()))
        self.assertEqual(len(np.unique(joined)), len(ids))
        self.assertEqual(set(joined), set(ids))

    def test_binary_metrics_q_identity(self):
        targets = np.array([3, 8, 3, 8])
        pred = np.array([[0, 0], [1, 1], [0, 1], [0, 1]], dtype=np.uint8)
        m = binary_metrics(pred, targets)
        self.assertAlmostEqual(m["all_colors_accuracy"], .5)
        self.assertAlmostEqual(m["mean_color_accuracy"], .75)
        self.assertAlmostEqual(m["flip_rate"], .5)
        self.assertAlmostEqual(m["shortcut_gap"], .5)
        self.assertAlmostEqual(m["recovery_score"], .1875)
        self.assertAlmostEqual(m["all_colors_accuracy"],
                               m["mean_color_accuracy"] - m["flip_rate"]/2)

    def test_per_exposure_reference(self):
        rows = []
        for exp, ref in ((4, .9), (32, .8)):
            for p, q0, qend in ((.5, .5, ref), (.99, .2, .6)):
                for step, value in ((0, q0), (3, qend)):
                    rows.append(dict(dataset="mnist_38", split="validation", seed=42,
                                     exposure_epochs=exp, p=p, recovery_step=step,
                                     all_colors_accuracy=value))
        derived = derive(rows, {})
        expected = {4: 100 * (.6-.2)/(.9-.2),
                    32: 100 * (.6-.2)/(.8-.2)}
        for exp, target in expected.items():
            measured = next(r["q_recovery_percent"] for r in derived
                            if r["exposure_epochs"] == exp and r["p"] == .99
                            and r["recovery_step"] == 3)
            self.assertAlmostEqual(measured, target)
        self.assertTrue(all(r["q_recovery_percent"] is None for r in derived
                            if r["p"] == .5))

    def test_invalid_schedule_arguments(self):
        for params in [(0, 5, 5), (3, 0, 5), (3, 5, 0)]:
            with self.assertRaises(ValueError):
                scheduled_recovery_steps(*params)


if __name__ == "__main__":
    unittest.main()
