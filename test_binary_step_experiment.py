"""Fast unit tests for binary_step_experiment.py (no dataset downloads)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

import binary_step_experiment as exp
from experiment_io import load_checkpoint, read_json, verify_manifest
from recovery_metrics import calculate_paired_recovery

from binary_step_experiment import (
    assign_binary_colors, binary_metrics, derive, stratified_split, summarize,
)


def protocol(**overrides):
    return {"datasets": ["mnist"], "seeds": [42], "p_values": [.5, .99],
            "exposure_epochs": [1], "recovery_epochs": 1, "batch_size": 64, **overrides}


def observation(dataset, seed, exposure, split, p, step, q, num_recovery=128):
    return {"dataset": dataset, "split": split, "seed": seed,
            "exposure_epochs": exposure, "p": p, "recovery_step": step,
            "actual_recovery_epoch": step / 2, "all_colors_accuracy": q,
            "mean_color_accuracy": q, "neutral_accuracy": q, "flip_rate": 0.,
            "initial_model_sha256": f"model-{seed}", "design_sha256": f"{dataset}-{seed}",
            "num_exposure": 128, "num_recovery": num_recovery,
            "num_sources": 32 if split == "validation" else 20, "num_colors": 2}


def paired_rows():
    return [observation("mnist_38", 42, 1, "validation", p, step, q)
            for p, values in ((.5, [.4, .8, .9]), (.99, [.2, .55, .95]))
            for step, q in enumerate(values)]


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
                    rows.append(observation(dataset,42,1,"validation",p,step,q,num_recovery=64))
        out = derive(rows,protocol(datasets=["mnist", "emnist_digits"]))
        for name,expected in (("mnist_38",100*(.6-.2)/(.9-.2)),
                              ("emnist_digits_38",100*(.6-.2)/(.8-.2))):
            measured = next(r["q_recovery_percent"] for r in out
                if r["dataset"]==name and r["p"]==.99 and r["recovery_step"]==1)
            self.assertAlmostEqual(measured,expected)
        self.assertTrue(all(r["q_recovery_percent"] is None for r in out if r["p"]==.5))


class BinaryStepIntegrationTests(unittest.TestCase):
    def test_epoch_and_step_recovery_agree(self):
        rows = paired_rows()
        local = [{**r, "dataset": "mnist", "num_classes": 2, "num_validation": 32,
                  "recovery_epoch": r["recovery_step"]} for r in rows]
        for a, b in zip(calculate_paired_recovery(local, 2), derive(rows, protocol())):
            for field in ("q_initial", "q_reference", "q_recovery_percent"):
                self.assertEqual(a[field], b[field])
            self.assertEqual(a["recovery_status"], b["recovery_percent_status"])
            self.assertEqual(a["q_gap_same_epoch_pp"], b["q_gap_same_step_pp"])
            self.assertEqual(b["reference_recovery_step"], 2)

    def test_partial_control_is_not_used_as_final_reference(self):
        rows = [r for r in paired_rows() if not (r["p"] == .5 and r["recovery_step"] == 2)]
        biased = [r for r in derive(rows, protocol()) if r["p"] == .99]
        self.assertTrue(all(r["q_reference"] is None and r["q_recovery_percent"] is None
                            and r["reference_recovery_step"] is None
                            and r["recovery_percent_status"] == "reference_incomplete" for r in biased))
        missing = derive([r for r in rows if r["p"] == .99], protocol())
        self.assertTrue(all(r["recovery_percent_status"] == "reference_missing" for r in missing))

    def test_reference_cannot_cross_seed_exposure_or_split(self):
        original = paired_rows()
        for changed in ({"seed": 43, "initial_model_sha256": "model-43", "design_sha256": "mnist_38-43"},
                        {"exposure_epochs": 2}, {"split": "test", "num_sources": 20}):
            rows = [r if r["p"] != .5 else {**r, **changed} for r in original]
            rows = [r for r in rows if not (r["split"] == "test" and r["recovery_step"] == 1)]
            settings = protocol(seeds=[42, 43], exposure_epochs=[1, 2])
            result = [r for r in derive(rows, settings) if r["p"] == .99]
            self.assertTrue(all(r["q_recovery_percent"] is None for r in result))

    def test_duplicate_missing_steps_and_mismatched_sources_are_rejected(self):
        rows = paired_rows()
        bad_inputs = [rows + [rows[0]], rows[:1] + rows[2:]]
        for field, value in (("initial_model_sha256", "different"), ("design_sha256", "different"),
                             ("num_sources", 33), ("num_recovery", 127), ("num_exposure", 127)):
            bad_inputs.append([{**rows[0], field: value}, *rows[1:]])
        for bad in bad_inputs:
            with self.subTest(bad=bad[0]), self.assertRaises(ValueError):
                derive(bad, protocol())
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            summarize([rows[0], rows[0]])

    def test_complete_requires_all_conditions_and_test_endpoints(self):
        rows = paired_rows()
        with self.assertRaisesRegex(ValueError, "evaluation split"):
            derive(rows, protocol(), complete=True)
        rows += [{**r, "split": "test", "num_sources": 20}
                 for r in rows if r["recovery_step"] in (0, 2)]
        self.assertEqual(len(derive(rows, protocol(), complete=True)), 10)
        with self.assertRaises(ValueError):
            derive(rows[:-1], protocol(), complete=True)
        with self.assertRaises(ValueError):
            derive(rows, protocol(seeds=[42, 43]), complete=True)

    def test_step_run_and_exact_resume_with_shared_metrics(self):
        def fixture(name, root, download):
            rng = np.random.default_rng(723)
            # The official source contains all ten digits; the runner filters 3/8.
            return {split: (rng.integers(0, 256, (n, 28, 28), dtype=np.uint8),
                            np.arange(n, dtype=np.int64) % 10)
                    for split, n in (("train", 400), ("test", 100))}

        with tempfile.TemporaryDirectory() as td, patch.object(exp, "load_source", side_effect=fixture):
            root = Path(td)
            common = ["--smoke", "--datasets", "mnist", "--exposure-epochs", "1",
                      "--recovery-epochs", "2", "--batch-size", "16", "--eval-batch-size", "32",
                      "--checkpoint-every-steps", "1", "--num-threads", "1", "--device", "cpu",
                      "--offline", "--no-keep-awake"]
            continuous, resumed = root / "continuous", root / "resumed"
            self.assertEqual(exp.main(common + ["--output", str(continuous)]), 0)
            self.assertEqual(exp.main(common + ["--output", str(resumed), "--stop-after-steps", "3"]), 130)
            # The interrupted neutral run has no final reference, even during analysis-only.
            partial, derived = exp.build_report(resumed, read_json(resumed / "protocol.json"))
            self.assertEqual(len(partial), 2)
            self.assertTrue(all(r["q_reference"] is None for r in derived))
            self.assertEqual(exp.main(common + ["--output", str(resumed)]), 0)
            for pointer in continuous.glob("runs/*/seed*/exp*/p*/latest.json"):
                rel = pointer.parent.relative_to(continuous)
                first = load_checkpoint(pointer.parent / read_json(pointer)["file"])
                other = resumed / rel
                second = load_checkpoint(other / read_json(other / "latest.json")["file"])
                for key, value in first["model"].items():
                    self.assertTrue(torch.equal(value, second["model"][key]), key)
                for a, b in zip(first["observations"], second["observations"]):
                    for metric in exp.METRICS:
                        self.assertEqual(a[metric], b[metric])
            settings = read_json(resumed / "protocol.json")
            for name in ("metrics.py", "recovery_metrics.py"):
                self.assertIn(name, settings["code_sha256"])
                self.assertTrue((resumed / "provenance" / "code" / name).is_file())
            self.assertEqual(read_json(resumed / "status.json")["status"], "complete")
            verify_manifest(resumed)


if __name__ == "__main__":
    unittest.main()
