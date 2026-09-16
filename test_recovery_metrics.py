"""서로 다른 정답/예측 조합과 실제 pilot 출력 형식에 대한 검증."""

import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from contextlib import contextmanager
from uuid import uuid4

from recovery_metrics import calculate_recovery, calculate_scores, color_scores_from_predictions


SCRIPT = Path(__file__).with_name("recovery_metrics.py")


@contextmanager
def fixture_directory():
    # Windows sandbox에서도 subprocess가 읽을 수 있는 workspace 내부 fixture.
    root = (SCRIPT.parent / "tmp" / "recovery_metric_tests").resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / uuid4().hex
    folder.mkdir()
    try:
        yield folder
    finally:
        resolved = folder.resolve()
        if resolved.parent != root:
            raise RuntimeError("테스트 fixture가 작업 경로를 벗어났습니다.")
        shutil.rmtree(resolved)


class RecoveryMetricsTests(unittest.TestCase):
    def test_binary_identity_against_every_pair(self):
        # 정답과 예측 조합을 직접 세어 수학적 복원식과 대조한다.
        for target, a, c in itertools.product(range(2), repeat=3):
            aligned, conflict = int(a == target), int(c == target)
            row = {"aligned_accuracy": aligned, "conflict_accuracy": conflict,
                   "neutral_accuracy": (aligned + conflict) / 2,
                   "flip_rate": int(a != c)}
            score = calculate_scores(row, 2, 2)
            self.assertEqual(score["all_colors_accuracy"], int(a == c == target))

    def test_multiclass_both_different_and_wrong(self):
        # 이진 복원식이 실패하는 핵심 사례: 서로 다른 예측인데 모두 오답.
        score = color_scores_from_predictions([[1, 2], [0, 0]], [0, 0])
        self.assertEqual(score["mean_color_accuracy"], 0.5)
        self.assertEqual(score["all_colors_accuracy"], 0.5)
        with self.assertRaises(ValueError):
            calculate_scores({"neutral_accuracy": 0.5, "flip_rate": 0.5}, 3, 2)

    def test_ten_colors(self):
        predictions = [[0] * 10, [1] * 8 + [2] * 2, [2] * 6 + [3] * 4]
        score = color_scores_from_predictions(predictions, [0, 1, 2])
        self.assertAlmostEqual(score["mean_color_accuracy"], 0.8)
        self.assertAlmostEqual(score["all_colors_accuracy"], 1 / 3)

    def test_invalid_data(self):
        for predictions, targets in (([], []), ([[0, 0], [1]], [0, 1]),
                                     ([[0, 0]], [0, 1]), ([[True, 0]], [0])):
            with self.assertRaises(ValueError):
                color_scores_from_predictions(predictions, targets)
        for row in ({"neutral_accuracy": 95, "flip_rate": 0},
                    {"neutral_accuracy": float("nan"), "flip_rate": 0},
                    {"neutral_accuracy": 0.1, "flip_rate": 1},
                    {"neutral_accuracy": 0.5, "flip_rate": 0,
                     "aligned_accuracy": 1, "conflict_accuracy": 0}):
            with self.assertRaises(ValueError):
                calculate_scores(row, 2, 2)

    def test_recovery_and_units(self):
        initial = {"mean_color_accuracy": 0.5, "all_colors_accuracy": 0.1}
        reference = {"mean_color_accuracy": 0.98, "all_colors_accuracy": 0.9}
        current = {"mean_color_accuracy": 0.85, "all_colors_accuracy": 0.7}
        result = calculate_recovery(current, initial, reference)
        self.assertAlmostEqual(result["q_recovery_percent"], 75)
        self.assertAlmostEqual(result["q_gap_to_reference_pp"], 20)
        self.assertAlmostEqual(result["mean_gap_to_reference_pp"], 13)
        self.assertIsNone(calculate_recovery(current)["q_recovery_percent"])
        self.assertEqual(calculate_recovery(current, reference=reference)["recovery_status"],
                         "missing_initial")
        for q in (0.1, 0.099, 0.10001):
            ref = {**reference, "all_colors_accuracy": q}
            self.assertIsNone(calculate_recovery(current, initial, ref)["q_recovery_percent"])
        for q, expected in ((0.0, -12.5), (1.0, 112.5)):
            result = calculate_recovery({**current, "all_colors_accuracy": q}, initial, reference)
            self.assertAlmostEqual(result["q_recovery_percent"], expected)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                              capture_output=True, encoding="utf-8",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    def test_cli_multiple_epochs_no_reference_and_preservation(self):
        with fixture_directory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"digits": [3, 8], "color_ids": [3, 8]}))
            source = root / "results.csv"
            text = ("exposure_epoch,aligned_accuracy,conflict_accuracy,neutral_accuracy,flip_rate\n"
                    "1,1,0,0.5,1\n2,1,0.2,0.6,0.8\n")
            source.write_text(text, encoding="utf-8-sig")
            for suffix in ("", "_02"):
                run = self.run_cli(root)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((root / f"recovery_metrics{suffix}" / "recovery_metrics.json")
                                    .read_text(encoding="utf-8"))
                self.assertEqual(len(report["results"]), 2)
                self.assertAlmostEqual(report["results"][1]["all_colors_accuracy"], 0.2)
                self.assertIsNone(report["results"][1]["q_recovery_percent"])
                self.assertEqual(report["results"][1]["phase"], "exposure")
            self.assertEqual(source.read_text(encoding="utf-8-sig"), text)

    def test_cli_reference_and_initial_last_row(self):
        with fixture_directory() as folder:
            root = Path(folder)
            for name, rows in {
                "initial": [{"neutral_accuracy": 0.5, "all_colors_accuracy": 0.0},
                            {"neutral_accuracy": 0.5, "all_colors_accuracy": 0.1}],
                "reference": {"neutral_accuracy": 0.95, "all_colors_accuracy": 0.9, "p": 0.1},
                "current": {"neutral_accuracy": 0.8, "all_colors_accuracy": 0.7, "recovery_epoch": 3},
            }.items():
                items = rows if isinstance(rows, list) else [rows]
                for row in items:
                    row.update(num_classes=10, num_colors=10)
                (root / f"{name}.json").write_text(json.dumps(items), encoding="utf-8")
            run = self.run_cli(root / "current.json", "--initial", root / "initial.json",
                               "--reference", root / "reference.json")
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root / "recovery_metrics" / "recovery_metrics.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual(report["initial"]["selected_row"], 2)
            self.assertAlmostEqual(report["results"][0]["q_recovery_percent"], 75)
            self.assertEqual(report["results"][0]["phase"], "recovery")


if __name__ == "__main__":
    unittest.main()
