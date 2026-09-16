"""Protocol regressions using synthetic source images; no download required."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from dataset import (
    Digit38MNIST, SpuriousColoredMNIST, make_biased_color_ids,
    make_recovery_color_ids, split_exposure_recovery_validation, subset_labels,
)
from metrics import recovery_score
from model import BinarySmallCNN
from recovery_experiment import (
    METRICS, main, make_loader, parse_args, prepare_seed, reset_adam,
    seed_plan, state_hash, summarize, validate_colors,
)
from train import evaluate_binary, train_one_epoch


class Sources(Dataset):
    def __init__(self):
        self.targets = torch.tensor([3, 8, 0, 4] * 200)
        self.images = torch.rand((800, 1, 28, 28), generator=torch.Generator().manual_seed(5))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


class RecoveryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.sources = Digit38MNIST(Sources())

    def test_defaults(self):
        args = parse_args([])
        self.assertEqual((args.seeds, args.p_values, args.recovery_epochs, args.batch_size),
                         ([42, 43, 44], [.5, .9, .99], 10, 64))

    def test_composite_extremes_negative_gap_and_invalid_inputs(self):
        self.assertEqual(recovery_score(1, 0, 0), 1)
        self.assertEqual(recovery_score(.5, 1, 1), 0)
        self.assertAlmostEqual(recovery_score(.95, -.05, .05), .857375)
        self.assertEqual(recovery_score(.6, 0, 0), .6)
        for values in ((math.nan, 0, 0), (1, math.inf, 0), (1, 0, -1), (1.1, 0, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                recovery_score(*values)

    def test_splits_cover_only_target_sources_and_are_reproducible(self):
        first = split_exposure_recovery_validation(self.sources, 42)
        second = split_exposure_recovery_validation(self.sources, 42)
        self.assertEqual([len(s) for s in first], [160, 160, 80])
        indices = [set(self.sources.original_indices[s.indices].tolist()) for s in first]
        self.assertEqual(set.union(*indices), set(self.sources.original_indices.tolist()))
        self.assertFalse(indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
        self.assertEqual([s.indices for s in first], [s.indices for s in second])

    def test_colors_and_loader_orders_are_paired_across_p(self):
        prepared = prepare_seed(self.sources, 42, 10)
        exp, rec, _ = prepared['splits']
        labels = subset_labels(exp)
        plan = seed_plan(42, 10)
        previous = None
        for epoch in range(1, 11):
            colors = prepared['assignments']['recovery_color_ids'][epoch]
            self.assertTrue(torch.equal(colors, make_recovery_color_ids(subset_labels(rec), 42, epoch)))
            validate_colors(subset_labels(rec), colors, .5)
            if previous is not None:
                self.assertFalse(torch.equal(previous, colors))
            previous = colors
        orders = []
        recovery_orders = []
        for p in [.5, .9, .99]:
            colors = make_biased_color_ids(labels, p, 42)
            validate_colors(labels, colors, p)
            ds = SpuriousColoredMNIST(exp, colors, prepared['assignments']['hue_offsets'])
            orders.append(list(make_loader(ds, 64, 0, plan['exposure_shuffle_seed'], True).sampler))
            recovery_orders.append([
                list(make_loader(rec, 64, 0, plan['recovery_shuffle_seeds'][str(e)], True).sampler)
                for e in range(1, 11)
            ])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0], orders[2])
        self.assertEqual(recovery_orders[0], recovery_orders[1])
        self.assertEqual(recovery_orders[0], recovery_orders[2])
        self.assertNotEqual(recovery_orders[0][0], recovery_orders[0][1])

    def test_optimizer_reset_preserves_trained_weights_and_restarts_steps(self):
        model = BinarySmallCNN()
        old = torch.optim.Adam(model.parameters(), lr=.001)
        x = torch.rand(4, 3, 28, 28)
        torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 0, 1])).backward()
        old.step()
        before = state_hash(model.state_dict())
        self.assertTrue(old.state)
        new = reset_adam(model, old)
        self.assertIsNot(new, old)
        self.assertFalse(new.state)
        self.assertEqual(before, state_hash(model.state_dict()))
        new.zero_grad()
        model(x).sum().backward()
        new.step()
        self.assertTrue(all(state['step'].item() == 1 for state in new.state.values()))

    def test_binary_evaluation_known_predictions_and_hue_guard(self):
        prepared = prepare_seed(self.sources, 42, 2)
        val = prepared['splits'][2]
        a = prepared['assignments']
        aligned = SpuriousColoredMNIST(val, a['validation_aligned_color_ids'], a['hue_offsets'])
        conflict = SpuriousColoredMNIST(val, a['validation_conflict_color_ids'], a['hue_offsets'])

        class ColorOnly(torch.nn.Module):
            def forward(self, image):
                means = image.mean((2, 3))
                return torch.stack((means[:, 1], means[:, 2]), dim=1)

        loaders = [make_loader(ds, 17, 0, 42, False) for ds in (aligned, conflict)]
        metrics = evaluate_binary(ColorOnly(), *loaders, 'cpu')
        self.assertEqual(metrics, dict(aligned_accuracy=1, conflict_accuracy=0,
                                      neutral_accuracy=.5, shortcut_gap=1, flip_rate=1))
        conflict.hue_offsets = conflict.hue_offsets + .1
        with self.assertRaisesRegex(ValueError, 'hue'):
            evaluate_binary(ColorOnly(), *loaders, 'cpu')

    def test_summary_is_sample_sd_and_handles_single_seed(self):
        rows = [dict(seed=i, p=.99, recovery_epoch=0, **{m: v for m in METRICS})
                for i, v in enumerate((.2, .4, .6))]
        result = summarize(rows)[0]
        self.assertAlmostEqual(result['recovery_score_mean'], .4)
        self.assertAlmostEqual(result['recovery_score_std'], .2)
        self.assertIsNone(summarize(rows[:1])[0]['recovery_score_std'])

    def test_end_to_end_protocol_and_outputs(self):
        import csv
        import json
        with tempfile.TemporaryDirectory() as temporary:
            initial_hashes, events, reset_events = [], [], []

            def train_spy(model, loader, optimizer, device):
                events.append((id(model), len(optimizer.state), id(loader.dataset.subset)))
                if len(events) % 3 == 1:
                    initial_hashes.append(state_hash(model.state_dict()))
                return train_one_epoch(model, loader, optimizer, device)

            def reset_spy(model, old):
                reset_events.append(state_hash(model.state_dict()))
                return reset_adam(model, old)

            with patch('recovery_experiment.load_digit38_mnist', return_value=(self.sources, None)), \
                 patch('recovery_experiment.train_one_epoch', side_effect=train_spy), \
                 patch('recovery_experiment.reset_adam', side_effect=reset_spy):
                run = main(['--seeds', '42', '43', '--recovery-epochs', '2',
                            '--device', 'cpu', '--output', temporary])
            rows = list(csv.DictReader((run / 'raw_results.csv').open(encoding='utf-8-sig')))
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(reset_events), 6)  # includes p=.5, exactly once per run
            self.assertEqual(initial_hashes[:3], [initial_hashes[0]] * 3)
            self.assertEqual(initial_hashes[3:], [initial_hashes[3]] * 3)
            for index in range(0, len(events), 3):
                self.assertEqual(events[index][0], events[index + 1][0])
                self.assertEqual(events[index + 1][1], 0)
                self.assertGreater(events[index + 2][1], 0)
                self.assertNotEqual(events[index][2], events[index + 1][2])
            self.assertTrue(all(r['recovery_train_loss'] == '' for r in rows if r['recovery_epoch'] == '0'))
            self.assertEqual(len(list((run / 'checkpoints').glob('*.pt'))), 12)
            self.assertEqual(len(list((run / 'plots').glob('*.png'))), 4)
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'complete')
            self.assertEqual(len(list(csv.DictReader((run / 'summary_results.csv').open(encoding='utf-8-sig')))), 9)


if __name__ == '__main__':
    unittest.main()
