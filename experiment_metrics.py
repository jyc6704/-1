"""Ten-color paired metrics; all base metrics are fractions, R_Q is percent."""
from __future__ import annotations

import math
import numpy as np

METRICS = ('mean_color_accuracy', 'all_colors_accuracy', 'aligned_accuracy',
           'conflict_accuracy', 'shortcut_gap', 'abs_shortcut_gap', 'flip_rate', 'recovery_score')


def calculate(predictions, targets, permutation):
    p = np.asarray(predictions)
    y = np.asarray(targets)
    pi = np.asarray(permutation)
    if p.shape != (len(y), 10) or len(y) == 0:
        raise ValueError('Expected nonempty N x 10 predictions and N targets.')
    for a in (p, y, pi):
        if not np.issubdtype(a.dtype, np.integer) or (a < 0).any() or (a > 9).any():
            raise ValueError('Labels/colors must be integer values in 0..9.')
    if pi.shape != (10,) or sorted(pi.tolist()) != list(range(10)):
        raise ValueError('Label-to-color mapping must be a permutation.')
    correct = p == y[:, None]
    m = float(correct.mean())
    q = float(correct.all(axis=1).mean())
    a = float(correct[np.arange(len(y)), pi[y]].mean())
    c = (10 * m - a) / 9
    c = float(np.clip(c, 0, 1))
    counts = np.stack([(p == k).sum(axis=1) for k in range(10)], axis=1)
    f = float((1 - (counts * (counts - 1)).sum(axis=1) / 90).mean())
    gap = a - c
    return dict(zip(METRICS, (m, q, a, c, gap, abs(gap), f, m*(1-abs(gap))*(1-f))))


def recovery(q, q0, qref, min_gap=.001):
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (q, q0, qref)):
        raise ValueError('Q values must be finite fractions.')
    gap = qref - q0
    if gap <= min_gap:
        return None, 'initial_gap_too_small_or_nonpositive'
    return 100 * (q - q0) / gap, 'available'


def self_check():
    pi = np.arange(10)
    for pred, target, expected in [([3]*10, 3, (1,1,0)),
                                  ([2]*10, 3, (0,0,0)),
                                  ([3]*9+[2], 3, (.9,0,.2)),
                                  (list(range(10)), 3, (.1,0,1))]:
        r = calculate(np.array([pred]), np.array([target]), pi)
        assert np.allclose([r['mean_color_accuracy'], r['all_colors_accuracy'], r['flip_rate']], expected)
        assert np.isclose(r['mean_color_accuracy'], (r['aligned_accuracy']+9*r['conflict_accuracy'])/10)
    assert np.isclose(recovery(.55, .2, .9)[0], 50)
    assert recovery(.8, .9, .9)[0] is None
    assert recovery(.95, .2, .9)[0] > 100
