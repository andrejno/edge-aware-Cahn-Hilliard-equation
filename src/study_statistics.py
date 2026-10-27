from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

def _seed(base, label):
    digest = hashlib.sha256(str(label).encode()).digest()
    return (int(base) + int.from_bytes(digest[:8], 'little')) % (2 ** 63 - 1)

def mean_interval(values, bootstrap=20000, seed=20261026, confidence=0.95):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not x.size or (not np.all(np.isfinite(x))):
        raise ValueError('mean_interval needs a nonempty finite one-dimensional vector')
    rng = np.random.default_rng(seed)
    indices = rng.integers(x.size, size=(int(bootstrap), x.size))
    sampled = x[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(sampled, [alpha, 1 - alpha])
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std(ddof=1)) if x.size > 1 else 0.0, median=float(np.median(x)), ci_low=float(lo), ci_high=float(hi))

def paired_signflip(differences, seed=20261026, monte_carlo=200000):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not d.size or (not np.isfinite(d).all()):
        raise ValueError('differences must be nonempty and finite')
    observed = abs(float(d.mean()))
    eps = 1e-12 * max(1.0, observed)
    n = d.size
    count = 0
    if n <= 18:
        total = 1 << n
        for first in range(0, total, 4096):
            labels = np.arange(first, min(first + 4096, total), dtype=np.uint64)
            bits = (labels[:, None] >> np.arange(n, dtype=np.uint64) & 1).astype(np.int8)
            signed_means = ((2 * bits - 1) * d).mean(axis=1)
            count += int(np.sum(np.abs(signed_means) >= observed - eps))
        p = count / total
        return dict(p_value=float(p), kind='exact paired sign-flip', enumerations=total, statistic=observed, alternative='two-sided')
    rng = np.random.default_rng(seed)
    total = int(monte_carlo)
    for first in range(0, total, 4096):
        size = min(4096, total - first)
        signs = 2 * rng.integers(0, 2, size=(size, n), dtype=np.int8) - 1
        means = (signs * d).mean(axis=1)
        count += int(np.sum(np.abs(means) >= observed - eps))
    return dict(p_value=float((count + 1) / (total + 1)), kind='Monte Carlo paired sign-flip', enumerations=total, statistic=observed, alternative='two-sided')

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)) or (not np.all(np.isfinite(p))):
        raise ValueError('p-values must be finite numbers between zero and one')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, previous)
    return adjusted.tolist()
