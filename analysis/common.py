"""Shared numerical routines and portable trace loading (no model imports)."""
import json
from pathlib import Path

import numpy as np


def visual_statistics(scores, thresholds):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not scores.size or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Expected finite nonnegative visual scores with shape [steps, tokens].')
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError('Activation thresholds must be finite positive numbers.')
    return scores.sum(axis=-1), np.stack([(scores > t).sum(axis=-1) for t in thresholds], axis=-1)


def cosine_matrix(layer_vectors):
    """Head-mean visual vectors, L2 normalized per layer as in the research code."""
    vectors = np.asarray(layer_vectors, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.size or not np.isfinite(vectors).all():
        raise ValueError('Expected finite layer vectors with shape [layers, visual tokens].')
    normalized = vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def pearson(x, y):
    """Return None for constant / undersized inputs instead of a spurious score."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Pearson inputs must be paired finite vectors.')
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def trace_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    paths = sorted(path.rglob('trace.npz'))
    if not paths:
        raise FileNotFoundError(f'No trace.npz files found under {path}')
    return paths


def load_trace(path):
    with np.load(path, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    required = {'visual_scores', 'token_ids', 'token_texts', 'grid_hw', 'metadata'}
    if required - trace.keys():
        raise ValueError(f'Missing trace fields: {required - trace.keys()}')
    scores = trace['visual_scores']
    visual_statistics(scores, [0.001])
    if len(trace['token_ids']) != len(scores) or len(trace['token_texts']) != len(scores):
        raise ValueError('Attention rows and generated tokens are not aligned.')
    grid = trace['grid_hw']
    if grid.shape != (2,) or (grid <= 0).any() or int(np.prod(grid)) != scores.shape[1]:
        raise ValueError('Visual token count does not match the spatial grid.')
    trace['metadata'] = json.loads(str(trace['metadata'].item()))
    return trace


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
