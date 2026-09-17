"""Frozen independent region probes and descriptive joint G/S OMP."""
import hashlib
import numpy as np


def masks_for(qid, region_ids):
    """22 Bernoulli(.5) draws; deliberately no token-budget repair/resampling."""
    if not region_ids or len(set(region_ids)) != len(region_ids):
        raise ValueError('Nonempty unique stable region IDs required')
    seed = int.from_bytes(hashlib.sha256(('omp10-20260917-v1|' + qid).encode()).digest()[:8], 'big')
    return np.random.Generator(np.random.PCG64(seed)).integers(0, 2, (22, len(region_ids)), dtype=np.int8)


def fit_omp(masks, scores):
    """Original-scale G/S, centered normalized columns, at most four joint atoms.

    Coefficients and in-sample fit describe hypotheses, not causal labels or
    held-out accuracy. Constant columns remain explicitly unidentifiable.
    """
    x, y = np.asarray(masks, dtype=float), np.asarray(scores, dtype=float)
    if x.ndim != 2 or y.shape != (len(x), 2) or len(x) != 22:
        raise ValueError('Require the complete 22-mask G/S bank')
    if not np.isin(x, [0, 1]).all() or not np.isfinite(y).all():
        raise ValueError('Invalid masks or measurements')
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    norm = np.linalg.norm(xc, axis=0)
    z = xc / np.where(norm > 0, norm, 1)
    residual, support = yc.copy(), []
    beta = np.zeros((x.shape[1], 2))
    for _ in range(min(4, x.shape[1])):
        association = np.linalg.norm(z.T @ residual, axis=1)
        association[norm == 0] = -1
        association[support] = -1
        j = int(np.argmax(association))
        if association[j] < 1e-10:
            break
        support.append(j)
        coeff = np.linalg.lstsq(z[:, support], yc, rcond=None)[0]
        residual = yc - z[:, support] @ coeff
        beta[support] = coeff / norm[support, None]
    intercept = ym - xm @ beta
    marginal = np.linalg.norm(z.T @ yc, axis=1)
    strength = np.linalg.norm(beta, axis=1)
    ranked_support = sorted(support, key=lambda j: (-strength[j], j))
    others = sorted((j for j in range(x.shape[1]) if j not in support and norm[j] > 0), key=lambda j: (-marginal[j], j))
    prediction = x @ beta + intercept
    return dict(support=support, shortlist=(ranked_support+others)[:8], coefficients=beta.tolist(),
                intercept=intercept.tolist(), marginal_association=marginal.tolist(),
                constant_columns=np.flatnonzero(norm == 0).tolist(),
                design_rank=int(np.linalg.matrix_rank(xc)),
                in_sample_rmse=np.sqrt(((prediction-y)**2).mean(0)).tolist(),
                predictions=prediction.tolist(), scope='descriptive discovery fit; not independently validated')
