from typing import List

import numpy as np
import torch

from sklearn.metrics import pairwise_distances
from snf import compute


# =========================================================
# HELPERS
# =========================================================

def _row_normalize(a: np.ndarray) -> np.ndarray:
    rs = a.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return a / rs


def _cosine_similarity(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)

    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    ).clip(1e-8)

    x = x / norm

    sim = x @ x.T

    return np.clip(sim, 0.0, 1.0)


def _cross_cosine_similarity(
    te: np.ndarray,
    tr: np.ndarray,
) -> np.ndarray:

    te = te.astype(np.float32)
    tr = tr.astype(np.float32)

    te_norm = np.linalg.norm(
        te,
        axis=1,
        keepdims=True
    ).clip(1e-8)

    tr_norm = np.linalg.norm(
        tr,
        axis=1,
        keepdims=True
    ).clip(1e-8)

    sim = (te / te_norm) @ (tr / tr_norm).T

    return np.clip(sim, 0.0, 1.0)


def _rbf_from_dist(
    dist: np.ndarray,
    sigma: float = None,
) -> np.ndarray:

    dist = np.asarray(dist, dtype=np.float32)

    if sigma is None:
        vals = dist[np.isfinite(dist)]

        sigma = np.median(vals) if vals.size else 1.0

        if sigma <= 0:
            sigma = 1.0

    return np.exp(
        -(dist ** 2) / (2.0 * sigma ** 2 + 1e-12)
    ).astype(np.float32)


def _topk_mask(
    sim: np.ndarray,
    k: int,
) -> np.ndarray:

    out = np.zeros_like(sim, dtype=np.float32)

    if sim.shape[1] == 0:
        return out

    k = max(1, min(k, sim.shape[1]))

    idx = np.argpartition(
        -sim,
        kth=k - 1,
        axis=1
    )[:, :k]

    rows = np.arange(sim.shape[0])[:, None]

    out[rows, idx] = sim[rows, idx]

    return out


# =========================================================
# EDGE DROPOUT
# =========================================================

def apply_edge_dropout(
    adj: np.ndarray,
    p: float,
    seed: int = 42,
) -> np.ndarray:

    if p <= 0:
        return adj

    rng = np.random.default_rng(seed)

    keep = rng.random(adj.shape) > p

    keep = np.triu(keep, 1)

    keep = keep + keep.T + np.eye(
        adj.shape[0],
        dtype=bool,
    )

    out = adj * keep.astype(np.float32)

    np.fill_diagonal(out, 1.0)

    return _row_normalize(out.astype(np.float32))


# =========================================================
# ORIGINAL COSINE GRAPH (BRCA)
# =========================================================

def build_train_test_graph_from_views_cosine(
    train_views: List[np.ndarray],
    test_views: List[np.ndarray],
    k: int = 10,
    threshold: float = 0.15,
    self_loop: bool = True,
) -> np.ndarray:

    train_sims = [
        _cosine_similarity(v)
        for v in train_views
    ]

    cross_sims = [
        _cross_cosine_similarity(te, tr)
        for tr, te in zip(train_views, test_views)
    ]

    fused_train = np.mean(train_sims, axis=0)

    fused_cross = np.mean(cross_sims, axis=0)

    train_adj = _topk_mask(fused_train, k)

    train_adj = np.maximum(
        train_adj,
        train_adj.T,
    )

    train_adj[train_adj < threshold] = 0.0

    if self_loop:
        np.fill_diagonal(train_adj, 1.0)

    train_adj = _row_normalize(train_adj)

    n_tr = train_adj.shape[0]
    n_te = fused_cross.shape[0]

    full = np.zeros(
        (n_tr + n_te, n_tr + n_te),
        dtype=np.float32,
    )

    full[:n_tr, :n_tr] = train_adj

    cross_adj = _topk_mask(fused_cross, k)

    cross_adj[cross_adj < threshold] = 0.0

    full[n_tr:, :n_tr] = cross_adj
    full[:n_tr, n_tr:] = cross_adj.T

    if self_loop:
        np.fill_diagonal(full, 1.0)

    full = _row_normalize(full)

    return full.astype(np.float32)


# =========================================================
# SNF INDUCTIVE GRAPH (LGG + KIPAN)
# =========================================================

def build_train_test_graph_from_views(
    train_views: List[np.ndarray],
    test_views: List[np.ndarray],
    snf_k: int = 20,
    snf_mu: float = 0.5,
    cross_k: int = 20,
    edge_dropout: float = 0.0,
    seed: int = 42,
) -> np.ndarray:

    train_aff = compute.make_affinity(
        train_views,
        metric="sqeuclidean",
        K=snf_k,
        mu=snf_mu,
    )

    fused_train = compute.snf(
        train_aff,
        K=snf_k,
    ).astype(np.float32)

    fused_train = (
        fused_train + fused_train.T
    ) / 2.0

    cross_mats = []

    for tr, te in zip(
        train_views,
        test_views,
    ):

        d = pairwise_distances(
            te,
            tr,
            metric="sqeuclidean",
        )

        s = _rbf_from_dist(d)

        s = _topk_mask(
            s,
            cross_k,
        )

        cross_mats.append(s)

    cross = np.mean(
        cross_mats,
        axis=0,
    ).astype(np.float32)

    n_tr = fused_train.shape[0]
    n_te = cross.shape[0]

    full = np.zeros(
        (n_tr + n_te, n_tr + n_te),
        dtype=np.float32,
    )

    full[:n_tr, :n_tr] = fused_train

    full[n_tr:, :n_tr] = cross
    full[:n_tr, n_tr:] = cross.T

    np.fill_diagonal(full, 1.0)

    full = _row_normalize(full)

    full = apply_edge_dropout(
        full,
        edge_dropout,
        seed=seed,
    )

    return full.astype(np.float32)


# =========================================================
# SNF TRANSDUCTIVE GRAPH
# =========================================================

def build_transductive_graph_from_views(
    train_views: List[np.ndarray],
    test_views: List[np.ndarray],
    snf_k: int = 20,
    snf_mu: float = 0.5,
    edge_dropout: float = 0.0,
    seed: int = 42,
) -> np.ndarray:

    all_views = [
        np.vstack([tr, te]).astype(np.float32)
        for tr, te in zip(
            train_views,
            test_views,
        )
    ]

    aff = compute.make_affinity(
        all_views,
        metric="sqeuclidean",
        K=snf_k,
        mu=snf_mu,
    )

    fused = compute.snf(
        aff,
        K=snf_k,
    ).astype(np.float32)

    fused = (fused + fused.T) / 2.0

    np.fill_diagonal(fused, 1.0)

    fused = _row_normalize(fused)

    fused = apply_edge_dropout(
        fused,
        edge_dropout,
        seed=seed,
    )

    return fused.astype(np.float32)


# =========================================================
# DYNAMIC GRAPH REFINEMENT
# =========================================================

def refine_adjacency_from_embeddings(
    adj: torch.Tensor,
    embeddings: torch.Tensor,
    k: int = 20,
    blend: float = 0.25,
) -> torch.Tensor:

    if blend <= 0:
        return adj

    n = embeddings.shape[0]

    k = max(
        1,
        min(int(k), n),
    )

    z = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    sim = torch.relu(z @ z.T)

    _, idx = torch.topk(
        sim,
        k=k,
        dim=1,
    )

    dyn = torch.zeros_like(sim)

    dyn.scatter_(
        1,
        idx,
        sim.gather(1, idx),
    )

    dyn = 0.5 * (dyn + dyn.T)

    dyn = dyn * (adj > 0).to(dyn.dtype)

    dyn.fill_diagonal_(1.0)

    dyn = dyn / dyn.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)

    out = (1.0 - blend) * adj + blend * dyn

    return out / out.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)