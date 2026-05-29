"""
kmeans_clustering.py
====================
K-means clustering for poroelastic parameter zone identification.

Incorporates the core K-means algorithm (ASA136) for partitioning
nodes or elements into homogeneous material zones based on their
wave velocity, permeability, or other physical attributes.

This enables identification of geological facies or lithological
units within a heterogeneous porous medium.
"""

import numpy as np


def kmeans(a, k, iter_max=100, init_centers=None):
    """
    K-means clustering algorithm (Hartigan-Wong).

    Parameters
    ----------
    a : ndarray, shape (m, n)
        Data points in n-dimensional space.
    k : int
        Number of clusters.
    iter_max : int
        Maximum iterations.
    init_centers : ndarray, optional
        Initial cluster centers, shape (k, n).

    Returns
    -------
    c : ndarray, shape (k, n)
        Final cluster centers.
    ic1 : ndarray, shape (m,)
        Cluster assignment for each point.
    nc : ndarray, shape (k,)
        Number of points in each cluster.
    wss : ndarray, shape (k,)
        Within-cluster sum of squares.
    ifault : int
        Error indicator (0=ok, 1=empty cluster, 2=max iter, 3=bad k).
    """
    a = np.asarray(a, dtype=float)
    m, n = a.shape

    if k <= 1 or m <= k:
        return None, None, None, None, 3

    if init_centers is not None:
        c = np.asarray(init_centers, dtype=float).copy()
    else:
        # Random initialization
        idx = np.random.choice(m, k, replace=False)
        c = a[idx, :].copy()

    ic1 = np.zeros(m, dtype=int)
    ic2 = np.zeros(m, dtype=int)
    dt = np.zeros(2)

    # Initial assignment: find two closest centers for each point
    for i in range(m):
        ic1[i] = 0
        ic2[i] = 1
        for il in range(2):
            dt[il] = np.sum((a[i, :] - c[il, :]) ** 2)
        if dt[1] < dt[0]:
            ic1[i], ic2[i] = ic2[i], ic1[i]
            dt[0], dt[1] = dt[1], dt[0]

        for l in range(2, k):
            db = np.sum((a[i, :] - c[l, :]) ** 2)
            if db < dt[1]:
                if dt[0] <= db:
                    dt[1] = db
                    ic2[i] = l
                else:
                    dt[1] = dt[0]
                    ic2[i] = ic1[i]
                    dt[0] = db
                    ic1[i] = l

    # Update centers
    nc = np.zeros(k, dtype=int)
    c[:, :] = 0.0
    for i in range(m):
        l = ic1[i]
        nc[l] += 1
        c[l, :] += a[i, :]

    if np.any(nc == 0):
        return c, ic1, nc, None, 1

    for l in range(k):
        c[l, :] /= nc[l]

    # Iterative refinement (simplified optimal-transfer + quick-transfer)
    an1 = np.zeros(k)
    an2 = np.zeros(k)
    for l in range(k):
        aa = float(nc[l])
        an2[l] = aa / (aa + 1.0)
        an1[l] = aa / (aa - 1.0) if aa > 1.0 else np.inf

    ncp = np.zeros(k, dtype=int)
    d = np.zeros(m)
    itran = np.ones(k, dtype=int)
    live = np.zeros(k, dtype=int)
    indx = 0

    for _ in range(iter_max):
        # Optimal transfer stage
        for i in range(m):
            l1 = ic1[i]
            l2 = ic2[i]
            r2 = np.sum((a[i, :] - c[l2, :]) ** 2) * an2[l2]
            r1 = np.sum((a[i, :] - c[l1, :]) ** 2) * an1[l1]

            if r2 < r1 and nc[l1] > 1:
                # Transfer point i from l1 to l2
                nc[l1] -= 1
                nc[l2] += 1
                aa = float(nc[l1])
                an1[l1] = aa / (aa - 1.0) if aa > 1.0 else np.inf
                an2[l1] = aa / (aa + 1.0)
                aa = float(nc[l2])
                an1[l2] = aa / (aa - 1.0) if aa > 1.0 else np.inf
                an2[l2] = aa / (aa + 1.0)

                # Update centers incrementally
                c[l1, :] = (c[l1, :] * (nc[l1] + 1) - a[i, :]) / max(nc[l1], 1)
                c[l2, :] = (c[l2, :] * (nc[l2] - 1) + a[i, :]) / max(nc[l2], 1)

                ic1[i] = l2
                # Recompute ic2
                dt_min = np.inf
                dt_second = np.inf
                ic2[i] = l1
                for l in range(k):
                    if l == ic1[i]:
                        continue
                    dd = np.sum((a[i, :] - c[l, :]) ** 2)
                    if dd < dt_min:
                        dt_second = dt_min
                        dt_min = dd
                        ic2[i] = ic1[i]
                        ic1[i] = l
                    elif dd < dt_second:
                        dt_second = dd
                        ic2[i] = l
                indx = 0
            else:
                indx += 1

        if indx >= m:
            break

    # Compute WSS
    wss = np.zeros(k)
    c[:, :] = 0.0
    for i in range(m):
        l = ic1[i]
        c[l, :] += a[i, :]
    for l in range(k):
        c[l, :] /= max(nc[l], 1)
    for i in range(m):
        l = ic1[i]
        wss[l] += np.sum((a[i, :] - c[l, :]) ** 2)

    return c, ic1, nc, wss, 0


def cluster_velocity_zones(nodes, velocity_field, k=3):
    """
    Cluster nodes into k zones based on wave velocity magnitude.

    Parameters
    ----------
    nodes : ndarray, shape (m, 2)
        Spatial coordinates.
    velocity_field : ndarray, shape (m, 2)
        Velocity vectors.
    k : int
        Number of clusters.

    Returns
    -------
    zones : ndarray, shape (m,)
        Zone index for each node.
    centers : ndarray, shape (k, 3)
        Cluster centers (x, y, |v|).
    """
    v_mag = np.linalg.norm(velocity_field, axis=1).reshape(-1, 1)
    features = np.hstack([nodes, v_mag])
    c, ic1, nc, wss, fault = kmeans(features, k)
    if fault != 0:
        # Fallback: equal partitioning
        ic1 = np.floor(np.linspace(0, k - 1e-6, len(nodes))).astype(int)
    return ic1, c
