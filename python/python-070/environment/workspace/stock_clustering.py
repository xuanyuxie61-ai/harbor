
import numpy as np
from utils import NumericalConfig


def euclidean_distance(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.sqrt(np.sum(diff ** 2))


def calcdist(data, center):
    data = np.asarray(data, dtype=float)
    center = np.asarray(center, dtype=float)
    diff = data - center
    return np.sqrt(np.sum(diff ** 2, axis=1))


def alldist(centers):
    k = centers.shape[0]
    distmat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            d = euclidean_distance(centers[i], centers[j])
            distmat[i, j] = d
            distmat[j, i] = d
    return distmat


def furthest_first_init(data, k):
    n, dim = data.shape
    centers = np.zeros((k, dim), dtype=float)


    centers[0, :] = np.mean(data, axis=0) + 0.01 * np.std(data, axis=0) * np.random.randn(dim)

    mincenter = np.zeros(n, dtype=int)
    mindist = np.full(n, np.inf, dtype=float)
    lower = np.zeros((n, k), dtype=float)
    computed = 0

    for j in range(1, k):

        dists = calcdist(data, centers[j - 1, :])
        update = dists < mindist
        mindist[update] = dists[update]
        mincenter[update] = j - 1
        lower[:, j - 1] = dists
        computed += n


        farthest_idx = np.argmax(mindist)
        centers[j, :] = data[farthest_idx, :]

    return centers, mincenter, mindist, lower, computed


def kmeans_fast(data, k, init_centers=None, max_iter=100, tol=1e-6):
    n, dim = data.shape

    if init_centers is None:
        centers, labels, mindist, lower, _ = furthest_first_init(data, k)
    else:
        centers = np.asarray(init_centers, dtype=float).copy()
        labels = np.zeros(n, dtype=int)
        mindist = np.full(n, np.inf, dtype=float)
        lower = np.zeros((n, k), dtype=float)
        for j in range(k):
            dists = calcdist(data, centers[j, :])
            lower[:, j] = dists
            update = dists < mindist
            mindist[update] = dists[update]
            labels[update] = j


    centdist = 0.5 * alldist(centers) + np.diag(np.full(k, np.inf))
    pop = np.zeros(k, dtype=int)
    for j in range(k):
        pop[j] = np.sum(labels == j)

    old_labels = labels.copy()

    for iteration in range(max_iter):

        nndist = np.min(centdist + np.diag(np.full(k, np.inf)), axis=1)
        mobile = np.where(mindist > nndist[labels])[0]

        recalculated = np.zeros(n, dtype=bool)

        for j in range(k):

            mdm = mindist[mobile]
            mcm = labels[mobile]
            track = np.where(mdm > centdist[mcm, j])[0]
            if len(track) == 0:
                continue

            alt = np.where(mdm[track] > lower[mobile[track], j])[0]
            if len(alt) == 0:
                continue

            track1 = mobile[track[alt]]


            redo = track1[~recalculated[track1]]
            if len(redo) > 0:
                c_redo = labels[redo]
                for jj in np.unique(c_redo):
                    rp = redo[c_redo == jj]
                    udist = calcdist(data[rp, :], centers[jj, :])
                    lower[rp, jj] = udist
                    mindist[rp] = udist
                recalculated[redo] = True


            track2 = np.where(mindist[track1] > centdist[labels[track1], j])[0]
            track1 = track1[track2]
            if len(track1) == 0:
                continue

            track4 = np.where(lower[track1, j] < mindist[track1])[0]
            if len(track4) == 0:
                continue

            track5 = track1[track4]
            jdist = calcdist(data[track5, :], centers[j, :])
            lower[track5, j] = jdist


            update = jdist < mindist[track5]
            track3 = track5[update]
            mindist[track3] = jdist[update]
            labels[track3] = j


        diff = np.where(labels != old_labels)[0]
        if len(diff) == 0:
            break

        diffj = np.unique(np.concatenate([labels[diff], old_labels[diff]]))
        diffj = diffj[diffj >= 0]

        for j in diffj:
            track = np.where(labels == j)[0]
            pop[j] = len(track)
            if pop[j] == 0:
                continue
            centers[j, :] = np.mean(data[track, :], axis=0)


        for j in diffj:
            offset = euclidean_distance(centers[j, :], centers[j, :])
            if offset == 0:
                continue
            track = np.where(labels == j)[0]
            mindist[track] += offset
            lower[:, j] = np.maximum(lower[:, j] - offset, 0.0)


        centdist = 0.5 * alldist(centers) + np.diag(np.full(k, np.inf))
        recalculated = np.zeros(n, dtype=bool)

        old_labels = labels.copy()


        if len(diff) < tol * n and iteration > 0:
            break


    inertia = 0.0
    for j in range(k):
        track = np.where(labels == j)[0]
        if len(track) > 0:
            inertia += np.sum(calcdist(data[track, :], centers[j, :]) ** 2)

    return centers, labels, inertia


def cluster_habitat_zones(n_stations, env_features, n_zones=4):
    if env_features.ndim == 1:
        env_features = env_features.reshape(-1, 1)


    mean_feat = np.mean(env_features, axis=0)
    std_feat = np.std(env_features, axis=0)
    std_feat = np.where(std_feat < NumericalConfig.EPS, NumericalConfig.EPS, std_feat)
    normalized = (env_features - mean_feat) / std_feat

    centers, zones, inertia = kmeans_fast(normalized, n_zones)

    zone_stats = {
        'inertia': inertia,
        'n_points_per_zone': [int(np.sum(zones == j)) for j in range(n_zones)],
        'zone_centers_original_scale': centers * std_feat + mean_feat,
    }

    return zones, centers, zone_stats
