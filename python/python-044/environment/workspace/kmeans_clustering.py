
import numpy as np


def kmeans(a, k, iter_max=100, init_centers=None):
    a = np.asarray(a, dtype=float)
    m, n = a.shape

    if k <= 1 or m <= k:
        return None, None, None, None, 3

    if init_centers is not None:
        c = np.asarray(init_centers, dtype=float).copy()
    else:

        idx = np.random.choice(m, k, replace=False)
        c = a[idx, :].copy()

    ic1 = np.zeros(m, dtype=int)
    ic2 = np.zeros(m, dtype=int)
    dt = np.zeros(2)


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

        for i in range(m):
            l1 = ic1[i]
            l2 = ic2[i]
            r2 = np.sum((a[i, :] - c[l2, :]) ** 2) * an2[l2]
            r1 = np.sum((a[i, :] - c[l1, :]) ** 2) * an1[l1]

            if r2 < r1 and nc[l1] > 1:

                nc[l1] -= 1
                nc[l2] += 1
                aa = float(nc[l1])
                an1[l1] = aa / (aa - 1.0) if aa > 1.0 else np.inf
                an2[l1] = aa / (aa + 1.0)
                aa = float(nc[l2])
                an1[l2] = aa / (aa - 1.0) if aa > 1.0 else np.inf
                an2[l2] = aa / (aa + 1.0)


                c[l1, :] = (c[l1, :] * (nc[l1] + 1) - a[i, :]) / max(nc[l1], 1)
                c[l2, :] = (c[l2, :] * (nc[l2] - 1) + a[i, :]) / max(nc[l2], 1)

                ic1[i] = l2

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
    v_mag = np.linalg.norm(velocity_field, axis=1).reshape(-1, 1)
    features = np.hstack([nodes, v_mag])
    c, ic1, nc, wss, fault = kmeans(features, k)
    if fault != 0:

        ic1 = np.floor(np.linspace(0, k - 1e-6, len(nodes))).astype(int)
    return ic1, c
