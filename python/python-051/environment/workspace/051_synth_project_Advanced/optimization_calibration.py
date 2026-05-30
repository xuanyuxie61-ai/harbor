
import numpy as np






def nelder_mead_optimize(func, x0, rho=1.0, xi=2.0, gam=0.5, sig=0.5,
                         tol=1e-6, max_feval=500):
    x = x0.copy().astype(float)
    n_dim = x.shape[1]
    n_vert = x.shape[0]
    if n_vert != n_dim + 1:
        raise ValueError("单纯形顶点数必须为 n_dim + 1")

    def evaluate_simplex(pts):
        return np.array([func(pts[i, :]) for i in range(pts.shape[0])])

    f = evaluate_simplex(x)
    n_feval = n_vert


    idx = np.argsort(f)
    f = f[idx]
    x = x[idx, :]

    converged = False
    diverged = False

    while not converged and not diverged:
        x_bar = np.mean(x[:-1, :], axis=0)


        x_r = (1.0 + rho) * x_bar - rho * x[-1, :]
        f_r = func(x_r)
        n_feval += 1

        if f[0] <= f_r <= f[-2]:
            x[-1, :] = x_r
            f[-1] = f_r
        elif f_r < f[0]:

            x_e = (1.0 + rho * xi) * x_bar - rho * xi * x[-1, :]
            f_e = func(x_e)
            n_feval += 1
            if f_e < f_r:
                x[-1, :] = x_e
                f[-1] = f_e
            else:
                x[-1, :] = x_r
                f[-1] = f_r
        elif f[-2] <= f_r < f[-1]:

            x_c = (1.0 + rho * gam) * x_bar - rho * gam * x[-1, :]
            f_c = func(x_c)
            n_feval += 1
            if f_c <= f_r:
                x[-1, :] = x_c
                f[-1] = f_c
            else:
                x, f = _shrink_simplex(x, f, func, sig)
                n_feval += n_dim
        else:

            x_c = (1.0 - gam) * x_bar + gam * x[-1, :]
            f_c = func(x_c)
            n_feval += 1
            if f_c < f[-1]:
                x[-1, :] = x_c
                f[-1] = f_c
            else:
                x, f = _shrink_simplex(x, f, func, sig)
                n_feval += n_dim

        idx = np.argsort(f)
        f = f[idx]
        x = x[idx, :]

        converged = (f[-1] - f[0] < tol)
        diverged = (n_feval > max_feval)

    return x[0, :], f[0], n_feval


def _shrink_simplex(x, f, func, sig):
    n_dim = x.shape[1]
    x_best = x[0, :]
    f[0] = func(x_best)
    for i in range(1, n_dim + 1):
        x[i, :] = sig * x[i, :] + (1.0 - sig) * x_best
        f[i] = func(x[i, :])
    return x, f






def path_cost(n, distance, p):
    cost = 0.0
    i1 = n - 1
    for i2 in range(n):
        cost += distance[p[i1], p[i2]]
        i1 = i2
    return cost


def tsp_descent(distance, variation_num=2000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = distance.shape[0]
    if n < 4:
        raise ValueError("城市数 n >= 4")


    if not np.allclose(distance, distance.T):
        raise ValueError("距离矩阵必须对称")
    if np.any(np.diag(distance) != 0.0):
        raise ValueError("距离矩阵对角线必须为零")

    p = np.random.permutation(n)
    cost = path_cost(n, distance, p)

    for _ in range(variation_num):

        c = np.random.choice(n, 2, replace=False)
        c = np.sort(c)
        i1, i2 = c[0], c[1]
        if i1 + 1 < i2:
            p2 = p.copy()
            p2[i1 + 1:i2 + 1] = np.roll(p2[i1 + 1:i2 + 1], 1)
            p2[i1 + 1] = p[i2]
            cost2 = path_cost(n, distance, p2)
            if cost2 < cost:
                p = p2
                cost = cost2


        c = np.random.choice(n, 2, replace=False)
        c = np.sort(c)
        i1, i2 = c[0], c[1]
        p2 = p.copy()
        p2[i1:i2 + 1] = p2[i1:i2 + 1][::-1]
        cost2 = path_cost(n, distance, p2)
        if cost2 < cost:
            p = p2
            cost = cost2

    return p, cost


def generate_sampling_stations(n_stations, Lx, Lz, depth_min=50.0):
    x = np.random.uniform(0.0, Lx, n_stations)
    z = np.random.uniform(depth_min, Lz, n_stations)
    return np.column_stack([x, z])


def build_distance_matrix(stations):
    n = stations.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(stations[i] - stations[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist
