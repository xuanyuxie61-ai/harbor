
import numpy as np






def comp_next(n, k, a=None, more=False, h=0, t=0):
    if a is None:
        a = np.zeros(k, dtype=int)
    if not more:
        t = n
        h = 0
        a[0] = n
        if k > 1:
            a[1:] = 0
        more = True
        return a, more, h, t

    if 1 < t:
        h = 0
    h += 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] = a[h] + 1
    more = (a[k - 1] != n)
    return a, more, h, t


def composition_count(n, k):
    from math import comb
    return comb(n + k - 1, k - 1)






def clencurt_weights(N1):
    if N1 == 1:
        return np.array([2.0])
    N = N1 - 1
    c = np.zeros(N1)

    idx = np.arange(0, N1, 2)
    vals = 2.0 / (1.0 - idx[1:] ** 2) if len(idx) > 1 else np.array([])
    c[0] = 2.0
    if len(vals) > 0:
        c[2:N1:2] = vals


    cc = np.concatenate([c, c[N:0:-1]])
    f = np.real(np.fft.ifft(cc))
    w = 2.0 * np.concatenate([[f[0]], 2.0 * f[1:N], [f[N]]])
    return w


def clencurt_nodes_weights(level):
    if level == 0:
        return np.array([0.0]), np.array([2.0])
    n = 2 ** level
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)
    w = clencurt_weights(n + 1)
    return x, w






def tensor_grid_1d_values(levels, dim):
    nodes_1d = []
    weights_1d = []
    for d in range(dim):
        x, w = clencurt_nodes_weights(levels[d])
        nodes_1d.append(x)
        weights_1d.append(w)


    grids = np.meshgrid(*nodes_1d, indexing='ij')
    points = np.stack([g.ravel() for g in grids], axis=1)

    w_grids = np.meshgrid(*weights_1d, indexing='ij')
    weights = np.ones(points.shape[0])
    for wg in w_grids:
        weights *= wg.ravel()

    return points, weights


def sparse_grid_quadrature(dim, max_level, func):
    if dim < 1:
        raise ValueError("dim >= 1")
    if max_level < 0:
        raise ValueError("max_level >= 0")

    if dim == 1:
        x, w = clencurt_nodes_weights(max_level)
        integral = 0.0
        for i in range(len(x)):
            integral += w[i] * func(x[i:i + 1])
        return integral, len(x)


    all_points = []
    all_weights = []

    q = max_level + dim - 1

    a = np.zeros(dim, dtype=int)
    more = False
    h = 0
    t = 0
    count = 0
    while True:
        a, more, h, t = comp_next(q, dim, a if count > 0 else None, more, h, t)
        count += 1


        levels = a

        pts, wts = tensor_grid_1d_values(levels, dim)


        s = np.sum(levels)
        if s > q:
            if not more:
                break
            continue
        k = q - s
        if k < 0 or k >= dim:
            if not more:
                break
            continue
        coeff = (-1) ** k
        from math import comb
        coeff *= comb(dim - 1, k)

        all_points.append(pts)
        all_weights.append(coeff * wts)

        if not more:
            break

    if len(all_points) == 0:

        return func(np.zeros(dim)), 1


    pts_all = np.vstack(all_points)
    wts_all = np.concatenate(all_weights)


    tol = 1e-12
    pts_rounded = np.round(pts_all / tol) * tol
    unique_pts, inv_idx = np.unique(pts_rounded, axis=0, return_inverse=True)
    n_unique = unique_pts.shape[0]
    wts_condensed = np.zeros(n_unique)
    for i in range(len(wts_all)):
        wts_condensed[inv_idx[i]] += wts_all[i]


    result = 0.0
    for i in range(n_unique):
        result += wts_condensed[i] * func(unique_pts[i])

    return result, n_unique






def stroud_cn_leg_5(dim):
    if dim not in (4, 5, 6):

        return None, None

    volume = 2.0 ** dim
    o = dim ** 2 + dim + 2
    x = np.zeros((dim, o))
    w = np.zeros(o)

    if dim == 4:
        eta = 0.778984505799815
        lam = 1.284565137874656
        xsi = -0.713647298819253
        mu = -0.715669761974162
        gamma = 0.217089151000943
        a = 0.206186096875899e-1 * volume
        b = 0.975705820221664e-2 * volume
        c = 0.733921929172573e-1 * volume
    elif dim == 5:
        eta = 0.522478547481276
        lam = 0.936135175985774
        xsi = -0.246351362101519
        mu = -0.496308106093758
        gamma = 0.827180176822930
        a = 0.631976901960153e-1 * volume
        b = 0.511464127430166e-1 * volume
        c = 0.181070246088902e-1 * volume
    else:
        eta = 0.660225291773525
        lam = 1.064581294844754
        xsi = 0.0
        mu = -0.660225291773525
        gamma = 0.660225291773525
        a = 0.182742214532872e-1 * volume
        b = 0.346020761245675e-1 * volume
        c = 0.182742214532872e-1 * volume

    k = 0

    x[:, k] = eta
    w[k] = a
    k += 1

    x[:, k] = -eta
    w[k] = a
    k += 1


    for i1 in range(dim):
        x[:, k] = xsi
        x[i1, k] = lam
        w[k] = b
        k += 1


    for i1 in range(dim):
        x[:, k] = -xsi
        x[i1, k] = -lam
        w[k] = b
        k += 1


    for i1 in range(dim - 1):
        for i2 in range(i1 + 1, dim):
            x[:, k] = gamma
            x[i1, k] = mu
            x[i2, k] = mu
            w[k] = c
            k += 1


    for i1 in range(dim - 1):
        for i2 in range(i1 + 1, dim):
            x[:, k] = -gamma
            x[i1, k] = -mu
            x[i2, k] = -mu
            w[k] = c
            k += 1

    return x.T, w






def legendre_monomial_integral(expon):
    if expon % 2 == 0:
        return 2.0 / (expon + 1)
    else:
        return 0.0


def test_quadrature_exactness(points, weights, degree_max=9):
    tol = 1e-12
    max_exact = -1
    for degree in range(degree_max + 1):
        exact = legendre_monomial_integral(degree)
        quad = np.sum(weights * (points ** degree))
        if exact == 0.0:
            err = abs(quad)
        else:
            err = abs((quad - exact) / exact)
        if err < tol:
            max_exact = degree
        else:
            break
    return max_exact
