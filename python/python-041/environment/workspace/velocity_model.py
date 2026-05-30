
import numpy as np
from fractal_scattering import fractal_porosity_field


def hermite_cubic_value(x1, f1, d1, x2, f2, d2, x):
    h = x2 - x1
    if abs(h) < 1e-14:
        return f1, d1, 0.0, 0.0
    df = (f2 - f1) / h
    c2 = -(2.0 * d1 - 3.0 * df + d2) / h
    c3 = (d1 - 2.0 * df + d2) / (h ** 2)
    dx = x - x1
    f = f1 + dx * (d1 + dx * (c2 + dx * c3))
    d = d1 + dx * (2.0 * c2 + dx * 3.0 * c3)
    s = 2.0 * c2 + dx * 6.0 * c3
    t = 6.0 * c3
    return f, d, s, t


def hermite_cubic_spline_value(xn, fn, dn, x):
    xn = np.asarray(xn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    dn = np.asarray(dn, dtype=float)
    x_arr = np.atleast_1d(x)
    f_out = np.zeros_like(x_arr, dtype=float)
    nn = len(xn)
    for j in range(len(x_arr)):
        xv = x_arr[j]

        if xv <= xn[0]:
            i1, i2 = 0, 1
        elif xv >= xn[-1]:
            i1, i2 = nn - 2, nn - 1
        else:
            i1 = np.searchsorted(xn, xv) - 1
            i2 = i1 + 1
        f_val, _, _, _ = hermite_cubic_value(
            xn[i1], fn[i1], dn[i1], xn[i2], fn[i2], dn[i2], xv
        )
        f_out[j] = f_val
    if np.isscalar(x):
        return float(f_out[0])
    return f_out


def hermite_cubic_spline_integral(xn, fn, dn):
    xn = np.asarray(xn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    dn = np.asarray(dn, dtype=float)
    nn = len(xn)
    if nn < 2:
        return 0.0
    il = np.arange(0, nn - 1)
    ir = np.arange(1, nn)
    h = xn[ir] - xn[il]
    q = np.sum(0.5 * h * (fn[il] + fn[ir] + h * (dn[il] - dn[ir]) / 6.0))
    return q


def ising_2d_initialize(m, n, thresh=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    c1 = np.ones((m, n), dtype=int)
    r = rng.random((m, n))
    c1[r <= thresh] = -1
    return c1


def ising_2d_agree(m, n, c1):
    c5 = (
        c1
        + np.roll(c1, -1, axis=0)
        + np.roll(c1, 1, axis=0)
        + np.roll(c1, -1, axis=1)
        + np.roll(c1, 1, axis=1)
    )
    pos_mask = c1 > 0
    neg_mask = c1 < 0
    c5 = c5.astype(float)
    c5[pos_mask] = (5.0 + c5[pos_mask]) / 2.0
    c5[neg_mask] = (5.0 - c5[neg_mask]) / 2.0
    return c5.astype(int)


def ising_2d_transition(m, n, iterations, prob, c1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    for step in range(iterations):
        c5 = ising_2d_agree(m, n, c1)
        threshold = np.zeros((m, n))
        for j in range(5):
            mask = (c5 == j + 1)
            threshold[mask] = prob[j]
        r = rng.random((m, n))
        flip = r < threshold
        c1[flip] = -c1[flip]
    return c1


def cvt_sample_generators(n, xlim=(0.0, 1.0), zlim=(0.0, 1.0), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    gen_x = rng.uniform(xlim[0], xlim[1], n)
    gen_z = rng.uniform(zlim[0], zlim[1], n)
    return gen_x, gen_z


def cvt_centroid_estimate(gen_x, gen_z, sample_num, xlim, zlim, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(gen_x)
    cen_x = np.zeros(n)
    cen_z = np.zeros(n)
    cen_count = np.zeros(n)
    x_samples = rng.uniform(xlim[0], xlim[1], sample_num)
    z_samples = rng.uniform(zlim[0], zlim[1], sample_num)
    for s in range(sample_num):
        dx = x_samples[s] - gen_x
        dz = z_samples[s] - gen_z
        dist2 = dx ** 2 + dz ** 2
        i_min = np.argmin(dist2)
        cen_x[i_min] += x_samples[s]
        cen_z[i_min] += z_samples[s]
        cen_count[i_min] += 1

    for i in range(n):
        if cen_count[i] == 0:
            cen_x[i] = gen_x[i]
            cen_z[i] = gen_z[i]
        else:
            cen_x[i] /= cen_count[i]
            cen_z[i] /= cen_count[i]
    return cen_x, cen_z


def cvt_optimize(n, xlim, zlim, n_steps=10, sample_num=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    gen_x, gen_z = cvt_sample_generators(n, xlim, zlim, rng=rng)
    for _ in range(n_steps):
        gen_x, gen_z = cvt_centroid_estimate(gen_x, gen_z, sample_num, xlim, zlim, rng=rng)
    return gen_x, gen_z


def build_velocity_model(nx, nz, v0=3000.0, dv_ising=500.0, dv_fractal=200.0,
                         ising_thresh=0.5, ising_iter=15, fractal_dim=1.8,
                         use_cvt=False, n_cvt=20, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x_coords = np.linspace(0.0, 1.0, nx)
    z_coords = np.linspace(0.0, 1.0, nz)
    

    ising_field = ising_2d_initialize(nz, nx, thresh=ising_thresh, rng=rng)
    prob = np.array([0.98, 0.85, 0.50, 0.15, 0.02])
    ising_field = ising_2d_transition(nz, nx, ising_iter, prob, ising_field, rng=rng)
    

    fractal_field = fractal_porosity_field(nx, nz, fractal_dim=fractal_dim, rng=rng)
    

    velocity = v0 + dv_ising * ising_field + dv_fractal * (fractal_field - 0.5)
    

    if use_cvt and n_cvt >= 2:
        gen_x, gen_z = cvt_optimize(n_cvt, (0.0, 1.0), (0.0, 1.0), n_steps=5,
                                     sample_num=1000, rng=rng)

        gen_v = np.zeros(n_cvt)
        for i in range(n_cvt):
            ix = min(int(gen_x[i] * (nx - 1)), nx - 1)
            iz = min(int(gen_z[i] * (nz - 1)), nz - 1)
            gen_v[i] = velocity[iz, ix]

        gen_dv = np.zeros(n_cvt)
        gen_dv[0] = (gen_v[1] - gen_v[0]) / (gen_x[1] - gen_x[0])
        gen_dv[-1] = (gen_v[-1] - gen_v[-2]) / (gen_x[-1] - gen_x[-2])
        for i in range(1, n_cvt - 1):
            gen_dv[i] = 0.5 * ((gen_v[i] - gen_v[i - 1]) / (gen_x[i] - gen_x[i - 1])
                                + (gen_v[i + 1] - gen_v[i]) / (gen_x[i + 1] - gen_x[i]))

        for iz in range(nz):

            v_profile = hermite_cubic_spline_value(gen_x, gen_v, gen_dv, x_coords)
            velocity[iz, :] = v_profile
    

    velocity = np.clip(velocity, 1000.0, 8000.0)
    return velocity, x_coords, z_coords
