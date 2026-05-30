
import numpy as np






def lorenz96_parameters(n=4, force=8.0, perturb=0.001, t0=0.0, y0=None, tstop=30.0):
    if n < 3:
        raise ValueError("lorenz96_parameters: n 必须至少为 3")
    if y0 is None:
        s = perturb * np.random.randn(n)
        y0 = force * np.ones(n) + s
    y0 = np.asarray(y0, dtype=float)
    if y0.shape[0] != n:
        raise ValueError("lorenz96_parameters: y0 长度必须与 n 一致")
    return n, force, perturb, t0, y0, tstop


def lorenz96_deriv(t, y, force=8.0):
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 3:
        raise ValueError("lorenz96_deriv: y 长度必须至少为 3")

    i = np.arange(n)
    ip1 = np.roll(i, -1)
    im1 = np.roll(i, 1)
    im2 = np.roll(i, 2)

    dydt = (y[ip1] - y[im2]) * y[im1] - y[i] + force
    return dydt






def duffing_parameters(alpha=1.0, beta=5.0, gamma=8.0, delta=0.02,
                       omega=0.5, t0=0.0, y0=None, tstop=100.0):
    if y0 is None:
        y0 = np.array([1.0, 0.0], dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if y0.shape[0] != 2:
        raise ValueError("duffing_parameters: y0 必须为二维向量 [x, x']")
    return alpha, beta, gamma, delta, omega, t0, y0, tstop


def duffing_deriv(t, y, alpha=1.0, beta=5.0, gamma=8.0, delta=0.02, omega=0.5):
    y = np.asarray(y, dtype=float)
    if y.shape[0] != 2:
        raise ValueError("duffing_deriv: y 必须为二维向量")
    y1, y2 = y[0], y[1]
    dy1dt = y2
    dy2dt = -delta * y2 - alpha * y1 - beta * (y1 ** 3) + gamma * np.cos(omega * t)
    return np.array([dy1dt, dy2dt], dtype=float)






def oregonator_parameters(eta1=None, eta2=None, q=None, f=1.0,
                          t0=0.0, y0=None, tstop=25.0):
    a = 0.06
    b = 0.02
    k2 = 2.4e6
    k3 = 1.28
    k4 = 3.0e3
    k5 = 33.6
    kc = 1.0

    if eta1 is None:
        eta1 = kc * b / (k5 * a)
    if eta2 is None:
        eta2 = 2.0 * kc * k4 * b / (k2 * k5 * a)
    if q is None:
        q = 2.0 * k3 * k4 / (k2 * k5)
    if y0 is None:
        y0 = np.array([1.0, 1.0, 1.0], dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if y0.shape[0] != 3:
        raise ValueError("oregonator_parameters: y0 必须为三维向量")
    return eta1, eta2, q, f, t0, y0, tstop


def oregonator_deriv(t, y, eta1, eta2, q, f):
    y = np.asarray(y, dtype=float)
    if y.shape[0] != 3:
        raise ValueError("oregonator_deriv: y 必须为三维向量")
    u, v, w = y[0], y[1], y[2]

    u = np.clip(u, -100.0, 100.0)
    v = np.clip(v, -100.0, 100.0)
    w = np.clip(w, -100.0, 100.0)
    dudt = (q * v - u * v + u * (1.0 - u)) / eta1
    dvdt = (-q * v - u * v + f * w) / eta2
    dwdt = u - w
    return np.array([dudt, dvdt, dwdt], dtype=float)






def multi_factor_coupling(t, lorenz_state, duffing_state, oregonator_state,
                          n_factors=3, coupling_matrix=None):
    lorenz_state = np.asarray(lorenz_state, dtype=float)
    duffing_state = np.asarray(duffing_state, dtype=float)
    oregonator_state = np.asarray(oregonator_state, dtype=float)









    raise NotImplementedError("HOLE_3: 多因子耦合投影尚未实现")
