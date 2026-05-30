import numpy as np
from constants import TINY, M_HIGGS, M_Z




def chebyshev_norm(j, x):
    x = np.clip(x, -1.0, 1.0)
    if j == 0:
        return np.ones_like(x)
    return np.sqrt(2.0) * np.cos(j * np.arccos(x))


def chebyshev_eval_matrix(degree, pts):
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    m = degree + 1
    P = np.zeros((n, m))
    for j in range(m):
        P[:, j] = chebyshev_norm(j, pts)
    return P





def padua_points(deg):
    if deg < 0:
        return np.zeros((0, 2)), np.array([])
    

    grid = np.cos(np.arange(deg + 1) * np.pi / deg) if deg > 0 else np.array([0.0])
    
    pts = []
    wts = []
    for i in range(deg + 1):
        for j in range(deg + 1):
            if (i + j) % 2 == 0:
                x = grid[i]
                y = grid[j]

                w = 1.0
                if i == 0 or i == deg:
                    w *= 0.5
                if j == 0 or j == deg:
                    w *= 0.5
                pts.append([x, y])
                wts.append(w)
    
    pts = np.array(pts)
    wts = np.array(wts)

    if np.sum(wts) > TINY:
        wts = wts / np.sum(wts) * 4.0
    return pts, wts





def bivariate_chebyshev_coeffs(deg, pts, fvals, weights):
    N = len(pts)
    fvals = np.asarray(fvals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    

    P = chebyshev_eval_matrix(deg, pts[:, 0])
    Q = chebyshev_eval_matrix(deg, pts[:, 1])
    

    G = np.diag(weights * fvals)
    


    C = np.zeros((deg + 1, deg + 1))
    for j in range(deg + 1):
        for l in range(deg + 1 - j):
            C[j, l] = np.sum(weights * fvals * P[:, j] * Q[:, l])
    

    esterr = 0.0
    for j in range(deg + 1):
        for l in range(deg + 1 - j):
            if j + l == deg:
                esterr += C[j, l] ** 2
    esterr = np.sqrt(esterr)
    
    return C, esterr


def bivariate_chebyshev_eval(deg, coeffs, x, y):
    x = np.clip(float(x), -1.0, 1.0)
    y = np.clip(float(y), -1.0, 1.0)
    
    P = chebyshev_eval_matrix(deg, np.array([x]))[0, :]
    Q = chebyshev_eval_matrix(deg, np.array([y]))[0, :]
    
    val = 0.0
    for j in range(deg + 1):
        for l in range(deg + 1 - j):
            val += coeffs[j, l] * P[j] * Q[l]
    return val





def continuum_background_model(m1, m2, params=None):
    if params is None:
        params = {"alpha": 1.5, "beta": 0.3, "norm": 1.0e-3}
    
    alpha = params["alpha"]
    beta = params["beta"]
    norm = params["norm"]
    
    if m1 <= 0.0 or m2 <= 0.0 or m1 + m2 > M_HIGGS:
        return 0.0
    
    val = norm * (m1 * m2) ** (-alpha) * np.exp(-beta * (m1 + m2) / M_Z)
    return float(val)


def build_background_interpolant(deg=8, m_range=(10.0, 120.0)):
    pts, wts = padua_points(deg)
    

    m_min, m_max = m_range
    scale = (m_max - m_min) / 2.0
    shift = (m_max + m_min) / 2.0
    phys_pts = pts * scale + shift
    
    fvals = []
    for p in phys_pts:
        fvals.append(continuum_background_model(p[0], p[1]))
    fvals = np.array(fvals)
    
    coeffs, esterr = bivariate_chebyshev_coeffs(deg, pts, fvals, wts)
    
    def eval_func(m1, m2):

        x1 = 2.0 * (m1 - m_min) / (m_max - m_min) - 1.0
        x2 = 2.0 * (m2 - m_min) / (m_max - m_min) - 1.0
        return bivariate_chebyshev_eval(deg, coeffs, x1, x2)
    
    return eval_func, coeffs, esterr





def s_b_ratio(m1, m2, signal_func, background_func):
    s = float(signal_func(m1, m2))
    b = float(background_func(m1, m2))
    if b < TINY:
        if s < TINY:
            return 0.0
        return 100.0
    return s / b
