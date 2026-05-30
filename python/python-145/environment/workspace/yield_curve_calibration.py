
import numpy as np


def shepard_interp_2d(xd, yd, zd, p, xi, yi):
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    zd = np.asarray(zd, dtype=float)
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)

    nd = xd.shape[0]
    ni = xi.shape[0]

    if nd == 0:
        raise ValueError("shepard_interp_2d: 数据点不能为空")
    if xd.shape != yd.shape or xd.shape != zd.shape:
        raise ValueError("shepard_interp_2d: xd, yd, zd 形状必须一致")
    if xi.shape != yi.shape:
        raise ValueError("shepard_interp_2d: xi, yi 形状必须一致")

    zi = np.zeros(ni, dtype=float)

    for i in range(ni):
        if p == 0.0:
            w = np.ones(nd, dtype=float) / nd
        else:
            dx = xi[i] - xd
            dy = yi[i] - yd
            dist = np.sqrt(dx * dx + dy * dy)


            exact = np.where(dist < 1e-14)[0]
            if len(exact) > 0:
                zi[i] = zd[exact[0]]
                continue

            w = 1.0 / (dist ** p)
            s = np.sum(w)
            if s < 1e-30:
                w = np.ones(nd, dtype=float) / nd
            else:
                w = w / s

        zi[i] = np.dot(w, zd)

    return zi


def horner_eval(c, x):
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    m = c.shape[0] - 1

    if m < 0:
        return np.zeros_like(x)

    p = np.full_like(x, c[m], dtype=float)
    for i in range(m - 1, -1, -1):
        p = p * x + c[i]
    return p


def fit_yield_polynomial(maturities, yields_, degree=5):
    maturities = np.asarray(maturities, dtype=float)
    yields_ = np.asarray(yields_, dtype=float)

    if maturities.shape != yields_.shape:
        raise ValueError("fit_yield_polynomial: maturities 与 yields_ 形状必须一致")
    if len(maturities) <= degree:
        raise ValueError("fit_yield_polynomial: 数据点数量必须大于多项式次数")


    T_max = np.max(maturities)
    if T_max < 1e-14:
        T_max = 1.0
    t_norm = maturities / T_max

    V = np.vander(t_norm, degree + 1, increasing=True)
    c, residuals, rank, s = np.linalg.lstsq(V, yields_, rcond=None)
    residual = np.linalg.norm(V @ c - yields_)
    cond_num = np.linalg.cond(V)


    c_scaled = c.copy()
    for k in range(degree + 1):
        c_scaled[k] = c[k] / (T_max ** k)

    return c_scaled, residual, cond_num


def extract_curve_features(maturities, yields_):
    maturities = np.asarray(maturities, dtype=float)
    yields_ = np.asarray(yields_, dtype=float)

    if len(maturities) < 3:
        return {
            'start': (maturities[0], yields_[0]) if len(maturities) > 0 else None,
            'end': (maturities[-1], yields_[-1]) if len(maturities) > 0 else None,
            'peaks': [],
            'valleys': [],
            'inflection': []
        }


    dy = np.diff(yields_)
    d2y = np.diff(dy)

    peaks = []
    valleys = []
    for i in range(1, len(yields_) - 1):
        if yields_[i] > yields_[i - 1] and yields_[i] > yields_[i + 1]:
            peaks.append((maturities[i], yields_[i]))
        elif yields_[i] < yields_[i - 1] and yields_[i] < yields_[i + 1]:
            valleys.append((maturities[i], yields_[i]))

    inflection = []
    for i in range(len(d2y) - 1):
        if d2y[i] * d2y[i + 1] < 0:
            idx = i + 1
            if 0 < idx < len(maturities):
                inflection.append((maturities[idx], yields_[idx]))

    return {
        'start': (maturities[0], yields_[0]),
        'end': (maturities[-1], yields_[-1]),
        'peaks': peaks,
        'valleys': valleys,
        'inflection': inflection
    }


def calibrate_yield_curve(market_maturities, market_yields,
                          interp_method='shepard', poly_degree=5):
    market_maturities = np.asarray(market_maturities, dtype=float)
    market_yields = np.asarray(market_yields, dtype=float)

    features = extract_curve_features(market_maturities, market_yields)
    c, residual, cond_num = fit_yield_polynomial(market_maturities, market_yields, poly_degree)

    def yield_func(T):
        T = np.asarray(T, dtype=float)

        return horner_eval(c, T)

    def yield_func_shepard(T):

        T_arr = np.asarray(T, dtype=float)
        return shepard_interp_2d(market_maturities, np.zeros_like(market_maturities),
                                  market_yields, 2.0, T_arr, np.zeros_like(T_arr))

    result = {
        'coefficients': c,
        'residual': residual,
        'condition_number': cond_num,
        'features': features,
        'yield_function': yield_func,
        'yield_function_shepard': yield_func_shepard,
        'method': interp_method
    }
    return result
