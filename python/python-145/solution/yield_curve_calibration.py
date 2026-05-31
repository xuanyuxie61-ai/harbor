"""
yield_curve_calibration.py
==========================
博士级收益率曲线校准：Shepard 插值、Horner 多项式求值与边界提取

本模块实现将不规则市场交易数据（不同期限的非均匀报价）插值与拟合为
连续光滑的收益率曲线函数：

  1. Shepard 二维插值：对不规则空间分布的收益率数据进行距离加权插值。
     权重公式:
         w_j = 1 / ||x - x_j||^p
         若 x = x_j，则 w_j = δ_{j,k}
     插值:
         f(x) = Σ_j w_j z_j / Σ_j w_j

  2. Horner 多项式求值：快速计算收益率曲线的多项式近似。
     对于 p(x) = c0 + c1 x + c2 x^2 + ... + cm x^m
     Horner 格式: p(x) = c0 + x*(c1 + x*(c2 + ... + x*cm)...)
     计算复杂度从 O(m^2) 降至 O(m)。

  3. 边界轮廓提取：从收益率曲线数据中提取关键形状特征点
     （基于 human_data 的边界点选取思想）。

数学背景
--------
收益率曲线 y(T) 表示在时刻 t 期限为 T 的零息债券收益率。
市场报价通常为离散点 {(T_i, y_i)}_{i=1}^{N_d}。
需要构造光滑延拓 ŷ(T) 使得:
    ŷ(T_i) ≈ y_i  （拟合）
    ŷ'(T) >= 0    （单调性约束，通常要求远期利率非负）
    ŷ''(T) 有界   （曲率约束）
"""

import numpy as np


def shepard_interp_2d(xd, yd, zd, p, xi, yi):
    """
    二维 Shepard 插值（逆距离加权）。

    公式:
        w_j = ||(x,y) - (x_j, y_j)||^{-p}
        z(x,y) = Σ_j w_j z_j / Σ_j w_j

    当插值点与数据点重合时，直接返回该数据点值。

    Parameters
    ----------
    xd : np.ndarray, shape (nd,)
        数据点 x 坐标。
    yd : np.ndarray, shape (nd,)
        数据点 y 坐标。
    zd : np.ndarray, shape (nd,)
        数据点值。
    p : float
        幂指数，p=0 时等权平均；通常 p=2。
    xi : np.ndarray, shape (ni,)
        插值点 x 坐标。
    yi : np.ndarray, shape (ni,)
        插值点 y 坐标。

    Returns
    -------
    zi : np.ndarray, shape (ni,)
        插值结果。
    """
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

            # 检查是否重合
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
    """
    Horner 法求多项式值。

    多项式:
        p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[m]*x^m

    Horner 格式:
        p(x) = c[0] + x*(c[1] + x*(c[2] + ... + x*c[m])...)

    Parameters
    ----------
    c : np.ndarray, shape (m+1,)
        系数数组，c[k] 为 x^k 的系数。
    x : float or np.ndarray
        求值点。

    Returns
    -------
    float or np.ndarray
        多项式值。
    """
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
    """
    使用最小二乘法拟合收益率曲线的多项式近似。

    目标泛函:
        min_c Σ_i (Σ_{k=0}^{degree} c_k T_i^k - y_i)^2

    Parameters
    ----------
    maturities : np.ndarray
        期限数据 T_i。
    yields_ : np.ndarray
        收益率数据 y_i。
    degree : int
        多项式次数。

    Returns
    -------
    c : np.ndarray
        多项式系数（升幂）。
    residual : float
        残差范数。
    cond_num : float
        Vandermonde 矩阵条件数。
    """
    maturities = np.asarray(maturities, dtype=float)
    yields_ = np.asarray(yields_, dtype=float)

    if maturities.shape != yields_.shape:
        raise ValueError("fit_yield_polynomial: maturities 与 yields_ 形状必须一致")
    if len(maturities) <= degree:
        raise ValueError("fit_yield_polynomial: 数据点数量必须大于多项式次数")

    # 构造 Vandermonde 矩阵并归一化期限以改善条件数
    T_max = np.max(maturities)
    if T_max < 1e-14:
        T_max = 1.0
    t_norm = maturities / T_max

    V = np.vander(t_norm, degree + 1, increasing=True)
    c, residuals, rank, s = np.linalg.lstsq(V, yields_, rcond=None)
    residual = np.linalg.norm(V @ c - yields_)
    cond_num = np.linalg.cond(V)

    # 将系数转换回原始尺度
    c_scaled = c.copy()
    for k in range(degree + 1):
        c_scaled[k] = c[k] / (T_max ** k)

    return c_scaled, residual, cond_num


def extract_curve_features(maturities, yields_):
    """
    从收益率曲线数据中提取关键形状特征点（边界轮廓思想）。

    特征:
      1. 起始点 (T_min, y_min)
      2. 终止点 (T_max, y_max)
      3. 局部极大值点（峰值）
      4. 局部极小值点（谷值）
      5. 拐点（二阶差分变号）

    Parameters
    ----------
    maturities : np.ndarray
        期限数组（已排序）。
    yields_ : np.ndarray
        收益率数组。

    Returns
    -------
    features : dict
        包含 'start', 'end', 'peaks', 'valleys', 'inflection' 的字典。
    """
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

    # 一阶、二阶差分
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
    """
    综合校准流程：将市场数据校准为连续的收益率曲线函数。

    步骤:
      1. 特征提取
      2. 多项式最小二乘拟合
      3. Shepard 插值验证（用于交叉检验）

    Parameters
    ----------
    market_maturities : np.ndarray
        市场报价期限。
    market_yields : np.ndarray
        市场报价收益率。
    interp_method : str
        'shepard' 或 'polynomial'。
    poly_degree : int
        多项式拟合次数。

    Returns
    -------
    result : dict
        包含拟合系数、残差、特征点、插值函数的字典。
    """
    market_maturities = np.asarray(market_maturities, dtype=float)
    market_yields = np.asarray(market_yields, dtype=float)

    features = extract_curve_features(market_maturities, market_yields)
    c, residual, cond_num = fit_yield_polynomial(market_maturities, market_yields, poly_degree)

    def yield_func(T):
        T = np.asarray(T, dtype=float)
        # 使用多项式拟合
        return horner_eval(c, T)

    def yield_func_shepard(T):
        # 将一维期限映射到二维 (T, 0) 进行 Shepard 插值
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
