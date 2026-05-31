"""
variational_spectrum.py
=======================
变分法提取强子能谱：广义本征值问题、Muller 求根、样条插值与 Hooke-Jeeves 优化。

原项目映射：
  - 209_conte_deboor：Muller 复根法、样条插值（calccf）
  - 1266_toms178：Hooke-Jeeves 直接搜索优化

物理背景
--------
在格点 QCD 中，为了同时提取基态与激发态质量，使用变分方法（Generalized Eigenvalue Problem, GEVP）：

    C(t) v_n = λ_n(t, t_0) C(t_0) v_n

其中 C_{ij}(t) = ⟨ O_i(t) O_j†(0) ⟩ 为 N×N 关联矩阵，
v_n 为最优算符组合系数，λ_n 满足：

    λ_n(t, t_0) = A_n exp( -m_n (t - t_0) ) + O(exp( -m_{n+1} (t - t_0) ))

有效质量由本征值给出：

    m_n^{eff}(t) = log( λ_n(t) / λ_n(t+1) )

为了精确定位关联矩阵的零点（或特征多项式的根），
采用 Muller 方法求解：

    det[ C(t) - λ C(t_0) ] = 0

同时，对关联函数数据进行样条平滑，并用 Hooke-Jeeves 优化
寻找最佳 smearing 参数，使基态信号最大化。

核心公式
--------
1. 样条插值（Hermite cubic spline）：
   给定节点 {x_i} 和函数值 f(x_i), f'(x_i)，构造分段三次多项式：

   S_i(x) = c_{1i} + (x-x_i)[c_{2i} + (x-x_i)(c_{3i} + (x-x_i)c_{4i})]

   系数由 divided differences 确定（calccf 算法）。

2. Muller 求根：
   给定三个近似根 z0, z1, z2，构造通过 (z_k, f(z_k)) 的二次插值，
   取其靠近 z2 的根作为下一次迭代。

3. Hooke-Jeeves 模式搜索：
   沿坐标轴方向进行探测移动（exploratory move），
   成功后沿该方向加速（pattern move），步长按因子 ρ 收缩。
"""

import numpy as np


def calccf(xi: np.ndarray, c: np.ndarray) -> tuple:
    """
    计算 Hermite 三次样条的系数（Conte-de Boor 算法）。

    给定节点 XI 和端点值 C[0,:]、导数值 C[1,:]，
    构造满足 S(XI_j) = C[0,j], S'(XI_j) = C[1,j] 的分段三次样条。

    系数计算：
        DX = diff(XI)
        DIVDF1 = diff(C[0,:]) / DX
        DIVDF3 = C[1,0:N] - 2 DIVDF1 + C[1,1:N+1]
        c[3,:] = (DIVDF1 - c[2,:] - DIVDF3) / DX
        c[4,:] = DIVDF3 / (DX * DX)

    Parameters
    ----------
    xi : np.ndarray
        节点，形状 (N+1,)。
    c : np.ndarray
        2×(N+1) 数组，第一行为函数值，第二行为导数值。

    Returns
    -------
    breaks : np.ndarray
        节点。
    coefs : np.ndarray
        4×N 系数矩阵。
    """
    n = len(xi) - 1
    dx = np.diff(xi)
    divdf1 = np.diff(c[0, :]) / dx
    divdf3 = c[1, 0:n] - 2.0 * divdf1 + c[1, 1:n + 1]
    coefs = np.zeros((4, n))
    coefs[0, :] = c[0, 0:n]
    coefs[1, :] = c[1, 0:n]
    coefs[2, :] = (divdf1 - coefs[1, :] - divdf3) / dx
    coefs[3, :] = divdf3 / (dx * dx)
    return xi, coefs


def spline_eval(breaks: np.ndarray, coefs: np.ndarray, x: float) -> float:
    """求值分段三次样条。"""
    if x <= breaks[0]:
        i = 0
    elif x >= breaks[-1]:
        i = len(breaks) - 2
    else:
        i = np.searchsorted(breaks, x) - 1
        i = max(0, min(i, len(breaks) - 2))
    dx = x - breaks[i]
    return coefs[0, i] + dx * (coefs[1, i] + dx * (coefs[2, i] + dx * coefs[3, i]))


def muller_method(f, z0: complex, z1: complex, z2: complex,
                  eps1: float = 1e-12, eps2: float = 1e-20,
                  maxit: int = 50) -> complex:
    """
    Muller 方法求复函数零点。

    算法：
        给定三点 (z0, f0), (z1, f1), (z2, f2)，构造二次插值：
            q(x) = a (x - z2)^2 + b (x - z2) + c
        取离 z2 最近的根：
            z_new = z2 - 2c / (b ± sqrt(b^2 - 4ac))
        其中符号选择使分母模更大。

    Parameters
    ----------
    f : callable
        复变函数。
    z0, z1, z2 : complex
        初始三个近似值。
    eps1, eps2 : float
        收敛判据。
    maxit : int
        最大迭代次数。

    Returns
    -------
    root : complex
        近似根。
    """
    eps1 = max(eps1, 1e-12)
    eps2 = max(eps2, 1e-20)
    z = [z0, z1, z2]
    fz = [f(zz) for zz in z]

    for _ in range(maxit):
        # 构造通过最近三点的二次多项式
        h0 = z[-2] - z[-3]
        h1 = z[-1] - z[-2]
        d0 = (fz[-2] - fz[-3]) / h0
        d1 = (fz[-1] - fz[-2]) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = fz[-1]

        disc = np.sqrt(b * b - 4.0 * a * c)
        if abs(b + disc) > abs(b - disc):
            denom = b + disc
        else:
            denom = b - disc
        if abs(denom) < 1e-30:
            denom = 1.0

        dz = -2.0 * c / denom
        z_new = z[-1] + dz
        fz_new = f(z_new)

        z.append(z_new)
        fz.append(fz_new)

        if abs(dz) < eps1 * max(abs(z_new), 1.0):
            break
        if max(abs(fz_new), abs(fz[-2])) <= eps2:
            break

    return z[-1]


def gevp_solve(c_t: np.ndarray, c_t0: np.ndarray) -> tuple:
    """
    求解广义本征值问题 C(t) v = λ C(t_0) v。

    通过同时 Cholesky 分解 C(t_0) = L L^† 转换为标准本征值问题。

    Parameters
    ----------
    c_t : np.ndarray
        时刻 t 的关联矩阵。
    c_t0 : np.ndarray
        参考时刻 t_0 的关联矩阵。

    Returns
    -------
    lambdas : np.ndarray
        本征值（从大到小排序）。
    vectors : np.ndarray
        本征向量矩阵（每列一个向量）。
    """
    # 确保厄米性
    c_t0 = 0.5 * (c_t0 + c_t0.T.conj())
    c_t = 0.5 * (c_t + c_t.T.conj())

    # 正则化
    eps = 1e-10
    c_t0 += eps * np.eye(c_t0.shape[0])

    # 使用 numpy 的广义本征值求解
    lambdas, vectors = np.linalg.eig(np.linalg.solve(c_t0, c_t))
    # 取实部并排序
    lambdas = lambdas.real
    idx = np.argsort(-lambdas)
    lambdas = lambdas[idx]
    vectors = vectors[:, idx]
    return lambdas, vectors


def hooke_jeeves(f, x0: np.ndarray, rho: float = 0.85,
                 eps: float = 1e-7, itermax: int = 500) -> tuple:
    """
    Hooke-Jeeves 直接搜索法求多变量函数极小值。

    算法：
        1. 探测移动：沿各坐标轴方向以步长 delta 搜索更优解；
        2. 模式移动：若探测成功，沿新方向加速；
        3. 若失败，缩小步长 delta ← rho * delta。

    Parameters
    ----------
    f : callable
        目标函数 f(x) → scalar。
    x0 : np.ndarray
        初始点。
    rho : float
        步长收缩因子 (0 < rho < 1)。
    eps : float
        收敛阈值。
    itermax : int
        最大迭代次数。

    Returns
    -------
    xbest : np.ndarray
        最优解。
    fbest : float
        最优函数值。
    """
    nvars = len(x0)
    xbefore = x0.copy().astype(float)
    delta = np.where(xbefore == 0.0, rho, rho * np.abs(xbefore))
    steplength = rho
    iters = 0
    fbefore = f(xbefore)

    def best_nearby(delta_loc, xbase, fbase):
        x = xbase.copy()
        fmin = fbase
        for i in range(nvars):
            for sign in [1.0, -1.0]:
                x[i] += sign * delta_loc[i]
                fn = f(x)
                if fn < fmin:
                    fmin = fn
                else:
                    x[i] -= sign * delta_loc[i]
        return fmin, x

    while iters < itermax and steplength > eps:
        iters += 1
        newf, newx = best_nearby(delta, xbefore, fbefore)

        keep = True
        while newf < fbefore and keep:
            for i in range(nvars):
                if newx[i] <= xbefore[i]:
                    delta[i] = -abs(delta[i])
                else:
                    delta[i] = abs(delta[i])
                tmp = xbefore[i]
                xbefore[i] = newx[i]
                newx[i] = newx[i] + newx[i] - tmp
            fbefore = newf
            newf, newx = best_nearby(delta, newx, fbefore)
            if fbefore <= newf:
                break
            keep = False
            for i in range(nvars):
                if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                    keep = True
                    break

        if steplength >= eps and fbefore <= newf:
            steplength *= rho
            delta *= rho

    return xbefore, fbefore


def variational_masses(correlator_matrix: np.ndarray, t0: int = 2) -> dict:
    """
    对关联矩阵进行变分分析，提取强子能级。

    Parameters
    ----------
    correlator_matrix : np.ndarray
        形状 (nt, Nop, Nop) 的关联矩阵 C_{ij}(t)。
    t0 : int
        参考时间切片。

    Returns
    -------
    results : dict
        包含本征值、有效质量、最优算符组合。
    """
    nt, nop, _ = correlator_matrix.shape
    lambdas = np.zeros((nt, nop))
    vectors = np.zeros((nt, nop, nop))
    masses = np.zeros((nt - t0 - 1, nop))

    c_t0 = correlator_matrix[t0].real
    for t in range(t0, nt):
        lam, vec = gevp_solve(correlator_matrix[t].real, c_t0)
        lambdas[t] = lam
        vectors[t] = vec.real

    for n in range(nop):
        for t in range(t0, nt - 1):
            if abs(lambdas[t, n]) > 1e-15 and lambdas[t, n] / lambdas[t + 1, n] > 0:
                masses[t - t0, n] = np.log(lambdas[t, n] / lambdas[t + 1, n])
            else:
                masses[t - t0, n] = np.nan

    return {
        "eigenvalues": lambdas,
        "eigenvectors": vectors,
        "masses": masses,
        "t0": t0,
    }


def optimize_smearing_parameter(correlator_func, param_bounds: tuple,
                                t0: int = 2) -> tuple:
    """
    使用 Hooke-Jeeves 优化 smearing 参数，使变分基态质量最稳定。

    目标函数：有效质量 plateau 的方差（越小越好）。
    """
    def objective(alpha):
        corr = correlator_func(alpha[0])
        nt = len(corr)
        if nt <= t0 + 3:
            return 1e6
        # 简化：直接计算单通道有效质量的标准差
        m_eff = np.zeros(nt - 1)
        for t in range(nt - 1):
            if corr[t] > 1e-15 and corr[t + 1] > 1e-15:
                m_eff[t] = np.log(corr[t] / corr[t + 1])
        # 取中间 plateau 段的标准差
        plateau = m_eff[t0 + 1:nt // 2]
        if len(plateau) < 2:
            return 1e6
        return np.nanvar(plateau)

    x0 = np.array([(param_bounds[0] + param_bounds[1]) / 2.0])
    xbest, fbest = hooke_jeeves(objective, x0, rho=0.75, eps=1e-4, itermax=100)
    # 限制在边界内
    xbest[0] = np.clip(xbest[0], param_bounds[0], param_bounds[1])
    return xbest[0], fbest
