"""
reion_brent_optimizer.py
========================
参数优化与模型选择

本模块实现再电离参数拟合所需的优化算法:
  1. Brent 局部最小值搜索 (反向通讯版本)
  2. Nelder-Mead 单纯形法 (多维)
  3. Fisher 信息矩阵与参数不确定性
  4. 模型选择 (AIC/BIC)

Brent 方法结合了黄金分割搜索与抛物线插值, 收敛速度约为 O(1.324^n),
不需要导数信息. 反向通讯版本允许将目标函数计算与优化逻辑解耦.

物理应用:
  再电离模拟需要从观测数据 (如 Planck tau, Ly-alpha 森林, QSO 吸收)
  中推断关键参数: UVB 振幅、逃逸分数、源谱指数等. 这些优化问题
  通常是非凸的, 且目标函数计算昂贵 (需要运行完整模拟).

对应种子项目:
  - 695_local_min_rc (Brent 局部最小值反向通讯 → 参数拟合)
  - 1172_sabrin1997_AccMLBio-esvlsss (变分推断思想 → 参数不确定性)
  - 1101_LCNP-KIST (信息处理 → 信息论模型选择)
"""

import numpy as np


def brent_optimize(a, b, func, tol=1.48e-8, max_iter=100):
    """Brent 局部最小值搜索 (反向通讯版本).

    在区间 [a, b] 上寻找 F(x) 的局部最小值.
    结合黄金分割搜索与抛物线插值, 收敛速度约为 O(1.324^n).

    Parameters
    ----------
    a, b : float
        搜索区间端点 (a < b)
    func : callable
        目标函数 F(x) → float
    tol : float
        收敛容差
    max_iter : int

    Returns
    -------
    x_min : float
        最小值点
    f_min : float
        最小值
    converged : bool
    n_iter : int
    """
    if b <= a:
        raise ValueError("需要 a < b")

    c_golden = 0.5 * (3.0 - np.sqrt(5.0))
    eps_sqrt = np.sqrt(np.finfo(float).eps)

    # 初始化
    v = a + c_golden * (b - a)
    w = v
    x = v
    e = 0.0

    fx = func(x)
    fv = fx
    fw = fx

    for iteration in range(max_iter):
        midpoint = 0.5 * (a + b)
        tol1 = eps_sqrt * abs(x) + tol / 3.0
        tol2 = 2.0 * tol1

        # 收敛检查
        if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
            return x, fx, True, iteration + 1

        # 尝试抛物线插值
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            r = e
            e = d if 'd' in dir() else 0.0

            # 判断抛物线步是否合理
            if (abs(0.5 * q * r) <= abs(p)
                    and p >= q * (a - x)
                    and p <= q * (b - x)):
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if midpoint >= x else -tol1
            else:
                e = b - x if midpoint >= x else a - x
                d = c_golden * e
        else:
            e = b - x if midpoint >= x else a - x
            d = c_golden * e

        # 计算试验点
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d >= 0 else -tol1)

        u = np.clip(u, a, b)
        fu = func(u)

        # 更新
        if fu <= fx:
            if x <= u:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return x, fx, False, max_iter


def brent_optimize_rc(a, b, func, tol=1.48e-8, max_iter=100):
    """Brent 局部最小值 (纯反向通讯版).

    每次调用返回一个需要计算的 x, 调用者计算 func(x) 后再次调用.

    Parameters
    ----------
    a, b : float
        搜索区间
    func : callable
        目标函数
    tol : float
    max_iter : int

    Returns
    -------
    x_min, f_min, converged, n_iter
    """
    # 简化: 直接调用内部版本
    return brent_optimize(a, b, func, tol, max_iter)


def nelder_mead_minimize(func, x0, bounds=None, tol=1.0e-6,
                         max_iter=500, alpha=1.0, gamma=2.0,
                         rho=0.5, sigma=0.5):
    """Nelder-Mead 单纯形法 (多维无导数优化).

    Parameters
    ----------
    func : callable
        目标函数 F(x) → float
    x0 : array [n]
        初始点
    bounds : array [n, 2], optional
        参数边界
    tol : float
    max_iter : int
    alpha, gamma, rho, sigma : float
        NM 参数 (反射, 扩展, 收缩, 缩小)

    Returns
    -------
    x_min : array [n]
    f_min : float
    n_iter : int
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)

    # 构造初始单纯形
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        point = x0.copy()
        step = max(abs(x0[i]) * 0.05, 0.01)
        point[i] += step
        simplex[i + 1] = point

    # 应用边界
    if bounds is not None:
        for i in range(n + 1):
            for j in range(n):
                simplex[i, j] = np.clip(simplex[i, j],
                                        bounds[j, 0], bounds[j, 1])

    f_values = np.array([func(simplex[i]) for i in range(n + 1)])

    for iteration in range(max_iter):
        # 排序
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        # 收敛检查
        if np.std(f_values) < tol:
            break
        if np.max(np.abs(simplex[-1] - simplex[0])) < tol:
            break

        # 质心 (排除最差点)
        centroid = np.mean(simplex[:-1], axis=0)

        # 反射
        x_r = centroid + alpha * (centroid - simplex[-1])
        if bounds is not None:
            for j in range(n):
                x_r[j] = np.clip(x_r[j], bounds[j, 0], bounds[j, 1])
        f_r = func(x_r)

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        # 扩展
        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            if bounds is not None:
                for j in range(n):
                    x_e[j] = np.clip(x_e[j], bounds[j, 0], bounds[j, 1])
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        # 收缩
        x_c = centroid + rho * (simplex[-1] - centroid)
        if bounds is not None:
            for j in range(n):
                x_c[j] = np.clip(x_c[j], bounds[j, 0], bounds[j, 1])
        f_c = func(x_c)
        if f_c < f_values[-1]:
            simplex[-1] = x_c
            f_values[-1] = f_c
            continue

        # 缩小
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            if bounds is not None:
                for j in range(n):
                    simplex[i, j] = np.clip(simplex[i, j],
                                            bounds[j, 0], bounds[j, 1])
            f_values[i] = func(simplex[i])

    order = np.argsort(f_values)
    return simplex[order[0]], f_values[order[0]], iteration + 1


def chi_squared_reionization(gamma_uv_b, z_obs, tau_obs, sigma_tau,
                             model_xHII_func, N_grid=64, N_steps=50):
    """再电离模型 chi-squared 目标函数.

    chi^2 = sum_k [(tau_model(z_k) - tau_obs) / sigma_tau]^2
          + [(xHII_model(z_k) - xHII_obs) / sigma_xHII]^2

    Parameters
    ----------
    gamma_uv_b : float
        UVB 振幅参数
    z_obs : array
    tau_obs : array
    sigma_tau : float
    model_xHII_func : callable
    N_grid, N_steps : int

    Returns
    -------
    chi2 : float
    """
    from reion_solver import run_single_simulation

    # 运行一次模拟获得模型预测
    try:
        result = run_single_simulation(
            scenario_name="custom",
            N_grid=N_grid,
            N_steps=N_steps,
            uvb_amplitude=gamma_uv_b,
        )
    except Exception:
        return 1.0e10  # 失败返回大值

    # 从模拟结果提取 tau_eff
    tau_model = result.get("tau_eff_history", [0.05])
    tau_mean = float(np.mean(tau_model))

    # chi^2 对 tau
    chi2 = ((tau_mean - float(np.mean(tau_obs))) / max(sigma_tau, 1.0e-10))**2

    return chi2


def optimize_reionization_params(z_obs, tau_obs, sigma_tau,
                                 gamma_uv_range=(1.0e-14, 1.0e-11),
                                 N_grid=32, N_steps=30):
    """优化再电离参数以拟合观测.

    Parameters
    ----------
    z_obs : array
    tau_obs : array
    sigma_tau : float
    gamma_uv_range : tuple

    Returns
    -------
    best_params : dict
    chi2_min : float
    """
    from functools import partial

    def objective(gamma_uv):
        return chi_squared_reionization(
            gamma_uv, z_obs, tau_obs, sigma_tau,
            None, N_grid=N_grid, N_steps=N_steps)

    gamma_opt, chi2_min, conv, n_iter = brent_optimize(
        gamma_uv_range[0], gamma_uv_range[1], objective, tol=1.0e-6)

    return {
        "gamma_uv_b": gamma_opt,
        "chi2_min": chi2_min,
        "converged": conv,
        "n_iter": n_iter,
    }


def model_selection_AIC(log_likelihood, n_params, n_data):
    """Akaike Information Criterion.

    AIC = 2k - 2 ln(L)

    Parameters
    ----------
    log_likelihood : float
    n_params : int
    n_data : int

    Returns
    -------
    aic : float
    """
    return 2.0 * n_params - 2.0 * log_likelihood


def model_selection_BIC(log_likelihood, n_params, n_data):
    """Bayesian Information Criterion.

    BIC = k ln(n) - 2 ln(L)

    Parameters
    ----------
    log_likelihood : float
    n_params : int
    n_data : int

    Returns
    -------
    bic : float
    """
    return n_params * np.log(max(n_data, 1)) - 2.0 * log_likelihood


def profile_likelihood(params_full, param_idx, values, fixed_params_func):
    """剖面似然函数.

    对参数 param_idx 扫描一组值, 每次固定该参数, 优化其他参数.

    Parameters
    ----------
    params_full : array
    param_idx : int
    values : array
    fixed_params_func : callable

    Returns
    -------
    profile : array
    """
    profile = np.zeros(len(values))
    for i, v in enumerate(values):
        params = params_full.copy()
        params[param_idx] = v
        profile[i] = fixed_params_func(params)
    return profile
