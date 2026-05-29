"""
nonlinear_solver.py
===================
基于 807_nonlin_fixed_point 改造的非线性方程组求解器。

在浆态床气泡柱反应器中，气含率 α_g、液速 u_l 与压力 p 通过耦合的
代数方程组相互约束。本模块提供定点迭代与带阻尼的 Newton 法求解
该非线性系统。

核心公式
--------
1. 定点迭代：
       x^{(k+1)} = g(x^{(k)})
   收敛判据：|f(x)| < tol 或 |x^{(k+1)} - x^{(k)}| < tol

2. Newton 迭代（阻尼）：
       J(x^{(k)}) Δx = -f(x^{(k)})
       x^{(k+1)} = x^{(k)} + λ Δx,   λ ∈ (0,1]
   其中 Jacobian 元素：
       J_{ij} = ∂f_i / ∂x_j

3. 反应器耦合残差（气含率方程）：
       f_1(α_g) = α_g - u_g / (u_g + u_slip(α_g))
       f_2(u_l) = -∇p + μ_l ∇²u_l + M_{gl}(α_g, u_l) + ρ_l g
       f_3(p)   = ρ_m c_{pm} (∂T/∂t + u_l·∇T) - k_eff ∇²T - (-ΔH) r_FT

边界处理与鲁棒性
----------------
- 迭代次数上限：max_iter = 200
- 导数过小保护：|J_{ii}| < 1e-12 时触发正则化 J_{ii} += 1e-10
- 发散检测：|f| > 100 |f_0| 时自动降低阻尼因子 λ
- 气含率物理约束：0 < α_g < 1 - α_{g,min}（α_{g,min}=1e-6）
"""

import numpy as np


def fixed_point_iteration(g_func, x0, tol=1e-10, max_iter=200,
                          bounds=None, verbose=False):
    """
    定点迭代法求解 x = g(x)。

    Parameters
    ----------
    g_func : callable
        定点函数 g(x)，输入输出均为 ndarray。
    x0 : ndarray
        初始猜测。
    tol : float
        收敛容差。
    max_iter : int
        最大迭代次数。
    bounds : tuple(ndarray, ndarray) or None
        (lower, upper) 物理边界约束。
    verbose : bool
        是否打印迭代信息。

    Returns
    -------
    x : ndarray
        收敛解。
    residual : float
        最终残差范数。
    it : int
        实际迭代次数。
    converged : bool
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    residual_history = []

    for it in range(1, max_iter + 1):
        x_new = g_func(x)
        if bounds is not None:
            low, upp = bounds
            x_new = np.clip(x_new, low, upp)
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        residual_history.append(diff)
        x = x_new

        if diff < tol:
            if verbose:
                print(f"[FixedPoint] Converged in {it} iterations, diff={diff:.3e}")
            return x, diff, it, True

        # 发散检测：连续三次残差不降
        if it > 10 and len(residual_history) >= 4:
            if residual_history[-1] > residual_history[-2] > residual_history[-3] > residual_history[-4]:
                if verbose:
                    print(f"[FixedPoint] Divergence detected at iter {it}")
                return x, diff, it, False

    if verbose:
        print(f"[FixedPoint] Max iter reached, diff={diff:.3e}")
    return x, diff, max_iter, False


def newton_solver(f_func, j_func, x0, tol=1e-10, max_iter=100,
                  lambda_init=1.0, bounds=None, verbose=False):
    """
    带线搜索阻尼的 Newton 法求解 f(x)=0。

    Parameters
    ----------
    f_func : callable
        残差函数 f(x)。
    j_func : callable
        Jacobian 矩阵函数 J(x)。
    x0 : ndarray
        初始猜测。
    tol : float
        收敛容差。
    max_iter : int
        最大迭代次数。
    lambda_init : float
        初始步长因子。
    bounds : tuple or None
        物理边界。
    verbose : bool

    Returns
    -------
    x : ndarray
    fx_norm : float
    it : int
    converged : bool
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    fx = f_func(x)
    fx_norm0 = np.linalg.norm(fx, ord=np.inf)
    fx_norm = fx_norm0
    big = 100.0 * fx_norm0
    small = 1e-12

    for it in range(1, max_iter + 1):
        J = j_func(x)
        # 正则化：对角元素过小保护
        for i in range(n):
            if abs(J[i, i]) < small:
                J[i, i] += 1e-10

        try:
            dx = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            # 退化为最小二乘
            dx = np.linalg.lstsq(J, -fx, rcond=None)[0]

        # 线搜索：阻尼因子调整
        lam = lambda_init
        for _ in range(10):
            x_trial = x + lam * dx
            if bounds is not None:
                low, upp = bounds
                x_trial = np.clip(x_trial, low, upp)
            fx_trial = f_func(x_trial)
            fx_trial_norm = np.linalg.norm(fx_trial, ord=np.inf)
            if fx_trial_norm < fx_norm:
                x = x_trial
                fx = fx_trial
                fx_norm = fx_trial_norm
                break
            lam *= 0.5
        else:
            # 线搜索失败，接受最小步长并继续
            x = x + 0.1 * dx
            if bounds is not None:
                low, upp = bounds
                x = np.clip(x, low, upp)
            fx = f_func(x)
            fx_norm = np.linalg.norm(fx, ord=np.inf)

        if big < fx_norm:
            if verbose:
                print(f"[Newton] Divergence: |f| grew too large at iter {it}")
            return x, fx_norm, it, False

        if fx_norm < tol:
            if verbose:
                print(f"[Newton] Converged in {it} iterations, |f|={fx_norm:.3e}")
            return x, fx_norm, it, True

    if verbose:
        print(f"[Newton] Max iter reached, |f|={fx_norm:.3e}")
    return x, fx_norm, max_iter, False


def reactor_algebraic_residual(state, params):
    """
    气泡柱反应器稳态代数残差（二维简化模型）。

    state = [α_g, u_l]  (标量或向量)
    模型基于 Zuber-Findlay 漂移流理论与总体积通量守恒：
        f1 = u_g/α_g - C_0·j - u_∞            (气相速度-漂移流关系)
        f2 = α_g·u_g + (1-α_g)·u_l - j_in     (总体积通量守恒)
    其中 j = α_g·u_g + (1-α_g)·u_l 为总体积通量 [m/s]。

    Parameters
    ----------
    state : ndarray, shape (2,)
    params : dict
        包含 u_g_in, j_in 等

    Returns
    -------
    f : ndarray, shape (2,)
    """
    # TODO: 实现气泡柱反应器稳态代数残差计算
    # 基于 Zuber-Findlay 漂移流理论与总体积通量守恒
    # state = [alpha_g, u_l]
    # 需要计算 f1(气相速度-漂移流关系) 和 f2(总体积通量守恒)
    raise NotImplementedError("Hole 1: 请实现 reactor_algebraic_residual 的残差公式")


def reactor_jacobian(state, params):
    """
    reactor_algebraic_residual 的数值 Jacobian。
    """
    eps = 1e-8
    n = len(state)
    J = np.zeros((n, n))
    f0 = reactor_algebraic_residual(state, params)
    for j in range(n):
        state_pert = state.copy()
        state_pert[j] += eps
        f_pert = reactor_algebraic_residual(state_pert, params)
        J[:, j] = (f_pert - f0) / eps
    return J
