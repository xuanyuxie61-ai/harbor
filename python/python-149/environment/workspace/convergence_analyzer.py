"""
convergence_analyzer.py
收敛性与稳定性分析工具

融合种子项目:
  - 1063_sde (stab_meansquare): 均方稳定性分析

科学背景:
  对于数值求解SDE的算法，需要严格分析其收敛阶与稳定性:

  1. 强收敛阶 p_s:
      E[ |X_T - X^h_T| ] ≤ C h^{p_s}

  2. 弱收敛阶 p_w:
      | E[φ(X_T)] - E[φ(X^h_T)] | ≤ C h^{p_w}

  3. 均方稳定性:
      对线性测试方程 dX = λX dt + μX dW,
      数值方法均方稳定当:
          lim_{n→∞} E[|X_n|^2] = 0

  对于神经SDE，还需分析控制系统的指数稳定性:
      若 Lyapunov函数 V(x) 满足:
          E[dV] ≤ -α V dt + β dt
      则系统指数均方稳定。
"""

import numpy as np
from typing import Callable, Optional, Tuple, List


def analyze_ms_stability_region(
    lambda_range: Tuple[float, float],
    mu_range: Tuple[float, float],
    dt: float,
    n_lambda: int = 50,
    n_mu: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    绘制Euler-Maruyama方法的均方稳定性区域。

    稳定条件:
        (1 + λΔt)^2 + μ^2 Δt < 1

    Parameters
    ----------
    lambda_range : (lmin, lmax)
    mu_range : (mumin, mumax)
    dt : float
    n_lambda, n_mu : int
        网格分辨率

    Returns
    -------
    L, MU : ndarray
        网格坐标
    stable_mask : ndarray
        布尔掩码，True表示稳定
    """
    lvals = np.linspace(lambda_range[0], lambda_range[1], n_lambda)
    muvals = np.linspace(mu_range[0], mu_range[1], n_mu)
    L, MU = np.meshgrid(lvals, muvals)

    stable_mask = (1.0 + L * dt) ** 2 + (MU ** 2) * dt < 1.0
    return L, MU, stable_mask


def estimate_convergence_rate(
    h_vals: np.ndarray,
    err_vals: np.ndarray,
) -> Tuple[float, float, float]:
    """
    通过最小二乘拟合估计收敛阶:

        err = C · h^p
        log(err) = log(C) + p · log(h)

    Returns
    -------
    p : float
        估计收敛阶
    logC : float
        log(C)
    residual : float
        拟合残差范数
    """
    # 过滤非正数据
    valid = (h_vals > 0) & (err_vals > 0) & np.isfinite(h_vals) & np.isfinite(err_vals)
    if np.sum(valid) < 2:
        return 0.0, 0.0, np.inf

    log_h = np.log(h_vals[valid])
    log_e = np.log(err_vals[valid])

    A = np.vstack([np.ones(len(log_h)), log_h]).T
    sol, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)

    logC = sol[0]
    p = sol[1]
    residual = float(np.linalg.norm(A @ sol - log_e))
    return float(p), float(logC), residual


def lyapunov_exponential_decay_rate(
    t: np.ndarray,
    y: np.ndarray,
    norm_order: int = 2,
) -> float:
    """
    估计轨迹的指数衰减速率:

        ||y(t)|| ≈ C exp(-λ t)
        log(||y||) ≈ log(C) - λ t

    Returns
    -------
    lambda_est : float
        估计的指数衰减速率（正数表示稳定）
    """
    norms = np.linalg.norm(y, ord=norm_order, axis=1)
    valid = (norms > 1e-12) & np.isfinite(norms)
    if np.sum(valid) < 2:
        return 0.0

    log_norms = np.log(norms[valid])
    t_valid = t[valid]

    # 线性回归
    A = np.vstack([np.ones(len(t_valid)), t_valid]).T
    sol, _, _, _ = np.linalg.lstsq(A, log_norms, rcond=None)
    lambda_est = -sol[1]
    return float(lambda_est)


def compute_maximum_lyapunov_exponent(
    f: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tspan: Tuple[float, float],
    n_steps: int = 1000,
    n_perturbations: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    通过有限差分近似最大Lyapunov指数:

        λ_max ≈ lim_{t→∞} (1/t) ln( ||δx(t)|| / ||δx(0)|| )

    其中 δx(t) 满足变分方程:
        d(δx)/dt = J(x(t)) δx(t)

    Parameters
    ----------
    f : callable
        向量场
    jacobian_fn : callable
        Jacobian矩阵
    x0 : ndarray
        参考轨迹初始点
    n_perturbations : int
        扰动方向数

    Returns
    -------
    lambda_max : float
        最大Lyapunov指数估计
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps

    # 参考轨迹（Euler积分）
    x_ref = x0.copy()
    lyapunov_estimates = []

    for _ in range(n_perturbations):
        # 随机小扰动
        delta = rng.normal(0, 1e-8, len(x0))
        delta_norm0 = np.linalg.norm(delta)
        if delta_norm0 < 1e-15:
            continue

        x_pert = x_ref + delta

        for step in range(n_steps):
            # 参考步进
            x_ref = x_ref + dt * f(x_ref)
            # 扰动步进
            x_pert = x_pert + dt * f(x_pert)

            # 重正化
            delta = x_pert - x_ref
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-15:
                break
            ratio = delta_norm / delta_norm0
            lyapunov_estimates.append(np.log(ratio) / (t0 + (step + 1) * dt))
            delta = delta / delta_norm * delta_norm0
            x_pert = x_ref + delta

    if len(lyapunov_estimates) == 0:
        return 0.0
    return float(np.mean(lyapunov_estimates))


def perform_stability_sweep(
    integrator: Callable,
    lambda_list: List[float],
    mu_list: List[float],
    dt: float,
    n_paths: int = 1000,
    tmax: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    对不同的 (λ, μ) 组合进行均方稳定性数值验证。

    对每个 (λ, μ)，运行 N 条轨迹，估计 E[X^2(T)]。
    若 E[X^2(T)] < E[X^2(0)] 则判定为稳定。

    Returns
    -------
    stability_matrix : ndarray, shape (len(lambda_list), len(mu_list))
        1=稳定, 0=不稳定
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    nL = len(lambda_list)
    nM = len(mu_list)
    stability = np.zeros((nL, nM), dtype=int)

    for i, lam in enumerate(lambda_list):
        for j, mu in enumerate(mu_list):
            # 理论预测
            theory_stable = (1.0 + lam * dt) ** 2 + (mu ** 2) * dt < 1.0

            # 数值验证
            n_steps = int(tmax / dt)
            x0 = 1.0
            x_final_sq = []
            for _ in range(n_paths):
                xtemp = x0
                for _ in range(n_steps):
                    dW = np.sqrt(dt) * rng.standard_normal()
                    xtemp = xtemp + lam * xtemp * dt + mu * xtemp * dW
                x_final_sq.append(xtemp ** 2)

            mean_sq = np.mean(x_final_sq)
            num_stable = mean_sq < x0 ** 2

            # 理论与数值一致时采用数值结果
            if num_stable:
                stability[i, j] = 1

    return stability
