"""
stability_analysis.py
=====================
非厄米时间演化的稳定性分析.
融合种子项目:
  - 357_fd1d_burgers_leap: leapfrog 稳定性
  - 138_cauchy_method: Cauchy theta 方法稳定性
  - 003_allen_cahn_pde: 扩散型方程稳定性约束

物理框架:
  非厄米薛定谔方程: i dψ/dt = H ψ, H = H_0 + iΓ
  解: ψ(t) = exp(-iHt) ψ(0)
  稳定性取决于谱: 若 Im(E_n) > 0 → 模式指数增长 (不稳定/增益).
  数值稳定性: 离散格式需满足 CFL 条件.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List
import high_order_fd as hfd


# ---------------------------------------------------------------------------
# 谱稳定性判据
# ---------------------------------------------------------------------------
def spectral_stability_analysis(H: np.ndarray) -> Dict:
    """基于谱的稳定性分析.

    对 H ψ = E ψ:
      - 若所有 Im(E) ≤ 0: 系统耗散, 稳定.
      - 若存在 Im(E) > 0: 存在增长模式 (PT 对称破缺).
      - 若所有 E 为实数: PT 对称未破缺相.

    Returns
    -------
    info : dict with 'eigenvalues', 'max_imag', 'min_imag',
           'pt_broken', 'growth_rate', 'stable_modes', 'unstable_modes'.
    """
    E = np.linalg.eigvals(H)
    imag_parts = E.imag
    real_parts = E.real
    max_imag = float(np.max(imag_parts))
    min_imag = float(np.min(imag_parts))
    n_unstable = int(np.sum(imag_parts > 1e-10))
    n_stable = int(np.sum(imag_parts <= 1e-10))
    all_real = bool(np.all(np.abs(imag_parts) < 1e-8))
    return {
        "eigenvalues": E,
        "max_imag": max_imag,
        "min_imag": min_imag,
        "pt_broken": not all_real and max_imag > 1e-8,
        "all_real_spectrum": all_real,
        "growth_rate": max_imag,
        "stable_modes": n_stable,
        "unstable_modes": n_unstable,
        "spectral_radius": float(np.max(np.abs(E))),
    }


def pseudospectrum_epsilon(H: np.ndarray, epsilon: float = 1e-3,
                           grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 ε-伪谱.

    ε-伪谱: σ_ε(H) = {z ∈ C : ||(zI - H)^{-1}|| > 1/ε}
    等价于: σ_ε(H) = ∪_{||E||<ε} σ(H + E)

    数值: 计算 s_min(zI - H), 若 s_min < ε 则 z ∈ σ_ε.

    Returns
    -------
    Re_grid, Im_grid : 网格
    log_smin : log10(s_min(zI - H))
    """
    E_vals = np.linalg.eigvals(H)
    re_center = np.mean(E_vals.real)
    im_center = np.mean(E_vals.imag)
    span = max(np.std(E_vals) * 3, 0.5)
    re_grid = np.linspace(re_center - span, re_center + span, grid_size)
    im_grid = np.linspace(im_center - span, im_center + span, grid_size)
    log_smin = np.zeros((grid_size, grid_size))
    I = np.eye(H.shape[0])
    for i, re in enumerate(re_grid):
        for j, im in enumerate(im_grid):
            z = re + 1j * im
            M = z * I - H
            s = np.linalg.svd(M, compute_uv=False)[-1]
            log_smin[i, j] = np.log10(max(s, 1e-16))
    return re_grid, im_grid, log_smin


# ---------------------------------------------------------------------------
# 数值格式稳定性 (源自 leapfrog + Cauchy 方法)
# ---------------------------------------------------------------------------
def leapfrog_stability_bound(nu: float, gamma: float,
                             dx: float, order: int = 2) -> float:
    """Leapfrog 格式的稳定性上界 dt_max.

    对 ∂u/∂t = ν ∂²u/∂x² + iγu, 纯空间离散后:
      d û_k / dt = (-ν k² + iγ) û_k
    Leapfrog: û^{n+1} = û^{n-1} + 2 dt λ_k û^n
    稳定条件: |dt * λ_k| ≤ 1 对所有 k.

    λ_k = ν λ_fd(k) + iγ
    |λ_k|² = ν² |λ_fd(k)|² + γ²

    dt_max = 1 / max_k |λ_k|

    Returns
    -------
    dt_max : 最大允许时间步.
    """
    p = order // 2
    c = hfd.fd_coefficients_second_derivative(p)
    n_k = 256
    kx_vals = np.linspace(0, np.pi / dx, n_k)
    max_abs_lambda = 0.0
    for kx in kx_vals:
        lambda_fd = 0.0 + 0.0j
        for m in range(-p, p + 1):
            lambda_fd += c[m + p] * np.exp(1j * m * kx * dx)
        lambda_fd /= dx * dx
        lambda_k = nu * lambda_fd + 1j * gamma
        abs_lam = abs(lambda_k)
        if abs_lam > max_abs_lambda:
            max_abs_lambda = abs_lam
    if max_abs_lambda < 1e-15:
        return np.inf
    return 1.0 / max_abs_lambda


def cauchy_theta_stability(theta: float, nu: float, gamma: float,
                           dx: float, order: int = 2) -> Dict:
    """Cauchy theta 方法的稳定性区域.

    theta = 1: 后向 Euler (无条件稳定, A-稳定).
    theta = 0.5: Crank-Nicolson (|g| = 1 对纯虚数 λ).
    theta < 0.5: 可能不稳定.

    增长因子: g = (1 + (1-θ)z) / (1 - θ z), z = dt*λ.

    Returns
    -------
    info : dict with 'theta', 'A_stable', 'max_growth_for_pure_imag'.
    """
    if theta >= 0.5:
        a_stable = True
    else:
        a_stable = False
    n_test = 200
    z_vals = 1j * np.linspace(-10, 10, n_test)
    max_g = 0.0
    for z in z_vals:
        num = 1.0 + (1.0 - theta) * z
        den = 1.0 - theta * z
        if abs(den) > 1e-15:
            g = abs(num / den)
            if g > max_g:
                max_g = g
    return {
        "theta": theta,
        "A_stable": a_stable,
        "max_growth_for_pure_imag": float(max_g),
        "L_stable": theta == 1.0,
    }


# ---------------------------------------------------------------------------
# 时间演化模拟 + 范数监测
# ---------------------------------------------------------------------------
def time_evolution_norm(H: np.ndarray, psi0: np.ndarray,
                        t_max: float, n_steps: int,
                        method: str = "cauchy",
                        theta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """模拟 ||ψ(t)|| 的时间演化, 检测稳定性.

    对非厄米 H, ||ψ(t)|| 可指数增长/衰减:
      ||ψ(t)|| ~ exp(Im(E_max) * t)

    方法:
      - 'exact': 矩阵指数 exp(-iHt)
      - 'cauchy': Cauchy theta 方法
      - 'leapfrog': leapfrog (二阶, 可能不稳定)

    Returns
    -------
    t_vals, norm_vals : 时间和范数序列.
    """
    N = len(psi0)
    dt = t_max / n_steps
    t_vals = np.linspace(0, t_max, n_steps + 1)
    norm_vals = np.zeros(n_steps + 1)
    psi = psi0.astype(complex).copy()
    norm_vals[0] = np.linalg.norm(psi)
    if method == "exact":
        try:
            from scipy.linalg import expm
            exp_Hdt = expm(-1j * H * dt)
        except ImportError:
            exp_Hdt = _matrix_exp_approx(-1j * H * dt, order=30)
        for n in range(n_steps):
            psi = exp_Hdt @ psi
            norm_vals[n + 1] = np.linalg.norm(psi)
    elif method == "cauchy":
        def rhs(p):
            return -1j * (H @ p)
        for n in range(n_steps):
            psi = hfd.cauchy_theta_step(psi, rhs, dt, theta=theta)
            norm_vals[n + 1] = np.linalg.norm(psi)
    elif method == "leapfrog":
        def rhs(p):
            return -1j * (H @ p)
        psi_prev = psi - dt * rhs(psi)
        for n in range(n_steps):
            psi_next = hfd.leapfrog_step(psi_prev, psi, rhs, dt)
            psi_prev = psi
            psi = psi_next
            norm_vals[n + 1] = np.linalg.norm(psi)
    else:
        raise ValueError(f"unknown method: {method}")
    return t_vals, norm_vals


def _matrix_exp_approx(A: np.ndarray, order: int = 20) -> np.ndarray:
    """Padé 近似矩阵指数 (截断 Taylor 用于小规模)."""
    I = np.eye(A.shape[0])
    result = I.copy()
    term = I.copy()
    for k in range(1, order + 1):
        term = term @ A / k
        result = result + term
        if np.linalg.norm(term, 'fro') < 1e-15:
            break
    return result


# ---------------------------------------------------------------------------
# CFL 条件扫描 (源自 Burgers leapfrog)
# ---------------------------------------------------------------------------
def cfl_scan(nu: float, gamma: float, dx: float,
             r_values: np.ndarray,
             order: int = 2) -> Dict:
    """扫描 CFL 数 r = dt/dx², 报告稳定性.

    Returns
    -------
    info : dict with 'r_values', 'g_max', 'stable_mask', 'critical_r'.
    """
    g_max = hfd.stability_domain_scan(nu, gamma, dx, r_values,
                                      order=order, scheme="leapfrog")
    stable_mask = g_max <= 1.0 + 1e-10
    critical_r = None
    for i in range(len(r_values) - 1):
        if stable_mask[i] and not stable_mask[i + 1]:
            critical_r = float(r_values[i])
            break
    return {
        "r_values": r_values,
        "g_max": g_max,
        "stable_mask": stable_mask,
        "critical_r": critical_r,
    }
