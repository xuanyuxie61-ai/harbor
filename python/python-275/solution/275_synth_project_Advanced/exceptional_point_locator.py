"""
exceptional_point_locator.py
============================
在参数空间中定位非厄米系统的例外点 (EP).
融合种子项目:
  - 138_cauchy_method: Cauchy 法求解隐式方程 (寻找 Δ=0 的参数点)
  - 893_polynomial: 特征多项式判别式分析
  - 254_cvt_circle_uniform: CVT 采样辅助全局搜索

物理定义:
  n 阶 EP: 哈密顿量 H(λ) 的 n 个本征值与本征矢同时简并.
  数学条件 (2 阶 EP):
    det(H - E*I) = 0,
    d/dλ det(H - E*I) = 0,
    等价于判别式 Δ(λ) = tr(H)^2 - 4*det(H) = 0.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
import nonhermitian_hamiltonian as nh


# ---------------------------------------------------------------------------
# 一维参数扫描: 沿 k 轴寻找 EP
# ---------------------------------------------------------------------------
def scan_discriminant_1d(t1: float, t2: float, gamma: float,
                         k_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """沿布里渊区 k 扫描判别式 Δ(k).

    Returns
    -------
    delta_vals : |Δ(k)|
    k_vals : 输入的 k 网格
    """
    delta_vals = np.zeros(len(k_vals))
    for i, k in enumerate(k_vals):
        H = nh.ssh_hamiltonian_1d(t1, t2, gamma, k)
        delta_vals[i] = abs(nh.discriminant_ep_indicator(H))
    return delta_vals, k_vals


def locate_ep_1d_newton(t1: float, t2: float, gamma: float,
                        k0: float, tol: float = 1e-12,
                        max_iter: int = 30) -> Dict:
    """Newton 法精确定位 k 空间中的 EP.

    目标函数: f(k) = |Δ(k)|^2 = 0.
    数值导数: f'(k) ≈ (f(k+h) - f(k-h)) / (2h).

    Returns
    -------
    result : dict with 'k_ep', 'delta_min', 'converged', 'iterations'.
    """
    h = 1e-7
    k = k0
    for iteration in range(max_iter):
        H = nh.ssh_hamiltonian_1d(t1, t2, gamma, k)
        delta = nh.discriminant_ep_indicator(H)
        f_val = abs(delta) ** 2
        if f_val < tol:
            return {"k_ep": float(k), "delta_min": float(np.sqrt(f_val)),
                    "converged": True, "iterations": iteration}
        H_p = nh.ssh_hamiltonian_1d(t1, t2, gamma, k + h)
        H_m = nh.ssh_hamiltonian_1d(t1, t2, gamma, k - h)
        d_p = nh.discriminant_ep_indicator(H_p)
        d_m = nh.discriminant_ep_indicator(H_m)
        f_prime = (abs(d_p) ** 2 - abs(d_m) ** 2) / (2 * h)
        if abs(f_prime) < 1e-15:
            break
        step = f_val / f_prime
        k = k - step
        k = k % (2 * np.pi)
    H = nh.ssh_hamiltonian_1d(t1, t2, gamma, k)
    delta = abs(nh.discriminant_ep_indicator(H))
    return {"k_ep": float(k), "delta_min": float(delta),
            "converged": delta < 1e-6, "iterations": max_iter}


# ---------------------------------------------------------------------------
# 二维参数空间 EP 定位 (源自 138_cauchy_method 的 fsolve 思想)
# ---------------------------------------------------------------------------
def ep_residual_2d(params: np.ndarray, t2: float,
                   k_fixed: float) -> np.ndarray:
    """二维 EP 残差: (Re(Δ), Im(Δ)) = (0, 0).

    params = (t1, gamma). 对固定 t2, k, 寻找 (t1, gamma) 使 Δ = 0.
    """
    t1, gamma = params
    H = nh.ssh_hamiltonian_1d(t1, t2, gamma, k_fixed)
    delta = nh.discriminant_ep_indicator(H)
    return np.array([delta.real, delta.imag])


def cauchy_fsolve_ep(residual_func: Callable, x0: np.ndarray,
                     tol: float = 1e-10, max_iter: int = 50) -> Dict:
    """Cauchy 型迭代求解 EP 方程 (源自 cauchy_fsolve).

    实值残差 F: R^n -> R^n, 求 x 使 F(x) = 0.
    采用 damped Newton: x_{n+1} = x_n - α J^{-1} F(x_n).

    Returns
    -------
    result : dict with 'x_sol', 'residual', 'converged', 'iterations'.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    h = 1e-7
    for iteration in range(max_iter):
        F = residual_func(x)
        res_norm = np.linalg.norm(F)
        if res_norm < tol:
            return {"x_sol": x.copy(), "residual": float(res_norm),
                    "converged": True, "iterations": iteration}
        J = np.zeros((n, n))
        for j in range(n):
            x_p = x.copy()
            x_p[j] += h
            F_p = residual_func(x_p)
            J[:, j] = (F_p - F) / h
        try:
            delta_x = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            delta_x = np.linalg.lstsq(J, F, rcond=None)[0]
        alpha = 1.0
        for _ in range(5):
            x_new = x - alpha * delta_x
            F_new = residual_func(x_new)
            if np.linalg.norm(F_new) < res_norm:
                break
            alpha *= 0.5
        x = x_new
    F = residual_func(x)
    return {"x_sol": x.copy(), "residual": float(np.linalg.norm(F)),
            "converged": False, "iterations": max_iter}


# ---------------------------------------------------------------------------
# 多阶 EP 识别
# ---------------------------------------------------------------------------
def ep_order_estimation(H: np.ndarray, tol: float = 1e-8) -> Dict:
    """估计 EP 的阶数 (通过 Jordan 块结构).

    数学: 对简并本征值 E, Jordan 块大小 = 代数重数.
    几何重数 = 线性无关本征矢数 = nullity(H - E*I).
    代数重数 - 几何重数 > 0 表明非平凡 Jordan 结构.

    Returns
    -------
    info : dict with 'eigenvalues', 'algebraic_multiplicity',
           'geometric_multiplicity', 'ep_order'.
    """
    E, _ = np.linalg.eig(H)
    unique_E = []
    alg_mult = []
    geom_mult = []
    used = np.zeros(len(E), dtype=bool)
    for i in range(len(E)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(E)):
            if not used[j] and abs(E[i] - E[j]) < tol:
                cluster.append(j)
                used[j] = True
        E_val = np.mean([E[k] for k in cluster])
        unique_E.append(E_val)
        alg_mult.append(len(cluster))
        M = H - E_val * np.eye(H.shape[0])
        rank = np.linalg.matrix_rank(M, tol=tol * 10)
        geom = H.shape[0] - rank
        geom_mult.append(geom)
    return {
        "eigenvalues": unique_E,
        "algebraic_multiplicity": alg_mult,
        "geometric_multiplicity": geom_mult,
        "ep_order": [a - g for a, g in zip(alg_mult, geom_mult)],
    }


# ---------------------------------------------------------------------------
# CVT 辅助全局 EP 搜索 (源自 254_cvt_circle_uniform)
# ---------------------------------------------------------------------------
def cvt_parameter_sampling(n_samples: int, param_bounds: List[Tuple[float, float]],
                           it_num: int = 20,
                           seed: int = 42) -> np.ndarray:
    """Centroidal Voronoi Tessellation 在参数空间中生成均匀采样.

    用于全局搜索 EP, 避免陷入局部极小.
    算法: Lloyd 迭代, 每次将点移到其 Voronoi cell 的质心.

    Returns
    -------
    points : (n_samples, dim) 采样点.
    """
    rng = np.random.default_rng(seed)
    dim = len(param_bounds)
    pts = np.zeros((n_samples, dim))
    for d in range(dim):
        pts[:, d] = rng.uniform(param_bounds[d][0], param_bounds[d][1],
                                size=n_samples)
    for _ in range(it_num):
        new_pts = np.zeros_like(pts)
        for i in range(n_samples):
            dists = np.sum((pts - pts[i]) ** 2, axis=1)
            weights = np.exp(-dists / (np.mean(dists) + 1e-15))
            weights /= weights.sum()
            for d in range(dim):
                lo, hi = param_bounds[d]
                samples = rng.uniform(lo, hi, size=50)
                w = np.exp(-((samples - pts[i, d]) ** 2) / (np.mean(dists) + 1e-15))
                w /= w.sum()
                new_pts[i, d] = np.sum(samples * w)
                new_pts[i, d] = np.clip(new_pts[i, d], lo, hi)
        pts = new_pts
    return pts


def global_ep_search(t2: float, k_range: Tuple[float, float] = (0, 2 * np.pi),
                     t1_range: Tuple[float, float] = (0, 3),
                     gamma_range: Tuple[float, float] = (0, 2),
                     n_candidates: int = 20,
                     tol: float = 1e-6) -> List[Dict]:
    """全局搜索 (t1, gamma, k) 空间中的 EP.

    流程: CVT 采样 → 评估 |Δ| → 对候选点 Newton 精化.

    Returns
    -------
    eps : 列表, 每个元素为 dict 含 't1', 'gamma', 'k', 'delta'.
    """
    bounds = [t1_range, gamma_range, k_range]
    candidates = cvt_parameter_sampling(n_candidates, bounds)
    results = []
    for cand in candidates:
        t1_c, gamma_c, k_c = cand
        res = locate_ep_1d_newton(t1_c, t2, gamma_c, k_c)
        if res["converged"]:
            results.append({
                "t1": t1_c,
                "gamma": gamma_c,
                "k": res["k_ep"],
                "delta": res["delta_min"],
            })
    results.sort(key=lambda r: r["delta"])
    return results
