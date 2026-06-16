"""
fractal_perturbation.py
-----------------------
共振态参数的分形扰动: 用分形曲线生成方法 (仿 446_fractal_coastline)
对 Dalitz 振幅中的共振参数进行系统性扰动, 模拟理论不确定性,
并研究有限差分格式在扰动下的稳定性。

物理动机:
  在 B 衰变的 isobar 模型中, 每个共振道有参数 (m_r, Gamma_r, a_r, delta_r),
  这些参数有实验误差和理论模型误差。为评估系统误差, 需要对参数进行
  系统性扰动扫描。传统方法: 对每个参数独立变化 ±1σ。
  本模块采用分形方法:
    将 N 个共振道的参数组合视为 N 维空间中的一个点,
    在参数空间中沿分形路径 (Koch 曲线类比分形扰动链) 采样,
    可更有效地探索参数空间的相关方向。

  分形扰动链 (仿 446.coastline_perturb):
    初始: N 个共振道参数的中心值 p^{(0)} = (p_1, ..., p_N)
    扰动: p^{(k+1)} = p^{(k)} + mu_k * (p^{(k)} - p^{(k-1)})
                       + noise * sigma_k
    其中 mu_k 控制扰动强度 (类似分形维数 D = 1 + 2 mu),
    noise 为随机扰动。
    迭代 n_iter 次后, 得到 2^n_iter 个参数点, 构成参数空间的分形采样。

  稳定性分析:
    对每个扰动参数点, 计算有限差分格式的放大因子 G 和残差,
    评估格式对参数扰动的鲁棒性。

数学公式:
  对闭合多边形 p (N 个顶点), 分形扰动 (仿 446):
      q_{2k-1} = p_k   (保留原顶点)
      q_{2k}   = 0.5 (p_k + p_{k+1}) + w_k (p_k - p_{k-1})
               - w_k (p_{k+2} - p_{k+1})
    其中 w_k = mu + mu^2 * N(0,1), mu 为扰动强度参数。
  分形维数 D ≈ 2 - log(1 + 2 mu) / log(2)  (对 mu << 1)。

  参数空间中的分形采样用于:
    1) 系统误差评估 (参数扰动对 CP 不对称的影响);
    2) 有限差分格式的鲁棒性研究;
    3) 共振模型不确定性的量化。
"""

from __future__ import annotations
import math
import random
from typing import Callable, List, Tuple

import numpy as np


# ========================================================================== #
#              分形扰动链 (仿 446_fractal_coastline)                       #
# ========================================================================== #
def fractal_perturb_closed_curve(
        p: np.ndarray,
        mu: float,
        seed: int = 42,
) -> np.ndarray:
    """
    在闭合多边形 p (shape (N, d)) 中插入中间点, 生成扰动曲线。
    返回 shape (2N, d) 的新曲线, 其中奇数索引为原顶点,
    偶数索引为中间扰动点。

    算法 (仿 446.coastline_perturb):
        sig = mu^2
        w_k = mu + sig * N(0, 1)
        perturb_k = 0.5 (p_k + p_{k+1})
                    + w_k (p_k - p_{k-1})
                    - w_k (p_{k+2} - p_{k+1})
        q[2k-1] = p[k]   (原顶点)
        q[2k]   = perturb[k]   (扰动中点)
    """
    if mu < 0.0 or mu > 0.5:
        raise ValueError("mu 应在 [0, 0.5] 范围内以保证收敛")
    rng = np.random.default_rng(seed)
    n, d = p.shape
    sig = mu * mu
    w = mu + sig * rng.standard_normal(n)
    # 循环位移: p_{k-1}, p_k, p_{k+1}, p_{k+2}
    p_m1 = np.roll(p, 1, axis=0)
    p_p1 = np.roll(p, -1, axis=0)
    p_p2 = np.roll(p, -2, axis=0)
    perturb = (
        0.5 * (p + p_p1)
        + w[:, None] * (p - p_m1)
        - w[:, None] * (p_p2 - p_p1)
    )
    perturb = np.roll(perturb, -1, axis=0)
    q = np.zeros((2 * n, d), dtype=float)
    q[0::2] = p
    q[1::2] = perturb
    return q


def fractal_iteration(
        initial: np.ndarray,
        mu: float,
        n_iter: int,
        seed: int = 42,
) -> List[np.ndarray]:
    """
    对初始曲线迭代分形扰动 n_iter 次, 每次点数翻倍。
    返回 [p^{(0)}, p^{(1)}, ..., p^{(n_iter)}] 的列表。
    """
    history = [initial.copy()]
    current = initial.copy()
    for k in range(n_iter):
        current = fractal_perturb_closed_curve(
            current, mu, seed=seed + k
        )
        history.append(current.copy())
    return history


def fractal_dimension_estimate(curve: np.ndarray) -> float:
    """
    估计闭合曲线的分形维数 (box-counting 简化版):
        D ≈ log(N(eps)) / log(1/eps)
    其中 N(eps) 为覆盖曲线所需的 eps-方格数。
    对非分形闭合多边形, D = 1; 对充分扰动的分形曲线, D > 1。
    """
    if len(curve) < 4:
        return 1.0
    # 计算曲线总长度 L
    L = 0.0
    for k in range(len(curve) - 1):
        L += np.linalg.norm(curve[k + 1] - curve[k])
    L += np.linalg.norm(curve[0] - curve[-1])
    # 估计直径
    all_pairs = np.max(np.linalg.norm(
        curve[:, None, :] - curve[None, :, :], axis=-1
    ))
    if all_pairs < 1.0e-30:
        return 1.0
    eps = all_pairs / 10.0
    n_boxes = max(int(L / eps), 1)
    D = math.log(n_boxes) / math.log(1.0 / eps) if eps < 1.0 else 1.0
    return max(1.0, min(D, 2.0))


# ========================================================================== #
#           共振参数空间的分形采样                                          #
# ========================================================================== #
def build_resonance_parameter_curve(
        resonances: List[dict],
) -> np.ndarray:
    """
    将共振参数列表转换为 N 维空间中的闭合曲线顶点。
    resonances: [{"m_r": float, "gamma_r": float, "a": float, "delta": float}, ...]
    将每个共振的参数 (m_r, gamma_r, a, delta) 拼接为一个 4N 维点,
    作为曲线的单个顶点。构造 N_v 个顶点形成闭合曲线 (用于分形扰动)。
    """
    n_res = len(resonances)
    if n_res == 0:
        raise ValueError("共振列表不能为空")
    n_dim = 4 * n_res
    n_vertices = 8   # 初始闭合多边形顶点数
    rng = np.random.default_rng(42)
    # 中心点: 所有参数的中心值
    center = np.zeros(n_dim)
    for k, r in enumerate(resonances):
        center[4 * k + 0] = r["m_r"]
        center[4 * k + 1] = r["gamma_r"]
        center[4 * k + 2] = r["a"]
        center[4 * k + 3] = r["delta"]
    # 构造 n_vertices 个顶点: 中心 + 小扰动
    vertices = np.zeros((n_vertices, n_dim))
    vertices[0] = center
    for v in range(1, n_vertices):
        pert = 0.05 * rng.standard_normal(n_dim) * np.maximum(np.abs(center), 1.0)
        vertices[v] = center + pert
    return vertices


def scan_systematic_uncertainty(
        resonances: List[dict],
        amplitude_func: Callable,
        mu_scan: float = 0.1,
        n_iter: int = 3,
        n_sample_obs: int = 20,
        seed: int = 42,
) -> dict:
    """
    用分形扰动扫描共振参数的系统不确定性:
      1) 构造初始参数曲线;
      2) 迭代分形扰动, 得到分形采样的参数点;
      3) 对每个参数点计算观测量 amplitude_func;
      4) 统计观测量的分布: 均值, 标准差, min, max。
    返回 dict:
        "mean": float, "std": float, "min": float, "max": float,
        "fractal_dim": float, "n_points": int
    """
    initial = build_resonance_parameter_curve(resonances)
    history = fractal_iteration(initial, mu_scan, n_iter, seed=seed)
    final = history[-1]
    # 对 final 曲线的每个顶点计算观测量
    n_res = len(resonances)
    observables = []
    for v in range(len(final)):
        # 将 4N 维向量还原为共振参数
        params_v = []
        for k in range(n_res):
            params_v.append({
                "m_r": final[v, 4 * k + 0],
                "gamma_r": final[v, 4 * k + 1],
                "a": final[v, 4 * k + 2],
                "delta": final[v, 4 * k + 3],
            })
        try:
            obs = amplitude_func(params_v)
            observables.append(float(obs))
        except Exception:
            continue
    if not observables:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "fractal_dim": 1.0, "n_points": 0}
    obs_arr = np.array(observables)
    return {
        "mean": float(obs_arr.mean()),
        "std": float(obs_arr.std(ddof=1)) if len(obs_arr) > 1 else 0.0,
        "min": float(obs_arr.min()),
        "max": float(obs_arr.max()),
        "fractal_dim": fractal_dimension_estimate(final),
        "n_points": len(obs_arr),
    }


# ========================================================================== #
#          有限差分格式的鲁棒性测试 (在分形扰动下)                          #
# ========================================================================== #
def fd_robustness_under_fractal_perturbation(
        fd_operator: Callable[[np.ndarray], np.ndarray],
        exact_solution: np.ndarray,
        mu_values: List[float],
        noise_level: float = 0.01,
        seed: int = 42,
) -> List[Tuple[float, float, float]]:
    """
    研究有限差分算子在解受分形扰动时的鲁棒性:
      对每个 mu 值:
        1) 在 exact_solution 上加分形扰动 (noise_level * mu * 随机);
        2) 应用 fd_operator, 计算残差 r = fd(perturbed) - fd(exact);
        3) 记录 ||r||_infty / ||perturbed - exact||_infty。
    返回 [(mu, relative_residual, perturbation_norm), ...]
    """
    rng = np.random.default_rng(seed)
    results = []
    exact_img = fd_operator(exact_solution)
    for mu in mu_values:
        if mu < 0.0 or mu > 1.0:
            raise ValueError("mu 应在 [0, 1]")
        # 分形扰动: 多重尺度的随机噪声
        pert = np.zeros_like(exact_solution)
        scale = 1.0
        for level in range(4):
            noise = rng.standard_normal(exact_solution.shape)
            pert += scale * noise
            scale *= 0.5
        pert = pert * noise_level * mu * np.maximum(np.abs(exact_solution).max(), 1.0)
        perturbed = exact_solution + pert
        perturbed_img = fd_operator(perturbed)
        residual = perturbed_img - exact_img
        rel_res = float(np.abs(residual).max()) / max(
            float(np.abs(pert).max()), 1.0e-30
        )
        pert_norm = float(np.abs(pert).max())
        results.append((mu, rel_res, pert_norm))
    return results
