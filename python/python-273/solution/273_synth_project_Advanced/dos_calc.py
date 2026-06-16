"""
dos_calc.py — 声子态密度 (DOS) 计算
====================================

融合种子项目:
  - 1242_ce335805_PhotonDosReference: 光子 DOS 计算方法
    (模式求和, 根搜索, 频率积分)
  - 004_alpert_rule: Alpert 求积规则 (处理 van Hove 奇点)
  - 195_coin_simulation: 统计采样与运行平均收敛

物理背景:
  声子态密度 g(omega) 定义:
    g(omega) = (1/N_k) * sum_{k,lambda} delta(omega - omega_lambda(k))

  数值计算使用 Gaussian 展宽:
    delta(x) ≈ (1/(sigma*sqrt(2*pi))) * exp(-x^2/(2*sigma^2))

  或使用 Tetrahedron 方法 (更精确但更复杂)。

  van Hove 奇点:
    在 d(g(omega))/d(omega) 发散处, 对应临界点
    grad_k omega(k) = 0

  对 3D 晶体:
    g(omega) ~ omega^2  (Debye 模型, omega -> 0)
    g(omega) ~ |omega - omega_vH|^{-1/2}  (van Hove 奇点附近)
"""

import numpy as np
from typing import Tuple, Dict, List


def compute_dos_gaussian_broadening(
    omega_all: np.ndarray,
    omega_grid: np.ndarray,
    sigma: float = 0.5,
    weights: np.ndarray = None,
) -> np.ndarray:
    """
    使用 Gaussian 展宽计算声子 DOS。

    g(omega) = (1/N) * sum_i w_i * G(omega - omega_i; sigma)
    其中 G(x; sigma) = exp(-x^2/(2*sigma^2)) / (sigma*sqrt(2*pi))

    融合 PhotonDosReference 的频率网格离散化策略。

    参数:
        omega_all: (N_total,) 所有 k 点的声子频率
        omega_grid: (N_omega,) 输出频率网格
        sigma: Gaussian 展宽宽度 (THz)
        weights: (N_total,) 权重 (默认等权)

    返回:
        dos: (N_omega,) 态密度
    """
    if weights is None:
        weights = np.ones(len(omega_all)) / len(omega_all)

    dos = np.zeros(len(omega_grid))
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    for i, w_val in enumerate(omega_grid):
        diff = w_val - omega_all
        gauss = np.exp(-0.5 * (diff / sigma) ** 2) * norm
        dos[i] = np.sum(weights * gauss)

    return dos


def compute_dos_tetrahedron(
    omega_all: np.ndarray,
    omega_grid: np.ndarray,
    tetra_indices: np.ndarray = None,
) -> np.ndarray:
    """
    Tetrahedron 方法计算 DOS (更精确)。

    将 Brillouin 区划分为四面体, 在每个四面体内
    对 omega(k) 做线性插值, 精确计算 delta 函数的积分。

    对四面体顶点频率 w1 < w2 < w3 < w4:
    在 [w1, w2] 区间:
      g(w) = 3*(w-w1)^2 / ((w2-w1)*(w3-w1)*(w4-w1))
    ... (分三段)

    简化实现: 使用最近邻插值。
    """
    if tetra_indices is None:
        # 简化: 使用 Gaussian 方法近似
        omega_range = omega_all.max() - omega_all.min()
        sigma = max(omega_range / 50.0, 0.1)
        return compute_dos_gaussian_broadening(omega_all, omega_grid, sigma)

    n_tetra = len(tetra_indices)
    dos = np.zeros(len(omega_grid))

    for tet in tetra_indices:
        if len(tet) < 4:
            continue
        omegas = omega_all[tet[:4]] if len(tet) >= 4 else omega_all[tet]
        sorted_omega = np.sort(omegas)
        if len(sorted_omega) < 4:
            continue
        w1, w2, w3, w4 = sorted_omega
        vol = max((w4 - w1) * (w3 - w1) * (w2 - w1), 1e-30)

        for iw, w in enumerate(omega_grid):
            if w1 <= w <= w2:
                dos[iw] += 3.0 * (w - w1) ** 2 / vol
            elif w2 < w <= w3:
                dos[iw] += (3.0 / vol) * ((w - w1) * (w3 - w) + (w - w2) * (w4 - w)) / 3.0
            elif w3 < w <= w4:
                dos[iw] += 3.0 * (w4 - w) ** 2 / vol

    # 归一化
    total = np.trapz(dos, omega_grid) if len(omega_grid) > 1 else 1.0
    if total > 1e-30:
        dos /= total

    return dos


def compute_projected_dos(
    omega_all: np.ndarray,
    eigenvectors: np.ndarray,
    omega_grid: np.ndarray,
    atom_indices: List[int],
    sigma: float = 0.5,
) -> np.ndarray:
    """
    投影态密度 (PDOS): 特定原子的部分 DOS。

    g_a(omega) = sum_{k,lambda} |e_{lambda,a}(k)|^2 * delta(omega - omega_{lambda,k})

    其中 e_{lambda,a} 是模式 lambda 在原子 a 上的偏振分量。
    """
    n_total = len(omega_all)
    n_atoms_total = eigenvectors.shape[0] // 3
    weights = np.zeros(n_total)

    for idx in range(n_total):
        e = eigenvectors[:, idx] if eigenvectors.ndim > 1 else eigenvectors
        mode_weight = 0.0
        for a_idx in atom_indices:
            for dim in range(3):
                comp_idx = a_idx * 3 + dim
                if comp_idx < len(e):
                    mode_weight += np.abs(e[comp_idx]) ** 2
        weights[idx] = mode_weight

    # 归一化
    total_w = np.sum(weights)
    if total_w > 1e-30:
        weights /= total_w

    return compute_dos_gaussian_broadening(omega_all, omega_grid, sigma, weights)


def find_van_hove_singularities(
    omega_grid: np.ndarray,
    dos: np.ndarray,
) -> List[Dict]:
    """
    检测 van Hove 奇点 (DOS 的局部极大值/拐点)。

    van Hove 奇点对应 grad_k omega = 0 的临界点。
    在 DOS 中表现为:
      - 1D: g(omega) ~ |omega - omega_c|^{-1/2} (发散)
      - 2D: g(omega) 有阶跃不连续
      - 3D: g(omega) 有拐点 (d^2g/d(omega)^2 发散)
    """
    singularities = []
    if len(dos) < 5:
        return singularities

    # 计算 DOS 的一阶和二阶导数
    d_dos = np.gradient(dos, omega_grid)
    d2_dos = np.gradient(d_dos, omega_grid)

    # 寻找 d_dos 的过零点 (极大值)
    for i in range(1, len(d_dos) - 1):
        if d_dos[i - 1] > 0 and d_dos[i + 1] < 0:
            singularities.append({
                'omega': omega_grid[i],
                'dos_value': dos[i],
                'type': 'maximum',
                'curvature': d2_dos[i],
            })
        elif d_dos[i - 1] < 0 and d_dos[i + 1] > 0:
            singularities.append({
                'omega': omega_grid[i],
                'dos_value': dos[i],
                'type': 'minimum',
                'curvature': d2_dos[i],
            })

    return singularities


def compute_debye_dos(
    omega_grid: np.ndarray,
    v_sound: float,
    volume: float,
    n_atoms: int = 1,
) -> np.ndarray:
    """
    Debye 模型 DOS: g(omega) = V * omega^2 / (2*pi^2 * v_s^3)

    参数:
        omega_grid: 频率网格
        v_sound: 声速
        volume: 体积
        n_atoms: 原胞原子数
    """
    dos = volume * omega_grid ** 2 / (2.0 * np.pi ** 2 * max(v_sound, 1e-15) ** 3)
    # 截断到 Debye 频率
    omega_D = v_sound * (6.0 * np.pi ** 2 * n_atoms / volume) ** (1.0 / 3.0)
    dos = np.where(omega_grid <= omega_D, dos, 0.0)
    # 归一化: integral g(omega) d_omega = 3*N
    total = np.trapz(dos, omega_grid) if len(omega_grid) > 1 else 1.0
    if total > 1e-30:
        dos *= 3.0 * n_atoms / total
    return dos


def alpert_singular_dos_integral(
    dos_func, omega_range: Tuple[float, float],
    singularity_omega: float,
    n_quad: int = 64,
) -> float:
    """
    Alpert 求积计算 van Hove 奇点附近的 DOS 积分 (融合 alpert_rule)。

    integral g(omega) d_omega 在奇点附近
    使用混合 Gauss-梯形规则处理可积奇异性。
    """
    from fd_stencil import alpert_quadrature_nodes_weights
    nodes, weights = alpert_quadrature_nodes_weights('power', rule_index=4)

    # 映射到积分区间
    a, b = omega_range
    omega_pts = a + (b - a) * nodes
    dos_vals = dos_func(omega_pts)
    integral = (b - a) * np.sum(weights * dos_vals)
    return integral


def running_average_dos(
    omega_all: np.ndarray,
    omega_grid: np.ndarray,
    sigma: float,
    n_batches: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    运行平均 DOS (融合 coin_simulation 的运行统计)。

    将 k 点分成 n_batches 批, 逐批累加 DOS,
    监测运行平均的收敛: |g_n - g_{n-1}| / |g_n| < tol

    返回:
        dos_final: 最终 DOS
        convergence: 每批的相对变化
    """
    rng = np.random.RandomState(seed)
    n_total = len(omega_all)
    batch_size = n_total // n_batches
    indices = rng.permutation(n_total)

    dos_cumulative = np.zeros(len(omega_grid))
    convergence = []

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size if batch < n_batches - 1 else n_total
        batch_idx = indices[start:end]

        dos_batch = compute_dos_gaussian_broadening(
            omega_all[batch_idx], omega_grid, sigma
        )
        dos_cumulative += dos_batch
        dos_avg = dos_cumulative / (batch + 1)

        if batch > 0:
            prev_avg = (dos_cumulative - dos_batch) / batch
            rel_change = np.max(np.abs(dos_avg - prev_avg)) / (np.max(np.abs(dos_avg)) + 1e-30)
            convergence.append(rel_change)
        else:
            convergence.append(1.0)

    return dos_avg, np.array(convergence)
