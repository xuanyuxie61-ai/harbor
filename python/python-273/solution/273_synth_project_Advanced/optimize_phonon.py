"""
optimize_phonon.py — 声子力常数优化与拟合
=========================================

融合种子项目:
  - 218_coordinate_search: 坐标搜索直接优化
    stencil search, pattern move, delta shrink
    收敛: max(f_s) - min(f_s) < tol 或 delta < tol
  - 1149_BanerjeeLab: scipy.optimize + basin hopping
  - 170_chinese_remainder_theorem: 参数空间的模运算分解

物理背景:
  力常数拟合: 从第一性原理或实验数据反推力常数参数。
  目标函数:
    min_{Phi} sum_{k,lambda} w_{k,lambda} * |omega_calc(k,lambda) - omega_exp(k,lambda)|^2
        + lambda_reg * ||Phi||^2

  约束:
    1. 声学求和规则: sum_j Phi_{ij} = 0
    2. 对称性: Phi_{ij,ab} = Phi_{ji,ba}
    3. 正定性: omega^2 >= 0 (稳定性)

  坐标搜索 (coordinate search) 的优势:
    - 不需要梯度信息
    - 适合非光滑目标函数 (如含绝对值的损失)
    - 可处理离散参数 (如近邻壳层截断)
"""

import numpy as np
from typing import Tuple, Dict, Callable, Optional


def coordinate_search_minimize(
    func: Callable,
    x0: np.ndarray,
    delta: float = 1.0,
    tolerance: float = 1e-6,
    max_feval: int = 250,
    stencil_type: str = 'coordinate',
) -> Dict:
    """
    坐标搜索最小化 (直接融合 coordinate_search)。

    算法:
      1. 在当前点 x 的邻域生成模板点
      2. 若模板中存在更优点: 移动到最优点 (pattern move)
      3. 若所有模板点更差: delta -> delta/2
      4. 收敛条件: delta < tol 或 stencil 值变化 < tol

    参数:
        func: 目标函数 f(x) -> float
        x0: 初始点
        delta: 初始步长
        tolerance: 收敛容限
        max_feval: 最大函数评估次数
        stencil_type: 'coordinate' (坐标轴) 或 'water_stick' (2D特殊)

    返回:
        result: {x_opt, f_opt, n_feval, converged, history}
    """
    n = len(x0)
    x = x0.copy()
    fc = func(x)
    n_feval = 1
    history = [(x.copy(), fc)]

    # 生成模板方向
    if stencil_type == 'coordinate':
        stencil = np.zeros((2 * n, n))
        for i in range(n):
            stencil[2 * i, i] = 1.0
            stencil[2 * i + 1, i] = -1.0
    elif stencil_type == 'water_stick' and n == 2:
        stencil = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    else:
        stencil = np.zeros((2 * n, n))
        for i in range(n):
            stencil[2 * i, i] = 1.0
            stencil[2 * i + 1, i] = -1.0

    while delta >= tolerance and n_feval < max_feval:
        # 评估模板点
        f_values = []
        x_stencil = []
        for s in range(len(stencil)):
            x_s = x + delta * stencil[s]
            f_s = func(x_s)
            n_feval += 1
            f_values.append(f_s)
            x_stencil.append(x_s)
            if n_feval >= max_feval:
                break

        f_values = np.array(f_values)
        best_idx = np.argmin(f_values)

        if f_values[best_idx] < fc:
            # Pattern move: 移动到最优点
            x_new = x_stencil[best_idx]
            fc_new = f_values[best_idx]

            # Pattern move 外推
            direction = x_new - x
            x_extrap = x_new + direction
            f_extrap = func(x_extrap)
            n_feval += 1

            if f_extrap < fc_new:
                x = x_extrap
                fc = f_extrap
            else:
                x = x_new
                fc = fc_new

            history.append((x.copy(), fc))

            # 检查 stencil 值变化
            fc_values = np.append(f_values, fc)
            if np.max(fc_values) - np.min(fc_values) < tolerance:
                break
        else:
            # 收缩步长
            delta /= 2.0

    converged = delta < tolerance
    return {
        'x_opt': x,
        'f_opt': fc,
        'n_feval': n_feval,
        'converged': converged,
        'final_delta': delta,
        'history': history,
    }


def fit_force_constants(
    target_frequencies: np.ndarray,
    k_points: np.ndarray,
    lattice_type: str,
    a: float,
    masses: np.ndarray,
    n_shells: int = 3,
) -> Dict:
    """
    拟合力常数以匹配目标声子频率。

    参数化: 每个近邻壳层 l 有一个力常数参数 alpha_l。
    力常数矩阵: Phi_l = alpha_l * Phi_l^unit

    目标: min sum_{k,lambda} (omega_calc - omega_target)^2

    使用坐标搜索优化。
    """
    from lattice_geometry import generate_bravais_lattice, compute_neighbor_shells
    from interatomic_potential import InteratomicPotential
    from dynamical_matrix import build_dynamical_matrix, apply_acoustic_sum_rule
    from dynamical_matrix import fourier_transform_dynamical_matrix
    from eigen_solver import solve_phonon_eigenproblem, extract_phonon_frequencies

    positions, _ = generate_bravais_lattice(lattice_type, a, (2, 2, 2))
    n_atoms = len(positions)
    box_length = 2 * a
    all_masses = np.ones(n_atoms) * np.mean(masses)

    shells = compute_neighbor_shells(positions, box_length, n_shells)

    def objective(alpha_vec):
        """力常数参数 alpha 的目标函数"""
        potential = InteratomicPotential('morse', {
            'D': alpha_vec[0],
            'alpha': 1.5,
            'r0': shells[0]['distance'] if shells else a / np.sqrt(2),
        })

        try:
            D_real, _ = build_dynamical_matrix(
                positions, np.zeros(n_atoms, dtype=int), all_masses,
                box_length, potential, n_shells,
            )
            D_real = apply_acoustic_sum_rule(D_real, n_atoms)

            total_error = 0.0
            for ki in range(min(len(k_points), 8)):  # 限制 k 点数
                D_q = fourier_transform_dynamical_matrix(
                    D_real, positions, k_points[ki] * 2 * np.pi / a, box_length
                )
                omega_sq, _ = solve_phonon_eigenproblem(D_q)
                omega, _ = extract_phonon_frequencies(omega_sq)
                omega = np.maximum(omega, 0.0)

                # 与目标比较
                n_compare = min(len(omega), len(target_frequencies))
                if n_compare > 0:
                    total_error += np.sum((omega[:n_compare] - target_frequencies[:n_compare]) ** 2)

            # 正则化
            total_error += 0.01 * np.sum(alpha_vec ** 2)
        except Exception:
            total_error = 1e10

        return total_error

    # 初始猜测
    alpha0 = np.ones(n_shells) * 0.5

    result = coordinate_search_minimize(
        objective, alpha0, delta=0.5, tolerance=1e-4, max_feval=150,
    )

    return {
        'optimal_alphas': result['x_opt'],
        'final_error': result['f_opt'],
        'n_evaluations': result['n_feval'],
        'converged': result['converged'],
    }


def regularized_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    lambda_reg: float = 0.01,
) -> np.ndarray:
    """
    正则化最小二乘: min ||Ax - b||^2 + lambda * ||x||^2
    解: x = (A^T A + lambda I)^{-1} A^T b
    """
    n = A.shape[1]
    ATA = A.T @ A + lambda_reg * np.eye(n)
    ATb = A.T @ b
    return np.linalg.solve(ATA, ATb)


def bootstrap_uncertainty(
    fit_func: Callable,
    data: np.ndarray,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap 不确定性估计。

    对数据有放回重采样 n_bootstrap 次,
    每次拟合得到参数估计, 计算标准差作为不确定度。
    """
    rng = np.random.RandomState(seed)
    n_data = len(data)
    params_list = []

    for b in range(n_bootstrap):
        sample_idx = rng.choice(n_data, n_data, replace=True)
        sample_data = data[sample_idx]
        params = fit_func(sample_data)
        params_list.append(params)

    params_arr = np.array(params_list)
    mean_params = np.mean(params_arr, axis=0)
    std_params = np.std(params_arr, axis=0)

    return mean_params, std_params
