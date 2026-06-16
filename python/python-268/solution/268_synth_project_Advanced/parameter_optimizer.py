"""
parameter_optimizer.py - Hubbard 模型参数的直接搜索优化
========================================================

科学背景 (Scientific Background):
    在研究 Hubbard 模型的量子相变时, 我们需要定位:
      - Mott 转变点: U_c(t, T)
      - 反铁磁量子临界点: (U_c, n_c)
      - 超导最佳掺杂: 最大化 T_c 的掺杂浓度

    这些临界参数通常通过坐标搜索 (coordinate search) 或
    更高级的优化算法确定.

    坐标搜索法 (融合 218_coordinate_search):
        沿每个坐标轴方向逐步搜索, 每次找到一个方向使目标函数
        下降后, 切换到下一个方向. 简单但鲁棒, 不需要梯度.

    目标函数可以是:
        - 双占据的导数 dD/dU (Mott 转变的指示)
        - 自旋结构因子的峰值 (反铁磁序)
        - 配对关联函数的最大值 (超导倾向)

融合种子项目:
    - 218_coordinate_search: 坐标直接搜索优化

核心公式 (Key Formulas):
    Mott 判据 (Brinkman-Rice):
        D(U) = (1 - U/U_c)² / 4  (Gutzwiller 近似)
        → U_c = 2 √(⟨ε²⟩ - ⟨ε⟩²) (动能方差)

    对半填充三角晶格 (DMFT):
        U_c2 ≈ 2.94 W  (W = 带宽 ≈ 12t 对三角晶格)
        U_c1 ≈ 2.22 W
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict


def coordinate_search(x0: np.ndarray,
                      func: Callable[[np.ndarray], float],
                      delta: float = 0.1,
                      tolerance: float = 1e-6,
                      max_iterations: int = 200,
                      verbose: bool = False) -> Dict[str, object]:
    """
    坐标直接搜索法 (Coordinate Direct Search).

    融合种子项目 218_coordinate_search.

    算法:
        1. 从初始点 x0 开始
        2. 对每个坐标方向 e_i:
            a. 试探 x0 ± δ e_i
            b. 若 f 下降, 移动到更优点
            c. 若所有方向都不下降, 缩小 δ → δ/2
        3. 重复直到 δ < tolerance 或达到最大迭代

    收敛性:
        对凸函数, 收敛到全局最小值.
        对非凸函数, 可能陷入局部最小值.

    物理应用:
        寻找 U_c 使得某个序参量 (如双占据导数) 最大.

    参数:
        x0: 初始参数向量
        func: 目标函数 f(x) → float (越小越好)
        delta: 初始步长
        tolerance: 收敛精度
        max_iterations: 最大迭代数
        verbose: 是否打印中间过程

    返回:
        'x_opt': 最优点
        'f_opt': 最优函数值
        'n_iterations': 迭代次数
        'n_evaluations': 函数调用次数
        'converged': 是否收敛
    """
    x = x0.copy()
    n_var = len(x)
    f_current = func(x)
    n_eval = 1
    iteration = 0
    converged = False

    # 坐标方向 (标准基)
    directions = np.eye(n_var)

    while iteration < max_iterations and delta > tolerance:
        improved = False

        for i in range(n_var):
            # 正向试探
            x_plus = x.copy()
            x_plus[i] += delta
            f_plus = func(x_plus)
            n_eval += 1

            # 负向试探
            x_minus = x.copy()
            x_minus[i] -= delta
            f_minus = func(x_minus)
            n_eval += 1

            # 选择最优点
            if f_plus < f_current and f_plus <= f_minus:
                x = x_plus
                f_current = f_plus
                improved = True
            elif f_minus < f_current:
                x = x_minus
                f_current = f_minus
                improved = True

            if verbose and improved:
                print(f"  iter={iteration}, dir={i}, f={f_current:.6e}, delta={delta:.4e}")

        if not improved:
            # 所有方向都无改善 → 缩小步长
            delta *= 0.5
        iteration += 1

    if delta <= tolerance:
        converged = True

    return {
        'x_opt': x,
        'f_opt': f_current,
        'n_iterations': iteration,
        'n_evaluations': n_eval,
        'converged': converged,
        'final_delta': delta,
    }


# ==========================================================================
#  Hubbard 模型特定的优化目标
# ==========================================================================

def mott_transition_objective(U: float, t: float = 1.0,
                              beta: float = 10.0,
                              n_sites: int = 4) -> float:
    """
    Mott 转变的目标函数.

    寻找 U_c 使得双占据 D(U) 的导数 |dD/dU| 最大.

    简化模型 (Gutzwiller 近似):
        D(U) = (1/4) (1 - (U/U_c)²)  for U < U_c
        D(U) = 0                       for U ≥ U_c

    实际 DQMC 计算中, D(U) 是连续的, 但 dD/dU 在 U_c 处有峰值.

    目标函数: -|dD/dU| (负号因为我们求最小值)
    """
    # 使用 Gutzwiller 近似作为代理模型
    U_c_approx = 8.0 * t  # 三角晶格的近似值
    if U < 0.01:
        U = 0.01
    if U < U_c_approx:
        D = 0.25 * max(0, 1.0 - (U / U_c_approx) ** 2)
        dD_dU = -0.5 * U / (U_c_approx ** 2)
    else:
        D = 0.0
        dD_dU = 0.0

    # 添加高斯噪声 (模拟 DQMC 统计误差)
    noise = 0.001 * np.exp(-0.5 * ((U - U_c_approx) / (0.5 * t)) ** 2)
    dD_dU += noise

    return -abs(dD_dU)  # 负号: 我们要最大化 |dD/dU|


def find_mott_uc(t: float = 1.0, beta: float = 10.0,
                 U_range: Tuple[float, float] = (0.5, 16.0)) -> float:
    """
    用坐标搜索寻找 Mott 转变点 U_c.

    参数:
        t: 跳跃积分
        beta: 逆温度
        U_range: 搜索范围

    返回:
        U_c: Mott 转变的临界 U 值
    """
    x0 = np.array([0.5 * (U_range[0] + U_range[1])])

    def objective(x):
        return mott_transition_objective(x[0], t, beta)

    result = coordinate_search(
        x0, objective,
        delta=2.0,
        tolerance=0.01,
        max_iterations=100
    )

    return result['x_opt'][0]


def optimize_cluster_size(target_condition: float = 1e8,
                          max_Ns: int = 100) -> int:
    """
    优化簇大小: 在条件数和系统大小间找平衡.

    目标: 最大化 Ns, 使得 κ(B_total) < target_condition.

    近似: κ ~ exp(β W), 与 Ns 弱相关 (对有限簇, κ 随 Ns 增大).
    """
    def neg_cluster_size(Ns_arr):
        Ns = int(round(Ns_arr[0]))
        if Ns < 2:
            return 1e6
        if Ns > max_Ns:
            return 1e6
        # 估计条件数: κ ~ Ns * exp(β * bandwidth / Ns)
        beta = 5.0
        bandwidth = 12.0  # 三角晶格
        kappa_est = Ns * np.exp(beta * bandwidth / max(Ns, 1))
        penalty = max(0, kappa_est - target_condition) ** 2
        return penalty - Ns * 0.01  # 鼓励大簇

    x0 = np.array([16.0])
    result = coordinate_search(
        x0, neg_cluster_size,
        delta=4.0,
        tolerance=0.5,
        max_iterations=50
    )
    return int(round(result['x_opt'][0]))


# ==========================================================================
#  参数空间扫描
# ==========================================================================

def scan_U_values(U_values: np.ndarray, t: float = 1.0,
                  beta: float = 10.0, n_sites: int = 4
                  ) -> Dict[str, np.ndarray]:
    """
    对一系列 U 值扫描物理量.

    返回:
        'U_values': U 值数组
        'energy': 总能量 vs U
        'double_occ': 双占据 vs U
        'compressibility': 压缩率 vs U
    """
    energies = []
    doub_occs = []
    compressibilities = []

    for U_val in U_values:
        # 使用 Gutzwiller 近似 + 微扰论
        U_c = 8.0 * t
        if U_val < U_c:
            D = 0.25 * max(0, 1.0 - (U_val / U_c) ** 2)
            E_kin = -2.0 * t * (1.0 - (U_val / U_c) ** 2) * n_sites
            E_pot = U_val * D * n_sites
        else:
            D = 0.0
            E_kin = -4.0 * t ** 2 / U_val * n_sites  # 超交换
            E_pot = 0.0

        E_total = E_kin + E_pot
        energies.append(E_total / n_sites)
        doub_occs.append(D)

        # 压缩率 κ = dn/dμ ≈ dD/dU (近似)
        if U_val < U_c:
            kappa = abs(-0.5 * U_val / (U_c ** 2))
        else:
            kappa = 0.0
        compressibilities.append(kappa)

    return {
        'U_values': U_values,
        'energy': np.array(energies),
        'double_occ': np.array(doub_occs),
        'compressibility': np.array(compressibilities),
    }
