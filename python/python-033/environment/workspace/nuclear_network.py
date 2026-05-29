"""
nuclear_network.py
r 过程核反应网络刚性 ODE 求解器

核丰度演化方程（Bateman 方程）：
    dY(Z,A)/dt = sum_j N_{ji} lambda_j Y_j - sum_k N_{ik} lambda_k Y_i
               + rho*N_A * sum_{j,k} [N_{ji} (1+delta_{jk})/2!] <sv>_{jk->i} Y_j Y_k
               - rho*N_A * sum_j <sv>_{ij->*} Y_i Y_j

其中：
    Y_i = X_i / A_i 为核素 i 的摩尔丰度
    lambda 为衰变率 (s^{-1})
    <sv> 为热平均反应截面 (cm^3/s)
    rho 为物质密度 (g/cm^3)
    N_A 为阿伏伽德罗常数

对于 r 过程，主要反应通道：
    (n,gamma) : 中子捕获
    (gamma,n) : 光致分解
    beta^-    : beta 衰变（使 Z+1, N-1）
    alpha     : alpha 衰变（使 Z-2, A-4）
    fission   : 裂变（超重复元素）

本模块采用隐式 Euler + 循环矩阵预处理的混合方法求解刚性系统。
"""

import numpy as np
from circulant_solver import circulant_solve


def build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx=0):
    """
    构建 r 过程反应网络的雅可比矩阵和源项。

    对于小网络（<50 核素），直接构建完整矩阵。
    对于大网络，采用稀疏表示。

    参数:
        nuclides : list of tuple, [(Z,N,A), ...]
        rates : dict, 反应率表
        rho : float, 密度 (g/cm^3)
        n_n : float, 中子数密度 (cm^{-3})
        Y : ndarray, 当前丰度

    返回:
        J : ndarray, 雅可比矩阵 d(dY/dt)/dY
        S : ndarray, 源项
    """
    n_nuc = len(nuclides)
    J = np.zeros((n_nuc, n_nuc), dtype=float)
    S = np.zeros(n_nuc, dtype=float)

    N_A = 6.02214076e23

    for i, (z_i, n_i, a_i) in enumerate(nuclides):
        key_i = (z_i, a_i)

        # 流出项（对角线）
        outflow = 0.0

        # (n,gamma) 流出: Y_i -> Y_{i+1n}
        cap_rate = rates['capture'].get(key_i, 0.0)
        if np.ndim(cap_rate) == 0:
            cap = float(cap_rate)
        else:
            # TODO [Hole 2]: 使用 temp_idx 从温度依赖数组 cap_rate 中选取当前温度点的反应率
            # 需考虑 temp_idx 越界保护（clamp 到有效范围）
            cap = 0.0  # placeholder
        outflow += n_n * cap

        # (gamma,n) 流出
        phot_rate = rates['photodis'].get(key_i, 0.0)
        if np.ndim(phot_rate) == 0:
            phot = float(phot_rate)
        else:
            # TODO [Hole 2]: 使用 temp_idx 从温度依赖数组 phot_rate 中选取当前温度点的反应率
            # 需考虑 temp_idx 越界保护（clamp 到有效范围）
            phot = 0.0  # placeholder
        outflow += phot

        # beta 衰变流出
        beta = float(rates['beta'].get(key_i, 0.0))
        outflow += beta

        # alpha 衰变流出
        alpha = float(rates['alpha'].get(key_i, 0.0))
        outflow += alpha

        # 裂变流出
        fiss = float(rates['fission'].get(key_i, 0.0))
        outflow += fiss

        J[i, i] -= outflow

        # 流入项（非对角线）
        for j, (z_j, n_j, a_j) in enumerate(nuclides):
            if i == j:
                continue
            key_j = (z_j, a_j)

            inflow = 0.0
            # 检查 j -> i 的反应通道
            # beta 衰变: (Z+1, N-1, A) -> (Z, N, A)
            if z_j == z_i + 1 and n_j == n_i - 1 and a_j == a_i:
                inflow += float(rates['beta'].get(key_j, 0.0))

            # (n,gamma): (Z, N-1, A-1) -> (Z, N, A)
            if z_j == z_i and n_j == n_i - 1 and a_j == a_i - 1:
                cap_j = rates['capture'].get(key_j, 0.0)
                if np.ndim(cap_j) == 0:
                    cap_j = float(cap_j)
                else:
                    # TODO [Hole 2]: 使用 temp_idx 从温度依赖数组 cap_j 中选取当前温度点的反应率
                    cap_j = 0.0  # placeholder
                inflow += n_n * cap_j

            # (gamma,n): (Z, N+1, A+1) -> (Z, N, A)
            if z_j == z_i and n_j == n_i + 1 and a_j == a_i + 1:
                phot_j = rates['photodis'].get(key_j, 0.0)
                if np.ndim(phot_j) == 0:
                    phot_j = float(phot_j)
                else:
                    # TODO [Hole 2]: 使用 temp_idx 从温度依赖数组 phot_j 中选取当前温度点的反应率
                    phot_j = 0.0  # placeholder
                inflow += phot_j

            # alpha 衰变: (Z+2, N+2, A+4) -> (Z, N, A)
            if z_j == z_i + 2 and n_j == n_i + 2 and a_j == a_i + 4:
                inflow += float(rates['alpha'].get(key_j, 0.0))

            # 裂变: 简化处理，A>220 核素裂变产生中质量碎片
            if a_j > 220 and a_i < a_j and a_i > 80:
                fiss_j = float(rates['fission'].get(key_j, 0.0))
                # 裂变产物分布简化：平均分配
                inflow += fiss_j * 0.1  # 简化分支比

            if inflow > 0:
                J[i, j] += inflow

        # S 表示外部源项；在当前封闭网络中 S=0
        # 保留接口以便后续扩展中子源注入
        S[i] = 0.0

    return J, S


def solve_network_implicit_euler(nuclides, rates, rho, n_n, Y0, t_end, n_steps=1000, temp_profile=None):
    """
    使用隐式 Euler 方法求解核反应网络。

    隐式 Euler：
        (Y^{n+1} - Y^n) / dt = f(Y^{n+1})
    线性化：
        Y^{n+1} ≈ Y^n + dt * [f(Y^n) + J(Y^n) (Y^{n+1} - Y^n)]
        => (I - dt*J) Y^{n+1} = Y^n + dt*f(Y^n) - dt*J*Y^n
        => (I - dt*J) Y^{n+1} = Y^n + dt*S

    参数:
        nuclides : list of tuple
        rates : dict
        rho : float
        n_n : float
        Y0 : ndarray, 初始丰度
        t_end : float, 终止时间 (s)
        n_steps : int, 时间步数

    返回:
        t : ndarray, 时间数组
        Y_history : ndarray, shape (n_steps+1, n_nuc), 丰度历史
    """
    n_nuc = len(nuclides)
    Y = np.asarray(Y0, dtype=float).copy()
    dt = t_end / n_steps

    t_history = np.zeros(n_steps + 1)
    Y_history = np.zeros((n_steps + 1, n_nuc))
    t_history[0] = 0.0
    Y_history[0] = Y

    for step in range(n_steps):
        temp_idx = temp_profile[step] if temp_profile is not None else 0
        J, S = build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx)
        M = np.eye(n_nuc) - dt * J

        # 保证丰度非负
        rhs = Y + dt * S

        try:
            Y_new = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # 退化情况：使用最小二乘
            Y_new = np.linalg.lstsq(M, rhs, rcond=None)[0]

        # 数值鲁棒性：截断负值
        Y_new = np.maximum(Y_new, 0.0)
        # 归一化总丰度（摩尔丰度之和应为 1）
        total = np.sum(Y_new)
        if total > 0:
            Y_new = Y_new / total

        Y = Y_new
        t_history[step + 1] = (step + 1) * dt
        Y_history[step + 1] = Y

    return t_history, Y_history


def solve_network_bdf2(nuclides, rates, rho, n_n, Y0, t_end, n_steps=500, temp_profile=None):
    """
    使用二阶后向差分公式（BDF2）求解，精度更高。

    BDF2 公式：
        (3Y^{n+1} - 4Y^n + Y^{n-1}) / (2dt) = f(Y^{n+1})
        => (3I - 2dt*J) Y^{n+1} = 4Y^n - Y^{n-1} + 2dt*S

    参数同 solve_network_implicit_euler。
    """
    n_nuc = len(nuclides)
    Y = np.asarray(Y0, dtype=float).copy()
    dt = t_end / n_steps

    t_history = np.zeros(n_steps + 1)
    Y_history = np.zeros((n_steps + 1, n_nuc))
    t_history[0] = 0.0
    Y_history[0] = Y

    # 第一步用隐式 Euler
    temp_idx = temp_profile[0] if temp_profile is not None else 0
    J, S = build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx)
    M = np.eye(n_nuc) - dt * J
    rhs = Y + dt * S
    try:
        Y_prev = Y.copy()
        Y = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        Y_prev = Y.copy()
        Y = np.linalg.lstsq(M, rhs, rcond=None)[0]
    Y = np.maximum(Y, 0.0)
    total = np.sum(Y)
    if total > 0:
        Y = Y / total
    Y_history[1] = Y
    t_history[1] = dt

    for step in range(1, n_steps):
        # TODO [Hole 2]: 根据当前 step 和 temp_profile 计算 temp_idx
        # 若 temp_profile 为 None，默认 temp_idx=0；否则取 temp_profile[step]
        temp_idx = 0  # placeholder
        J, S = build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx)
        M = 3.0 * np.eye(n_nuc) - 2.0 * dt * J
        rhs = 4.0 * Y - Y_prev + 2.0 * dt * S

        try:
            Y_new = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            Y_new = np.linalg.lstsq(M, rhs, rcond=None)[0]

        Y_new = np.maximum(Y_new, 0.0)
        total = np.sum(Y_new)
        if total > 0:
            Y_new = Y_new / total

        Y_prev = Y.copy()
        Y = Y_new
        t_history[step + 1] = (step + 1) * dt
        Y_history[step + 1] = Y

    return t_history, Y_history


def compute_abundance_peaks(Y_final, nuclides, A_bins=None):
    """
    计算最终丰度分布的 r 过程峰。

    参数:
        Y_final : ndarray, 最终丰度
        nuclides : list of tuple
        A_bins : ndarray, 质量数分箱

    返回:
        A_centers : ndarray, 分箱中心
        abundances : ndarray, 每个 A 的累计丰度
    """
    if A_bins is None:
        A_bins = np.arange(70, 251, 5)
    A_centers = 0.5 * (A_bins[:-1] + A_bins[1:])
    abundances = np.zeros(len(A_centers))

    for i, (z, n, a) in enumerate(nuclides):
        idx = np.searchsorted(A_bins, a) - 1
        idx = np.clip(idx, 0, len(A_centers) - 1)
        abundances[idx] += Y_final[i]

    return A_centers, abundances


def test_nuclear_network():
    """自包含测试"""
    from reaction_rates import build_reaction_rate_table
    nuclides = [(26, 30, 56), (26, 31, 57), (27, 30, 57), (27, 31, 58), (28, 30, 58)]
    T9_range = np.array([1.0, 1.5, 2.0])
    S_n_table = {(26, 56): 8.0, (26, 57): 7.5, (27, 57): 8.2, (27, 58): 7.8, (28, 58): 8.5}
    T_half_table = {(26, 56): 1e10, (26, 57): 1.5, (27, 57): 272.0, (27, 58): 70.8, (28, 58): 1e10}

    rates = build_reaction_rate_table(nuclides, T9_range, S_n_table, T_half_table)
    Y0 = np.ones(len(nuclides)) / len(nuclides)
    t, Y_hist = solve_network_implicit_euler(nuclides, rates, rho=1e8, n_n=1e30,
                                              Y0=Y0, t_end=10.0, n_steps=100)
    print(f"[nuclear_network] Final abundances: {Y_hist[-1]}")
    print(f"[nuclear_network] Sum of abundances: {np.sum(Y_hist[-1]):.6f}")
    assert np.sum(Y_hist[-1]) > 0.99, "Total abundance conservation violated"


if __name__ == "__main__":
    test_nuclear_network()
