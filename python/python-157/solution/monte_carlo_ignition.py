"""
monte_carlo_ignition.py
点火概率蒙特卡洛采样与统计分析模块
融合来源：331_ellipse_monte_carlo（椭圆内采样与Cholesky分解）
           711_mandelbrot_area（逃逸迭代与区域判断）
           1092_snakes_and_ladders_simulation（批量统计与批次分析）
"""
import numpy as np
from combustion_utils import (
    check_positive, check_nonnegative, cholesky_factor, solve_lower_triangular,
    arrhenius_rate, R_UNIVERSAL, DEFAULT_E_A, DEFAULT_A_PRE, DEFAULT_T_IGN
)


def sample_ellipse(n, A_mat, r):
    r"""
    在椭圆 X' A X <= r^2 内均匀随机采样 n 个点。
    融合来源：331_ellipse_monte_carlo 的 ellipse_sample。

    算法:
        1. Cholesky 分解 A = L L^T
        2. 在单位圆盘内均匀采样 Y
        3. 解 L X = r * Y 得椭圆内点 X
    """
    check_positive(n, "n")
    check_positive(r, "r")
    A_mat = np.asarray(A_mat, dtype=float)
    if A_mat.shape != (2, 2):
        raise ValueError("A_mat must be 2x2")

    L = cholesky_factor(A_mat)

    # 在单位圆内均匀采样（拒绝采样）
    samples = []
    max_trials = n * 20
    trial = 0
    while len(samples) < n and trial < max_trials:
        trial += 1
        y = np.random.uniform(-1.0, 1.0, size=2)
        if np.dot(y, y) <= 1.0:
            samples.append(y)
    if len(samples) < n:
        # 若拒绝采样不足，补充均匀分布
        while len(samples) < n:
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            rad = np.sqrt(np.random.uniform(0.0, 1.0))
            samples.append([rad * np.cos(angle), rad * np.sin(angle)])

    Y = np.array(samples[:n]).T  # shape (2, n)
    Y = r * Y
    X = np.zeros_like(Y)
    for j in range(n):
        X[:, j] = solve_lower_triangular(L, Y[:, j])
    return X.T  # shape (n, 2)


def ignition_probability_monte_carlo(n_samples, T_mean, T_std, p_mean, p_std,
                                     phi_mean, phi_std, Ea=DEFAULT_E_A,
                                     A=DEFAULT_A_PRE, T_ign=DEFAULT_T_IGN,
                                     n_batches=5):
    r"""
    蒙特卡洛评估点火概率。

    物理模型:
        在湍流燃烧中，局部温度 T、压强 p、当量比 phi 存在涨落。
        当局部 Arrhenius 反应速率超过阈值时视为点火成功:
            k = A * exp(-Ea/(R*T))
            点火条件: k > k_ign = A * exp(-Ea/(R*T_ign))

    对 n_samples 个随机样本进行判断，分 n_batches 批次统计均值与方差。
    融合来源：1092_snakes_and_ladders_simulation 的批量统计思想。
    """
    check_positive(n_samples, "n_samples")
    check_positive(n_batches, "n_batches")

    k_ign = A * np.exp(-Ea / (R_UNIVERSAL * max(T_ign, 1.0e-6)))
    batch_size = n_samples // n_batches

    batch_probs = []
    for b in range(n_batches):
        count = 0
        for _ in range(batch_size):
            # 从截断正态分布采样（确保物理正值）
            T = max(np.random.normal(T_mean, T_std), 200.0)
            p = max(np.random.normal(p_mean, p_std), 1000.0)
            phi = max(np.random.normal(phi_mean, phi_std), 0.1)
            # 考虑当量比影响：富燃/贫燃降低反应速率
            phi_eff = np.exp(-0.5 * ((phi - 1.0) / 0.3) ** 2)
            k = A * np.exp(-Ea / (R_UNIVERSAL * T)) * phi_eff
            if k > k_ign:
                count += 1
        prob = count / batch_size
        batch_probs.append(prob)

    mean_prob = np.mean(batch_probs)
    std_prob = np.std(batch_probs, ddof=1)
    return mean_prob, std_prob, batch_probs


def critical_kernel_escape_time(n_grid, it_max, D_wave, gamma, Q,
                                rho0, p0, x_range=(-0.5, 0.5),
                                y_range=(-0.5, 0.5)):
    r"""
    临界核逃逸时间分析。

    物理背景：在爆轰波阵面附近，局部热点（hot spot）的临界尺寸
    可通过类似 Mandelbrot 逃逸时间的迭代来判断。
    将反应区映射到复平面，若热点温度迭代后超过临界值则认为
    "逃逸"（即成功点燃）。

    融合来源：711_mandelbrot_area 的逃逸迭代思想。

    返回:
        area_fraction: 成功点火区域占比
        avg_escape_time: 平均逃逸时间（迭代步数）
    """
    check_positive(n_grid, "n_grid")
    check_positive(it_max, "it_max")
    x_min, x_max = x_range
    y_min, y_max = y_range

    X = np.linspace(x_min, x_max, n_grid)
    Y = np.linspace(y_min, y_max, n_grid)
    escape_times = np.zeros((n_grid, n_grid), dtype=int)

    # 从 CJ 速度得到初始温度参考
    from combustion_utils import cj_detonation_velocity
    D_cj = cj_detonation_velocity(gamma, Q, p0, rho0)
    T_ref = p0 / (rho0 * (R_UNIVERSAL / 0.029))
    T_critical = 2.0 * T_ref

    for i in range(n_grid):
        for j in range(n_grid):
            # 初始"复数": z = x + i*y 映射到局部温度扰动
            z_real = X[i]
            z_imag = Y[j]
            T_local = T_ref + 50.0 * z_real
            # 迭代模拟反应放热导致的温度演化
            for it in range(1, it_max + 1):
                # 简化模型: T_{n+1} = T_n + dt * Q * k(T_n)
                k_val = DEFAULT_A_PRE * np.exp(-DEFAULT_E_A / (R_UNIVERSAL * max(T_local, 100.0)))
                T_local += 1.0e-7 * Q * k_val
                if T_local > T_critical:
                    escape_times[j, i] = it
                    break
            if escape_times[j, i] == 0:
                escape_times[j, i] = it_max + 1

    ignited = escape_times <= it_max
    area_fraction = np.sum(ignited) / (n_grid * n_grid)
    avg_escape_time = np.mean(escape_times[ignited]) if np.any(ignited) else it_max
    return area_fraction, avg_escape_time, escape_times
