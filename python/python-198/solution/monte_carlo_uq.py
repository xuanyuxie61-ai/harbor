"""
monte_carlo_uq.py
=================
蒙特卡洛不确定性量化验证模块（融合 331_ellipse_monte_carlo + 292_disk_distance）

功能：
- 椭圆域上的均匀随机采样（Cholesky变换）
- 圆盘域上的随机距离统计
- 对PCE结果进行蒙特卡洛对照验证
- 计算统计量：均值、方差、高阶矩

数学公式：
- 椭圆域: x' A x <= R², A 正定对称
  采样方法: 先采单位球 Y ~ Uniform(Ball(0,1))，再解 U X = R Y，其中 A = U'U
- 圆盘距离: d = ||P - Q||₂, P,Q ~ Uniform(Disk(0,1))
  E[d²] = 1, Var[d²] 可通过积分计算
- MC收敛率: RMSE ~ O(1/√N)
"""

import numpy as np


def uniform_in_unit_ball(m, n):
    """
    在单位球B(0,1)内均匀采样n个m维点。
    方法：方向均匀 + 半径^{1/m}分布。
    """
    # 方向：高斯归一化
    z = np.random.randn(m, n)
    norms = np.linalg.norm(z, axis=0)
    norms = np.clip(norms, 1e-15, None)
    directions = z / norms
    
    # 半径
    r = np.random.uniform(0.0, 1.0, n) ** (1.0 / m)
    points = directions * r
    return points


def ellipse_sample(n, A_mat, R):
    """
    在椭圆域 x' A x <= R² 内均匀采样n个点。
    融合 331_ellipse_monte_carlo 的核心算法：
    1. Cholesky分解 A = U' U
    2. 在单位球内采样 Y
    3. 解 U X = R Y
    
    参数:
        n: 采样点数
        A_mat: (2,2) 正定对称矩阵
        R: 半径参数
    
    返回:
        X: (2, n) 采样点
    """
    A_mat = np.asarray(A_mat, dtype=float)
    m = 2
    
    # Cholesky分解
    try:
        U = np.linalg.cholesky(A_mat).T  # A = U' U，U是上三角
    except np.linalg.LinAlgError:
        # 非正定，使用特征值修正
        eigvals, eigvecs = np.linalg.eigh(A_mat)
        eigvals = np.clip(eigvals, 1e-8, None)
        A_mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        U = np.linalg.cholesky(A_mat).T
    
    # 单位球采样
    Y = uniform_in_unit_ball(m, n) * R
    
    # 解 U X = Y  =>  X = U^{-1} Y
    X = np.linalg.solve(U, Y)
    return X


def monte_carlo_pce_verify(n_samples, pce_degree, alpha_mu, alpha_sigma,
                           u0_scalar, tf, exact_mean_func):
    """
    对PCE随机ODE进行蒙特卡洛验证。
    du/dt = -α(ξ) u, α(ξ) = α_μ + α_σ ξ
    
    精确解: u(t;ξ) = u0 * exp(-α(ξ) t)
    PCE近似: u_pce(t) = Σ u_k(t) He_k(ξ)
    
    返回MC统计量与PCE解析均值的对比。
    """
    # 生成随机样本
    xi = np.random.randn(n_samples)
    alpha = alpha_mu + alpha_sigma * xi
    
    # 精确解在时间tf处的值
    u_exact = u0_scalar * np.exp(-alpha * tf)
    
    mc_mean = np.mean(u_exact)
    mc_var = np.var(u_exact, ddof=1)
    
    # PCE解析均值（0阶系数）: u0 * exp(-α_μ t + 0.5 α_σ² t²)
    pce_mean_analytical = u0_scalar * np.exp(-alpha_mu * tf + 0.5 * alpha_sigma ** 2 * tf ** 2)
    
    error_mean = abs(mc_mean - pce_mean_analytical) / (abs(pce_mean_analytical) + 1e-15)
    
    return {
        'mc_mean': mc_mean,
        'mc_var': mc_var,
        'pce_mean_analytical': pce_mean_analytical,
        'error_mean': error_mean,
        'n_samples': n_samples
    }


def disk_distance_monte_carlo(n_samples=50000):
    """
    蒙特卡洛估计单位圆盘上两点距离统计量。
    融合 292_disk_distance 的核心算法。
    """
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples)
    r1 = np.sqrt(np.random.uniform(0, 1, n_samples))
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples)
    r2 = np.sqrt(np.random.uniform(0, 1, n_samples))
    
    p1 = np.column_stack((r1 * np.cos(theta1), r1 * np.sin(theta1)))
    p2 = np.column_stack((r2 * np.cos(theta2), r2 * np.sin(theta2)))
    
    d = np.linalg.norm(p1 - p2, axis=1)
    return {
        'mean': float(np.mean(d)),
        'variance': float(np.var(d, ddof=1)),
        'theoretical_mean': 128.0 / (45.0 * np.pi)
    }
