"""
bubble_shape_deformation.py
气泡形状变形与混沌破碎模型

核心物理模型:
1. 非球形气泡变形（椭圆映射）:
   气泡界面由椭圆矩阵 A 映射单位圆得到:
   X(θ) = V + R * A^{-1/2} * [cosθ, sinθ]^T
   条件数 κ(A) = λ_max / λ_min 描述变形程度。

2. 高阶模式扰动（Legendre 多项式）:
   r(θ, t) = R(t) * [1 + Σ_{n=2}^{N} a_n(t) P_n(cosθ)]
   其中 P_n 为 n 阶 Legendre 多项式。

3. 混沌破碎（迭代函数系统 IFS）:
   将 leaf_chaos 的仿射变换映射到气泡微团破碎:
   x_{k+1} = A_j * x_k + b_j,  j ∈ {0,1,2,3}
   用于模拟气泡崩溃后期的微团随机运动。

映射来源:
- 180_circle_map: 矩阵对单位圆的映射 → 气泡椭圆变形分析
- 655_leaf_chaos: 迭代函数系统 → 气泡破碎微团混沌运动
"""

import numpy as np
from numpy.linalg import eigvals, cond, inv
from scipy.linalg import sqrtm
from scipy.special import legendre


def ellipse_condition_number(A):
    """
    计算描述气泡变形的椭圆矩阵条件数。
    对应 180_circle_map 中对矩阵映射单位圆的分析。
    κ(A) = λ_max / λ_min，越大表示变形越严重。
    """
    eigenvals = eigvals(A)
    eigenvals = np.abs(eigenvals)
    eigenvals = np.maximum(eigenvals, 1e-15)
    return np.max(eigenvals) / np.min(eigenvals)


def bubble_ellipse_shape(V, A, R, num_points=100):
    """
    生成变形气泡的椭圆边界点。
    对应 circle_map_dots 的核心思想:
    矩阵 A 将单位圆映射为椭圆，描述气泡变形。

    参数:
        V: 中心位置 [2]
        A: 2x2 正定对称形状矩阵
        R: 等效半径
        num_points: 采样点数
    返回:
        points: 2 x num_points 的边界坐标
    """
    # 确保 A 正定
    A = 0.5 * (A + A.T)
    eigs = eigvals(A)
    if np.any(eigs <= 0):
        A = A + np.eye(2) * (1.1 * abs(np.min(eigs)) + 0.1)

    theta = np.linspace(0, 2 * np.pi, num_points)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])

    # A^{-1/2} 映射: 将单位圆映射为椭圆
    try:
        A_inv_sqrt = inv(sqrtm(A))
    except (np.linalg.LinAlgError, ValueError):
        A_inv_sqrt = np.eye(2)

    points = V[:, None] + R * (A_inv_sqrt @ unit_circle)
    return points


def deformation_velocity_potential(R, dRdt, a_n, da_ndt, theta, N_modes=4):
    """
    计算非球形气泡的速度势。
    在球坐标下，速度势展开为:
    φ(r,θ,t) = Σ_{n=0}^{∞} (R^{n+2}/((n+1)r^{n+1})) * d a_n/dt * P_n(cosθ)
    其中 a_0 = R, a_1 = 0（质心守恒）。

    参数:
        R: 等效半径
        dRdt: 径向速度
        a_n: 形状模式振幅 [N_modes]
        da_ndt: 模式速度 [N_modes]
        theta: 极角数组
        N_modes: 保留模式数
    """
    cos_theta = np.cos(theta)
    phi = np.zeros_like(theta)

    # n=0 项（球形膨胀/收缩）
    phi += (R**2 / np.maximum(np.abs(theta), 1e-15)) * dRdt * 1.0  # P_0 = 1
    # 这里简化处理，仅计算表面处 (r=R) 的势
    phi = R * dRdt

    for n in range(2, N_modes + 1):
        if n - 1 < len(a_n):
            Pn = legendre(n)(cos_theta)
            phi += (R / (n + 1)) * da_ndt[n - 1] * Pn

    return phi


def mode_amplitude_odes(t, y, R_eq, sigma, rho, mu, N_modes=4):
    """
    非球形模式振幅的 ODE 系统。
    对应 axon_ode 的 ODE 框架，迁移到气泡形状模式。

    第 n 阶模式的 Rayleigh-Taylor 不稳定性增长:
    d²a_n/dt² + 3(dR/dt/R) * da_n/dt - (n-1)(d²R/dt²/R) * a_n
    = -(n-1)σ/(ρR³) * (n+2)(n-1) * a_n - 2μ(n-1)(n+2)/(ρR²) * da_n/dt

    参数:
        y: [a_2, da_2/dt, a_3, da_3/dt, ..., R, dR/dt]
        R_eq: 平衡半径
    """
    N = N_modes - 1  # 从 n=2 开始
    a = np.zeros(N_modes + 1)
    dadt = np.zeros(N_modes + 1)

    for n in range(2, N_modes + 1):
        idx = 2 * (n - 2)
        a[n] = y[idx]
        dadt[n] = y[idx + 1]

    R = y[-2]
    dRdt = y[-1]

    dydt = np.zeros(len(y))

    # 径向运动（简化）
    d2Rdt2 = -safe_divide(2.0 * sigma, rho * R**2) - 1.5 * dRdt**2 / (R + 1e-15)
    dydt[-2] = dRdt
    dydt[-1] = d2Rdt2

    for n in range(2, N_modes + 1):
        idx = 2 * (n - 2)
        an = a[n]
        dan_dt = dadt[n]

        # 恢复力（表面张力）
        restoring = -(n - 1) * sigma / (rho * (R**3 + 1e-30)) * (n + 2) * (n - 1) * an
        # 阻尼（粘性）
        damping = -2.0 * mu * (n - 1) * (n + 2) / (rho * (R**2 + 1e-30)) * dan_dt
        # 耦合项
        coupling = (n - 1) * d2Rdt2 / (R + 1e-15) * an
        # 径向膨胀阻尼
        expansion = -3.0 * dRdt / (R + 1e-15) * dan_dt

        d2an_dt2 = restoring + damping + coupling + expansion

        dydt[idx] = dan_dt
        dydt[idx + 1] = d2an_dt2

    return dydt


def safe_divide(a, b, default=0.0):
    """安全除法"""
    b = np.asarray(b)
    a = np.asarray(a)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-30
    result[mask] = a[mask] / b[mask]
    result[~mask] = default
    return result


def chaotic_microfragmentation(num_points=5000, iterations=3000):
    """
    气泡破碎微团的混沌运动模拟。
    将 655_leaf_chaos 的 IFS 映射迁移到气泡微团破碎:
    每个仿射变换代表不同尺度涡旋对微团的输运作用。

    变换定义:
      A0 = [[0.80, 0.00], [0.00, 0.80]],  b0 = [0.10, 0.04]  → 大尺度涡旋拉伸
      A1 = [[0.50, 0.00], [0.00, 0.50]],  b1 = [0.25, 0.40]  → 中等尺度压缩
      A2 = [[0.355, -0.355], [0.355, 0.355]], b2 = [0.266, 0.078]  → 旋转剪切
      A3 = [[0.355, 0.355], [-0.355, 0.355]], b3 = [0.378, 0.434]  → 反向旋转

    参数:
        num_points: 追踪的微团数量
        iterations: 每个微团的迭代次数
    返回:
        x_final: 2 x num_points 的最终位置
        lyapunov: 平均 Lyapunov 指数估计
    """
    A = [
        np.array([[0.80, 0.00], [0.00, 0.80]]),
        np.array([[0.50, 0.00], [0.00, 0.50]]),
        np.array([[0.355, -0.355], [0.355, 0.355]]),
        np.array([[0.355, 0.355], [-0.355, 0.355]]),
    ]
    b = [
        np.array([0.10, 0.04]),
        np.array([0.25, 0.40]),
        np.array([0.266, 0.078]),
        np.array([0.378, 0.434]),
    ]

    x = np.random.rand(2, num_points)

    # Lyapunov 指数估计（前几个点）
    lyap_sum = 0.0
    lyap_count = 0
    delta = 1e-10

    for i in range(iterations):
        j = np.random.randint(0, 4, size=num_points)
        for k in range(num_points):
            Aj = A[j[k]]
            bj = b[j[k]]
            x[:, k] = Aj @ x[:, k] + bj

        # 前 100 步估计 Lyapunov 指数
        if i < 100 and num_points > 1:
            x_pert = x[:, 0].copy()
            x_pert[0] += delta
            j0 = np.random.randint(0, 4)
            x_pert = A[j0] @ x_pert + b[j0]
            dist = np.linalg.norm(x_pert - x[:, 0])
            if dist > 1e-30:
                lyap_sum += np.log(dist / delta)
                lyap_count += 1

    lyapunov = lyap_sum / max(lyap_count, 1)
    return x, lyapunov


def compute_deformation_tensor(points):
    """
    从离散边界点计算等效变形张量。
    通过最小二乘拟合椭圆 Ax² + Bxy + Cy² = 1。
    """
    x = points[0, :]
    y = points[1, :]
    # 构造最小二乘问题: [x², xy, y²] * [A, B, C] = 1
    M = np.vstack([x**2, x * y, y**2]).T
    rhs = np.ones(len(x))
    coeffs, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    A_mat = np.array([[coeffs[0], coeffs[1] / 2.0], [coeffs[1] / 2.0, coeffs[2]]])
    return A_mat


def fragmentation_dimension(x):
    """
    计算微团分布的计盒维数（Box-counting dimension）。
    D_box = lim_{ε→0} log(N(ε)) / log(1/ε)
    """
    mins = np.min(x, axis=1)
    maxs = np.max(x, axis=1)
    L = np.max(maxs - mins)
    if L < 1e-15:
        return 0.0

    epsilons = L / (2.0 ** np.arange(1, 8))
    N_boxes = []
    for eps in epsilons:
        nx = int(np.ceil((maxs[0] - mins[0]) / eps)) + 1
        ny = int(np.ceil((maxs[1] - mins[1]) / eps)) + 1
        boxes = set()
        for k in range(x.shape[1]):
            ix = int((x[0, k] - mins[0]) / eps)
            iy = int((x[1, k] - mins[1]) / eps)
            boxes.add((ix, iy))
        N_boxes.append(len(boxes))

    log_eps = np.log(1.0 / epsilons)
    log_N = np.log(np.maximum(N_boxes, 1))
    # 线性拟合求斜率
    D_box = np.polyfit(log_eps, log_N, 1)[0]
    return max(0.0, min(D_box, 2.0))
