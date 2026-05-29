"""
radiative_transfer_solver.py
辐射传输方程求解器

整合原项目:
  - 1099_sor: 逐次超松弛迭代 (Successive Over-Relaxation)

功能:
  使用 SOR 方法求解一维平面平行大气中的辐射传输方程。
  考虑多次散射、吸收和发射过程。

核心方程 (辐射传输方程, RTE):
  μ dI(τ, μ) / dτ = I(τ, μ) - J(τ, μ)

其中源函数 J 包含:
  J(τ, μ) = (ω/2) ∫_{-1}^{1} P(μ, μ') I(τ, μ') dμ' + (1-ω) B(τ)

离散化后形成线性系统 A·I = b，使用 SOR 迭代求解:
  x_i^{new} = (1-ω_SOR) x_i + ω_SOR / A_{ii} * (b_i - Σ_{j<i} A_{ij} x_j^{new} - Σ_{j>i} A_{ij} x_j)
"""

import numpy as np
from math import sqrt, pi, exp


class RTESolverError(Exception):
    pass


def build_rte_matrix(num_depth, num_angle, tau_total, omega, g, B=None):
    """
    构建离散化辐射传输方程的线性系统 A·I = b。

    离散化策略:
      - 深度方向: 均匀光学厚度分层，Δτ = τ_total / (N_τ - 1)
      - 角度方向: 高斯-勒让德节点 μ_j 和权重 w_j
      - 散射相函数: Henyey-Greenstein 近似
      - 边界条件: 层顶入射 I_top=1，层底出射 I_bottom=0

    参数:
      num_depth: 深度层数
      num_angle: 角度方向离散数 (Gauss-Legendre 阶数)
      tau_total: 总光学厚度
      omega: 单次散射反照率
      g: 不对称因子
      B: 普朗克函数数组 (num_depth,)，若 None 则假设无热发射

    返回:
      A: 系统矩阵 (num_depth*num_angle, num_depth*num_angle)
      b: 右端项
      mu: 角度节点
      w: 角度权重
    """
    if not (0.0 <= omega <= 1.0):
        raise RTESolverError("build_rte_matrix: ω 必须在 [0,1]")
    if not (-1.0 < g < 1.0):
        raise RTESolverError("build_rte_matrix: g 必须在 (-1,1)")

    N = num_depth * num_angle
    # [HOLE 1] 根据辐射传输离散化策略定义层间距 dtau
    dtau = None

    # Gauss-Legendre 节点和权重
    mu, w = _gauss_legendre_nodes_weights(num_angle)

    A = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    if B is None:
        B = np.zeros(num_depth, dtype=np.float64)

    def idx(t, a):
        # [HOLE 1] 实现 (depth_layer, angle) → flat_index 映射
        pass

    for t in range(num_depth):
        for a in range(num_angle):
            i = idx(t, a)
            mu_a = mu[a]

            # 源函数中的散射积分权重项
            scatter_diag = 0.0
            for aj in range(num_angle):
                p_val = _hg_phase_function(mu_a, mu[aj], g)
                scatter_diag += 0.5 * omega * w[aj] * p_val

            if mu_a > 0:
                # 上行辐射，从层底向层顶传播 (μ > 0)
                # 注意: 在光学厚度坐标中，τ=0 为层顶，τ=τ_total 为层底
                # μ > 0 表示向上传播 (从层底到层顶)，即 τ 减小的方向
                if t == num_depth - 1:
                    # 底层边界: 入射 (从下方来) = 0 (无向上入射)
                    A[i, i] = 1.0
                    b[i] = 0.0
                else:
                    # 内点: μ (I_{t+1} - I_t) / Δτ = I_t - J_t
                    # 注意: t 从 0(层顶) 到 N-1(层底)
                    # μ>0 向上: t 增加是向下，所以 I_t 的 upstream 是 I_{t+1}
                    coeff = mu_a / dtau
                    A[i, i] = coeff + 1.0
                    A[i, idx(t + 1, a)] = -coeff
                    # 减去散射积分耦合项
                    for aj in range(num_angle):
                        p_val = _hg_phase_function(mu_a, mu[aj], g)
                        A[i, idx(t, aj)] -= 0.5 * omega * w[aj] * p_val
                    b[i] = (1.0 - omega) * B[t]
            else:
                # 下行辐射 (μ < 0)，从层顶向层底传播
                if t == 0:
                    # 顶层边界: 入射辐射 = 1.0
                    A[i, i] = 1.0
                    b[i] = 1.0
                else:
                    coeff = -mu_a / dtau  # |μ| / Δτ
                    A[i, i] = coeff + 1.0
                    A[i, idx(t - 1, a)] = -coeff
                    for aj in range(num_angle):
                        p_val = _hg_phase_function(mu_a, mu[aj], g)
                        A[i, idx(t, aj)] -= 0.5 * omega * w[aj] * p_val
                    b[i] = (1.0 - omega) * B[t]

    return A, b, mu, w


def _gauss_legendre_nodes_weights(n):
    """
    近似生成 Gauss-Legendre 节点和权重 (简化版，n<=8)。
    """
    if n == 2:
        x = np.array([-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)])
        w = np.array([1.0, 1.0])
    elif n == 4:
        x = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])
        w = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
    elif n == 6:
        x = np.array([-0.9324695142, -0.6612093865, -0.2386191861,
                      0.2386191861, 0.6612093865, 0.9324695142])
        w = np.array([0.1713244924, 0.3607615730, 0.4679139346,
                      0.4679139346, 0.3607615730, 0.1713244924])
    elif n == 8:
        x = np.array([-0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
                      0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565])
        w = np.array([0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
                      0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363])
    else:
        # 回退: 均匀节点 (精度较低但稳健)
        x = np.linspace(-1.0 + 1.0 / n, 1.0 - 1.0 / n, n)
        w = np.full(n, 2.0 / n)
    return x, w


def _hg_phase_function(mu1, mu2, g):
    """
    Henyey-Greenstein 相函数在 μ1 和 μ2 之间的值。
    近似使用 cos Θ = μ1 μ2 (简化，忽略方位角耦合)。
    """
    cos_theta = mu1 * mu2
    denom = (1.0 + g ** 2 - 2.0 * g * cos_theta) ** 1.5
    if denom < 1e-15:
        denom = 1e-15
    return (1.0 - g ** 2) / denom


def sor_solve(A, b, x0=None, omega_sor=1.5, tol=1e-10, max_iter=2000):
    """
    逐次超松弛 (SOR) 方法求解 A x = b。

    迭代公式:
      x_i^{new} = (1 - ω) x_i + ω / A_{ii} * (b_i - Σ_{j≠i} A_{ij} x_j)

    其中对于 j < i 使用已更新的 x_j^{new}，对于 j > i 使用旧的 x_j。

    参数:
      A: (n, n) 系数矩阵
      b: (n,) 右端项
      x0: 初始猜测
      omega_sor: 松弛因子 (0 < ω < 2)
      tol: 残差容差
      max_iter: 最大迭代次数

    返回:
      x: 解向量
      it: 实际迭代次数
      residual: 最终残差范数
    """
    n = A.shape[0]
    if A.shape[1] != n or b.shape[0] != n:
        raise RTESolverError("sor_solve: 维度不匹配")
    if not (0.0 < omega_sor < 2.0):
        raise RTESolverError("sor_solve: ω_SOR 必须在 (0,2)")

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64, copy=True)

    for it in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j == i:
                    continue
                if j < i:
                    sigma += A[i, j] * x_new[j]
                else:
                    sigma += A[i, j] * x[j]
            if abs(A[i, i]) < 1e-15:
                raise RTESolverError(f"sor_solve: A[{i},{i}] 为零，无法迭代")
            x_new[i] = (1.0 - omega_sor) * x[i] + omega_sor * (b[i] - sigma) / A[i, i]

        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            residual = np.linalg.norm(A @ x - b, ord=np.inf)
            return x, it, residual

        # 发散检测
        if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.linalg.norm(x, ord=np.inf) > 1e30:
            residual = np.linalg.norm(A @ x - b, ord=np.inf)
            return x, it, residual

    residual = np.linalg.norm(A @ x - b, ord=np.inf)
    return x, max_iter, residual


def compute_radiative_flux(I, mu, w):
    """
    由辐射强度场计算辐射通量。

    公式:
      F = 2π ∫_{-1}^{1} I(μ) μ dμ
    """
    integrand = I * mu
    return 2.0 * pi * np.trapezoid(integrand, mu)


def compute_heating_rate(F_up, F_down, dtau, rho_cp):
    """
    计算大气加热率 (K/day)。

    公式:
      dT/dt = - (g / c_p) * dF_net / dp
            = (1 / (ρ c_p)) * dF_net / dz
    近似:
      dF_net/dτ = (F_up - F_down) / Δτ
      dT/dt = dF_net / (ρ c_p Δz) = dF_net / dtau * (dτ/dz) / (ρ c_p)
    """
    dF_net = F_up - F_down
    # 简化: 加热率正比于净通量散度
    heating = dF_net / dtau / rho_cp
    # 转换为 K/day (假设典型值)
    heating_k_day = heating * 86400.0
    return heating_k_day
