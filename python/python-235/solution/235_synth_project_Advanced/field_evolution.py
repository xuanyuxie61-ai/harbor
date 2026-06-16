"""
field_evolution.py — 场时间演化模块
=====================================
种子项目映射:
  1206_EMIT_SIM (反应扩散/PDE) → 格点场的时间步进
  270_dfield9 (方向场/ODE) → 初值问题积分器

物理: 1+1D KG方程数值求解
  ∂²φ/∂t² = ∂²φ/∂x² - m²φ - λ/6 φ³ - μ³
  使用 Newmark-β (或 leapfrog) 时间积分 + 高阶空间差分
"""
import numpy as np
from finite_difference import fd_second_deriv


def create_gaussian_packet(x, x0, k0, sigma, amplitude=1.0):
    """
    高斯波包初始条件:
      φ(x) = A * exp(-(x-x0)²/(2σ²)) * cos(k0*(x-x0))
    """
    return amplitude * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.cos(k0 * (x - x0))


def compute_energy(phi, phi_dot, h, mass, lam, fd_order=4):
    """
    计算总能量 (Hamiltonian):
      E = Σ_i [½π² + ½(Dφ)² + ½m²φ² + λ/24 φ⁴] * h
    其中 π = ∂φ/∂t
    """
    D2phi = fd_second_deriv(phi, h, fd_order)
    # 动能密度
    kinetic = 0.5 * phi_dot**2
    # 梯度能 (使用 -φ*D²φ 近似 (Dφ)², 对周期性边界等价)
    gradient = -0.5 * phi * D2phi
    # 质量能
    mass_term = 0.5 * mass**2 * phi**2
    # 相互作用能
    interaction = lam / 24.0 * phi**4
    # 总能量密度积分
    energy_density = kinetic + gradient + mass_term + interaction
    return np.sum(energy_density) * h


def leapfrog_evolve(phi0, phi_dot0, N_x, h, dt, N_t, mass=1.0, lam=0.1,
                    mu3=0.0, fd_order=4):
    """
    Leapfrog (Störmer-Verlet) 时间演化.

    φ^{n+1} = 2φ^n - φ^{n-1} + dt² * [D²φ^n - m²φ^n - λ/6 (φ^n)³ - μ³]

    这是二阶辛积分器, 能量守恒性质好.
    """
    phi = phi0.copy()
    phi_prev = phi0 - dt * phi_dot0  # 初始半步

    energy_history = []
    phi_max_history = []

    for step in range(N_t):
        # 计算加速度
        D2phi = fd_second_deriv(phi, h, fd_order)
        accel = D2phi - mass**2 * phi - (lam / 6.0) * phi**3 - mu3

        # Leapfrog更新
        phi_next = 2.0 * phi - phi_prev + dt**2 * accel

        # 能量记录
        phi_dot = (phi_next - phi_prev) / (2.0 * dt)
        E = compute_energy(phi, phi_dot, h, mass, lam, fd_order)
        energy_history.append(E)
        phi_max_history.append(float(np.max(np.abs(phi))))

        # 更新
        phi_prev = phi.copy()
        phi = phi_next

    return phi, phi_dot, np.array(energy_history), np.array(phi_max_history)


def newmark_beta_evolve(phi0, phi_dot0, N_x, h, dt, N_t, mass=1.0, lam=0.1,
                        mu3=0.0, fd_order=4, beta=0.25, gamma=0.5):
    """
    Newmark-β 隐式时间演化.

    对于非线性问题, 每步需要 Newton-Raphson 迭代.
    β=0.25, γ=0.5 → 梯形规则 (无条件稳定, 二阶精度).

    隐式方程:
      φ^{n+1} = φ^n + dt*φ̇^n + dt²/2 * [(1-2β)*a^n + 2β*a^{n+1}]
      a^{n+1} = D²φ^{n+1} - m²φ^{n+1} - λ/6 (φ^{n+1})³ - μ³
    """
    from finite_difference import build_diff_matrix, lu_solve_no_pivot

    phi = phi0.copy()
    phi_dot = phi_dot0.copy()
    D2_mat = build_diff_matrix(N_x, h, fd_order)

    # 初始加速度
    accel = D2_mat @ phi - mass**2 * phi - (lam / 6.0) * phi**3 - mu3

    energy_history = []
    phi_max_history = []

    for step in range(N_t):
        # 预测值
        phi_pred = phi + dt * phi_dot + 0.5 * dt**2 * (1 - 2 * beta) * accel

        # Newton-Raphson 迭代
        phi_new = phi_pred.copy()
        for nr_iter in range(20):
            # 残差
            D2phi_new = D2_mat @ phi_new
            F_new = D2phi_new - mass**2 * phi_new - (lam / 6.0) * phi_new**3 - mu3
            residual = phi_new - phi_pred - beta * dt**2 * F_new

            # Jacobian
            J = np.eye(N_x) - beta * dt**2 * (D2_mat - mass**2 * np.eye(N_x)
                                                - 0.5 * lam * np.diag(phi_new**2))

            # 求解修正
            try:
                delta = lu_solve_no_pivot(J, -residual)
            except ValueError:
                # LU失败, 用伪逆
                delta = np.linalg.lstsq(J, -residual, rcond=None)[0]

            phi_new += delta
            if np.max(np.abs(delta)) < 1e-10:
                break

        # 更新速度和加速度
        phi_dot = phi_dot + dt * ((1 - gamma) * accel + gamma * (
            D2_mat @ phi_new - mass**2 * phi_new - (lam / 6.0) * phi_new**3 - mu3))
        accel = D2_mat @ phi_new - mass**2 * phi_new - (lam / 6.0) * phi_new**3 - mu3
        phi = phi_new

        E = compute_energy(phi, phi_dot, h, mass, lam, fd_order)
        energy_history.append(E)
        phi_max_history.append(float(np.max(np.abs(phi))))

    return phi, phi_dot, np.array(energy_history), np.array(phi_max_history)


def spectral_content(phi, k):
    """计算场的谱含量 |φ̃(k)|²."""
    phi_k = np.fft.fft(phi)
    return np.abs(phi_k)**2 / len(phi)
