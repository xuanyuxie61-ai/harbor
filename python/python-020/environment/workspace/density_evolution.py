# -*- coding: utf-8 -*-
"""
density_evolution.py
密度矩阵的时间演化与反应-扩散动力学

核心物理：
  在开放量子系统中，密度矩阵 ρ(t) 满足Lindblad主方程：
      dρ/dt = -(i/ħ)[H, ρ] + Σ_k [ L_k ρ L_k^† - (1/2){L_k^† L_k, ρ} ]

  对于密度分布 n(r,t) = ⟨ψ^†(r)ψ(r)⟩，在平均场近似下退化为
  反应-扩散方程（融合Fisher-KPP方程）。

  Fisher-KPP方程：
      ∂n/∂t = D ∇²n + r n (1 - n/K)

  其中 D 为扩散系数，r 为增长率，K 为饱和密度（承载容量）。
  该方程描述密度扰动从初始不均匀状态向均匀稳态的演化。

  在量子霍尔系统中，密度演化还受磁场约束，涡旋的产生与湮灭
  类似于反应项：
      ∂n/∂t = D ∇²n + Γ(n) - R(n)

  其中 Γ(n) 为产生率，R(n) 为复合/湮灭率。

  对于波动方程（与Fisher方程耦合），采用有限差分法：
      c² U_{xx} = U_{tt}

  离散格式（CTCS）：
      U^{n+1}_j = α² U^n_{j+1} + 2(1-α²) U^n_j + α² U^n_{j-1} - U^{n-1}_j
    其中 α = c Δt / Δx，稳定性要求 |α| ≤ 1。

本模块融合原项目：
  - 366_fd1d_wave（一维波动方程有限差分）
  - 101_blowup_ode（爆炸ODE）
  - 433_fisher_exact（Fisher-KPP方程精确解）
"""
import numpy as np
from utils import safe_exp

# ============================================================================
# 1. 一维波动方程有限差分求解（融合原项目 366_fd1d_wave）
# ============================================================================

def fd1d_wave_solve(x_num, x1, x2, t_num, t1, t2, c, u_x1_fn, u_x2_fn,
                    u_t1_fn, ut_t1_fn):
    """
    使用有限差分法求解一维波动方程：
        c² U_{xx} = U_{tt}

    离散化（中心时间中心空间，CTCS）：
        U^{n+1}_j = α² U^n_{j+1} + 2(1-α²) U^n_j + α² U^n_{j-1} - U^{n-1}_j
    其中 α = c Δt / Δx。

    初始步（无 U^{n-1}）：
        U^{1}_j = (α²/2) U^0_{j+1} + (1-α²) U^0_j + (α²/2) U^0_{j-1}
                  + Δt · U_t^0_j

    稳定性条件：|α| ≤ 1。

    参数:
        x_num   : int, 空间区间数
        x1, x2  : float, 空间域 [x1, x2]
        t_num   : int, 时间步数
        t1, t2  : float, 时间域 [t1, t2]
        c       : float, 波速
        u_x1_fn : callable, 左边界 Dirichlet 条件 U(t, x1)
        u_x2_fn : callable, 右边界 Dirichlet 条件 U(t, x2)
        u_t1_fn : callable, 初始条件 U(t1, x)
        ut_t1_fn: callable, 初始速度 ∂U/∂t(t1, x)

    返回:
        u       : ndarray, shape (t_num+1, x_num+1)
    """
    if x_num < 2:
        raise ValueError("x_num 必须 ≥ 2")
    if t_num < 1:
        raise ValueError("t_num 必须 ≥ 1")

    dt = (t2 - t1) / t_num
    dx = (x2 - x1) / x_num
    alpha = c * dt / dx

    if abs(alpha) > 1.0:
        # 数值不稳定，减小时间步
        dt_new = dx / (abs(c) + 1e-10)
        t_num_new = int(np.ceil((t2 - t1) / dt_new))
        dt = (t2 - t1) / t_num_new
        alpha = c * dt / dx
        # 重新分配数组
        u = np.zeros((t_num_new + 1, x_num + 1))
        t_num = t_num_new
    else:
        u = np.zeros((t_num + 1, x_num + 1))

    x = np.linspace(x1, x2, x_num + 1)

    # 边界条件
    times = np.linspace(t1, t2, t_num + 1)
    for n in range(t_num + 1):
        u[n, 0] = u_x1_fn(times[n])
        u[n, x_num] = u_x2_fn(times[n])

    # 初始条件
    u[0, :] = u_t1_fn(x)
    ut0 = ut_t1_fn(x)

    # 第一时间步（使用初始速度）
    for j in range(1, x_num):
        u[1, j] = (0.5 * alpha ** 2) * u[0, j + 1] \
                  + (1.0 - alpha ** 2) * u[0, j] \
                  + (0.5 * alpha ** 2) * u[0, j - 1] \
                  + dt * ut0[j]

    # 后续时间步
    for n in range(1, t_num):
        for j in range(1, x_num):
            u[n + 1, j] = (alpha ** 2) * u[n, j + 1] \
                          + 2.0 * (1.0 - alpha ** 2) * u[n, j] \
                          + (alpha ** 2) * u[n, j - 1] \
                          - u[n - 1, j]

    return u


# ============================================================================
# 2. Fisher-KPP反应扩散方程（融合原项目 433_fisher_exact）
# ============================================================================

def fisher_kpp_exact_solution(t, x, a=1.0, c=2.0, k=-1.0 / np.sqrt(2.0)):
    """
    Fisher-KPP方程的行波精确解（融合原项目 433_fisher_exact）。

    Fisher方程：
        u_t = u_{xx} + u(1 - u)

    行波解形式：
        u(t, x) = 1 / [1 + a exp(k(x - ct))]²

    其中波速 c = 5/√6（对于特定参数），k = -1/√2。

    参数:
        t, x : float or ndarray, 时间和空间
        a    : float, 振幅参数
        c    : float, 波速
        k    : float, 波数

    返回:
        u    : ndarray, 解
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    z = x - c * t
    denom = 1.0 + a * np.exp(k * z)
    u = 1.0 / (denom ** 2)
    return u


def fisher_kpp_derivatives(t, x, a=1.0, c=2.0, k=-1.0 / np.sqrt(2.0)):
    """
    Fisher-KPP精确解的时间、空间一阶和二阶导数。

    u = [1 + a exp(kz)]^{-2}, z = x - ct

    u_t = 2ck · a exp(kz) / [1 + a exp(kz)]³
    u_x = -2k · a exp(kz) / [1 + a exp(kz)]³
    u_xx = 6k² · a² exp(2kz) / [1 + a exp(kz)]^4
           - 2k² · a exp(kz) / [1 + a exp(kz)]³
    """
    z = x - c * t
    exp_kz = np.exp(k * z)
    denom1 = 1.0 + a * exp_kz
    denom3 = denom1 ** 3
    denom4 = denom1 ** 4

    ut = 2.0 * c * k * a * exp_kz / denom3
    ux = -2.0 * k * a * exp_kz / denom3
    uxx = 6.0 * (k ** 2) * (a ** 2) * np.exp(2.0 * k * z) / denom4 \
          - 2.0 * (k ** 2) * a * exp_kz / denom3
    return ut, ux, uxx


def fisher_kpp_fd_solve(x_num, x1, x2, t_num, t1, t2, D=1.0, r=1.0, K=1.0,
                         u0_fn=None):
    """
    使用有限差分法求解Fisher-KPP方程：
        ∂u/∂t = D ∂²u/∂x² + r u (1 - u/K)

    空间离散（中心差分）：
        ∂²u/∂x² ≈ (u_{j+1} - 2u_j + u_{j-1}) / Δx²

    时间离散（向前Euler）：
        u^{n+1}_j = u^n_j + Δt [ D (u^n_{j+1} - 2u^n_j + u^n_{j-1}) / Δx²
                                + r u^n_j (1 - u^n_j/K) ]

    稳定性条件：D Δt / Δx² ≤ 0.5。

    参数:
        x_num, x1, x2 : 空间离散参数
        t_num, t1, t2 : 时间离散参数
        D             : float, 扩散系数
        r             : float, 增长率
        K             : float, 饱和密度
        u0_fn         : callable, 初始条件函数

    返回:
        u             : ndarray, (t_num+1, x_num+1)
    """
    if x_num < 2 or t_num < 1:
        raise ValueError("网格数不足")

    dx = (x2 - x1) / x_num
    dt = (t2 - t1) / t_num

    # 稳定性检查
    lambda_val = D * dt / (dx ** 2)
    if lambda_val > 0.5:
        # 自动调整时间步
        dt_new = 0.45 * dx ** 2 / (D + 1e-10)
        t_num_new = int(np.ceil((t2 - t1) / dt_new))
        dt = (t2 - t1) / t_num_new
        lambda_val = D * dt / (dx ** 2)
        u = np.zeros((t_num_new + 1, x_num + 1))
        t_num = t_num_new
    else:
        u = np.zeros((t_num + 1, x_num + 1))

    x = np.linspace(x1, x2, x_num + 1)

    # 初始条件
    if u0_fn is None:
        u[0, :] = np.exp(-x ** 2)
    else:
        u[0, :] = u0_fn(x)

    # 边界条件（Neumann零通量）
    for n in range(t_num):
        for j in range(1, x_num):
            diffusion = D * (u[n, j + 1] - 2.0 * u[n, j] + u[n, j - 1]) / (dx ** 2)
            reaction = r * u[n, j] * (1.0 - u[n, j] / K)
            u[n + 1, j] = u[n, j] + dt * (diffusion + reaction)
        # 零通量边界
        u[n + 1, 0] = u[n + 1, 1]
        u[n + 1, x_num] = u[n + 1, x_num - 1]

    return u


# ============================================================================
# 3. 密度矩阵的粗粒化演化（Lindblad平均场近似）
# ============================================================================

def density_matrix_evolution_lindblad(
    rho0, H, L_list, t_span, n_steps
):
    """
    在Lindblad主方程下的密度矩阵演化（简化离散化）。

    dρ/dt = -(i/ħ)[H, ρ] + Σ_k D[L_k](ρ)

    其中耗散超算符：
        D[L](ρ) = L ρ L^† - (1/2){L^† L, ρ}

    采用一阶Euler离散化：
        ρ(t+Δt) ≈ ρ(t) + Δt · dρ/dt

    参数:
        rho0    : ndarray, 初始密度矩阵
        H       : ndarray, 系统哈密顿量
        L_list  : list of ndarray, Lindblad算符列表
        t_span  : tuple, (t0, t_stop)
        n_steps : int, 时间步数

    返回:
        times   : ndarray, 时间数组
        rhos    : list, 每个时间步的密度矩阵
    """
    rho = np.asarray(rho0, dtype=complex)
    H = np.asarray(H, dtype=complex)
    t0, t_stop = t_span
    dt = (t_stop - t0) / n_steps

    times = np.linspace(t0, t_stop, n_steps + 1)
    rhos = [rho.copy()]

    for _ in range(n_steps):
        # 幺正演化项
        commutator = H @ rho - rho @ H
        d_rho = -1j * commutator

        # 耗散项
        for L in L_list:
            L = np.asarray(L, dtype=complex)
            L_dag = np.conj(L.T)
            d_rho += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)

        # Euler步进
        rho = rho + dt * d_rho

        # 确保密度矩阵的厄米性和迹为1
        rho = 0.5 * (rho + np.conj(rho.T))
        trace = np.trace(rho)
        if abs(trace) > 1e-14:
            rho = rho / trace

        # 半正定性修正（特征值截断）
        eigs, V = np.linalg.eigh(rho)
        eigs = np.maximum(eigs, 0.0)
        eigs = eigs / np.sum(eigs)
        rho = V @ np.diag(eigs) @ np.conj(V.T)

        rhos.append(rho.copy())

    return times, rhos


# ============================================================================
# 4. 测试接口
# ============================================================================
def test_density_evolution():
    """测试密度演化模块。"""
    print("=" * 60)
    print("[density_evolution.py] 密度演化测试")
    print("=" * 60)

    # 测试波动方程
    print("\n1. 一维波动方程测试 (c=1, 初始条件 sin(πx)):")
    def u_x1(t):
        return 0.0
    def u_x2(t):
        return 0.0
    def u_t1(x):
        return np.sin(np.pi * x)
    def ut_t1(x):
        return np.zeros_like(x)

    u = fd1d_wave_solve(50, 0.0, 1.0, 100, 0.0, 2.0, 1.0, u_x1, u_x2, u_t1, ut_t1)
    print(f"   解的形状: {u.shape}")
    print(f"   t=0 时最大振幅: {np.max(np.abs(u[0,:])):.6f}")
    print(f"   t=2 时最大振幅: {np.max(np.abs(u[-1,:])):.6f}")

    # 测试Fisher-KPP精确解
    print("\n2. Fisher-KPP精确解测试:")
    t_test, x_test = 0.5, 0.0
    u_exact = fisher_kpp_exact_solution(t_test, x_test)
    ut, ux, uxx = fisher_kpp_derivatives(t_test, x_test)
    # 验证 u_t = u_xx + u(1-u)
    lhs = ut
    rhs = uxx + u_exact * (1.0 - u_exact)
    print(f"   u(0.5, 0.0) = {u_exact:.6f}")
    print(f"   u_t = {lhs:.6f}")
    print(f"   u_xx + u(1-u) = {rhs:.6f}")
    print(f"   残差 = {abs(lhs - rhs):.2e}")

    # 测试Fisher-KPP数值解
    print("\n3. Fisher-KPP有限差分解测试:")
    u_fisher = fisher_kpp_fd_solve(80, -10.0, 10.0, 200, 0.0, 5.0, D=1.0, r=1.0, K=1.0)
    print(f"   解的形状: {u_fisher.shape}")
    print(f"   初始总密度: {np.sum(u_fisher[0,:]):.4f}")
    print(f"   最终总密度: {np.sum(u_fisher[-1,:]):.4f}")
    print(f"   最终最大密度: {np.max(u_fisher[-1,:]):.4f}")

    # 测试Lindblad演化
    print("\n4. Lindblad密度矩阵演化测试:")
    H = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    L = np.array([[0.0, 0.1], [0.0, 0.0]], dtype=complex)
    times, rhos = density_matrix_evolution_lindblad(rho0, H, [L], (0.0, 2.0), 100)
    print(f"   时间步数: {len(times)}")
    print(f"   初始 ρ_11: {rhos[0][0,0].real:.4f}")
    print(f"   最终 ρ_11: {rhos[-1][0,0].real:.4f}")
    print(f"   最终迹: {np.trace(rhos[-1]).real:.6f}")

    print("\n[density_evolution.py] 测试完成。\n")


if __name__ == "__main__":
    test_density_evolution()
