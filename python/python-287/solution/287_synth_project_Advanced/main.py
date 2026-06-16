# -*- coding: utf-8 -*-
"""
main.py
=======

PROJECT 287: 计算等离子体 — MHD 不稳定性数值模拟
=================================================
高阶有限差分与稳定性分析 (小规模可复现实验)

统一入口, 零参数可运行.

本程序模拟 Harris 电流片中的 resistive 撕裂模 (tearing mode) 不稳定性,
综合运用以下数值方法:
  - 高阶 Lagrange 有限差分算子 (Fornberg 算法)
  - ADI 时间分裂 (Douglas-Gunn)
  - FFT 谱分析与 Hankel 矩阵 Prony 方法
  - Cholesky 分解用于 Poisson 求解
  - RK4 / velocity-Verlet 时间积分
  - MCMC 不确定性量化
  - Markov 链模式耦合模型
  - 相位-频率振荡分析
  - 3D 磁场重构 (encoder-decoder 上采样)

科学问题:
  Harris 电流片在 resistive MHD 框架下的撕裂模不稳定性
  是磁约束聚变等离子体中磁岛形成和磁重联的基本机制.
  本程序通过线性稳定性分析和非线性初值模拟, 验证 FKR 标度律,
  并量化等离子体参数不确定性对增长率的影响.

作者: PROJECT 287 合成
"""

from __future__ import annotations
import sys
import time as walltime
import numpy as np

# ================================================================
#  导入合成模块
# ================================================================
import plasma_constants as pc
from mesh_generator import StructuredMesh2D, extend_to_3d
from high_order_fd import (
    fornberg_weights,
    build_first_derivative_matrix,
    build_second_derivative_matrix,
    lagrange_interpolant_value,
)
from mhd_physics import (
    MHDUnits,
    harris_equilibrium_1d,
    tearing_mode_perturbation,
    compute_delta_prime,
)
from time_integrator import (
    rk4_step,
    rk4_integrate,
    ADISolver,
    velocity_verlet_step,
    cfl_time_step,
    diffusion_time_step,
    solve_tridiagonal,
)
from spectral_tools import (
    complex_fft,
    power_spectrum,
    prony_analysis,
    hankel_matrix,
    hankel_inverse,
    autocorrelation,
)
from matrix_solvers import (
    cholesky_factor,
    cholesky_solve,
    laplacian_2d_sparse,
    eigenvalues_generalized,
)
from stability_analyzer import (
    solve_tearing_eigenvalues,
    compute_delta_w,
    compute_tearing_delta_prime,
    fkr_scaling_test,
)
from oscillation_analyzer import (
    hilbert_transform,
    instantaneous_amplitude_phase,
    instantaneous_frequency,
    instantaneous_growth_rate,
    modulation_index,
    detect_oscillation_bursts,
)
from monte_carlo_uq import (
    MetropolisHastings,
    log_posterior_tearing,
    ModeCouplingMarkov,
    summarize_chain,
)
from field_reconstructor import (
    multiscale_upsample,
    reconstruct_3d_field,
    check_divergence_free,
    magnetic_energy,
)
from plasma_diagnostics import (
    magnetic_island_width,
    reconnection_rate,
    total_energy,
    safety_factor_profile,
    find_rational_surface,
    current_sheet_thickness,
    diagnostic_summary,
)


# ================================================================
#  输出分隔线
# ================================================================
def header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subheader(title: str) -> None:
    print()
    print(f"--- {title} ---")


# ================================================================
#  第 1 部分: 等离子体参数
# ================================================================
def part1_parameters() -> None:
    header("第 1 部分: 等离子体参数与无量纲数")
    print(pc.summary())

    params = pc.get_dimensionless_params()
    print(f"\n  无量纲参数:")
    for key, val in params.items():
        print(f"    {key:20s} = {val:.4e}")

    eta_hat = MHDUnits.to_dimensionless_eta()
    S = MHDUnits.lundquist_number()
    print(f"\n  归一化电阻率 eta_hat = {eta_hat:.4e}")
    print(f"  Lundquist 数 S      = {S:.4e}")

    gamma_fkr = pc.fkr_growth_rate(delta_prime=2.0)
    print(f"\n  FKR 增长率 (Delta'=2): gamma*tau_R = {gamma_fkr:.4e}")

    lam = pc.copling_parameter()
    print(f"  耦合参数 Lambda     = {lam:.4e}")


# ================================================================
#  第 2 部分: 网格生成
# ================================================================
def part2_mesh() -> StructuredMesh2D:
    header("第 2 部分: 结构化网格生成 (Harris 电流片)")

    nx, ny = 32, 48
    lx = 2.0 * np.pi * pc.L_CS  # 对应 k = 1/L_cs
    ly = 4.0 * pc.L_CS
    alpha = 2.5

    mesh = StructuredMesh2D(nx, ny, lx, ly, alpha_stretch=alpha)
    print(mesh.summary())

    # Harris 平衡场
    bx0_2d = mesh.harris_bx()
    jz0_2d = mesh.harris_jz()
    print(f"\n  Harris 平衡:")
    print(f"    max |Bx| = {np.max(np.abs(bx0_2d)):.4e} T")
    print(f"    max |Jz| = {np.max(np.abs(jz0_2d)):.4e} A/m^2")

    return mesh


# ================================================================
#  第 3 部分: 高阶有限差分验证
# ================================================================
def part3_fd_validation(mesh: StructuredMesh2D) -> None:
    header("第 3 部分: 高阶 Lagrange 有限差分验证")

    # 在 y 方向 (非均匀) 测试 Fornberg 权重
    y = mesh.y
    n_test = min(9, y.size)
    print(f"\n  Fornberg 权重测试 (n={n_test}, 求 2 阶导数):")

    nodes = y[:n_test]
    x_eval = y[n_test // 2]
    w = fornberg_weights(nodes, x_eval, max_deriv=2)
    print(f"    节点: {nodes[:5]} ... (共 {n_test} 个)")
    print(f"    求导点: x_eval = {x_eval:.6f}")
    print(f"    0 阶权重之和 = {np.sum(w[0]):.10f} (应为 1)")
    print(f"    1 阶权重之和 = {np.sum(w[1]):.2e} (应接近 0)")
    print(f"    2 阶权重之和 = {np.sum(w[2]):.6f}")

    # 测试: 对 f(y) = sin(k y) 求二阶导数
    # 解析: f'' = -k^2 sin(k y)
    k_test = 2.0 * np.pi / (2.0 * mesh.ly)
    f = np.sin(k_test * y)
    f_exact_dd = -k_test ** 2 * np.sin(k_test * y)

    D2 = build_second_derivative_matrix(y, order=3, bc="dirichlet")
    f_num_dd = D2 @ f

    # 内部点误差
    err = np.max(np.abs(f_num_dd[2:-2] - f_exact_dd[2:-2]))
    print(f"\n  二阶导数验证: f(y) = sin(k y), k = {k_test:.4f}")
    print(f"    最大误差 (内部点) = {err:.4e}")

    # Lagrange 插值测试
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.sin(x_data)
    x_test = 1.5
    y_interp = lagrange_interpolant_value(x_data, y_data, x_test)
    y_exact = np.sin(x_test)
    print(f"\n  Lagrange 插值测试: f(x) = sin(x), x = {x_test}")
    print(f"    插值值 = {y_interp:.10f}")
    print(f"    精确值 = {y_exact:.10f}")
    print(f"    误差   = {abs(y_interp - y_exact):.4e}")


# ================================================================
#  第 4 部分: Harris 平衡与撕裂模扰动
# ================================================================
def part4_equilibrium(mesh: StructuredMesh2D) -> dict:
    header("第 4 部分: Harris 电流片平衡与撕裂模扰动")

    y = mesh.y
    bx0, jz0, p0 = harris_equilibrium_1d(y)

    print(f"\n  Harris 平衡 (1D):")
    print(f"    y 范围: [{y[0]:.4e}, {y[-1]:.4e}]")
    print(f"    max |Bx| = {np.max(np.abs(bx0)):.4e}")
    print(f"    max |Jz| = {np.max(np.abs(jz0)):.4e}")
    print(f"    min p    = {np.min(p0):.4e}")

    # 撕裂模扰动
    k_mode = 2.0 * np.pi / mesh.lx
    psi, bx_pert, by_pert = tearing_mode_perturbation(
        mesh.x, mesh.y, k_mode, psi_amp=1.0e-3
    )
    print(f"\n  撕裂模扰动:")
    print(f"    k_mode  = {k_mode:.4e}")
    print(f"    max |psi|    = {np.max(np.abs(psi)):.4e}")
    print(f"    max |Bx_pert| = {np.max(np.abs(bx_pert)):.4e}")

    # Delta' 计算
    # 沿 y=0 处取 psi 的 x 平均
    psi_y_profile = np.mean(psi, axis=1)
    dp = compute_delta_prime(y, psi_y_profile, k_mode)
    print(f"\n  撕裂模稳定性参数:")
    print(f"    Delta' (数值) = {dp:.4e}")
    dp_analytic = 2.0 * (1.0 - (k_mode * pc.L_CS) ** 2) / (k_mode * pc.L_CS)
    print(f"    Delta' (解析) = {dp_analytic:.4e}")

    return {
        "bx0": bx0,
        "jz0": jz0,
        "p0": p0,
        "k_mode": k_mode,
        "psi": psi,
        "bx_pert": bx_pert,
        "by_pert": by_pert,
    }


# ================================================================
#  第 5 部分: 线性稳定性分析
# ================================================================
def part5_stability(mesh: StructuredMesh2D, eq: dict) -> None:
    header("第 5 部分: 线性稳定性分析 (本征值问题)")

    y = mesh.y
    bx0 = eq["bx0"]
    k_mode = eq["k_mode"]

    eta_hat = MHDUnits.to_dimensionless_eta()
    print(f"\n  参数: eta_hat = {eta_hat:.4e}, k = {k_mode:.4e}")

    # 本征值求解
    evals, evecs = solve_tearing_eigenvalues(
        y, k_mode, bx0, eta_hat, n_modes=8, order=4
    )

    print(f"\n  前 8 个本征值 (按 Re(gamma) 降序):")
    print(f"    {'n':>3s}  {'Re(gamma)':>14s}  {'Im(gamma)':>14s}  {'|gamma|':>14s}")
    for n in range(min(8, len(evals))):
        g = evals[n]
        print(f"    {n:3d}  {np.real(g):14.6e}  {np.imag(g):14.6e}  {abs(g):14.6e}")

    gamma_max = np.real(evals[0])
    omega_max = np.imag(evals[0])
    print(f"\n  最不稳定模式:")
    print(f"    增长率 gamma = {gamma_max:.6e}")
    print(f"    频率   omega = {omega_max:.6e}")

    # 外部理想 MHD: δW
    xi_test = np.sin(np.pi * (y - y[0]) / (y[-1] - y[0]))
    delta_w = compute_delta_w(y, xi_test, bx0, eq["p0"], k_mode)
    print(f"\n  理想 MHD 能量原理 δW = {delta_w:.6e}")
    if delta_w > 0:
        print(f"    => 理想 MHD 稳定 (δW > 0)")
    else:
        print(f"    => 理想 MHD 不稳定 (δW < 0)")

    # Delta' (从本征函数)
    dp_num = compute_tearing_delta_prime(y, np.real(evecs[:, 0]))
    print(f"\n  从本征函数计算的 Delta' = {dp_num:.4e}")


# ================================================================
#  第 6 部分: ADI 时间演化
# ================================================================
def part6_time_evolution(mesh: StructuredMesh2D, eq: dict) -> dict:
    header("第 6 部分: ADI 时间演化 (磁通量扩散)")

    eta_hat = MHDUnits.to_dimensionless_eta()
    k_mode = eq["k_mode"]
    bx0 = eq["bx0"]

    # ADI 求解器
    adi = ADISolver(
        nx=mesh.nx,
        ny=mesh.ny,
        dx=mesh.dx,
        dy=mesh.dy,
        eta=eta_hat,
        bc_x="periodic",
        bc_y="dirichlet",
    )

    # 初始条件: Harris + 撕裂模扰动
    psi_init = eq["psi"].copy()

    # 时间步长
    dt_cfl = cfl_time_step(mesh.min_cell_size(), pc.V_ALFVEN, cfl_number=0.3)
    dt_diff = diffusion_time_step(mesh.min_cell_size(), eta_hat, safety=0.3)
    dt = min(dt_cfl, dt_diff)
    dt = min(dt, 0.1 * pc.TAU_ALFVEN)  # 限制最大步长

    print(f"\n  时间步长:")
    print(f"    dt_CFL  = {dt_cfl:.4e}")
    print(f"    dt_diff = {dt_diff:.4e}")
    print(f"    dt_used = {dt:.4e}")

    # 时间演化
    n_steps = 50
    psi = psi_init.copy()
    psi_history = [psi.copy()]
    time_arr = [0.0]
    t = 0.0

    print(f"\n  ADI 时间演化 ({n_steps} 步):")
    print(f"    {'step':>5s}  {'time':>12s}  {'max|psi|':>14s}  {'energy':>14s}")

    for step in range(n_steps):
        psi = adi.step(psi, dt)
        t += dt
        max_psi = np.max(np.abs(psi))

        # 磁能 (正比于 |∇ψ|²) — 使用内部点, 与 area_element 尺寸匹配
        dpsi_dy = np.gradient(psi, mesh.y, axis=0)
        dpsi_dx = np.gradient(psi, mesh.x, axis=1)
        # area_element 为 (ny-1, nx-1), 取单元中心梯度
        grad_cell = (
            0.25 * (
                dpsi_dy[:-1, :-1] ** 2 + dpsi_dy[:-1, 1:] ** 2
                + dpsi_dy[1:, :-1] ** 2 + dpsi_dy[1:, 1:] ** 2
            )
            + 0.25 * (
                dpsi_dx[:-1, :-1] ** 2 + dpsi_dx[:-1, 1:] ** 2
                + dpsi_dx[1:, :-1] ** 2 + dpsi_dx[1:, 1:] ** 2
            )
        )
        energy = np.sum(grad_cell * mesh.area_element)

        if step % 10 == 0 or step == n_steps - 1:
            print(f"    {step:5d}  {t:12.4e}  {max_psi:14.6e}  {energy:14.6e}")

        psi_history.append(psi.copy())
        time_arr.append(t)

    # 分析最终增长率
    max_psi_arr = np.array([np.max(np.abs(p)) for p in psi_history])
    if max_psi_arr[-1] > 1.0e-30 and max_psi_arr[0] > 1.0e-30:
        gamma_num = np.log(max_psi_arr[-1] / max_psi_arr[0]) / t
        print(f"\n  数值增长率 gamma_num = {gamma_num:.6e}")

    return {
        "time_arr": np.array(time_arr),
        "psi_history": psi_history,
        "max_psi_arr": max_psi_arr,
    }


# ================================================================
#  第 7 部分: 谱分析
# ================================================================
def part7_spectral(time_arr: np.ndarray, max_psi_arr: np.ndarray) -> None:
    header("第 7 部分: 谱分析 (FFT, Hankel-Prony, 自相关)")

    dt = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 1.0

    # FFT 功率谱
    freq, psd = power_spectrum(np.log(max_psi_arr + 1e-30), dt)
    dom_freq = freq[np.argmax(psd[1:]) + 1] if len(psd) > 1 else 0.0
    print(f"\n  FFT 功率谱:")
    print(f"    主导频率 = {dom_freq:.4e}")

    # Hankel-Prony
    if len(max_psi_arr) >= 8:
        gamma_prony, omega_prony, amp_prony = prony_analysis(
            np.log(max_psi_arr + 1e-30), n_modes=3
        )
        print(f"\n  Prony 分析 (Hankel SVD):")
        for n in range(min(3, len(gamma_prony))):
            print(f"    模式 {n}: gamma = {gamma_prony[n]:.4e}, omega = {omega_prony[n]:.4e}")

    # Hankel 矩阵测试
    n_hank = 5
    x_hank = np.arange(1.0, 2 * n_hank, 1.0)
    H = hankel_matrix(n_hank, x_hank)
    print(f"\n  Hankel 矩阵 ({n_hank}x{n_hank}):")
    print(f"    行列式 = {np.linalg.det(H):.4e}")
    print(f"    条件数 = {np.linalg.cond(H):.4e}")

    # 自相关
    R = autocorrelation(np.log(max_psi_arr + 1e-30), max_lag=min(10, len(max_psi_arr) // 2))
    print(f"\n  自相关函数:")
    for k in range(min(5, len(R))):
        print(f"    R[{k}] = {R[k]:.6e}")


# ================================================================
#  第 8 部分: 矩阵求解器与 Cholesky
# ================================================================
def part8_matrix() -> None:
    header("第 8 部分: 矩阵求解器 (Cholesky 分解)")

    # 构造 SPD 矩阵 (离散 Laplace + 对角优势)
    n = 16
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 4.0
        if i > 0:
            A[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0
    A += 0.5 * np.eye(n)  # 确保 SPD

    print(f"\n  SPD 矩阵 ({n}x{n}):")
    print(f"    条件数 = {np.linalg.cond(A):.4e}")

    # Cholesky 分解
    U, nullity, ifault = cholesky_factor(A)
    print(f"\n  Cholesky 分解:")
    print(f"    ifault = {ifault}")
    print(f"    nullity = {nullity}")

    # 验证: U^T U ≈ A
    if ifault == 0:
        A_recon = U.T @ U
        err = np.max(np.abs(A_recon - A))
        print(f"    ||U^T U - A|| = {err:.4e}")

    # 求解线性系统
    b = np.ones(n)
    x_sol = cholesky_solve(U, b)
    residual = np.max(np.abs(A @ x_sol - b))
    print(f"    ||A x - b|| = {residual:.4e}")

    # Laplace 矩阵
    L = laplacian_2d_sparse(8, 8, 0.1, 0.1, bc_x="periodic", bc_y="dirichlet")
    print(f"\n  2D Laplace 矩阵 (8x8 网格):")
    print(f"    维度 = {L.shape}")
    print(f"    非零元 = {np.count_nonzero(L)}")


# ================================================================
#  第 9 部分: ODE 积分与测试粒子
# ================================================================
def part9_ode() -> None:
    header("第 9 部分: ODE 积分 (RK4) 与测试粒子 (velocity-Verlet)")

    # 非线性摆 (参考 861_pendulum_nonlinear_ode)
    # y1' = y2, y2' = -(g/L) sin(y1)
    g_over_L = 9.8

    def pendulum_rhs(t, y):
        dydt = np.zeros(2)
        dydt[0] = y[1]
        dydt[1] = -g_over_L * np.sin(y[0])
        return dydt

    y0 = np.array([0.5 * np.pi, 0.0])  # 初始角度 90°, 静止
    t_arr, y_arr = rk4_integrate(pendulum_rhs, (0.0, 10.0), y0, n_steps=1000)

    print(f"\n  非线性摆 (RK4):")
    print(f"    初始角度 = {y0[0]:.4f} rad")
    print(f"    积分时间 = 10.0 s, 步数 = 1000")
    print(f"    最终角度 = {y_arr[-1, 0]:.6f} rad")
    print(f"    最终角速度 = {y_arr[-1, 1]:.6f} rad/s")

    # 能量守恒检验
    E0 = 0.5 * y0[1] ** 2 + g_over_L * (1.0 - np.cos(y0[0]))
    E_final = 0.5 * y_arr[-1, 1] ** 2 + g_over_L * (1.0 - np.cos(y_arr[-1, 0]))
    print(f"    初始能量 = {E0:.10f}")
    print(f"    最终能量 = {E_final:.10f}")
    print(f"    相对误差 = {abs(E_final - E0) / abs(E0):.4e}")

    # 测试粒子 (velocity-Verlet, 参考 744_md)
    # 简谐势: F = -k x
    k_spring = 1.0
    mass = 1.0
    dt_vv = 0.1

    def spring_force(pos):
        return -k_spring * pos

    pos = np.array([[1.0]])
    vel = np.array([[0.0]])
    acc = spring_force(pos) / mass

    print(f"\n  测试粒子 (velocity-Verlet):")
    for step in range(100):
        pos, vel, acc = velocity_verlet_step(pos, vel, acc, spring_force, mass, dt_vv)

    E_final_vv = 0.5 * k_spring * pos[0, 0] ** 2 + 0.5 * mass * vel[0, 0] ** 2
    E0_vv = 0.5 * k_spring * 1.0 ** 2
    print(f"    100 步后位置 = {pos[0, 0]:.6f}")
    print(f"    初始能量 = {E0_vv:.10f}")
    print(f"    最终能量 = {E_final_vv:.10f}")
    print(f"    相对误差 = {abs(E_final_vv - E0_vv) / abs(E0_vv):.4e}")


# ================================================================
#  第 10 部分: 振荡分析
# ================================================================
def part10_oscillation(time_arr: np.ndarray, max_psi_arr: np.ndarray) -> None:
    header("第 10 部分: 振荡分析 (Hilbert, PAC, burst)")

    dt = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 1.0
    signal = np.log(max_psi_arr + 1e-30)

    # Hilbert 变换
    z = hilbert_transform(signal)
    amp, phase = instantaneous_amplitude_phase(signal)
    omega_inst = instantaneous_frequency(phase, dt)
    gamma_inst = instantaneous_growth_rate(amp, dt)

    print(f"\n  Hilbert 变换:")
    print(f"    振幅范围: [{np.min(amp):.4e}, {np.max(amp):.4e}]")
    print(f"    瞬时频率范围: [{np.min(omega_inst):.4e}, {np.max(omega_inst):.4e}]")
    print(f"    瞬时增长率范围: [{np.min(gamma_inst):.4e}, {np.max(gamma_inst):.4e}]")

    # Burst 检测
    bursts = detect_oscillation_bursts(amp, dt, threshold_factor=1.0, min_duration=3)
    print(f"\n  振荡爆发检测:")
    print(f"    检测到 {len(bursts['start_idx'])} 个爆发事件")
    if len(bursts['peak_amp']) > 0:
        print(f"    最大峰值振幅 = {np.max(bursts['peak_amp']):.4e}")

    # 调制指数
    if len(signal) > 20:
        # 合成慢频和快频
        n = len(signal)
        t = np.arange(n) * dt
        slow = np.sin(2 * np.pi * 0.1 * t)
        fast = (1.0 + 0.5 * np.sin(2 * np.pi * 0.1 * t)) * np.sin(2 * np.pi * 1.0 * t)
        amp_fast, _ = instantaneous_amplitude_phase(fast)
        _, phase_slow = instantaneous_amplitude_phase(slow)
        mi = modulation_index(amp_fast, phase_slow, num_bins=12)
        print(f"\n  调制指数 (合成信号):")
        print(f"    MI = {mi:.4f} (0=无耦合, 1=完全耦合)")


# ================================================================
#  第 11 部分: MCMC 不确定性量化
# ================================================================
def part11_mcmc() -> None:
    header("第 11 部分: MCMC 不确定性量化")

    # 合成 "观测" 数据
    true_params = np.array([4.0, 0.5, 2.0])  # log10(S), kL, Delta'
    from monte_carlo_uq import tearing_growth_rate_model
    gamma_obs = tearing_growth_rate_model(true_params)
    data_std = 0.1 * gamma_obs + 1e-10

    print(f"\n  合成观测:")
    print(f"    真实参数 = {true_params}")
    print(f"    观测增长率 = {gamma_obs:.4e}")

    # MCMC 采样
    prior_std = np.array([2.0, 1.0, 2.0])
    proposal_std = np.array([0.3, 0.1, 0.3])

    def logp(params):
        return log_posterior_tearing(params, gamma_obs, data_std, prior_std)

    sampler = MetropolisHastings(logp, proposal_std, seed=42)
    x0 = true_params + 0.1 * np.random.randn(3)
    chain, acc_rate = sampler.sample(x0, n_samples=500, burn_in=200, thin=2)

    print(f"\n  MCMC 结果:")
    print(f"    样本数 = {chain.shape[0]}")
    print(f"    接受率 = {acc_rate:.4f}")

    stats = summarize_chain(chain, ["log10_S", "kL", "Delta_prime"])
    for key, val in sorted(stats.items()):
        print(f"    {key:25s} = {val:.4e}")


# ================================================================
#  第 12 部分: Markov 模式耦合
# ================================================================
def part12_markov() -> None:
    header("第 12 部分: Markov 模式耦合模型")

    n_modes = 4
    model = ModeCouplingMarkov(n_modes, seed=42)

    # 耦合强度 (非对称)
    coupling = np.array([
        [0.0, 0.3, 0.1, 0.05],
        [0.2, 0.0, 0.4, 0.1],
        [0.1, 0.3, 0.0, 0.3],
        [0.05, 0.1, 0.2, 0.0],
    ])
    decay = np.array([0.1, 0.2, 0.15, 0.3])

    T = model.build_transition_matrix(coupling, decay)
    print(f"\n  转移矩阵 T ({n_modes}x{n_modes}):")
    for i in range(n_modes):
        row_str = "    [" + ", ".join(f"{T[i, j]:.4f}" for j in range(n_modes)) + "]"
        print(row_str)

    # 行和检验
    row_sums = np.sum(T, axis=1)
    print(f"\n  行和 (应为 1): {row_sums}")

    # 平稳分布
    pi = model.stationary_distribution()
    print(f"  平稳分布 π = {pi}")

    # 模拟链
    chain = model.simulate_chain(200, initial_state=0)
    print(f"\n  Markov 链模拟 (200 步):")
    unique, counts = np.unique(chain, return_counts=True)
    for s, c in zip(unique, counts):
        print(f"    状态 {s}: {c} 次 ({100.0 * c / 200:.1f}%)")


# ================================================================
#  第 13 部分: 3D 磁场重构
# ================================================================
def part13_reconstruction(mesh: StructuredMesh2D, eq: dict) -> None:
    header("第 13 部分: 3D 磁场重构 (encoder-decoder 上采样)")

    psi_2d = eq["psi"]

    # 多级上采样
    target_shape = (2 * mesh.ny, 2 * mesh.nx)
    psi_upsampled = multiscale_upsample(psi_2d, target_shape, n_levels=2)

    print(f"\n  多级上采样:")
    print(f"    粗网格: {psi_2d.shape}")
    print(f"    细网格: {psi_upsampled.shape}")

    # 3D 重构
    n_phi = 8
    B_R, B_phi, B_Z = reconstruct_3d_field(
        psi_2d, mesh, n_phi=n_phi, n_harmonics=2
    )

    print(f"\n  3D 磁场重构:")
    print(f"    形状: B_R.shape = {B_R.shape}")
    print(f"    max |B_R|   = {np.max(np.abs(B_R)):.4e}")
    print(f"    max |B_phi| = {np.max(np.abs(B_phi)):.4e}")
    print(f"    max |B_Z|   = {np.max(np.abs(B_Z)):.4e}")

    # 散度检验
    div_rel = check_divergence_free(B_R, B_phi, B_Z, mesh, R0=1.0)
    print(f"\n  散度检验:")
    print(f"    max|∇·B|/max|B| = {div_rel:.4e}")

    # 磁能
    vol_elem = mesh.area_element.mean() * (2 * np.pi / n_phi)
    W_B = magnetic_energy(B_R, B_phi, B_Z, vol_elem)
    print(f"    磁能 W_B = {W_B:.4e}")


# ================================================================
#  第 14 部分: 等离子体诊断
# ================================================================
def part14_diagnostics(mesh: StructuredMesh2D, eq: dict) -> None:
    header("第 14 部分: 等离子体诊断")

    y = mesh.y
    bx0 = eq["bx0"]
    jz0 = eq["jz0"]
    k_mode = eq["k_mode"]
    psi_y = np.mean(eq["psi"], axis=1)

    # 汇总诊断
    diag = diagnostic_summary(y, bx0, psi_y, jz0, k_mode)

    print(f"\n  诊断量汇总:")
    for key, val in diag.items():
        print(f"    {key:30s} = {val:.4e}")

    # 安全因子
    r_arr = np.abs(y)
    # 简化: B_theta 正比于 jz * r
    b_theta = np.abs(jz0) * r_arr * 0.01
    b_phi = 1.0
    R0 = 1.0
    q_profile = safety_factor_profile(r_arr, b_theta, b_phi, R0)

    print(f"\n  安全因子剖面:")
    print(f"    q(0) = {q_profile[0]:.4e}")
    print(f"    q(ly) = {q_profile[-1]:.4e}")

    # 有理面 q=1
    r_res = find_rational_surface(r_arr, q_profile, m=1, n=1)
    if r_res > 0:
        print(f"    q=1 有理面位置: r = {r_res:.4e}")
    else:
        print(f"    未找到 q=1 有理面")

    # 电流片厚度
    delta = current_sheet_thickness(y, jz0)
    print(f"\n  电流片厚度 (1/e): delta = {delta:.4e}")

    # 总能量
    bx_2d = np.broadcast_to(bx0[:, None], (mesh.ny, mesh.nx))
    by_2d = np.zeros_like(bx_2d)
    bz_2d = np.zeros_like(bx_2d)
    vx = np.zeros_like(bx_2d)
    vy = np.zeros_like(bx_2d)
    vz = np.zeros_like(bx_2d)
    rho_2d = np.full_like(bx_2d, pc.RHO0)

    energies = total_energy(
        bx_2d, by_2d, bz_2d, vx, vy, vz, eq["p0"], rho_2d, mesh.area_element
    )

    print(f"\n  能量分解:")
    for key, val in energies.items():
        print(f"    W_{key:10s} = {val:.4e}")


# ================================================================
#  第 15 部分: FKR 标度律验证
# ================================================================
def part15_fkr_scaling(mesh: StructuredMesh2D, eq: dict) -> None:
    header("第 15 部分: FKR 标度律验证")

    y = mesh.y
    bx0 = eq["bx0"]
    k_mode = eq["k_mode"]

    # 在不同电阻率下计算增长率
    eta_values = np.logspace(-4, -2, 5)
    result = fkr_scaling_test(y, bx0, eta_values, k_mode)

    print(f"\n  参数扫描 (k = {k_mode:.4e}):")
    print(f"    {'eta':>12s}  {'gamma_num':>14s}  {'gamma_fkr':>14s}  {'ratio':>10s}")
    for i in range(len(eta_values)):
        print(f"    {result['eta'][i]:12.4e}  "
              f"{result['gamma_numerical'][i]:14.6e}  "
              f"{result['gamma_fkr'][i]:14.6e}  "
              f"{result['ratio'][i]:10.4f}")


# ================================================================
#  主程序
# ================================================================
def main() -> int:
    t_start = walltime.time()

    print()
    print("#" * 70)
    print("#  PROJECT 287: MHD 不稳定性数值模拟")
    print("#  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("#" * 70)
    print(f"\n  Python 版本: {sys.version.split()[0]}")
    print(f"  NumPy 版本:  {np.__version__}")
    print(f"  开始时间:    {walltime.strftime('%Y-%m-%d %H:%M:%S')}")

    # 第 1 部分: 参数
    part1_parameters()

    # 第 2 部分: 网格
    mesh = part2_mesh()

    # 第 3 部分: FD 验证
    part3_fd_validation(mesh)

    # 第 4 部分: 平衡与扰动
    eq = part4_equilibrium(mesh)

    # 第 5 部分: 稳定性分析
    part5_stability(mesh, eq)

    # 第 6 部分: ADI 时间演化
    tevo = part6_time_evolution(mesh, eq)

    # 第 7 部分: 谱分析
    part7_spectral(tevo["time_arr"], tevo["max_psi_arr"])

    # 第 8 部分: 矩阵求解器
    part8_matrix()

    # 第 9 部分: ODE 与粒子
    part9_ode()

    # 第 10 部分: 振荡分析
    part10_oscillation(tevo["time_arr"], tevo["max_psi_arr"])

    # 第 11 部分: MCMC UQ
    part11_mcmc()

    # 第 12 部分: Markov 耦合
    part12_markov()

    # 第 13 部分: 3D 重构
    part13_reconstruction(mesh, eq)

    # 第 14 部分: 诊断
    part14_diagnostics(mesh, eq)

    # 第 15 部分: FKR 标度
    part15_fkr_scaling(mesh, eq)

    # 完成
    t_end = walltime.time()
    elapsed = t_end - t_start

    header("完成")
    print(f"\n  总运行时间: {elapsed:.2f} 秒")
    print(f"  所有模块运行成功, 无错误.")
    print()
    print("=" * 70)
    print("  PROJECT 287 正常结束")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
