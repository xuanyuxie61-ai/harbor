# -*- coding: utf-8 -*-
"""
main.py
分数量子霍尔效应Laughlin态的多体数值模拟平台

统一入口，零参数可运行。
执行从单粒子基函数构建到多体关联分析、拓扑不变量计算的完整流程。
"""
import sys
import numpy as np

# 导入所有模块
from utils import (
    magnetic_length, cyclotron_frequency, landau_level_energy,
    filling_factor, fermi_dirac, gram_schmidt_qr, condition_number,
    H_BAR, E_CHARGE
)
from landau_levels import (
    landau_orbital_wavefunction, build_spectral_stiffness_matrix,
    local_basis_1d_lagrange, local_fem_1d,
    landau_degeneracy, density_of_states_landau
)
from laughlin_wavefunction import (
    laughlin_wavefunction, laughlin_log_probability,
    quasihole_wavefunction, quasielectron_wavefunction,
    pair_correlation_function, structure_factor_s_q
)
from monte_carlo_sampler import (
    hammersley_sequence, map_hammersley_to_disk,
    variational_monte_carlo_energy, qmc_integration
)
from hartree_fock_solver import (
    cg_ne_solve, newton_solve,
    self_consistent_hf, coulomb_interaction_2d
)
from edge_state_solver import (
    euler_integrate, shooting_method_bvp,
    edge_state_radial_ode, chiral_luttinger_dispersion,
    edge_state_density_of_states, blowup_ode_stabilized
)
from density_evolution import (
    fd1d_wave_solve, fisher_kpp_exact_solution,
    fisher_kpp_fd_solve, density_matrix_evolution_lindblad
)
from correlation_functions import (
    find_nearest_neighbors, hyperball_distance_stats,
    hypercube_surface_distance_stats,
    two_point_correlation, density_correlation_function
)
from quadrature_integrals import (
    comp_next, monomial_value, wedge01_integral,
    gauss_legendre_1d, multidimensional_gauss_legendre,
    wedge_exactness_test, integrate_coulomb_2d_gauss
)
from topological_invariants import (
    berry_connection, berry_curvature_discrete,
    chern_number_from_berry_curvature, tknn_conductivity,
    flux_quantization_phase, orbital_evolution_parameters
)


def print_header(title):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_landau_level_analysis(B, m_star, N_electrons, A_sample):
    """
    模块1: Landau能级分析与态密度计算
    """
    print_header("模块1: Landau能级与单粒子基函数分析")

    lB = magnetic_length(B, m_star)
    omega_c = cyclotron_frequency(B, m_star)
    N_phi = landau_degeneracy(B, A_sample, m_star)
    nu = filling_factor(N_electrons, B, A_sample, m_star)

    print(f"\n物理参数:")
    print(f"  磁场强度 B = {B:.2f} T")
    print(f"  有效质量 m* = {m_star:.4f}")
    print(f"  回旋频率 ω_c = {omega_c:.6f}")
    print(f"  磁长度 l_B = {lB:.6f}")
    print(f"  样品面积 A = {A_sample:.2f}")
    print(f"  磁通量子数 N_Φ = {N_phi:.2f}")
    print(f"  电子数 N_e = {N_electrons}")
    print(f"  填充因子 ν = {nu:.6f}")

    print(f"\n前6个Landau能级:")
    for n in range(6):
        En = landau_level_energy(n, B, m_star)
        print(f"  n={n}: E_n/ħω_c = {En / (H_BAR * omega_c):.4f}")

    # 态密度
    E_range = np.linspace(0.0, 6.0 * H_BAR * omega_c, 200)
    dos = density_of_states_landau(E_range, B, m_star, gamma=0.05)
    print(f"\n态密度特征:")
    print(f"  最大态密度: {np.max(dos):.4f}")
    print(f"  平均态密度: {np.mean(dos):.4f}")

    # 谱FEM刚度矩阵
    print(f"\n谱有限元刚度矩阵条件数:")
    for N in [4, 8, 12]:
        K = build_spectral_stiffness_matrix(N, domain=(0.0, 1.0))
        cond = np.linalg.cond(K)
        print(f"  N={N:2d}: cond(K) = {cond:.4e}")

    return lB, omega_c, N_phi, nu


def run_laughlin_wavefunction_analysis(N_electrons, m, lB, B):
    """
    模块2: Laughlin波函数与准粒子激发
    """
    print_header("模块2: Laughlin多体波函数与准粒子激发")

    # 生成初始电子构型（在圆盘内准均匀分布）
    np.random.seed(42)
    R_max = np.sqrt(2.0 * m * N_electrons) * lB * 0.6
    theta = np.random.uniform(0.0, 2.0 * np.pi, N_electrons)
    r = np.sqrt(np.random.uniform(0.0, 1.0, N_electrons)) * R_max
    z = r * np.exp(1j * theta)

    log_psi = laughlin_wavefunction(z, m, lB, return_log=True)
    print(f"\nLaughlin波函数 (m={m}, ν=1/{m}):")
    print(f"  电子数 N = {N_electrons}")
    print(f"  系统半径 R ≈ {R_max:.4f}")
    print(f"  ln|Ψ|² = {log_psi.real:.4f}")

    # 准空穴
    z0 = 0.5 * lB + 0.3j * lB
    log_psi_qh = quasihole_wavefunction(z, z0, m, lB, return_log=True)
    print(f"\n准空穴激发 (z0 = {z0:.3f}):")
    print(f"  ln|Ψ_qh|² = {log_psi_qh.real:.4f}")
    print(f"  准空穴携带分数电荷 e* = e/{m} = {1.0/m:.6f}")

    # 配对关联函数
    r_edges, g_r, r_centers = pair_correlation_function(z, m, lB, r_bins=40)
    print(f"\n配对关联函数 g^(2)(r) 统计:")
    print(f"  r 范围: [{r_centers[0]:.4f}, {r_centers[-1]:.4f}]")
    print(f"  g(r) 最大值: {np.max(g_r):.4f}")
    print(f"  g(r) 最小值: {np.min(g_r):.4f}")

    # 结构因子
    q_vals, S_q = structure_factor_s_q(z, m, lB, q_bins=25)
    print(f"\n结构因子 S(q):")
    print(f"  q 范围: [{q_vals[0]:.4f}, {q_vals[-1]:.4f}]")
    print(f"  S(q) 范围: [{np.min(S_q):.4f}, {np.max(S_q):.4f}]")

    return z, log_psi, g_r, r_centers


def run_monte_carlo_analysis(N_electrons, m, lB):
    """
    模块3: 准蒙特卡洛采样
    """
    print_header("模块3: 准蒙特卡洛采样与积分")

    # Hammersley序列生成
    n_samples = 500
    h_points = hammersley_sequence(0, n_samples - 1, 2)
    z_samples = map_hammersley_to_disk(h_points, radius=np.sqrt(2.0 * m * N_electrons) * lB * 0.5)
    print(f"\nHammersley低差异序列:")
    print(f"  采样点数: {n_samples}")
    print(f"  映射到圆盘半径: {np.max(np.abs(z_samples)):.4f}")

    # QMC积分测试
    def f_integrand(pt):
        x, y = pt[0], pt[1]
        return x ** 2 + y ** 2

    integral_qmc = qmc_integration(f_integrand, h_points, domain_volume=np.pi * (np.sqrt(2.0 * m * N_electrons) * lB * 0.5) ** 2)
    print(f"\n准Monte Carlo积分 (圆盘上 ∫(x²+y²)dA):")
    print(f"  估计值: {integral_qmc:.6f}")

    return z_samples


def run_hartree_fock_analysis(B, lB, grid_res=12):
    """
    模块4: Hartree-Fock自洽场
    """
    print_header("模块4: Hartree-Fock自洽场分析")

    x = np.linspace(-2.0 * lB, 2.0 * lB, grid_res)
    y = np.linspace(-2.0 * lB, 2.0 * lB, grid_res)
    grid_x, grid_y = np.meshgrid(x, y)

    energies, orbitals, density, converged = self_consistent_hf(
        N_electrons=2, N_basis=4, B=B, lB=lB,
        grid_x=grid_x, grid_y=grid_y,
        max_iter=8, tol=1e-4
    )

    print(f"\nHartree-Fock自洽场结果:")
    print(f"  收敛状态: {converged}")
    print(f"  单粒子能级: {np.real(energies[:4])}")
    print(f"  最大电子密度: {np.max(density):.6f}")
    print(f"  总电子数（格点积分）: {np.sum(density) * (x[1]-x[0]) * (y[1]-y[0]):.4f}")

    # CGNE测试
    A_test = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, 1.0]])
    b_test = np.array([4.0, 5.0, 3.0])
    x_cg, conv_cg, res_cg = cg_ne_solve(A_test, b_test, tol=1e-10)
    print(f"\n共轭梯度(CGNE)测试:")
    print(f"  解: {x_cg}")
    print(f"  残差: {res_cg:.2e}")

    # Newton迭代测试
    def F_newton(x):
        return np.array([x[0] ** 2 - 2.0])
    def J_newton(x):
        return np.array([[2.0 * x[0]]])
    x_newton, conv_newton, nit_newton = newton_solve(F_newton, J_newton, np.array([1.5]))
    print(f"\nNewton迭代测试 (x²=2):")
    print(f"  解: {x_newton[0]:.8f}")
    print(f"  收敛: {conv_newton}, 迭代: {nit_newton}")

    return energies, density


def run_edge_state_analysis(B, m_star):
    """
    模块5: 边缘态与手性Luttinger液体
    """
    print_header("模块5: 边缘态BVP求解与手性Luttinger液体")

    # 打靶法测试
    def harmonic_ode(r, y):
        u, up = y
        return np.array([up, -u])

    r_grid, u_sol, conv_shoot, nit_shoot = shooting_method_bvp(
        harmonic_ode, a=0.0, b=np.pi, ya=0.0, yb_target=0.0,
        alpha_guess1=0.5, alpha_guess2=1.5, max_iter=20, tol=1e-5, n_steps=500
    )
    err_shoot = np.max(np.abs(u_sol - np.sin(r_grid)))
    print(f"\n打靶法测试 (u''+u=0, u(0)=u(π)=0):")
    print(f"  收敛: {conv_shoot}, 迭代: {nit_shoot}")
    print(f"  与sin(r)最大误差: {err_shoot:.2e}")

    # 边缘态径向方程
    ode_rhs = edge_state_radial_ode(B, m_star, angular_m=1, E=5.0)
    r_edge, u_edge, conv_edge, nit_edge = shooting_method_bvp(
        ode_rhs, a=1e-3, b=3.0, ya=0.0, yb_target=0.0,
        alpha_guess1=0.1, alpha_guess2=1.0, max_iter=10, tol=1e-3, n_steps=600
    )
    print(f"\n边缘态径向方程:")
    print(f"  收敛: {conv_edge}, 迭代: {nit_edge}")
    print(f"  解范数: {np.linalg.norm(u_edge):.4f}")

    # 手性Luttinger色散
    k_vals = np.linspace(0.1, 5.0, 20)
    v_F = 1.0
    eps = chiral_luttinger_dispersion(k_vals, v_F, g_factor=0.5)
    print(f"\n手性Luttinger液体 (g=0.5):")
    print(f"  k=0.1 时 ε = {eps[0]:.4f}")
    print(f"  k=5.0 时 ε = {eps[-1]:.4f}")

    return u_edge


def run_density_evolution():
    """
    模块6: 密度演化与Fisher-KPP方程
    """
    print_header("模块6: 密度演化动力学")

    # Fisher-KPP数值解
    u_fisher = fisher_kpp_fd_solve(
        60, -8.0, 8.0, 150, 0.0, 4.0,
        D=1.0, r=1.0, K=1.0
    )
    print(f"\nFisher-KPP反应扩散方程:")
    print(f"  解形状: {u_fisher.shape}")
    print(f"  初始峰值: {np.max(u_fisher[0,:]):.4f}")
    print(f"  最终峰值: {np.max(u_fisher[-1,:]):.4f}")
    print(f"  总密度守恒: init={np.sum(u_fisher[0,:]):.2f}, final={np.sum(u_fisher[-1,:]):.2f}")

    # 波动方程
    def u_x1(t): return 0.0
    def u_x2(t): return 0.0
    def u_t1(x): return np.sin(np.pi * x)
    def ut_t1(x): return np.zeros_like(x)
    u_wave = fd1d_wave_solve(40, 0.0, 1.0, 80, 0.0, 2.0, 1.0, u_x1, u_x2, u_t1, ut_t1)
    print(f"\n一维波动方程:")
    print(f"  解形状: {u_wave.shape}")
    print(f"  t=0 最大振幅: {np.max(np.abs(u_wave[0,:])):.4f}")
    print(f"  t=2 最大振幅: {np.max(np.abs(u_wave[-1,:])):.4f}")

    # Lindblad演化
    H = np.array([[1.0, 0.3], [0.3, -0.5]], dtype=complex)
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    L = np.array([[0.0, 0.05], [0.0, 0.0]], dtype=complex)
    times, rhos = density_matrix_evolution_lindblad(rho0, H, [L], (0.0, 2.0), 80)
    print(f"\nLindblad密度矩阵演化:")
    print(f"  时间步: {len(times)}")
    print(f"  ρ_11: {rhos[0][0,0].real:.4f} → {rhos[-1][0,0].real:.4f}")
    print(f"  最终迹: {np.trace(rhos[-1]).real:.6f}")

    return u_fisher


def run_correlation_analysis(z_samples, lB, m):
    """
    模块7: 关联函数与最近邻分析
    """
    print_header("模块7: 关联函数与最近邻分析")

    # 最近邻
    m_dim, nr, ns = 2, 5, 3
    R_pts = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]])
    S_pts = np.array([[0.3, 2.5, 4.2],
                      [0.0, 0.0, 0.0]])
    idx_nn, dists_nn = find_nearest_neighbors(m_dim, nr, R_pts, ns, S_pts)
    print(f"\n最近邻搜索:")
    print(f"  查询点索引: {idx_nn}")
    print(f"  最小距离: {dists_nn}")

    # 超球距离统计
    print(f"\n超球距离统计:")
    for dim in [2, 3]:
        mu, var, _ = hyperball_distance_stats(dim, 200)
        print(f"  dim={dim}: μ={mu:.4f}, σ²={var:.6f}")

    # 量子霍尔关联函数
    r_edges, g2, r_centers = two_point_correlation(z_samples, lB, r_bins=30)
    print(f"\n两点关联函数:")
    print(f"  g2 前3个值: {g2[:3]}")
    print(f"  g2 最大值: {np.max(g2):.4f}")

    # 密度关联（Fourier）
    n_grid = np.random.rand(24, 24)
    q_vals, S_q = density_correlation_function(n_grid, dx=0.1, dy=0.1)
    print(f"\n密度结构因子 S(q):")
    print(f"  S(q=0附近) ≈ {S_q[0]:.4f}")

    return g2


def run_quadrature_analysis():
    """
    模块8: 高维积分与精确性验证
    """
    print_header("模块8: 高维数值积分与精确性验证")

    # 楔形体积分
    print(f"\n楔形体精确积分:")
    for e in [[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,2]]:
        val = wedge01_integral(e)
        print(f"  I{e} = {val:.6f}")

    # 多维Gauss积分
    def f2(x, y): return x**2 * y**2
    val2d = multidimensional_gauss_legendre(f2, 2, 4, [(0.0,1.0),(0.0,1.0)])
    print(f"\n二维Gauss积分 (x²y² over [0,1]²):")
    print(f"  数值: {val2d:.8f}, 精确: {1.0/9.0:.8f}")

    # 楔形体精确性检验
    n_pts = 10
    x_pts = np.random.rand(3, n_pts)
    x_pts[1,:] *= (1.0 - x_pts[0,:])
    x_pts[2,:] = 2.0 * x_pts[2,:] - 1.0
    w = np.ones(n_pts) / n_pts
    results = wedge_exactness_test(x_pts, w, degree_max=2)
    max_err = max([r[4] for r in results])
    print(f"\n楔形体求积精确性 (degree≤2): max_err = {max_err:.4f}")

    # 库仑积分
    val_coul, exact_coul = integrate_coulomb_2d_gauss(8, 16, epsilon_r=12.0, R_max=1.5)
    print(f"\n二维库仑势积分:")
    print(f"  数值: {val_coul:.6f}, 精确: {exact_coul:.6f}")

    return val2d


def run_topological_analysis():
    """
    模块9: 拓扑不变量计算
    """
    print_header("模块9: 拓扑不变量与Chern数")

    # Berry曲率与Chern数
    Nk = 30
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    m_mass = 1.5
    d_x = np.sin(KX)
    d_y = np.sin(KY)
    d_z = m_mass + np.cos(KX) + np.cos(KY)
    d_mag = np.sqrt(d_x**2 + d_y**2 + d_z**2)

    u_grid = np.zeros((Nk, Nk, 2), dtype=complex)
    for i in range(Nk):
        for j in range(Nk):
            dx, dy, dz = d_x[i,j], d_y[i,j], d_z[i,j]
            dm = d_mag[i,j]
            if dm < 1e-14:
                u_grid[i,j] = [1.0, 0.0]
            elif abs(dm - dz) < 1e-14:
                u_grid[i,j] = [0.0, 1.0]
            else:
                u_grid[i,j] = [
                    np.sqrt((1.0 - dz/dm)/2.0),
                    (dx + 1j*dy) / np.sqrt(2.0*dm*(dm-dz))
                ]

    Omega, _, _ = berry_curvature_discrete(u_grid, kx, ky)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    C = chern_number_from_berry_curvature(Omega, dkx, dky)
    print(f"\nBerry曲率与Chern数 (m={m_mass}):")
    print(f"  Chern数 C = {C:.4f}")
    print(f"  霍尔电导 σ_xy = (e²/h) · {C:.4f}")

    # 磁通量子化
    print(f"\n磁通量子化Berry相位:")
    for n_phi in [3, 6, 9]:
        n_e = n_phi // 3
        phase, charge = flux_quantization_phase(n_phi, n_e, m_laughlin=3)
        print(f"  n_Φ={n_phi}, n_e={n_e}: phase={phase:.4f}, charge={charge:.6f}")

    # 轨道参数演化
    tau, theta, x2, y2, z2 = orbital_evolution_parameters(
        ecc=0.01671, lon_deg=77.0, obliq_deg=23.44, n_points=50
    )
    print(f"\n绝热参数演化:")
    print(f"  角度范围: [{np.min(theta):.4f}, {np.max(theta):.4f}]")
    print(f"  轨道闭合误差: {abs(np.mean(x2**2+y2**2+z2**2)-1.0):.6f}")

    return C


def main():
    """
    主函数：零参数运行，执行完整的FQHE数值模拟流程。
    """
    print("\n" + "=" * 70)
    print("  分数量子霍尔效应：Laughlin态的多体数值模拟平台")
    print("  Fractional Quantum Hall Effect: Laughlin State Numerical Platform")
    print("=" * 70)
    print("\n本程序围绕凝聚态物理中的分数量子霍尔效应展开，")
    print("融合15个种子项目的核心算法，实现从单粒子Landau能级")
    print("到多体Laughlin波函数、自洽场、边缘态、拓扑不变量的")
    print("完整数值模拟链路。\n")

    # 全局物理参数
    B = 10.0          # 磁场强度 (T)
    m_star = 1.0      # 有效质量
    N_electrons = 8   # 电子数
    A_sample = 50.0   # 样品面积
    m_Laughlin = 3    # Laughlin指数 (ν = 1/3)

    # 执行各模块
    lB, omega_c, N_phi, nu = run_landau_level_analysis(B, m_star, N_electrons, A_sample)
    z, log_psi, g_r, r_centers = run_laughlin_wavefunction_analysis(N_electrons, m_Laughlin, lB, B)
    z_samples = run_monte_carlo_analysis(N_electrons, m_Laughlin, lB)
    energies_hf, density_hf = run_hartree_fock_analysis(B, lB)
    u_edge = run_edge_state_analysis(B, m_star)
    u_fisher = run_density_evolution()
    g2 = run_correlation_analysis(z, lB, m_Laughlin)
    val2d = run_quadrature_analysis()
    C = run_topological_analysis()

    # 总结
    print("\n" + "=" * 70)
    print("  模拟结果总结")
    print("=" * 70)
    print(f"\n物理参数:")
    print(f"  磁场 B = {B:.2f} T")
    print(f"  磁长度 l_B = {lB:.6f}")
    print(f"  填充因子 ν = {nu:.6f}")
    print(f"  Laughlin指数 m = {m_Laughlin}")
    print(f"\n核心计算结果:")
    print(f"  Laughlin波函数 ln|Ψ|² = {log_psi.real:.4f}")
    print(f"  配对关联函数峰值 = {np.max(g_r):.4f}")
    print(f"  Hartree-Fock基态能量 = {np.real(energies_hf[0]):.4f}")
    print(f"  边缘态解范数 = {np.linalg.norm(u_edge):.4f}")
    print(f"  Fisher-KPP最终峰值 = {np.max(u_fisher[-1,:]):.4f}")
    print(f"  二维Gauss积分精度 = {abs(val2d - 1.0/9.0):.2e}")
    print(f"  拓扑Chern数 C = {C:.4f}")
    print(f"  霍尔电导 σ_xy = (e²/h) × {C:.4f}")
    print("\n所有模块执行完毕，程序正常结束。\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: magnetic_length returns correct value for B=10, m*=1 ----
lB_val = magnetic_length(10.0, 1.0)
assert abs(lB_val - np.sqrt(0.1)) < 1e-12, '[TC01] magnetic_length FAILED'

# ---- TC02: cyclotron_frequency returns correct value for B=10, m*=1 ----
omega_val = cyclotron_frequency(10.0, 1.0)
assert abs(omega_val - 10.0) < 1e-12, '[TC02] cyclotron_frequency FAILED'

# ---- TC03: landau_level_energy for n=0 equals 0.5*hbar*omega_c ----
E0 = landau_level_energy(0, 10.0, 1.0)
assert abs(E0 - 5.0) < 1e-12, '[TC03] landau_level_energy FAILED'

# ---- TC04: filling_factor returns finite positive value ----
nu_val = filling_factor(8, 10.0, 50.0, 1.0)
assert nu_val > 0 and np.isfinite(nu_val), '[TC04] filling_factor FAILED'

# ---- TC05: fermi_dirac at T->0 behaves as step function around mu ----
f_low = fermi_dirac(0.5, 1.0, 1e-10)
f_high = fermi_dirac(1.5, 1.0, 1e-10)
assert f_low > 0.999 and f_high < 0.001, '[TC05] fermi_dirac step FAILED'

# ---- TC06: gram_schmidt_qr produces orthonormal Q columns ----
np.random.seed(42)
V_gs = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
Q_gs, R_gs = gram_schmidt_qr(V_gs)
for i in range(3):
    for j in range(3):
        dot = np.vdot(Q_gs[:, i], Q_gs[:, j])
        target = 1.0 if i == j else 0.0
        assert abs(dot - target) < 1e-10, '[TC06] gram_schmidt_qr FAILED'

# ---- TC07: condition_number of rank-1 2x2 matrix is inf ----
sing = np.array([[1.0, 2.0], [2.0, 4.0]])
assert condition_number(sing) == np.inf, '[TC07] condition_number singular FAILED'

# ---- TC08: safe_exp handles very large input without overflow ----
from utils import safe_exp
big_result = safe_exp(1000.0)
assert np.isfinite(big_result), '[TC08] safe_exp overflow FAILED'

# ---- TC09: safe_log handles zero without crash or nan ----
from utils import safe_log
log_result = safe_log(0.0)
assert np.isfinite(log_result), '[TC09] safe_log zero FAILED'

# ---- TC10: landau_orbital_wavefunction raises ValueError for invalid n ----
try:
    landau_orbital_wavefunction(-1, 0, np.array([1.0+0j]), 1.0)
    assert False, '[TC10] landau_orbital exception FAILED'
except ValueError:
    pass

# ---- TC11: landau_degeneracy returns positive value ----
Nphi = landau_degeneracy(10.0, 50.0, 1.0)
assert Nphi > 0, '[TC11] landau_degeneracy FAILED'

# ---- TC12: density_of_states_landau is non-negative everywhere ----
E_range = np.linspace(0, 10, 50)
dos = density_of_states_landau(E_range, 10.0, 1.0, gamma=0.05)
assert np.all(dos >= 0), '[TC12] DOS negative FAILED'

# ---- TC13: laughlin_wavefunction with return_log gives complex scalar ----
np.random.seed(42)
z_laugh = np.array([1.0+0j, 0.5+0.5j])
log_psi = laughlin_wavefunction(z_laugh, 3, 1.0, return_log=True)
assert isinstance(log_psi, (complex, np.complexfloating)), '[TC13] laughlin log type FAILED'

# ---- TC14: pair_correlation_function g(r) is non-negative ----
np.random.seed(42)
z_g2 = np.array([1.0+0j, -1.0+0j, 0.0+1.0j, 0.0-1.0j])
r_e, g_r, r_c = pair_correlation_function(z_g2, 3, 1.0, r_bins=10)
assert np.all(g_r >= 0), '[TC14] pair_correlation negative FAILED'

# ---- TC15: hammersley_sequence output shape is (m, N) ----
pts = hammersley_sequence(0, 49, 3)
assert pts.shape == (3, 50), '[TC15] hammersley shape FAILED'

# ---- TC16: radical_inverse produces identical results for same input ----
from monte_carlo_sampler import radical_inverse
phi_a = radical_inverse(7, 3)
phi_b = radical_inverse(7, 3)
assert abs(phi_a - phi_b) < 1e-15, '[TC16] radical_inverse FAILED'

# ---- TC17: cg_ne_solve satisfies normal equation A^T A x = A^T b ----
A_cg = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, 1.0]])
b_cg = np.array([4.0, 5.0, 3.0])
x_cg, conv_cg, res_cg = cg_ne_solve(A_cg, b_cg, tol=1e-10)
assert np.allclose(A_cg.T @ A_cg @ x_cg, A_cg.T @ b_cg, atol=1e-8), '[TC17] CGNE FAILED'

# ---- TC18: newton_solve finds sqrt(2) from initial guess 1.5 ----
def F_n(x): return np.array([x[0]**2 - 2.0])
def J_n(x): return np.array([[2.0*x[0]]])
x_n, conv_n, _ = newton_solve(F_n, J_n, np.array([1.5]), tol=1e-8)
assert conv_n and abs(x_n[0] - np.sqrt(2.0)) < 1e-6, '[TC18] Newton FAILED'

# ---- TC19: euler_integrate approximates exp(-t) within tolerance ----
def ode_decay(t, y): return np.array([-y[0]])
t_eu, y_eu = euler_integrate(ode_decay, (0.0, 1.0), np.array([1.0]), n_steps=500)
err_eu = np.max(np.abs(y_eu[:, 0] - np.exp(-t_eu)))
assert err_eu < 0.01, '[TC19] Euler FAILED'

# ---- TC20: chiral_luttinger_dispersion is linear in k ----
k_lut = np.array([1.0, 2.0, 3.0])
eps_lut = chiral_luttinger_dispersion(k_lut, 2.0, g_factor=0.5)
assert np.allclose(eps_lut, k_lut * 4.0), '[TC20] Luttinger FAILED'

# ---- TC21: fisher_kpp_exact_solution at t=0, x=0 equals 1/(1+a)^2 ----
u_fk = fisher_kpp_exact_solution(0.0, 0.0, a=1.0)
assert abs(u_fk - 0.25) < 1e-12, '[TC21] fisher_kpp exact FAILED'

# ---- TC22: fisher_kpp_fd_solve produces bounded solution ----
u_fd = fisher_kpp_fd_solve(20, -4.0, 4.0, 20, 0.0, 1.0, D=1.0, r=1.0, K=1.0)
assert u_fd.shape[0] > 0 and u_fd.shape[1] > 0, '[TC22] fisher_kpp shape FAILED'
assert np.all(u_fd >= 0) and np.all(u_fd <= 2.0), '[TC22] fisher_kpp bounds FAILED'

# ---- TC23: Lindblad evolution preserves trace of density matrix ----
H_lb = np.array([[1.0, 0.3], [0.3, -0.5]], dtype=complex)
rho0_lb = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
L_lb = np.array([[0.0, 0.05], [0.0, 0.0]], dtype=complex)
times_lb, rhos_lb = density_matrix_evolution_lindblad(rho0_lb, H_lb, [L_lb], (0.0, 1.0), 50)
tr_final = np.trace(rhos_lb[-1]).real
assert abs(tr_final - 1.0) < 1e-6, '[TC23] Lindblad trace FAILED'

# ---- TC24: fd1d_wave_solve satisfies boundary conditions ----
def ux1(t): return 0.0
def ux2(t): return 0.0
def ut1(x): return np.sin(np.pi * x)
def utt1(x): return np.zeros_like(x)
u_wv = fd1d_wave_solve(20, 0.0, 1.0, 40, 0.0, 1.0, 1.0, ux1, ux2, ut1, utt1)
assert u_wv.shape == (41, 21), '[TC24] wave shape FAILED'
assert abs(u_wv[-1, 0]) < 1e-6 and abs(u_wv[-1, -1]) < 1e-6, '[TC24] wave boundary FAILED'

# ---- TC25: find_nearest_neighbors returns correct indices and distances ----
m_nn, nr_nn, ns_nn = 2, 3, 2
R_nn = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
S_nn = np.array([[0.1, 1.9], [0.0, 0.0]])
idx_nn, dist_nn = find_nearest_neighbors(m_nn, nr_nn, R_nn, ns_nn, S_nn)
assert idx_nn[0] == 0 and idx_nn[1] == 2, '[TC25] nearest neighbor idx FAILED'
assert abs(dist_nn[0] - 0.1) < 1e-12, '[TC25] nearest neighbor dist FAILED'

# ---- TC26: hyperball_distance_stats is reproducible with fixed seed ----
mu1, var1, _ = hyperball_distance_stats(2, 50, seed=42)
mu2, var2, _ = hyperball_distance_stats(2, 50, seed=42)
assert abs(mu1 - mu2) < 1e-12 and abs(var1 - var2) < 1e-12, '[TC26] hyperball repro FAILED'

# ---- TC27: wedge01_integral for (0,0,0) equals unit volume 1 ----
vol = wedge01_integral([0, 0, 0])
assert abs(vol - 1.0) < 1e-12, '[TC27] wedge01 volume FAILED'

# ---- TC28: 2D Gauss-Legendre integrates x^2*y^2 exactly on [0,1]^2 ----
def f28(x, y): return x**2 * y**2
val28 = multidimensional_gauss_legendre(f28, 2, 4, [(0.0, 1.0), (0.0, 1.0)])
assert abs(val28 - 1.0/9.0) < 1e-12, '[TC28] 2D Gauss FAILED'

# ---- TC29: flux_quantization_phase for n_phi=3, n_e=1 gives phase=2*pi/3 ----
phase, charge = flux_quantization_phase(3, 1, m_laughlin=3)
assert abs(phase - 2.0*np.pi/3.0) < 1e-12, '[TC29] flux phase FAILED'
assert abs(charge - 1.0/3.0) < 1e-12, '[TC29] flux charge FAILED'

# ---- TC30: orbital_evolution_parameters returns arrays of length n_points ----
tau, theta, x2, y2, z2 = orbital_evolution_parameters(0.1, 30.0, 20.0, n_points=60)
assert len(tau) == 60 and len(theta) == 60, '[TC30] orbital length FAILED'

# ---- TC31: qmc_integration of constant function equals domain_volume ----
pts_qmc = hammersley_sequence(0, 99, 2)
def f_const(pt): return 1.0
I_qmc = qmc_integration(f_const, pts_qmc, domain_volume=3.5)
assert abs(I_qmc - 3.5) < 1e-12, '[TC31] QMC constant FAILED'

# ---- TC32: two_point_correlation g2 is non-negative ----
np.random.seed(42)
z_corr = np.array([0.5+0j, -0.5+0j, 0.0+0.5j, 0.0-0.5j])
r_e2, g2, r_c2 = two_point_correlation(z_corr, 1.0, r_bins=10)
assert np.all(g2 >= 0), '[TC32] two_point_correlation FAILED'

# ---- TC33: spectral stiffness matrix is symmetric ----
K_mat = build_spectral_stiffness_matrix(6, domain=(0.0, 1.0))
assert np.allclose(K_mat, K_mat.T), '[TC33] stiffness symmetry FAILED'

# ---- TC34: Lagrange basis equals identity at nodes ----
nodes = np.array([0.0, 0.5, 1.0])
phi_nodes = local_basis_1d_lagrange(3, nodes, nodes)
assert np.allclose(phi_nodes, np.eye(3)), '[TC34] Lagrange Kronecker FAILED'

# ---- TC35: coulomb_interaction_2d is finite at r=0 due to cutoff ----
V0 = coulomb_interaction_2d(0.0, epsilon_r=12.0)
assert np.isfinite(V0), '[TC35] Coulomb r=0 FAILED'

# ---- TC36: edge_state_density_of_states is non-negative ----
omega_test = np.linspace(-1.0, 1.0, 20)
dos_edge = edge_state_density_of_states(omega_test, 1.0, 10.0, T=0.01)
assert np.all(dos_edge >= 0), '[TC36] edge DOS negative FAILED'

# ---- TC37: tknn_conductivity matches chern_number_from_berry_curvature ----
Omega_dummy = np.ones((2, 2))
C_dummy = chern_number_from_berry_curvature(Omega_dummy, 1.0, 1.0)
sigma_dummy = tknn_conductivity(np.sum(Omega_dummy), 1.0, 1.0)
assert abs(sigma_dummy - C_dummy) < 1e-12, '[TC37] TKNN consistency FAILED'

# ---- TC38: berry_connection returns finite for normalized states ----
u_a = np.array([1.0, 0.0])
u_b = np.array([0.0, 1.0])
A_conn = berry_connection(u_a, u_b, 0.1)
assert np.isfinite(A_conn), '[TC38] berry_connection FAILED'

# ---- TC39: quasihole_wavefunction with return_log returns complex ----
np.random.seed(42)
z_qh = np.array([1.0+0j, 0.5+0.5j])
log_psi_qh = quasihole_wavefunction(z_qh, 0.2+0.1j, 3, 1.0, return_log=True)
assert isinstance(log_psi_qh, (complex, np.complexfloating)), '[TC39] quasihole type FAILED'

# ---- TC40: hypercube_surface_distance_stats reproducible with fixed seed ----
mu1_c, var1_c, _ = hypercube_surface_distance_stats(2, 50, seed=42)
mu2_c, var2_c, _ = hypercube_surface_distance_stats(2, 50, seed=42)
assert abs(mu1_c - mu2_c) < 1e-12, '[TC40] hypercube repro FAILED'

print('\n全部 40 个测试通过!\n')
