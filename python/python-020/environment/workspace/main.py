# -*- coding: utf-8 -*-
import sys
import numpy as np


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


    E_range = np.linspace(0.0, 6.0 * H_BAR * omega_c, 200)
    dos = density_of_states_landau(E_range, B, m_star, gamma=0.05)
    print(f"\n态密度特征:")
    print(f"  最大态密度: {np.max(dos):.4f}")
    print(f"  平均态密度: {np.mean(dos):.4f}")


    print(f"\n谱有限元刚度矩阵条件数:")
    for N in [4, 8, 12]:
        K = build_spectral_stiffness_matrix(N, domain=(0.0, 1.0))
        cond = np.linalg.cond(K)
        print(f"  N={N:2d}: cond(K) = {cond:.4e}")

    return lB, omega_c, N_phi, nu


def run_laughlin_wavefunction_analysis(N_electrons, m, lB, B):
    print_header("模块2: Laughlin多体波函数与准粒子激发")


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


    z0 = 0.5 * lB + 0.3j * lB
    log_psi_qh = quasihole_wavefunction(z, z0, m, lB, return_log=True)
    print(f"\n准空穴激发 (z0 = {z0:.3f}):")
    print(f"  ln|Ψ_qh|² = {log_psi_qh.real:.4f}")
    print(f"  准空穴携带分数电荷 e* = e/{m} = {1.0/m:.6f}")


    r_edges, g_r, r_centers = pair_correlation_function(z, m, lB, r_bins=40)
    print(f"\n配对关联函数 g^(2)(r) 统计:")
    print(f"  r 范围: [{r_centers[0]:.4f}, {r_centers[-1]:.4f}]")
    print(f"  g(r) 最大值: {np.max(g_r):.4f}")
    print(f"  g(r) 最小值: {np.min(g_r):.4f}")


    q_vals, S_q = structure_factor_s_q(z, m, lB, q_bins=25)
    print(f"\n结构因子 S(q):")
    print(f"  q 范围: [{q_vals[0]:.4f}, {q_vals[-1]:.4f}]")
    print(f"  S(q) 范围: [{np.min(S_q):.4f}, {np.max(S_q):.4f}]")

    return z, log_psi, g_r, r_centers


def run_monte_carlo_analysis(N_electrons, m, lB):
    print_header("模块3: 准蒙特卡洛采样与积分")


    n_samples = 500
    h_points = hammersley_sequence(0, n_samples - 1, 2)
    z_samples = map_hammersley_to_disk(h_points, radius=np.sqrt(2.0 * m * N_electrons) * lB * 0.5)
    print(f"\nHammersley低差异序列:")
    print(f"  采样点数: {n_samples}")
    print(f"  映射到圆盘半径: {np.max(np.abs(z_samples)):.4f}")


    def f_integrand(pt):
        x, y = pt[0], pt[1]
        return x ** 2 + y ** 2

    integral_qmc = qmc_integration(f_integrand, h_points, domain_volume=np.pi * (np.sqrt(2.0 * m * N_electrons) * lB * 0.5) ** 2)
    print(f"\n准Monte Carlo积分 (圆盘上 ∫(x²+y²)dA):")
    print(f"  估计值: {integral_qmc:.6f}")

    return z_samples


def run_hartree_fock_analysis(B, lB, grid_res=12):
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


    A_test = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, 1.0]])
    b_test = np.array([4.0, 5.0, 3.0])
    x_cg, conv_cg, res_cg = cg_ne_solve(A_test, b_test, tol=1e-10)
    print(f"\n共轭梯度(CGNE)测试:")
    print(f"  解: {x_cg}")
    print(f"  残差: {res_cg:.2e}")


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
    print_header("模块5: 边缘态BVP求解与手性Luttinger液体")


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


    ode_rhs = edge_state_radial_ode(B, m_star, angular_m=1, E=5.0)
    r_edge, u_edge, conv_edge, nit_edge = shooting_method_bvp(
        ode_rhs, a=1e-3, b=3.0, ya=0.0, yb_target=0.0,
        alpha_guess1=0.1, alpha_guess2=1.0, max_iter=10, tol=1e-3, n_steps=600
    )
    print(f"\n边缘态径向方程:")
    print(f"  收敛: {conv_edge}, 迭代: {nit_edge}")
    print(f"  解范数: {np.linalg.norm(u_edge):.4f}")


    k_vals = np.linspace(0.1, 5.0, 20)
    v_F = 1.0
    eps = chiral_luttinger_dispersion(k_vals, v_F, g_factor=0.5)
    print(f"\n手性Luttinger液体 (g=0.5):")
    print(f"  k=0.1 时 ε = {eps[0]:.4f}")
    print(f"  k=5.0 时 ε = {eps[-1]:.4f}")

    return u_edge


def run_density_evolution():
    print_header("模块6: 密度演化动力学")


    u_fisher = fisher_kpp_fd_solve(
        60, -8.0, 8.0, 150, 0.0, 4.0,
        D=1.0, r=1.0, K=1.0
    )
    print(f"\nFisher-KPP反应扩散方程:")
    print(f"  解形状: {u_fisher.shape}")
    print(f"  初始峰值: {np.max(u_fisher[0,:]):.4f}")
    print(f"  最终峰值: {np.max(u_fisher[-1,:]):.4f}")
    print(f"  总密度守恒: init={np.sum(u_fisher[0,:]):.2f}, final={np.sum(u_fisher[-1,:]):.2f}")


    def u_x1(t): return 0.0
    def u_x2(t): return 0.0
    def u_t1(x): return np.sin(np.pi * x)
    def ut_t1(x): return np.zeros_like(x)
    u_wave = fd1d_wave_solve(40, 0.0, 1.0, 80, 0.0, 2.0, 1.0, u_x1, u_x2, u_t1, ut_t1)
    print(f"\n一维波动方程:")
    print(f"  解形状: {u_wave.shape}")
    print(f"  t=0 最大振幅: {np.max(np.abs(u_wave[0,:])):.4f}")
    print(f"  t=2 最大振幅: {np.max(np.abs(u_wave[-1,:])):.4f}")


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
    print_header("模块7: 关联函数与最近邻分析")


    m_dim, nr, ns = 2, 5, 3
    R_pts = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]])
    S_pts = np.array([[0.3, 2.5, 4.2],
                      [0.0, 0.0, 0.0]])
    idx_nn, dists_nn = find_nearest_neighbors(m_dim, nr, R_pts, ns, S_pts)
    print(f"\n最近邻搜索:")
    print(f"  查询点索引: {idx_nn}")
    print(f"  最小距离: {dists_nn}")


    print(f"\n超球距离统计:")
    for dim in [2, 3]:
        mu, var, _ = hyperball_distance_stats(dim, 200)
        print(f"  dim={dim}: μ={mu:.4f}, σ²={var:.6f}")


    r_edges, g2, r_centers = two_point_correlation(z_samples, lB, r_bins=30)
    print(f"\n两点关联函数:")
    print(f"  g2 前3个值: {g2[:3]}")
    print(f"  g2 最大值: {np.max(g2):.4f}")


    n_grid = np.random.rand(24, 24)
    q_vals, S_q = density_correlation_function(n_grid, dx=0.1, dy=0.1)
    print(f"\n密度结构因子 S(q):")
    print(f"  S(q=0附近) ≈ {S_q[0]:.4f}")

    return g2


def run_quadrature_analysis():
    print_header("模块8: 高维数值积分与精确性验证")


    print(f"\n楔形体精确积分:")
    for e in [[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,2]]:
        val = wedge01_integral(e)
        print(f"  I{e} = {val:.6f}")


    def f2(x, y): return x**2 * y**2
    val2d = multidimensional_gauss_legendre(f2, 2, 4, [(0.0,1.0),(0.0,1.0)])
    print(f"\n二维Gauss积分 (x²y² over [0,1]²):")
    print(f"  数值: {val2d:.8f}, 精确: {1.0/9.0:.8f}")


    n_pts = 10
    x_pts = np.random.rand(3, n_pts)
    x_pts[1,:] *= (1.0 - x_pts[0,:])
    x_pts[2,:] = 2.0 * x_pts[2,:] - 1.0
    w = np.ones(n_pts) / n_pts
    results = wedge_exactness_test(x_pts, w, degree_max=2)
    max_err = max([r[4] for r in results])
    print(f"\n楔形体求积精确性 (degree≤2): max_err = {max_err:.4f}")


    val_coul, exact_coul = integrate_coulomb_2d_gauss(8, 16, epsilon_r=12.0, R_max=1.5)
    print(f"\n二维库仑势积分:")
    print(f"  数值: {val_coul:.6f}, 精确: {exact_coul:.6f}")

    return val2d


def run_topological_analysis():
    print_header("模块9: 拓扑不变量与Chern数")


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


    print(f"\n磁通量子化Berry相位:")
    for n_phi in [3, 6, 9]:
        n_e = n_phi // 3
        phase, charge = flux_quantization_phase(n_phi, n_e, m_laughlin=3)
        print(f"  n_Φ={n_phi}, n_e={n_e}: phase={phase:.4f}, charge={charge:.6f}")


    tau, theta, x2, y2, z2 = orbital_evolution_parameters(
        ecc=0.01671, lon_deg=77.0, obliq_deg=23.44, n_points=50
    )
    print(f"\n绝热参数演化:")
    print(f"  角度范围: [{np.min(theta):.4f}, {np.max(theta):.4f}]")
    print(f"  轨道闭合误差: {abs(np.mean(x2**2+y2**2+z2**2)-1.0):.6f}")

    return C


def main():
    print("\n" + "=" * 70)
    print("  分数量子霍尔效应：Laughlin态的多体数值模拟平台")
    print("  Fractional Quantum Hall Effect: Laughlin State Numerical Platform")
    print("=" * 70)
    print("\n本程序围绕凝聚态物理中的分数量子霍尔效应展开，")
    print("融合15个种子项目的核心算法，实现从单粒子Landau能级")
    print("到多体Laughlin波函数、自洽场、边缘态、拓扑不变量的")
    print("完整数值模拟链路。\n")


    B = 10.0
    m_star = 1.0
    N_electrons = 8
    A_sample = 50.0
    m_Laughlin = 3


    lB, omega_c, N_phi, nu = run_landau_level_analysis(B, m_star, N_electrons, A_sample)
    z, log_psi, g_r, r_centers = run_laughlin_wavefunction_analysis(N_electrons, m_Laughlin, lB, B)
    z_samples = run_monte_carlo_analysis(N_electrons, m_Laughlin, lB)
    energies_hf, density_hf = run_hartree_fock_analysis(B, lB)
    u_edge = run_edge_state_analysis(B, m_star)
    u_fisher = run_density_evolution()
    g2 = run_correlation_analysis(z, lB, m_Laughlin)
    val2d = run_quadrature_analysis()
    C = run_topological_analysis()


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
