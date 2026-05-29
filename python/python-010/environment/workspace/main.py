#!/usr/bin/env python3
"""
main.py
=======
宇宙大尺度结构 N 体模拟统一入口

零参数直接运行，执行以下完整科学计算流程:
    1. ΛCDM 宇宙学背景初始化（尺度因子、Hubble 参数、线性增长因子）
    2. Zeldovich 近似生成高斯随机场初始条件
    3. 粒子网格（PM）N 体演化（Leapfrog + RK12 自适应步长测试）
    4. 密度场对流方程数值演化（Lax-Wendroff 守恒性检验）
    5. 功率谱与相关函数估计
    6. 暗物质晕识别（FOF + 球形过密度）
    7. 质量函数统计与 Press-Schechter 理论比较
    8. 球面方向采样与各向同性检验
    9. Monte Carlo 多维积分验证
    10. 数值鲁棒性与能量守恒性评估

科学问题
--------
在 ΛCDM 平坦宇宙学框架下，追踪 4096 个暗物质粒子从红移 z≈50 到 z=0 的非线性引力演化，
研究宇宙大尺度结构的形成过程，包括：
    - 密度扰动的线性增长与非线性坍缩
    - 功率谱 P(k) 的演化
    - 暗物质晕的质量函数 dn/dM
    - 结构形成的统计各向同性
"""

import numpy as np
import time

from cosmology import Cosmology
from initial_conditions import (
    PowerSpectrum,
    generate_zeldovich_displacement,
    particle_mass_from_cosmology,
    latin_edge_sample,
    gauss_hermite_nodes_weights,
)
from pm_solver import PMSolver
from nbody_integrator import NBodyIntegrator
from density_field import AdvectionSolver, test_mass_conservation
from power_spectrum import (
    PowerSpectrumEstimator,
    monte_carlo_nd_integral,
    press_schechter_mass_function,
    compute_sigma_r,
    spherical_overdensity_criterion,
)
from halo_finder import (
    HaloFinder,
    level_set_volume_analysis,
    sample_sphere_positive_distance,
    angular_distance_histogram,
    halo_mass_function_from_groups,
)
from statistics import (
    alnorm,
    sample_discrete_cdf,
    casino_random_walk,
    variance_from_power_spectrum,
    tophat_window,
)
from linalg_utils import DenseLU, SparseCRS, build_laplacian_1d, solve_tridiagonal


def run_simulation():
    print("=" * 70)
    print("宇宙大尺度结构 N 体模拟 —— ΛCDM 粒子网格演化")
    print("=" * 70)
    t_start = time.time()

    # =================================================================
    # 1. 宇宙学参数初始化
    # =================================================================
    print("\n[1/10] 初始化 ΛCDM 宇宙学参数...")
    cosmo = Cosmology(
        h=0.6732,
        Omega_m=0.3158,
        Omega_b=0.0494,
        Omega_Lambda=0.6842,
        Omega_r=9.2e-5,
        T_cmb=2.7255,
        sigma8=0.811,
        ns=0.965,
    )
    print(f"  H0 = {cosmo.H0:.2f} km/s/Mpc")
    print(f"  Ω_m = {cosmo.Omega_m:.4f}, Ω_Λ = {cosmo.Omega_Lambda:.4f}")
    print(f"  宇宙年龄（a=1）≈ {cosmo.age_of_universe(a_target=1.0):.3f} Gyr")
    print(f"  δ_c(z=0) ≈ {cosmo.delta_c(z=0.0):.4f}")

    # 线性增长因子
    a_arr, D_arr, _ = cosmo.compute_linear_growth_factor(
        a_min=1e-4, a_max=1.0, n_steps=2000
    )
    D_growth = np.interp(1.0, a_arr, D_arr)
    print(f"  线性增长因子 D(z=0) = {D_growth:.4f}")

    # =================================================================
    # 2. 初始条件生成（Zeldovich 近似）
    # =================================================================
    print("\n[2/10] 生成高斯随机场初始条件（Zeldovich 近似）...")
    N = 16
    L = 100.0  # Mpc/h
    ps = PowerSpectrum(cosmo)
    pos, vel, delta_ic = generate_zeldovich_displacement(
        N, L, ps, D_growth=D_growth
    )
    n_part = pos.shape[0]
    m_part = particle_mass_from_cosmology(N, L, cosmo)
    mass = np.full(n_part, m_part)
    rho_mean = cosmo.Omega_m * cosmo.rho_crit_0
    print(f"  粒子数: {n_part}")
    print(f"  单粒子质量: {m_part:.3e} M⊙")
    print(f"  初始密度场 std: {delta_ic.std():.4f}")

    # Latin Edge 采样测试（宇宙学参数空间均匀性）
    latin_samples = latin_edge_sample(dim_num=3, point_num=8)
    print(f"  Latin edge 采样完成: shape={latin_samples.shape}")

    # Gauss-Hermite 节点验证速度分布
    gh_nodes, gh_weights = gauss_hermite_nodes_weights(16)
    # 验证 ∫ exp(-x²) dx = √π
    gh_integral = np.sum(gh_weights)
    print(f"  Gauss-Hermite 求积验证: ∫exp(-x²)dx = {gh_integral:.8f} (理论 √π={np.sqrt(np.pi):.8f})")

    # =================================================================
    # 3. N 体 PM 演化
    # =================================================================
    print("\n[3/10] PM N 体演化（Leapfrog 积分）...")
    solver = PMSolver(N, L, G=cosmo.G)
    integrator = NBodyIntegrator(
        cosmo, softening=0.5, eta=0.2, use_adaptive_step=False
    )

    def compute_acc(p):
        # TODO: 实现加速度计算，需考虑随时间变化的尺度因子 a(t)
        raise NotImplementedError("请实现 compute_acc 函数")

    t_arr, pos_arr, vel_arr, acc_arr = integrator.evolve(
        pos, vel, t_span=(0.0, 1.0), L=L, compute_acc=compute_acc, n_steps=20
    )
    pos_final = pos_arr[-1]
    vel_final = vel_arr[-1]
    print(f"  演化步数: {len(t_arr) - 1}")
    print(f"  最终位置均值: {pos_final.mean(axis=0)}")
    print(f"  最终速度 rms: {np.sqrt((vel_final ** 2).mean()):.4f} km/s")

    # =================================================================
    # 4. 密度场对流测试（Lax-Wendroff）
    # =================================================================
    print("\n[4/10] 密度场 Lax-Wendroff 对流演化测试...")
    rho_final = solver.cic_deposit(pos_final, mass)
    delta_final = solver.compute_density_contrast(rho_final, rho_mean)
    # 对流测试：以恒定速度平移密度场并检验质量守恒
    advect = AdvectionSolver(nx=N, dx=L / N, c=1.0)
    t_adv, rho_adv = advect.evolve_3d_density_field(
        rho_final.copy(), t_final=0.05, n_steps=20
    )
    mass0 = rho_final.sum()
    massf = rho_adv[-1].sum()
    rel_err = abs(massf - mass0) / abs(mass0)
    print(f"  初始总质量: {mass0:.6e}")
    print(f"  最终总质量: {massf:.6e}")
    print(f"  质量守恒相对误差: {rel_err:.4e}")

    # =================================================================
    # 5. 功率谱与相关函数
    # =================================================================
    print("\n[5/10] 估计功率谱 P(k) 与相关函数 ξ(r)...")
    estimator = PowerSpectrumEstimator(N, L)
    k_bins, Pk, Nmodes = estimator.estimate(delta_final, n_bins=N // 2)
    # 只保留有模式的 bin
    mask = Nmodes > 0
    k_bins = k_bins[mask]
    Pk = Pk[mask]
    print(f"  k 范围: [{k_bins.min():.4f}, {k_bins.max():.4f}] h/Mpc")
    print(f"  P(k) 范围: [{Pk.min():.4e}, {Pk.max():.4e}] (Mpc/h)³")

    r_bins, xi = estimator.compute_correlation_function(delta_final)
    print(f"  ξ(r=0) ≈ {xi[0]:.4f} (应接近 σ²)")

    # =================================================================
    # 6. 暗物质晕识别
    # =================================================================
    print("\n[6/10] 暗物质晕识别（FOF + 球形过密度）...")
    finder = HaloFinder(L=L, linking_length=0.2 * (L ** 3 / n_part) ** (1.0 / 3.0))
    groups, halo_mass = finder.fof_groups(pos_final, mass)
    n_halos = len(groups)
    print(f"  识别到 {n_halos} 个 FOF 晕")
    if n_halos > 0:
        print(f"  最大晕质量: {halo_mass.max():.3e} M⊙")
        print(f"  平均晕质量: {halo_mass.mean():.3e} M⊙")

        # 球形过密度质量（取最大晕）
        max_idx = np.argmax(halo_mass)
        center = pos_final[groups[max_idx]].mean(axis=0)
        M_SO, R_SO = finder.spherical_overdensity_mass(
            pos_final, mass, center, rho_crit=cosmo.rho_crit_0, Delta=200.0
        )
        print(f"  最大晕的 M_200 = {M_SO:.3e} M⊙, R_200 = {R_SO:.3f} Mpc/h")

    # =================================================================
    # 7. 质量函数统计与 Press-Schechter 理论
    # =================================================================
    print("\n[7/10] 晕质量函数统计...")
    if n_halos > 5:
        logM_bins, dn_dlnM, err = halo_mass_function_from_groups(
            halo_mass, volume=L ** 3, n_bins=8
        )
        print(f"  质量函数 bin 数: {len(logM_bins)}")
        print(f"  dn/dlnM (最大值): {dn_dlnM.max():.4e} M⊙⁻¹ (Mpc/h)⁻³")

        # Press-Schechter 理论比较
        M_theory = np.logspace(11, 15, 50)
        # 近似 σ(M) 关系
        sigma_theory = 2.0 * (M_theory / 1e14) ** (-0.2)
        nM_theory = press_schechter_mass_function(
            M_theory, sigma_theory, rho_mean, delta_c=cosmo.delta_c(z=0.0)
        )
        print(f"  Press-Schechter 理论 n(M=1e14) = {np.interp(1e14, M_theory, nM_theory):.4e}")
    else:
        print("  晕数量不足，跳过质量函数统计")

    # =================================================================
    # 8. 水平集分析与各向同性检验
    # =================================================================
    print("\n[8/10] 密度场水平集分析与各向同性检验...")
    levels, volumes = level_set_volume_analysis(delta_final, L, n_levels=12)
    print(f"  水平集阈值范围: [{levels.min():.4f}, {levels.max():.4f}]")
    print(f"  最大过密度体积占比: {volumes[0]:.4f}")

    # 球面方向采样
    dirs = sample_sphere_positive_distance(500)
    bc, pdf = angular_distance_histogram(dirs, n_bins=10)
    mean_angle = bc.mean()
    print(f"  随机方向角距离均值: {mean_angle:.4f} (理论 π/2={np.pi/2:.4f})")

    # =================================================================
    # 9. 统计与 Monte Carlo 积分验证
    # =================================================================
    print("\n[9/10] 统计检验与 Monte Carlo 积分...")
    # 正态分布 CDF 检验（融入 asa005）
    phi_0 = alnorm(0.0)
    phi_2 = alnorm(2.0)
    print(f"  alnorm(0) = {phi_0:.6f} (理论 0.5)")
    print(f"  alnorm(2) = {phi_2:.6f} (理论 0.9772)")

    # 离散采样检验（融入 fair_dice_simulation）
    pmf = np.array([1, 2, 3, 4, 5, 6]) / 21.0
    samples = sample_discrete_cdf(10000, pmf)
    print(f"  离散采样均值: {samples.mean():.4f} (理论 3.333)")

    # Casino 随机行走（融入 casino_simulation）
    traj, w, l = casino_random_walk(1.0, 100)
    print(f"  Casino 随机行走最终值: {traj[-1]:.4f}, wins={w}, losses={l}")

    # Monte Carlo 多维积分（融入 nintlib）
    def unit_cube_integrand(x):
        return np.prod(x ** 2)

    mc_val, mc_err = monte_carlo_nd_integral(
        unit_cube_integrand, 3, [0, 0, 0], [1, 1, 1], 50000
    )
    print(f"  Monte Carlo ∫ x²y²z² dV = {mc_val:.6f} ± {mc_err:.6f} (理论 1/27={1/27:.6f})")

    # =================================================================
    # 10. 数值鲁棒性与线性代数检验
    # =================================================================
    print("\n[10/10] 数值鲁棒性与线性代数检验...")
    # 稠密 LU 分解检验（融入 r8ge）
    A_test = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
    lu = DenseLU(A_test)
    b_test = np.array([1.0, 2.0, 3.0])
    x_lu = lu.solve(b_test)
    resid = np.linalg.norm(A_test @ x_lu - b_test)
    print(f"  DenseLU 残差: {resid:.4e}")

    # 稀疏矩阵检验（融入 r8crs）
    crs = SparseCRS.from_dense(A_test)
    y_crs = crs.matvec(b_test)
    diff_sparse = np.linalg.norm(A_test @ b_test - y_crs)
    print(f"  SparseCRS matvec 差: {diff_sparse:.4e}")

    # 三对角求解检验（Thomas 算法）
    n_tri = 100
    a_tri = np.ones(n_tri)
    b_tri = -2.0 * np.ones(n_tri)
    c_tri = np.ones(n_tri)
    d_tri = np.ones(n_tri)
    d_tri[0] = d_tri[-1] = 0.0  # 边界条件
    b_tri[0] = b_tri[-1] = 1.0
    a_tri[0] = c_tri[-1] = 0.0
    x_tri = solve_tridiagonal(a_tri, b_tri, c_tri, d_tri)
    # 逐元素验证 Ax = d
    resid_tri = 0.0
    for i in range(n_tri):
        ax_i = b_tri[i] * x_tri[i]
        if i > 0:
            ax_i += a_tri[i] * x_tri[i - 1]
        if i < n_tri - 1:
            ax_i += c_tri[i] * x_tri[i + 1]
        resid_tri = max(resid_tri, abs(ax_i - d_tri[i]))
    print(f"  Thomas 算法残差: {resid_tri:.4e}")

    # =================================================================
    # 总结
    # =================================================================
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("模拟完成")
    print(f"总耗时: {t_elapsed:.2f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation()
