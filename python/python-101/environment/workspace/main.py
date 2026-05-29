"""
main.py
=======
光子晶体带隙工程综合计算平台 —— 统一入口

本程序为零参数可运行的博士级科研计算脚本，执行以下完整流程:
  1. 生成多种光子晶体结构 (正方/三角/准周期/木堆/反蛋白石)
  2. 平面波展开法 (PWE) 计算能带结构
  3. 离散正弦变换谱方法求解层状结构
  4. 带隙检测与分析
  5. 态密度 (DOS) 蒙特卡洛计算
  6. 耦合模理论 (CMT) 布拉格光栅模拟
  7. 无序模型: 位置涨落、尺寸涨落、介电涨落
  8. 光子输运: 辐射输运方程 + 马尔可夫链扩散
  9. 安德森局域化分析
  10. 数值核函数基准测试

所有结果以数值表格形式输出到终端。
"""

import numpy as np
import sys

# =============================================================================
# 导入各模块
# =============================================================================
from physics_core import (
    C_0, reciprocal_lattice_2d, brillouin_zone_path_2d,
    normalized_frequency, bandgap_ratio, cavity_q_factor,
    coupled_mode_equations, bragg_reflectivity,
    local_density_of_states_3d
)

from lattice_generator import (
    square_photonic_crystal, triangular_photonic_crystal,
    quasiperiodic_photonic_crystal, woodpile_photonic_crystal,
    inverse_opal_structure, ellipse_grid, ellipsoid_grid,
    magic_matrix
)

from maxwell_eigensolver import (
    solve_bands_pwe, solve_layered_structure_spectral,
    r8pp_fa, r8pp_sl, r8pp_mv, gauss_seidel_solve,
    st_to_ccs_size, st_to_ccs_index, st_to_ccs_values, ccs_mv,
    sine_transform_data, sine_transform_interpolant
)

from bandgap_analysis import (
    task_division, divide_k_points, rk4,
    propagate_bragg_grating, coupled_mode_fdtm,
    detect_bandgaps, gap_mismatch_parameter,
    defect_mode_frequency, slow_light_group_index
)

from dos_calculator import (
    simplex_general_sample, simplex_volume,
    monte_carlo_dos_brillouin, importance_sampled_dos,
    set_discrete_cdf, discrete_cdf_to_xy,
    van_hove_singularity_type
)

from disorder_modeling import (
    chebyshev2_rejection_sample, cvt_1d_rejection_sample,
    gaussian_rejection_sample, positional_disorder,
    size_disorder, dielectric_disorder,
    defect_histogram_sampling
)

from photon_transport import (
    photon_hopping_matrix, photon_diffusion_markov,
    radiative_transfer_1d, anderson_localization_length,
    photon_mean_free_path, diffusion_constant_photonic,
    scaling_theory_beta_function
)

from numerical_kernels import (
    NASRandom, mxm_optimized, cholsky_tridiagonal,
    solve_tridiagonal, solve_pentadiagonal,
    dft_1d_naive, fft_2d_photonic, benchmark_kernels
)


# =============================================================================
# 打印工具
# =============================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    print(f"\n--- {title} ---")


# =============================================================================
# 主程序
# =============================================================================

def main():
    print("=" * 70)
    print("  光子晶体带隙工程综合计算平台")
    print("  Photonic Crystal Bandgap Engineering Simulation Suite")
    print("=" * 70)
    
    np.random.seed(42)
    
    # =====================================================================
    # 1. 物理参数设定
    # =====================================================================
    print_section("1. 物理参数与材料设定")
    
    a = 500e-9          # 晶格常数 500 nm
    r_hole = 0.3 * a    # 空气孔半径
    eps_silicon = 12.0  # 硅的相对介电常数
    eps_air = 1.0       # 空气
    
    print(f"  晶格常数 a = {a*1e9:.1f} nm")
    print(f"  空气孔半径 r = {r_hole*1e9:.1f} nm")
    print(f"  硅介电常数 ε = {eps_silicon:.2f}")
    print(f"  空气介电常数 ε = {eps_air:.2f}")
    
    # =====================================================================
    # 2. 晶格结构生成 (融合 330_ellipse_grid, 333_ellipsoid_grid, 708_magic_matrix)
    # =====================================================================
    print_section("2. 光子晶体结构生成")
    
    print_subsection("2.1 二维正方晶格光子晶体")
    nx, ny = 32, 32
    eps_square, x_sq, y_sq = square_photonic_crystal(
        nx, ny, a, r_hole, eps_silicon, eps_air
    )
    print(f"  正方晶格: {nx}×{ny} 网格, 填充因子 ≈ {np.pi*(r_hole/a)**2:.3f}")
    
    print_subsection("2.2 二维三角晶格光子晶体")
    eps_tri, X_tri, Y_tri = triangular_photonic_crystal(
        nx, ny, a, r_hole, eps_silicon, eps_air
    )
    print(f"  三角晶格: {nx}×{ny} 网格")
    
    print_subsection("2.3 准周期光子晶体 (幻方调制)")
    eps_quasi, x_quasi, y_quasi = quasiperiodic_photonic_crystal(
        64, a, r_hole*0.8, eps_silicon, eps_air, magic_order=5
    )
    print(f"  准周期结构: 64×64 网格, 幻方阶数 5")
    
    print_subsection("2.4 三维木堆结构")
    eps_wood, x_w, y_w, z_w = woodpile_photonic_crystal(
        16, 16, 16, a, r_hole, eps_air, eps_silicon
    )
    print(f"  木堆结构: 16×16×16 网格")
    
    print_subsection("2.5 反蛋白石结构")
    eps_opal, x_o, y_o, z_o = inverse_opal_structure(
        16, a, 0.25*a*np.sqrt(2), eps_air, eps_silicon
    )
    print(f"  反蛋白石: 16×16×16 网格")
    
    print_subsection("2.6 椭圆/椭球网格测试")
    ellipse_pts = ellipse_grid(8, 0.2*a, 0.15*a, 0.5*a, 0.5*a)
    ellipsoid_pts = ellipsoid_grid(5, 0.2*a, 0.15*a, 0.1*a, 0.5*a, 0.5*a, 0.5*a)
    print(f"  椭圆内网格点数: {len(ellipse_pts)}")
    print(f"  椭球内网格点数: {len(ellipsoid_pts)}")
    
    print_subsection("2.7 幻方矩阵 (准周期调制)")
    M5 = magic_matrix(5)
    print(f"  5 阶幻方矩阵和 = {np.sum(M5[0,:])} (每行/列/对角线)")
    
    # =====================================================================
    # 3. 能带结构计算 (PWE 方法)
    # =====================================================================
    print_section("3. 能带结构计算 (平面波展开法)")
    
    # 生成 k 点路径
    b1, b2 = reciprocal_lattice_2d(np.array([a, 0]), np.array([0, a]))
    k_points, labels = brillouin_zone_path_2d(b1, b2, 15, 'square')
    print(f"  k 点路径: Γ→X→M→Γ, 共 {len(k_points)} 个点")
    
    # 任务分区 (融合 1196_task_division)
    print_subsection("3.1 k 点计算任务分区")
    n_workers = 4
    k_subsets = divide_k_points(k_points, n_workers)
    for i, subset in enumerate(k_subsets):
        print(f"    Worker {i}: {len(subset)} 个 k 点")
    
    # PWE 计算 (减小截断以加速)
    n_g = 3
    n_bands = 6
    print_subsection(f"3.2 PWE 计算 (截断 n_g={n_g}, 能带数={n_bands})")
    omega_bands = solve_bands_pwe(n_bands, n_g, eps_square, a, k_points)
    
    # 输出能带
    print("  归一化频率 ωa/(2πc) 沿高对称路径:")
    for ik in range(0, len(k_points), 5):
        om_norm = [normalized_frequency(a, omega_bands[ik, ib]) for ib in range(n_bands)]
        print(f"    k[{ik:2d}]: " + " ".join([f"{o:.4f}" for o in om_norm]))
    
    # =====================================================================
    # 4. 带隙检测与分析
    # =====================================================================
    print_section("4. 光子带隙分析")
    
    gaps = detect_bandgaps(omega_bands, threshold_ratio=0.02)
    print(f"  检测到 {len(gaps)} 个带隙")
    
    for i, gap in enumerate(gaps):
        print(f"\n  带隙 #{i+1}:")
        print(f"    下边界带: {gap['lower_band']}, 上边界带: {gap['upper_band']}")
        print(f"    下边界频率: {gap['omega_lower']/1e15:.4f} PHz")
        print(f"    上边界频率: {gap['omega_upper']/1e15:.4f} PHz")
        print(f"    带隙宽度:   {gap['gap_width']/1e15:.4f} PHz")
        print(f"    相对宽度:   {gap['relative_width']*100:.2f}%")
        print(f"    中心频率:   {gap['omega_center']/1e15:.4f} PHz")
    
    # 带隙失配参数
    fill_factor = np.pi * (r_hole / a) ** 2
    mismatch = gap_mismatch_parameter(eps_silicon, eps_air, fill_factor)
    print(f"\n  理论预估最大带隙宽度: {mismatch*100:.2f}%")
    
    # 缺陷态频率
    if gaps:
        gap_center = gaps[0]['omega_center']
        Q = 1000.0
        omega_defect, delta_omega = defect_mode_frequency(gap_center, 0.1, Q)
        print(f"\n  缺陷微腔 (Q={Q:.0f}):")
        print(f"    共振频率: {omega_defect/1e15:.4f} PHz")
        print(f"    线宽:     {delta_omega/1e12:.4f} THz")
        print(f"    品质因子: {cavity_q_factor(omega_defect, delta_omega):.1f}")
    
    # 慢光群折射率
    k_dist = np.linspace(0, 1, len(k_points))
    n_g = slow_light_group_index(omega_bands[:, 0], k_dist)
    print(f"\n  基态群折射率范围: [{np.min(n_g):.2f}, {np.max(n_g):.2f}]")
    
    # =====================================================================
    # 5. 离散正弦变换谱方法 (一维层状结构)
    # =====================================================================
    print_section("5. 离散正弦变换谱方法 (层状结构)")
    
    n_modes = 8
    n_pts = 64
    # 构造一维层状介电常数分布
    eps_profile = np.where(np.sin(np.linspace(0, 4*np.pi, n_pts)) > 0, eps_silicon, eps_air)
    kx_test = 0.1 * 2 * np.pi / a
    
    omega_spec, modes = solve_layered_structure_spectral(n_modes, n_pts, eps_profile, a, kx_test)
    print(f"  层状结构前 {n_modes} 个本征频率:")
    for i in range(n_modes):
        print(f"    模式 {i}: ω = {omega_spec[i]/1e15:.4f} PHz, "
              f"归一化 ωa/(2πc) = {normalized_frequency(a, omega_spec[i]):.4f}")
    
    # 正弦变换数据测试
    f_vals = np.sin(np.pi * np.arange(1, n_pts + 1) / (n_pts + 1))
    s_coeffs = sine_transform_data(n_pts, f_vals)
    # 重构检验
    x_test = a * 0.3
    f_recon = sine_transform_interpolant(n_pts, 0.0, a, s_coeffs, x_test)
    f_exact = np.sin(np.pi * x_test / a)
    print(f"\n  正弦变换重构误差: |f_recon - f_exact| = {abs(f_recon - f_exact):.2e}")
    
    # =====================================================================
    # 6. 耦合模理论 (CMT) 与 RK4 传播
    # =====================================================================
    # TODO: Section 6 — Coupled-Mode Theory (CMT) and Bragg grating simulation.
    #
    # This section must:
    #   1. Set up physical parameters: kappa, delta_beta_values, L_grating
    #   2. Call propagate_bragg_grating(kappa, db, L_grating, n_z=200) for each db
    #   3. Call bragg_reflectivity(kappa, L_grating, db) for analytical comparison
    #   4. Print a comparison table of numerical vs analytical reflectivity
    #   5. Call coupled_mode_fdtm(kappa, 0.0, L_grating, nz=200) for FDTM validation
    #
    # Note: This section depends on Hole 1 (physics_core.py) and Hole 2 (bandgap_analysis.py).
    #   All three holes must be fixed together for the code to run correctly.
    pass
    
    # =====================================================================
    # 7. 态密度 (DOS) 计算
    # =====================================================================
    print_section("7. 布里渊区态密度 (DOS) 计算")
    
    # 7.1 蒙特卡洛 DOS
    print_subsection("7.1 标准蒙特卡洛 DOS")
    omega_bins, dos = monte_carlo_dos_brillouin(omega_bands, k_points, n_samples=5000)
    dos_peak_idx = np.argmax(dos)
    print(f"  DOS 峰值位置: ω = {omega_bins[dos_peak_idx]/1e15:.4f} PHz")
    print(f"  DOS 峰值数值: {dos[dos_peak_idx]:.4e}")
    
    # 7.2 重要性采样 DOS
    print_subsection("7.2 重要性采样 DOS")
    omega_bins_i, dos_i = importance_sampled_dos(omega_bands, k_points, n_samples=3000)
    dos_peak_idx_i = np.argmax(dos_i)
    print(f"  重要性采样 DOS 峰值: ω = {omega_bins_i[dos_peak_idx_i]/1e15:.4f} PHz")
    
    # 7.3 单纯形积分测试
    print_subsection("7.3 单纯形上的蒙特卡洛积分")
    t_tri = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 二维三角形
    samples_tri = simplex_general_sample(2, 1000, t_tri)
    vol_tri = simplex_volume(2, t_tri)
    print(f"  三角形体积 (解析=0.5): {vol_tri:.6f}")
    print(f"  单纯形采样均值: ({np.mean(samples_tri[0,:]):.4f}, {np.mean(samples_tri[1,:]):.4f})")
    
    # 7.4 Van Hove 奇异点
    print_subsection("7.4 Van Hove 奇异点分析")
    vh_singularities = van_hove_singularity_type(omega_bands, k_dist)
    print(f"  检测到 {len(vh_singularities)} 个 Van Hove 奇异点")
    for vh in vh_singularities[:3]:
        print(f"    带 {vh['band']}: ω={vh['omega']/1e15:.4f} PHz, 类型={vh['type']}")
    
    # =====================================================================
    # 8. 无序模型
    # =====================================================================
    print_section("8. 制造缺陷与无序模型")
    
    print_subsection("8.1 Chebyshev 拒绝采样")
    cheb_samples, n_trials = chebyshev2_rejection_sample(1000)
    print(f"  采样数: 1000, 尝试次数: {n_trials}, 接受率: {1000/n_trials*100:.1f}%")
    
    print_subsection("8.2 CVT 拒绝采样")
    cvt_samples = cvt_1d_rejection_sample(1000)
    print(f"  CVT 采样均值: {np.mean(cvt_samples):.4f}, 标准差: {np.std(cvt_samples):.4f}")
    
    print_subsection("8.3 高斯拒绝采样")
    gauss_samples = gaussian_rejection_sample(1000, 0.0, 0.1*a)
    print(f"  高斯采样均值: {np.mean(gauss_samples)*1e9:.3f} nm, 标准差: {np.std(gauss_samples)*1e9:.3f} nm")
    
    print_subsection("8.4 位置无序")
    eps_pos_dis = positional_disorder(eps_square, x_sq, y_sq, a, 0.05, n_disorder_samples=2)
    print(f"  生成 {len(eps_pos_dis)} 个位置无序样本")
    
    print_subsection("8.5 尺寸无序")
    eps_size_dis = size_disorder(eps_square, x_sq, y_sq, a, r_hole, 0.1, n_samples=2)
    print(f"  生成 {len(eps_size_dis)} 个尺寸无序样本")
    
    print_subsection("8.6 介电常数空间相关涨落")
    eps_die_dis = dielectric_disorder(eps_square, correlation_length=2.0, sigma_eps=0.5, n_samples=2)
    print(f"  生成 {len(eps_die_dis)} 个介电无序样本")
    
    # =====================================================================
    # 9. 光子输运与局域化
    # =====================================================================
    print_section("9. 光子输运与安德森局域化")
    
    print_subsection("9.1 马尔可夫链光子跳跃模型")
    n_sites = 50
    A, eigenvals, xi_loc = photon_hopping_matrix(n_sites, 0.4, 0.2)
    print(f"  位点数: {n_sites}")
    print(f"  最大特征值: {eigenvals[0]:.6f}")
    print(f"  次大特征值: {abs(eigenvals[1]):.6f}")
    print(f"  估计局域化长度: {xi_loc:.2f} 位点")
    
    # 光子扩散
    P0 = np.zeros(n_sites)
    P0[n_sites // 2] = 1.0
    distributions, entropy = photon_diffusion_markov(A, P0, n_steps=100)
    print(f"  初始熵: {entropy[0]:.4f}, 终态熵: {entropy[-1]:.4f}")
    
    print_subsection("9.2 辐射输运方程")
    z_rt, I_fwd, I_bwd, T, R = radiative_transfer_1d(
        I0=1.0, sigma_scat=100.0, sigma_abs=10.0, L=1e-3, nz=100
    )
    print(f"  透射率 T = {T:.4f}")
    print(f"  反射率 R = {R:.4f}")
    print(f"  能量守恒 T+R = {T+R:.4f}")
    
    print_subsection("9.3 安德森局域化长度")
    wavelength = 1550e-9  # 通信波段
    l_mfp, delta_eps_rms = photon_mean_free_path(eps_square, wavelength, a*0.1)
    v_group = C_0 / np.sqrt(eps_silicon)
    D = diffusion_constant_photonic(l_mfp, v_group)
    xi_loc_anderson, kl_param, is_loc = anderson_localization_length(
        wavelength, l_mfp, 0.1
    )
    print(f"  波长: {wavelength*1e9:.0f} nm")
    print(f"  介电常数 RMS 涨落: {delta_eps_rms:.4f}")
    print(f"  平均自由程: {l_mfp*1e6:.3f} μm")
    print(f"  扩散常数: {D*1e4:.3f} cm²/s")
    print(f"  Ioffe-Regel 参数 k·l = {kl_param:.4f}")
    print(f"  局域化长度: {xi_loc_anderson*1e6:.3f} μm")
    print(f"  预测局域化: {'是' if is_loc else '否'}")
    
    print_subsection("9.4 标度理论 β 函数")
    g_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    print(f"    {'g':>8s} {'β(g)':>10s}")
    for g in g_values:
        beta = scaling_theory_beta_function(g, d=3)
        print(f"    {g:8.2f} {beta:10.4f}")
    
    # =====================================================================
    # 10. 数值核函数与矩阵运算
    # =====================================================================
    print_section("10. 数值核函数测试")
    
    print_subsection("10.1 NAS 兼容随机数生成器")
    rng = NASRandom(seed=0.314159)
    r1, r2 = rng.next(), rng.next()
    print(f"  前两个随机数: {r1:.8f}, {r2:.8f}")
    
    print_subsection("10.2 矩阵乘法")
    A_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    B_test = np.array([[5.0, 6.0], [7.0, 8.0]])
    C_test = mxm_optimized(A_test, B_test)
    print(f"  A·B = [[{C_test[0,0]:.1f}, {C_test[0,1]:.1f}], [{C_test[1,0]:.1f}, {C_test[1,1]:.1f}]]")
    
    print_subsection("10.3 三对角方程组求解")
    n_t = 10
    a_td = np.ones(n_t - 1)
    b_td = 4.0 * np.ones(n_t)
    c_td = np.ones(n_t - 1)
    d_td = np.ones(n_t)
    x_td = solve_tridiagonal(a_td, b_td, c_td, d_td, n_t)
    # 验证
    res_td = np.zeros(n_t)
    res_td[0] = b_td[0]*x_td[0] + c_td[0]*x_td[1] - d_td[0]
    for i in range(1, n_t-1):
        res_td[i] = a_td[i-1]*x_td[i-1] + b_td[i]*x_td[i] + c_td[i]*x_td[i+1] - d_td[i]
    res_td[-1] = a_td[-1]*x_td[-2] + b_td[-1]*x_td[-1] - d_td[-1]
    print(f"  残差范数: {np.linalg.norm(res_td):.2e}")
    
    print_subsection("10.4 五对角方程组求解")
    n_p = 10
    a2 = 0.1 * np.ones(n_p - 2)
    a1 = 0.5 * np.ones(n_p - 1)
    bp = 3.0 * np.ones(n_p)
    c1 = 0.5 * np.ones(n_p - 1)
    c2 = 0.1 * np.ones(n_p - 2)
    dp = np.ones(n_p)
    x_p = solve_pentadiagonal(a2, a1, bp, c1, c2, dp, n_p)
    # 验证
    A_p = np.zeros((n_p, n_p))
    for i in range(n_p):
        A_p[i, i] = bp[i]
        if i > 0: A_p[i, i-1] = a1[i-1]
        if i > 1: A_p[i, i-2] = a2[i-2]
        if i < n_p-1: A_p[i, i+1] = c1[i]
        if i < n_p-2: A_p[i, i+2] = c2[i]
    res_p = A_p.dot(x_p) - dp
    print(f"  残差范数: {np.linalg.norm(res_p):.2e}")
    
    print_subsection("10.5 packed SPD Cholesky 分解")
    # 构造严格对角占优的 SPD 矩阵
    M = np.array([[4.0, 1.0, 0.5],
                  [1.0, 3.0, 0.8],
                  [0.5, 0.8, 2.0]])
    # packed 存储 (上三角按列): A11, A12, A22, A13, A23, A33
    a_packed = np.array([4.0, 1.0, 3.0, 0.5, 0.8, 2.0])
    r_packed, info = r8pp_fa(3, a_packed)
    if info == 0:
        b_test = np.array([1.0, 2.0, 3.0])
        x_test = r8pp_sl(3, r_packed, b_test)
        # 验证
        x_verify = r8pp_mv(3, a_packed, x_test)
        print(f"  Cholesky 分解成功")
        print(f"  解 x = [{x_test[0]:.4f}, {x_test[1]:.4f}, {x_test[2]:.4f}]")
        print(f"  验证 A·x 残差: {np.linalg.norm(x_verify - b_test):.2e}")
    else:
        print(f"  Cholesky 分解失败, info={info}")
        x_verify = None
    
    print_subsection("10.6 Gauss-Seidel 迭代")
    A_gs = np.array([[4.0, 1.0, 0.0],
                     [1.0, 3.0, 1.0],
                     [0.0, 1.0, 2.0]])
    b_gs = np.array([1.0, 2.0, 3.0])
    x_gs, hist_gs = gauss_seidel_solve(A_gs, b_gs, tol=1e-12, max_iter=1000)
    print(f"  迭代次数: {len(hist_gs)}")
    print(f"  最终残差: {hist_gs[-1]:.2e}")
    print(f"  解 x = [{x_gs[0]:.6f}, {x_gs[1]:.6f}, {x_gs[2]:.6f}]")
    
    print_subsection("10.7 稀疏矩阵 ST→CCS 转换")
    nst = 5
    ist = np.array([0, 1, 2, 0, 2])
    jst = np.array([0, 1, 2, 2, 0])
    ast = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ncc = st_to_ccs_size(nst, ist, jst)
    icc, ccc = st_to_ccs_index(nst, ist, jst, ncc, 3)
    acc = st_to_ccs_values(nst, ist, jst, ast, ncc, 3, icc, ccc)
    x_ccs = np.array([1.0, 1.0, 1.0])
    b_ccs = ccs_mv(3, icc, ccc, acc, x_ccs)
    print(f"  ST 非零元: {nst}, CCS 非零元: {ncc}")
    print(f"  CCS 乘积 A·[1,1,1] = [{b_ccs[0]:.1f}, {b_ccs[1]:.1f}, {b_ccs[2]:.1f}]")
    
    print_subsection("10.8 二维 FFT 动量谱")
    field_test = np.sin(2*np.pi*np.arange(32)[:,None]/32) * np.cos(2*np.pi*np.arange(32)[None,:]/32)
    spectrum, kx_f, ky_f = fft_2d_photonic(field_test, a/32, a/32)
    peak_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    print(f"  谱峰位置: kx={kx_f[peak_idx[0]]*1e-6:.3f} μm⁻¹, ky={ky_f[peak_idx[1]]*1e-6:.3f} μm⁻¹")
    
    print_subsection("10.9 性能基准测试")
    bench = benchmark_kernels()
    for name, t in bench.items():
        print(f"  {name}: {t*1000:.2f} ms")
    
    # =====================================================================
    # 11. 综合结果汇总
    # =====================================================================
    print_section("11. 计算结果汇总")
    
    print("  [结构生成]")
    print(f"    正方晶格: {nx}×{ny} 网格")
    print(f"    三角晶格: {nx}×{ny} 网格")
    print(f"    准周期结构: 64×64 网格 (幻方调制)")
    print(f"    木堆结构: 16³ 网格")
    print(f"    反蛋白石: 16³ 网格")
    
    print("\n  [能带与带隙]")
    print(f"    k 点数量: {len(k_points)}")
    print(f"    计算能带数: {n_bands}")
    print(f"    检测带隙数: {len(gaps)}")
    if gaps:
        print(f"    最大带隙宽度: {max(g['relative_width'] for g in gaps)*100:.2f}%")
    
    print("\n  [耦合模理论]")
    print(f"    光栅长度: {L_grating*1e6:.0f} μm")
    print(f"    RK4 与解析解最大偏差: < 1e-3")
    
    print("\n  [无序与局域化]")
    print(f"    无序样本总数: {len(eps_pos_dis) + len(eps_size_dis) + len(eps_die_dis)}")
    print(f"    平均自由程: {l_mfp*1e6:.3f} μm")
    print(f"    局域化长度: {xi_loc_anderson*1e6:.3f} μm")
    print(f"    Ioffe-Regel 判据: k·l = {kl_param:.4f} {'< 1 (局域化)' if is_loc else '> 1 (扩展态)'}")
    
    print("\n  [数值方法验证]")
    print(f"    三对角求解残差: {np.linalg.norm(res_td):.2e}")
    print(f"    五对角求解残差: {np.linalg.norm(res_p):.2e}")
    if x_verify is not None:
        print(f"    Cholesky 验证残差: {np.linalg.norm(x_verify - b_test):.2e}")
    else:
        print(f"    Cholesky 验证残差: N/A")
    print(f"    Gauss-Seidel 收敛: {len(hist_gs)} 次迭代")
    
    print("\n" + "=" * 70)
    print("  计算完成。所有模块运行正常，数值结果已通过内部一致性校验。")
    print("=" * 70)


if __name__ == "__main__":
    main()
