"""
main.py — 量子霍尔效应数值对角化：高阶有限差分与稳定性分析
================================================================

统一入口：零参数可运行

本项目融合 15 个种子项目的核心算法, 围绕 "计算凝聚态：量子霍尔效应
数值对角化" 展开, 涵盖:

  (1) 连续域有限差分求解 Landau 能级 (058_atkinson → 高阶 FD)
  (2) 六角格点 Hofstadter 模型 (107_boundary_word_equilateral → 格点)
  (3) Suzuki-Trotter 时间演化 (1015_sheffieldquantum_qsim → Trotter)
  (4) 矩阵性质分析 (737_matrix_analyze → 矩阵分析)
  (5) 哈密顿量并行组装 (738_matrix_assemble_parfor → 组装)
  (6) 态分类与递归分析 (199_collatz_recursive → Collatz)
  (7) 多尺度波函数分析 (575_image_decimate → 降采样)
  (8) Dijkstra 最短路径 (287_dijkstra → 关联路径)
  (9) 分子动力学时间积分 (746_md_parfor → Verlet)
  (10) GPC 贝叶斯推理 (1262_aybo_gpc → 不确定度)
  (11) Fekete 求积 (678_line_fekete_rule → 矩阵元积分)
  (12) Bernstein 多项式 (078_bernstein_polynomial → 分类光滑)
  (13) 边界词分析 (107_boundary_word_equilateral → 边界拓扑)
  (14) SLDS 切换动力学 (1239_DurationModulatedDynamics → 态切换)
  (15) HyperMPC 模型预测 (1177_HyperMPC → 参数优化)
  (16) 网格 I/O (571_ice_to_medit → 格式转换)

物理模型:
  H = (1/2m)(p - A)² + V(r)
  在 Landau 规范 A = (0, Bx, 0) 下:
  H = -(1/2m)∇² - (iBx/m)∂_y + B²x²/(2m) + V(r)

运行方式:
  python main.py
"""

import sys
import os
import time
import numpy as np
from scipy import sparse

# 设置随机种子保证可复现性
np.random.seed(269)

# ─── 添加当前目录到路径 ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_constants import (
    landau_level_energy, magnetic_length, filling_factor,
    peierls_phase, dirac_notation_inner, expectation_value
)
from gauge_field import LandauGauge, SymmetricGauge, HofstadterGauge
from high_order_fd import (
    build_1d_laplacian, build_1d_first_derivative,
    fd_derivative_1d, truncation_error, richardson_extrapolation
)
from hamiltonian_assemble import (
    assemble_hamiltonian_landau, ground_state_energy,
    build_impurity_potential
)
from matrix_analyze_qhe import analyze_hamiltonian, print_analysis
from landau_levels import (
    analytical_landau_levels, numerical_landau_levels,
    landau_wavefunction, landau_level_comparison
)
from stability_analysis import (
    von_neumann_analysis, energy_conservation_test,
    grid_convergence_study, cfl_condition
)
from suzuki_trotter import evolve_step, time_evolution_sequence
from hexagonal_lattice_hofstadter import (
    generate_honeycomb_flake, build_hofstadter_hamiltonian,
    hofstadter_spectrum
)
from wavelet_downsample import (
    participation_ratio, inverse_participation_ratio,
    downsample_2d, haar_wavelet_decompose
)
from uncertainty_quantify import (
    monte_carlo_uncertainty, sensitivity_analysis,
    convergence_extrapolation
)
from observables import (
    local_density_of_states, current_density, particle_density,
    guiding_center, streda_formula_density
)
from mesh_io import save_wavefunction_data, export_xyz_format, generate_report
from state_classifier import (
    classify_states, collatz_classification,
    bernstein_smooth_classification, boundary_word_analysis
)


def separator(title: str) -> str:
    """打印分隔线"""
    line = "=" * 68
    return f"\n{line}\n  {title}\n{line}"


# ═══════════════════════════════════════════════════════════════
# Phase 1: 物理参数与系统设置
# ═══════════════════════════════════════════════════════════════
def phase1_setup():
    """设置物理参数"""
    print(separator("Phase 1: 物理参数设置"))

    # 磁场参数 (原子单位: ℏ = m = e = 1)
    B = 1.0
    l_B = magnetic_length(B)
    omega_c = B  # 回旋频率

    print(f"  磁场 B = {B} (原子单位)")
    print(f"  磁长度 l_B = {l_B:.4f}")
    print(f"  回旋频率 ω_c = {omega_c:.4f}")

    # 系统尺寸
    Lx, Ly = 8.0, 8.0
    Nx, Ny = 20, 20
    hx, hy = Lx / Nx, Ly / Ny

    print(f"  系统尺寸: Lx={Lx}, Ly={Ly}")
    print(f"  格点数: Nx={Nx}, Ny={Ny}")
    print(f"  网格间距: hx={hx:.4f}, hy={hy:.4f}")

    # 解析朗道能级
    n_max = 5
    E_analytical = analytical_landau_levels(B, n_max)
    print(f"\n  解析朗道能级 (前{n_max+1}个):")
    for n, E in enumerate(E_analytical):
        print(f"    E_{n} = B(n+1/2) = {E:.6f}")

    # 规范设置
    gauge = LandauGauge(B)
    print(f"\n  使用规范: {gauge.name}")
    Ax, Ay = gauge.A(1.0, 0.5)
    print(f"  A(1.0, 0.5) = ({Ax:.4f}, {Ay:.4f})")
    curl_B = gauge.curl_A(1.0, 0.5)
    print(f"  ∇×A(1.0, 0.5) = {curl_B:.6f} (应等于 B={B})")

    return {
        'B': B, 'l_B': l_B, 'Lx': Lx, 'Ly': Ly,
        'Nx': Nx, 'Ny': Ny, 'hx': hx, 'hy': hy,
        'E_analytical': E_analytical, 'gauge': gauge
    }


# ═══════════════════════════════════════════════════════════════
# Phase 2: 有限差分格式验证
# ═══════════════════════════════════════════════════════════════
def phase2_fd_validation(params):
    """验证有限差分格式的精度"""
    print(separator("Phase 2: 有限差分格式验证"))

    # 测试函数: f(x) = sin(2πx/L)
    L = 4.0
    N = 50
    h = L / N
    x = np.linspace(0, L, N, endpoint=False)

    f_exact = np.sin(2 * np.pi * x / L)
    # f''(x) = -(2π/L)² sin(2πx/L)
    fdd_exact = -(2 * np.pi / L) ** 2 * f_exact

    print(f"  测试函数: f(x) = sin(2πx/L), L={L}")
    print(f"  格点数: N={N}, h={h:.4f}")

    errors = {}
    for order in [2, 4, 6]:
        fdd_fd = fd_derivative_1d(f_exact, h, order=order, derivative=2)
        error = np.max(np.abs(fdd_fd - fdd_exact))
        errors[order] = error
        print(f"    {order}阶模板: 最大误差 = {error:.6e}")

    # Richardson 外推估计收敛阶
    if len(errors) >= 2:
        orders_used = list(errors.keys())
        p_est = richardson_extrapolation(errors[orders_used[0]],
                                          errors[orders_used[1]],
                                          h, h/2)
        print(f"  Richardson 估计收敛阶: p ≈ {p_est:.2f}")

    # 截断误差测试
    test_func = lambda x: np.exp(-x**2)
    te = truncation_error(test_func, 0.5, h, order=4, derivative=2)
    print(f"  高斯函数在 x=0.5 的截断误差: {te:.6e}")

    return errors


# ═══════════════════════════════════════════════════════════════
# Phase 3: 哈密顿量组装与矩阵分析
# ═══════════════════════════════════════════════════════════════
def phase3_hamiltonian(params):
    """组装哈密顿量并进行矩阵分析"""
    print(separator("Phase 3: 哈密顿量组装与矩阵分析"))

    B = params['B']
    Nx, Ny = params['Nx'], params['Ny']
    Lx, Ly = params['Lx'], params['Ly']

    t0 = time.time()
    H = assemble_hamiltonian_landau(Nx, Ny, Lx, Ly, B, fd_order=4)
    t_build = time.time() - t0

    print(f"  哈密顿量维度: {H.shape}")
    print(f"  非零元数: {H.nnz}")
    print(f"  组装时间: {t_build:.4f}s")

    # 矩阵分析
    results = analyze_hamiltonian(H, "QHE Hamiltonian (Landau gauge)")
    print(print_analysis(results))

    return H, results


# ═══════════════════════════════════════════════════════════════
# Phase 4: 朗道能级求解
# ═══════════════════════════════════════════════════════════════
def phase4_landau_levels(params, H):
    """求解朗道能级并与解析结果比较"""
    print(separator("Phase 4: 朗道能级数值求解"))

    B = params['B']
    E_analytical = params['E_analytical']
    n_levels = min(5, H.shape[0] - 1)

    t0 = time.time()
    E_numerical, evecs = ground_state_energy(H, num_states=n_levels)
    t_diag = time.time() - t0

    print(f"  对角化时间: {t_diag:.4f}s")
    print(f"\n  {'n':>4s}  {'E_num':>12s}  {'E_exact':>12s}  {'Error':>12s}")
    print(f"  {'─'*46}")

    errors = []
    for n in range(n_levels):
        E_num = E_numerical[n]
        E_ex = E_analytical[n]
        err = abs(E_num - E_ex)
        errors.append(err)
        print(f"  {n:4d}  {E_num:12.6f}  {E_ex:12.6f}  {err:12.4e}")

    mean_err = np.mean(errors)
    print(f"\n  平均误差: {mean_err:.4e}")

    # 正交性检验
    print("\n  本征态正交性检验:")
    for i in range(min(3, n_levels)):
        for j in range(i+1, min(3, n_levels)):
            overlap = np.abs(np.vdot(evecs[:, i], evecs[:, j]))
            print(f"    ⟨ψ_{i}|ψ_{j}⟩ = {overlap:.2e}")

    return E_numerical, evecs, errors


# ═══════════════════════════════════════════════════════════════
# Phase 5: 稳定性分析
# ═══════════════════════════════════════════════════════════════
def phase5_stability(params, H):
    """进行稳定性分析"""
    print(separator("Phase 5: 数值稳定性分析"))

    B = params['B']
    hx = params['hx']

    # von Neumann 分析
    print("  von Neumann 稳定性分析:")
    for scheme in ['euler_explicit', 'crank_nicolson', 'suzuki_trotter']:
        result = von_neumann_analysis(scheme, hx, 0.01, B)
        stable_str = "稳定" if result['stable'] else "不稳定"
        print(f"    {scheme}: {stable_str}")

    # CFL 条件
    dt_max = cfl_condition(H)
    print(f"\n  CFL 最大时间步长: dt_max = {dt_max:.6f}")

    # 能量守恒测试
    N_total = H.shape[0]
    psi0 = np.zeros(N_total, dtype=complex)
    psi0[N_total // 2] = 1.0  # 中心初始态
    psi0 /= np.linalg.norm(psi0)

    dt = min(0.01, dt_max * 0.5)
    ec_result = energy_conservation_test(H, psi0, dt, n_steps=20,
                                          method='suzuki_trotter')
    print(f"\n  能量守恒测试 (20步, dt={dt:.4f}):")
    print(f"    初始能量: {ec_result['E0']:.6f}")
    print(f"    最终能量: {ec_result['E_final']:.6f}")
    print(f"    最大漂移: {ec_result['max_drift']:.4e}")
    print(f"    稳定性: {'通过' if ec_result['stable'] else '失败'}")

    return ec_result


# ═══════════════════════════════════════════════════════════════
# Phase 6: Hofstadter 六角格点模型
# ═══════════════════════════════════════════════════════════════
def phase6_hofstadter(params):
    """Hofstadter 模型计算"""
    print(separator("Phase 6: Hofstadter 六角格点模型"))

    # 生成蜂窝格点
    radius = 3.0
    pos_A, pos_B, bonds = generate_honeycomb_flake(radius, a=1.0)
    N_A, N_B = len(pos_A), len(pos_B)
    N_total = N_A + N_B

    print(f"  Flake 半径: {radius}")
    print(f"  A 子晶格: {N_A} 个格点")
    print(f"  B 子晶格: {N_B} 个格点")
    print(f"  总格点: {N_total}")
    print(f"  近邻键: {len(bonds)}")

    # Hofstadter 规范
    hof_gauge = HofstadterGauge(B=0.5)
    phi = hof_gauge.flux_per_plaquette()
    print(f"  磁通/plaquette: φ = {phi:.4f} (Φ₀)")

    # 构建 Hofstadter 哈密顿量
    B_values = [0.0, 0.1, 0.3, 0.5]
    print(f"\n  Hofstadter 能谱 (B = {B_values}):")
    for B_val in B_values:
        H_hof = build_hofstadter_hamiltonian(pos_A, pos_B, bonds, B_val)
        if N_total <= 100:
            evals = np.linalg.eigvalsh(H_hof.toarray())
        else:
            k = min(N_total - 2, 10)
            evals = sparse.linalg.eigsh(H_hof, k=k, which='SA',
                                         return_eigenvectors=False)
        print(f"    B={B_val:.1f}: E_min={evals[0]:.4f}, "
              f"E_max={evals[-1]:.4f}, "
              f"gap={evals[min(1,len(evals)-1)]-evals[0]:.4f}")

    # 边界词分析
    all_pos = np.vstack([pos_A, pos_B])
    center = np.mean(all_pos, axis=0)
    # 近似边界: 距中心最远的格点
    dists = np.linalg.norm(all_pos - center, axis=1)
    boundary_mask = dists > np.percentile(dists, 80)
    boundary_sites = all_pos[boundary_mask]
    bw_result = boundary_word_analysis(boundary_sites, center)
    print(f"\n  边界词分析:")
    print(f"    边界格点: {len(boundary_sites)}")
    print(f"    边界词长度: {bw_result['length']}")
    print(f"    合法性: {'合法' if bw_result['legal'] else '不合法'}")

    return H_hof, N_total


# ═══════════════════════════════════════════════════════════════
# Phase 7: 波函数分析与态分类
# ═══════════════════════════════════════════════════════════════
def phase7_wavefunction_analysis(params, E_numerical, evecs):
    """波函数分析与态分类"""
    print(separator("Phase 7: 波函数分析与态分类"))

    Nx, Ny = params['Nx'], params['Ny']
    N_total = Nx * Ny  # 内部点, Dirichlet 边界

    # 态分类
    class_result = classify_states(E_numerical, evecs, N_total)
    print(f"  态分类统计:")
    print(f"    扩展态: {class_result['n_extended']}")
    print(f"    临界态: {class_result['n_critical']}")
    print(f"    局域态: {class_result['n_localized']}")

    # 每个态的详细信息
    print(f"\n  各态详情:")
    print(f"  {'n':>3s}  {'E':>10s}  {'PR/N':>10s}  {'IPR':>10s}  {'Type':>10s}")
    for c in class_result['classifications'][:min(8, len(class_result['classifications']))]:
        print(f"  {c['level']:3d}  {c['energy']:10.4f}  {c['PR_norm']:10.4f}  "
              f"{c['IPR']:10.6f}  {c['type']:>10s}")

    # Collatz 递归分类
    print(f"\n  Collatz 递归分类指纹:")
    for n in range(min(5, len(E_numerical))):
        collatz_result = collatz_classification(n + 1)
        print(f"    Level {n}: seq_length={collatz_result['length']}, "
              f"max={collatz_result['max_value']}, "
              f"converged={collatz_result['converged']}")

    # Bernstein 光滑
    pr_norms = np.array([c['PR_norm'] for c in class_result['classifications']])
    smooth = bernstein_smooth_classification(pr_norms, degree=8)
    print(f"\n  Bernstein 光滑分类函数 (度=8):")
    print(f"    输入 PR/N: {pr_norms[:5]}")
    print(f"    光滑输出: {smooth[:5]}")

    # Haar 小波分解
    print(f"\n  Haar 小波多分辨率分析:")
    psi_test = evecs[:, 0]
    coeffs = haar_wavelet_decompose(np.abs(psi_test[:64]), levels=3)
    for i, c in enumerate(coeffs):
        print(f"    Level {i}: {len(c)} 系数, "
              f"能量 = {np.sum(c**2):.6f}")

    # 降采样 (H 是内部点 Nx × Ny, 不含边界)
    psi_2d = np.abs(evecs[:, 0]).reshape(Nx, Ny)
    psi_coarse = downsample_2d(psi_2d, factor=2)
    print(f"\n  降采样 (因子=2):")
    print(f"    原始: {psi_2d.shape}")
    print(f"    粗化: {psi_coarse.shape}")

    return class_result


# ═══════════════════════════════════════════════════════════════
# Phase 8: 不确定度量化
# ═══════════════════════════════════════════════════════════════
def phase8_uncertainty(params):
    """不确定度量化"""
    print(separator("Phase 8: 不确定度量化"))

    B = params['B']
    Nx, Ny = params['Nx'], params['Ny']
    Lx, Ly = params['Lx'], params['Ly']

    # Monte Carlo 不确定度 (磁场不确定度)
    def energy_model(p):
        B_val = p['B']
        H = assemble_hamiltonian_landau(Nx, Ny, Lx, Ly, B_val, fd_order=4)
        evals, _ = ground_state_energy(H, num_states=1)
        return evals[0]

    print("  Monte Carlo 不确定度分析 (磁场 B ± 5%):")
    mc_result = monte_carlo_uncertainty(
        energy_model,
        param_ranges={'B': (B, 0.05 * B, 'normal')},
        n_samples=30,
        seed=42
    )
    print(f"    采样数: {mc_result['n_valid']}/{mc_result['n_samples']}")
    print(f"    E₀ 均值: {mc_result['mean']:.6f}")
    print(f"    E₀ 标准差: {mc_result['std']:.6f}")
    print(f"    95% 置信区间: [{mc_result['ci_95'][0]:.6f}, "
          f"{mc_result['ci_95'][1]:.6f}]")

    # 敏感度分析
    print(f"\n  磁场敏感度分析:")
    B_range = np.linspace(0.5, 1.5, 5)
    energies, sensitivity = sensitivity_analysis(
        energy_model, {'B': B}, 'B', B_range
    )
    print(f"    dE/dB 归一化敏感度: {sensitivity:.4f}")

    return mc_result


# ═══════════════════════════════════════════════════════════════
# Phase 9: 可观测物理量
# ═══════════════════════════════════════════════════════════════
def phase9_observables(params, E_numerical, evecs):
    """计算物理可观测量"""
    print(separator("Phase 9: 物理可观测量"))

    B = params['B']
    Nx, Ny = params['Nx'], params['Ny']
    Lx, Ly = params['Lx'], params['Ly']
    hx, hy = params['hx'], params['hy']

    # 导向中心 (内部点 Nx × Ny, 坐标从 hx 到 (Nx)*hx)
    print("  导向中心位置:")
    x_coords = np.array([i * hx for i in range(1, Nx+1) for _ in range(Ny)])
    y_coords = np.array([j * hy for _ in range(Nx) for j in range(1, Ny+1)])
    for n in range(min(3, len(E_numerical))):
        X_c, Y_c = guiding_center(evecs[:, n], x_coords, y_coords, B)
        print(f"    Level {n}: (X, Y) = ({X_c:.4f}, {Y_c:.4f})")

    # 粒子数密度
    density = particle_density([evecs[:, n] for n in range(min(3, len(E_numerical)))],
                                3, Nx, Ny)
    total_particles = np.sum(density)
    print(f"\n  填充 3 个态的总粒子数: {total_particles:.4f}")

    # Streda 公式
    area = Lx * Ly
    n_streda = streda_formula_density(B, 1, area)
    print(f"  Streda 公式预期密度 (ν=1): n = {n_streda:.6f}")

    # LDOS
    x_grid = np.array([i * hx for i in range(1, Nx+1)])
    y_grid = np.array([j * hy for j in range(1, Ny+1)])
    E_target = E_numerical[0]
    ldos = local_density_of_states(
        [evecs[:, n] for n in range(min(3, len(E_numerical)))],
        E_numerical[:3], x_grid, y_grid, E_target, sigma=0.1
    )
    print(f"\n  LDOS (E={E_target:.4f}):")
    print(f"    最大值: {np.max(ldos):.6f}")
    print(f"    平均值: {np.mean(ldos):.6f}")

    return density, ldos


# ═══════════════════════════════════════════════════════════════
# Phase 10: 时间演化
# ═══════════════════════════════════════════════════════════════
def phase10_time_evolution(params, H):
    """Suzuki-Trotter 时间演化"""
    print(separator("Phase 10: Suzuki-Trotter 时间演化"))

    N_total = H.shape[0]  # 内部点 Nx × Ny

    # 构造初始波包 (高斯波包)
    Nx, Ny = params['Nx'], params['Ny']
    hx, hy = params['hx'], params['hy']
    Lx, Ly = params['Lx'], params['Ly']

    psi0 = np.zeros(N_total, dtype=complex)
    x0, y0 = Lx / 2, Ly / 2  # 中心
    sigma_wave = 0.5
    idx = 0
    for ix in range(1, Nx + 1):
        for iy in range(1, Ny + 1):
            x = ix * hx
            y = iy * hy
            psi0[idx] = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma_wave**2))
            idx += 1

    psi0 /= np.linalg.norm(psi0)

    # 时间演化
    dt = 0.01
    n_steps = 30
    print(f"  初始波包: 高斯, 中心=({x0:.1f}, {y0:.1f}), σ={sigma_wave}")
    print(f"  时间步长: dt={dt}, 步数={n_steps}")

    psi_final, energies = time_evolution_sequence(
        H, psi0, dt, n_steps, method='expm_direct'
    )

    E_initial = energies[0]
    E_final = energies[-1]
    drift = abs(E_final - E_initial) / (abs(E_initial) + 1e-30)

    print(f"\n  时间演化结果:")
    print(f"    初始能量: {E_initial:.6f}")
    print(f"    最终能量: {E_final:.6f}")
    print(f"    相对漂移: {drift:.4e}")
    print(f"    保模性: ||ψ||² = {np.linalg.norm(psi_final)**2:.10f}")

    return psi_final, energies


# ═══════════════════════════════════════════════════════════════
# Phase 11: 数据输出
# ═══════════════════════════════════════════════════════════════
def phase11_output(params, E_numerical, evecs, mc_result):
    """输出结果"""
    print(separator("Phase 11: 数据输出"))

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # 保存波函数数据 (内部点 Nx × Ny)
    Nx, Ny = params['Nx'], params['Ny']
    x_grid = np.array([i * params['hx'] for i in range(1, Nx+1) for _ in range(Ny)])
    y_grid = np.array([j * params['hy'] for _ in range(Nx) for j in range(1, Ny+1)])

    npz_path = os.path.join(output_dir, "qhe_results.npz")
    save_wavefunction_data(npz_path, x_grid, y_grid,
                            E_numerical, evecs,
                            metadata={'B': params['B']})
    print(f"  波函数数据已保存: {npz_path}")

    # 生成报告
    report = {
        '物理参数': {
            '磁场 B': params['B'],
            '磁长度 l_B': params['l_B'],
            '系统尺寸': f"{params['Lx']} × {params['Ly']}",
            '格点数': f"{params['Nx']} × {params['Ny']}",
        },
        '朗道能级': {
            '解析值': params['E_analytical'][:5],
            '数值值': E_numerical[:5],
        },
        '不确定度': {
            'E₀ 均值': mc_result['mean'],
            'E₀ 标准差': mc_result['std'],
        }
    }

    report_path = os.path.join(output_dir, "computation_report.txt")
    generate_report(report, report_path)
    print(f"  计算报告已保存: {report_path}")


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "★" * 68)
    print("  量子霍尔效应数值对角化：高阶有限差分与稳定性分析")
    print("  Computational Condensed Matter — Project 269")
    print("  小规模可复现实验")
    print("★" * 68)

    t_start = time.time()

    # Phase 1: 设置
    params = phase1_setup()

    # Phase 2: FD 验证
    fd_errors = phase2_fd_validation(params)

    # Phase 3: 哈密顿量组装
    H, matrix_info = phase3_hamiltonian(params)

    # Phase 4: 朗道能级
    E_numerical, evecs, ll_errors = phase4_landau_levels(params, H)

    # Phase 5: 稳定性
    ec_result = phase5_stability(params, H)

    # Phase 6: Hofstadter
    H_hof, N_hof = phase6_hofstadter(params)

    # Phase 7: 波函数分析
    class_result = phase7_wavefunction_analysis(params, E_numerical, evecs)

    # Phase 8: 不确定度
    mc_result = phase8_uncertainty(params)

    # Phase 9: 可观测物理量
    density, ldos = phase9_observables(params, E_numerical, evecs)

    # Phase 10: 时间演化
    psi_final, energies = phase10_time_evolution(params, H)

    # Phase 11: 输出
    phase11_output(params, E_numerical, evecs, mc_result)

    t_total = time.time() - t_start

    print(separator("计算完成"))
    print(f"  总耗时: {t_total:.2f}s")
    print(f"  朗道能级平均误差: {np.mean(ll_errors):.4e}")
    print(f"  能量守恒漂移: {ec_result['max_drift']:.4e}")
    print(f"  态分类: {class_result['n_extended']} 扩展 + "
          f"{class_result['n_critical']} 临界 + "
          f"{class_result['n_localized']} 局域")
    print(f"  不确定度 σ(E₀): {mc_result['std']:.6f}")
    print("  ★ 所有计算正常完成 ★\n")


if __name__ == "__main__":
    main()
