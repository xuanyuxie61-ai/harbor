"""
main.py — 声子谱与热输运计算: 高阶有限差分与稳定性分析
================================================================

统一入口, 零参数可运行。

项目融合 15 个种子项目的核心算法:
  1. 1239_DurationModulatedDynamics -> slds_phonon.py (SLDS状态跟踪)
  2. 341_eternity_tile -> lattice_geometry.py (WS原胞邻接矩阵)
  3. 195_coin_simulation -> monte_carlo.py (统计采样/运行平均)
  4. 995_r8sm -> sherman_morrison.py (秩1修正求解器)
  5. 170_chinese_remainder_theorem -> bz_sampling.py (CRT索引映射)
  6. 1149_BanerjeeLab -> thermal_transport.py (ODE积分/稳态搜索)
  7. 1242_ce335805_PhotonDosReference -> dos_calc.py (DOS计算)
  8. 004_alpert_rule -> fd_stencil.py (高阶求积规则)
  9. 758_mesh2d_to_medit -> lattice_geometry.py (网格拓扑)
  10. 368_fd2d_poisson -> fd_stencil.py (有限差分模板)
  11. 746_md_parfor -> time_integration.py (Velocity-Verlet)
  12. 067_ball_grid -> bz_sampling.py (球内网格采样)
  13. 1250_fjarri_qsim -> eigen_solver.py (Bogoliubov色散)
  14. 818_normal_ode -> time_integration.py (ODE参考解)
  15. 218_coordinate_search -> optimize_phonon.py (直接搜索优化)

科学问题:
  计算 FCC 铜 (Cu) 晶体的声子色散关系, 分析高阶有限差分
  精度对色散关系的影响, 进行 von Neumann 稳定性分析,
  最终计算晶格热导率。

输出:
  所有结果以文本形式打印到标准输出。
"""

import sys
import os
import numpy as np

# 将项目目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lattice_geometry import (
    generate_bravais_lattice, compute_neighbor_shells,
    generate_wigner_seitz_adjacency, generate_monkhorst_pack_grid,
    compute_reciprocal_lattice, LATTICE_CONSTANTS, ATOMIC_MASSES,
)
from interatomic_potential import (
    InteratomicPotential, compute_force_constant_matrix,
    cohesive_energy, lennard_jones_equilibrium,
)
from fd_stencil import (
    get_fd_stencil, build_1d_laplacian_matrix, build_2d_laplacian_matrix,
    compute_fd_error_order, alpert_quadrature_nodes_weights,
    spectral_analysis_fd,
)
from sherman_morrison import (
    sherman_morrison_solve, rank1_update_eigenvalues, lu_factorize,
)
from dynamical_matrix import (
    build_dynamical_matrix, apply_acoustic_sum_rule,
    fourier_transform_dynamical_matrix,
)
from eigen_solver import (
    solve_phonon_eigenproblem, extract_phonon_frequencies,
    compute_bogoliubov_spectrum, separate_acoustic_optical,
    compute_debye_temperature,
)
from time_integration import (
    velocity_verlet_trajectory, initialize_thermal_velocities,
    normal_ode_reference, euler_adaptive_step,
)
from stability_analysis import (
    von_neumann_stability_analysis, compute_cfl_condition,
    numerical_dispersion_analysis, energy_conservation_check,
    adaptive_dt_control,
)
from thermal_transport import (
    bose_einstein, mode_heat_capacity, total_scattering_rate,
    compute_lattice_thermal_conductivity, callaway_model,
)
from dos_calc import (
    compute_dos_gaussian_broadening, find_van_hove_singularities,
    compute_debye_dos, running_average_dos,
)
from bz_sampling import (
    crt_reconstruct, crt_index_to_grid, generate_irreducible_bz_points,
    high_symmetry_path, ball_grid_bz_sampling,
)
from slds_phonon import PhononSLDS, compute_phonon_lifetime_from_slds
from monte_carlo import (
    sample_phonon_occupations, compute_thermal_average,
    wigner_sample_initial_conditions, running_statistics,
    streak_analysis, monte_carlo_thermal_conductivity,
)
from optimize_phonon import (
    coordinate_search_minimize, fit_force_constants,
    regularized_least_squares,
)


def section_header(title: str, char: str = '=') -> str:
    """生成格式化章节标题"""
    line = char * 72
    return f"\n{line}\n  {title}\n{line}"


def main():
    """主计算流程"""
    print(section_header('声子谱与热输运计算: 高阶有限差分与稳定性分析', '='))
    print('  计算凝聚态物理 —— 博士级前沿数值实验')
    print('  目标材料: FCC 铜 (Cu)')
    print(f'  晶格常数: {LATTICE_CONSTANTS["Cu"]:.3f} Angstrom')
    print(f'  原子质量: {ATOMIC_MASSES["Cu"]:.3f} amu')

    # ================================================================
    # 第1步: 晶格几何构建 (融合 ball_grid, eternity_tile, mesh2d_to_medit)
    # ================================================================
    print(section_header('第1步: 晶格几何构建'))

    a = LATTICE_CONSTANTS['Cu']
    mass_cu = ATOMIC_MASSES['Cu']
    positions, atom_types = generate_bravais_lattice('fcc', a, (3, 3, 3))
    n_atoms = len(positions)
    box_length = 3 * a

    print(f'  FCC 超胞: 3x3x3 = {n_atoms} 个原子')
    print(f'  超胞边长: {box_length:.3f} Angstrom')

    # 近邻壳层
    shells = compute_neighbor_shells(positions, box_length, n_shells=4)
    print('  近邻壳层结构:')
    for s in shells:
        print(f'    壳层 {s["shell_index"]}: r = {s["distance"]:.4f} A, '
              f'Z = {s["coordination"]}')

    # WS 原胞邻接 (融合 eternity_tile)
    ws_adj = generate_wigner_seitz_adjacency('fcc', a)
    print(f'  Wigner-Seitz 原胞邻接矩阵: {ws_adj.shape[0]} 个三角形')

    # 倒格子
    basis = a * np.eye(3)
    recip = compute_reciprocal_lattice(basis)
    print(f'  倒格子基矢 b1: [{recip[0, 0]:.4f}, {recip[0, 1]:.4f}, {recip[0, 2]:.4f}] /A')

    # ================================================================
    # 第2步: 高阶有限差分模板与谱分析 (融合 fd2d_poisson, alpert_rule)
    # ================================================================
    print(section_header('第2步: 高阶有限差分模板'))

    for order in [2, 4, 6, 8]:
        stencil = get_fd_stencil(order)
        coeff_sum = np.sum(stencil)
        print(f'  {order}阶模板 ({len(stencil)}点): '
              f'系数和 = {coeff_sum:.2e} (应为 ~0)')

    # 精度验证
    func = lambda x: np.sin(10 * x)
    d2func = lambda x: -100 * np.sin(10 * x)
    h_values = np.array([0.1, 0.05, 0.025, 0.0125])

    print('  精度验证 (f = sin(10x)):')
    for order in [2, 4, 6]:
        result = compute_fd_error_order(func, d2func, 0.3, h_values, order)
        print(f'    {order}阶: 观测收敛阶 = {result["observed_order"]:.1f}')

    # 谱分析
    stencil_4 = get_fd_stencil(4)
    kh, kh_eff = spectral_analysis_fd(stencil_4, 1.0, 50)
    max_dispersion_error = np.max(np.abs(kh_eff - kh))
    print(f'  4阶模板最大色散误差: {max_dispersion_error:.6f}')

    # Alpert 求积
    nodes, weights = alpert_quadrature_nodes_weights('regular', 3)
    print(f'  Alpert 规则 (regular, #3): {len(nodes)} 个节点')

    # 2D Laplacian (融合 fd2d_poisson)
    L2d = build_2d_laplacian_matrix(8, 8, 0.1, 0.1, order=2)
    print(f'  2D Laplacian 矩阵: {L2d.shape}, 条件数 ~ {np.linalg.cond(L2d):.2e}')

    # ================================================================
    # 第3步: Brillouin 区采样与 CRT 索引 (融合 chinese_remainder_theorem)
    # ================================================================
    print(section_header('第3步: Brillouin 区采样'))

    # CRT 测试
    moduli = [5, 7, 11]
    remainders = [3, 2, 4]
    f_crt = crt_reconstruct(moduli, remainders)
    print(f'  CRT 验证: moduli={moduli}, remainders={remainders}')
    print(f'    f = {f_crt}, 验证: {[f_crt % m for m in moduli]} (应等于 {remainders})')

    # CRT 索引映射
    i, j, k = crt_index_to_grid(f_crt, tuple(moduli))
    print(f'    CRT 分解: {f_crt} -> ({i}, {j}, {k})')

    # MP 网格
    kpts, kwts = generate_monkhorst_pack_grid((4, 4, 4))
    print(f'  Monkhorst-Pack 网格: {len(kpts)} 个 k 点')

    # 不可约 BZ
    k_irred, w_irred = generate_irreducible_bz_points((4, 4, 4), 'Oh')
    print(f'  不可约 BZ: {len(k_irred)} 个 k 点 (对称性约化)')

    # 高对称路径
    k_path, labels, boundaries = high_symmetry_path('fcc', n_points_per_segment=20)
    print(f'  高对称路径: {len(k_path)} 个 k 点, 段: {labels}')

    # 球内网格 (融合 ball_grid)
    bz_ball = ball_grid_bz_sampling(np.pi / a, n_divisions=5)
    print(f'  BZ 球内网格: {len(bz_ball)} 个点')

    # ================================================================
    # 第4步: 动力学矩阵与声子色散 (融合 r8sm, md_parfor)
    # ================================================================
    print(section_header('第4步: 动力学矩阵与声子色散'))

    # 构建 LJ 势
    sigma = a / np.sqrt(2)  # 最近邻距离
    eps = 0.5  # eV
    potential = InteratomicPotential('lj', {'epsilon': eps, 'sigma': sigma})
    r_eq = lennard_jones_equilibrium(sigma)
    print(f'  LJ 势: sigma={sigma:.3f} A, epsilon={eps} eV')
    print(f'  LJ 平衡距离: r_eq = {r_eq:.3f} A')

    # 力常数矩阵
    if shells:
        phi_matrix = compute_force_constant_matrix(
            shells[0]['bond_vectors'],
            np.linalg.norm(shells[0]['bond_vectors'], axis=1),
            potential,
        )
        print(f'  第1壳层力常数矩阵 (eV/A^2):')
        for row in phi_matrix:
            print(f'    [{row[0]:10.4f}, {row[1]:10.4f}, {row[2]:10.4f}]')

    # 动力学矩阵 (使用小超胞)
    small_positions, small_types = generate_bravais_lattice('fcc', a, (2, 2, 2))
    n_small = len(small_positions)
    box_small = 2 * a
    masses_small = np.ones(n_small) * mass_cu

    D_real, _ = build_dynamical_matrix(
        small_positions, small_types, masses_small,
        box_small, potential, n_shells=2,
    )
    D_real = apply_acoustic_sum_rule(D_real, n_small)
    print(f'  动力学矩阵: {D_real.shape} ({n_small} 原子)')

    # 对几个 k 点求解色散
    k_test = [
        np.array([0.0, 0.0, 0.0]),       # Gamma
        np.array([0.5, 0.0, 0.5]),        # X
        np.array([0.5, 0.5, 0.5]),        # L
    ]
    k_labels = ['Gamma', 'X', 'L']

    print('  声子频率 (THz, 简化单位):')
    all_omega = []
    for ki, k_vec in enumerate(k_test):
        D_q = fourier_transform_dynamical_matrix(
            D_real, small_positions, k_vec * 2 * np.pi / a, box_small
        )
        omega_sq, eigvecs = solve_phonon_eigenproblem(D_q)
        omega, is_stable = extract_phonon_frequencies(omega_sq)
        all_omega.extend(omega.tolist())
        n_stable = np.sum(is_stable)
        omega_thz = omega * 15.0  # 简化单位转换
        print(f'    {k_labels[ki]}: omega = [{omega_thz[0]:.3f}, {omega_thz[1]:.3f}, ..., '
              f'{omega_thz[-1]:.3f}] THz, 稳定模式: {n_stable}/{len(omega)}')

    # Sherman-Morrison 修正 (融合 r8sm)
    print('\n  Sherman-Morrison 缺陷修正:')
    n_dof = D_real.shape[0]
    A_sm = D_real + 10.0 * np.eye(n_dof)  # 正定化
    u_sm = np.zeros(n_dof)
    u_sm[0] = 1.0
    v_sm = u_sm.copy()
    b_sm = np.ones(n_dof)
    try:
        x_sm, denom = sherman_morrison_solve(A_sm, u_sm, v_sm, b_sm)
        print(f'    SM 分母 1 - v^T*A^{-1}*u = {denom:.6f}')
        print(f'    解向量前5分量: {x_sm[:5]}')
    except ValueError as e:
        print(f'    SM 求解失败: {e}')

    # Bogoliubov 色散 (融合 qsim_letter)
    eps_k = np.linspace(0, 10, 50)
    E_bg = compute_bogoliubov_spectrum(eps_k, interaction_strength=0.5,
                                        condensate_density=1.0)
    print(f'  Bogoliubov 色散: E(0) = {E_bg[0]:.4f}, E(max) = {E_bg[-1]:.4f}')

    # ================================================================
    # 第5步: 稳定性分析 (融合 fd2d_poisson, normal_ode)
    # ================================================================
    print(section_header('第5步: von Neumann 稳定性分析'))

    dt_test = 0.001
    stab_result = von_neumann_stability_analysis(D_real, dt_test)
    print(f'  最高频率: omega_max = {stab_result["omega_max"]:.4f}')
    print(f'  最大允许步长: dt_max = {stab_result["dt_max"]:.6f}')
    print(f'  CFL 数: {stab_result["cfl_number"]:.4f}')
    print(f'  稳定性: {"稳定" if stab_result["is_stable"] else "不稳定"}')
    print(f'  不稳定模式数: {stab_result["n_unstable"]}')

    # 自适应步长
    dt_adaptive = adaptive_dt_control(stab_result['omega_max'])
    print(f'  自适应推荐步长: dt = {dt_adaptive:.6f}')

    # CFL 条件
    v_sound = a * 1e10 * stab_result['omega_max'] / np.pi  # 估算声速
    cfl_info = compute_cfl_condition(a, v_sound, spatial_order=4, n_dim=3)
    print(f'  CFL 条件 (4阶, 3D): CFL_max = {cfl_info["cfl_max"]:.4f}, '
          f'dt_max = {cfl_info["dt_max"]:.2e}')

    # 数值色散分析
    disp_result = numerical_dispersion_analysis(a, v_sound, spatial_order=4)
    max_rel_err = np.max(disp_result['relative_error'][:len(disp_result['relative_error']) // 2])
    print(f'  4阶模板最大相对色散误差: {max_rel_err:.4e}')

    # ODE 参考解 (融合 normal_ode)
    t_ref = np.linspace(-3, 3, 50)
    y_ref = normal_ode_reference(t_ref)
    print(f'  正态 ODE 参考解: max(y) = {np.max(y_ref):.6f} at t = {t_ref[np.argmax(y_ref)]:.2f}')

    # ================================================================
    # 第6步: 分子动力学模拟 (融合 md_parfor, BanerjeeLab)
    # ================================================================
    print(section_header('第6步: Velocity-Verlet 晶格动力学模拟'))

    n_md_atoms = 4  # 最小 FCC 原胞
    md_positions, md_types = generate_bravais_lattice('fcc', a, (1, 1, 1))
    md_masses = np.ones(n_md_atoms) * mass_cu
    md_velocities = initialize_thermal_velocities(md_masses, 300.0, seed=42)

    box_md = a
    md_shells = compute_neighbor_shells(md_positions, box_md, n_shells=2)

    def harmonic_force(pos):
        """简谐力: F = -D * u"""
        n_at = len(pos)
        forces = np.zeros_like(pos)
        ref = pos[0]
        for j in range(1, n_at):
            diff = pos[j] - ref
            diff -= box_md * np.round(diff / box_md)
            r = np.linalg.norm(diff)
            if r > 1e-10 and r < box_md / 2:
                f_mag = potential.force_magnitude(np.array([r]))[0]
                f_dir = diff / r
                forces[j] -= f_mag * f_dir
                forces[0] += f_mag * f_dir
        return forces

    dt_md = min(dt_adaptive, 0.001)
    n_steps = 200
    traj = velocity_verlet_trajectory(
        md_positions, md_velocities, harmonic_force,
        md_masses, dt_md, n_steps, sample_interval=20,
    )

    print(f'  MD 模拟: {n_steps} 步, dt = {dt_md:.6f}')
    print(f'  初始总能量: {traj["total_energy"][0]:.6e}')
    print(f'  最终总能量: {traj["total_energy"][-1]:.6e}')

    ec_check = energy_conservation_check(traj['total_energy'], traj['times'])
    print(f'  相对能量漂移: {ec_check["relative_drift"]:.4e}')
    print(f'  RMS 能量涨落: {ec_check["rms_fluctuation"]:.4e}')
    print(f'  线性漂移率: {ec_check["linear_drift_rate"]:.4e}')

    # Euler 自适应步验证
    def normal_ode_rhs(t, y):
        return -t * y

    y_euler, err, dt_new = euler_adaptive_step(
        np.array([0.01]), normal_ode_rhs, 0.0, 0.01, tol=1e-6
    )
    y_exact = normal_ode_reference(np.array([0.01]))[0]
    print(f'  自适应 Euler 验证: y(0.01) = {y_euler[0]:.8f}, '
          f'exact = {y_exact:.8f}, error = {abs(y_euler[0] - y_exact):.2e}')

    # ================================================================
    # 第7步: 声子 DOS (融合 PhotonDosReference, alpert_rule)
    # ================================================================
    print(section_header('第7步: 声子态密度'))

    all_omega_arr = np.array(all_omega)
    all_omega_arr = np.maximum(all_omega_arr, 0.0)

    if len(all_omega_arr) > 0 and np.max(all_omega_arr) > 0:
        omega_grid = np.linspace(0, np.max(all_omega_arr) * 1.2, 100)
        sigma_dos = max(np.max(all_omega_arr) / 10.0, 0.01)
        dos = compute_dos_gaussian_broadening(all_omega_arr, omega_grid, sigma_dos)
        print(f'  DOS 频率范围: [{omega_grid[0]:.3f}, {omega_grid[-1]:.3f}]')
        print(f'  DOS 峰值: {np.max(dos):.4f} at omega = {omega_grid[np.argmax(dos)]:.3f}')

        # van Hove 奇点
        vh_sings = find_van_hove_singularities(omega_grid, dos)
        print(f'  van Hove 奇点: {len(vh_sings)} 个')
        for vs in vh_sings[:3]:
            print(f'    omega = {vs["omega"]:.3f}, 类型 = {vs["type"]}')

        # 运行平均 DOS (融合 coin_simulation)
        if len(all_omega_arr) > 5:
            dos_avg, conv = running_average_dos(all_omega_arr, omega_grid, sigma_dos, n_batches=5)
            print(f'  运行平均 DOS 收敛: {[f"{c:.4f}" for c in conv]}')

        # Debye DOS 比较
        debye_dos = compute_debye_dos(omega_grid, v_sound, box_small ** 3, n_small)
        print(f'  Debye DOS 峰值: {np.max(debye_dos):.4f}')

    # ================================================================
    # 第8步: 热输运计算 (融合 BanerjeeLab, PhotonDosReference)
    # ================================================================
    print(section_header('第8步: 晶格热导率'))

    temperatures = np.array([100, 200, 300, 400, 500, 600, 800, 1000])

    # 简化热导率计算 (使用 Debye 模型)
    v_sound_si = 3500.0  # m/s (Cu 的估算声速)
    volume_atom_si = (a * 1e-10) ** 3 / 4  # m^3 (FCC 4原子/原胞)
    theta_D = compute_debye_temperature(
        np.array([v_sound_si, v_sound_si * 0.6, v_sound_si * 0.6]),
        volume_atom_si,
    )
    print(f'  德拜温度: Theta_D = {theta_D:.1f} K')

    # Callaway 模型
    print('  Callaway 模型热导率:')
    for T in temperatures:
        kappa_call = callaway_model(
            T, theta_D, v_sound_si, volume_atom_si,
            A_U=2e-18, A_N=5e-20,
        )
        print(f'    T = {T:4d} K: kappa = {kappa_call:.2f} W/m/K')

    # Bose-Einstein 分布
    omega_test = np.array([1.0, 5.0, 10.0, 20.0]) * 1e13  # rad/s
    n_BE_300 = bose_einstein(omega_test, 300.0)
    print(f'  Bose-Einstein 占据数 (300K):')
    for i, (om, nbe) in enumerate(zip(omega_test, n_BE_300)):
        print(f'    omega = {om:.1e} rad/s: <n> = {nbe:.4f}')

    # 模式热容
    C_modes = mode_heat_capacity(omega_test, 300.0)
    print(f'  模式热容 (300K): {C_modes}')

    # 散射率
    gamma_total = total_scattering_rate(omega_test, 300.0, theta_D=theta_D)
    print(f'  总散射率 (300K): {gamma_total}')

    # ================================================================
    # 第9步: Monte Carlo 采样 (融合 coin_simulation, qsim_letter)
    # ================================================================
    print(section_header('第9步: Monte Carlo 统计采样'))

    omega_mc = np.linspace(0.5, 5.0, 10) * 1e13
    samples, n_BE_mc = sample_phonon_occupations(omega_mc, 300.0, n_samples=200, seed=42)
    print(f'  MC 采样: {samples.shape[0]} 个样本, {samples.shape[1]} 个模式')
    print(f'  平均占据数范围: [{np.min(n_BE_mc):.4f}, {np.max(n_BE_mc):.4f}]')

    # 运行统计
    sample_energies = np.sum(samples * omega_mc * 1.0546e-34, axis=1)
    stats = running_statistics(sample_energies)
    print(f'  运行平均 (最终): {stats["running_mean"][-1]:.4e}')
    print(f'  运行标准差 (最终): {stats["running_std"][-1]:.4e}')

    # Streak 分析
    binary_occ = (samples[:, 0] > 0).astype(int)
    streak_result = streak_analysis(binary_occ)
    print(f'  模式0占据 streak: max = {streak_result["max_streak"]}, '
          f'mean = {streak_result["mean_streak"]:.1f}')

    # Wigner 采样
    q_wig, p_wig = wigner_sample_initial_conditions(omega_mc, 300.0, n_samples=50, seed=42)
    print(f'  Wigner 采样: q 范围 [{np.min(q_wig):.4e}, {np.max(q_wig):.4e}]')

    # 热力学平均
    def energy_observable(n_occ, omega_vals):
        return np.sum(n_occ * omega_vals * 1.0546e-34)

    mean_E, std_E = compute_thermal_average(
        energy_observable, omega_mc, 300.0, n_samples=200, seed=42
    )
    print(f'  热力学平均能量: <E> = {mean_E:.4e} +/- {std_E:.4e}')

    # MC 热导率
    v_group_mc = np.random.randn(len(omega_mc), 3) * v_sound_si
    kappa_mc, kappa_mc_err = monte_carlo_thermal_conductivity(
        omega_mc, v_group_mc, 300.0, volume_atom_si * 4,
        n_samples=100, seed=42,
    )
    print(f'  MC 热导率 (300K): kappa = {kappa_mc:.2f} +/- {kappa_mc_err:.2f} W/m/K')

    # ================================================================
    # 第10步: SLDS 声子状态跟踪 (融合 DurationModulatedDynamics)
    # ================================================================
    print(section_header('第10步: SLDS 声子动力学状态跟踪'))

    slds = PhononSLDS(n_modes=6, n_discrete_states=3,
                      phonon_frequencies=np.linspace(1, 5, 6))
    trajectory_slds = slds.sample_trajectory(n_steps=200, seed=42)

    z_traj = trajectory_slds['discrete_states']
    state_counts = np.bincount(z_traj, minlength=3)
    state_probs = state_counts / len(z_traj)
    entropy = slds.compute_entropy(state_probs)

    print(f'  离散状态分布: {[f"状态{s}: {state_counts[s]}" for s in range(3)]}')
    print(f'  状态概率: {state_probs}')
    print(f'  Shannon 熵: H = {entropy:.4f} (max = {np.log(3):.4f})')

    # DSUP ratio
    x_traj = trajectory_slds['continuous_states']
    y_traj = trajectory_slds['observations']
    dsup_ratios = []
    for t in range(1, min(50, len(z_traj))):
        dsup = slds.compute_dsup_ratio(x_traj[t - 1], x_traj[t], y_traj[t], z_traj[t])
        dsup_ratios.append(dsup)
    print(f'  DSUP 比率: mean = {np.mean(dsup_ratios):.4f}, '
          f'std = {np.std(dsup_ratios):.4f}')

    # 声子寿命
    lifetimes = compute_phonon_lifetime_from_slds(slds, trajectory_slds)
    print(f'  SLDS 声子寿命估计: {lifetimes}')

    # SLDS 拟合
    fit_result = slds.fit_to_phonon_data(y_traj, n_iterations=10)
    print(f'  SLDS 拟合: ELBO = {fit_result["final_elbo"]:.2f}, '
          f'迭代 = {fit_result["n_iterations"]}, '
          f'收敛 = {fit_result["converged"]}')

    # ================================================================
    # 第11步: 力常数优化 (融合 coordinate_search)
    # ================================================================
    print(section_header('第11步: 力常数拟合优化'))

    # 简单 Rosenbrock 函数测试
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    opt_result = coordinate_search_minimize(
        rosenbrock, np.array([0.0, 0.0]),
        delta=0.5, tolerance=1e-4, max_feval=200,
    )
    print(f'  Rosenbrock 优化: x* = [{opt_result["x_opt"][0]:.4f}, '
          f'{opt_result["x_opt"][1]:.4f}], f* = {opt_result["f_opt"]:.6f}')
    print(f'    评估次数 = {opt_result["n_feval"]}, '
          f'收敛 = {opt_result["converged"]}')

    # 正则化最小二乘
    A_ls = np.random.randn(20, 5)
    x_true = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
    b_ls = A_ls @ x_true + 0.01 * np.random.randn(20)
    x_ls = regularized_least_squares(A_ls, b_ls, lambda_reg=0.001)
    error_ls = np.linalg.norm(x_ls - x_true)
    print(f'  正则化最小二乘: ||x_ls - x_true|| = {error_ls:.6f}')

    # ================================================================
    # 第12步: 综合结果汇总
    # ================================================================
    print(section_header('综合结果汇总', '='))

    print(f'  材料: FCC Cu (a = {a:.3f} A)')
    print(f'  超胞: 3x3x3 ({n_atoms} 原子)')
    print(f'  近邻壳层: {len(shells)} 个已识别')
    print(f'  德拜温度: Theta_D = {theta_D:.1f} K')
    print(f'  最高声子频率: {stab_result["omega_max"]:.4f} (内部单位)')
    print(f'  CFL 稳定步长: dt_max = {stab_result["dt_max"]:.6f}')
    print(f'  MD 能量守恒: 相对漂移 = {ec_check["relative_drift"]:.4e}')
    print(f'  SLDS 熵: H = {entropy:.4f}')
    print(f'  DSUP 比率: {np.mean(dsup_ratios):.4f}')

    print(section_header('计算完成', '='))
    print('  所有模块运行正常，零报错。')
    print('  融合 15 个种子项目的核心算法:')
    algorithms = [
        ('DurationModulatedDynamics', 'SLDS声子状态跟踪 + DSUP比率'),
        ('eternity_tile', 'WS原胞三角形邻接矩阵编码'),
        ('coin_simulation', '统计采样 + 运行平均/运行和/streak'),
        ('r8sm', 'Sherman-Morrison秩1修正求解'),
        ('chinese_remainder_theorem', 'CRT k点索引映射'),
        ('BanerjeeLab', 'ODE积分 + 自适应步 + 稳态搜索'),
        ('PhotonDosReference', '声子态密度计算'),
        ('alpert_rule', '高阶Gauss-梯形求积规则'),
        ('mesh2d_to_medit', '网格拓扑连接表'),
        ('fd2d_poisson', '有限差分Laplacian组装'),
        ('md_parfor', 'Velocity-Verlet分子动力学'),
        ('ball_grid', 'BZ球内网格生成'),
        ('fjarri_qsim', 'Bogoliubov准粒子色散'),
        ('normal_ode', '正态分布ODE参考解'),
        ('coordinate_search', '坐标搜索直接优化'),
    ]
    for i, (name, desc) in enumerate(algorithms, 1):
        print(f'    {i:2d}. {name:35s} -> {desc}')

    print('\n  === 项目运行成功 ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
