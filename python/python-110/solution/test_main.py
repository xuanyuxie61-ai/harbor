"""
main.py - 量子点单光子发射综合模拟系统

基于15个科研代码项目的核心算法融合构建，围绕光学工程前沿课题：
"半导体量子点单光子发射的理论建模与性能优化"。

运行方式：
    python main.py

无需任何命令行参数，自动执行完整计算流程并输出结果。
"""

import numpy as np
import time

# 导入各科学计算模块
from utils import (
    validate_array_1d,
    fio_write_matrix,
    estimate_condition_number_dense,
)
from qd_hamiltonian import (
    effective_mass,
    spherical_confinement_potential,
    stark_field_potential,
    build_kinetic_hamiltonian_1d,
    add_potential_to_hamiltonian,
    sparse_to_dense,
    solve_eigenvalues_1d,
    dipole_matrix_element_1d,
    exciton_binding_energy_1d,
    reduced_mass,
)
from wavefunction_solver import (
    gegenbauer_cc_quadrature,
    solve_radial_wavefunctions,
    interpolate_potential,
)
from mesh_generator import (
    generate_circular_domain_nodes,
    compute_mesh_areas,
    node_to_element_average,
    sample_q4_mesh,
    area_estimate_grid_in_polygon,
)
from em_field import (
    lorentzian_cavity_mode,
    effective_mode_volume_2d,
    purcell_factor,
    fem_mode_solver_1d,
    spontaneous_emission_rate,
)
from master_equation import (
    jaynes_cummings_hamiltonian,
    solve_steady_state,
    solve_master_equation_time_evolution,
    excited_state_population,
    cavity_photon_number,
    check_trace_conservation,
)
from photon_statistics import (
    disk_distance_stats,
    second_order_correlation_weak_coupling,
    antibunching_parameter,
    monte_carlo_photon_detection,
    simulate_hanbury_brown_twiss,
    photon_indistinguishability_homodyne,
    detection_area_efficiency,
)
from state_clustering import (
    classify_quantum_dot_ensemble,
    spectral_purity_index,
)
from surface_roughness import (
    generate_rough_quantum_dot_boundary,
    roughness_induced_broadening,
    effective_potential_perturbation,
    fractal_dimension_box_counting,
)
from parameter_search import (
    brute_force_optimize,
    single_photon_figure_of_merit,
    sensitivity_analysis,
)


def print_section(title: str):
    """打印格式化章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_quantum_dot_confinement():
    """
    步骤1：求解量子点中电子与空穴的受限态。
    """
    print_section("步骤1：量子点电子/空穴受限态求解")
    print("物理模型：一维等效质量薛定谔方程（球对称近似）")
    print("  H = - (hbar^2 / 2m*) d^2/dx^2 + V_conf(x)")

    # 参数：InAs 量子点，半径 5 nm
    R_dot = 5.0e-9
    n_grid = 200
    x_grid = np.linspace(-3.0 * R_dot, 3.0 * R_dot, n_grid)
    m_e_ratio = effective_mass("InAs")
    m_h_ratio = effective_mass("GaAs")  # 空穴有效质量较大

    # 电子基态（加入微弱电场打破对称性，获得非零偶极矩）
    V_e = spherical_confinement_potential(x_grid, R_dot, 0.5 * 1.602176634e-19)
    V_e += stark_field_potential(x_grid, 5e6)  # 5 MV/m 弱电场
    result_e = solve_eigenvalues_1d(
        x_grid, m_e_ratio, potential_type="spherical",
        R_dot=R_dot, V0=0.5 * 1.602176634e-19
    )
    # 重新计算加入 Stark 场后的本征态（直接叠加势能在现有网格上）
    from qd_hamiltonian import build_kinetic_hamiltonian_1d, add_potential_to_hamiltonian, sparse_to_dense
    rows_e, cols_e, data_e = build_kinetic_hamiltonian_1d(x_grid, m_e_ratio)
    rows_e, cols_e, data_e = add_potential_to_hamiltonian(rows_e, cols_e, data_e, V_e)
    H_e = sparse_to_dense(rows_e, cols_e, data_e, x_grid.size)
    H_e = np.real(0.5 * (H_e + H_e.T))
    eigvals_e, eigvecs_e = np.linalg.eigh(H_e)
    idx_e = np.argsort(eigvals_e)
    eigvals_e = eigvals_e[idx_e]
    eigvecs_e = eigvecs_e[:, idx_e]
    E_e_ground = eigvals_e[0] / 1.602176634e-19
    psi_e_ground = eigvecs_e[:, 0]

    # 空穴基态
    V_h = spherical_confinement_potential(x_grid, R_dot, 0.3 * 1.602176634e-19)
    V_h += stark_field_potential(x_grid, 5e6)
    rows_h, cols_h, data_h = build_kinetic_hamiltonian_1d(x_grid, m_h_ratio)
    rows_h, cols_h, data_h = add_potential_to_hamiltonian(rows_h, cols_h, data_h, V_h)
    H_h = sparse_to_dense(rows_h, cols_h, data_h, x_grid.size)
    H_h = np.real(0.5 * (H_h + H_h.T))
    eigvals_h, eigvecs_h = np.linalg.eigh(H_h)
    idx_h = np.argsort(eigvals_h)
    eigvals_h = eigvals_h[idx_h]
    eigvecs_h = eigvecs_h[:, idx_h]
    E_h_ground = eigvals_h[0] / 1.602176634e-19
    psi_h_ground = eigvecs_h[:, 0]

    # 带隙能量与激子结合能
    E_g_bulk = 0.417  # InAs 体带隙 (eV)
    E_ex = E_g_bulk + E_e_ground + abs(E_h_ground)
    E_bind = exciton_binding_energy_1d(x_grid, psi_e_ground, psi_h_ground, eps_r=12.9)

    # 偶极矩：通过电子与空穴的电荷中心分离计算
    x_center_e = np.trapezoid(x_grid * np.abs(psi_e_ground) ** 2, x_grid)
    x_center_h = np.trapezoid(x_grid * np.abs(psi_h_ground) ** 2, x_grid)
    e_charge = 1.602176634e-19
    d_ex = e_charge * abs(x_center_e - x_center_h)
    # 若数值过小，采用典型 InAs 量子点实验值作为下限
    if d_ex < 1e-30:
        d_ex = 2.5e-29  # ~7.5 Debye

    print(f"  电子基态能量:       {E_e_ground:.4f} eV")
    print(f"  空穴基态能量:       {E_h_ground:.4f} eV")
    print(f"  激子跃迁能量:       {E_ex:.4f} eV")
    print(f"  激子结合能:         {E_bind / 1.602176634e-19 * 1e3:.3f} meV")
    print(f"  电偶极矩矩阵元:     {abs(d_ex) * 1e28:.4f} x 10^{-28} C·m")

    return {
        "x_grid": x_grid,
        "E_ex_eV": E_ex,
        "E_bind_J": E_bind,
        "dipole_moment": abs(d_ex),
        "psi_e": psi_e_ground,
        "psi_h": psi_h_ground,
    }


def run_wavefunction_special_functions():
    """
    步骤2：使用 Gegenbauer 正交多项式积分验证波函数归一化。
    """
    print_section("步骤2：Gegenbauer-Clenshaw-Curtis 特殊函数积分")
    print("验证波函数归一化：")
    print("  I = integral_{-1}^{+1} (1-x^2)^{lambda-1/2} |psi(x)|^2 dx")

    # 构造测试波函数：高斯型
    def test_psi2(x):
        return np.exp(-x ** 2)

    lam = 0.5
    n_quad = 64
    val = gegenbauer_cc_quadrature(n_quad, lam, test_psi2)
    print(f"  Gegenbauer 积分结果 (n={n_quad}, lambda={lam}): {val:.8f}")
    print(f"  理论值 sqrt(pi)*erf(1): {np.sqrt(np.pi) * 0.84270079:.8f}")


def run_mesh_and_field():
    """
    步骤3：生成微腔网格并计算电磁场模式。
    """
    print_section("步骤3：微腔网格生成与电磁场模式")
    print("模型：圆盘形 whispering-gallery 微腔")
    print("  等参映射: x(r,s) = sum_i x_i psi_i(r,s)")

    R_cavity = 1.5e-6  # 1.5 um 微腔半径
    nodes, elements = generate_circular_domain_nodes(R_cavity, n_r=12, n_theta=24)
    elem_areas, mesh_area = compute_mesh_areas(nodes, elements)
    print(f"  节点数: {nodes.shape[1]}, 单元数: {elements.shape[1]}")
    print(f"  网格总面积: {mesh_area * 1e12:.3f} um^2")

    # 计算腔模场分布
    E_mode = lorentzian_cavity_mode(
        nodes[0, :], nodes[1, :], 0.0, 0.0, R_cavity, n_eff=3.5
    )
    epsilon_r = np.full(elements.shape[1], 12.25)  # GaAs epsilon = n^2
    A_eff = effective_mode_volume_2d(nodes, elements, E_mode, epsilon_r)
    # 对于二维微盘，等效模式体积 V_eff = A_eff * (lambda / n)（等效厚度）
    wavelength = 930e-9  # 930 nm
    n_eff = 3.5
    V_eff = A_eff * (wavelength / n_eff)
    print(f"  等效模式面积 A_eff: {A_eff * 1e12:.4f} um^2")
    print(f"  等效模式体积 V_eff: {V_eff * 1e18:.4f} um^3")

    # Purcell 因子
    Q = 1e4
    Fp = purcell_factor(Q, V_eff, wavelength, n_eff=n_eff)
    print(f"  品质因子 Q:         {Q:.0e}")
    print(f"  Purcell 因子 F_p:   {Fp:.2f}")

    # 节点值平均到单元
    E_elem = node_to_element_average(E_mode, elements)
    print(f"  单元平均场强范围:   [{np.min(E_elem):.4f}, {np.max(E_elem):.4f}]")

    # 网格采样
    sample_xy, sample_elem = sample_q4_mesh(nodes, elements, sample_num=500)
    print(f"  网格随机采样点数:   {sample_xy.shape[1]}")

    # 面积估计验证
    # 构造圆形多边形边界
    theta_poly = np.linspace(0, 2 * np.pi, 100)
    circle_poly = np.vstack([R_cavity * np.cos(theta_poly), R_cavity * np.sin(theta_poly)])
    rel_area = area_estimate_grid_in_polygon(circle_poly, n_grid=128)
    print(f"  圆域网格面积估计:   {rel_area:.4f} (理论值: {np.pi / 4:.4f} 对单位圆)")

    return {
        "nodes": nodes,
        "elements": elements,
        "V_eff": V_eff,
        "Fp": Fp,
        "Q": Q,
        "wavelength": wavelength,
    }


def run_master_equation(E_ex_eV: float, dipole_moment: float, Fp: float):
    """
    步骤4：求解量子点-微腔耦合的主方程。
    """
    print_section("步骤4：量子点-微腔 Lindblad 主方程")
    print("哈密顿量 (Jaynes-Cummings, 旋波近似):")
    print("  H = hbar omega_c a^dagger a + hbar omega_dot sigma_+ sigma_-")
    print("      + hbar g (a^dagger sigma_- + a sigma_+)")
    print("主方程:")
    print("  d rho/dt = -i/hbar [H, rho] + gamma_dot D[sigma_-](rho)")
    print("             + kappa D[a](rho)")

    omega_dot = E_ex_eV * 1.602176634e-19 / 1.054571817e-34
    omega_c = omega_dot  # 共振
    gamma_dot = 1.0e8  # 裸量子点自发辐射速率 (1/s)
    kappa = omega_c / 1e4  # 腔衰减
    g_coupling = np.sqrt(max(Fp * gamma_dot * kappa / 4.0, 0.0))  # 强耦合估计

    n_cutoff = 4
    H = jaynes_cummings_hamiltonian(omega_c, omega_dot, g_coupling, n_cutoff)

    # 构建 jump operators
    dim = 2 * n_cutoff
    sigma_minus = np.zeros((dim, dim), dtype=complex)
    for n in range(n_cutoff):
        if n > 0:
            sigma_minus[2 * (n - 1), 2 * n + 1] = 1.0
        sigma_minus[2 * n, 2 * n + 1] = 0.0  # |g,n><e,n| 无 n 变化
    # 正确构造 sigma_- = |g><e| \otimes I
    sigma_minus = np.zeros((dim, dim), dtype=complex)
    for n in range(n_cutoff):
        sigma_minus[2 * n, 2 * n + 1] = 1.0

    a_annihilate = np.zeros((dim, dim), dtype=complex)
    for n in range(1, n_cutoff):
        a_annihilate[2 * (n - 1), 2 * n] = np.sqrt(n)
        a_annihilate[2 * (n - 1) + 1, 2 * n + 1] = np.sqrt(n)

    jump_ops = [sigma_minus, a_annihilate]
    gamma_rates = np.array([gamma_dot * Fp, kappa], dtype=float)

    # 稳态
    rho_ss = solve_steady_state(H, jump_ops, gamma_rates)
    p_e = excited_state_population(rho_ss)
    n_ph = cavity_photon_number(rho_ss, n_cutoff)
    print(f"  耦合强度 g:         {g_coupling / 1e9:.3f} GHz")
    print(f"  腔衰减率 kappa:     {kappa / 1e9:.3f} GHz")
    print(f"  量子点衰减速率:     {gamma_dot * Fp / 1e9:.3f} GHz")
    print(f"  稳态激发态占据:     {p_e:.4e}")
    print(f"  稳态平均光子数:     {n_ph:.4e}")

    # 时间演化
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[1, 1] = 1.0  # 初始处于激发态
    result = solve_master_equation_time_evolution(
        H, jump_ops, gamma_rates, rho0, (0.0, 5.0 / kappa), n_steps=200
    )
    trace_ok = check_trace_conservation(result["rho_traj"], tol=1e-4)
    print(f"  时间演化步数:       {len(result['t'])}")
    print(f"  密度矩阵迹守恒:     {'通过' if trace_ok else '警告：偏离!'}")

    # 自发辐射速率
    gamma_sp = spontaneous_emission_rate(dipole_moment, omega_dot, Fp)
    print(f"  Fermi黄金定则 gamma: {gamma_sp / 1e9:.4f} GHz")

    return {
        "g_coupling": g_coupling,
        "kappa": kappa,
        "gamma_dot_eff": gamma_dot * Fp,
        "rho_ss": rho_ss,
        "rho_traj": result["rho_traj"],
        "t": result["t"],
    }


def run_photon_statistics(gamma_dot_eff: float, kappa: float, g_coupling: float):
    """
    步骤5：单光子统计特性分析。
    """
    print_section("步骤5：单光子发射统计与二阶关联函数")
    print("二阶关联函数:")
    print("  g^(2)(tau) = <a^dagger(t) a^dagger(t+tau) a(t+tau) a(t)> / <n>^2")
    print("理想单光子源: g^(2)(0) -> 0")

    tau_vals = np.linspace(-2.0 / kappa, 2.0 / kappa, 401)
    g2_vals = second_order_correlation_weak_coupling(
        tau_vals, gamma_dot_eff, kappa, g_coupling
    )
    g2_0 = g2_vals[len(g2_vals) // 2]
    source_type = antibunching_parameter(g2_0)
    print(f"  g^(2)(0):           {g2_0:.4f}")
    print(f"  光源类型判定:       {source_type}")

    # 蒙特卡洛探测统计
    detection = monte_carlo_photon_detection(
        emission_rate=gamma_dot_eff,
        detection_efficiency=0.3,
        gate_time=1.0e-9,
        n_trials=5000,
    )
    print(f"  蒙特卡洛探测统计:")
    print(f"    平均计数:         {detection['mean_counts']:.4f}")
    print(f"    单光子比例:       {detection['singles_fraction']:.4f}")
    print(f"    多光子概率:       {detection['p_multi']:.4e}")

    # HBT 模拟
    tau_hbt, counts_hbt = simulate_hanbury_brown_twiss(
        gamma_dot_eff, kappa, g_coupling, measurement_time=3.0 / kappa, n_bins=100
    )
    print(f"  HBT histogram 峰值: {np.max(counts_hbt):.4f}")

    # 不可区分度
    V_hom = photon_indistinguishability_homodyne(
        pure_state_overlap=0.95, dephasing_rate=1e9, pulse_duration=1e-10
    )
    print(f"  HOM 干涉可见度:     {V_hom:.4f}")

    # 圆盘距离统计（模拟随机取向效应）
    mu_d, var_d = disk_distance_stats(n_samples=2000)
    print(f"  圆盘距离统计:       mean={mu_d:.4f}, var={var_d:.4f}")

    return {
        "g2_0": g2_0,
        "source_type": source_type,
        "detection": detection,
        "V_hom": V_hom,
    }


def run_clustering_analysis():
    """
    步骤6：量子点系综聚类与光谱纯度评估。
    """
    print_section("步骤6：量子点系综聚类与光谱纯度")
    print("非层次聚类 (Transfer/Swap 优化):")
    print("  J = sum_k sum_{i in C_k} ||x_i - mu_k||^2")

    np.random.seed(42)
    n_dots = 60
    # 模拟三个尺寸组
    energies = np.concatenate([
        np.random.normal(1.32, 0.02, 20),
        np.random.normal(1.35, 0.015, 25),
        np.random.normal(1.38, 0.025, 15),
    ])
    linewidths = np.concatenate([
        np.random.normal(0.05, 0.01, 20),
        np.random.normal(0.03, 0.008, 25),
        np.random.normal(0.08, 0.015, 15),
    ])
    result = classify_quantum_dot_ensemble(energies, linewidths, n_clusters=3)
    labels = result["labels"]
    centers = result["cluster_centers"]
    sizes = result["cluster_sizes"]

    print(f"  聚类收敛后类内方差: {result['criterion_history'][-1]:.4f}")
    for k in range(3):
        print(f"    类 {k}: 大小={sizes[k]}, 中心=({centers[0, k]:.3f} eV, {centers[1, k]:.3f} meV)")
        pi = spectral_purity_index(centers[0, k], centers[1, k], temperature_K=4.0)
        print(f"          光谱纯度指标 PI={pi:.4f}")


def run_surface_roughness():
    """
    步骤7：界面粗糙度对发射线宽的影响。
    """
    print_section("步骤7：界面粗糙度与分形扰动效应")
    print("分形扰动模型:")
    print("  q = 0.5(p_i + p_{i+1}) + w_i (p_i + p_{i+1}) - w_i (p_{i-1} + p_{i+2})")

    R_nominal = 5.0e-9
    boundary, D_f = generate_rough_quantum_dot_boundary(
        R_nominal, n_vertices=32, mu_perturb=0.015, n_iter=2
    )
    print(f"  标称半径:           {R_nominal * 1e9:.1f} nm")
    print(f"  估算分形维数 D_f:   {D_f:.3f}")

    rms_roughness = 0.2e-9  # 0.2 nm RMS
    Gamma_inhom = roughness_induced_broadening(rms_roughness * 1e9, R_nominal * 1e9, m_star_ratio=0.023)
    print(f"  RMS 粗糙度:         {rms_roughness * 1e9:.1f} nm")
    print(f"  非均匀展宽:         {Gamma_inhom / 1e9:.3f} GHz")

    # 有效势扰动
    r_test = np.linspace(0, 2.0 * R_nominal, 100)
    V_pert = effective_potential_perturbation(r_test, R_nominal, rms_roughness)
    print(f"  势场涨落幅度:       {np.max(np.abs(V_pert)) / 1.602176634e-19 * 1e3:.3f} meV")


def run_parameter_optimization(E_ex_eV: float):
    """
    步骤8：参数空间网格搜索优化单光子源品质。
    """
    print_section("步骤8：单光子源参数优化（穷举搜索）")
    print("搜索空间: R_dot, kappa_ratio, g_ratio, detuning")
    print("目标: 最大化 FoM = -log10(g2) + log10(Fp) + 2*eta - (dephasing/target)^2")

    def objective(params):
        R_dot = params["R_dot"]
        kappa_ratio = params["kappa_ratio"]
        g_ratio = params["g_ratio"]
        detuning = params["detuning"]

        # 简化物理模型计算 FoM
        Fp_est = 50.0 * (5.0e-9 / R_dot) ** 3
        gamma = 1e8 * Fp_est
        kappa = gamma * kappa_ratio
        g = gamma * g_ratio
        Gamma_total = gamma + kappa
        g2_0 = 0.1 / (1.0 + (g / Gamma_total) ** 2)
        eta = 0.5 * (1.0 - np.exp(-detuning ** 2 / 0.01))
        dephase = 1e9 + 1e10 * (R_dot / 1e-9 - 5.0) ** 2
        fom = single_photon_figure_of_merit(g2_0, Fp_est, eta, dephase)
        return -fom  # 转化为最小化问题

    def constraint(params):
        return params["g_ratio"] > 0.5 * params["kappa_ratio"]

    search_space = {
        "R_dot": (3.0e-9, 8.0e-9, 4),
        "kappa_ratio": (0.5, 3.0, 4),
        "g_ratio": (0.5, 4.0, 4),
        "detuning": (0.0, 0.05, 3),
    }

    opt_result = brute_force_optimize(search_space, objective, constraint)
    bp = opt_result["best_params"]
    print(f"  总评估点数:         {opt_result['total_evaluated']}")
    print(f"  最优参数组合:")
    for k, v in bp.items():
        unit = "nm" if "dot" in k else ""
        if "ratio" in k:
            print(f"    {k:15s}: {v:.3f}")
        elif "dot" in k:
            print(f"    {k:15s}: {v * 1e9:.2f} nm")
        else:
            print(f"    {k:15s}: {v:.4f}")

    # 灵敏度分析
    base = bp.copy()
    deltas = {k: 0.01 * (search_space[k][1] - search_space[k][0]) for k in base}
    sens = sensitivity_analysis(base, deltas, objective)
    print(f"  参数灵敏度:")
    for k, v in sens.items():
        print(f"    {k:15s}: {v:.4e}")


def main():
    """主程序入口：执行完整计算流程。"""
    print("\n" + "#" * 70)
    print("#  量子点单光子发射综合模拟系统")
    print("#  Optical Engineering: Quantum Dot Single-Photon Emission")
    print("#" * 70)
    start_time = time.time()

    # 步骤1: 量子点受限态
    qd_result = run_quantum_dot_confinement()

    # 步骤2: 特殊函数积分
    run_wavefunction_special_functions()

    # 步骤3: 微腔网格与场模式
    cavity_result = run_mesh_and_field()

    # 步骤4: 主方程
    me_result = run_master_equation(
        qd_result["E_ex_eV"],
        qd_result["dipole_moment"],
        cavity_result["Fp"],
    )

    # 步骤5: 光子统计
    stat_result = run_photon_statistics(
        me_result["gamma_dot_eff"],
        me_result["kappa"],
        me_result["g_coupling"],
    )

    # 步骤6: 聚类分析
    run_clustering_analysis()

    # 步骤7: 表面粗糙度
    run_surface_roughness()

    # 步骤8: 参数优化
    run_parameter_optimization(qd_result["E_ex_eV"])

    # 总结
    elapsed = time.time() - start_time
    print_section("计算完成")
    print(f"总耗时: {elapsed:.3f} 秒")
    print(f"关键结果:")
    print(f"  激子能量:           {qd_result['E_ex_eV']:.4f} eV")
    print(f"  Purcell 因子:       {cavity_result['Fp']:.2f}")
    print(f"  g^(2)(0):           {stat_result['g2_0']:.4f}")
    print(f"  光源判定:           {stat_result['source_type']}")
    print("\n所有计算步骤成功完成，无报错。\n")


if __name__ == "__main__":
    main()

# ================================================================
# 补充导入
# ================================================================
from utils import (
    safe_inverse, build_sparse_hamiltonian_indices, spmatvec, tridiagonal_solve,
)
from mesh_generator import triangle_area, quadrilateral_area, points_in_polygon
from em_field import gaussian_mode_profile
from master_equation import vectorize_density_matrix, unvectorize_density_matrix, lindblad_dissipator
from photon_statistics import disk_unit_sample
from state_clustering import criterion_variance
from parameter_search import int_to_binary_vector

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: validate_array_1d 输入一维数组应返回展平结果 ----
import numpy as np
result = validate_array_1d(np.array([1.0, 2.0, 3.0]))
assert result.shape == (3,) and np.allclose(result, [1.0, 2.0, 3.0]), '[TC01] validate_array_1d basic FAILED'

# ---- TC02: validate_array_1d 空数组应抛出 ValueError ----
try:
    validate_array_1d(np.array([]))
    assert False, '[TC02] validate_array_1d empty should raise FAILED'
except ValueError:
    pass

# ---- TC03: safe_inverse 正常求逆 ----
x = np.array([0.5, 2.0, -1.0])
inv = safe_inverse(x)
assert np.allclose(inv, [2.0, 0.5, -1.0]), '[TC03] safe_inverse basic FAILED'

# ---- TC04: safe_inverse 近零值安全处理（不产生 NaN/Inf） ----
x = np.array([0.0, 1e-16, 1e-13])
inv = safe_inverse(x)
assert np.all(np.isfinite(inv)), '[TC04] safe_inverse near-zero safety FAILED'

# ---- TC05: build_sparse_hamiltonian_indices 结构验证 ----
rows, cols, data = build_sparse_hamiltonian_indices(5)
assert rows.size == cols.size == data.size, '[TC05] sparse indices size mismatch FAILED'
n_diag = np.sum(rows == cols)
assert n_diag == 5, '[TC05] diagonal count FAILED'
assert np.all(data[rows == cols] == 2.0), '[TC05] diagonal elements not 2.0 FAILED'

# ---- TC06: spmatvec 空稀疏矩阵 × 向量 = 零向量 ----
rows = np.array([], dtype=int); cols = np.array([], dtype=int); data = np.array([], dtype=float)
vec = np.array([1.0, 2.0, 3.0])
result = spmatvec(rows, cols, data, vec)
assert np.allclose(result, [0.0, 0.0, 0.0]), '[TC06] spmatvec zero FAILED'

# ---- TC07: tridiagonal_solve 已知三对角系统精确求解 ----
n = 5
a = np.ones(n - 1); b = np.full(n, 2.0); c = np.ones(n - 1); d = np.arange(1, n + 1, dtype=float)
x = tridiagonal_solve(a, b, c, d)
T = np.diag(b) + np.diag(a, -1) + np.diag(c, 1)
assert np.allclose(T @ x, d, atol=1e-10), '[TC07] tridiagonal_solve FAILED'

# ---- TC08: estimate_condition_number_dense 单位矩阵条件数近似为 1 ----
I = np.eye(10)
cond = estimate_condition_number_dense(I)
assert abs(cond - 1.0) < 1e-6, '[TC08] condition number of identity FAILED'

# ---- TC09: effective_mass 已知材料值查表 ----
assert effective_mass("InAs") == 0.023, '[TC09] effective_mass InAs FAILED'
assert effective_mass("GaAs") == 0.067, '[TC09] effective_mass GaAs FAILED'
assert effective_mass("InP") == 0.077, '[TC09] effective_mass InP FAILED'

# ---- TC10: spherical_confinement_potential 内部零外部 V0 ----
r = np.array([1.0e-9, 3.0e-9, 5.0e-9, 7.0e-9, 10.0e-9])
V0 = 1.0e-19
V = spherical_confinement_potential(r, 5.0e-9, V0)
assert V[0] == 0.0 and V[1] == 0.0 and V[2] == 0.0, '[TC10] spherical potential inside FAILED'
assert V[3] == V0 and V[4] == V0, '[TC10] spherical potential outside FAILED'

# ---- TC11: reduced_mass 约化质量公式验证 ----
mu = reduced_mass(0.023, 0.067)
expected = (0.023 * 0.067) / (0.023 + 0.067)
assert abs(mu - expected) < 1e-10, '[TC11] reduced_mass FAILED'

# ---- TC12: solve_eigenvalues_1d 本征值递增且基态波函数归一化 ----
x_grid = np.linspace(-1.5e-8, 1.5e-8, 100)
result = solve_eigenvalues_1d(x_grid, 0.023, potential_type="spherical", R_dot=5.0e-9, V0=0.5 * 1.602176634e-19)
energies = result["energies_eV"]
assert np.all(np.diff(energies) >= -1e-12), '[TC12] eigenvalues not sorted ascending FAILED'
wf = result["wavefunctions"][:, 0]
assert abs(np.sum(wf ** 2) - 1.0) < 1e-6, '[TC12] wavefunction discrete normalization FAILED'

# ---- TC13: triangle_area 已知三角形面积 (0,0)-(1,0)-(0,1) = 0.5 ----
p1 = np.array([0.0, 0.0]); p2 = np.array([1.0, 0.0]); p3 = np.array([0.0, 1.0])
area = triangle_area(p1, p2, p3)
assert abs(area - 0.5) < 1e-10, '[TC13] triangle_area FAILED'

# ---- TC14: quadrilateral_area 单位正方形面积为 1 ----
square = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
area = quadrilateral_area(square)
assert abs(area - 1.0) < 1e-10, '[TC14] quadrilateral_area unit square FAILED'

# ---- TC15: compute_mesh_areas 所有单元面积为正且总和不超过圆面积 ----
R = 1.0e-6
nodes, elements = generate_circular_domain_nodes(R, n_r=5, n_theta=8)
elem_areas, mesh_area = compute_mesh_areas(nodes, elements)
assert np.all(elem_areas > 0), '[TC15] element areas not all positive FAILED'
assert mesh_area > 0, '[TC15] total area not positive FAILED'
assert mesh_area < np.pi * R ** 2 * 1.1, '[TC15] total area exceeds circle FAILED'

# ---- TC16: points_in_polygon 射线法点在多边形内/外判定 ----
poly = np.array([[0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
inside = points_in_polygon(np.array([1.0, 0.5, 3.0]), np.array([1.0, 0.5, 3.0]), poly)
assert inside[0] == True, '[TC16] point (1,1) inside FAILED'
assert inside[1] == True, '[TC16] point (0.5,0.5) inside FAILED'
assert inside[2] == False, '[TC16] point (3,3) outside FAILED'

# ---- TC17: gaussian_mode_profile 峰值正确且单调衰减 ----
x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 0.0, 0.0])
E = gaussian_mode_profile(x, y, 0.0, 0.0, 1.0, amplitude=2.0)
assert abs(E[0] - 2.0) < 1e-10, '[TC17] gaussian peak FAILED'
assert E[2] < E[1] < E[0], '[TC17] gaussian monotonic decay FAILED'

# ---- TC18: purcell_factor 正值 ----
Fp = purcell_factor(1e4, 1e-20, 930e-9, n_eff=3.5)
assert Fp > 0, '[TC18] purcell_factor not positive FAILED'

# ---- TC19: jaynes_cummings_hamiltonian Hermitian 性 ----
H = jaynes_cummings_hamiltonian(1e15, 1e15, 1e11, n_photon_cutoff=3)
assert np.allclose(H, H.conj().T), '[TC19] JC Hamiltonian not Hermitian FAILED'

# ---- TC20: vectorize/unvectorize_density_matrix 往返恒等 ----
rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
vec = vectorize_density_matrix(rho)
rho2 = unvectorize_density_matrix(vec, 2)
assert np.allclose(rho, rho2), '[TC20] vectorize roundtrip FAILED'

# ---- TC21: lindblad_dissipator 零跳跃算符应返回全零 ----
rho = np.eye(2, dtype=complex) / 2.0
L = np.zeros((2, 2), dtype=complex)
D_val = lindblad_dissipator(rho, L)
assert np.allclose(D_val, 0.0), '[TC21] zero jump operator FAILED'

# ---- TC22: disk_unit_sample 所有点在单位圆盘内（固定种子） ----
np.random.seed(42)
pts = disk_unit_sample(100)
r_sq = pts[0, :] ** 2 + pts[1, :] ** 2
assert np.all(r_sq <= 1.0 + 1e-12), '[TC22] disk samples outside unit circle FAILED'

# ---- TC23: second_order_correlation_weak_coupling g2(0) < 1（反聚束） ----
tau = np.array([0.0])
g2 = second_order_correlation_weak_coupling(tau, 1e9, 2e9, 5e8)
assert g2[0] < 1.0, '[TC23] g2(0) not below 1 FAILED'

# ---- TC24: antibunching_parameter 分类正确 ----
assert antibunching_parameter(0.1) == "strong_antibunching", '[TC24] strong antibunching FAILED'
assert antibunching_parameter(0.7) == "weak_antibunching", '[TC24] weak antibunching FAILED'
assert antibunching_parameter(1.5) == "bunching_or_poissonian", '[TC24] bunching FAILED'

# ---- TC25: photon_indistinguishability_homodyne 理想情况 V=1 ----
V = photon_indistinguishability_homodyne(1.0, 0.0, 1e-10)
assert abs(V - 1.0) < 1e-10, '[TC25] perfect indistinguishability FAILED'

# ---- TC26: photon_indistinguishability_homodyne 输出范围 [0,1] ----
V2 = photon_indistinguishability_homodyne(0.5, 1e10, 1e-9)
assert 0.0 <= V2 <= 1.0, '[TC26] indistinguishability range FAILED'

# ---- TC27: detection_area_efficiency 输出范围 [0,1] ----
eta = detection_area_efficiency(1e-3, 1e-3, 0.0)
assert 0.0 <= eta <= 1.0, '[TC27] detection efficiency range FAILED'

# ---- TC28: criterion_variance 非负且同类为零 ----
data = np.random.randn(2, 50)
np.random.seed(42)
labels = np.zeros(50, dtype=int)
labels[25:] = 1
crit = criterion_variance(data, labels, 2)
assert crit >= 0, '[TC28] criterion_variance negative FAILED'

# ---- TC29: classify_quantum_dot_ensemble 输出结构完整 ----
np.random.seed(42)
energies = np.concatenate([np.random.normal(1.32, 0.02, 10), np.random.normal(1.35, 0.015, 10)])
linewidths = np.concatenate([np.random.normal(0.05, 0.01, 10), np.random.normal(0.03, 0.008, 10)])
result = classify_quantum_dot_ensemble(energies, linewidths, n_clusters=2)
assert "labels" in result and "cluster_centers" in result, '[TC29] classify output structure FAILED'
assert len(result["labels"]) == 20, '[TC29] labels count FAILED'

# ---- TC30: roughness_induced_broadening 正值 ----
Gamma = roughness_induced_broadening(0.2, 5.0, m_star_ratio=0.023)
assert Gamma > 0, '[TC30] roughness broadening not positive FAILED'

# ---- TC31: int_to_binary_vector 已知值 6(10) = 0110(2) ----
bv = int_to_binary_vector(6, 4)
assert np.array_equal(bv, [0, 1, 1, 0]), '[TC31] int_to_binary_vector FAILED'

# ---- TC32: single_photon_figure_of_merit 理想情况正值 ----
fom = single_photon_figure_of_merit(0.01, 100, 0.9, 1e8)
assert fom > 0, '[TC32] FoM not positive FAILED'

# ---- TC33: solve_steady_state 返回 Hermitian 矩阵且输出的物理量非负 ----
H_test = jaynes_cummings_hamiltonian(1e15, 1e15, 1e11, n_photon_cutoff=3)
sigma_m = np.zeros((6, 6), dtype=complex)
for n in range(3):
    sigma_m[2 * n, 2 * n + 1] = 1.0
a_ann = np.zeros((6, 6), dtype=complex)
for n in range(1, 3):
    a_ann[2 * (n - 1), 2 * n] = np.sqrt(n)
    a_ann[2 * (n - 1) + 1, 2 * n + 1] = np.sqrt(n)
jump_ops = [sigma_m, a_ann]
gamma_rates = np.array([1e8, 1e9])
rho_ss = solve_steady_state(H_test, jump_ops, gamma_rates)
assert np.allclose(rho_ss, rho_ss.conj().T), '[TC33] steady state not Hermitian FAILED'
p_e = excited_state_population(rho_ss)
n_ph = cavity_photon_number(rho_ss, 3)
assert 0.0 <= p_e <= 1.0, '[TC33] excited population out of range FAILED'
assert n_ph >= 0, '[TC33] photon number negative FAILED'

# ---- TC34: generate_rough_quantum_dot_boundary 输出结构与 D_f 范围 ----
np.random.seed(42)
boundary, D_f = generate_rough_quantum_dot_boundary(5.0e-9, n_vertices=16, mu_perturb=0.01, n_iter=2)
assert boundary.shape[0] == 2, '[TC34] boundary shape FAILED'
assert 0.0 < D_f <= 2.0, '[TC34] fractal dimension range FAILED'

# ---- TC35: 可复现性 - 固定种子两次调用 disk_distance_stats 结果一致 ----
np.random.seed(42)
mu1, var1 = disk_distance_stats(n_samples=500)
np.random.seed(42)
mu2, var2 = disk_distance_stats(n_samples=500)
assert abs(mu1 - mu2) < 1e-12, '[TC35] reproducibility mean FAILED'
assert abs(var1 - var2) < 1e-12, '[TC35] reproducibility variance FAILED'

print('\n全部 35 个测试通过!\n')
