# -*- coding: utf-8 -*-
"""
main.py
恒星演化与核合成多物理耦合模拟器 (Stellar Evolution & Nucleosynthesis Simulator)
统一入口，零参数可运行。

科学问题：模拟一颗 1 M_sun 恒星从主序到红巨星分支的早期演化，
包括核燃烧网络、对流混合、星震学模式分析与不确定性量化。
"""

import numpy as np
import time

from stellar_grid import StellarGrid
from stellar_structure import StellarStructure
from nuclear_network import NuclearNetwork
from reaction_rates import NuclearReactionRates
from convection_diffusion import ConvectionDiffusion
from stellar_integration import StellarIntegrator
from seismic_analysis import SeismicAnalysis
from uncertainty_quantification import UncertaintyQuantification
from composition_analysis import CompositionAnalysis
from io_utils import IOUtils
from numerical_utils import safe_divide

# 太阳参数 CGS
M_SUN = 1.98847e33
R_SUN = 6.957e10
L_SUN = 3.828e33


def initialize_stellar_model(M_star: float, N_shells: int = 200) -> StellarGrid:
    """
    初始化恒星模型。
    使用多方球指数 n=3 (Eddington 标准模型) 作为初始猜测。
    密度分布：ρ = ρ_c * (sin(ξ)/ξ)^3, 其中 ξ = π r / R
    """
    grid = StellarGrid(M_star, N_shells, core_fraction=0.05, envelope_fraction=0.95)
    R_init = R_SUN * (M_star / M_SUN) ** 0.8  # 质量-半径关系近似

    # 多方球 n=3 密度分布
    xi = np.pi * np.linspace(0.0, 1.0, N_shells)
    xi[0] = 1e-6  # 避免除零
    theta = np.sin(xi) / xi
    rho_c = 54.18 * M_star / (4.0 / 3.0 * np.pi * R_init ** 3)  # n=3 中心密度近似
    rho = rho_c * theta ** 3
    rho = np.maximum(rho, 1e-10)

    # 积分求半径
    m = grid.mass
    r = np.zeros(N_shells, dtype=np.float64)
    r[0] = 1e5  # 避免中心除零 [cm]
    for i in range(1, N_shells):
        dm = m[i] - m[i - 1]
        rho_avg = 0.5 * (rho[i] + rho[i - 1])
        r[i] = (r[i - 1] ** 3 + 3.0 * dm / (4.0 * np.pi * rho_avg)) ** (1.0 / 3.0)
    r = np.maximum(r, 1e3)

    # 初始温度（粗略估计：T ∝ ρ^{1/3}）
    T_c = 1.5e7 * (M_star / M_SUN) ** 0.5
    T = T_c * (rho / rho_c) ** (1.0 / 3.0)
    T = np.maximum(T, 1e4)

    # 初始组成
    X_h1 = 0.7
    X_he4 = 0.28
    X_c12 = 0.01
    X_n14 = 0.005
    X_o16 = 0.003
    X_ne20 = 0.001
    X_mg24 = 0.001
    composition = np.array([X_h1, 0.0, X_he4, X_c12, X_n14, X_o16, X_ne20, X_mg24])

    # 物态方程初始化压强
    struct = StellarStructure(M_star, R_init, composition)
    P = np.zeros(N_shells, dtype=np.float64)
    for i in range(N_shells):
        P[i], _, _, _ = struct.equation_of_state(rho[i], T[i], composition)

    grid.radius = r
    grid.density = rho
    grid.temperature = T
    grid.pressure = P
    return grid, composition


def solve_hydrostatic_equilibrium(grid: StellarGrid, composition: np.ndarray,
                                  struct: StellarStructure, max_iter: int = 30) -> StellarGrid:
    """
    迭代求解流体静力学平衡。
    使用 Henyey 方法思想：交替更新 P(rho,T) 和 dP/dm = -Gm/(4πr⁴)。
    """
    n = grid.N_shells
    for iteration in range(max_iter):
        r_old = grid.radius.copy()
        P_old = grid.pressure.copy()
        rho_old = grid.density.copy()

        # 阶段1: 从中心向外积分半径
        for i in range(n - 1):
            dm = grid.mass[i + 1] - grid.mass[i]
            rho_avg = max(0.5 * (grid.density[i] + grid.density[i + 1]), 1e-10)
            r_cubed = grid.radius[i] ** 3 + 3.0 * dm / (4.0 * np.pi * rho_avg)
            grid.radius[i + 1] = max(r_cubed, 0.0) ** (1.0 / 3.0)

        # 阶段2: 从中心向外积分压强（使用已更新的半径）
        for i in range(n - 1):
            dm = grid.mass[i + 1] - grid.mass[i]
            m_avg = 0.5 * (grid.mass[i] + grid.mass[i + 1])
            r_avg = max(0.5 * (grid.radius[i] + grid.radius[i + 1]), 1e5)
            dP = -struct.G * m_avg * dm / (4.0 * np.pi * r_avg ** 4)
            grid.pressure[i + 1] = max(grid.pressure[i] + dP, 1e-5)

        # 阶段3: 从 P, T 更新 rho（理想气体近似）
        mu = struct.mean_molecular_weight(composition)
        for i in range(n):
            P_gas = max(grid.pressure[i] - struct.A_RAD / 3.0 * grid.temperature[i] ** 4, 1e-5)
            grid.density[i] = P_gas * mu / (struct.R_GAS * grid.temperature[i])
            grid.density[i] = max(grid.density[i], 1e-10)

        # 收敛检查
        dr_rel = safe_divide(np.abs(grid.radius - r_old), r_old, 1e10)
        dP_rel = safe_divide(np.abs(grid.pressure - P_old), P_old, 1e10)
        if np.max(dr_rel) < 1e-4 and np.max(dP_rel) < 1e-4:
            break

    return grid


def evolve_stellar_model(grid: StellarGrid, composition: np.ndarray,
                         struct: StellarStructure, network: NuclearNetwork,
                         rates: NuclearReactionRates, n_steps: int = 50,
                         dt_years: float = 1e6) -> dict:
    """
    恒星演化主循环。
    每步更新：核燃烧 -> 能量产生 -> 光度 -> 温度 -> 流体静力学平衡。
    """
    n = grid.N_shells
    dt = dt_years * 3.154e7  # 转换为秒

    # 存储演化轨迹
    track_times = []
    track_lum = []
    track_radius = []
    track_teff = []

    Y_current = composition.copy()
    # 转换为摩尔丰度 (粗略: Y_i = X_i / A_i)
    Y_molar = Y_current / network.MASS_NUMBERS
    Y_molar = Y_molar / np.sum(Y_molar) * 1e-3  # 归一化到合理量级

    for step in range(n_steps):
        # 1. 核网络演化（每个壳层）
        epsilon_nuc = np.zeros(n, dtype=np.float64)
        for i in range(n):
            T_i = grid.temperature[i]
            rho_i = grid.density[i]
            if T_i < 1e6 or rho_i < 1e-10:
                continue
            # 子步进（对核心加密）
            n_sub = 10 if i < n // 5 else 1
            dt_sub = dt / n_sub
            Y_local = Y_molar.copy()
            for _ in range(n_sub):
                Y_local = network.solve_network_rk4(Y_local, T_i, rho_i, dt_sub, n_steps=1)
            epsilon_nuc[i] = network.energy_generation_rate(Y_local, T_i, rho_i)

        # 2. 计算光度剖面
        L = struct.solve_luminosity_profile(grid.mass, epsilon_nuc, L_SUN)

        # 3. 温度梯度与对流判定
        convection_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            X_shell = network.abundances_to_mass_fractions(Y_molar)
            nabla_rad, nabla_ad, nabla_actual, is_conv = struct.temperature_gradients(
                grid.mass[i], grid.radius[i], grid.pressure[i],
                grid.temperature[i], L[i], grid.density[i], X_shell
            )
            convection_mask[i] = is_conv
            # 更新温度
            if i > 0:
                dm = grid.mass[i] - grid.mass[i - 1]
                dT = nabla_actual * grid.temperature[i] / grid.pressure[i] * (
                    -struct.G * grid.mass[i] * dm / (4.0 * np.pi * grid.radius[i] ** 4)
                )
                grid.temperature[i] += dT
            grid.temperature[i] = max(grid.temperature[i], 1e3)

        # 4. 对流区混合（仅对对流壳层）
        if np.any(convection_mask):
            conv_idx = np.where(convection_mask)[0]
            if len(conv_idx) > 2:
                r_conv = grid.radius[conv_idx]
                rho_conv = grid.density[conv_idx]
                T_conv = grid.temperature[conv_idx]
                P_conv = grid.pressure[conv_idx]
                conv_solver = ConvectionDiffusion(n_points=len(conv_idx),
                                                   r_min=r_conv.min(), r_max=r_conv.max())
                X_shell = network.abundances_to_mass_fractions(Y_molar)
                D_mix = conv_solver.convective_diffusivity(
                    r_conv, rho_conv, T_conv, P_conv,
                    nabla=0.4, nabla_ad=0.25, alpha_mlt=1.5
                )
                # 简单的均匀化混合（扩散系数极大时）
                D_mean = np.mean(D_mix)
                if D_mean > 1e12:
                    X_shell = np.mean(X_shell) * np.ones_like(X_shell)

        # 5. 流体静力学平衡更新
        grid = solve_hydrostatic_equilibrium(grid, composition, struct, max_iter=5)

        # 6. 记录演化轨迹
        R_surf = grid.radius[-1]
        L_surf = L[-1]
        T_eff = (L_surf / (4.0 * np.pi * struct.A_RAD / 3.0 * R_surf ** 2)) ** 0.25
        T_eff = max(T_eff, 1000.0)

        track_times.append(step * dt_years)
        track_lum.append(L_surf / L_SUN)
        track_radius.append(R_surf / R_SUN)
        track_teff.append(T_eff)

        # 每10步打印状态
        if step % 10 == 0:
            print(f"[Step {step:3d}] t={step*dt_years:.2e} yr, "
                  f"L={L_surf/L_SUN:.3f} L_sun, R={R_surf/R_SUN:.3f} R_sun, "
                  f"Teff={T_eff:.0f} K")

    return {
        'grid': grid,
        'times': np.array(track_times),
        'luminosities': np.array(track_lum),
        'radii': np.array(track_radius),
        'temperatures': np.array(track_teff),
        'composition': network.abundances_to_mass_fractions(Y_molar),
        'convection_mask': convection_mask,
    }


def run_seismic_analysis(grid: StellarGrid, struct: StellarStructure,
                         composition: np.ndarray) -> dict:
    """
    星震学分析：计算振动模式频率与 FFT 功率谱。
    """
    seismic = SeismicAnalysis(n_modes=30)
    cs = np.array([struct.sound_speed(grid.density[i], grid.temperature[i], composition)
                   for i in range(grid.N_shells)])

    dnu = seismic.large_frequency_separation(grid.radius, cs)
    dnu02 = seismic.small_frequency_separation(grid.radius, grid.density, cs)

    # p-模式频率
    p_modes = seismic.compute_p_mode_frequencies(n_max=15, l_max=2, dnu=dnu, epsilon=1.5)

    # g-模式频率（简化的 Brunt-Väisälä 频率）
    N_brunt = np.sqrt(struct.G * np.mean(grid.density) / np.max(grid.radius))
    g_modes = seismic.compute_g_mode_frequencies(n_g=5, l=1, N_brunt=N_brunt, R=np.max(grid.radius))

    # 模拟光变曲线并做 FFT
    time = np.linspace(0, 100 * 86400, 4096)  # 100天，4096点
    # 构造包含 p-模式频率的合成光变
    flux = np.ones_like(time)
    if len(p_modes) > 0:
        for row in p_modes[:10]:
            nu_hz = row['nu'] * 1e-6  # μHz -> Hz
            amp = 1e-4
            flux += amp * np.sin(2.0 * np.pi * nu_hz * time)
    freqs, power = seismic.frequency_spectrum_fft(flux, dt=time[1] - time[0])

    return {
        'dnu': dnu,
        'dnu02': dnu02,
        'p_modes': p_modes,
        'g_modes': g_modes,
        'fft_freqs': freqs,
        'fft_power': power,
    }


def run_uncertainty_analysis() -> dict:
    """
    不确定性量化：IMF 采样与核反应参数蒙特卡洛传播。
    """
    uq = UncertaintyQuantification(seed=42)

    # IMF 采样
    masses = uq.sample_stellar_masses(n_stars=1000, m_min=0.5, m_max=25.0, imf_type='kroupa')
    mass_mean = np.mean(masses)
    mass_std = np.std(masses)

    # 核反应参数不确定性传播
    base_params = np.array([1.0, 1.0, 1.5])  # [S_pp, S_CNO, alpha_MLT]
    param_cov = np.array([
        [0.01, 0.002, 0.0],
        [0.002, 0.02, 0.0],
        [0.0, 0.0, 0.04]
    ])

    def dummy_model(p):
        return p[0] * p[1] * (1.0 + 0.05 * (p[2] - 1.5))

    samples, outputs, stats = uq.propagate_nuclear_uncertainty(
        base_params, param_cov, n_mc=500, model_func=dummy_model
    )

    return {
        'imf_masses': masses,
        'imf_mean': mass_mean,
        'imf_std': mass_std,
        'mc_samples': samples,
        'mc_outputs': outputs,
        'mc_stats': stats,
    }


def run_composition_analysis(initial_comp: np.ndarray, final_comp: np.ndarray) -> dict:
    """
    化学丰度分析：Jaccard 距离、CNO 比值、熵等。
    """
    analyzer = CompositionAnalysis()

    jaccard = analyzer.jaccard_index(initial_comp, final_comp, threshold=1e-4)
    cosine = analyzer.cosine_similarity(initial_comp, final_comp)
    euclid = analyzer.euclidean_distance(initial_comp, final_comp)

    species = NuclearNetwork.SPECIES
    cn, on, co = analyzer.cno_ratios(final_comp, species)
    Z_initial = np.sum(initial_comp[3:])
    Z_final = np.sum(final_comp[3:])
    feh_initial = analyzer.metallicity_feh(Z_initial)
    feh_final = analyzer.metallicity_feh(Z_final)
    entropy = analyzer.entropy_abundance(final_comp)

    return {
        'jaccard_index': jaccard,
        'cosine_similarity': cosine,
        'euclidean_distance': euclid,
        'C_N_ratio': cn,
        'O_N_ratio': on,
        'C_O_ratio': co,
        'feh_initial': feh_initial,
        'feh_final': feh_final,
        'entropy': entropy,
    }


def run_integration_tests(grid: StellarGrid, struct: StellarStructure) -> dict:
    """
    数值积分测试：Newton-Cotes、三角形求积、引力结合能等。
    """
    integrator = StellarIntegrator()

    # 测试 Newton-Cotes
    f_test = lambda x: np.exp(-x ** 2)
    x_nodes, w_nodes = integrator.newton_cotes_weights(5, 0.0, 2.0)
    ncc_result = np.dot(w_nodes, f_test(x_nodes))

    # 测试三角形求积
    f_tri = lambda x, y: x ** 2 + y ** 2
    tri_result = integrator.integrate_triangle(
        f_tri, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), degree=5
    )

    # 引力结合能
    Omega = integrator.gravitational_binding_energy(grid.mass, grid.radius)
    I_rot = integrator.moment_of_inertia(grid.radius, grid.density, grid.dm)

    return {
        'newton_cotes_gaussian': ncc_result,
        'triangle_integral': tri_result,
        'gravitational_binding_energy': Omega,
        'moment_of_inertia': I_rot,
    }


def main():
    print("=" * 70)
    print("  恒星演化与核合成多物理耦合模拟器 (SENS)")
    print("  Stellar Evolution & Nucleosynthesis Simulator")
    print("  科学领域: 天体物理 - 恒星演化与核合成模拟")
    print("=" * 70)
    t_start = time.time()

    # ---------------------------------------------------------------
    # 1. 初始化恒星模型
    # ---------------------------------------------------------------
    print("\n[1/6] 初始化恒星模型 (1 M_sun, 200 shells)...")
    M_star = 1.0 * M_SUN
    grid, composition = initialize_stellar_model(M_star, N_shells=200)
    struct = StellarStructure(M_star, R_SUN, composition)
    grid = solve_hydrostatic_equilibrium(grid, composition, struct, max_iter=20)
    print(f"      初始半径: {grid.radius[-1]/R_SUN:.3f} R_sun")
    print(f"      初始中心密度: {grid.density[0]:.3e} g/cm³")
    print(f"      初始中心温度: {grid.temperature[0]:.3e} K")

    # ---------------------------------------------------------------
    # 2. 恒星演化
    # ---------------------------------------------------------------
    print("\n[2/6] 运行恒星演化 (50 步 × 1 Myr)...")
    rates = NuclearReactionRates()
    network = NuclearNetwork(rates)
    evolution = evolve_stellar_model(
        grid, composition, struct, network, rates,
        n_steps=50, dt_years=1e6
    )
    print(f"      最终光度: {evolution['luminosities'][-1]:.3f} L_sun")
    print(f"      最终半径: {evolution['radii'][-1]:.3f} R_sun")
    print(f"      最终有效温度: {evolution['temperatures'][-1]:.0f} K")

    # ---------------------------------------------------------------
    # 3. 星震学分析
    # ---------------------------------------------------------------
    print("\n[3/6] 星震学模式分析...")
    seismic = run_seismic_analysis(evolution['grid'], struct, evolution['composition'])
    print(f"      大频率分离 Δν: {seismic['dnu']:.2f} μHz")
    print(f"      小频率分离 δν_02: {seismic['dnu02']:.2f} μHz")
    print(f"      p-模式数量: {len(seismic['p_modes'])}")
    print(f"      g-模式数量: {len(seismic['g_modes'])}")

    # ---------------------------------------------------------------
    # 4. 不确定性量化
    # ---------------------------------------------------------------
    print("\n[4/6] 不确定性量化 (IMF 采样 + MC 传播)...")
    uq_results = run_uncertainty_analysis()
    print(f"      IMF 平均质量: {uq_results['imf_mean']:.3f} M_sun")
    print(f"      IMF 质量标准差: {uq_results['imf_std']:.3f} M_sun")
    print(f"      MC 输出均值: {uq_results['mc_stats'][0]:.4f}")
    print(f"      MC 输出标准差: {uq_results['mc_stats'][1]:.4f}")
    print(f"      95% 置信区间: [{uq_results['mc_stats'][4]:.4f}, {uq_results['mc_stats'][5]:.4f}]")

    # ---------------------------------------------------------------
    # 5. 化学丰度分析
    # ---------------------------------------------------------------
    print("\n[5/6] 化学丰度分析...")
    initial_comp = np.array([0.7, 0.0, 0.28, 0.01, 0.005, 0.003, 0.001, 0.001])
    comp_results = run_composition_analysis(initial_comp, evolution['composition'])
    print(f"      Jaccard 指数: {comp_results['jaccard_index']:.4f}")
    print(f"      余弦相似度: {comp_results['cosine_similarity']:.4f}")
    print(f"      C/N 比值: {comp_results['C_N_ratio']:.4f}")
    print(f"      O/N 比值: {comp_results['O_N_ratio']:.4f}")
    print(f"      [Fe/H] 初始: {comp_results['feh_initial']:.3f}")
    print(f"      [Fe/H] 最终: {comp_results['feh_final']:.3f}")
    print(f"      化学熵: {comp_results['entropy']:.4f}")

    # ---------------------------------------------------------------
    # 6. 数值积分验证
    # ---------------------------------------------------------------
    print("\n[6/6] 数值积分与物理量计算...")
    integ_results = run_integration_tests(evolution['grid'], struct)
    print(f"      Newton-Cotes 积分 exp(-x²)|₀²: {integ_results['newton_cotes_gaussian']:.6f}")
    print(f"      三角形积分 (x²+y²): {integ_results['triangle_integral']:.6f}")
    print(f"      引力结合能: {integ_results['gravitational_binding_energy']:.3e} erg")
    print(f"      转动惯量: {integ_results['moment_of_inertia']:.3e} g cm²")

    # ---------------------------------------------------------------
    # 7. 保存结果
    # ---------------------------------------------------------------
    print("\n[保存] 序列化演化数据...")
    IOUtils.write_evolution_track(
        evolution['times'], evolution['luminosities'],
        evolution['radii'], evolution['temperatures'],
        'evolution_track.txt'
    )
    model_data = {
        'mass': evolution['grid'].mass,
        'radius': evolution['grid'].radius,
        'density': evolution['grid'].density,
        'temperature': evolution['grid'].temperature,
        'pressure': evolution['grid'].pressure,
        'composition': evolution['composition'],
    }
    IOUtils.serialize_stellar_model(model_data, 'stellar_model.npz')

    # 测试矩阵条件数
    M_test, cond_test = IOUtils.test_matrix_condition(n=5)
    print(f"      5阶魔方阵条件数: {cond_test:.2e}")

    t_end = time.time()
    print("\n" + "=" * 70)
    print(f"  模拟完成，总耗时: {t_end - t_start:.2f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
