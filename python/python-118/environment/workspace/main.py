"""
main.py
分子动力学：合金固液界面动力学 — 统一入口

零参数运行，完成以下完整流程:
1. 构建二元合金(Ni-Cu)固液双相初始构型
2. EAM势场下的速度Verlet + Nosé-Hoover链 NVT分子动力学模拟
3. Steinhardt参数识别固液界面
4. 界面位置、粗糙度与毛细波谱分析
5. 溶质浓度场、分凝系数与扩散系数计算
6. 径向分布函数与谱展开分析
7. 高维参数空间稀疏网格采样

科学背景:
    二元合金凝固过程中，固液界面的原子尺度结构、溶质分凝行为
    以及界面迁移动力学是决定最终微观组织的关键。本项目采用
    嵌入原子法(EAM)描述Ni-Cu合金相互作用，通过NVT系综分子
    动力学模拟界面演化，并结合多尺度分析工具提取界面特征。

核心公式:
    EAM总能量:
        E = \sum_i F_i(\bar{\rho}_i) + \frac{1}{2}\sum_{i\neq j}\phi_{ij}(r_{ij})
    Steinhardt序参量:
        q_l(i) = \sqrt{ \frac{4\pi}{2l+1}\sum_{m=-l}^{l}|q_{lm}(i)|^2 }
    毛细波涨落谱:
        \langle |h_k|^2 \rangle = k_B T / (\gamma k^2)
    Fick扩散:
        \partial C/\partial t = D \nabla^2 C
"""

import numpy as np
import time
import sys
import os

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lattice_builder import LatticeBuilder
from eam_potential import eam_parameterized_alloy
from velocity_verlet import VelocityVerletNVT
from structure_factor import (
    LocalStructureAnalyzer, build_neighbor_list_with_cutoff, radial_distribution_function
)
from interface_dynamics import InterfaceTracker, CapillaryWaveTheory
from solute_diffusion import SoluteField, LagrangeInterpolator3D
from sparse_quadrature import AlloyPhaseSpaceSampler
from spectral_decomposition import RDFSpectralExpansion, SpectralEntropy
from utils_numeric import RandomState


def run_simulation():
    print("=" * 70)
    print("  分子动力学模拟：合金固液界面动力学")
    print("  Molecular Dynamics: Alloy Solid-Liquid Interface Dynamics")
    print("=" * 70)

    # ========== 1. 系统参数设置 ==========
    print("\n[1/7] 初始化系统参数与晶格构型...")
    lattice_type = "fcc"
    a0 = 3.52  # Ni的晶格常数 (Angstrom)
    n_solid = (4, 4, 4)  # 固体区晶胞数
    n_liquid_z = 4  # 液体区晶胞数
    concentration_b = 0.15  # 溶质B(Cu)的初始浓度
    T_target = 1400.0  # K (低于Ni熔点1728K，但高于共晶区，形成过冷液)
    dt = 1.0  # fs
    n_steps = 200  # 模拟步数 (轻量版，保证快速运行)
    print(f"    晶格类型: {lattice_type}, 晶格常数: {a0} A")
    print(f"    固相区: {n_solid}, 液相区厚度: {n_liquid_z} 晶胞")
    print(f"    溶质浓度: {concentration_b}, 目标温度: {T_target} K")
    print(f"    时间步长: {dt} fs, 模拟步数: {n_steps}")

    # 构建初始构型
    builder = LatticeBuilder(lattice_type=lattice_type, a0=a0, rng_seed=42)
    positions, species_idx, box, is_solid_init = builder.build_solid_liquid_interface(
        n_solid_x=n_solid[0], n_solid_y=n_solid[1], n_solid_z=n_solid[2],
        n_liquid_z=n_liquid_z, concentration_b=concentration_b
    )
    n_atoms = positions.shape[0]
    print(f"    总原子数: {n_atoms}")
    print(f"    模拟盒子: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}] A")

    # 初始化速度 (Maxwell-Boltzmann分布)
    rng = RandomState(seed=123)
    masses_amu = np.where(species_idx == 0, 58.69, 63.55).astype(np.float64)
    # 原子质量单位转 MD 单位 (eV * fs^2 / A^2): amu * 1.66054e-27 kg = ...
    # 简化: 直接使用amu，力单位为 eV/A，则加速度 a = F/m 需要换算
    # 统一使用 1 amu = 1.0 的简化单位系，通过调节势参数保证稳定性
    masses = masses_amu
    velocities = rng.maxwell_boltzmann(n_atoms, T=T_target, m=1.0)
    # 扣除质心速度
    v_cm = np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)
    velocities -= v_cm

    # 施加热位移扰动
    positions = builder.apply_thermal_displacement(positions, T_target, masses_amu, species_idx)
    positions -= box * np.floor(positions / box)

    # ========== 2. EAM势与积分器初始化 ==========
    print("\n[2/7] 初始化EAM势场与NVT积分器...")
    potential = eam_parameterized_alloy(species_pair=("Ni", "Cu"))
    integrator = VelocityVerletNVT(dt=dt, T_target=T_target,
                                    nhc_chain_length=3, Q_factor=1.0)
    print("    EAM势参数: Ni-Ni, Cu-Cu, Ni-Cu Morse对势 + 嵌入函数")
    print("    恒温器: Nosé-Hoover链 (长度=3)")

    # 预热 (平衡化)
    print("    预热平衡中 (50步)...")
    for step in range(50):
        positions, velocities, _, _, _, _, _ = integrator.step(
            positions, velocities, masses, species_idx, potential, box
        )

    # ========== 3. 生产模拟 + 数据采集 ==========
    print("\n[3/7] 生产模拟运行与数据采集...")
    trajectory = []
    energies = []
    temperatures = []
    kinetic_energies = []
    potential_energies = []

    sample_interval = 10  # 每10步采样一次
    for step in range(n_steps):
        positions, velocities, total_E, T_inst, pot_E, kin_E, virial = integrator.step(
            positions, velocities, masses, species_idx, potential, box
        )
        if step % sample_interval == 0:
            trajectory.append(positions.copy())
            energies.append(total_E)
            temperatures.append(T_inst)
            kinetic_energies.append(kin_E)
            potential_energies.append(pot_E)
        if step % 50 == 0:
            print(f"      Step {step:4d}/{n_steps}: T={T_inst:.1f}K  E_total={total_E:.3f}eV")

    n_frames = len(trajectory)
    print(f"    采集帧数: {n_frames}")

    # ========== 4. 局域结构分析与界面识别 ==========
    print("\n[4/7] Steinhardt序参量分析与固液界面识别...")
    analyzer = LocalStructureAnalyzer(l=6, q_threshold=0.40, r_bond_factor=1.35)
    r_cut_neigh = a0 * 1.35

    q_values_frames = []
    is_solid_frames = []
    for t in range(n_frames):
        neigh_list = build_neighbor_list_with_cutoff(trajectory[t], box, r_cut_neigh)
        q_vals, is_sol, qlm = analyzer.analyze_system(trajectory[t], box, neigh_list)
        q_values_frames.append(q_vals)
        is_solid_frames.append(is_sol)

    # 最后一帧的界面追踪
    tracker = InterfaceTracker(q_threshold=0.40, delta_q=0.05, n_bins_xy=12)
    z_interface, x_edges, y_edges = tracker.locate_interface(
        trajectory[-1], q_values_frames[-1], box
    )
    W, z_mean = tracker.compute_roughness(z_interface)
    max_cluster, n_clusters, cluster_sizes = tracker.cluster_analysis(
        is_solid_frames[-1], trajectory[-1], box, r_cut=3.5
    )

    print(f"    平均q_6 (固相): {np.mean(q_values_frames[-1][is_solid_frames[-1]]):.3f}")
    print(f"    平均q_6 (液相): {np.mean(q_values_frames[-1][~is_solid_frames[-1]]):.3f}")
    print(f"    界面平均位置: {z_mean:.2f} A, 界面粗糙度 W: {W:.3f} A")
    print(f"    最大固相团簇: {max_cluster} 原子, 团簇数: {n_clusters}")

    # 毛细波谱分析
    cwt = CapillaryWaveTheory(gamma=0.15, T=T_target, a0=a0)
    k_centers, spectrum = tracker.compute_capillary_waves_spectrum(
        z_interface, box_xy=(box[0], box[1])
    )
    gamma_fit = cwt.fit_gamma_from_spectrum(k_centers[1:], spectrum[1:])
    print(f"    拟合界面能 gamma: {gamma_fit:.4f} eV/A^2")

    # ========== 5. 溶质扩散与分凝分析 ==========
    print("\n[5/7] 溶质浓度场、分凝系数与扩散分析...")
    solute = SoluteField(sigma=1.0, n_grid=(16, 16, 16))
    C_final, density_final = solute.compute_concentration_on_grid(
        trajectory[-1], species_idx, box
    )
    grad_C = solute.compute_concentration_gradient(C_final, box)
    lap_C = solute.compute_laplacian(C_final, box)

    k_eff = solute.compute_segregation_coefficient(C_final, z_interface, box)
    dt_msd, msd = solute.mean_squared_displacement(trajectory, species_idx, target_species=1)
    D_coeff = solute.fit_diffusion_coefficient(dt_msd, msd)

    print(f"    溶质平均浓度: {np.mean(C_final):.4f}")
    print(f"    浓度梯度最大值: {np.max(np.abs(grad_C)):.4f}")
    print(f"    有效分凝系数 k_eff: {k_eff:.4f}")
    print(f"    溶质扩散系数 D: {D_coeff:.4f} A^2/fs = {D_coeff * 1e-5:.4e} cm^2/s")

    # 插值验证
    interp = LagrangeInterpolator3D(order=3)
    test_points = np.array([
        [box[0] * 0.25, box[1] * 0.25, box[2] * 0.5],
        [box[0] * 0.75, box[1] * 0.75, box[2] * 0.5]
    ])
    C_interp = interp.interpolate(C_final, box, test_points)
    print(f"    插值测试点浓度: {C_interp}")

    # ========== 6. 径向分布函数与谱分析 ==========
    print("\n[6/7] 径向分布函数与谱分解分析...")
    r_rdf, g_r = radial_distribution_function(trajectory[-1], box, species_idx, dr=0.1)

    rdf_spec = RDFSpectralExpansion(alpha=2.0, beta=0.5, n_modes=8)
    coeffs, g_recon = rdf_spec.expand(r_rdf, g_r)
    S_index = rdf_spec.compute_structure_index(coeffs)
    entropy = SpectralEntropy.shannon_entropy(coeffs)
    pr = SpectralEntropy.participation_ratio(coeffs)

    print(f"    RDF主峰位置: {r_rdf[np.argmax(g_r)]:.2f} A")
    print(f"    拉盖尔展开系数 (前4项): {coeffs[:4]}")
    print(f"    结构指数 S: {S_index:.4f}")
    print(f"    谱熵: {entropy:.4f}, 参与比: {pr:.2f}")

    # ========== 7. 高维参数空间稀疏网格采样 ==========
    print("\n[7/7] 合金相图参数空间稀疏网格采样...")
    param_names = ["Temperature_K", "Concentration_Cu", "InterfaceEnergy_eV"]
    param_bounds = [(1200.0, 1600.0), (0.05, 0.30), (0.05, 0.30)]
    sampler = AlloyPhaseSpaceSampler(param_names, param_bounds, level_max=2)

    # 定义一个响应函数：界面粗糙度关于参数的近似响应
    def roughness_response(params_array):
        """简化的响应面模型: W ~ sqrt(k_B T / gamma) * f(c)。"""
        # TODO: Hole 4 — 实现界面粗糙度响应面模型
        # 提示: 基于毛细波理论 W ~ sqrt(k_B T / gamma)，并考虑浓度修正因子。
        return np.zeros(params_array.shape[0])

    sample_points, weights = sampler.sample()
    responses = roughness_response(sample_points)
    expected_W = sampler.compute_expectation(responses, param_bounds)

    print(f"    采样点数: {sample_points.shape[0]}")
    print(f"    参数空间维度: {sampler.dim}")
    print(f"    期望界面粗糙度 <W>: {expected_W:.4f} A")
    print(f"    采样点范围:")
    for d in range(sampler.dim):
        print(f"      {param_names[d]}: [{sample_points[:, d].min():.3f}, {sample_points[:, d].max():.3f}]")

    # ========== 8. 结果汇总 ==========
    print("\n" + "=" * 70)
    print("  模拟结果汇总")
    print("=" * 70)
    print(f"  系统: Ni-Cu 二元合金, 原子数: {n_atoms}")
    print(f"  最终温度: {temperatures[-1]:.1f} K (目标: {T_target} K)")
    print(f"  总能量: {energies[-1]:.4f} eV")
    print(f"  界面位置: {z_mean:.2f} A, 粗糙度 W: {W:.3f} A")
    print(f"  拟合界面能: {gamma_fit:.4f} eV/A^2")
    print(f"  有效分凝系数: {k_eff:.4f}")
    print(f"  溶质扩散系数: {D_coeff * 1e-5:.4e} cm^2/s")
    print(f"  结构指数: {S_index:.4f}")
    print(f"  谱熵: {entropy:.4f}")
    print(f"  参数空间期望粗糙度: {expected_W:.4f} A")
    print("=" * 70)
    print("  模拟正常结束。")
    print("=" * 70)
    return True


if __name__ == "__main__":
    start_time = time.time()
    success = run_simulation()
    elapsed = time.time() - start_time
    print(f"\n运行时间: {elapsed:.2f} 秒")
    if not success:
        sys.exit(1)
    sys.exit(0)
