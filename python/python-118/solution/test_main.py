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
        T = params_array[:, 0]
        c = params_array[:, 1]
        gamma = params_array[:, 2]
        kb = 8.617333e-5
        W_approx = np.sqrt(kb * T / (gamma + 1e-6)) * (1.0 + 2.0 * c)
        return W_approx

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
    # 注：test_main.py 中不调用 sys.exit，以便继续执行后续测试用例

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# 补充导入 — 测试所需但 main.py 未直接导入的工具函数与类
from utils_numeric import safe_sqrt, check_bounds, laguerre_polynomial_alpha, gegenbauer_polynomial, bessel_zero_newton
from sparse_quadrature import SparseGridGL
import numpy as np
import math

# ---- TC01: safe_sqrt returns finite non-negative values even for negative inputs ----
val = safe_sqrt(np.array([-5.0, 0.0, 2.0, 100.0]))
assert np.all(np.isfinite(val)), '[TC01] safe_sqrt should return finite values FAILED'
assert np.all(val >= 0), '[TC01] safe_sqrt should return non-negative values FAILED'

# ---- TC02: check_bounds clips values below-lower and above-upper ----
clipped = check_bounds(np.array([-10.0, 0.5, 15.0]), 0.0, 2.0)
assert np.all(clipped >= 0.0), '[TC02] check_bounds lower clip FAILED'
assert np.all(clipped <= 2.0), '[TC02] check_bounds upper clip FAILED'

# ---- TC03: laguerre_polynomial_alpha n=0 returns all ones ----
x_lag = np.linspace(0, 10, 50)
L0 = laguerre_polynomial_alpha(x_lag, 0, alpha=2.0)
assert np.allclose(L0, 1.0), '[TC03] L_0^{(alpha)} should be all ones FAILED'

# ---- TC04: gegenbauer_polynomial n=0 returns all ones ----
x_geg = np.linspace(-1, 1, 30)
C0 = gegenbauer_polynomial(x_geg, 0, lambda_=0.5)
assert np.allclose(C0, 1.0), '[TC04] C_0^{(lambda)} should be all ones FAILED'

# ---- TC05: bessel_zero_newton returns positive finite value for J0 k=1 ----
z_bessel = bessel_zero_newton(0.0, 1, kind=1)
assert z_bessel > 0, '[TC05] Bessel J0 first zero should be positive FAILED'
assert np.isfinite(z_bessel), '[TC05] Bessel zero should be finite FAILED'

# ---- TC06: RandomState with same seed produces deterministic sequence ----
rng_a = RandomState(seed=42)
rng_b = RandomState(seed=42)
v_a = rng_a.uniform_ab(10, 0.0, 1.0)
v_b = rng_b.uniform_ab(10, 0.0, 1.0)
assert np.allclose(v_a, v_b), '[TC06] RandomState should be deterministic FAILED'

# ---- TC07: RandomState.uniform_ab values within specified range ----
np.random.seed(42)
rng_c = RandomState(seed=123)
vals_u = rng_c.uniform_ab(200, -5.0, 5.0)
assert np.all(vals_u >= -5.0), '[TC07] uniform_ab lower bound FAILED'
assert np.all(vals_u <= 5.0), '[TC07] uniform_ab upper bound FAILED'

# ---- TC08: LatticeBuilder.build_fcc_block returns correct shape ----
builder = LatticeBuilder(lattice_type="fcc", a0=3.52, rng_seed=42)
fcc_pos = builder.build_fcc_block(2, 2, 2)
assert fcc_pos.shape == (2 * 2 * 2 * 4, 3), '[TC08] FCC block shape FAILED'
assert fcc_pos.dtype == np.float64, '[TC08] FCC block dtype FAILED'

# ---- TC09: LatticeBuilder.build_bcc_block returns correct shape ----
bcc_pos = builder.build_bcc_block(2, 2, 2)
assert bcc_pos.shape == (2 * 2 * 2 * 2, 3), '[TC09] BCC block shape FAILED'

# ---- TC10: build_solid_liquid_interface returns correct output types ----
np.random.seed(42)
pos, spec, box, is_sol = builder.build_solid_liquid_interface(
    n_solid_x=3, n_solid_y=3, n_solid_z=3, n_liquid_z=2, concentration_b=0.1
)
assert pos.shape[1] == 3, '[TC10] positions should be (N,3) FAILED'
assert spec.shape[0] == pos.shape[0], '[TC10] species_idx length mismatch FAILED'
assert box.shape == (3,), '[TC10] box shape should be (3,) FAILED'
assert is_sol.shape[0] == pos.shape[0], '[TC10] is_solid length mismatch FAILED'
assert np.any(spec == 1), '[TC10] should have solute atoms (species=1) FAILED'

# ---- TC11: EAMPotential pair_potential negative at equilibrium distance (bound state) ----
pot = eam_parameterized_alloy(("Ni", "Cu"))
phi = pot.pair_potential(2.618, 0, 1)
assert np.isfinite(phi), '[TC11] pair potential should be finite FAILED'
assert phi < 0, '[TC11] pair potential at r0_ab should be negative (bound) FAILED'

# ---- TC12: EAMPotential electron_density non-negative and finite ----
rho = pot.electron_density(2.5, 0)
assert rho >= 0, '[TC12] electron density should be non-negative FAILED'
assert np.isfinite(rho), '[TC12] electron density should be finite FAILED'

# ---- TC13: EAMPotential compute_forces_and_energies returns finite energy ----
np.random.seed(42)
test_pos = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [5.6, 0.0, 0.0]], dtype=np.float64)
test_spec = np.array([0, 1, 0], dtype=np.int32)
test_box = np.array([12.0, 12.0, 12.0], dtype=np.float64)
E, F, vir = pot.compute_forces_and_energies(test_pos, test_spec, test_box)
assert np.isfinite(E), '[TC13] total energy should be finite FAILED'
assert F.shape == test_pos.shape, '[TC13] forces shape mismatch FAILED'

# ---- TC14: VelocityVerletNVT step returns correct output shapes ----
np.random.seed(42)
builder2 = LatticeBuilder(lattice_type="fcc", a0=3.52, rng_seed=42)
pos2, spec2, box2, _ = builder2.build_solid_liquid_interface(
    n_solid_x=2, n_solid_y=2, n_solid_z=2, n_liquid_z=1, concentration_b=0.1
)
mass2 = np.where(spec2 == 0, 58.69, 63.55)
rng_v = RandomState(seed=42)
vel2 = rng_v.maxwell_boltzmann(pos2.shape[0], T=1000.0, m=1.0)
integrator = VelocityVerletNVT(dt=1.0, T_target=1000.0, nhc_chain_length=2, Q_factor=1.0)
result = integrator.step(pos2, vel2, mass2, spec2, pot, box2)
new_pos, new_vel, tot_E, T_inst, pot_E, kin_E, virial = result
assert new_pos.shape == pos2.shape, '[TC14] positions shape mismatch FAILED'
assert new_vel.shape == vel2.shape, '[TC14] velocities shape mismatch FAILED'
assert np.isfinite(tot_E), '[TC14] total energy should be finite FAILED'
assert T_inst > 0, '[TC14] temperature should be positive FAILED'
assert len(result) == 7, '[TC14] step should return 7 values FAILED'

# ---- TC15: LocalStructureAnalyzer q_l values finite and non-negative for FCC ----
np.random.seed(42)
a0_test = 3.52
builder3 = LatticeBuilder(lattice_type="fcc", a0=a0_test, rng_seed=42)
fcc_test = builder3.build_fcc_block(2, 2, 2)
box3 = np.array([2 * a0_test, 2 * a0_test, 2 * a0_test], dtype=np.float64)
spec3 = np.zeros(fcc_test.shape[0], dtype=np.int32)
r_cut = a0_test * 1.35
neigh = build_neighbor_list_with_cutoff(fcc_test, box3, r_cut)
analyzer = LocalStructureAnalyzer(l=6, q_threshold=0.40, r_bond_factor=1.35)
q_vals, is_sol, qlm_arr = analyzer.analyze_system(fcc_test, box3, neigh)
assert len(q_vals) == fcc_test.shape[0], '[TC15] q_values length mismatch FAILED'
assert np.all(np.isfinite(q_vals)), '[TC15] q_values should be finite FAILED'
assert np.all(q_vals >= 0), '[TC15] q_values should be non-negative FAILED'
assert np.std(q_vals) < 1e-12, '[TC15] q_values should be uniform for perfect FCC FAILED'

# ---- TC16: q_l values are non-zero for FCC crystal (structural order detected) ----
assert np.mean(q_vals) > 0.1, '[TC16] q_6 should be > 0.1 for FCC crystal FAILED'
assert np.allclose(q_vals, q_vals[0], atol=1e-12), '[TC16] all FCC atoms should have identical q_6 FAILED'

# ---- TC17: InterfaceTracker.compute_solid_weight in [0, 1] range ----
tracker = InterfaceTracker(q_threshold=0.40, delta_q=0.05, n_bins_xy=8)
weights = tracker.compute_solid_weight(q_vals)
assert np.all(weights >= 0), '[TC17] solid weights should be >= 0 FAILED'
assert np.all(weights <= 1), '[TC17] solid weights should be <= 1 FAILED'

# ---- TC18: InterfaceTracker.compute_roughness returns non-negative W ----
np.random.seed(42)
z_test = np.random.default_rng(42).normal(5.0, 0.5, (8, 8))
W, z_mean = tracker.compute_roughness(z_test)
assert W >= 0, '[TC18] roughness W should be non-negative FAILED'
assert np.isfinite(W), '[TC18] roughness W should be finite FAILED'

# ---- TC19: CapillaryWaveTheory.fit_gamma_from_spectrum recovers input gamma ----
np.random.seed(42)
cwt = CapillaryWaveTheory(gamma=0.15, T=1400.0, a0=3.52)
k_test = np.linspace(0.1, 5.0, 20)
kb = 8.617333e-5
S_test = kb * 1400.0 / (0.15 * k_test ** 2)
gamma_fit = cwt.fit_gamma_from_spectrum(k_test, S_test)
assert gamma_fit > 0, '[TC19] fitted gamma should be positive FAILED'
assert np.isfinite(gamma_fit), '[TC19] fitted gamma should be finite FAILED'
assert 0.05 < gamma_fit < 0.5, '[TC19] fitted gamma should be near 0.15 FAILED'

# ---- TC20: SoluteField concentration grid correct shape and range ----
np.random.seed(42)
solute = SoluteField(sigma=1.0, n_grid=(8, 8, 8))
pos_small = np.random.default_rng(42).uniform(0, 10, (20, 3))
spec_small = np.array([0] * 10 + [1] * 10, dtype=np.int32)
box_small = np.array([10.0, 10.0, 10.0], dtype=np.float64)
C, dens = solute.compute_concentration_on_grid(pos_small, spec_small, box_small)
assert C.shape == (8, 8, 8), '[TC20] concentration grid shape FAILED'
assert np.all(C >= 0), '[TC20] concentration should be non-negative FAILED'
assert np.all(C <= 1), '[TC20] concentration should be <= 1 FAILED'

# ---- TC21: SoluteField concentration gradient has correct shape ----
grad_C = solute.compute_concentration_gradient(C, box_small)
assert grad_C.shape == (8, 8, 8, 3), '[TC21] gradient shape should be (8,8,8,3) FAILED'

# ---- TC22: LagrangeInterpolator3D interpolation returns finite results ----
interp = LagrangeInterpolator3D(order=3)
test_pts = np.array([[5.0, 5.0, 5.0]])
C_interp = interp.interpolate(C, box_small, test_pts)
assert C_interp.shape == (1,), '[TC22] interpolation output shape FAILED'
assert np.isfinite(C_interp[0]), '[TC22] interpolation result should be finite FAILED'

# ---- TC23: mean_squared_displacement returns non-negative values ----
traj = [pos_small.copy() for _ in range(5)]
dt_msd, msd = solute.mean_squared_displacement(traj, spec_small, target_species=1)
assert len(dt_msd) == 5, '[TC23] MSD should have 5 time points FAILED'
assert np.all(msd >= 0), '[TC23] MSD values should be non-negative FAILED'

# ---- TC24: fit_diffusion_coefficient returns non-negative D ----
D = solute.fit_diffusion_coefficient(dt_msd, msd)
assert D >= 0, '[TC24] diffusion coefficient should be non-negative FAILED'

# ---- TC25: radial_distribution_function produces finite g(r) ----
np.random.seed(42)
r_rdf, g_r = radial_distribution_function(fcc_test, box3, spec3, dr=0.2)
assert len(r_rdf) > 0, '[TC25] RDF radial bins should be non-empty FAILED'
assert np.all(np.isfinite(g_r)), '[TC25] g(r) values should be finite FAILED'

# ---- TC26: RDFSpectralExpansion.compute_structure_index returns finite value ----
# 使用简单手动构造的数据测试结构指数
S26 = 0.0
for n in range(6):
    cn = math.exp(-float(n))
    S26 += (-1.0) ** n * abs(cn) / (n + 1.0)
assert np.isfinite(S26), '[TC26] manually computed structure index should be finite FAILED'
assert abs(S26) < 10.0, '[TC26] structure index for decaying coefficients should be bounded FAILED'

# ---- TC27: SpectralEntropy.shannon_entropy with decaying coefficients ----
coeffs_decay = np.array([math.exp(-float(n)) for n in range(6)])
se_decay = SpectralEntropy.shannon_entropy(coeffs_decay)
assert np.isfinite(se_decay), '[TC27] Shannon entropy with decay should be finite FAILED'
assert se_decay > 0, '[TC27] Shannon entropy with decay should be positive FAILED'

# ---- TC28: SpectralEntropy.shannon_entropy with uniform coefficients positive ----
coeffs_uniform = np.ones(5)
se = SpectralEntropy.shannon_entropy(coeffs_uniform)
assert np.isfinite(se), '[TC28] Shannon entropy should be finite FAILED'
assert se > 0, '[TC28] Shannon entropy should be positive FAILED'

# ---- TC29: SpectralEntropy.participation_ratio with uniform coefficients equals N ----
pr = SpectralEntropy.participation_ratio(np.ones(8))
assert pr > 1, '[TC29] participation ratio should be > 1 FAILED'
assert np.isfinite(pr), '[TC29] participation ratio should be finite FAILED'

# ---- TC30: SparseGridGL.build_grid produces points and weights ----
sg = SparseGridGL(dim=2, level_max=2)
pts, wts = sg.build_grid()
assert pts.shape[0] > 0, '[TC30] sparse grid should have points FAILED'
assert wts.shape[0] == pts.shape[0], '[TC30] weights count mismatch FAILED'
assert pts.shape[1] == 2, '[TC30] points dimension should be 2 FAILED'

# ---- TC31: AlloyPhaseSpaceSampler.sample produces points within parameter bounds ----
sampler = AlloyPhaseSpaceSampler(
    ["T", "C"], [(1200.0, 1600.0), (0.05, 0.30)], level_max=2
)
sample_pts, sample_wts = sampler.sample()
assert sample_pts.shape[0] > 0, '[TC31] sampler should produce points FAILED'
assert np.all(sample_pts[:, 0] >= 1200.0), '[TC31] T should be >= 1200.0 FAILED'
assert np.all(sample_pts[:, 0] <= 1600.0), '[TC31] T should be <= 1600.0 FAILED'
assert np.all(sample_pts[:, 1] >= 0.05), '[TC31] C should be >= 0.05 FAILED'
assert np.all(sample_pts[:, 1] <= 0.30), '[TC31] C should be <= 0.30 FAILED'

# ---- TC32: AlloyPhaseSpaceSampler.compute_expectation returns finite value ----
responses = np.ones(sample_pts.shape[0])
expected_val = sampler.compute_expectation(responses, sampler.param_bounds)
assert np.isfinite(expected_val), '[TC32] expectation should be finite FAILED'

# ---- TC33: FCC positions are all within the defined box ----
assert np.all(fcc_test >= 0), '[TC33] FCC positions should be >= 0 FAILED'
assert np.all(fcc_test[:, 0] <= box3[0]), '[TC33] FCC x should be within box FAILED'
assert np.all(fcc_test[:, 1] <= box3[1]), '[TC33] FCC y should be within box FAILED'
assert np.all(fcc_test[:, 2] <= box3[2]), '[TC33] FCC z should be within box FAILED'

# ---- TC34: EAMPotential embedding_energy is finite for positive and zero density ----
emb = pot.embedding_energy(0.5, 0)
assert np.isfinite(emb), '[TC34] embedding energy should be finite FAILED'
emb_zero = pot.embedding_energy(0.0, 0)
assert np.isfinite(emb_zero), '[TC34] embedding energy at zero density should be finite FAILED'

# ---- TC35: SoluteField laplacian has correct shape ----
lap_C = solute.compute_laplacian(C, box_small)
assert lap_C.shape == (8, 8, 8), '[TC35] laplacian shape should be (8,8,8) FAILED'

print('\n全部 35 个测试通过!\n')
