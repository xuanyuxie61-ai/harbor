#!/usr/bin/env python3
"""
PROJECT_292: 计算等离子体 — 磁重联与太阳耀斑能量释放
        高阶有限差分与稳定性分析（小规模可复现实验）

统一入口：零参数运行完整磁重联模拟流水线
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_generator import MeshGenerator
from high_order_fd import HighOrderFD
from resistive_mhd import ResistiveMHD1D
from stability_analysis import StabilityAnalyzer
from current_sheet import CurrentSheetEquilibrium
from spectral_laguerre import LaguerreSpectralSolver
from volume_integrals import VolumeIntegralEngine
from monte_carlo_kinetic import KineticMonteCarloSampler
from causality_detector import CausalityDetector
from reconnection_dynamics import ReconnectionDynamics
from iterative_equilibrium import DiffusiveEquilibriumSolver
from verification_metrics import VerificationMetrics
from ensemble_analyzer import EnsembleAnalyzer


def main():
    np.random.seed(42)

    print("=" * 72)
    print("  PROJECT 292: 磁重联与太阳耀斑能量释放")
    print("  高阶有限差分与 von Neumann 稳定性分析")
    print("=" * 72)

    # ===== Phase 1: Mesh =====
    print("\n[Phase 1] 生成 Harris 电流片计算域网格 ...")
    mesh = MeshGenerator(nx=128, ny=64,
                         x_range=(-6.4, 6.4), y_range=(-3.2, 3.2),
                         stretching_factor=1.2)
    mesh.build_structured_mesh()
    mesh.compute_metrics()
    print(f"  网格尺寸: {mesh.nx} x {mesh.ny}")
    print(f"  最小 dx = {mesh.dx_min:.6f}, 最小 dy = {mesh.dy_min:.6f}")
    print(f"  最大纵横比 = {mesh.max_aspect_ratio:.4f}")

    # ===== Phase 2: High-order FD =====
    print("\n[Phase 2] 构建高阶有限差分算子 ...")
    fd = HighOrderFD(order=6, nx=mesh.nx, dx=mesh.dx_uniform)
    fd.build_central_stencils()
    fd.build_upwind_stencils()
    fd.compute_dispersion_relation()
    print(f"  差分阶数: {fd.order}")
    print(f"  中心模板半宽: {fd.half_width}")
    print(f"  最大色散误差 (k*dx=pi): {fd.max_dispersion_error:.2e}")

    # ===== Phase 3: Harris sheet =====
    print("\n[Phase 3] 求解 Harris 电流片平衡态 ...")
    cs = CurrentSheetEquilibrium(B0=1.0, n0=1.0, T_e=0.1, T_i=0.1, a_sheet=0.5)
    cs.compute_harris_profile(mesh.y_centers)
    cs.check_force_balance()
    cs.compute_beta_profile()
    print(f"  Harris 磁场 Bz(x) = B0 * tanh(y/a)")
    print(f"  最大电流密度 Jx_max = {cs.J_max:.4f}")
    print(f"  最小等离子体 beta = {cs.beta_min:.4f}")
    print(f"  力平衡残差 = {cs.force_balance_residual:.2e}")

    # ===== Phase 4: Resistive MHD =====
    print("\n[Phase 4] 电阻性 MHD 时间演化 (1D 切割模式) ...")
    mhd = ResistiveMHD1D(nx=mesh.nx, dx=mesh.dx_uniform,
                         eta=1e-3, nu=5e-4, S_lundquist=1e4)
    mhd.initialize_perturbation(cs, amplitude=1e-3, k_mode=1)
    n_steps = 200
    dt = mhd.compute_cfl_timestep(fd, cfl=0.4)
    mhd.evolve(n_steps=n_steps, dt=dt, fd=fd)
    print(f"  Lundquist 数 S = {mhd.S_lundquist:.2e}")
    print(f"  电阻率 eta = {mhd.eta:.2e}")
    print(f"  CFL 时间步 dt = {dt:.6f}")
    print(f"  最大增长率 gamma = {mhd.max_growth_rate:.4e}")

    # ===== Phase 5: von Neumann stability =====
    print("\n[Phase 5] von Neumann 稳定性分析 ...")
    sa = StabilityAnalyzer(mhd=mhd, fd=fd)
    sa.analyze_advection()
    sa.analyze_diffusion()
    sa.analyze_combined_scheme()
    sa.find_critical_timestep()
    print(f"  纯对流 CFL 限 = {sa.cfl_advect_crit:.4f}")
    print(f"  纯扩散 Fo 限 = {sa.fo_diff_crit:.4f}")
    print(f"  组合方案临界 dt = {sa.dt_critical:.6f}")
    print(f"  实际 dt/dt_crit = {dt/sa.dt_critical:.4f}")

    # ===== Phase 6: Laguerre spectral =====
    print("\n[Phase 6] Laguerre 谱展开 — 电子速度空间分布 ...")
    lspec = LaguerreSpectralSolver(n_modes=12, v_max=6.0)
    lspec.build_quadrature()
    lspec.compute_expansion_coefficients(mhd.electron_distribution)
    lspec.reconstruct_distribution()
    energy_moment = lspec.compute_energy_moment()
    heat_flux = lspec.compute_heat_flux()
    print(f"  Laguerre 模式数 N = {lspec.n_modes}")
    print(f"  高斯积分点数 = {lspec.n_quad}")
    print(f"  电子热能 <v^2/2> = {energy_moment:.6f}")
    print(f"  电子热通量 q_e = {heat_flux:.6f}")
    print(f"  截断误差 = {lspec.truncation_error:.2e}")

    # ===== Phase 7: Volume integrals =====
    print("\n[Phase 7] 计算磁能、磁螺旋度体积分 ...")
    vie = VolumeIntegralEngine(mesh)
    magnetic_energy = vie.compute_magnetic_energy(mhd.Bx, mhd.By)
    magnetic_helicity = vie.compute_magnetic_helicity(mhd.Az, mhd.Bx, mhd.By)
    kinetic_energy = vie.compute_kinetic_energy(mhd.vx, mhd.vy)
    dissipation = vie.compute_ohmic_dissipation(mhd.Jz, mhd.eta)
    print(f"  磁能 W_B = {magnetic_energy:.6f}")
    print(f"  磁螺旋度 H_m = {magnetic_helicity:.6f}")
    print(f"  动能 W_K = {kinetic_energy:.6f}")
    print(f"  欧姆耗散 P_eta = {dissipation:.6f}")
    ratio = magnetic_energy / (magnetic_energy + kinetic_energy + 1e-30)
    print(f"  磁能/总能量 = {ratio:.4f}")

    # ===== Phase 8: Monte Carlo kinetic =====
    print("\n[Phase 8] 扩散区域动力学蒙特卡洛采样 ...")
    kmcs = KineticMonteCarloSampler(
        n_particles=2000, B_x=mhd.Bx, B_y=mhd.By,
        E_z=mhd.Ez, mesh=mesh)
    kmcs.initialize_particles()
    kmcs.advance_orbits(n_steps=50, dt_orbit=dt * 0.1)
    energization = kmcs.compute_energy_gain()
    pitch_angle = kmcs.compute_pitch_angle_distribution()
    print(f"  粒子数 = {kmcs.n_particles}")
    print(f"  平均能量增益 <dE> = {energization:.6f}")
    print(f"  最大能量增益 dE_max = {kmcs.max_energy_gain:.6f}")
    print(f"  平均投掷角 <alpha> = {pitch_angle:.2f} deg")

    # ===== Phase 9: Causality =====
    print("\n[Phase 9] 磁场-电流密度-电场 Granger 因果性分析 ...")
    cd = CausalityDetector(
        time_series={'B_y': mhd.By_history, 'J_z': mhd.Jz_history,
                     'E_z': mhd.Ez_history},
        max_lag=5)
    cd.compute_granger_causality()
    cd.compute_cross_correlation()
    cd.detect_causal_chains()
    print(f"  因果性矩阵 (p-value):")
    for key, row in cd.causality_matrix.items():
        vals = ", ".join([f"{v:.4f}" for v in row.values()])
        print(f"    {key}: [{vals}]")
    print(f"  检测到因果链数 = {len(cd.causal_chains)}")

    # ===== Phase 10: Reconnection dynamics =====
    print("\n[Phase 10] 重联率与 X 点动力学 ...")
    rd = ReconnectionDynamics(mhd=mhd, mesh=mesh)
    rd.locate_x_point()
    rd.compute_reconnection_rate()
    rd.analyze_outflow_jets()
    rd.compute_energy_conversion_rate()
    print(f"  X 点位置 = ({rd.x_point[0]:.3f}, {rd.x_point[1]:.3f})")
    print(f"  无量纲重联率 R = {rd.reconnection_rate:.4f}")
    print(f"  出流速度 v_out = {rd.outflow_velocity:.4f}")
    print(f"  能量转换率 dW/dt = {rd.energy_conversion_rate:.4e}")

    # ===== Phase 11: Diffusive equilibrium =====
    print("\n[Phase 11] 扩散松弛法求解力自由磁场平衡 ...")
    des = DiffusiveEquilibriumSolver(
        mesh=mesh, target_divB=1e-8,
        max_iter=100, tolerance=1e-6)
    des.initialize_from_mhd(mhd)
    des.iterate()
    des.check_force_free_condition()
    print(f"  迭代次数 = {des.n_iterations}")
    print(f"  最终 |div B| = {des.final_divB:.2e}")
    print(f"  力自由条件 alpha = J·B/B² = {des.alpha_ff:.4f}")
    print(f"  力自由残差 = {des.force_free_residual:.2e}")

    # ===== Phase 12: Ensemble statistics =====
    print("\n[Phase 12] 参数集合统计分析 ...")
    ea = EnsembleAnalyzer(base_mhd=mhd, base_cs=cs, n_runs=5)
    ea.run_parameter_sweep(eta_range=[5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    ea.compute_statistics()
    ea.analyze_transient_response()
    print(f"  参数范围: eta in [5e-4, 1e-2]")
    print(f"  重联率均值 = {ea.mean_reconnection_rate:.4f} +/- {ea.std_reconnection_rate:.4f}")
    print(f"  响应时间均值 = {ea.mean_response_time:.4f}")
    print(f"  能量释放方差 = {ea.variance_energy_release:.6f}")

    # ===== Phase 13: Verification metrics =====
    print("\n[Phase 13] 数值验证度量计算 ...")
    vm = VerificationMetrics(mhd=mhd, mesh=mesh)
    vm.compute_divergence_error()
    vm.compute_energy_conservation()
    vm.compute_symmetry_residual()
    vm.compute_convergence_order(fd)
    print(f"  |div B| 最大 = {vm.max_divB:.2e}")
    print(f"  能量守恒误差 = {vm.energy_conservation_error:.2e}")
    print(f"  对称性残差 = {vm.symmetry_residual:.2e}")
    print(f"  实测收敛阶 = {vm.measured_order:.2f}")

    print("\n" + "=" * 72)
    print("  全部计算完成。PROJECT_292 成功执行。")
    print("=" * 72)


if __name__ == "__main__":
    main()
