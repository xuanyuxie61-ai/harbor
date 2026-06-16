# -*- coding: utf-8 -*-
"""
main.py
=======
Fast Ignition ICF: 电子束能量沉积高阶有限差分与稳定性分析
统一入口 (零参数可运行)

============================================================
问题描述:
---------
在 fast ignition 惯性约束聚变方案中, 超短超强激光脉冲在稠密
等离子体临界面产生相对论电子束, 电子束穿越日冕等离子体并
将能量沉积在稠密芯部. 本代码模拟这一能量沉积过程, 使用
高阶有限差分方法并分析数值稳定性.

核心方程:
---------
    ∂u/∂t = ∇·(κ(u)∇u) + S(x, y, t)
其中 κ(u) = κ_0 u^{5/2} (Spitzer-Härm),
      S 为快电子束沉积源.

============================================================
模块结构与种子项目映射:
-----------------------
(1)  plasma_parameters     ← 物理常数, 等离子体参数
(2)  electron_beam_source  ← 816_normal (Box-Muller), 910_prime (素数种子)
(3)  vandermonde_fd        ← 1381_vandermonde (Vandermonde FD 系数)
(4)  plasma_mesh           ← 1329_triangulate_rectangle (靶区网格)
(5)  energy_deposition_ftcs← 434_fisher_pde_ftcs (FTCS PDE 求解)
(6)  electron_trajectory_rk4← 830_ode_rk4 (RK4 轨迹), 019_arneodo_ode, 018_arenstorf_ode
(7)  von_neumann_stability ← 稳定性分析, CFL 条件
(8)  conservation_monitor  ← 018_arenstorf_ode/conserved (守恒量)
(9)  convergence_benchmark ← 901_porous_medium_exact (Barenblatt 精确解)
(10) spectral_tools        ← 537_hilbert_curve (Hilbert 索引), 910_prime (谱滤波)
(11) plasma_diagnostics    ← 1234_HackBio (QC 过滤), 192_closest_point_brute, 1158_MSD
(12) phase_space_io        ← 1424_xyz_io (相空间 I/O)
(13) error_analysis        ← 338_errors (数值误差分析)
============================================================
"""

import os
import sys
import time as timer

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入所有模块
from plasma_parameters import (
    get_plasma_config, print_plasma_header,
    CORONA_ELECTRON_DENSITY, CORONA_ELECTRON_TEMPERATURE,
    CORONA_ION_CHARGE_Z, FAST_ELECTRON_TEMPERATURE,
    FAST_ELECTRON_BEAM_RADIUS, FAST_ELECTRON_PULSE_DURATION,
    FAST_ELECTRON_TOTAL_ENERGY, FAST_ELECTRON_DIVERGENCE_HALF_ANGLE,
    DOMAIN_LENGTH_X, DOMAIN_LENGTH_Y, NX, NY,
    TOTAL_TIME, MAX_TIME_STEPS, CFL_NUMBER, FD_ORDER,
    spitzer_harm_conductivity, EV_TO_JOULE,
)
from electron_beam_source import FastElectronBeam, prime_sieve, prime_count
from vandermonde_fd import print_fd_summary
from plasma_mesh import plasma_target_mesh, print_mesh_summary
from energy_deposition_ftcs import (
    run_deposition_simulation_1d, print_simulation_summary,
)
from electron_trajectory_rk4 import (
    ElectronTrajectoryParams, integrate_beam_electrons,
    print_trajectory_summary,
)
from von_neumann_stability import (
    full_stability_analysis, print_stability_summary,
)
from conservation_monitor import (
    compute_total_energy_1d, compute_conservation_drift,
    print_conservation_summary,
)
from convergence_benchmark import (
    run_barenblatt_convergence_test, print_convergence_summary,
)
from spectral_tools import (
    hilbert_order_2d_grid, prime_mode_filter,
    hilbert_locality_score, print_spectral_summary,
)
from plasma_diagnostics import (
    plasma_qc_filter, deposit_beam_to_grid,
    compute_msd_1d, estimate_diffusion_coefficient,
    print_diagnostics_summary,
)
from phase_space_io import (
    write_phase_space_xyz, read_phase_space_xyz,
    print_io_summary,
)
from error_analysis import full_error_analysis, print_error_summary


def main():
    """
    Fast Ignition ICF 电子束能量沉积模拟主程序.

    流程:
    1. 打印物理参数
    2. 生成等离子体靶网格
    3. 计算高阶 FD 系数
    4. 采样快电子束
    5. QC 过滤 + 最近邻沉积
    6. RK4 轨迹积分
    7. 一维能量沉积 PDE 求解
    8. von Neumann 稳定性分析
    9. 守恒量监测
    10. Barenblatt 收敛性验证
    11. 谱工具 (Hilbert + 素数滤波)
    12. MSD 扩散率估计
    13. 数值误差分析
    14. 相空间 I/O
    """
    start_time = timer.time()

    print("\n" + "=" * 72)
    print(" PROJECT 296: Fast Ignition ICF")
    print(" 电子束能量沉积 高阶有限差分与稳定性分析")
    print(" 小规模可复现实验 (博士级)")
    print("=" * 72)

    # ==========================================================
    # 阶段 1: 物理参数
    # ==========================================================
    print("\n[阶段 1/14] 物理参数初始化...")
    config = get_plasma_config()
    print_plasma_header()

    # ==========================================================
    # 阶段 2: 等离子体靶网格
    # ==========================================================
    print("\n[阶段 2/14] 生成等离子体靶区三角形网格...")
    mesh = plasma_target_mesh(
        Lx=DOMAIN_LENGTH_X,
        Ly=DOMAIN_LENGTH_Y,
        nx=min(NX, 32),  # 小规模
        ny=min(NY, 20),
        interface_x_frac=0.6,
    )
    print_mesh_summary(mesh)

    # ==========================================================
    # 阶段 3: 高阶 FD 系数
    # ==========================================================
    print("\n[阶段 3/14] Vandermonde 高阶有限差分系数...")
    print_fd_summary()

    # ==========================================================
    # 阶段 4: 快电子束采样
    # ==========================================================
    print("\n[阶段 4/14] Monte Carlo 快电子束采样...")
    beam = FastElectronBeam(
        n_particles=min(config["n_beam"], 128),  # 小规模
        E0_ev=FAST_ELECTRON_TEMPERATURE,
        sigma_frac=config["energy_spread"],
        r_beam_m=FAST_ELECTRON_BEAM_RADIUS,
        theta_div_rad=FAST_ELECTRON_DIVERGENCE_HALF_ANGLE,
        total_energy_j=FAST_ELECTRON_TOTAL_ENERGY,
        seed=42,
    )
    beam_summary = beam.summary()
    print("  束流统计:")
    for key, val in beam_summary.items():
        if isinstance(val, float):
            print("    {:20s}: {:.4e}".format(key, val))
        else:
            print("    {:20s}: {}".format(key, val))

    # ==========================================================
    # 阶段 5: QC 过滤 + 沉积映射
    # ==========================================================
    print("\n[阶段 5/14] 等离子体相空间 QC 过滤 + 最近邻沉积...")
    qc_result = plasma_qc_filter(
        beam.particles,
        min_energy_ev=1.0e4,
        max_energy_ev=1.0e8,
        max_divergence_rad=FAST_ELECTRON_DIVERGENCE_HALF_ANGLE * 1.5,
        sigma_clip_factor=3.0,
    )

    # 沉积到一维网格 (简化)
    x_grid = [DOMAIN_LENGTH_X * i / 31 for i in range(32)]
    y_grid = [DOMAIN_LENGTH_Y * j / 15 for j in range(16)]

    # 为 QC 通过的粒子添加位置偏移 (使在域内)
    for p in qc_result["passed"]:
        # 将相对位置映射到域内
        p["x"] = 0.3 * DOMAIN_LENGTH_X + p["x"]
        p["y"] = 0.5 * DOMAIN_LENGTH_Y + p["y"]

    source_2d = deposit_beam_to_grid(
        qc_result["passed"], x_grid[:min(len(x_grid), 16)],
        y_grid[:min(len(y_grid), 8)])

    deposit_info = {
        "nx": min(len(x_grid), 16),
        "ny": min(len(y_grid), 8),
        "source": source_2d,
    }

    # ==========================================================
    # 阶段 6: RK4 轨迹积分
    # ==========================================================
    print("\n[阶段 6/14] RK4 电子轨迹积分...")
    traj_params = ElectronTrajectoryParams(
        n_e=CORONA_ELECTRON_DENSITY,
        T_e_ev=CORONA_ELECTRON_TEMPERATURE,
        Z_eff=CORONA_ION_CHARGE_Z,
    )

    # 选择前 16 个电子进行轨迹积分
    n_traj = min(16, len(qc_result["passed"]))
    traj_particles = qc_result["passed"][:n_traj]
    traj_results = []
    for p in traj_particles:
        from electron_trajectory_rk4 import integrate_single_electron
        res = integrate_single_electron(
            E0_ev=p["energy_ev"],
            x0=p["x"],
            vx0=p["vz"],
            t_end=min(TOTAL_TIME, 2.0e-12),
            n_steps=50,
            params=traj_params,
        )
        res["particle_id"] = p.get("id", 0)
        res["weight"] = p["weight_j"]
        traj_results.append(res)
    print_trajectory_summary(traj_results)

    # ==========================================================
    # 阶段 7: 能量沉积 PDE 求解
    # ==========================================================
    print("\n[阶段 7/14] FTCS 能量沉积 PDE 求解 (1D)...")
    kappa_0 = spitzer_harm_conductivity(
        CORONA_ELECTRON_TEMPERATURE, CORONA_ELECTRON_DENSITY,
        CORONA_ION_CHARGE_Z)
    # 缩放到合理单位
    kappa_0_scaled = kappa_0 * (EV_TO_JOULE ** 2.5) * 1.0e-60

    beam_params = {
        "x0": 0.3 * DOMAIN_LENGTH_X,
        "r_beam": FAST_ELECTRON_BEAM_RADIUS,
        "tau_pulse": FAST_ELECTRON_PULSE_DURATION,
        "total_energy": FAST_ELECTRON_TOTAL_ENERGY * 0.01,  # 小规模
        "t_start": 0.0,
    }

    sim_result = run_deposition_simulation_1d(
        nx=48,
        Lx=DOMAIN_LENGTH_X,
        nt=40,
        T_total=min(TOTAL_TIME, 2.0e-12),
        kappa_0=max(kappa_0_scaled, 1.0e-40),
        beam_params=beam_params,
        fd_order=FD_ORDER,
    )
    print_simulation_summary(sim_result)

    # ==========================================================
    # 阶段 8: von Neumann 稳定性分析
    # ==========================================================
    print("\n[阶段 8/14] von Neumann 稳定性分析...")
    dx = sim_result["dx"]
    kappa_max = kappa_0_scaled * max(sim_result["u_history"][-1]) ** 2.5 \
        if sim_result["u_history"] else 1.0e-30
    kappa_max = max(kappa_max, 1.0e-40)

    stability = full_stability_analysis(
        kappa_max=kappa_max,
        dx=dx,
        dy=dx,
        fd_orders=(2, 4),
        n_modes=32,
        n_random_tests=3,
    )
    print_stability_summary(stability)

    # ==========================================================
    # 阶段 9: 守恒量监测
    # ==========================================================
    print("\n[阶段 9/14] 能量守恒监测...")
    energies = [compute_total_energy_1d(u, sim_result["dx"])
                for u in sim_result["u_history"]]
    conservation = {
        "drift": compute_conservation_drift(energies),
        "M0_history": energies,
        "M1_history": [0.0] * len(energies),
        "M2_history": [0.0] * len(energies),
    }
    # 简单矩计算
    for n_step, u in enumerate(sim_result["u_history"]):
        m1 = sum(sim_result["x"][i] * u[i] for i in range(len(u))) * sim_result["dx"]
        m2 = sum(sim_result["x"][i] ** 2 * u[i] for i in range(len(u))) * sim_result["dx"]
        conservation["M1_history"][n_step] = m1
        conservation["M2_history"][n_step] = m2
    print_conservation_summary(conservation)

    # ==========================================================
    # 阶段 10: Barenblatt 收敛性
    # ==========================================================
    print("\n[阶段 10/14] Barenblatt 精确解收敛性验证...")
    conv_result = run_barenblatt_convergence_test(
        n_levels=config["refine_levels"],
        base_nx=16,
        T_end=0.05,
    )
    print_convergence_summary(conv_result)

    # ==========================================================
    # 阶段 11: 谱工具 (Hilbert + 素数滤波)
    # ==========================================================
    print("\n[阶段 11/14] Hilbert 索引 + 素数谱滤波...")

    # Hilbert 局部性
    small_nx, small_ny = 16, 8
    locality = hilbert_locality_score(small_nx, small_ny)
    print("  Hilbert 局部性评分:")
    print("    行主序得分  : {}".format(locality["row_major_score"]))
    print("    Hilbert 得分: {}".format(locality["hilbert_score"]))
    print("    改善因子    : {:.2f}x".format(locality["improvement"]))

    # 素数滤波 (对能量密度剖面)
    if sim_result["u_history"]:
        signal = sim_result["u_history"][-1][:min(32, len(sim_result["u_history"][-1]))]
        filtered, spectrum, filt_spectrum = prime_mode_filter(signal, keep_prime_modes=True)
        print_spectral_summary(signal, filtered, spectrum, filt_spectrum)

    # ==========================================================
    # 阶段 12: MSD 扩散率
    # ==========================================================
    print("\n[阶段 12/14] MSD 扩散率估计...")
    # 从轨迹中提取 x(t)
    x_trajectories = [r["x"] for r in traj_results if len(r["x"]) > 2]
    if x_trajectories:
        lags, msd_vals = compute_msd_1d(x_trajectories, max_lag=10)
        dt_traj = min(TOTAL_TIME, 2.0e-12) / 50
        diffusion = estimate_diffusion_coefficient(
            lags, msd_vals, dt_traj, dimension=1)
    else:
        diffusion = {"D": 0.0, "slope": 0.0, "r_squared": 0.0}
    print_diagnostics_summary(qc_result, deposit_info, diffusion)

    # ==========================================================
    # 阶段 13: 数值误差分析
    # ==========================================================
    print("\n[阶段 13/14] 数值误差分析...")
    error_results = full_error_analysis()
    print_error_summary(error_results)

    # ==========================================================
    # 阶段 14: 相空间 I/O
    # ==========================================================
    print("\n[阶段 14/14] 相空间 I/O...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    xyz_file = os.path.join(output_dir, "beam_phase_space.xyz")
    write_phase_space_xyz(
        xyz_file,
        beam.particles[:min(32, len(beam.particles))],
        comment="Fast ignition electron beam, t=0"
    )
    # 读回验证
    read_back, comment = read_phase_space_xyz(xxyz_file := xyz_file)
    print_io_summary(xyz_file,
                     n_written=min(32, len(beam.particles)),
                     n_read=len(read_back))
    # 清理临时文件
    try:
        if os.path.exists(xyz_file):
            os.remove(xyz_file)
    except Exception:
        pass

    # ==========================================================
    # 最终汇总
    # ==========================================================
    elapsed = timer.time() - start_time

    print("\n" + "=" * 72)
    print(" 计算完成!")
    print("=" * 72)
    print("  总计算时间: {:.3f} s".format(elapsed))
    print("  物理问题  : Fast Ignition ICF 电子束能量沉积")
    print("  数值方法  : 高阶有限差分 (Vandermonde 系数)")
    print("  时间积分  : RK4 + FTCS")
    print("  稳定性    : Von Neumann 分析")
    print("  收敛验证  : Barenblatt 精确解")
    print("  种子项目  : 15 个项目全部融入")
    print("")
    print("  模块清单 (13 个):")
    print("    1. plasma_parameters     (物理常数)")
    print("    2. electron_beam_source  (Box-Muller + 素数种子)")
    print("    3. vandermonde_fd        (高阶 FD 系数)")
    print("    4. plasma_mesh           (三角形网格)")
    print("    5. energy_deposition_ftcs(FTCS PDE 求解)")
    print("    6. electron_trajectory_rk4 (RK4 轨迹)")
    print("    7. von_neumann_stability (稳定性分析)")
    print("    8. conservation_monitor  (守恒量监测)")
    print("    9. convergence_benchmark (收敛性验证)")
    print("   10. spectral_tools        (Hilbert + 素数滤波)")
    print("   11. plasma_diagnostics    (QC + 沉积 + MSD)")
    print("   12. phase_space_io        (XYZ I/O)")
    print("   13. error_analysis        (误差分析)")
    print("=" * 72)


if __name__ == "__main__":
    main()
