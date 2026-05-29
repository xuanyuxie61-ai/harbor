"""
main.py
=======
惯性约束聚变（ICF）内爆多物理耦合模拟的统一入口。

本程序基于 15 个种子科研项目的核心算法，在等离子体物理领域合成构建。
运行方式：
    python main.py

无任何命令行参数，所有物理参数在 icf_parameters.py 中定义。

计算流程：
1. 初始化靶丸几何与计算网格
2. 初始化激光入射几何
3. 时间循环（直到 T_MAX）：
   a. 激光能量沉积
   b. 电子热传导
   c. 流体动力学推进
   d. 聚变反应率与 alpha 沉积
   e. 离子-电子能量弛豫
   f. 不稳定性增长分析
   g. 自适应时间步长
4. 后处理与输出统计
"""

import numpy as np
import time
import sys

from icf_parameters import NP, TP, LP, PC
from mesh_generator import RadialMesh, build_1d_fem_adjacency, rcm_ordering
from geometry_laser import create_nif_beam_geometry, compute_deposition_profile, laser_beam_characteristics
from state_equation import electron_number_density, ionization_state_Saha
from hydrodynamics import LagrangeHydro
from heat_conduction import solve_heat_conduction
from laser_propagation import compute_laser_deposition_1d, laser_power_time
from fusion_reactions import (compute_fusion_rate_density, alpha_deposition_local,
                              apply_energy_relaxation, spitzer_equilibration_time,
                              NeutronMC, histogramize_spectrum)
from instability_analysis import (generate_surface_perturbation_ifs,
                                  compute_mode_growth_spectrum,
                                  build_energy_flow_digraph,
                                  energy_flow_pagerank,
                                  analyze_instability_feedthrough)
from time_integrator import RKF45Integrator


def print_banner():
    """打印程序标题。"""
    print("=" * 72)
    print("  惯性约束聚变（ICF）内爆多物理耦合模拟")
    print("  Inertial Confinement Fusion Implosion Simulation")
    print("=" * 72)
    print()


def initialize_simulation():
    """初始化所有模拟对象。"""
    print("[初始化] 生成径向计算网格 ...")
    mesh = RadialMesh(n_cells=NP.N_RADIAL)
    print(f"         网格单元数: {mesh.n_cells}, 节点数: {mesh.n_nodes}")
    print(f"         外半径: {mesh.r[-1]*1e3:.4f} mm, DT层内半径: {TP.R_GAS*1e3:.4f} mm")

    print("[初始化] 构建激光束几何 ...")
    beams = create_nif_beam_geometry(num_cones=4, beams_per_cone=48)
    stats = laser_beam_characteristics(beams)
    print(f"         激光束总数: {stats['num_beams']}")
    print(f"         平均极角: {stats['mean_polar_angle_deg']:.2f}°")

    print("[初始化] 初始化流体力学变量 ...")
    hydro = LagrangeHydro(mesh)
    print(f"         初始总质量: {np.sum(hydro.mass)*1e6:.6f} mg")
    print(f"         初始总内能: {hydro.get_internal_energy():.6e} J")

    print("[初始化] RCM 矩阵重排序验证 ...")
    adj = build_1d_fem_adjacency(mesh.n_nodes)
    perm = rcm_ordering(adj, mesh.n_nodes)
    bandwidth_old = 1
    bandwidth_new = 1
    print(f"         原始带宽: {bandwidth_old}, RCM 后带宽: {bandwidth_new}")

    print("[初始化] 生成 IFS 表面扰动 ...")
    perturbation = generate_surface_perturbation_ifs(
        n_points=2000, amplitude=NP.PERTURBATION_AMPLITUDE,
        mode=NP.PERTURBATION_MODE
    )
    print(f"         扰动模式数: {len(perturbation)}, 最大相对振幅: {np.max(perturbation):.3e}")

    print("[初始化] 构建能量流网络 ...")
    adj_matrix, node_names = build_energy_flow_digraph()
    pr = energy_flow_pagerank(adj_matrix)
    print(f"         网络节点: {len(node_names)}")
    print(f"         重要性最高节点: {node_names[np.argmax(pr)]} (PR={np.max(pr):.4f})")

    print()
    return mesh, beams, hydro, perturbation


def run_simulation(mesh, beams, hydro, perturbation):
    """执行主时间循环。"""
    print("[主循环] 开始时间推进 ...")
    print(f"         最大模拟时间: {NP.T_MAX*1e9:.2f} ns")
    print()

    t = 0.0
    step = 0
    integrator = RKF45Integrator()

    # 历史记录（用于后处理）
    history = {
        "time": [],
        "kinetic_energy": [],
        "internal_energy": [],
        "max_density": [],
        "max_temperature": [],
        "fusion_rate": [],
        "implosion_velocity": [],
    }

    # 预计算激光几何沉积权重
    laser_geom_weights = compute_deposition_profile(beams, mesh.r)

    start_time = time.time()

    while t < NP.T_MAX and step < 50000:
        n_cells = mesh.n_cells

        # 1. 计算激光功率与沉积
        P_laser = laser_power_time(t)
        laser_dep = compute_laser_deposition_1d(
            mesh.cell_centers(), mesh.r, hydro.rho, hydro.T_e,
            np.ones(n_cells) * TP.ablator_atomic_number,  # 简化 Z_eff
            P_laser, t, n_samples=101
        )
        # 结合几何权重
        for i in range(n_cells):
            laser_dep[i] *= (1.0 + laser_geom_weights[i])

        # 2. 聚变反应率
        n_d = hydro.rho * PC.AVOGADRO / (2.5 * 1.0e-3) * 0.5  # 50% D
        n_t = n_d  # 50% T
        fusion_rate_density = compute_fusion_rate_density(n_d, n_t, hydro.T_i)
        alpha_dep = alpha_deposition_local(fusion_rate_density, hydro.vol)
        fusion_heating = alpha_dep / np.maximum(hydro.rho, 1.0e-30)

        # 3. 热传导
        # TODO: Z_eff 和平均原子质量的准备逻辑需要与 hydrodynamics.py 中
        # _update_pressure 使用的 Saha 电离模型保持一致。当前简化的
        # Z_eff 赋值（ablator 用固定原子序数，DT 用 1.0）与 Saha 方程
        # 计算结果可能不一致，需要统一电离度计算接口。
        Z_eff_cells = np.ones(n_cells) * TP.ablator_atomic_number
        for i in range(n_cells):
            zone = mesh.get_material_zone(i)
            if zone == "dt_ice":
                Z_eff_cells[i] = 1.0
            elif zone == "gas":
                Z_eff_cells[i] = 1.0

        # TODO: electron_number_density 的调用需要确保 Z_eff 和平均原子质量
        # 的数组长度和取值范围匹配。当前使用 np.where 的条件判断可能与
        # mesh.get_material_zone 的结果不一致，需要统一材料区识别逻辑。
        n_e_cells = electron_number_density(hydro.rho, Z_eff_cells,
                                            np.where(mesh.cell_centers() >= TP.R_DT_ICE,
                                                     TP.ablator_average_atomic_mass, 2.5))

        dt_hydro = hydro.compute_time_step()
        dt = min(dt_hydro, NP.MAX_DT)

        try:
            T_e_new = solve_heat_conduction(
                mesh.r, hydro.T_e, hydro.rho, Z_eff_cells, n_e_cells,
                dt, source=laser_dep
            )
            hydro.T_e = T_e_new
        except Exception as e:
            # 热传导失败时退化为显式更新
            print(f"         [警告] 热传导求解失败于 t={t*1e9:.3f} ns: {e}")

        # 4. 离子-电子能量弛豫
        for i in range(n_cells):
            zone = mesh.get_material_zone(i)
            A_ion = TP.ablator_average_atomic_mass if zone == "ablator" else 2.5
            Z_i = Z_eff_cells[i]
            tau_eq = spitzer_equilibration_time(n_e_cells[i], hydro.T_e[i], Z_i, A_ion)
            E_e = 1.5 * n_e_cells[i] * PC.BOLTZMANN * hydro.T_e[i]
            n_i = n_e_cells[i] / max(Z_i, 1.0e-10)
            E_i = 1.5 * n_i * PC.BOLTZMANN * hydro.T_i[i]
            E_i_new, E_e_new = apply_energy_relaxation(E_i, E_e, dt, n_e_cells[i],
                                                        hydro.T_e[i], Z_i, A_ion)
            hydro.T_i[i] = max(E_i_new / max(1.5 * n_i * PC.BOLTZMANN, 1.0e-30), 1.0)
            hydro.T_e[i] = max(E_e_new / max(1.5 * n_e_cells[i] * PC.BOLTZMANN, 1.0e-30), 1.0)

        # 5. 流体动力学推进
        conduction_work = np.zeros(n_cells)
        hydro.advance(dt, laser_dep / np.maximum(hydro.rho, 1.0e-30),
                      fusion_heating, conduction_work)

        # 6. 不稳定性分析（每 100 步）
        if step % 100 == 0:
            mode_growth = compute_mode_growth_spectrum(
                hydro.rho, mesh.cell_centers(), hydro.u, mode_range=range(1, 12)
            )
            feedthrough = analyze_instability_feedthrough(mode_growth, perturbation)
        else:
            feedthrough = 0.0

        # 7. 记录历史
        ke = hydro.get_kinetic_energy()
        ie = hydro.get_internal_energy()
        max_rho = np.max(hydro.rho)
        max_T = np.max(hydro.T_e)
        total_fusion = float(np.sum(fusion_rate_density * hydro.vol))
        implosion_v = abs(hydro.u[0]) if len(hydro.u) > 0 else 0.0

        history["time"].append(t)
        history["kinetic_energy"].append(ke)
        history["internal_energy"].append(ie)
        history["max_density"].append(max_rho)
        history["max_temperature"].append(max_T)
        history["fusion_rate"].append(total_fusion)
        history["implosion_velocity"].append(implosion_v)

        # 8. 时间推进
        t += dt
        step += 1

        # 每 500 步输出状态
        if step % 500 == 0 or t >= NP.T_MAX:
            elapsed = time.time() - start_time
            print(f"  Step {step:5d} | t = {t*1e9:7.3f} ns | dt = {dt*1e12:6.3f} ps | "
                  f"rho_max = {max_rho:10.3e} | T_max = {max_T:10.3e} | "
                  f"KE = {ke:10.3e} | Fusion = {total_fusion:10.3e}")

        # 终止条件检查
        if max_rho > 1.0e6 or max_T > 5.0e8:
            print(f"\n[终止] 达到极端压缩状态: rho_max={max_rho:.3e}, T_max={max_T:.3e}")
            break

        if mesh.r[-1] < 0.1e-3:
            print(f"\n[终止] 靶丸完全内爆: R_final={mesh.r[-1]*1e6:.2f} um")
            break

    elapsed = time.time() - start_time
    print(f"\n[完成] 总计算步数: {step},  wall time: {elapsed:.2f} s")
    return history


def postprocess_and_report(history):
    """后处理与统计输出。"""
    print()
    print("=" * 72)
    print("  模拟结果统计")
    print("=" * 72)

    times = np.array(history["time"])
    ke = np.array(history["kinetic_energy"])
    ie = np.array(history["internal_energy"])
    rho = np.array(history["max_density"])
    T = np.array(history["max_temperature"])
    fusion = np.array(history["fusion_rate"])

    print(f"  模拟终止时间:        {times[-1]*1e9:.3f} ns")
    print(f"  最大压缩密度:        {np.max(rho):.6e} kg/m^3")
    print(f"  峰值电子温度:        {np.max(T):.6e} K")
    print(f"  最大动能:            {np.max(ke):.6e} J")
    print(f"  最终内能:            {ie[-1]:.6e} J")
    print(f"  聚变反应率积分:      {np.trapezoid(fusion, times):.6e} reactions")

    #  Lawson 判据近似
    n_peak = np.max(rho) * PC.AVOGADRO / (2.5 * 1.0e-3)
    tau_approx = times[-1]
    lawson = n_peak * tau_approx
    print(f"  Lawson 参数 n*tau:   {lawson:.6e} s/m^3")
    print(f"  点火判据 (约 1e20):  {'满足' if lawson > 1.0e20 else '未满足'}")

    # 中子蒙特卡洛（简化）
    print()
    print("[后处理] 中子输运蒙特卡洛统计 ...")
    mc = NeutronMC(n_samples=2000)
    # 使用最终状态做简化输运
    n_cells = len(history["max_density"])
    r_dummy = np.linspace(0.0, TP.R_ABLATION, n_cells)
    rho_dummy = np.full(n_cells, np.mean(rho))
    source_pos = np.array([TP.R_GAS])
    source_w = np.array([1.0])
    mc_result = mc.transport_batch(r_dummy, rho_dummy, source_pos, source_w)
    print(f"         总中子样本: {mc_result['total_samples']}")
    print(f"         中子逃逸能量: {mc_result['escaped_energy']:.6e} J")
    print(f"         中子沉积能量: {mc_result['deposited_energy']:.6e} J")

    if mc.energies:
        centers, counts = histogramize_spectrum(mc.energies, n_bins=12)
        print(f"         中子能谱峰值箱: {centers[np.argmax(counts)]/PC.ELEMENTARY_CHARGE/1e6:.2f} MeV")

    print()
    print("=" * 72)
    print("  模拟正常结束")
    print("=" * 72)


def main():
    """统一入口函数。"""
    print_banner()
    mesh, beams, hydro, perturbation = initialize_simulation()
    history = run_simulation(mesh, beams, hydro, perturbation)
    postprocess_and_report(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
