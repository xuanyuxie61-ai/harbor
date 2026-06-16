"""
main.py
=======

统一入口：固态电解质界面（SEI）高阶有限差分反应-扩散-电迁移耦合建模
与稳定性分析（小规模可复现实验）。

零参数运行方式：
    python main.py

本项目融合 15 个种子项目算法，面向计算材料领域前沿问题：
    - 电极/电解质界面 SEI 层的反应-扩散-力学耦合
    - 高阶有限差分空间离散（2阶/4阶/Gram 多项式重构）
    - 显式时间推进 + von Neumann 稳定性分析
    - Butler-Volmer 电化学动力学
    - SEI 形貌不稳定性的 Lorenz 降维分析
    - 蒙特卡罗随机成核与化学计量学
    - Gauss-Laguerre/Hermite 求积（Boltzmann 权重积分）
    - PCA 本征模分解（浓度场模式提取）
    - 无量纲不变量检测（类比引力波 ringdown）
    - 实验时间戳与循环调度

融合种子项目：
    360_fd1d_heat_explicit, 125_burgers_steady_viscous, 1402_wave_pde,
    479_gram_polynomial, 092_bioconvection_ode, 1256_AXVIAM_gw-ringdown-invariant,
    326_eigenfaces, 348_fair_dice_simulation, 323_e_spigot, 899_polyomino_parity,
    682_line_lines_packing, 467_gen_laguerre_rule, 519_hermite_exactness,
    1333_triangulation_boundary_nodes, 1412_weekday_zeller

作者: DA-Synthesis
"""

import sys
import os

# 确保可以导入同级模块（既支持作为脚本运行，也支持作为包运行）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


# ============================================================
#  导入所有模块
# ============================================================

import sei_parameters as P
import grid_mesh
import high_order_fd
import butler_volmer_kinetics as bv
import sei_diffusion_solver
import stability_analysis
import invariant_detector
import eigenmode_decomposition
import stochastic_nucleation
import quadrature_integration
import sei_lorenz_dynamics
import timestamp_utils


# ============================================================
#  分隔线辅助函数
# ============================================================

def section_header(title, char="=", width=72):
    """打印带装饰的节标题。"""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


# ============================================================
#  主流程
# ============================================================

def main():
    """
    统一入口：按序执行 SEI 建模的完整仿真流程。
    """
    print("=" * 72)
    print("  固态电解质界面反应建模 — 高阶有限差分与稳定性分析")
    print("  Solid Electrolyte Interphase (SEI) High-Order FD Modeling")
    print("=" * 72)
    print()

    # ----------------------------------------------------------
    #  第 1 阶段：参数校验与实验时间戳（1412_weekday_zeller）
    # ----------------------------------------------------------
    section_header("阶段 1：参数校验与实验时间戳")
    params_ok = P.validate_parameters()
    print(f"  参数自洽性校验: {'通过' if params_ok else '失败'}")

    ts_result = timestamp_utils.run_timestamp_demo()
    ts = ts_result['experiment_timestamp']
    print(f"  实验日期: {ts['year']}-{ts['month']:02d}-{ts['day']:02d} "
          f"({ts['weekday_name']})")
    print(f"  时间戳字符串: {ts['timestamp_string']}")
    print(f"  时间戳哈希（用于 RNG 种子）: {ts['timestamp_hash']}")

    # ----------------------------------------------------------
    #  第 2 阶段：网格构造与边界识别
    #            （1333_triangulation_boundary_nodes +
    #             682_line_lines_packing）
    # ----------------------------------------------------------
    section_header("阶段 2：SEI 网格构造与纳米孔隙密堆积")
    grid_result = grid_mesh.run_grid_construction_demo()
    print(f"  节点数 N = {P.N_GRID}, 域长 L = {P.L_DOMAIN * 1e9:.1f} nm")
    print(f"  空间步长 dx = {grid_result['dx'] * 1e9:.4f} nm")
    print(f"  边界: {grid_result['bc_type'][0]} | {grid_result['bc_type'][-1]}")
    pi = grid_result['packing_info']
    print(f"  Rényi 密堆积: {pi['n_pores']} 个孔隙, "
          f"密度 = {pi['density']:.4f} "
          f"(理论极限 ≈ {pi['renyi_theoretical']:.4f})")

    # ----------------------------------------------------------
    #  第 3 阶段：高精度基本常数（323_e_spigot）
    # ----------------------------------------------------------
    section_header("阶段 3：高精度基本常数")
    consts = stochastic_nucleation.compute_fundamental_constants()
    print(f"  元电荷 e  = {consts['e_charge']:.6e} C")
    print(f"  Avogadro  = {consts['N_Avogadro']:.6e} /mol")
    print(f"  Faraday F = {consts['Faraday']:.6e} C/mol")
    print(f"  气体常数 R= {consts['R_gas']:.6e} J/(mol K)")
    print(f"  e 的前 20 位: {consts['e_digits_20'][:22]}...")

    # ----------------------------------------------------------
    #  第 4 阶段：高阶有限差分验证（479_gram_polynomial）
    # ----------------------------------------------------------
    section_header("阶段 4：高阶差分算子验证（含 Gram 重构）")
    x = grid_result['x']
    dx = grid_result['dx']
    L = P.L_DOMAIN
    k_wave = 2.0 * 3.14159265358979 / L
    u_test = [__import__('math').sin(k_wave * xi) for xi in x]
    lu_exact = [-(k_wave ** 2) * __import__('math').sin(k_wave * xi)
                for xi in x]
    lu_2 = high_order_fd.laplacian_2nd_order(u_test, dx)
    lu_4 = high_order_fd.laplacian_4th_order(u_test, dx)
    lu_gram = high_order_fd.reconstruct_with_gram(u_test, dx, P.GRAM_ORDER)
    err_2 = max(abs(lu_2[i] - lu_exact[i]) for i in range(P.N_GRID))
    err_4 = max(abs(lu_4[i] - lu_exact[i]) for i in range(P.N_GRID))
    err_gram = max(abs(lu_gram[i] - lu_exact[i]) for i in range(P.N_GRID))
    print(f"  f(x) = sin(2πx/L) 二阶导数误差:")
    print(f"    二阶精度 O(dx²) : {err_2:.6e}")
    print(f"    四阶精度 O(dx⁴) : {err_4:.6e}")
    print(f"    Gram 重构       : {err_gram:.6e}")
    if err_4 > 1.0e-30:
        print(f"  四阶精度提升比: {err_2 / err_4:.1f}")

    # ----------------------------------------------------------
    #  第 5 阶段：稳定性分析（360_fd1d + 125_burgers）
    # ----------------------------------------------------------
    section_header("阶段 5：von Neumann 稳定性分析")
    stab_report = stability_analysis.generate_stability_report()
    print(f"  实际 CFL 数: {stab_report['cfl_actual']:.4f}")
    print(f"  二阶临界 CFL: {stab_report['cfl_crit_2nd']:.4f}")
    print(f"  四阶临界 CFL: {stab_report['cfl_crit_4th']:.4f}")
    print(f"  二阶最大 dt : {stab_report['dt_max_2nd']:.6e} s")
    print(f"  四阶最大 dt : {stab_report['dt_max_4th']:.6e} s")
    print(f"\n  von Neumann 扫描（二阶格式）:")
    for r in stab_report['von_neumann_2nd']:
        status = "✓ 稳定" if r['stable'] else "✗ 不稳定"
        print(f"    CFL = {r['cfl']:.3f}: "
              f"|G|_max = {r['max_abs_G']:.6f} [{status}]")
    rd = stab_report['reaction_diffusion']
    print(f"\n  反应-扩散耦合:")
    print(f"    Damköhler 数 Da = {rd['damkohler']:.6e}")
    print(f"    控制因素: {rd['controlling']}")

    # ----------------------------------------------------------
    #  第 6 阶段：Butler-Volmer 稳态 Newton 求解
    #            （125_burgers_steady_viscous）
    # ----------------------------------------------------------
    section_header("阶段 6：Butler-Volmer 动力学与 Newton 稳态求解")
    j_test = bv.butler_volmer_current(
        0.05, 1.0, P.ALPHA_A_GRAPHITE, P.ALPHA_C_GRAPHITE
    )
    dj_test = bv.butler_volmer_derivative(
        0.05, 1.0, P.ALPHA_A_GRAPHITE, P.ALPHA_C_GRAPHITE
    )
    print(f"  η = 50 mV 时:")
    print(f"    j  = {j_test:.6e} A/m^2")
    print(f"    dj = {dj_test:.6e} A/(m^2 V)")

    u_sol, n_step, conv, res_hist = bv.solve_steady_state_np(
        alpha_left=0.1, beta_right=0.0, nu_eff=1.0,
        n_nodes=21, dx=0.05
    )
    print(f"\n  Newton 稳态 NP 方程求解:")
    print(f"    收敛: {conv}, 步数: {n_step}")
    if res_hist:
        print(f"    初始残差: {res_hist[0]:.6e}")
        print(f"    最终残差: {res_hist[-1]:.6e}")

    # Newton 收敛性分析
    conv_analysis = stability_analysis.newton_convergence_analysis(res_hist)
    print(f"    单调递减: {conv_analysis['monotone']}")
    print(f"    二次收敛: {conv_analysis['quadratic']}")

    # ----------------------------------------------------------
    #  第 7 阶段：SEI Li+ 扩散-反应时间推进
    #            （360_fd1d_heat_explicit + 1402_wave_pde）
    # ----------------------------------------------------------
    section_header("阶段 7：SEI Li⁺ 扩散-反应耦合时间推进")
    diff_result = sei_diffusion_solver.run_diffusion_simulation(
        n_steps=100, fd_scheme="2nd_order", coupled_wave=True
    )
    print(f"  时间步数: 100")
    print(f"  最终 Li+ 浓度范围: "
          f"[{min(diff_result['c_final']):.4f}, "
          f"{max(diff_result['c_final']):.4f}] mol/m³")
    print(f"  最终 CFL: {diff_result['cfl_history'][-1]:.4f}")
    print(f"  初始总质量: {diff_result['mass_history'][0]:.6e}")
    print(f"  最终总质量: {diff_result['mass_history'][-1]:.6e}")
    mass_conservation = (abs(diff_result['mass_history'][-1]
                             - diff_result['mass_history'][0])
                         / max(diff_result['mass_history'][0], 1.0e-30))
    print(f"  质量守恒偏差: {mass_conservation:.4f}")

    # ----------------------------------------------------------
    #  第 8 阶段：SEI 无量纲不变量检测（1256_AXVIAM）
    # ----------------------------------------------------------
    section_header("阶段 8：SEI 演化无量纲不变量检测")
    inv_result = invariant_detector.run_invariant_analysis_demo()
    print(f"  特征时间 τ_char = {inv_result['tau_characteristic']:.6e} s")
    print(f"  Plateau 起始: {inv_result['plateau_start']}")
    print(f"  Plateau 值 Ξ = {inv_result['plateau_value']:.6e} s")
    print(f"  无量纲不变量 Ξ̃ = {inv_result['xi_tilde']:.6e}")

    # ----------------------------------------------------------
    #  第 9 阶段：PCA 本征模分解（326_eigenfaces）
    # ----------------------------------------------------------
    section_header("阶段 9：SEI 浓度场 PCA 本征模分解")
    pca_result = eigenmode_decomposition.run_pca_demo(n_modes=3)
    print(f"  快照数: {pca_result['n_snapshots']}")
    for i, (lam, frac) in enumerate(
            zip(pca_result['eigenvalues'], pca_result['energy_fractions'])):
        print(f"  模态 {i}: λ = {lam:.6e}, 能量占比 η = {frac:.4f}")

    # ----------------------------------------------------------
    #  第 10 阶段：Lorenz 形貌不稳定性（092_bioconvection）
    # ----------------------------------------------------------
    section_header("阶段 10：SEI 形貌不稳定性 Lorenz 降维")
    lorenz_result = sei_lorenz_dynamics.run_lorenz_trajectory(n_steps=300)
    print(f"  初始状态: [1.000, 1.000, 1.000]")
    print(f"  最终状态: [{lorenz_result['final_state'][0]:.4f}, "
          f"{lorenz_result['final_state'][1]:.4f}, "
          f"{lorenz_result['final_state'][2]:.4f}]")
    print(f"\n  不动点分析:")
    for fp_info in lorenz_result['fixed_points']:
        fp = fp_info['point']
        cls = fp_info['classification']
        print(f"    ({fp[0]:7.3f}, {fp[1]:7.3f}, {fp[2]:7.3f}) "
              f"-> trace={cls['trace']:.3f}, {cls['stability']}")

    # ----------------------------------------------------------
    #  第 11 阶段：蒙特卡罗成核 + 化学计量学（348 + 899 + 323）
    # ----------------------------------------------------------
    section_header("阶段 11：SEI 随机成核与化学计量学")
    mc_result = stochastic_nucleation.monte_carlo_nucleation(
        n_sites=P.N_NUCLEATION_SITES,
        n_steps=min(200, P.N_STEPS),
        energy_barrier=P.NUCLEATION_ENERGY_BARRIER,
        rng_seed=P.RNG_SEED
    )
    print(f"  成核概率/步: {mc_result['p_nucleation_per_step']:.6e}")
    print(f"  总成核数: {mc_result['n_total_nucleated']}"
          f"/{mc_result['n_sites']}")
    print(f"  成核比例: {mc_result['fraction_nucleated']:.4f}")
    print(f"  平均成核时间: {mc_result['mean_nucleation_time']:.6e} s")

    stoi_result = stochastic_nucleation.balance_sei_reaction()
    print(f"\n  化学计量不定方程:")
    print(f"    系数: {stoi_result['coefficients']}")
    print(f"    右侧: {stoi_result['rhs']}")
    print(f"    非负整数解数: {stoi_result['n_solutions']}")
    if stoi_result['solutions']:
        print(f"    前 5 个解: {stoi_result['solutions'][:5]}")

    # ----------------------------------------------------------
    #  第 12 阶段：Gauss 求积（467 + 519）
    # ----------------------------------------------------------
    section_header("阶段 12：Gauss 求积（Laguerre + Hermite）")
    quad_result = quadrature_integration.run_quadrature_demo()
    print(f"  Gauss-Laguerre 求积节点 (α={P.GL_ALPHA_PARAM}, n={P.GL_ORDER}):")
    for i, (xi, wi) in enumerate(
            zip(quad_result['laguerre_nodes'],
                quad_result['laguerre_weights'])):
        print(f"    x_{i} = {xi:8.5f}, w_{i} = {wi:.6e}")
    print(f"\n  Gauss-Hermite 精确性检验 (n={P.GH_ORDER}):")
    hermite_test = quad_result['hermite_exactness']
    n_exact = sum(1 for r in hermite_test if r['exact_within_tolerance'])
    print(f"    精确到 1e-8 的次数: {n_exact}/{len(hermite_test)}")
    print(f"  Boltzmann 平均反应速率 = {quad_result['boltzmann_avg_rate']:.6e}")

    # ----------------------------------------------------------
    #  完成总结
    # ----------------------------------------------------------
    section_header("合成项目运行完成", char="*")
    print("  全部 12 个阶段执行成功。")
    print()
    print("  项目结构（12 个 .py 模块）:")
    print("    sei_parameters.py           — 物理/化学参数")
    print("    grid_mesh.py                — 网格 + Rényi 密堆积")
    print("    high_order_fd.py            — 2/4 阶差分 + Gram 重构")
    print("    butler_volmer_kinetics.py   — Butler-Volmer + Newton")
    print("    sei_diffusion_solver.py     — 扩散-反应时间推进")
    print("    stability_analysis.py       — von Neumann 稳定性")
    print("    invariant_detector.py       — 无量纲不变量检测")
    print("    eigenmode_decomposition.py  — PCA 本征模分解")
    print("    stochastic_nucleation.py    — 蒙特卡罗成核 + 不定方程")
    print("    quadrature_integration.py   — Gauss 求积")
    print("    sei_lorenz_dynamics.py      — Lorenz 形貌降维")
    print("    timestamp_utils.py          — 时间戳 + Zeller 日历")
    print()
    print("  科学意义:")
    print("    本项目为锂离子负极 SEI 层的生长演化提供完整的高阶数值")
    print("    仿真框架，耦合扩散、反应、电迁移、力学效应，并辅以")
    print("    严格的稳定性分析、本征模分解、随机成核与不变量检测。")
    print("=" * 72)


if __name__ == "__main__":
    main()
