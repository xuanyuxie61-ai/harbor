"""
main.py
=======
格点 QCD 强子谱学博士级合成项目入口。

运行方式：
    python main.py

无需任何参数，程序自动执行完整的强子谱学计算流程：
1. 生成热化规范场构型
2. Wilson 梯度流平滑
3. Wilson-Dirac 传播子求解
4. 介子/重子关联函数构造
5. 变分法提取质量
6. 高阶积分计算衰变常数与自能
7. 手征动力学与 RG 流分析
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lattice_gauge import Lattice, GaugeConfig, ifs_thermalize_gauge
from gauge_dg_update import dg_gauge_evolve
from wilson_flow import wilson_flow_run
from fermion_solver import WilsonDiracOperator, point_source, solve_all_propagators
from correlator_builder import (
    meson_correlator_pion, baryon_correlator_nucleon,
    correlator_effective_mass, lagrange_interpolate
)
from variational_spectrum import (
    variational_masses, optimize_smearing_parameter,
    calccf, spline_eval
)
from quadrature_physics import (
    decay_constant_integral, self_energy_integral
)
from chiral_dynamics import (
    solve_chiral_oscillator, QuarkMesonReactionNetwork,
    chiral_condensate_from_reaction, pion_decay_constant_from_dynamics
)
from reaction_rg import (
    solve_rg_flow, alpha_s_running, lattice_coupling_from_beta,
    coupled_rg_reaction_network
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)

    # ============================================================
    # 0. 格点参数设置（使用较小格点以保证运行速度）
    # ============================================================
    print_section("格点 QCD 强子谱学：博士级合成计算")
    print("科学领域：粒子物理 — 格点 QCD 强子谱学")
    print("模拟参数：4³ × 8 格点，SU(2) 规范群（简化示范）")
    print("物理公式：Wilson plaquette action, Wilson-Dirac operator,")
    print("          GEVP variational method, GMOR relation, RG flow")

    nx, ny, nz, nt = 4, 4, 4, 8
    lat = Lattice(nx, ny, nz, nt)
    beta_lat = 2.4
    mass_quark = 0.1
    n_sources = 2
    flow_time = 0.3

    # ============================================================
    # 1. 规范场生成与热化
    # ============================================================
    print_section("1. 规范场构型生成与 IFS 热化")
    gauge = GaugeConfig(lat)
    gauge.randomize()
    print(f"初始平均 plaquette: {gauge.average_plaquette():.6f}")
    print("执行 IFS 伪热化 (50 iter)...")
    ifs_thermalize_gauge(gauge, n_iter=50)
    print(f"热化后平均 plaquette: {gauge.average_plaquette():.6f}")

    # ============================================================
    # 2. DG 谱方法规范场演化（冷却预处理）
    # ============================================================
    print_section("2. DG 谱方法规范场 fictitious-time 演化")
    print("使用间断 Galerkin 方法 + LSERK 时间推进")
    gauge = dg_gauge_evolve(gauge, beta=beta_lat, final_time=0.2, cfl=0.3)
    print(f"DG 演化后平均 plaquette: {gauge.average_plaquette():.6f}")

    # ============================================================
    # 3. Wilson 梯度流平滑
    # ============================================================
    print_section("3. Wilson 梯度流平滑（自适应隐式中点法）")
    print(f"目标流时间: {flow_time}")
    gauge = wilson_flow_run(gauge, flow_time=flow_time, dt_init=0.05, method="midpoint")
    print(f"梯度流后平均 plaquette: {gauge.average_plaquette():.6f}")
    print(f"Wilson 作用量: {gauge.wilson_action(beta_lat):.4f}")

    # ============================================================
    # 4. Wilson-Dirac 传播子求解
    # ============================================================
    print_section("4. Wilson-Dirac 传播子 CG 求解")
    print(f"夸克质量 m_q = {mass_quark}, κ = {1.0 / (2.0 * mass_quark + 8.0):.6f}")
    wd = WilsonDiracOperator(lat, gauge, mass=mass_quark)

    sources = []
    source_positions = []
    for i in range(n_sources):
        x0 = np.array([i % nx, (i * 2) % ny, (i * 3) % nz, 0])
        src = point_source(lat, x0, spin=0)
        sources.append(src)
        source_positions.append(x0)
    print(f"生成 {n_sources} 个点源，开始 CG 求解 (max_iter=150)...")
    propagators = solve_all_propagators(wd, sources)
    print("传播子求解完成。")

    # ============================================================
    # 5. 关联函数构造
    # ============================================================
    print_section("5. 强子关联函数构造")
    print("计算 π 介子关联函数（赝标量通道）...")
    corr_pion = meson_correlator_pion(lat, propagators, source_positions)
    print(f"π 关联函数: {np.round(corr_pion[:5], 6)}")

    print("计算核子关联函数（重子通道，含孤子波包增强）...")
    corr_nucleon = baryon_correlator_nucleon(lat, propagators, soliton_enhance=True)
    print(f"N 关联函数: {np.round(corr_nucleon[:5], 6)}")

    # 有效质量
    m_eff_pion = correlator_effective_mass(corr_pion, dt=1)
    m_eff_nucleon = correlator_effective_mass(corr_nucleon, dt=1)
    # 取中间 plateau 值
    plateau_slice = slice(nt // 4, 3 * nt // 4)
    mpi_est = np.nanmedian(m_eff_pion[plateau_slice])
    mn_est = np.nanmedian(m_eff_nucleon[plateau_slice])
    print(f"π 有效质量估计: {mpi_est:.6f} (格点单位)")
    print(f"N 有效质量估计: {mn_est:.6f} (格点单位)")

    # ============================================================
    # 6. 变分法能谱提取
    # ============================================================
    print_section("6. 变分广义本征值分析 (GEVP)")
    nop = 3
    corr_matrix = np.zeros((nt, nop, nop))
    for t in range(nt):
        for i in range(nop):
            for j in range(nop):
                smear_i = 0.5 + i * 0.3
                smear_j = 0.5 + j * 0.3
                corr_matrix[t, i, j] = corr_pion[t] * np.exp(-smear_i * smear_j * t / nt)

    var_results = variational_masses(corr_matrix, t0=2)
    masses_var = var_results["masses"]
    print("变分法提取的前 3 个能级（有效质量 plateau 中值）:")
    for n in range(nop):
        m_est = np.nanmedian(masses_var[:, n])
        print(f"  能级 {n}: m_{n} = {m_est:.6f}")

    # Hooke-Jeeves 优化 smearing 参数
    print("\n使用 Hooke-Jeeves 直接搜索优化 smearing 参数...")
    def corr_func(alpha):
        return np.array([corr_pion[t] * np.exp(-alpha * t / nt) for t in range(nt)])
    alpha_opt, obj_val = optimize_smearing_parameter(corr_func, (0.1, 3.0), t0=2)
    print(f"最优 smearing 参数: α = {alpha_opt:.4f}, 目标函数值 = {obj_val:.6e}")

    # ============================================================
    # 7. 样条插值平滑
    # ============================================================
    print_section("7. 关联函数样条插值平滑")
    t_nodes = np.arange(0, nt, 2, dtype=float)
    if len(t_nodes) < 4:
        t_nodes = np.arange(0, nt, 1, dtype=float)
    c_vals = np.zeros((2, len(t_nodes)))
    c_vals[0, :] = corr_pion[t_nodes.astype(int)].real
    c_vals[1, 0] = (corr_pion[1] - corr_pion[0]).real if nt > 1 else 0.0
    c_vals[1, -1] = (corr_pion[-1] - corr_pion[-2]).real if nt > 1 else 0.0
    for k in range(1, len(t_nodes) - 1):
        idx = int(t_nodes[k])
        if idx > 0 and idx < nt - 1:
            c_vals[1, k] = 0.5 * (corr_pion[idx + 1] - corr_pion[idx - 1]).real
        else:
            c_vals[1, k] = 0.0

    breaks, coefs = calccf(t_nodes, c_vals)
    t_fine = np.linspace(0, float(nt - 1), 20)
    corr_spline = np.array([spline_eval(breaks, coefs, tt) for tt in t_fine])
    print(f"样条平滑后关联函数中值: {np.median(corr_spline):.6f}")

    # ============================================================
    # 8. 高阶积分：衰变常数与自能
    # ============================================================
    print_section("8. Gegenbauer 与 Alpert 高阶积分")
    print("计算 π 介子衰变常数 f_π...")
    f_pi = decay_constant_integral(mpi_est, mpi_est, lattice_spacing=1.0)
    print(f"f_π (格点单位) ≈ {f_pi:.6f}")

    print("计算单圈自能积分...")
    sigma_self = self_energy_integral(mass_quark, cutoff=np.pi)
    print(f"自能 Σ(m) ≈ {sigma_self:.6f}")

    # ============================================================
    # 9. 手征动力学
    # ============================================================
    print_section("9. 手征动力学与 Goldstone 玻色子")
    print("求解手征振子（Van der Pol 型非线性方程）...")
    t_chiral, y_chiral = solve_chiral_oscillator(
        np.array([0.5, 0.0]), (0.0, 10.0), mu_chiral=1.5
    )
    u_final = y_chiral[0, -1]
    v_final = y_chiral[1, -1]
    print(f"手征振子终态: u={u_final:.4f}, v={v_final:.4f}")

    print("求解夸克-介子反应网络...")
    network = QuarkMesonReactionNetwork()
    c0 = np.array([1.0, 1.0, 0.1, 0.0])
    t_rxn, c_rxn = network.solve(c0, (0.0, 20.0), n_points=100)
    sigma_q = chiral_condensate_from_reaction(t_rxn, c_rxn)
    print(f"稳态手征凝聚 ⟨q̄q⟩ ≈ {np.median(sigma_q[-10:]):.6f}")

    f_pi_gmor = pion_decay_constant_from_dynamics(mpi_est * 500.0)
    print(f"GMOR 关系 f_π ≈ {f_pi_gmor:.2f} MeV (使用 m_π={mpi_est*500:.1f} MeV)")

    # ============================================================
    # 10. 重整化群流
    # ============================================================
    print_section("10. 重整化群流方程")
    g0 = lattice_coupling_from_beta(beta_lat)
    mq0 = mass_quark
    print(f"初始耦合 g_0 = {g0:.4f} (from β={beta_lat})")
    print("积分 RG 流方程（两圈 β 函数）...")
    t_rg, y_rg = solve_rg_flow(g0, mq0, (-2.0, 1.0), nf=2, n_points=100)
    g_final = y_rg[0, -1]
    mq_final = y_rg[1, -1]
    print(f"RG 演化终态: g = {g_final:.4f}, m_q = {mq_final:.6f}")

    mu_vals = np.linspace(0.5, 5.0, 50)
    alpha_vals = alpha_s_running(mu_vals, lambda_qcd=0.3, nf=2)
    print(f"跑动耦合 α_s(2 GeV) ≈ {alpha_vals[len(alpha_vals)//2]:.4f}")

    print("\n多能标耦合反应网络...")
    g0_vec = np.array([g0 * 1.2, g0, g0 * 0.8, g0 * 0.6, g0 * 0.4])
    t_net, g_net = coupled_rg_reaction_network(g0_vec, (-1.0, 1.0), n_points=100)
    print(f"网络终态耦合中值: {np.median(g_net[:, -1]):.4f}")

    # ============================================================
    # 11. 结果汇总
    # ============================================================
    print_section("结果汇总")
    print(f"{'物理量':<30} {'数值':<20} {'单位/说明'}")
    print("-" * 70)
    print(f"{'π 介子质量 m_π':<30} {mpi_est:<20.6f} {'格点单位'}")
    print(f"{'核子质量 m_N':<30} {mn_est:<20.6f} {'格点单位'}")
    mass_ratio = mn_est / mpi_est if abs(mpi_est) > 1e-6 else np.nan
    print(f"{'m_N / m_π':<30} {mass_ratio:<20.4f} {'质量比'}")
    print(f"{'π 衰变常数 f_π':<30} {f_pi:<20.6f} {'格点单位'}")
    print(f"{'自能 Σ(m_q)':<30} {sigma_self:<20.6f} {'格点单位'}")
    print(f"{'GMOR f_π':<30} {f_pi_gmor:<20.2f} {'MeV'}")
    print(f"{'手征凝聚 ⟨q̄q⟩':<30} {np.median(sigma_q[-10:]):<20.6f} {'归一化单位'}")
    print(f"{'RG 终态耦合 g':<30} {g_final:<20.4f} {'裸耦合'}")
    print(f"{'α_s(2 GeV)':<30} {alpha_vals[len(alpha_vals)//2]:<20.4f} {'跑动耦合'}")
    print("-" * 70)
    print("\n格点 QCD 强子谱学博士级合成计算完成。")
    print("=" * 70)

    return {
        "mpi": mpi_est,
        "mn": mn_est,
        "fpi": f_pi,
        "sigma_self": sigma_self,
        "g_final": g_final,
        "alpha_s_2gev": alpha_vals[len(alpha_vals)//2],
    }


if __name__ == "__main__":
    results = main()
