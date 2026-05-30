# -*- coding: utf-8 -*-

import numpy as np
import sys
import time


from nuclear_eos import nuclear_matter_properties, parameter_uncertainty_t_stat, SKYRME_PARAMS
from geometry_pasta import create_pasta_phase, pasta_energy_landscape, PastaPhase
from coulomb_solver import analytical_coulomb, wigner_seitz_coulomb
from reaction_diffusion import fd_reaction_diffusion_1d, beta_decay_rates, diffusion_coefficient
from ode_integrator import solve_crust_cooling, solve_unstable_system, solve_tough_system
from bessel_modes import besselj_zero, cylinder_vibration_frequencies, spherical_coulomb_potential
from cvt_sampler import monte_carlo_nd_integral, nd_integrand_gaussian, optimize_pasta_cvt
from phase_diagram import (
    total_energy_per_nucleon, optimal_filling, compute_phase_diagram,
    stability_analysis, transition_density
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_eos_calculation():
    print_section("1. 核物质状态方程 (Skyrme能量密度泛函)")

    rho_values = np.array([0.03, 0.05, 0.08, 0.10, 0.16])
    x_p = 0.3

    print(f"{'密度(fm^-3)':<12} {'E/A(MeV)':<12} {'P(MeV/fm^3)':<14} {'E_sym(MeV)':<12} {'K(MeV)':<10}")
    print("-" * 70)
    for rho in rho_values:
        try:
            props = nuclear_matter_properties(rho, x_p)
            print(f"{rho:<12.3f} {props['energy_per_nucleon']:<12.2f} "
                  f"{props['pressure']:<14.4f} {props['symmetry_energy']:<12.2f} "
                  f"{props['incompressibility']:<10.2f}")
        except Exception as e:
            print(f"rho={rho:.3f}: 计算失败 ({e})")


    print("\n  Skyrme参数t0拟合不确定性分析 (非中心t分布):")
    estimates = np.array([-1790, -1810, -1805, -1785, -1800])
    true_val = -1800.0
    std_errs = np.array([15.0, 12.0, 18.0, 14.0, 16.0])
    coverage, ci_l, ci_u = parameter_uncertainty_t_stat(estimates, true_val, std_errs)
    print(f"    覆盖率: {coverage:.2%}")
    print(f"    95% CI: [{ci_l:.1f}, {ci_u:.1f}] MeV·fm^3")


def run_geometry_modeling():
    print_section("2. 核pasta相几何结构")

    rho = 0.08
    x_p = 0.3

    print(f"密度 = {rho} fm^-3, 质子分数 = {x_p}")
    print(f"{'相':<18} {'特征尺度(fm)':<14} {'S/V(fm^-1)':<12} {'f_C(u)':<10}")
    print("-" * 60)

    for pid in range(1, 6):
        phase = create_pasta_phase(pid, rho, x_p)
        name = phase.PHASE_NAMES[pid]
        R_or_t = getattr(phase, 'R', getattr(phase, 't', phase.a_WS))
        print(f"{name:<18} {R_or_t:<14.3f} {phase.surface_to_volume():<12.3f} "
              f"{phase.coulomb_factor():<10.4f}")


    print("\n  能量景观计算 (不同密度):")
    rho_range = np.linspace(0.03, 0.12, 10)
    landscape = pasta_energy_landscape(rho_range, x_p, n_points=15)
    for name, energies in landscape.items():
        print(f"    {name}: min={np.min(energies):.4f}, max={np.max(energies):.4f} MeV")


def run_coulomb_solver():
    print_section("3. 库仑势与静电能")

    rho = 0.08
    x_p = 0.3

    print(f"解析近似库仑能 (密度={rho} fm^-3, x_p={x_p}):")
    print(f"{'相':<18} {'E_Coulomb/核子(MeV)':<20}")
    print("-" * 40)
    for pid in range(1, 6):
        e_c = analytical_coulomb(pid, rho, x_p)
        name = PastaPhase.PHASE_NAMES[pid]
        print(f"{name:<18} {e_c:<20.6f}")


    print("\n  有限元Wigner-Seitz单元求解 (球状相):")
    try:
        e_c_fem = wigner_seitz_coulomb(rho, x_p, 1, n_r=30)
        print(f"    FEM库仑能: {e_c_fem:.6f} MeV/核子")
    except Exception as e:
        print(f"    FEM求解异常 (网格过粗): {e}")


    from bessel_modes import spherical_coulomb_potential
    r_test = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    R_ws = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    rho_p = rho * x_p
    phi = spherical_coulomb_potential(r_test, R_ws, rho_p)
    print(f"\n  球对称库仑势 (R_WS={R_ws:.3f} fm):")
    for ri, phii in zip(r_test, phi):
        print(f"    r={ri:.1f} fm: Phi={phii:.4f} MeV")


def run_reaction_diffusion():
    print_section("4. 核子反应-扩散动力学")

    T = 1.0
    rho_n = 0.05
    rho_p = 0.02
    mu_e = 50.0

    lp, lm = beta_decay_rates(T, rho_n, rho_p, mu_e)
    print(f"温度={T} MeV, rho_n={rho_n}, rho_p={rho_p}")
    print(f"  beta+衰变率 (p->n): {lp:.4e} s^-1")
    print(f"  beta-衰变率 (n->p): {lm:.4e} s^-1")


    D_n = 0.5
    D_p = 0.3
    print(f"\n  扩散系数 (演示值):")
    print(f"    D_n = {D_n:.4e} fm^2/s")
    print(f"    D_p = {D_p:.4e} fm^2/s")


    print("\n  1D有限差分反应扩散模拟:")
    N = 50
    x = np.linspace(0, 10, N)
    rho_n0 = np.exp(-(x - 5)**2) * 0.1
    rho_p0 = np.ones(N) * 0.02

    try:
        lp_safe = min(lp, 0.1)
        lm_safe = min(lm, 0.1)
        rho_n, rho_p, _, _ = fd_reaction_diffusion_1d(
            rho_n0, rho_p0, dx=x[1]-x[0], dt=0.001, n_steps=200,
            D_n=D_n, D_p=D_p, lambda_plus=lp_safe, lambda_minus=lm_safe,
            bc_type='neumann'
        )
        print(f"    初始: rho_n=[{rho_n0.min():.4f}, {rho_n0.max():.4f}], "
              f"rho_p=[{rho_p0.min():.4f}, {rho_p0.max():.4f}]")
        print(f"    最终: rho_n=[{rho_n.min():.4f}, {rho_n.max():.4f}], "
              f"rho_p=[{rho_p.min():.4f}, {rho_p.max():.4f}]")
    except Exception as e:
        print(f"    FD模拟异常: {e}")


def run_ode_integration():
    print_section("5. 温度演化与ODE系统")


    print("  中子星crust冷却模拟 (100秒):")
    T0 = 1.0
    rho = 0.1
    x_p = 0.3
    try:
        sol = solve_crust_cooling([0, 100.0], T0, rho, x_p, heating_rate=0.0)
        T_final = sol.y[0, -1]
        Q_final = sol.y[1, -1]
        print(f"    初始温度: {T0:.4f} MeV")
        print(f"    最终温度: {T_final:.6f} MeV")
        print(f"    累积加热: {Q_final:.4e} MeV/fm^3")
        print(f"    时间步数: {len(sol.t)}")
    except Exception as e:
        print(f"    冷却模拟异常: {e}")


    print("\n  不稳定ODE测试 (mu=0.5):")
    try:
        t, y_exact, y_num = solve_unstable_system([0, 1.0], [1.0, 0.0], mu=0.5)
        err = np.abs(y_exact[0, -1] - y_num[0, -1])
        print(f"    精确解 y1(1) = {y_exact[0, -1]:.6f}")
        print(f"    数值解 y1(1) = {y_num[0, -1]:.6f}")
        print(f"    误差 = {err:.2e}")
    except Exception as e:
        print(f"    不稳定ODE异常: {e}")


    print("\n  刚性ODE测试:")
    try:
        sol = solve_tough_system([0, 1.0], [1.0, 1.0, 0.0, 1.0])
        print(f"    y1(1) = {sol.y[0, -1]:.6f}")
        print(f"    y2(1) = {sol.y[1, -1]:.6f}")
        print(f"    成功步数: {sol.nfev}")
    except Exception as e:
        print(f"    刚性ODE异常: {e}")


def run_bessel_modes():
    print_section("6. 柱坐标本征模式与贝塞尔零点")

    print("  贝塞尔函数 J_0 前10个零点:")
    zeros = besselj_zero(0, 10)
    for i, z in enumerate(zeros):
        print(f"    alpha_{i+1,0} = {z:.6f}")

    print("\n  柱相振动频率 (R=5 fm, sigma=1 MeV/fm^2, rho=0.08 fm^-3):")
    freqs = cylinder_vibration_frequencies(5.0, 1.0, 0.08 * 939.0)
    if len(freqs) > 0:
        for i, f in enumerate(freqs[:5]):
            print(f"    omega_{i+1} = {f:.6e} (自然单位)")
    else:
        print("    无有效频率")


def run_cvt_sampling():
    print_section("7. CVT结构优化与多维积分")


    print("  N维高斯积分 (蒙特卡洛):")
    for dim in [2, 3, 5]:
        integral, error = monte_carlo_nd_integral(
            nd_integrand_gaussian, dim, -3, 3, n_samples=20000
        )
        exact = np.pi ** (dim / 2.0)
        rel_err = abs(integral - exact) / exact
        print(f"    dim={dim}: I={integral:.6f} +/- {error:.6f}, "
              f"exact={exact:.6f}, rel_err={rel_err:.4%}")


    print("\n  2D CVT结构优化 (模拟核团分布):")
    try:
        generators, areas, energy, motion = optimize_pasta_cvt(
            0.08, 0.3, 1, n_generators=10, it_num=30
        )
        print(f"    生成点数: {len(generators)}")
        print(f"    平均Voronoi面积: {np.mean(areas):.4f} fm^2")
        print(f"    最终CVT能量: {energy[-1]:.6f}")
        print(f"    最终平均位移: {motion[-1]:.6e}")
    except Exception as e:
        print(f"    CVT优化异常: {e}")


def run_phase_diagram():
    print_section("8. 相图与稳定性分析")

    rho = 0.08
    x_p = 0.3

    print(f"  密度={rho} fm^-3, 质子分数={x_p} 下的各相能量:")
    print(f"{'相':<18} {'u_opt':<8} {'E/A(MeV)':<12} {'E_bulk':<10} {'E_surf':<10} {'E_Coul':<10}")
    print("-" * 75)

    for pid in range(1, 6):
        try:
            u_opt, _ = optimal_filling(pid, rho, x_p)
            E_total, comp = total_energy_per_nucleon(pid, rho, x_p, u=u_opt)
            name = PastaPhase.PHASE_NAMES[pid]
            print(f"{name:<18} {u_opt:<8.3f} {E_total:<12.4f} "
                  f"{comp.get('bulk', 0):<10.4f} {comp.get('surface', 0):<10.4f} "
                  f"{comp.get('coulomb', 0):<10.6f}")
        except Exception as e:
            print(f"Phase {pid}: 计算失败 ({e})")


    print(f"\n  球状相(Gnocchi)稳定性分析:")
    try:
        stab = stability_analysis(rho, x_p, 1, temperature=0.0)
        print(f"    力学稳定: {stab['mechanical_stable']} (dP/drho={stab['dP_drho']:.4f})")
        print(f"    化学稳定: {stab['chemical_stable']} (d2E/dx2={stab['d2E_dx2']:.4f})")
        print(f"    总体稳定: {stab['stable']}")
        for m_info in stab['modes']:
            print(f"    模式 m={m_info['mode']}: dE={m_info['deformation_energy']:.4f}, "
                  f"稳定={m_info['stable']}")
    except Exception as e:
        print(f"    稳定性分析异常: {e}")


    print(f"\n  相转变密度:")
    pairs = [(1, 2), (2, 3), (3, 4), (4, 5)]
    for p1, p2 in pairs:
        try:
            rho_t, found = transition_density(p1, p2, x_p, rho_min=0.02, rho_max=0.15)
            if found:
                name1 = PastaPhase.PHASE_NAMES[p1]
                name2 = PastaPhase.PHASE_NAMES[p2]
                print(f"    {name1} -> {name2}: rho = {rho_t:.4f} fm^-3")
            else:
                print(f"    {PastaPhase.PHASE_NAMES[p1]} -> {PastaPhase.PHASE_NAMES[p2]}: 未找到")
        except Exception as e:
            print(f"    转变密度计算异常: {e}")


    print(f"\n  简化相图 (T=0, x_p={x_p}):")
    rho_range = np.linspace(0.03, 0.12, 8)
    T_range = np.array([0.0])
    try:
        phase_map, energy_map = compute_phase_diagram(rho_range, T_range, x_p)
        for i_rho, rho_val in enumerate(rho_range):
            pid = phase_map[0, i_rho]
            name = PastaPhase.PHASE_NAMES.get(pid, 'unknown')
            print(f"    rho={rho_val:.3f}: {name}")
    except Exception as e:
        print(f"    相图计算异常: {e}")


def main():
    print("\n" + "#" * 70)
    print("#  中子星crust层核pasta相计算平台")
    print("#  Neutron Star Crust Nuclear Pasta Phase Computation")
    print("#" * 70)
    print(f"\n运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Python版本:", sys.version.split()[0])

    t_start = time.time()

    run_eos_calculation()
    run_geometry_modeling()
    run_coulomb_solver()
    run_reaction_diffusion()
    run_ode_integration()
    run_bessel_modes()
    run_cvt_sampling()
    run_phase_diagram()

    t_elapsed = time.time() - t_start
    print("\n" + "#" * 70)
    print(f"#  计算完成, 总耗时: {t_elapsed:.2f} 秒")
    print("#" * 70 + "\n")


if __name__ == '__main__':
    main()
