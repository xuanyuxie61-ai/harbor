# -*- coding: utf-8 -*-
"""
main.py
中子星crust层核pasta相计算平台

本程序为统一入口,零参数可运行. 完成以下计算流程:
1. 核物质状态方程计算 (Skyrme能量密度泛函)
2. 五种核pasta相的几何建模
3. 有限元库仑势求解
4. 反应-扩散动力学模拟
5. 温度演化ODE积分
6. 相图计算与稳定性分析
7. CVT结构优化采样

物理背景:
在中子星crust层(密度约10^11 - 10^14 g/cm^3, 即0.001-0.1 fm^{-3}),
核物质在强库仑相互作用与表面张力竞争下,自组织成各种非均匀结构,
即所谓的"核pasta相". 这些结构包括:
- 球状相(gnocchi): 核物质球嵌入核子气体
- 柱状相(spaghetti): 核物质柱
- 片状相(lasagna): 核物质片
- 管状相(anti-spaghetti): 气体柱嵌入核物质
- 泡状相(anti-gnocchi): 气体泡嵌入核物质

核心公式体系:
1. Skyrme能量密度泛函:
   H = (hbar^2/2m)tau + t0/2[(1+x0/2)rho^2 - (x0+1/2)(rho_n^2+rho_p^2)]
     + t3/24[(1+x3/2)rho^{alpha+2} - (x3+1/2)rho^alpha(rho_n^2+rho_p^2)]

2. 总能量/核子:
   E/A = E_bulk/A + E_surf/A + E_Coulomb/A + E_lattice/A
   E_surf/A = sigma * S/(rho*V)
   E_Coulomb/A = (3/10)(e^2/R_WS)(rho_p/rho)^2 * f_C(u)

3. Gibbs平衡:
   mu_n^I = mu_n^II,  mu_p^I = mu_p^II,  P^I = P^II

4. 冷却方程:
   C_V dT/dt = -epsilon_nu + epsilon_crust
"""

import numpy as np
import sys
import time

# 导入所有模块
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
    """打印章节标题."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_eos_calculation():
    """1. 核物质状态方程计算."""
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

    # 参数不确定性分析 (asa243融入)
    print("\n  Skyrme参数t0拟合不确定性分析 (非中心t分布):")
    estimates = np.array([-1790, -1810, -1805, -1785, -1800])
    true_val = -1800.0
    std_errs = np.array([15.0, 12.0, 18.0, 14.0, 16.0])
    coverage, ci_l, ci_u = parameter_uncertainty_t_stat(estimates, true_val, std_errs)
    print(f"    覆盖率: {coverage:.2%}")
    print(f"    95% CI: [{ci_l:.1f}, {ci_u:.1f}] MeV·fm^3")


def run_geometry_modeling():
    """2. 核pasta相几何建模."""
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

    # 能量景观
    print("\n  能量景观计算 (不同密度):")
    rho_range = np.linspace(0.03, 0.12, 10)
    landscape = pasta_energy_landscape(rho_range, x_p, n_points=15)
    for name, energies in landscape.items():
        print(f"    {name}: min={np.min(energies):.4f}, max={np.max(energies):.4f} MeV")


def run_coulomb_solver():
    """3. 库仑势计算."""
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

    # 有限元求解 (简化2D)
    print("\n  有限元Wigner-Seitz单元求解 (球状相):")
    try:
        e_c_fem = wigner_seitz_coulomb(rho, x_p, 1, n_r=30)
        print(f"    FEM库仑能: {e_c_fem:.6f} MeV/核子")
    except Exception as e:
        print(f"    FEM求解异常 (网格过粗): {e}")

    # 球对称精确解
    from bessel_modes import spherical_coulomb_potential
    r_test = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    R_ws = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    rho_p = rho * x_p
    phi = spherical_coulomb_potential(r_test, R_ws, rho_p)
    print(f"\n  球对称库仑势 (R_WS={R_ws:.3f} fm):")
    for ri, phii in zip(r_test, phi):
        print(f"    r={ri:.1f} fm: Phi={phii:.4f} MeV")


def run_reaction_diffusion():
    """4. 反应-扩散动力学."""
    print_section("4. 核子反应-扩散动力学")

    T = 1.0  # MeV
    rho_n = 0.05
    rho_p = 0.02
    mu_e = 50.0  # MeV

    lp, lm = beta_decay_rates(T, rho_n, rho_p, mu_e)
    print(f"温度={T} MeV, rho_n={rho_n}, rho_p={rho_p}")
    print(f"  beta+衰变率 (p->n): {lp:.4e} s^-1")
    print(f"  beta-衰变率 (n->p): {lm:.4e} s^-1")

    # 使用演示值避免数值不稳定
    D_n = 0.5
    D_p = 0.3
    print(f"\n  扩散系数 (演示值):")
    print(f"    D_n = {D_n:.4e} fm^2/s")
    print(f"    D_p = {D_p:.4e} fm^2/s")

    # 1D有限差分模拟
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
    """5. 温度演化与ODE积分."""
    print_section("5. 温度演化与ODE系统")

    # 中子星冷却
    print("  中子星crust冷却模拟 (100秒):")
    T0 = 1.0  # MeV
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

    # 不稳定ODE
    print("\n  不稳定ODE测试 (mu=0.5):")
    try:
        t, y_exact, y_num = solve_unstable_system([0, 1.0], [1.0, 0.0], mu=0.5)
        err = np.abs(y_exact[0, -1] - y_num[0, -1])
        print(f"    精确解 y1(1) = {y_exact[0, -1]:.6f}")
        print(f"    数值解 y1(1) = {y_num[0, -1]:.6f}")
        print(f"    误差 = {err:.2e}")
    except Exception as e:
        print(f"    不稳定ODE异常: {e}")

    # 刚性ODE
    print("\n  刚性ODE测试:")
    try:
        sol = solve_tough_system([0, 1.0], [1.0, 1.0, 0.0, 1.0])
        print(f"    y1(1) = {sol.y[0, -1]:.6f}")
        print(f"    y2(1) = {sol.y[1, -1]:.6f}")
        print(f"    成功步数: {sol.nfev}")
    except Exception as e:
        print(f"    刚性ODE异常: {e}")


def run_bessel_modes():
    """6. 柱坐标本征模式."""
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
    """7. CVT采样与多维积分."""
    print_section("7. CVT结构优化与多维积分")

    # N维高斯积分
    print("  N维高斯积分 (蒙特卡洛):")
    for dim in [2, 3, 5]:
        integral, error = monte_carlo_nd_integral(
            nd_integrand_gaussian, dim, -3, 3, n_samples=20000
        )
        exact = np.pi ** (dim / 2.0)
        rel_err = abs(integral - exact) / exact
        print(f"    dim={dim}: I={integral:.6f} +/- {error:.6f}, "
              f"exact={exact:.6f}, rel_err={rel_err:.4%}")

    # CVT优化
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
    """8. 相图计算与稳定性分析."""
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

    # 稳定性分析
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

    # 相转变密度
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

    # 简化相图
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
    """主程序入口."""
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

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: skyrme_energy_density 零密度返回零 ----
from nuclear_eos import skyrme_energy_density
ed, pr = skyrme_energy_density(0.0, 0.0)
assert ed == 0.0 and pr == 0.0, '[TC01] skyrme_energy_density 零密度 FAILED'

# ---- TC02: nuclear_matter_properties 饱和密度能量为有限负值 ----
props = nuclear_matter_properties(0.16, 0.5)
assert np.isfinite(props['energy_per_nucleon']), '[TC02] 能量非有限 FAILED'
assert props['energy_per_nucleon'] < 0, '[TC02] 能量非负 FAILED'

# ---- TC03: nuclear_matter_properties 对称能为正值 ----
props = nuclear_matter_properties(0.08, 0.3)
assert props['symmetry_energy'] > 0, '[TC03] 对称能非正 FAILED'

# ---- TC04: alnorm 正态CDF在x=0处为0.5 ----
from nuclear_eos import alnorm
val = alnorm(0.0, upper=True)
assert abs(val - 0.5) < 1e-10, '[TC04] alnorm(0,upper) FAILED'

# ---- TC05: tnc df<=0返回错误标志2 ----
from nuclear_eos import tnc
val, ifault = tnc(0.0, 0.0, 0.0)
assert ifault == 2, '[TC05] tnc df<=0 FAILED'

# ---- TC06: parameter_uncertainty_t_stat 覆盖率在[0,1]区间内 ----
cov, ci_l, ci_u = parameter_uncertainty_t_stat(np.array([-1800.0]), -1800.0, np.array([10.0]))
assert 0.0 <= cov <= 1.0, '[TC06] 覆盖率范围 FAILED'

# ---- TC07: create_pasta_phase 五种相S/V均为正 ----
for pid in range(1, 6):
    phase = create_pasta_phase(pid, 0.08, 0.3)
    assert phase.surface_to_volume() > 0, f'[TC07] Phase {pid} S/V FAILED'

# ---- TC08: PastaPhase.PHASE_NAMES 包含5个相 ----
assert len(PastaPhase.PHASE_NAMES) == 5, '[TC08] 相数量 FAILED'

# ---- TC09: triangle01_monomial_integral [0,0]等于0.5 ----
from geometry_pasta import triangle01_monomial_integral
val = triangle01_monomial_integral([0, 0])
assert abs(val - 0.5) < 1e-12, '[TC09] 单位三角形积分 FAILED'

# ---- TC10: tetrahedron_unit_monomial [0,0,0]等于1/6 ----
from geometry_pasta import tetrahedron_unit_monomial
val = tetrahedron_unit_monomial([0, 0, 0])
assert abs(val - 1.0 / 6.0) < 1e-12, '[TC10] 单位四面体积分 FAILED'

# ---- TC11: analytical_coulomb 五种相返回值非负 ----
for pid in range(1, 6):
    e_c = analytical_coulomb(pid, 0.08, 0.3)
    assert e_c >= 0, f'[TC11] Phase {pid} Coulomb FAILED'

# ---- TC12: beta_decay_rates 零温度返回零 ----
lp, lm = beta_decay_rates(0.0, 0.05, 0.02, 50.0)
assert lp == 0.0 and lm == 0.0, '[TC12] 零温度衰变率 FAILED'

# ---- TC13: diffusion_coefficient 正参数返回正值 ----
D = diffusion_coefficient(1.0, 1.0, 5.0)
assert D > 0, '[TC13] 扩散系数非正 FAILED'

# ---- TC14: fd_reaction_diffusion_1d 总核子数近似守恒 ----
N = 20
x = np.linspace(0, 5, N)
rho_n0 = np.ones(N) * 0.05
rho_p0 = np.ones(N) * 0.02
total0 = np.sum(rho_n0 + rho_p0)
rho_n, rho_p, _, _ = fd_reaction_diffusion_1d(
    rho_n0, rho_p0, dx=x[1] - x[0], dt=0.001, n_steps=50,
    D_n=0.1, D_p=0.1, lambda_plus=0.01, lambda_minus=0.01, bc_type='neumann'
)
total1 = np.sum(rho_n + rho_p)
assert abs(total1 - total0) / total0 < 0.1, '[TC14] 总核子数守恒 FAILED'

# ---- TC15: unstable_exact t=0时等于初始条件[1, mu] ----
from ode_integrator import unstable_exact
y1, y2 = unstable_exact(0.0, 0.5)
assert abs(y1 - 1.0) < 1e-12 and abs(y2 - 0.5) < 1e-12, '[TC15] unstable_exact初值 FAILED'

# ---- TC16: unstable_deriv mu=1时导数计算正确 ----
from ode_integrator import unstable_deriv
dydt = unstable_deriv(0.0, [1.0, 0.0], 1.0)
assert abs(dydt[0] - 1.0) < 1e-12 and abs(dydt[1] + 1.0) < 1e-12, '[TC16] unstable_deriv FAILED'

# ---- TC17: tough_deriv 输出长度为4 ----
from ode_integrator import tough_deriv
dydt = tough_deriv(0.0, [1.0, 1.0, 0.0, 1.0])
assert len(dydt) == 4, '[TC17] tough_deriv 维度 FAILED'

# ---- TC18: neutrino_luminosity 零温度返回0 ----
from ode_integrator import neutrino_luminosity
eps = neutrino_luminosity(0.0, 0.1, 0.3)
assert eps == 0.0, '[TC18] 零温度中微子发光度 FAILED'

# ---- TC19: heat_capacity_degenerate 正温度正比热 ----
from ode_integrator import heat_capacity_degenerate
cv = heat_capacity_degenerate(0.1, 0.3, 1.0)
assert cv > 0, '[TC19] 比热非正 FAILED'

# ---- TC20: besselj_zero J0第一个零点约2.4048 ----
zeros = besselj_zero(0, 3)
assert abs(zeros[0] - 2.4048255577) < 1e-4, '[TC20] J0第一个零点 FAILED'

# ---- TC21: spherical_coulomb_potential 球内递减球外衰减 ----
r = np.array([0.0, 0.5, 1.0, 2.0])
R_ws = (3.0 / (4.0 * np.pi * 0.08)) ** (1.0 / 3.0)
phi = spherical_coulomb_potential(r, R_ws, 0.08 * 0.3)
assert phi[0] > phi[2], '[TC21] 球内势递减 FAILED'
assert phi[-1] < phi[0], '[TC21] 球外势衰减 FAILED'

# ---- TC22: cylinder_vibration_frequencies 正参数返回非空递增序列 ----
freqs = cylinder_vibration_frequencies(5.0, 1.0, 0.08 * 939.0)
assert len(freqs) > 0, '[TC22] 振动频率为空 FAILED'
assert np.all(np.diff(freqs) >= 0), '[TC22] 频率非递增 FAILED'

# ---- TC23: monte_carlo_nd_integral 2D高斯积分接近理论值pi ----
np.random.seed(42)
integral, error = monte_carlo_nd_integral(nd_integrand_gaussian, 2, -3, 3, n_samples=50000)
exact = np.pi
assert abs(integral - exact) / exact < 0.05, '[TC23] 2D高斯积分 FAILED'

# ---- TC24: nd_integrand_gaussian 在零点值为1 ----
x = np.zeros((2, 5))
val = nd_integrand_gaussian(2, 5, x)
assert np.allclose(val, 1.0), '[TC24] 高斯被积函数零点 FAILED'

# ---- TC25: total_energy_per_nucleon 含bulk分量 ----
E_total, comp = total_energy_per_nucleon(1, 0.08, 0.3)
assert 'bulk' in comp and np.isfinite(comp['bulk']), '[TC25] bulk分量缺失 FAILED'

# ---- TC26: optimal_filling 返回u在(0,1)区间内 ----
u_opt, E_min = optimal_filling(1, 0.08, 0.3)
assert 0.0 < u_opt < 1.0, '[TC26] 最优填充率范围 FAILED'

# ---- TC27: surface_tension 对称物质为正 ----
from phase_diagram import surface_tension
sigma = surface_tension(0.16, 0.5)
assert sigma > 0, '[TC27] 表面张力非正 FAILED'

# ---- TC28: lattice_energy 为负值 ----
from phase_diagram import lattice_energy
E_lat = lattice_energy(0.08, 0.3)
assert E_lat < 0, '[TC28] 晶格能非负 FAILED'

# ---- TC29: pasta_deformation_energy m=2柱相为正 ----
from bessel_modes import pasta_deformation_energy
dE = pasta_deformation_energy(2, 5.0, 0.1, 2, 1.0)
assert dE > 0, '[TC29] 柱相形变能非正 FAILED'

# ---- TC30: transition_density 搜索返回值在合理范围 ----
rho_t, found = transition_density(1, 2, 0.3, rho_min=0.02, rho_max=0.15)
if found:
    assert 0.02 <= rho_t <= 0.15, '[TC30] 转变密度范围 FAILED'
assert (found and rho_t is not None) or (not found and rho_t is None), '[TC30] 转变密度返回值 FAILED'

print('\n全部 30 个测试通过!\n')
