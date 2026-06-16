# -*- coding: utf-8 -*-
"""
main.py
=======
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

统一入口: 零参数即可运行完整的 BNS 并合 + kilonova 辐射转移 +
高阶有限差分稳定性分析的小规模可复现实验.

执行流程
--------
1. 初始化物理常数与状态方程
2. 演化双中子星旋进轨道 (2.5PN)
3. 构造抛射物球坐标网格
4. 执行图划分 (域分解)
5. 构造 Chebyshev 不透明度表
6. 分析 HMNS f-mode 振荡
7. 执行 von Neumann 稳定性分析, 确定允许时间步长
8. 运行 Crank-Nicolson 辐射扩散
9. 运行蒙特卡洛光子包输运
10. 计算光度曲线
11. 收敛性分析
12. 输出最终摘要

映射种子项目清单
----------------
- 860  pendulum_nonlinear_exact         → hmfns_oscillation (Jacobi 椭圆)
- 1363 tsp_brute                        → von_neumann_stability (最严苛波数搜索)
                                        → domain_decomposition (组合遍历)
                                        → convergence_analysis (最坏误差搜索)
- 517  henon_orbit                      → von_neumann_stability (放大因子映射)
- 321  dueling_idiots                   → monte_carlo_transport (几何分布)
                                        → thermodynamic_eos (相变等待时间)
- 796  neighbors_to_metis_graph         → domain_decomposition (邻接→METIS)
- 577  image_diffuse4/8                 → high_order_fd (邻居模板 = FD)
- 1276 ConsistentMIClientSimulator      → (Python 包结构 + 脚本调度)
- 688  linpack_bench_backslash          → radiative_transfer (LU 求解器基准)
- 1369 two_body_ode                     → binary_inspiral (二体 ODE)
- 1192 Quantum Control (Hamiltonian)    → equation_of_state (哈密顿热力学)
                                        → thermodynamic_eos (相变动力学)
- 014  approx_chebyshev                 → opacity_tables (Chebyshev 插值)
- 406  fem2d_mesh_display               → ejecta_mesh (网格结构)
- 049  asa239 (alnorm/gammad)           → physical_constants (分布函数)
                                        → radiative_transfer (衰变热源)
- 1346 triangulation_q2l                → ejecta_mesh (二次→线性降阶)
- 677  line_distance                    → monte_carlo_transport (采样)
                                        → luminosity_lightcurve (统计)

运行
----
    python main.py
"""

from __future__ import annotations
import sys
import time
import math


def banner(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Stage 1: 物理常数
# ---------------------------------------------------------------------------
def stage_physical_constants() -> dict:
    banner("Stage 1 / Physical Constants & Derived Quantities")
    import physical_constants as pc
    ok = pc._self_check()
    derived = pc.all_derived_quantities()
    print(f"  [self-check] passed = {ok}")
    for k, v in list(derived.items())[:6]:
        print(f"    {k:32s} = {v:14.4e}")
    return {"constants": pc, "derived": derived}


# ---------------------------------------------------------------------------
# Stage 2: 状态方程
# ---------------------------------------------------------------------------
def stage_equation_of_state() -> dict:
    banner("Stage 2 / Equation of State (piecewise polytrope)")
    import equation_of_state as eos
    ok = eos._self_check()
    rho_test = 3.0e14
    P = eos.pressure_cold(rho_test)
    cs = eos.sound_speed_cold(rho_test)
    Gamma1 = eos.adiabatic_index(rho_test)
    print(f"  [self-check] passed = {ok}")
    print(f"  at rho = {rho_test:.2e} g/cm^3:")
    print(f"    P_cold   = {P:.4e} dyn/cm^2")
    print(f"    c_s      = {cs:.4e} cm/s = {cs / 3e10:.3f} c")
    print(f"    Gamma_1  = {Gamma1:.4f}")
    return {"eos": eos}


# ---------------------------------------------------------------------------
# Stage 3: 双中子星旋进
# ---------------------------------------------------------------------------
def stage_binary_inspiral() -> dict:
    banner("Stage 3 / Binary Neutron Star Inspiral (2.5PN)")
    import binary_inspiral as bi
    ok = bi._self_check()
    params = bi.default_binary_params()
    m1, m2 = params["m1_g"], params["m2_g"]
    r0 = params["initial_separation_cm"]
    print(f"  [self-check] passed = {ok}")
    print(f"  m1 = {m1 / 1.989e33:.3f} M_sun,  m2 = {m2 / 1.989e33:.3f} M_sun")
    print(f"  initial separation = {r0:.3e} cm")
    print(f"  chirp mass M_c = {bi.chirp_mass(m1, m2) / 1.989e33:.3f} M_sun")
    # 短时间演化 (仅 500 步, 验证可运行)
    t_start = time.perf_counter()
    traj = bi.evolve_orbit(m1, m2, r0, dt_s=1.0e-5, n_steps=500,
                           stop_at_contact=True)
    elapsed = time.perf_counter() - t_start
    print(f"  integrated {len(traj['time'])} steps in {elapsed:.3f} s")
    print(f"  initial r = {traj['r'][0]:.3e} cm, "
          f"final r = {traj['r'][-1]:.3e} cm")
    print(f"  peak f_GW = {max(traj['f_GW']):.2f} Hz")
    return {"traj": traj}


# ---------------------------------------------------------------------------
# Stage 4: 抛射物网格
# ---------------------------------------------------------------------------
def stage_ejecta_mesh() -> dict:
    banner("Stage 4 / Spherical Ejecta Mesh")
    import ejecta_mesh as em
    ok = em._self_check()
    mesh = em.SphericalMesh(N_r=20, N_theta=10, N_phi=20)
    V_num = mesh.total_volume()
    V_ana = mesh.expected_volume()
    rel = abs(V_num - V_ana) / V_ana
    print(f"  [self-check] passed = {ok}")
    print(f"  mesh = {mesh.N_r} x {mesh.N_theta} x {mesh.N_phi}")
    print(f"  V_num / V_ana = {V_num:.4e} / {V_ana:.4e} cm^3")
    print(f"  relative error = {rel:.3e}")
    # Q2L 降阶示例
    tri6 = [(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)]
    tri3 = em.quadratic_to_linear_triangles(tri6)
    print(f"  Q2L: {len(tri6)} quadratic -> {len(tri3)} linear triangles")
    return {"mesh": mesh}


# ---------------------------------------------------------------------------
# Stage 5: 域分解
# ---------------------------------------------------------------------------
def stage_domain_decomposition() -> dict:
    banner("Stage 5 / Graph-Based Domain Decomposition")
    import domain_decomposition as dd
    ok = dd._self_check()
    N_r, N_theta, N_phi = 6, 6, 12
    adj = dd.build_spherical_adjacency(N_r, N_theta, N_phi)
    coords = [(float(i), float(j), float(k))
              for i in range(N_r) for j in range(N_theta) for k in range(N_phi)]
    part = dd.rcb_partition(adj, coords, n_parts=8)
    metrics = dd.cut_metrics(adj, part)
    print(f"  [self-check] passed = {ok}")
    print(f"  graph: {len(adj)} vertices, {metrics['total_edges']} edges")
    print(f"  partition: {metrics['n_parts']} parts, sizes = {metrics['sizes']}")
    print(f"  balance = {metrics['balance']:.3f}")
    print(f"  edge cut = {metrics['edge_cut']}, ratio = {metrics['cut_ratio']:.3f}")
    return {"metrics": metrics}


# ---------------------------------------------------------------------------
# Stage 6: 不透明度表
# ---------------------------------------------------------------------------
def stage_opacity_tables() -> dict:
    banner("Stage 6 / Chebyshev Opacity Tables")
    import opacity_tables as ot
    ok = ot._self_check()
    T_K, rho, Ye = 5000.0, 1.0e-13, 0.2
    table = ot.build_chebyshev_opacity_table(T_K, rho, Ye, n_cheb=12)
    print(f"  [self-check] passed = {ok}")
    print(f"  T={T_K:.0f} K, rho={rho:.2e}, Ye={Ye:.2f}")
    print(f"  Chebyshev max error = {table['maxerr']:.4e} cm^2/g")
    print(f"  nodes (Angstrom): {[f'{x:.0f}' for x in table['nodes'][:6]]} ...")
    return {"table": table}


# ---------------------------------------------------------------------------
# Stage 7: HMNS f-mode 振荡
# ---------------------------------------------------------------------------
def stage_hmfns_oscillation() -> dict:
    banner("Stage 7 / HMNS F-mode Oscillation (Jacobi Elliptic)")
    import hmfns_oscillation as hmf
    ok = hmf._self_check()
    M_ns = 2.7 * 1.989e33
    R_ns = 1.3e6
    f_kHz = hmf.fmode_frequency(M_ns, R_ns)
    print(f"  [self-check] passed = {ok}")
    print(f"  HMNS mass  = {M_ns / 1.989e33:.2f} M_sun")
    print(f"  HMNS radius = {R_ns / 1e5:.1f} km")
    print(f"  f-mode freq = {f_kHz:.3f} kHz")
    # 振荡演化
    t_arr = [i * 5.0e-5 for i in range(200)]
    theta, thetadot = hmf.hmns_radial_oscillation(
        t_arr, amplitude_rad=0.1, omega_hz=f_kHz * 1000.0, k_modulus=0.3)
    print(f"  max |theta|    = {max(abs(th) for th in theta):.4f} rad")
    print(f"  max |thetadot| = {max(abs(td) for td in thetadot):.4e} rad/s")
    return {"f_kHz": f_kHz, "theta": theta}


# ---------------------------------------------------------------------------
# Stage 8: Von Neumann 稳定性分析
# ---------------------------------------------------------------------------
def stage_von_neumann() -> dict:
    banner("Stage 8 / Von Neumann Stability Analysis")
    import von_neumann_stability as vns
    ok = vns._self_check()
    print(f"  [self-check] passed = {ok}")
    h = 1.0e5  # 1 km
    v_max = 1.0e9
    c_s = 1.0e8
    kappa = 10.0
    rho = 1.0e-10
    dt_adv = vns.cfl_advection(h, v_max, c_s)
    dt_rad = vns.cfl_radiation_diffusion(h, kappa, rho)
    print(f"  grid h = {h:.2e} cm")
    print(f"  CFL dt_advection = {dt_adv:.4e} s")
    print(f"  CFL dt_radiation = {dt_rad:.4e} s")
    print(f"  recommended dt   = {min(dt_adv, dt_rad):.4e} s")
    # Henon-style 稳定性图
    henon_result = vns.henon_stability_scan(c_param=0.5, n_iter=500)
    print(f"  Henon stability: bounded_frac = {henon_result['bounded_fraction']:.3f}")
    return {"dt_adv": dt_adv, "dt_rad": dt_rad,
            "dt_rec": min(dt_adv, dt_rad)}


# ---------------------------------------------------------------------------
# Stage 9: 辐射扩散
# ---------------------------------------------------------------------------
def stage_radiative_transfer(dt_from_stage8: float) -> dict:
    banner("Stage 9 / Implicit Radiation Diffusion (Crank-Nicolson)")
    import radiative_transfer as rt
    ok = rt._self_check()
    print(f"  [self-check] passed = {ok}")
    # 1D 径向网格
    N = 50
    r_grid = [1.0e12 + i * 5.0e10 for i in range(N)]
    aT4 = 7.5657e-15
    T_init = 1.0e4
    E0 = [aT4 * T_init ** 4] * N
    kappa = [10.0] * N
    rho = [1.0e-13 * (r_grid[0] / r) ** 1.5 for r in r_grid]
    src = [rt.specific_heating_rate(1.0 * 86400.0, X_r=0.01) * rho[i]
           for i in range(N)]
    dt = min(dt_from_stage8, 10.0)
    E1 = rt.radiation_diffusion_step(E0, r_grid, dt, kappa, rho, src)
    print(f"  grid points = {N}")
    print(f"  dt = {dt:.4f} s")
    print(f"  E_rad (init)  = {E0[0]:.4e}")
    print(f"  E_rad (final) = {E1[N // 2]:.4e} (midpoint)")
    # LINPACK 基准
    bench = rt.linpack_style_benchmark(100)
    print(f"  LINPACK n={bench['n']}: {bench['time_s']:.4f} s, "
          f"{bench['MFLOPS']:.2f} MFLOPS")
    return {"E_final": E1, "bench": bench}


# ---------------------------------------------------------------------------
# Stage 10: 蒙特卡洛输运
# ---------------------------------------------------------------------------
def stage_monte_carlo() -> dict:
    banner("Stage 10 / Monte Carlo Photon Packet Transport")
    import monte_carlo_transport as mc
    ok = mc._self_check()
    print(f"  [self-check] passed = {ok}")
    env = mc.KilonovaEnvelope()
    stats = mc.run_mc_simulation(env, n_packets=300, seed=42)
    print(f"  packets launched : {stats['n_packets']}")
    print(f"  escaped          : {stats['n_escaped']}")
    print(f"  p_escape         : {stats['p_escape']:.3f}")
    print(f"  mean scatters    : {stats['mean_scatters']:.2f}")
    print(f"  mean t_escape    : {stats['mean_t_escape'] / 86400:.2f} days")
    # 几何分布校验
    geom = mc.geometric_distribution_check(0.1, n_trials=300, seed=1)
    print(f"  Geom check: E[N]_theory = {geom['theoretical_mean']:.2f}, "
          f"E[N]_sample = {geom['sample_mean']:.2f}")
    return {"stats": stats}


# ---------------------------------------------------------------------------
# Stage 11: 光度曲线
# ---------------------------------------------------------------------------
def stage_lightcurve() -> dict:
    banner("Stage 11 / Kilonova Lightcurve (Arnett model)")
    import luminosity_lightcurve as lc
    ok = lc._self_check()
    print(f"  [self-check] passed = {ok}")
    t_days = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    result = lc.arnett_lightcurve(t_days, M_ej_Msun=0.05, v_c_km_s=10000,
                                  kappa_cgs=10.0, X_r=0.01)
    L_peak, t_peak = lc.peak_luminosity(result["L_erg_s"], t_days)
    print(f"  t_days = {t_days}")
    for i, t in enumerate(t_days):
        print(f"    t = {t:5.1f} d: L = {result['L_erg_s'][i]:.3e} erg/s, "
              f"T_eff = {result['T_eff_K'][i]:.0f} K")
    print(f"  L_peak = {L_peak:.3e} erg/s at t = {t_peak:.2f} days")
    print(f"  tau_m  = {result['tau_m']:.2f}")
    return {"L_peak": L_peak, "t_peak": t_peak, "tau_m": result["tau_m"]}


# ---------------------------------------------------------------------------
# Stage 12: 收敛分析
# ---------------------------------------------------------------------------
def stage_convergence() -> dict:
    banner("Stage 12 / Convergence Analysis")
    import convergence_analysis as ca
    ok = ca._self_check()
    print(f"  [self-check] passed = {ok}")
    # 用 sin(1) + h^2 的测试函数
    import math
    def solver(h):
        return math.sin(1.0) + 0.5 * h * h
    result = ca.convergence_test(solver, [0.1, 0.05, 0.025, 0.0125],
                                 exact=math.sin(1.0))
    print(f"  errors         : {[f'{e:.3e}' for e in result['errors']]}")
    print(f"  observed order : {[f'{p:.3f}' for p in result['p_observed']]}")
    # 误差预算
    budget = ca.error_budget({
        "spatial": 0.01, "temporal": 0.005,
        "opacity": 0.02, "statistical": 0.015
    })
    print(f"  error budget   :")
    for k in ["spatial", "temporal", "opacity", "statistical", "total"]:
        if k in budget:
            print(f"    {k:16s} = {budget[k]:.4e}")
    return {"p_obs": result["p_observed"], "budget": budget}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_total_start = time.perf_counter()
    print("\n" + "#" * 72)
    print("#" + " " * 70 + "#")
    print("#   PROJECT_254 : Binary Neutron Star Merger & Kilonova         #")
    print("#   High-Order Finite Differences & Stability Analysis          #")
    print("#" + " " * 70 + "#")
    print("#" * 72)
    print(f"\n  Python executable : {sys.executable}")
    print(f"  Python version    : {sys.version.split()[0]}")

    s1 = stage_physical_constants()
    s2 = stage_equation_of_state()
    s3 = stage_binary_inspiral()
    s4 = stage_ejecta_mesh()
    s5 = stage_domain_decomposition()
    s6 = stage_opacity_tables()
    s7 = stage_hmfns_oscillation()
    s8 = stage_von_neumann()
    s9 = stage_radiative_transfer(s8["dt_rec"])
    s10 = stage_monte_carlo()
    s11 = stage_lightcurve()
    s12 = stage_convergence()

    elapsed = time.perf_counter() - t_total_start
    banner("Summary")
    print(f"  total wall-clock time      : {elapsed:.3f} s")
    print(f"  BNS peak GW frequency      : {max(s3['traj']['f_GW']):.1f} Hz")
    print(f"  HMNS f-mode                : {s7['f_kHz']:.3f} kHz")
    print(f"  CFL-recommended dt         : {s8['dt_rec']:.3e} s")
    print(f"  MC escape probability      : {s10['stats']['p_escape']:.3f}")
    print(f"  kilonova L_peak            : {s11['L_peak']:.3e} erg/s")
    print(f"  kilonova t_peak            : {s11['t_peak']:.2f} days")
    print(f"  convergence order observed : {s12['p_obs'][0]:.3f}")
    print("\n" + "=" * 72)
    print("  Normal end of execution. All 12 stages completed successfully.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
