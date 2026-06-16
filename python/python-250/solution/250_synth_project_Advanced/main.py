"""
超新星爆发辐射流体模拟：高阶有限差分与稳定性分析
====================================================

统一入口 (零参数). 运行流程：
  1. 设置物理常数和球对称网格
  2. 构造前身星初始剖面 (解析近似)
  3. 在激波处插入 Rankine-Hugoniot 跳跃
  4. 演化辐射流体若干时间步 (WENO5 + HLL + SSP-RK3)
  5. 隐式辐射扩散求解 (GMRES + 带状矩阵)
  6. GARCH 随机中微子光度
  7. 蒙特卡罗中微子包输运
  8. 激波几何 + 三角剖分直方图
  9. SASI 极限环 + SSD 模式分解
 10. Jordan 形式稳定性分析
 11. Bayesian 熵产率诊断
 12. Granger 因果检验
 13. 拉格朗日示踪粒子
 14. 全局诊断

所有输出以文本形式打印 (无可视化).
"""
from __future__ import annotations
import math
import sys
import time
import numpy as np

# 本地模块
import constants as C
from mesh import SphericalMesh
from eos import EquationOfState
from opacity import OpacityTable
from radiation_quadrature import DiscreteOrdinateQuadrature, level_symmetric_sn
from hydro import RadiationHydroSolver
from implicit_solver import BandMatrix, RestartedGMRES, ilu0_preconditioner
from shock_geometry import ShockSurface, IcosphereMesh, rankine_hugoniot
from neutrino import (GARCHLuminosity, NeutrinoSpectrum,
                       MonteCarloNeutrinoTransport, neutrino_heating_rate)
from stability import (jordan_block, jordan_decomposition, matrix_exponential,
                        von_neumann_amplification, stability_criterion)
from sasi_dynamics import (VanDerPolOscillator, vanderpol_period_estimate,
                            SpatialSpectralDecomposition)
from causality import bagged_granger_causality
from tracer import TracerParticles, escape_velocity
from bayesian_entropy import (BayesianEntropyProduction, total_entropy,
                                specific_entropy_gas)
from diagnostics import global_diagnostics
from initial import progenitor_profile, _convert, insert_shock


def _banner(title: str):
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def main() -> int:
    t0 = time.time()
    np.random.seed(20260607)
    rng = np.random.default_rng(20260607)

    # =============================================================
    # 1. 物理参数与网格
    # =============================================================
    _banner("1. 球对称网格构造")
    n_cells = 64
    r_inner = 1.0e6        # 10 km
    r_outer = 5.0e8        # 5000 km
    mesh = SphericalMesh(n_cells, r_inner, r_outer, q_ratio=1.05, cfl=0.4)
    print(f"  单元数: {mesh.n_cells},  r ∈ [{mesh.r_inner:.2e}, {mesh.r_outer:.2e}] cm")
    print(f"  最小 dr = {mesh.dr.min():.2e} cm,  最大 dr = {mesh.dr.max():.2e} cm")

    # =============================================================
    # 2. 物态 + 不透明度 + 角求积
    # =============================================================
    _banner("2. 物态、不透明度、离散纵标")
    eos = EquationOfState(gamma_ion=5.0 / 3.0, y_e=0.42)
    opacity = OpacityTable(logT_range=(7.0, 10.5), logRho_range=(2.0, 10.0),
                           n_T=10, n_rho=8)
    quadrature = DiscreteOrdinateQuadrature(n_order=6)
    print(f"  物态: γ={eos.gamma_ion:.3f}, μ={eos.mu:.3f}, Y_e={eos.y_e:.2f}")
    print(f"  不透明度: {opacity.n_T} × {opacity.n_rho} 切比雪夫节点")
    kappa_test = opacity.evaluate(np.array([5.0e9]), np.array([1.0e9]))
    print(f"  κ(T=5e9, ρ=1e9) = {kappa_test[0]:.3e} cm^2/g")
    print(f"  S_N 求积: {quadrature.n_dir} 方向, 权重和 = {quadrature.weights.sum():.4e}")

    sn4 = level_symmetric_sn(4)
    print(f"  S_4 level-symmetric: {sn4.n_dir} 方向")

    # =============================================================
    # 3. 前身星初值
    # =============================================================
    _banner("3. 前身星初始剖面")
    prof = progenitor_profile(mesh.r_centers)
    rho0, T0, v0, P0, Ye0 = prof['rho'], prof['T'], prof['v'], prof['P'], prof['Ye']
    print(f"  ρ ∈ [{rho0.min():.2e}, {rho0.max():.2e}] g/cm^3")
    print(f"  T ∈ [{T0.min():.2e}, {T0.max():.2e}] K")
    print(f"  |v| ∈ [{np.abs(v0).min():.2e}, {np.abs(v0).max():.2e}] cm/s")

    U0 = _convert(mesh, rho0, v0, P0, T0, eos)
    U = insert_shock(U0, mesh, r_shock=1.5e7, mach=4.0, gamma=5.0 / 3.0)
    print(f"  注入激波: r_shock = 1.5e7 cm, M = 4.0")

    # =============================================================
    # 4. 流体求解器 + 显式演化
    # =============================================================
    _banner("4. 辐射流体显式演化 (WENO5 + HLL)")
    hydro = RadiationHydroSolver(mesh, eos, opacity, quadrature,
                                  gravity_M=1.4 * C.M_SUN, cfl=0.3)
    rho, v, P, T = hydro.conservative_to_primitive(U)
    cs = eos.sound_speed(rho, T, P)
    dt = mesh.courant_dt(v, cs)
    print(f"  初始 dt_CFL = {dt:.3e} s")
    # 显式演化几步
    n_step = 3
    for step in range(n_step):
        dU = hydro.rhs_explicit(U)
        U = U + dt * dU
        U = hydro.positivity_limiter(U)
        rho, v, P, T = hydro.conservative_to_primitive(U)
        cs = eos.sound_speed(rho, T, P)
        dt = mesh.courant_dt(v, cs)
    print(f"  完成 {n_step} 步, 当前 dt = {dt:.3e} s")
    print(f"  ρ ∈ [{rho.min():.2e}, {rho.max():.2e}]")
    print(f"  T ∈ [{T.min():.2e}, {T.max():.2e}]")
    R_s = mesh.shock_radius(rho, P)
    print(f"  激波半径 R_s = {R_s:.3e} cm = {R_s / C.KM_TO_CM:.2f} km")

    # =============================================================
    # 5. 隐式辐射扩散 (GMRES + ILU)
    # =============================================================
    _banner("5. 隐式辐射扩散 (GMRES + 带状矩阵)")
    N = mesh.n_cells
    A = BandMatrix(N, kind='tridiag')
    # 构造 (I - Δt L) E_r = E_r^n
    # 简化：对角元 = 1 + Δt χ_a c, 上次/下次对角 = -Δt D / (2 dr^2)
    dt_implicit = 1.0e-4
    chi_a = 0.01 * np.ones(N)
    D_diff = 1.0e20 * np.ones(N)
    dr_min = mesh.dr.min()
    coeff = dt_implicit * D_diff / (dr_min ** 2)
    A.d[:] = 1.0 + dt_implicit * chi_a * C.C_LIGHT + 2.0 * coeff
    A.dl[:] = -coeff[1:]
    A.du[:] = -coeff[1:]
    # 边界
    A.d[0] -= coeff[0]
    A.d[-1] -= coeff[-1]
    rhs = 1.0 + 0.1 * rng.standard_normal(N)
    precond = ilu0_preconditioner(A)
    gmres = RestartedGMRES(max_iter=50, restart=10, tol_abs=1.0e-8, tol_rel=1.0e-8)
    x_sol, info = gmres.solve(A, rhs, precond=precond)
    print(f"  矩阵规模: {N} × {N}")
    print(f"  GMRES 收敛: {info['converged']}, "
          f"迭代 = {info['iters']}, 残差 = {info['final_residual']:.3e}")
    print(f"  解范围: [{x_sol.min():.3f}, {x_sol.max():.3f}]")

    # =============================================================
    # 6. GARCH 中微子光度 + 蒙特卡罗输运
    # =============================================================
    _banner("6. GARCH 随机中微子源 + 蒙特卡罗输运")
    garch = GARCHLuminosity(L0=3.0e52, tau=3.0,
                             omega=1.0e50, alpha=0.15, beta=0.80)
    t_grid = np.linspace(0, 5.0, 200)
    L_nu_seq = garch.generate_sequence(t_grid, rng)
    print(f"  GARCH(1,1) 光度序列: {len(L_nu_seq)} 点")
    print(f"  基础 L_ν(0) = {garch.base_luminosity(0.0):.3e} erg/s")
    print(f"  序列均值 = {L_nu_seq.mean():.3e}, 标准差 = {L_nu_seq.std():.3e}")
    print(f"  序列范围 = [{L_nu_seq.min():.3e}, {L_nu_seq.max():.3e}]")

    mc = MonteCarloNeutrinoTransport(n_packets=100, seed=42)
    r_edges = mesh.r_nodes
    chi_a_arr = 0.001 * np.ones(N)
    chi_s_arr = 0.0002 * np.ones(N)
    result_mc = mc.run(r_edges, chi_a_arr, chi_s_arr, L_inj=1.0e52)
    print(f"  蒙特卡罗: {mc.n_packets} 包")
    print(f"    被吸收 = {result_mc['n_absorbed']}, 逃逸 = {result_mc['n_escaped']}")
    print(f"    沉积分数 = {result_mc['fraction_deposited']:.3f}")

    nusp = NeutrinoSpectrum(T_MeV=5.0)
    print(f"  中微子谱: T_ν = {nusp.T_MeV} MeV, <ε> = {nusp.mean_energy_MeV():.2f} MeV")

    # 中微子加热率
    Q_nu = neutrino_heating_rate(3.0e52, mesh.r_centers, r_gain=1.0e7, eps_avg_MeV=15.0)
    print(f"  中微子加热率: max = {Q_nu.max():.3e} erg/g/s")

    # =============================================================
    # 7. 激波几何 + 三角剖分直方图
    # =============================================================
    _banner("7. 激波曲面几何 + 三角剖分直方图")
    shock = ShockSurface(R0=R_s, delta=0.1)
    shock.perturb(rng, amplitude=0.08)
    theta_g, phi_g, r_g = shock.sample_directions(n_theta=6, n_phi=12)
    print(f"  激波平均半径: {r_g.mean():.3e} cm")
    print(f"  激波半径起伏: {r_g.std() / r_g.mean():.3f}")
    rh_test = rankine_hugoniot(M=5.0, gamma=5.0 / 3.0)
    print(f"  Rankine-Hugoniot (M=5): "
          f"P_2/P_1 = {rh_test['pressure_ratio']:.3f}, "
          f"ρ_2/ρ_1 = {rh_test['density_ratio']:.3f}, "
          f"T_2/T_1 = {rh_test['temperature_ratio']:.3f}")
    mesh_tri = IcosphereMesh(n_theta=4, n_phi=8)
    # 生成采样点
    pts = [(theta_g[i, j], phi_g[i, j])
           for i in range(theta_g.shape[0]) for j in range(theta_g.shape[1])]
    counts = mesh_tri.histogram(pts)
    chi2 = mesh_tri.uniformity_chi2(counts)
    print(f"  三角剖分: {mesh_tri.n_tri} 三角形")
    print(f"  采样直方图: 最大落入 = {counts.max()}, 最小 = {counts.min()}")
    print(f"  χ² 均匀性 = {chi2:.3f}")

    # =============================================================
    # 8. SASI 极限环 + SSD 模式分解
    # =============================================================
    _banner("8. SASI 极限环振荡 + 空间-谱分解")
    mu_sasi = 2.0
    omega_sasi = 2.0 * math.pi / 0.030  # 30 ms 周期
    osc = VanDerPolOscillator(mu=mu_sasi, omega0=omega_sasi,
                               amplitude=0.15, drive_amp=0.02)
    osc.x = 0.05  # 初始位移打破对称
    osc.v = 0.01
    dt_sasi = 1.0e-4
    t_sasi = np.linspace(0, 0.2, 2000)
    x_sasi = osc.integrate(dt_sasi, t_sasi)
    T_pred = vanderpol_period_estimate(mu_sasi, omega_sasi)
    print(f"  Van der Pol μ = {mu_sasi}, ω_0 = {omega_sasi:.2f} rad/s")
    print(f"  Urabe 预测周期 = {T_pred * 1000:.2f} ms")
    print(f"  数值 x(t) 范围 = [{x_sasi.min():.4f}, {x_sasi.max():.4f}]")
    # SSD
    n_chan = 6
    X_active = np.zeros((n_chan, len(t_sasi)))
    for k in range(n_chan):
        X_active[k] = x_sasi * (1.0 + 0.1 * k) + 0.05 * rng.standard_normal(len(t_sasi))
    X_quiet = 0.05 * rng.standard_normal((n_chan, len(t_sasi)))
    ssd = SpatialSpectralDecomposition()
    fit = ssd.fit(X_active, X_quiet)
    print(f"  SSD 模式 (特征值): {fit['eigenvalues'][:3]}")
    f_dom = ssd.dominant_frequency(fit['scores_active'][0], dt_sasi,
                                    f_min=10.0, f_max=100.0)
    print(f"  SASI 主导频率 = {f_dom:.2f} Hz")

    # =============================================================
    # 9. 稳定性分析 (Jordan 形式)
    # =============================================================
    _banner("9. 线性稳定性分析 (Jordan 形式)")
    # 构造一个典型 Jacobian (3x3, 含对流 + 辐射耦合)
    # λ_1 = -0.1 (稳定), λ_2 = 0.05 ± 0.3i (SASI 模式)
    J_test = np.array([
        [-0.1, 0.0, 0.0],
        [0.0, 0.05, -0.3],
        [0.0, 0.3, 0.05]
    ], dtype=np.float64)
    sc = stability_criterion(J_test, dx=1.0e5, cfl_target=0.4)
    print(f"  测试 Jacobian 特征值: {sc['eigenvalues']}")
    print(f"  谱半径 = {sc['spectral_radius']:.4e}")
    print(f"  临界 dt_CFL = {sc['dt_cfl']:.4e} s")
    print(f"  非正规度 = {sc['non_normality']:.4e}")
    # Jordan 块演示
    JB = jordan_block(0.05 + 0.3j, size=2)
    print(f"  Jordan 块 (2×2): {JB[0]}")
    # 矩阵指数
    expJ = matrix_exponential(J_test, t=0.01)
    print(f"  ||exp(J·0.01)|| = {np.linalg.norm(expJ):.4e}")
    # 放大矩阵
    G = von_neumann_amplification(wave_number=1.0e-5, dx=1.0e5, dt=1.0e-4,
                                   jacobian=J_test)
    rho_G = np.max(np.abs(np.linalg.eigvals(G)))
    print(f"  von Neumann 放大矩阵谱半径 = {rho_G:.4f}")

    # =============================================================
    # 10. 贝叶斯熵产诊断
    # =============================================================
    _banner("10. 贝叶斯热力学第二律诊断")
    s_total = total_entropy(P, rho, T)
    dm = rho * mesh.volumes
    S_tot = float(np.sum(s_total * dm))
    print(f"  总熵 S = {S_tot:.4e} erg/K")
    print(f"  单位质量熵 s ∈ [{s_total.min():.3e}, {s_total.max():.3e}] erg/K/g")
    # 贝叶斯更新
    bayes = BayesianEntropyProduction(alpha_0=3.0, beta_0=1.0,
                                       K=1.0e50, sigma_obs=1.0e51)
    Phi_obs = 1.0e52 + 1.0e51 * rng.standard_normal(50)
    bayes.update(Phi_obs)
    mean_S, var_S = bayes.posterior_mean(), bayes.posterior_variance()
    lo, hi = bayes.credible_interval(0.95)
    print(f"  后验 ⟨Σ⟩ = {mean_S:.4e}, σ(Σ) = {math.sqrt(var_S):.4e}")
    print(f"  95% CI = ({lo:.4e}, {hi:.4e})")
    test_2nd = bayes.test_second_law()
    print(f"  z-score = {test_2nd['z_score']:.3f}, p = {test_2nd['p_value']:.4e}")
    print(f"  与第二律一致: {test_2nd['consistent_with_2nd_law']}")

    # =============================================================
    # 11. 因果诊断
    # =============================================================
    _banner("11. 后激波-中微子因果检验 (Bagged Granger)")
    # 构造时间序列：激波脉动 → 中微子光度
    n_t = 100
    shock_series = 0.1 * np.sin(2.0 * math.pi * np.arange(n_t) / 30.0)
    shock_series += 0.05 * rng.standard_normal(n_t)
    # 中微子信号对激波有 2 步延迟
    nu_series = np.zeros(n_t)
    for i in range(2, n_t):
        nu_series[i] = 0.7 * shock_series[i - 2] + 0.05 * rng.standard_normal()
    bag_res = bagged_granger_causality(shock_series, nu_series,
                                        p=3, block_len=8, n_bootstrap=10)
    print(f"  原始 Granger F = {bag_res['original']['F_statistic']:.3f}, "
          f"p = {bag_res['original']['p_value']:.4e}")
    print(f"  Bootstrap 中位 F = {bag_res['F_median']:.3f}")
    print(f"  Bootstrap 95% CI = ({bag_res['F_2.5']:.3f}, {bag_res['F_97.5']:.3f})")

    # =============================================================
    # 12. 拉格朗日示踪粒子
    # =============================================================
    _banner("12. 拉格朗日抛射物示踪粒子")
    tracers = TracerParticles(n_particles=30, seed=123)
    print(f"  初始 r ∈ [{tracers.r.min():.2e}, {tracers.r.max():.2e}] cm")
    n_tracer_step = 5
    for _ in range(n_tracer_step):
        tracers.velocity_verlet_step(
            dt=1.0e-4, r_mesh=mesh.r_centers, v_fluid=v, P=P, rho=rho,
            drag_time=1.0e-3)
    msd = tracers.mean_square_displacement(lag_max=3)
    looping = tracers.looping_path_length()
    esc = tracers.escape_fraction(r_escape=1.0e13)
    print(f"  完成 {n_tracer_step} 步")
    print(f"  最终 r ∈ [{tracers.r.min():.2e}, {tracers.r.max():.2e}] cm")
    print(f"  MSD(Δt=1) = {msd[1]:.3e}, MSD(Δt=2) = {msd[2]:.3e}")
    print(f"  Looping 平均 = {looping.mean():.3f}")
    print(f"  逃逸率 = {esc * 100:.1f} %")
    v_esc_test = escape_velocity(1.4 * C.M_SUN, 1.0e7)
    print(f"  逃逸速度 (r=100 km) = {v_esc_test:.3e} cm/s = {v_esc_test / C.KM_TO_CM:.1f} km/s")

    # =============================================================
    # 13. 全局诊断
    # =============================================================
    _banner("13. 全局诊断")
    diag = global_diagnostics(U, mesh, eos, E_rad_density=None)
    print(f"  引力结合能 E_grav = {diag['E_grav']:.3e} erg = {diag['E_grav'] / 1.0e51:.3f} foe")
    print(f"  动能 E_kin = {diag['E_kin']:.3e} erg = {diag['E_kin'] / 1.0e51:.3f} foe")
    print(f"  内能 E_int = {diag['E_int']:.3e} erg = {diag['E_int'] / 1.0e51:.3f} foe")
    print(f"  总质量 M = {diag['M_total'] / C.M_SUN:.3f} M_sun")
    print(f"  激波半径 = {diag['R_shock'] / C.KM_TO_CM:.2f} km")

    elapsed = time.time() - t0
    _banner("完成")
    print(f"  总运行时间: {elapsed:.3f} s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
