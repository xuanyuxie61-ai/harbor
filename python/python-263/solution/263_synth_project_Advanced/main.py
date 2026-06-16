# -*- coding: utf-8 -*-
"""
main.py - PROJECT 263 统一入口
------------------------------
计算太阳物理: 日冕加热与太阳风加速
高阶有限差分与稳定性分析 (小规模可复现实验)

流程
----
1. 构建 CVT 自适应网格 (日冕环 + 太阳风径向)
2. 初始化日冕等离子体状态
3. 运行纳耀斑弛豫振荡器
4. 计算 Parker 太阳风跨声速解 + committor 概率
5. 评估 MHD 能量原理稳定性
6. 运行多层 HEOM 耦合演化
7. 隐式热传导求解 (GMRES)
8. WKB 基模态分解
9. 磁场拓扑距离 + 加热率量化
10. 全系统 von Neumann 稳定性分析
11. 输出诊断报告
"""
from __future__ import annotations
import sys
import time
import numpy as np

# --- 项目模块 ---
import solar_constants as sc
import coronal_grid as cg
import fd_operators as fdo
import nanoflare_oscillator as nfo
import parker_wind as pw
import coronal_energy as ce
import heom_hierarchy as hh
import coronal_solver as cs
import coronal_basis as cb
import topology_tools as tt
import stability_analysis as sa


def banner(title: str) -> None:
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def phase_header(idx: int, title: str) -> None:
    print()
    print(f"--- Phase {idx:02d}: {title} " + "-" * max(0, 48 - len(title)))


def report_dict(d: dict, indent: int = 4) -> None:
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{pad}{k:<24}: {v:14.6e}")
        else:
            print(f"{pad}{k:<24}: {v}")


# ============================================================
# Phase 1: 自适应网格
# ============================================================
def phase_coronal_grid() -> dict:
    phase_header(1, "Coronal Adaptive Grid (CVT Lloyd Sampling)")
    z_loop = cg.build_corona_loop_grid(
        s_fp=1.0e8, n_gen=64, it_num=60, s_num=4000, seed=263
    )
    z_wind = cg.build_solar_wind_grid(
        r_out=20.0 * sc.SOLAR_RADIUS, n_gen=80, it_num=60,
        s_num=5000, t_corona=1.5e6, seed=263
    )
    q_loop = cg.grid_quality(z_loop)
    q_wind = cg.grid_quality(z_wind)
    print("  Corona loop grid (nodes={})".format(q_loop["n_nodes"]))
    report_dict(q_loop)
    print("  Solar wind grid (nodes={})".format(q_wind["n_nodes"]))
    report_dict(q_wind)
    return dict(z_loop=z_loop, z_wind=z_wind, q_loop=q_loop, q_wind=q_wind)


# ============================================================
# Phase 2: 等离子体初始化
# ============================================================
def phase_plasma_init(z_loop: np.ndarray) -> dict:
    phase_header(2, "Initial Coronal Plasma State")
    n = z_loop.size
    # RTV 标度律: T(s) = T_0 * |cos(pi s / (2 L))|^{2/7}
    L = z_loop.max()
    arg = np.pi * z_loop / (2.0 * L + 1.0e-12)
    T_profile = sc.CORONA_T_BASE * np.abs(np.cos(arg)) ** (2.0 / 7.0)
    T_profile = np.maximum(T_profile, 5.0e4)
    # 静力平衡密度: n(s) ~ n_0 * (T/T_0)^{-1}
    n_profile = sc.CORONA_N_BASE * (sc.CORONA_T_BASE / T_profile)
    n_profile = np.maximum(n_profile, 1.0e13)
    # 磁场: 偶极近似 B(s) = B_0 (R_sun / (R_sun + h(s)))^3
    R_sun = sc.SOLAR_RADIUS
    h_loop = np.sqrt(z_loop**2 + 1.0e16) * 0.01
    B_profile = sc.CORONA_B_BASE * (
        R_sun / (R_sun + h_loop)
    )**3
    v_profile = np.zeros(n)
    p_profile = n_profile * sc.BOLTZMANN * T_profile

    print(f"  T range : [{T_profile.min():.3e}, {T_profile.max():.3e}] K")
    print(f"  n range : [{n_profile.min():.3e}, {n_profile.max():.3e}] m^-3")
    print(f"  B range : [{B_profile.min():.3e}, {B_profile.max():.3e}] T")
    print(f"  beta    : {sc.plasma_beta(n_profile[0], T_profile[0], B_profile[0]):.4f}")
    return dict(T=T_profile, n=n_profile, B=B_profile, v=v_profile, p=p_profile,
                L=L)


# ============================================================
# Phase 3: 纳耀斑振荡
# ============================================================
def phase_nanoflare() -> dict:
    phase_header(3, "Nanoflare Relaxation Oscillation")
    params = nfo.nanoflare_parameters()
    y0 = np.array([0.5, 0.3])
    t_vals, y_vals = nfo.integrate_nanoflare(y0, (0.0, params["t_stop"]),
                                              params["dt"], params)
    events = nfo.detect_nanoflare_events(
        t_vals, y_vals[:, 0], threshold=0.5
    )
    alpha = nfo.power_law_index(events)
    print(f"  t_span        : [0, {t_vals[-1]:.2f}]")
    print(f"  E_mag range   : [{y_vals[:, 0].min():.4f}, {y_vals[:, 0].max():.4f}]")
    print(f"  nanoflares    : {len(events)}")
    print(f"  power-law idx : {alpha:.3f}  (Parker criterion alpha>2)")
    return dict(t_vals=t_vals, y_vals=y_vals, events=events, alpha=alpha)


# ============================================================
# Phase 4: Parker 太阳风 + committor
# ============================================================
def phase_parker_wind() -> dict:
    phase_header(4, "Parker Solar Wind Transonic Solution")
    cs_val = sc.sound_speed(1.5e6)
    r_full, v_full, r_c = pw.critical_solution(cs_val)
    mach = v_full / cs_val
    print(f"  c_s           : {cs_val:.3e} m/s")
    print(f"  r_c           : {r_c/sc.SOLAR_RADIUS:.3f} R_sun")
    print(f"  v(r_out)/c_s  : {v_full[-1]/cs_val:.3f}")
    print(f"  max Mach      : {mach.max():.3f}")

    # Committor (coarse grid)
    r_g = np.linspace(sc.SOLAR_RADIUS, 5.0 * sc.SOLAR_RADIUS, 20)
    v_g = np.linspace(0.3 * cs_val, 2.5 * cs_val, 20)
    q_table = pw.solve_committor(r_g, v_g, cs_val)
    q_at_rc = pw.escape_probability_at_state(r_c, cs_val, cs_val, q_table, r_g, v_g)
    print(f"  q(r_c, c_s)   : {q_at_rc:.4f} (transonic escape prob)")
    return dict(r_full=r_full, v_full=v_full, r_c=r_c, q_table=q_table,
                r_g=r_g, v_g=v_g, mach=mach)


# ============================================================
# Phase 5: 变分能量原理 + MHD 稳定性
# ============================================================
def phase_energy_principle(z_loop: np.ndarray, state: dict) -> dict:
    phase_header(5, "Variational Energy Principle & Kink/Torus")
    rho = state["n"] * sc.PROTON_MASS
    p = state["p"]
    B = state["B"]
    W = ce.total_potential_energy(rho, p, B, z_loop, z_loop)
    evals, evecs = ce.compute_unstable_mode(z_loop, B, p, rho, n_modes=3)
    q_kink = np.array([
        ce.kink_instability_threshold(B[i], 0.1 * B[i], 1.0e8, 1.0e7)
        for i in range(z_loop.size)
    ])
    q_diag = ce.krksll_shaf_local(q_kink)
    print(f"  total W       : {W:.6e} J/m^2")
    print(f"  3 smallest eigenvalues of delta^2 W:")
    for i, lam in enumerate(evals):
        status = "UNSTABLE" if lam < 0 else "stable"
        print(f"    lambda_{i} = {lam:.6e}  [{status}]")
    print(f"  K-S safety q  : min={q_diag['q_min']:.3f}, "
          f"unstable fraction={q_diag['unstable_fraction']:.3f}")
    return dict(evals=evals, evecs=evecs, W=W, q_diag=q_diag)


# ============================================================
# Phase 6: 多层 HEOM 耦合
# ============================================================
def phase_heom() -> dict:
    phase_header(6, "HEOM Multi-layer Atmosphere Coupling")
    hier = hh.HEOMHierarchy()
    hist = hier.run(t_total=500.0, dt=5.0)
    summary = hier.summary()
    for layer_name, info in summary.items():
        print(f"  [{layer_name}]")
        for k, v in info.items():
            print(f"      {k:<10}: {v:.4e}")
    print(f"  evolve steps recorded: {len(hist)}")
    return dict(summary=summary, history=hist)


# ============================================================
# Phase 7: 隐式热传导 (GMRES)
# ============================================================
def phase_implicit_heat(z_loop: np.ndarray, T0: np.ndarray) -> dict:
    phase_header(7, "Implicit Heat Conduction (Restarted GMRES)")
    kappa = np.array([sc.spitzer_conductivity(T) for T in T0])
    dt = 10.0
    # 多步推进
    T = T0.copy()
    res_norms = []
    for step in range(5):
        T_new = cs.implicit_heat_step(T, dt, theta=0.5,
                                      z_grid=z_loop, kappa=kappa)
        res = np.linalg.norm(T_new - T)
        res_norms.append(res)
        T = T_new
    print(f"  T_max after 5 implicit steps: {T.max():.3e} K")
    print(f"  T_min after 5 implicit steps: {T.min():.3e} K")
    print(f"  residual norms: {[f'{r:.3e}' for r in res_norms]}")
    return dict(T_final=T, residuals=res_norms)


# ============================================================
# Phase 8: WKB 基分解
# ============================================================
def phase_wkb_modes(z_loop: np.ndarray, B: np.ndarray) -> dict:
    phase_header(8, "WKB Basis Modal Decomposition")
    n_i = 1.0e15
    v_A = np.array([sc.alven_speed(b, n_i) for b in B])
    omega = 0.1  # 特征频率
    n_basis = 6
    basis = cb.wkb_basis(z_loop, v_A, omega, n_basis)
    # 合成一个测试场
    u_test = np.sin(np.pi * z_loop / z_loop.max()) * np.cos(omega * 0.0)
    coef = cb.modal_decomposition(u_test, basis)
    u_recon = cb.reconstuct_from_modes(coef, basis)
    err = np.linalg.norm(u_recon - u_test) / (np.linalg.norm(u_test) + 1.0e-12)
    print(f"  omega         : {omega:.3e}")
    print(f"  n_basis       : {n_basis}")
    print(f"  coef          : [{', '.join(f'{c:.4f}' for c in coef)}]")
    print(f"  relative err  : {err:.6e}")
    return dict(coef=coef, error=err)


# ============================================================
# Phase 9: 拓扑 + 量化
# ============================================================
def phase_topology(z_loop: np.ndarray) -> dict:
    phase_header(9, "Magnetic Topology & Heating Quantization")
    # 简单偶极场
    def b_func(x, y, z):
        r2 = x**2 + y**2 + z**2 + 1.0e20
        return (3.0 * x * z / r2**2.5, 3.0 * y * z / r2**2.5,
                (2 * z**2 - x**2 - y**2) / r2**2.5)
    x_grid = np.linspace(-1.0e7, 1.0e7, 8)
    y_grid = np.zeros(8)
    D = tt.connectivity_matrix(x_grid, y_grid, b_func,
                               z0=1.0e6, z_end=5.0e7, dz=1.0e6)
    print(f"  topology matrix shape: {D.shape}")
    print(f"  mean topo distance   : {D.mean():.4e}")

    # 加热率量化
    Q_field = cg.coronal_heating_profile(z_loop, z_loop.max(), z_loop.max())
    Q_quant, centers, labels = tt.quantize_heating_rate(Q_field, n_levels=6)
    grad_Q = tt.heating_gradient_identification(Q_quant, z_loop)
    dom = tt.dominant_heating_region(Q_quant, z_loop)
    print(f"  quantization centers  : {[f'{c:.4f}' for c in centers]}")
    print(f"  dominant region       : z in [{dom['z_start']:.3e}, {dom['z_end']:.3e}]")
    print(f"  dominant fraction     : {dom['fraction']:.3f}")
    return dict(D=D, Q_quant=Q_quant, centers=centers, dominant=dom)


# ============================================================
# Phase 10: 稳定性诊断
# ============================================================
def phase_stability(z_loop: np.ndarray, state: dict) -> dict:
    phase_header(10, "von Neumann & CFL Stability Analysis")
    stab = sa.stability_summary(
        z_loop, state["v"], state["B"], state["T"], state["n"], dt=1.0
    )
    report_dict(stab)

    # von Neumann 单模式
    vna = sa.von_neumann_analysis_fd4(
        v_flow=1.0e5, h=cg.grid_quality(z_loop)["h_min"],
        kappa=sc.spitzer_conductivity(state["T"].max()), dt=0.5
    )
    print(f"  von Neumann stable  : {vna['stable']}")
    print(f"  max |G|             : {vna['max_amplification']:.6e}")

    mwn = sa.modified_wavenumber_analysis(cg.grid_quality(z_loop)["h_min"])
    print(f"  dispersion (2nd)    : {mwn['dispersion_2']:.4f}")
    print(f"  dispersion (4th)    : {mwn['dispersion_4']:.4f}")
    print(f"  dispersion (compact): {mwn['dispersion_compact']:.4f}")
    return dict(stab=stab, vna=vna, mwn=mwn)


# ============================================================
# Phase 11: FD 算子 + 下三角求解器 + 带状矩阵演示
# ============================================================
def phase_fd_operators(z_loop: np.ndarray) -> dict:
    phase_header(11, "High-order FD Operators & R8LT Solver")
    D1 = fdo.derivative_matrix_4th(z_loop)
    kappa = np.ones_like(z_loop) * 1.0e6
    L = fdo.diffusion_matrix(z_loop, kappa)
    bih = fdo.biharmonic_matrix(z_loop)
    print(f"  D1 nnz            : {D1.nnz}")
    print(f"  L  nnz            : {L.nnz}")
    print(f"  biharmonic nnz    : {bih.nnz}")

    # R8LT 下三角求解演示
    n = z_loop.size
    diag = 2.0 * np.ones(n)
    sub = -1.0 * np.ones(n - 1)
    Ltri = cs.R8LT(diag, sub)
    b = np.random.default_rng(263).normal(size=n)
    x_sol = Ltri.solve(b)
    err = np.linalg.norm(Ltri.matvec(x_sol) - b)
    print(f"  R8LT solve err    : {err:.4e}")
    print(f"  det(L)            : {Ltri.determinant():.4e}")

    # 带状矩阵
    band = fdo.to_band_storage(L, kl=1, ku=1)
    print(f"  band storage shape: {band.shape}")

    # 人工粘性
    v = np.sin(np.linspace(0, np.pi, n)) * 1.0e5
    q = fdo.artificial_viscosity(v, z_loop)
    print(f"  art. viscosity max: {q.max():.4e}")
    return dict(D1=D1, L=L, bih=bih, r8lt_err=err)


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    np.seterr(all="warn", over="ignore", under="ignore")
    banner("PROJECT 263 - Coronal Heating & Solar Wind Acceleration")
    print("  High-Order Finite Differences with Stability Analysis")
    print(f"  Python {sys.version.split()[0]}, NumPy {np.__version__}")
    print(f"  Random seed: 263")
    t_start = time.time()

    # Phase 1
    grid = phase_coronal_grid()
    z_loop = grid["z_loop"]
    z_wind = grid["z_wind"]

    # Phase 2
    state = phase_plasma_init(z_loop)

    # Phase 3
    nfo_results = phase_nanoflare()

    # Phase 4
    pw_results = phase_parker_wind()

    # Phase 5
    energy = phase_energy_principle(z_loop, state)

    # Phase 6
    heom = phase_heom()

    # Phase 7
    heat = phase_implicit_heat(z_loop, state["T"])

    # Phase 8
    wkb = phase_wkb_modes(z_loop, state["B"])

    # Phase 9
    topo = phase_topology(z_loop)

    # Phase 10
    stab = phase_stability(z_loop, state)

    # Phase 11
    fdo_r = phase_fd_operators(z_loop)

    # 总览
    banner("Final Summary")
    print(f"  Corona loop nodes    : {grid['q_loop']['n_nodes']}")
    print(f"  Solar wind nodes     : {grid['q_wind']['n_nodes']}")
    print(f"  Nanoflare events     : {len(nfo_results['events'])}")
    print(f"  Power-law index      : {nfo_results['alpha']:.3f}")
    print(f"  Parker r_c / R_sun   : {pw_results['r_c']/sc.SOLAR_RADIUS:.3f}")
    print(f"  WKB reconstruction   : err={wkb['error']:.3e}")
    print(f"  Dominant heating     : {topo['dominant']['fraction']*100:.1f}%")
    print(f"  CFL stable           : {stab['stab']['cfl_stable']}")
    print(f"  von Neumann stable   : {stab['vna']['stable']}")
    print(f"  R8LT solver err      : {fdo_r['r8lt_err']:.3e}")
    print(f"  Wall time            : {time.time()-t_start:.2f} s")
    print("=" * 64)
    print("  PROJECT 263 COMPLETED SUCCESSFULLY")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
