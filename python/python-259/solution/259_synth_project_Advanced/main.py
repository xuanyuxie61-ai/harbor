# -*- coding: utf-8 -*-
"""
main.py
=======

统一入口: 计算宇宙学 — 重子声学振荡参数拟合
         高阶有限差分与稳定性分析 (小规模可复现实验)

本脚本把全部 14 个子模块串联成一条完整的 BAO 分析流水线:

  Phase 0. 初始化宇宙学参数与物理常数
         [bao_constants.py]

  Phase 1. 生成共动坐标网格 + 区域编号 (numgrid 风格)
         [comoving_grid.py]

  Phase 2. 计算背景演化: H(z), χ(z), D_A(z), D(z), f(z)
         [background_cosmology.py]

  Phase 3. 计算声视界 r_d 与拖拽红移 z_d
         [sound_horizon.py]

  Phase 4. 构造 mock 星系 survey + 随机 catalog
         [mock_galaxy_survey.py]

  Phase 5. 用 Landy-Szalay 估计器计算 ξ(s) 与多极矩 ξ_0(s)
         [correlation_landy_szalay.py]

  Phase 6. 用高阶有限差分求解重子-光子流体声学振荡 PDE
         [acoustic_wave_pde.py + fornberg_fd.py]

  Phase 7. Von Neumann 稳定性分析 + CFL 临界数
         [von_neumann_stability.py]

  Phase 8. 用 Bezier 曲面重建窗函数 W(s_⊥, s_∥)
         [bezier_window.py]

  Phase 9. 用双层 LM 拟合器拟合 (ω_b, ω_m, h)
         [chi2_fitter_bilevel.py + bao_observables.py]

  Phase 10. 稀疏协方差 + 精度矩阵构造
         [sparse_covariance.py]

  Phase 11. Hankel 变换验证 (ξ ↔ P)
         [spherical_hankel.py]

  Phase 12. 多项式基 (Legendre, Chebyshev, Collatz) 自检
         [polynomial_basis_toolkit.py]

每个 Phase 都打印关键数值结果, 使整个流程可复现、可审计.
本脚本不依赖任何可视化库, 仅用标准库 + numpy.

运行方式
--------
    python main.py

零参数即可执行.
"""

from __future__ import annotations
import math
import sys
import time
from typing import Dict, List

import numpy as np

# ─────────── 项目内部模块 ───────────
from bao_constants import (
    FiducialCosmology, get_fiducial, get_priors,
    OMEGA_GAMMA_H2, OMEGA_R_H2,
)
from background_cosmology import (
    H_z, E_z, comoving_distance, transverse_comoving_distance,
    angular_diameter_distance, D_H, D_V, Omega_m_at_z,
    growth_D, growth_f, growth_factor_array, bao_geometry,
)
from fornberg_fd import (
    fornberg_weights, build_derivative_matrix_1d,
)
from acoustic_wave_pde import (
    AcousticSolverConfig, AcousticWaveSolver,
    sound_horizon_from_field, sound_speed_over_c,
    baryon_photon_ratio,
)
from von_neumann_stability import (
    find_cfl_critical, full_stability_diagnosis,
    symbol_second_derivative, amplification_imex,
)
from sound_horizon import (
    drag_redshift_eisenstein_hu, sound_horizon_numerical,
    sound_horizon_analytic, rd_gradient,
)
from bao_observables import (
    get_bao_observations, predict_bao, chi2_bao_full,
)
from correlation_landy_szalay import (
    xi_of_s, xi_multipole, locate_bao_peak,
)
from mock_galaxy_survey import (
    sample_galaxies_from_box, generate_random_catalog,
    sigma8_squared, normalize_P,
)
from sparse_covariance import (
    CovarianceConfig, build_covariance_matrix,
    build_sparse_graph, incidence_to_precision,
    matrix_condition_report,
)
from bezier_window import (
    bernstein, bezier_patch_eval, reconstruct_bao_window,
    grid_polar, grid_triangular,
)
from polynomial_basis_toolkit import (
    legendre_P, chebyshev_T, lagrange_interpolant,
    collatz_poly_sequence, horner_eval,
)
from spherical_hankel import (
    j0, jl, besselzero_j0, besselzero_jl,
    hankel_transform_xi_to_Pk, hankel_transform_Pk_to_xi,
)
from comoving_grid import (
    numgrid_bao, default_bao_box, closest_pair_brute,
    grid_density_check, adaptive_mesh_1d,
)
from chi2_fitter_bilevel import (
    FitConfig, run_full_fit, theta_to_cosmo,
)


# =============================================================
# 辅助: 打印分隔
# =============================================================
def _banner(phase: int, title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  Phase {phase:02d} :: {title}")
    print(bar)


# =============================================================
# Phase 0
# =============================================================
def phase_00_init() -> FiducialCosmology:
    _banner(0, "初始化物理常数与基准宇宙学")
    cosmo = get_fiducial()
    print(f"  h            = {cosmo.h:.4f}")
    print(f"  ω_b          = {cosmo.omega_b:.5f}   (Ω_b = {cosmo.Omega_b:.5f})")
    print(f"  ω_m          = {cosmo.omega_m:.5f}   (Ω_m = {cosmo.Omega_m:.5f})")
    print(f"  Ω_Λ          = {cosmo.Omega_Lambda:.5f}")
    print(f"  Ω_r          = {cosmo.Omega_r:.5e}")
    print(f"  n_s          = {cosmo.n_s:.4f}")
    print(f"  σ_8          = {cosmo.sigma8:.4f}")
    print(f"  w_0, w_a     = {cosmo.w0:.3f}, {cosmo.wa:.3f}")
    print(f"  H0           = {cosmo.H0_km_s_Mpc:.3f} km/s/Mpc")
    print(f"  D_H0         = {cosmo.H0_inv_Mpc:.3f} Mpc")
    return cosmo


# =============================================================
# Phase 1
# =============================================================
def phase_01_grid() -> None:
    _banner(1, "生成共动坐标网格 + 区域编号")
    for region in ["S", "L", "C", "F", "D"]:
        G, X, Y = numgrid_bao(region, 16)
        n_int = int((G > 0).sum())
        print(f"  region='{region}': 内部点 {n_int}")
    box = default_bao_box(L=500.0, N=32)
    print(f"  共动盒子: L={box.L:.1f} Mpc/h, N={box.N}, dx={box.dx:.3f} Mpc/h")
    # 自适应网格
    x_adapt = adaptive_mesh_1d(0.0, 300.0, 20,
                               refinement=[(100.0, 180.0)])
    print(f"  自适应网格 (BAO 峰加密): {len(x_adapt)} 个点")


# =============================================================
# Phase 2
# =============================================================
def phase_02_background(cosmo: FiducialCosmology) -> None:
    _banner(2, "背景宇宙学: H(z), χ(z), D(z)")
    z_arr, D_arr, f_arr = growth_factor_array(cosmo, n_grid=64)
    print(f"  z 采样: {len(z_arr)} 点, 范围 [{z_arr.min():.3f}, {z_arr.max():.3f}]")
    for z in [0.0, 0.38, 0.51, 0.85, 1.48, 2.5]:
        Hz = H_z(cosmo, z)
        chi = comoving_distance(cosmo, z)
        DM = transverse_comoving_distance(cosmo, z)
        DH = D_H(cosmo, z)
        DV = D_V(cosmo, z)
        Omz = Omega_m_at_z(cosmo, z)
        Dz = growth_D(cosmo, z)
        fz = growth_f(cosmo, z)
        print(f"  z={z:4.2f}: H={Hz:7.2f} km/s/Mpc, χ={chi:7.2f} Mpc, "
              f"D_M={DM:7.2f}, D_H={DH:7.2f}, D_V={DV:7.2f}, "
              f"Ω_m(z)={Omz:.3f}, D(z)={Dz:.4f}, f(z)={fz:.4f}")
    bao = bao_geometry(cosmo, z=0.51, rd_fid=147.78)
    print(f"  BAO @ z=0.51: DM/r_d={bao.DM_over_rd:.4f}, "
          f"DH/r_d={bao.DH_over_rd:.4f}, DV/r_d={bao.DV_over_rd:.4f}")


# =============================================================
# Phase 3
# =============================================================
def phase_03_sound_horizon(cosmo: FiducialCosmology) -> float:
    _banner(3, "声视界 r_d 与拖拽红移 z_d")
    z_d = drag_redshift_eisenstein_hu(cosmo)
    r_analytic = sound_horizon_analytic(cosmo)
    r_numeric = sound_horizon_numerical(cosmo, z_d=z_d)
    print(f"  z_d (Eisenstein-Hu 拟合) = {z_d:.3f}")
    print(f"  r_d 解析拟合 (EH98 Eq.6) = {r_analytic:.4f} Mpc")
    print(f"  r_d 数值积分 (Gauss-Legendre 96) = {r_numeric:.4f} Mpc")
    print(f"  |r_d,num - r_d,ana| = {abs(r_numeric - r_analytic):.4f} Mpc")
    grad = rd_gradient(cosmo)
    print(f"  ∂r_d/∂ω_b = {grad['omega_b']:.4f} Mpc")
    print(f"  ∂r_d/∂ω_m = {grad['omega_m']:.4f} Mpc")
    return r_numeric


# =============================================================
# Phase 4
# =============================================================
def phase_04_mock_survey(cosmo: FiducialCosmology
                         ):
    _banner(4, "生成 mock 星系巡天 + 随机 catalog")
    s82 = sigma8_squared(cosmo)
    A = normalize_P(cosmo)
    print(f"  σ_8^2 (未归一) = {s82:.4f}")
    print(f"  归一化因子 A = {A:.4e}")
    data = sample_galaxies_from_box(
        L_box=400.0, n_galaxy=150, cosmo=cosmo,
        z_center=0.51, dz=0.10, bias=2.0, seed=0
    )
    randoms = generate_random_catalog(
        L_box=400.0, n_random=300,
        z_center=0.51, dz=0.10, tracer="LRG", seed=1,
    )
    print(f"  生成 {len(data)} 个星系 Agent, {len(randoms)} 个随机点")
    return data, randoms


# =============================================================
# Phase 5
# =============================================================
def phase_05_correlation(data, randoms):
    _banner(5, "Landy-Szalay 相关函数估计 + 多极矩")
    s_edges = np.linspace(60.0, 240.0, 15)
    s_c, xi_0 = xi_of_s(data, randoms, s_edges)
    s_peak, xi_peak, xi_dd = locate_bao_peak(s_c, xi_0, s_range=(100.0, 200.0))
    print(f"  ξ_0(s) 在 {len(s_c)} 个 s bin 上估计")
    print(f"  BAO 峰位置 s_peak = {s_peak:.2f} Mpc")
    print(f"  ξ_0(s_peak) = {xi_peak:.4f}")
    print(f"  ξ_0''(s_peak) = {xi_dd:.4e}")
    # 多极矩 (简化: 仅单极)
    s_c2, xi_2 = xi_multipole(data, randoms, s_edges, ell=2)
    print(f"  四极矩 ξ_2(s_peak) ≈ {xi_2[s_c2.searchsorted(s_peak)]:.4f}" if s_peak > 0 else "")
    return s_c, xi_0, s_edges


# =============================================================
# Phase 6
# =============================================================
def phase_06_acoustic_pde(cosmo: FiducialCosmology):
    _banner(6, "重子-光子声学 PDE 的高阶有限差分离散求解")
    cfg = AcousticSolverConfig(N_z=48, N_chi=40, chi_max=400.0, stencil=5)
    solver = AcousticWaveSolver(cosmo, cfg)
    t0 = time.time()
    field = solver.run()
    dt = time.time() - t0
    r_d_num = sound_horizon_from_field(field, cosmo)
    idx_150 = field.chi.searchsorted(150.0)
    idx_150 = min(max(idx_150, 0), len(field.chi) - 1)
    print(f"  计算耗时: {dt:.3f} s")
    print(f"  声视界 r_d (PDE 数值积分) = {r_d_num:.2f} Mpc")
    print(f"  δ_γ(z=0, χ=150 Mpc) = {field.delta[-1, idx_150]:+.4e}")
    print(f"  红移网格: {len(field.z)} 点, z ∈ [{field.z.min():.1f}, {field.z.max():.1f}]")
    print(f"  共动网格: {len(field.chi)} 点, χ ∈ [{field.chi[0]:.1f}, {field.chi[-1]:.1f}] Mpc")
    # Fornberg 差分矩阵
    D2 = build_derivative_matrix_1d(field.chi, deriv=2, stencil=5)
    print(f"  ∂^2/∂χ^2 差分矩阵形状: {D2.shape}, 非零元素: {np.count_nonzero(np.abs(D2) > 1e-12)}")
    return field, r_d_num


# =============================================================
# Phase 7
# =============================================================
def phase_07_von_neumann():
    _banner(7, "Von Neumann 稳定性分析")
    for scheme in ["explicit", "CN", "IMEX"]:
        cfl = find_cfl_critical(scheme)
        print(f"  {scheme:8s} 临界 CFL = {cfl:.5f}")
    report = full_stability_diagnosis(dx=1.0, cs=1.0 / math.sqrt(3.0),
                                      Hc=0.01, stencil=5)
    print(f"  IMEX 完整诊断: scheme={report.scheme}, stencil={report.stencil}")
    print(f"  CFL_critical = {report.cfl_critical:.5f}")
    print(f"  ρ(CFL=1) = {report.rho_at_cfl1:.5f}")
    print(f"  稳定 @ CFL=1? {report.is_stable_at_cfl1}")
    # 差分算子符号
    k_arr, lambda_k = symbol_second_derivative(dx=1.0, stencil=5, n_samples=64)
    print(f"  ∂^2/∂χ^2 符号: λ_max = {float(np.max(lambda_k)):.4e}, "
          f"λ_min = {float(np.min(lambda_k)):.4e}")


# =============================================================
# Phase 8
# =============================================================
def phase_08_bezier_window():
    _banner(8, "Bezier 曲面重建 BAO 窗函数")
    s1 = np.linspace(50.0, 250.0, 14)
    s2 = np.linspace(50.0, 250.0, 14)
    W_true = np.outer(
        np.exp(-0.5 * ((s1 - 150.0) / 60.0) ** 2),
        np.exp(-0.5 * ((s2 - 150.0) / 60.0) ** 2)
    )
    W_fit = reconstruct_bao_window(s1, s2, W_true)
    err = float(np.max(np.abs(W_fit - W_true)))
    print(f"  Bezier 窗函数重建最大误差 = {err:.4e}")
    # 极坐标 / 三角网格
    sp_pol, spar_pol = grid_polar(200.0, 10, 12)
    sp_tri, spar_tri = grid_triangular(50.0, 250.0, 10)
    print(f"  极坐标网格: {len(sp_pol)} 点")
    print(f"  三角网格: {len(sp_tri)} 点")
    # Bernstein 基
    for u in [0.0, 0.5, 1.0]:
        s = sum(bernstein(3, i, u) for i in range(4))
        print(f"  Bernstein Σ_i B_i^3({u}) = {s:.6f} (应为 1)")


# =============================================================
# Phase 9
# =============================================================
def phase_09_chi2_fit(cosmo: FiducialCosmology) -> Dict[str, float]:
    _banner(9, "双层 χ² LM 拟合 (ω_b, ω_m, h)")
    theta_init = {"omega_b": 0.0228, "omega_m": 0.1440, "h": 0.69}
    print(f"  初值: {theta_init}")
    theta_best, chi2_best, hist = run_full_fit(theta_init)
    print(f"  最优: {theta_best}")
    print(f"  χ²(θ_best) = {chi2_best:.4f}")
    print(f"  χ² 迭代: {[f'{c:.3f}' for c in hist[:8]]}...")
    # 各壳层分解
    obs = get_bao_observations()
    _, per_shell = chi2_bao_full(theta_to_cosmo(theta_best), obs)
    for name, c2 in per_shell.items():
        print(f"    {name:15s}: χ² = {c2:.4f}")
    return theta_best


# =============================================================
# Phase 10
# =============================================================
def phase_10_sparse_covariance():
    _banner(10, "稀疏协方差 + 精度矩阵构造")
    s_bins = np.linspace(60.0, 200.0, 13)
    cfg = CovarianceConfig(s_bins=s_bins, volume=5.0e9, n_k=48)
    C = build_covariance_matrix(cfg)
    rep = matrix_condition_report(C)
    print(f"  协方差矩阵维度: {C.shape}")
    print(f"  λ_min = {rep['lambda_min']:.4e}")
    print(f"  λ_max = {rep['lambda_max']:.4e}")
    print(f"  条件数 = {rep['condition']:.4e}")
    G = build_sparse_graph(s_bins, correlation_length=40.0)
    P = incidence_to_precision(G, regularization=0.01)
    print(f"  精度矩阵维度: {P.shape}")
    rep_P = matrix_condition_report(P)
    print(f"  精度矩阵条件数 = {rep_P['condition']:.4e}")


# =============================================================
# Phase 11
# =============================================================
def phase_11_hankel():
    _banner(11, "Hankel 变换 ξ ↔ P 自洽性检验")
    s_arr = np.linspace(10.0, 300.0, 64)
    xi_test = np.exp(-0.5 * ((s_arr - 150.0) / 20.0) ** 2)
    k_arr = np.geomspace(0.005, 0.5, 64)
    Pk = hankel_transform_xi_to_Pk(s_arr, xi_test, 0, k_arr)
    xi_recon = hankel_transform_Pk_to_xi(k_arr, Pk, 0, s_arr)
    err = float(np.max(np.abs(xi_recon - xi_test)))
    print(f"  Hankel 自洽误差 = {err:.4e}")
    # Bessel 零点
    j0_zeros = besselzero_j0(5)
    print(f"  j_0 前 5 个零点: {[f'{z:.4f}' for z in j0_zeros]}")
    j2_zeros = besselzero_jl(2, 4)
    print(f"  j_2 前 4 个零点: {[f'{z:.4f}' for z in j2_zeros]}")


# =============================================================
# Phase 12
# =============================================================
def phase_12_polynomial():
    _banner(12, "多项式基函数工具箱自检")
    for ell in [0, 1, 2, 3, 4]:
        val = legendre_P(ell, 0.5)
        print(f"  P_{ell}(0.5) = {val:.6f}")
    print(f"  T_5(0.5) = {chebyshev_T(5, 0.5):.6f}")
    x_nodes = np.array([0.0, 0.5, 1.0, 1.5])
    y_vals = np.array([1.0, 1.25, 2.0, 3.25])
    print(f"  Lagrange interp @ 0.7 = {lagrange_interpolant(x_nodes, y_vals, 0.7):.6f}")
    print(f"  Horner eval [1,2,3] @ 2 = {horner_eval(np.array([1.0,2.0,3.0]), 2.0):.1f}")
    seq = collatz_poly_sequence([1, 0, 1], max_steps=15)
    print(f"  Collatz 多项式序列长度: {len(seq)}")


# =============================================================
# 汇总报告
# =============================================================
def final_summary(cosmo: FiducialCosmology, theta_best: Dict[str, float],
                  rd_num: float) -> None:
    _banner(99, "最终合成报告")
    print("  [核心物理结果]")
    print(f"    基准声视界 r_d = {rd_num:.4f} Mpc")
    z_d = drag_redshift_eisenstein_hu(cosmo)
    print(f"    拖拽红移 z_d   = {z_d:.2f}")
    print(f"    最佳拟合参数:")
    for k, v in theta_best.items():
        print(f"      {k:10s} = {v:.6f}")
    print()
    print("  [种子项目贡献清单 (15/15)]")
    contributions = [
        ("1137 DRL-for-Personalized-Energy-Trading",
         "参数管理/梯度累积/BPTT → 拟合器 + 参数先验"),
        ("665 legendre_rule",
         "Gauss-Legendre 求积 → 红移积分 + 声视界积分"),
        ("300 disk01_integrands",
         "cos_power_int 递归积分 → 分段加密积分策略"),
        ("1299 AVSim",
         "Agent-based 模拟器 → 星系 survey mock"),
        ("492 gridlines",
         "极/三角网格 → survey 几何非规则采样"),
        ("198 collatz_polynomial",
         "Collatz 多项式 → 红移 bin 奇偶加权 + 迭代诊断"),
        ("515 helmholtz_exact",
         "Bessel 零点 + Helmholtz 精确解 → ξ 峰模板 + 声学初条件"),
        ("1405 web_matrix",
         "关联图 → 转移矩阵 → 协方差稀疏化 + 精度矩阵"),
        ("1135 bilevel-optim",
         "双层优化 + 字典学习稀疏约束 → LM + 正则化精度矩阵"),
        ("1080 safe-CCMPC-in-CARLA",
         "安全 CBF → 物理边界投影 + 可行域保证"),
        ("1205 Bayesian-Second-Law",
         "Fokker-Planck IMEX → 声学 PDE 隐式-显式推进"),
        ("990 r8poly",
         "多项式代数 (Legendre/Chebyshev/Lagrange) → 插值基工具箱"),
        ("190 closest_pair_brute",
         "暴力最近邻 → 网格密集性检查 + RR/DD 计数基准"),
        ("820 numgrid",
         "区域编号 → survey footprint 离散化 + 索引"),
        ("083 bezier_surface",
         "Bezier patch + 邻接 → 窗函数重建 + patch 连续性"),
    ]
    for i, (name, contrib) in enumerate(contributions, 1):
        print(f"    [{i:2d}] {name[:50]:50s}")
        print(f"         → {contrib}")
    print()
    print("  [可复现性]")
    print(f"    种子: 星系 mock seed=0, 随机 catalog seed=1")
    print(f"    红移壳层: 0.38, 0.51, 0.85, 1.48")
    print(f"    网格: 声学 PDE (N_z=48, N_chi=40), 共动盒子 (L=500, N=32)")


# =============================================================
# main
# =============================================================
def main() -> int:
    t_start = time.time()
    print("=" * 72)
    print(" PROJECT 259 : 计算宇宙学 — 重子声学振荡参数拟合")
    print("             高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 72)

    cosmo = phase_00_init()
    phase_01_grid()
    phase_02_background(cosmo)
    r_d = phase_03_sound_horizon(cosmo)
    data, randoms = phase_04_mock_survey(cosmo)
    phase_05_correlation(data, randoms)
    phase_06_acoustic_pde(cosmo)
    phase_07_von_neumann()
    phase_08_bezier_window()
    theta_best = phase_09_chi2_fit(cosmo)
    phase_10_sparse_covariance()
    phase_11_hankel()
    phase_12_polynomial()
    final_summary(cosmo, theta_best, r_d)

    t_total = time.time() - t_start
    print(f"\n[SUCCESS] 完整流水线执行完毕, 总耗时 {t_total:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
