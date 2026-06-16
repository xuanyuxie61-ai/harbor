"""
main.py
=======

PROJECT_229 : 计算高能物理 — 探测器响应矩阵与 unfolding 反演
                  高阶有限差分与稳定性分析 (小规模可复现实验)

统一入口, 零参数运行.

流程:
  1. 构造相空间网格 → 能量 bin 划分
  2. 生成真实能谱 (多模型)
  3. 组装探测器响应矩阵 (Wathen 风格有限元组装)
  4. 构造迁移图 (有向图 → Markov 转移矩阵)
  5. 生成观测数据 (R·T + 噪声)
  6. 执行多种 unfolding:
       - SVD 截断
       - D'Agostini 迭代贝叶斯
       - Tikhonov 正则化 (黄金分割选 λ)
       - 物理约束 PINN
  7. 稳定性分析:
       - 条件数, 有效秩
       - L-curve 与角点
       - FD 光滑度指标
       - 偏差-方差 FOM
  8. 输出综合报告
"""

from __future__ import annotations
import math
import sys
import os
import time

# 确保包导入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from special_functions import r8_gamma, r8_erf, r8_besk0, r8_besi0, r8_betai
from phase_space_grid import (
    ball_grid_3d, ball_grid_count, momentum_to_observables,
    acceptance_mask, gauss_legendre_cos_theta,
)
from energy_loss_ode import (
    EnergyLossParams, compute_energy_loss_trajectory,
    polar_ode_benchmark,
)
from response_matrix import (
    CSRMatrix, assemble_response_matrix,
    migration_adjacency, adjacency_to_transition,
)
from bin_integration import (
    gauss_laguerre_rule, gen_laguerre_exactness,
    triangle_exactness, spectral_moments,
)
from detector_mesh import (
    DetectorMesh, mesh_enrichment_for_response,
    mesh_quality_aspect_ratio,
)
from finite_difference import (
    fd_derivative, fd_amplification_matrix,
)
from unfolding_methods import (
    svd_unfold, dAgostini_unfold, PhysicsInformedUnfolder,
)
from regularization import (
    golden_section_search, tikhonov_unfold,
    compute_l_curve, l_curve_corner, optimal_lambda_by_golden,
)
from physics_models import (
    power_law_spectrum, thermal_spectrum,
    synchrotron_spectrum, polar_ode_spectrum,
    apply_response, add_poisson_noise,
    integrate_spectrum_in_bins, chi_squared,
)
from stability_analysis import (
    condition_number_from_singulars, effective_rank,
    hat_matrix_svd, hat_matrix_tikhonov,
    propagate_variance, figure_of_merit,
    fd_smoothness, stability_report,
)


# ===========================================================================
#                   工具函数
# ===========================================================================

def _section(title: str, char: str = "="):
    bar = char * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _subsection(title: str):
    print(f"\n--- {title} ---")


def _vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def _relative_error(a, b):
    na = _vec_norm(a)
    if na < 1e-300:
        na = 1e-300
    diff = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))
    return diff / na


# ===========================================================================
#                   主流程
# ===========================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  PROJECT_229: 计算高能物理")
    print("  探测器响应矩阵与 unfolding 反演")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 70)

    # =================================================================
    #  1. 相空间网格与能量 bin 构造
    # =================================================================
    _section("1. 相空间网格与能量 bin 构造")

    # 动量空间球网格 (来自 067_ball_grid)
    radius = 50.0  # GeV
    n_per_axis = 6
    pts = ball_grid_3d(radius, n_per_axis)
    n_expected = ball_grid_count(radius, n_per_axis)
    print(f"  动量空间网格: 半径 = {radius} GeV, n_per_axis = {n_per_axis}")
    print(f"  网格点数: 实际 {len(pts)}, 解析估计 {n_expected}")

    # 转换到观测变量
    obs = momentum_to_observables(pts, mass=0.139)  # pion mass
    n_accepted = sum(acceptance_mask(obs, pt_min=0.5, pt_max=100.0, eta_min=-2.5, eta_max=2.5))
    print(f"  探测器接受 (pT>0.5 GeV, |η|<2.5): {n_accepted}/{len(obs)}")

    # 能量 bin (对数等距)
    E_min, E_max = 1.0, 100.0
    N_true = 15
    N_rec = 15
    E_true_edges = [E_min * (E_max / E_min) ** (i / N_true) for i in range(N_true + 1)]
    E_rec_edges = [E_min * (E_max / E_max) ** (i / N_rec) for i in range(N_rec + 1)]
    # 修正 E_rec_edges (同样对数等距)
    E_rec_edges = [E_min * (E_max / E_min) ** (i / N_rec) for i in range(N_rec + 1)]
    print(f"  能量 bin: {N_true} (true) × {N_rec} (rec), E ∈ [{E_min}, {E_max}] GeV")

    # =================================================================
    #  2. 探测器网格与富化 (789 + 1353)
    # =================================================================
    _section("2. 探测器网格与富化 (来自 789 + 1353)")

    mesh = DetectorMesh(
        eta_range=(-2.5, 2.5),
        phi_range=(-math.pi, math.pi),
        n_eta=8, n_phi=10,
    )
    print(f"  探测器网格: {mesh.n_eta}×{mesh.n_phi} = {mesh.n_elements()} 单元")
    print(f"  节点数: {mesh.n_nodes()}")

    # 网格富化 T3 → T4
    enriched_nodes, t4_elems = mesh_enrichment_for_response(mesh)
    print(f"  T4 富化后: {len(enriched_nodes)} 节点, {len(t4_elems)} 单元")

    # 网格质量
    ar = mesh_quality_aspect_ratio(mesh)
    print(f"  纵横比: min = {min(ar):.3f}, max = {max(ar):.3f}, mean = {sum(ar)/len(ar):.3f}")

    # =================================================================
    #  3. 能量损失 ODE 基准 (488 + 880)
    # =================================================================
    _section("3. 能量损失 ODE 基准 (来自 488 + 880)")

    params = EnergyLossParams(alpha=0.02, E_max=100.0, beta=1e-3, E_c=10.0)
    xs, E_list = compute_energy_loss_trajectory(E0=50.0, x_max=10.0, params=params)
    print(f"  能量损失轨迹: x ∈ [0, {xs[-1]:.2f}], E: {E_list[0]:.2f} → {E_list[-1]:.4f} GeV")
    print(f"  积分步数: {len(xs)}")

    ts, zs_num, zs_exact = polar_ode_benchmark(t_end=2 * math.pi, n_steps=50)
    max_err = max(abs(zs_num[i] - zs_exact[i]) for i in range(len(ts)))
    print(f"  极坐标 ODE 基准: t ∈ [0, 2π], 最大误差 = {max_err:.2e}")

    # =================================================================
    #  4. 特殊函数验证 (443)
    # =================================================================
    _section("4. 特殊函数验证 (来自 443)")

    test_vals = [0.5, 1.0, 2.0, 5.0]
    for x in test_vals:
        g = r8_gamma(x)
        g_exact = math.gamma(x)
        print(f"  Γ({x}) = {g:.10f}, 精确 = {g_exact:.10f}, 相对误差 = {abs(g-g_exact)/(g_exact+1e-300):.2e}")
    print(f"  erf(1) = {r8_erf(1.0):.10f}")
    print(f"  K_0(1) = {r8_besk0(1.0):.10f}")
    print(f"  I_0(1) = {r8_besi0(1.0):.10f}")
    print(f"  I_{0.5}(0.3, 0.7) = {r8_betai(0.5, 0.7, 0.3):.10f}")

    # =================================================================
    #  5. 求积精确度检验 (466 + 1302)
    # =================================================================
    _section("5. 求积精确度检验 (来自 466 + 1302)")

    _subsection("5a. Gauss-Laguerre 精确度")
    n_quad = 8
    alpha = 0.0
    deg_max = 2 * n_quad
    errors = gen_laguerre_exactness(n_quad, alpha, deg_max)
    print(f"  N = {n_quad}, α = {alpha}")
    print(f"  最高精确度阶数 (误差 < 1e-10): ", end="")
    exact_deg = sum(1 for e in errors if e < 1e-10) - 1
    print(exact_deg)
    print(f"  理论: 2N-1 = {2*n_quad-1}")

    _subsection("5b. 三角形求积精确度")
    v1, v2, v3 = (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)
    tri_errors = triangle_exactness(v1, v2, v3, degree_max=6, quadrature_order=7)
    print(f"  最高精确度阶数 (误差 < 1e-10): {sum(1 for e in tri_errors if e < 1e-10) - 1}")

    # =================================================================
    #  6. 生成真实能谱
    # =================================================================
    _section("6. 真实能谱生成")

    # 使用组合谱
    # 功率律谱 (γ=2.7) 在 [1, 100] GeV 上的积分 ~ 0.59; 乘以因子使总计数 ~ 1000
    T_raw = integrate_spectrum_in_bins(power_law_spectrum, E_true_edges)
    T_raw_total = sum(T_raw)
    target_total = 1000.0
    T_true = [t * (target_total / T_raw_total) if T_raw_total > 0 else t for t in T_raw]
    print(f"  幂律谱 (γ=2.7): 总计数 = {sum(T_true):.2f} (归一化至 {target_total:.0f})")

    T_thermal = integrate_spectrum_in_bins(lambda E: thermal_spectrum(E, T=0.3), E_true_edges)
    T_therm_total = sum(T_thermal)
    if T_therm_total > 0:
        T_thermal = [t * (target_total / T_therm_total) for t in T_thermal]
    print(f"  热谱 (T=0.3 GeV): 总计数 = {sum(T_thermal):.2f}")

    T_sync = integrate_spectrum_in_bins(lambda E: synchrotron_spectrum(E, E_c=10.0), E_true_edges)
    T_sync_total = sum(T_sync)
    if T_sync_total > 0:
        T_sync = [t * (target_total / T_sync_total) for t in T_sync]
    print(f"  同步辐射谱: 总计数 = {sum(T_sync):.2f}")

    T_polar = integrate_spectrum_in_bins(lambda E: polar_ode_spectrum(E, A=5.0), E_true_edges)
    print(f"  极坐标 ODE 谱: 总计数 = {sum(T_polar):.2f}")

    # 选择幂律谱作为基准
    T_true_ref = T_true
    print(f"\n  基准真谱: 幂律, 总计数 = {sum(T_true_ref):.2f}")

    # =================================================================
    #  7. 组装响应矩阵
    # =================================================================
    _section("7. 探测器响应矩阵组装 (Wathen 风格)")

    R_csr = assemble_response_matrix(
        E_true_edges, E_rec_edges,
        resolution_a=0.10, resolution_b=0.01,
        efficiency=0.95,
        quadrature_order=5,
    )
    print(f"  响应矩阵: {R_csr.nrows} × {R_csr.ncols}")
    print(f"  非零元: {R_csr.nnz()} / {R_csr.nrows * R_csr.ncols}")
    print(f"  带宽: {R_csr.bandwidth()}")

    R_dense = R_csr.to_dense()

    # 列归一化检查
    col_sums = [sum(R_dense[i][j] for i in range(N_rec)) for j in range(N_true)]
    print(f"  列和: min = {min(col_sums):.4f}, max = {max(col_sums):.4f}")

    # 迁移图
    A_mig = migration_adjacency(R_csr, threshold=1e-6)
    P_trans = adjacency_to_transition(A_mig)
    print(f"  迁移图边数: {A_mig.nnz()}")

    # =================================================================
    #  8. 生成观测数据
    # =================================================================
    _section("8. 观测数据生成")

    O_exact = apply_response(R_dense, T_true_ref)
    O_noisy = add_poisson_noise(O_exact, seed=42, scale=1.0)
    print(f"  精确观测总计数: {sum(O_exact):.2f}")
    print(f"  含噪观测总计数: {sum(O_noisy):.2f}")
    snr = sum(O_exact) / max(math.sqrt(sum(O_exact)), 1e-10)
    print(f"  信噪比 (总): {snr:.1f}")

    # =================================================================
    #  9. Unfolding 方法对比
    # =================================================================
    _section("9. Unfolding 方法对比")

    _subsection("9a. SVD 截断 unfolding")
    T_svd, sigma_svd, U_svd, V_svd = svd_unfold(R_dense, O_noisy, k_cutoff=8)
    err_svd = _relative_error(T_svd, T_true_ref)
    print(f"  截断阶数 k = 8")
    print(f"  相对误差: {err_svd:.4f}")
    print(f"  总计数 (unfolded/true): {sum(T_svd):.2f} / {sum(T_true_ref):.2f}")

    _subsection("9b. D'Agostini 迭代贝叶斯")
    T_dag = dAgostini_unfold(R_dense, O_noisy, n_iter=15)
    err_dag = _relative_error(T_dag, T_true_ref)
    print(f"  迭代次数: 15")
    print(f"  相对误差: {err_dag:.4f}")

    _subsection("9c. Tikhonov 正则化 (黄金分割选 λ)")
    lam_opt = optimal_lambda_by_golden(
        R_dense, O_noisy, lam_range=(1e-6, 0.1), L_order=2,
    )
    print(f"  最优 λ (黄金分割): {lam_opt:.6f}")
    T_tik = tikhonov_unfold(R_dense, O_noisy, lam=lam_opt, L_order=2)
    err_tik = _relative_error(T_tik, T_true_ref)
    print(f"  相对误差: {err_tik:.4f}")

    _subsection("9d. 物理约束 PINN unfolding")
    E_centers = [0.5 * (E_true_edges[i] + E_true_edges[i + 1]) for i in range(N_true)]
    pinn = PhysicsInformedUnfolder(
        E_centers, N_hidden=12, lr=0.005, lambda_smooth=0.01, seed=42,
    )
    T_pinn = pinn.train(R_dense, O_noisy, n_epochs=50)
    err_pinn = _relative_error(T_pinn, T_true_ref)
    print(f"  训练轮数: 50")
    print(f"  相对误差: {err_pinn:.4f}")

    # =================================================================
    #  10. 稳定性分析
    # =================================================================
    _section("10. 稳定性分析")

    _subsection("10a. 条件数与有效秩")
    kappa = condition_number_from_singulars(sigma_svd)
    k_eff = effective_rank(sigma_svd, threshold=1e-3)
    print(f"  条件数 κ(R) = {kappa:.2e}")
    print(f"  有效秩 (阈值 1e-3): {k_eff} / {len(sigma_svd)}")

    _subsection("10b. L-curve 分析")
    lam_scan = [10 ** (-6 + 0.5 * i) for i in range(13)]
    log_xi, log_eta, _ = compute_l_curve(R_dense, O_noisy, lam_scan, L_order=2)
    corner_idx = l_curve_corner(log_xi, log_eta)
    print(f"  L-curve 采样点: {len(lam_scan)}")
    print(f"  角点索引: {corner_idx}, λ_corner = {lam_scan[corner_idx]:.2e}")

    _subsection("10c. 传播矩阵与方差")
    H_svd = hat_matrix_svd(U_svd, sigma_svd, V_svd, k_cutoff=8)
    sigma_T_svd = propagate_variance(H_svd, O_noisy)
    bias_svd, var_svd, fom_svd = figure_of_merit(T_svd, T_true_ref, sigma_T_svd)
    print(f"  SVD unfold FOM: bias = {bias_svd:.4f}, var = {var_svd:.4f}, FOM = {fom_svd:.4f}")

    H_tik = hat_matrix_tikhonov(R_dense, lam_opt, L_order=2)
    sigma_T_tik = propagate_variance(H_tik, O_noisy)
    bias_tik, var_tik, fom_tik = figure_of_merit(T_tik, T_true_ref, sigma_T_tik)
    print(f"  Tikhonov unfold FOM: bias = {bias_tik:.4f}, var = {var_tik:.4f}, FOM = {fom_tik:.4f}")

    _subsection("10d. FD 光滑度指标")
    h = (E_true_edges[-1] - E_true_edges[0]) / (N_true - 1)
    for label, T_u in [("SVD", T_svd), ("D'Agostini", T_dag), ("Tikhonov", T_tik), ("PINN", T_pinn)]:
        sm = fd_smoothness(T_u, E_true_edges, max_order=3)
        print(f"  {label:12s}: S1 = {sm[0]:.3e}, S2 = {sm[1]:.3e}, S3 = {sm[2]:.3e}")

    # =================================================================
    #  11. 方法对比总结
    # =================================================================
    _section("11. 方法对比总结")

    methods = [
        ("SVD (k=8)", T_svd, err_svd, fom_svd),
        ("D'Agostini (15 iter)", T_dag, err_dag, 0.0),
        ("Tikhonov (λ={:.1e})".format(lam_opt), T_tik, err_tik, fom_tik),
        ("PINN (50 epoch)", T_pinn, err_pinn, 0.0),
    ]
    print(f"  {'方法':30s}  {'相对误差':>10s}  {'总计数':>10s}  {'FOM':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name, T_u, err, fom in methods:
        print(f"  {name:30s}  {err:10.4f}  {sum(T_u):10.2f}  {fom:8.4f}")

    # =================================================================
    #  12. 综合稳定性报告 (最佳方法)
    # =================================================================
    _section("12. 综合稳定性报告 (SVD unfolding)")

    report = stability_report(
        sigma_svd, H_svd, O_noisy, T_svd, T_true_ref, E_true_edges,
    )
    for k, v in report.items():
        print(f"  {k:25s} = {v:.4e}" if isinstance(v, float) else f"  {k:25s} = {v}")

    # =================================================================
    #  13. χ² 拟合优度
    # =================================================================
    _section("13. χ² 拟合优度")

    from physics_models import chi_squared as chi2_func
    for name, T_u in [("SVD", T_svd), ("Tikhonov", T_tik), ("D'Agostini", T_dag)]:
        O_exp = apply_response(R_dense, T_u)
        chi2 = chi2_func(O_noisy, O_exp)
        ndf = N_rec - 1
        print(f"  {name:12s}: χ² = {chi2:.2f}, χ²/ndf = {chi2/ndf:.3f}")

    # =================================================================
    #  结束
    # =================================================================
    _section("完成")
    elapsed = time.time() - t0
    print(f"  总耗时: {elapsed:.2f} 秒")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
