"""
main.py - PROJECT_275 统一入口
==============================
非厄米 SSH 模型的谱结构、例外点与高阶有限差分稳定性分析.

零参数运行: python main.py

流程:
  1. 构造非厄米 SSH 哈密顿量
  2. 高阶有限差分离散化 + Von Neumann 稳定性分析
  3. 谱求解 + 双正交分解
  4. 例外点定位 (Newton + Cauchy + CVT 全局搜索)
  5. 参数延拓追踪谱流
  6. 布里渊区积分 (态密度 + Berry 相位)
  7. 谱统计分析 (RMT 比较)
  8. 谱相分类
  9. 自适应网格细化
"""
from __future__ import annotations
import sys
import os
import numpy as np

# 把本目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nonhermitian_hamiltonian import (
    ssh_hamiltonian_1d, hatano_nelson_hamiltonian,
    adjacency_to_tight_binding, blend_hamiltonian_101,
    discriminant_ep_indicator, build_spectral_phase_diagram,
)
from high_order_fd import (
    fd_coefficients_second_derivative, apply_laplacian_fd,
    build_nonhermitian_fd_matrix, von_neumann_growth_factor,
    stability_domain_scan, cauchy_theta_step, leapfrog_step,
)
from spectral_solver import (
    compute_spectrum, compute_left_eigenvectors,
    biorthogonal_projector, spectral_decomposition_residual,
    skin_mode_localization, chiral_decomposition,
)
from exceptional_point_locator import (
    scan_discriminant_1d, locate_ep_1d_newton,
    cauchy_fsolve_ep, ep_order_estimation,
    cvt_parameter_sampling, global_ep_search,
)
from stability_analysis import (
    spectral_stability_analysis, leapfrog_stability_bound,
    cauchy_theta_stability, time_evolution_norm, cfl_scan,
)
from brillouin_quadrature import (
    witherden_rule_2d, triangle_rule_2d, quadrature_exactness_test,
    brillouin_zone_integral_2d, berry_phase_1d,
)
from parameter_continuation import (
    track_eigenvalue_path, cauchy_continuation,
    ep_puiseux_expansion, winding_number_around_ep,
)
from spectral_statistics import (
    normal_cdf, normal_inverse_cdf,
    level_spacing_distribution, level_spacing_ratio,
    spectral_rigidity, disordered_average, spectral_summary,
)
from phase_classifier import (
    extract_spectral_features, classify_phase,
    build_adjacency_from_hamiltonian, is_connected,
    phase_diagram_classification,
)
from adaptive_ep_mesh import (
    adaptive_refinement_1d, build_ep_adaptive_k_grid,
    ep_localization_error, adaptive_condition_mesh,
)


# ============================================================================
# 打印工具
# ============================================================================
def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def subheader(title: str) -> None:
    print(f"\n--- {title} ---")


# ============================================================================
# 主流程
# ============================================================================
def main() -> None:
    np.random.seed(275)
    print("=" * 72)
    print("  PROJECT 275: Non-Hermitian Spectral Structure & Exceptional")
    print("  Points - High-Order Finite Difference & Stability Analysis")
    print("=" * 72)

    # =========================================================================
    # 1. 哈密顿量构造
    # =========================================================================
    header("1. Non-Hermitian SSH Hamiltonian Construction")

    t1, t2, gamma = 1.0, 0.6, 0.35
    k_test = np.pi / 3
    H_ssh = ssh_hamiltonian_1d(t1, t2, gamma, k_test)
    print(f"  SSH parameters: t1={t1}, t2={t2}, gamma={gamma}, k={k_test:.4f}")
    print(f"  H_SSH(k) =\n{H_ssh}")
    delta = discriminant_ep_indicator(H_ssh)
    print(f"  Discriminant Delta(k) = {delta.real:.6f} + {delta.imag:.6f}i")
    print(f"  |Delta| = {abs(delta):.6e}")

    # Hatano-Nelson
    N_hn = 12
    g_nh = 0.4
    H_hn = hatano_nelson_hamiltonian(N_hn, t=1.0, g=g_nh, boundary="periodic")
    print(f"\n  Hatano-Nelson: N={N_hn}, g={g_nh}")
    E_hn = np.linalg.eigvals(H_hn)
    print(f"  Spectrum range: Re in [{E_hn.real.min():.4f}, {E_hn.real.max():.4f}], "
          f"Im in [{E_hn.imag.min():.4f}, {E_hn.imag.max():.4f}]")

    # 邻接矩阵 → 紧束缚
    adj = np.zeros((6, 6))
    for i in range(5):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    adj[0, 5] = 1
    adj[5, 0] = 1
    H_tb = adjacency_to_tight_binding(adj)
    print(f"\n  Tight-binding from adjacency (6-site ring):")
    E_tb = np.linalg.eigvals(H_tb)
    print(f"  Eigenvalues: {[f'{e.real:.3f}+{e.imag:.3f}i' for e in E_tb]}")

    # 混合哈密顿量
    H0 = ssh_hamiltonian_1d(1.0, 0.5, 0.0, k_test)
    H1 = ssh_hamiltonian_1d(0.5, 1.0, 0.5, k_test)
    H_blend = blend_hamiltonian_101(0.5, H0, H1)
    print(f"\n  Blended H(r=0.5): trace = {np.trace(H_blend):.4f}")

    # =========================================================================
    # 2. 高阶有限差分
    # =========================================================================
    header("2. High-Order Finite Difference Operators")

    for order in [2, 4, 6]:
        c = fd_coefficients_second_derivative(order // 2)
        print(f"  Order {order} FD coefficients (2p+1={len(c)} pts): "
              f"c = [{', '.join(f'{ci:.6f}' for ci in c)}]")
        print(f"    Sum check (should be ~0): {np.sum(c):.2e}")

    # Laplacian 作用于测试函数
    N_fd = 64
    L = 2 * np.pi
    dx = L / N_fd
    x_grid = np.linspace(0, L, N_fd, endpoint=False)
    psi_test = np.sin(3 * x_grid) + 0.5j * np.cos(2 * x_grid)
    for order in [2, 4]:
        lap = apply_laplacian_fd(psi_test, dx, order=order, bc="periodic")
        exact = -9 * np.sin(3 * x_grid) - 2j * np.cos(2 * x_grid)
        err = np.max(np.abs(lap - exact))
        print(f"  Laplacian error (order {order}): {err:.3e}")

    # 非厄米 FD 矩阵
    gamma_profile = 0.3 * np.sin(x_grid)
    H_fd = build_nonhermitian_fd_matrix(N_fd, dx, nu=1.0,
                                        gamma_profile=gamma_profile,
                                        order=4, bc="periodic")
    print(f"\n  FD matrix: shape={H_fd.shape}, norm={np.linalg.norm(H_fd, 'fro'):.4f}")
    print(f"  Is Hermitian: {np.allclose(H_fd, H_fd.conj().T)}")

    # =========================================================================
    # 3. 谱求解 + 双正交分解
    # =========================================================================
    header("3. Spectral Solver & Biorthogonal Decomposition")

    E_ssh, psi_R = compute_spectrum(H_ssh, sort_by="real")
    print(f"  SSH eigenvalues: E_+ = {E_ssh[1]:.6f}, E_- = {E_ssh[0]:.6f}")
    psi_L = compute_left_eigenvectors(H_ssh, E_ssh, psi_R)
    res = spectral_decomposition_residual(H_ssh, E_ssh, psi_R)
    print(f"  Spectral decomposition residual: {res:.3e}")

    ipr, x_c = skin_mode_localization(psi_R)
    print(f"  IPR of eigenstates: {ipr}")
    chi = chiral_decomposition(E_ssh)
    print(f"  Chirality: {chi['chirality']:.4f} (N+={chi['N_plus']}, N-={chi['N_minus']})")

    # Hatano-Nelson 趋肤效应
    E_hn_sorted, psi_R_hn = compute_spectrum(H_hn, sort_by="real")
    ipr_hn, x_c_hn = skin_mode_localization(psi_R_hn)
    print(f"\n  Hatano-Nelson IPR: mean={np.mean(ipr_hn):.4f}, max={np.max(ipr_hn):.4f}")
    print(f"  (IPR >> 1 indicates skin modes)")

    # =========================================================================
    # 4. 例外点定位
    # =========================================================================
    header("4. Exceptional Point Location")

    # 一维扫描
    k_scan = np.linspace(0, 2 * np.pi, 200)
    delta_vals, _ = scan_discriminant_1d(t1, t2, gamma, k_scan)
    k_min_idx = np.argmin(delta_vals)
    print(f"  1D scan: min |Delta| = {delta_vals[k_min_idx]:.4e} at k = {k_scan[k_min_idx]:.4f}")

    # Newton 精化
    res_newton = locate_ep_1d_newton(t1, t2, gamma, k_scan[k_min_idx])
    print(f"  Newton refinement: k_EP = {res_newton['k_ep']:.8f}, "
          f"|Delta| = {res_newton['delta_min']:.3e}, "
          f"converged = {res_newton['converged']}")

    # EP 阶数估计
    H_at_ep = ssh_hamiltonian_1d(t1, t2, gamma, res_newton['k_ep'])
    ep_info = ep_order_estimation(H_at_ep, tol=1e-4)
    print(f"  EP order estimate: {ep_info['ep_order']}")
    print(f"  Algebraic multiplicities: {ep_info['algebraic_multiplicity']}")
    print(f"  Geometric multiplicities: {ep_info['geometric_multiplicity']}")

    # Cauchy 法求解
    def residual_2d(params):
        H = ssh_hamiltonian_1d(params[0], t2, params[1], res_newton['k_ep'])
        delta = discriminant_ep_indicator(H)
        return np.array([delta.real, delta.imag])

    res_cauchy = cauchy_fsolve_ep(residual_2d, np.array([t1, gamma]))
    print(f"\n  Cauchy 2D solve: (t1, gamma) = ({res_cauchy['x_sol'][0]:.6f}, "
          f"{res_cauchy['x_sol'][1]:.6f}), residual = {res_cauchy['residual']:.3e}")

    # CVT 采样
    cvt_pts = cvt_parameter_sampling(15, [(0.3, 1.5), (0.1, 0.8), (0, 2 * np.pi)],
                                     it_num=5)
    print(f"\n  CVT parameter sampling: {len(cvt_pts)} points generated")

    # =========================================================================
    # 5. 参数延拓
    # =========================================================================
    header("5. Parameter Continuation & Spectral Flow")

    def H_k_func(k):
        return ssh_hamiltonian_1d(t1, t2, gamma, k)

    k_path = np.linspace(0, 2 * np.pi, 80)
    E_band, overlap_path = track_eigenvalue_path(H_k_func, k_path, band_idx=1)
    print(f"  Tracked band 1 along BZ: E_start = {E_band[0]:.4f}, E_end = {E_band[-1]:.4f}")
    print(f"  Min overlap (EP indicator): {np.min(overlap_path):.4f}")

    # Berry 相位
    berry = berry_phase_1d(H_k_func, 0, 2 * np.pi, n_k=100)
    print(f"  Berry (Zak) phase: {berry:.4f} rad (should be ~0 or ~π for SSH)")

    # Cauchy 延拓
    lam_vals_c, E_cont = cauchy_continuation(
        lambda k: ssh_hamiltonian_1d(t1, t2, gamma, k),
        lambda_0=0.0, lambda_f=np.pi, n_steps=30, band_idx=1, theta=0.5)
    print(f"\n  Cauchy continuation E(0→π): E(0) = {E_cont[0]:.4f}, "
          f"E(π) = {E_cont[-1]:.4f}")

    # =========================================================================
    # 6. 布里渊区积分
    # =========================================================================
    header("6. Brillouin Zone Quadrature")

    for prec in [1, 3, 5]:
        x, y, w = witherden_rule_2d(prec)
        print(f"  Witherden precision {prec}: {len(w)} points, sum(w) = {w.sum():.10f}")

    # 精确度测试
    def exact_sq(a, b):
        return 1.0 / ((a + 1) * (b + 1))

    xact = quadrature_exactness_test(
        lambda: witherden_rule_2d(5), dim=2, degree_max=8,
        exact_integral=exact_sq)
    print(f"  Exactness test (prec 5): max exact degree = {xact['max_exact_degree']}")
    print(f"  Errors: {xact['errors_by_degree']}")

    # 态密度
    def H_2d_func(kx, ky):
        return ssh_hamiltonian_1d(t1, t2, gamma, kx) * np.eye(2) + 0.1 * np.eye(2) * np.cos(ky)

    dos_val = brillouin_zone_integral_2d(
        lambda kx, ky: -np.imag(np.trace(np.linalg.inv(
            (0.5 + 0.05j) * np.eye(2) - H_2d_func(kx, ky)))) / np.pi,
        lattice="square", precision=5)
    print(f"\n  DOS at E=0.5: {dos_val:.4f}")

    # =========================================================================
    # 7. 稳定性分析
    # =========================================================================
    header("7. Stability Analysis")

    stab = spectral_stability_analysis(H_hn)
    print(f"  Hatano-Nelson spectral stability:")
    print(f"    Max Im(E) = {stab['max_imag']:.4e}")
    print(f"    PT broken = {stab['pt_broken']}")
    print(f"    Unstable modes = {stab['unstable_modes']}")

    dt_max = leapfrog_stability_bound(nu=1.0, gamma=0.3, dx=dx, order=4)
    print(f"\n  Leapfrog stability bound: dt_max = {dt_max:.4e}")

    for theta in [0.3, 0.5, 1.0]:
        info = cauchy_theta_stability(theta, nu=1.0, gamma=0.3, dx=dx, order=4)
        print(f"  Cauchy theta={theta}: A-stable={info['A_stable']}, "
              f"max growth (imag axis) = {info['max_growth_for_pure_imag']:.4f}")

    # Von Neumann 扫描
    r_values = np.linspace(0.01, 0.5, 30)
    cfl_info = cfl_scan(nu=1.0, gamma=0.3, dx=dx, r_values=r_values, order=2)
    print(f"\n  CFL scan: critical r = {cfl_info['critical_r']}")

    # 时间演化
    psi0 = np.zeros(N_fd, dtype=complex)
    psi0[N_fd // 2] = 1.0
    t_vals, norm_cauchy = time_evolution_norm(H_fd, psi0, t_max=0.5, n_steps=100,
                                              method="cauchy", theta=1.0)
    print(f"\n  Time evolution (Cauchy θ=1): ||ψ(0)||={norm_cauchy[0]:.4f}, "
          f"||ψ(t_max)||={norm_cauchy[-1]:.4f}")

    # =========================================================================
    # 8. 谱统计
    # =========================================================================
    header("8. Spectral Statistics (RMT)")

    print(f"  Normal CDF: Φ(0) = {normal_cdf(0):.6f}, Φ(1) = {normal_cdf(1):.6f}, "
          f"Φ(-2) = {normal_cdf(-2):.6f}")
    print(f"  Normal inverse: Φ^{-1}(0.975) = {normal_inverse_cdf(0.975):.6f}")

    # 无序平均
    def H_builder():
        return hatano_nelson_hamiltonian(10, t=1.0, g=g_nh, boundary="periodic")

    dis_res = disordered_average(H_builder, n_samples=30,
                                 disorder_strength=0.2, seed=275)
    E_avg = dis_res["mean_spectrum"]
    print(f"\n  Disordered average (30 samples, W=0.2):")
    print(f"    Mean |E| = {np.mean(np.abs(E_avg)):.4f}")
    print(f"    Std |E| = {np.mean(dis_res['std_spectrum']):.4f}")

    # 间距分布
    E_large = np.linalg.eigvals(hatano_nelson_hamiltonian(40, t=1.0, g=0.3))
    s_centers, P_s = level_spacing_distribution(E_large)
    r_mean, r_std = level_spacing_ratio(E_large)
    print(f"\n  Level spacing ratio: <r> = {r_mean:.4f} ± {r_std:.4f}")
    print(f"    (GUE: 0.6027, Poisson: 0.3863)")

    # 谱刚性
    L_vals, Sigma2 = spectral_rigidity(E_large, L_max=10)
    print(f"\n  Spectral rigidity Σ²(L): L={L_vals[:5]}, Σ²={Sigma2[:5]}")

    # 谱摘要
    summary = spectral_summary(E_large)
    print(f"\n  Spectral summary (40-site HN):")
    print(f"    Re: [{summary['real_min']:.3f}, {summary['real_max']:.3f}], "
          f"median={summary['real_median']:.3f}")
    print(f"    Im: [{summary['imag_min']:.3f}, {summary['imag_max']:.3f}]")

    # =========================================================================
    # 9. 谱相分类
    # =========================================================================
    header("9. Spectral Phase Classification")

    for g_test in [0.0, 0.3, 0.8, 1.5]:
        H_test = hatano_nelson_hamiltonian(16, t=1.0, g=g_test)
        feats = extract_spectral_features(H_test)
        phase = classify_phase(feats)
        print(f"  g={g_test:.2f}: frac_real={feats['frac_real']:.2f}, "
              f"mean_ipr={feats['mean_ipr']:.3f} → {phase}")

    # 相图扫描
    gamma_grid = np.linspace(0, 2.0, 12)
    phases = phase_diagram_classification(
        lambda gamma: hatano_nelson_hamiltonian(16, t=1.0, g=gamma),
        gamma_grid, "gamma")
    print(f"\n  Phase diagram along gamma:")
    for g, ph in zip(gamma_grid, phases):
        print(f"    gamma={g:.3f} → {ph}")

    # 邻接矩阵分析
    A = build_adjacency_from_hamiltonian(H_hn)
    print(f"\n  Adjacency from HN Hamiltonian: connected = {is_connected(A)}")

    # =========================================================================
    # 10. 自适应网格
    # =========================================================================
    header("10. Adaptive Mesh Refinement near EP")

    k_adaptive = build_ep_adaptive_k_grid(t1, t2, gamma, tol=1e-3, max_levels=6)
    print(f"  Adaptive k-grid: {len(k_adaptive)} points (vs 200 uniform)")
    print(f"  Range: [{k_adaptive[0]:.4f}, {k_adaptive[-1]:.4f}]")

    loc_err = ep_localization_error(k_adaptive, t1, t2, gamma)
    print(f"  Localization: min |Delta| = {loc_err['min_delta']:.3e} at k = {loc_err['k_at_min']:.4f}")
    print(f"  Local spacing at EP: {loc_err['spacing_at_min']:.3e}")

    k_cond = adaptive_condition_mesh(t1, t2, gamma, tol=0.3, max_levels=5)
    print(f"\n  Condition-based adaptive: {len(k_cond)} points")

    # =========================================================================
    # 11. 相图扫描 (综合)
    # =========================================================================
    header("11. Summary: Phase Diagram in (t1, gamma) Space")

    t1_range = np.linspace(0.3, 1.5, 8)
    gamma_range = np.linspace(0.05, 1.0, 8)
    min_delta = build_spectral_phase_diagram(t1_range, gamma_range, 0.6,
                                             k_samples=40)
    print(f"  Phase diagram shape: {min_delta.shape}")
    print(f"  Min |Delta| across parameter space: {min_delta.min():.3e}")
    print(f"  Max |Delta| across parameter space: {min_delta.max():.3e}")
    ep_candidates = np.argwhere(min_delta < 0.1)
    print(f"  Near-EP candidates (|Delta| < 0.1): {len(ep_candidates)} points")
    if len(ep_candidates) > 0:
        for idx in ep_candidates[:3]:
            i, j = idx
            print(f"    (t1={t1_range[i]:.3f}, gamma={gamma_range[j]:.3f}): "
                  f"|Delta|={min_delta[i, j]:.3e}")

    # =========================================================================
    # 结束
    # =========================================================================
    header("Execution Complete")
    print("  All modules executed successfully.")
    print(f"  Non-Hermitian SSH model: t1={t1}, t2={t2}, gamma={gamma}")
    print(f"  EP candidate: k ~ {res_newton['k_ep']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
