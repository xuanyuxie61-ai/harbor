#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np


import utils
import test_eigen_validation
import optimal_sampling
import tight_binding
import bz_integration
import fermi_surface
import spectral_analysis
import tdgl_dynamics
import percolation_coherence
import bcs_gap_solver


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_tight_binding_and_validate():
    print_section("任务 1：紧束缚哈密顿量构建与数值验证")

    Nx, Ny = 8, 8
    t, tp, mu = 1.0, 0.3, 0.0
    H = tight_binding.build_tight_binding_hamiltonian(Nx, Ny, t=t, tp=tp, mu=mu, open_boundary=False)
    print(f"  晶格尺寸: {Nx}x{Ny}, t={t}, t'={tp}, μ={mu}")
    print(f"  哈密顿量维度: {H.shape}, 非零元数: {H.nnz}")


    is_herm = tight_binding.validate_hamiltonian_hermiticity(H, tol=1e-10)
    print(f"  厄米性验证: {'通过' if is_herm else '失败'}")


    H_dense = H.toarray()
    nz_num, colptr, rowind, vals = tight_binding.dense_to_ccs(H_dense)
    H_back = tight_binding.ccs_to_csr(colptr, rowind, vals, H_dense.shape[0], H_dense.shape[1])
    diff = np.linalg.norm(H_dense - H_back.toarray(), ord='fro')
    print(f"  CCS 转换往返误差 (Frobenius): {diff:.3e}")


    A_test, Q_test, lam_exact = test_eigen_validation.generate_symmetric_test_matrix(20, lambda_mean=0.0, lambda_std=1.0)
    val_res = test_eigen_validation.validate_eigensolver(A_test, lam_exact)
    print(f"  测试矩阵本征值最大绝对误差: {val_res['max_abs_err']:.3e}")
    print(f"  测试矩阵本征值平均相对误差: {val_res['mean_rel_err']:.3e}")


    eigvals, eigvecs = tight_binding.diagonalize_sparse_hamiltonian(H, k=6, sigma=0.0)
    print(f"  费米面附近 6 个本征值: {np.round(eigvals, 4)}")


    k_path = np.linspace(-np.pi, np.pi, 20)

    energies = tight_binding.dispersion_2d(k_path, np.zeros_like(k_path), t=t, tp=tp, mu=mu)
    k_fine = np.linspace(-np.pi, np.pi, 100)
    energies_fine = tight_binding.band_structure_interpolation(k_path, energies, k_fine)
    print(f"  能带插值: 粗网格 {len(k_path)} 点 -> 细网格 {len(k_fine)} 点，边界二阶导数=0（自然样条）")

    return H, eigvals, k_fine, energies_fine


def run_optimal_sampling_and_bz():
    print_section("任务 2：最优采样与 BZ 积分")


    g, e_hist, m_hist = optimal_sampling.cvtm_1d(g_num=16, it_num=30, s_num=3000, seed=42)
    k_opt = optimal_sampling.optimal_k_path_sampling(-np.pi, np.pi, n_points=16, it_num=30, s_num=3000)
    print(f"  CVT 生成 {len(g)} 个最优 k 点，最终能量泛函={e_hist[-1]:.4f}, 平均移动量={m_hist[-1]:.3e}")


    e_vec = np.array([2, 1, 0])
    exact = bz_integration.wedge01_monomial_integral(e_vec)

    pts = bz_integration.wedge01_sample(50000)
    vals = bz_integration.monomial_value(3, pts.shape[1], e_vec, pts)
    mc_est = np.mean(vals) * bz_integration.wedge01_volume()
    print(f"  楔形单项式积分 (e=[2,1,0]): 精确值={exact:.6f}, MC={mc_est:.6f}")



    def f_unit(p):
        return np.ones(p.shape[0])
    tet_int = bz_integration.integrate_tetrahedron(f_unit, degree=3)
    print(f"  四面体求积 (f=1): 结果={tet_int:.6f}, 理论=0.166667")


    x_nodes, w_nodes = bz_integration.gauss_jacobi_quadrature(8, 0.0, 0.0)

    int_x2 = np.sum(w_nodes * x_nodes ** 2)
    print(f"  Gauss-Jacobi (α=β=0, n=8) 积分 x^2: {int_x2:.6f}, 理论=0.666667")


    t, tp, mu = 1.0, 0.3, 0.0
    def dos_integrand(kpts):
        kx = kpts[:, 0]
        ky = kpts[:, 1]
        eps = -2.0 * t * (np.cos(kx) + np.cos(ky)) + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu

        eta = 0.05
        return (1.0 / np.pi) * eta / (eps ** 2 + eta ** 2)

    dos = bz_integration.integrate_bz_gauss_legendre_2d(dos_integrand, n_per_dim=40)
    print(f"  BZ 态密度 (η=0.05): N(0) ≈ {dos:.4f}")

    return k_opt


def run_fermi_surface_analysis():
    print_section("任务 3：费米面几何分析")


    nk = 64
    kx = np.linspace(-np.pi, np.pi, nk)
    ky = np.linspace(-np.pi, np.pi, nk)
    KX, KY = np.meshgrid(kx, ky)
    eps_grid = tight_binding.dispersion_2d(KX, KY, t=1.0, tp=0.3, mu=0.0)


    stats = fermi_surface.fermi_surface_annulus_stats(kx, ky, eps_grid, mu=0.0, delta_e=0.15, n_mc=3000)
    print(f"  费米环质心: ({stats['centroid'][0]:.3f}, {stats['centroid'][1]:.3f})")
    print(f"  等效内外半径: r1={stats['r1']:.3f}, r2={stats['r2']:.3f}")
    print(f"  环上动量距离均值={stats['mean_distance']:.3f}, 方差={stats['variance']:.3f}")


    boundary_info = fermi_surface.trace_fermi_surface_boundary(eps_grid, kx, ky, mu=0.0)
    print(f"  费米面边界多边形: 点数={boundary_info['boundary_points'].shape[0]}")
    print(f"  近似面积={boundary_info['area_approx']:.4f}, 近似质心=({boundary_info['centroid_approx'][0]:.3f}, {boundary_info['centroid_approx'][1]:.3f})")
    print(f"  近似转动惯量={boundary_info['moment_approx']:.4f}")


    q_nest, overlap = fermi_surface.fermi_surface_nesting_vector(kx, ky, eps_grid, mu=0.0, delta_e=0.1)
    print(f"  最优嵌套向量 Q=({q_nest[0]:.3f}, {q_nest[1]:.3f}), 重叠度={overlap:.3f}")

    return eps_grid, kx, ky


def run_spectral_analysis():
    print_section("任务 4：Haar 小波谱分析")


    N = 128
    x = np.linspace(0, 4 * np.pi, N)
    y = np.linspace(0, 4 * np.pi, N)
    X, Y = np.meshgrid(x, y)

    Delta_field = (1.0 * np.sin(X) * np.sin(Y)
                   + 0.3 * np.sin(4 * X) * np.sin(4 * Y)
                   + 0.1 * np.random.randn(N, N))


    coeffs = spectral_analysis.haar_2d(Delta_field)
    reconstructed = spectral_analysis.haar_2d_inverse(coeffs)
    recon_err = np.max(np.abs(Delta_field - reconstructed))
    print(f"  2D Haar 正逆变换最大误差: {recon_err:.3e}")

    spectrum, scales = spectral_analysis.analyze_superconducting_fluctuations(Delta_field)
    print(f"  各尺度能量谱（从低频到高频）:")
    for i, (e, s) in enumerate(zip(spectrum, scales)):
        print(f"    尺度 {s}x{s}: 能量={e:.4f}")

    return Delta_field


def run_tdgl_and_dynamics():
    print_section("任务 5：时间依赖动力学")


    t_wave, u_wave, v_wave, energy_wave = tdgl_dynamics.solve_wave_equation_mol(
        nx=32, c=1.0, t_span=(0.0, 2.0 * np.pi), nt_eval=50
    )
    E_conservation = np.max(np.abs(energy_wave - energy_wave[0]))
    print(f"  波动方程能量守恒偏差: {E_conservation:.3e}")


    t_chen, y_chen, lyap = tdgl_dynamics.solve_chen_system(
        t_span=(0.0, 10.0), nt_eval=1000
    )
    print(f"  Chen 系统 Lyapunov 指数估计: {lyap:.4f}")
    print(f"  终态: x={y_chen[-1, 0]:.3f}, y={y_chen[-1, 1]:.3f}, z={y_chen[-1, 2]:.3f}")


    t_tdgl, Delta_tdgl, max_amp = tdgl_dynamics.tdgl_evolution_1d(
        Nx=64, L=10.0, T=1.0, Tc=1.5, tau=1.0, xi=1.0,
        b_coeff=1.0, t_span=(0.0, 15.0), nt=300
    )
    rho_s = tdgl_dynamics.compute_superfluid_stiffness(Delta_tdgl, dx=10.0 / 64, dt=t_tdgl[1] - t_tdgl[0])
    print(f"  TDGL 稳态最大振幅: {max_amp[-1]:.4f}")
    print(f"  估算超流刚度 ρ_s: {rho_s:.4f}")

    return Delta_tdgl


def run_percolation_analysis(Delta_field):
    print_section("任务 6：渗流与超导相干性")



    nt, nx = Delta_field.shape
    ny = int(np.sqrt(nx))
    if ny * ny == nx:
        field_2d = Delta_field[-1, :].reshape(ny, ny)
    else:

        ny = max(8, nx // 8)
        if nx % ny == 0:
            field_2d = Delta_field[-1, :].reshape(ny, nx // ny)
        else:

            nx_use = (nx // ny) * ny
            field_2d = Delta_field[-1, :nx_use].reshape(ny, nx_use // ny)

    analysis = percolation_coherence.superconducting_percolation_analysis(field_2d, threshold=0.3)

    print(f"  超导占据填充率: {analysis['filling_fraction']:.3f}")
    print(f"  连通团簇数: {analysis['component_num']}")
    print(f"  平均团簇尺寸: {analysis['mean_cluster_size']:.1f}")
    print(f"  最大团簇尺寸: {analysis['largest_cluster_size']}")
    print(f"  是否存在跨越团簇: {'是' if analysis['spanning'] else '否'}")


    p_c_est = percolation_coherence.find_percolation_threshold(m=50, n=50, n_trials=12)
    print(f"  数值估计渗流阈值 p_c ≈ {p_c_est:.4f} (理论≈0.5927)")

    return analysis


def run_bcs_gap_and_phase_diagram():
    print_section("任务 7：d-波 BCS 能隙方程")

    U = 8.0
    T = 0.05
    beta = 1.0 / max(T, 1e-6)
    t, tp, mu = 1.0, 0.3, 0.0


    Delta_sc, history, converged = bcs_gap_solver.solve_gap_self_consistent(
        U=U, beta=beta, t=t, tp=tp, mu=mu,
        Delta_max=5.0, n_k=32, tol=1e-7, max_iter=80
    )
    print(f"  参数: U={U}, T={T}, t'={tp}, μ={mu}")
    print(f"  自洽 d-波能隙 Δ_0 = {Delta_sc:.6f}, 收敛={'是' if converged else '否'}, 迭代次数={len(history)}")


    Tc = bcs_gap_solver.compute_critical_temperature(
        U=U, t=t, tp=tp, mu=mu, beta_max=200.0, n_k=32
    )
    print(f"  估算临界温度 T_c = {Tc:.4f}")


    F_sc = bcs_gap_solver.compute_free_energy(Delta_sc, U, beta, t=t, tp=tp, mu=mu, n_k=24)
    F_n = bcs_gap_solver.compute_free_energy(0.0, U, beta, t=t, tp=tp, mu=mu, n_k=24)
    print(f"  超导态自由能 F_s = {F_sc:.6f}")
    print(f"  正常态自由能 F_n = {F_n:.6f}")
    print(f"  自由能差 ΔF = F_s - F_n = {F_sc - F_n:.6f} (负值表示超导稳定)")


    design = bcs_gap_solver.box_behnken_parameter_sweep(
        U_range=[1.0, 4.0],
        T_range=[0.05, 0.5],
        tp_range=[0.0, 0.5],
        mu_range=[-1.0, 1.0]
    )
    print(f"  Box-Behnken 4 因子实验设计: 生成 {design.shape[1]} 个参数组合点")

    for idx in range(min(3, design.shape[1])):
        U_pt, T_pt, tp_pt, mu_pt = design[:, idx]
        beta_pt = 1.0 / max(T_pt, 1e-6)
        try:
            d_pt, _, conv = bcs_gap_solver.solve_gap_self_consistent(
                U=U_pt, beta=beta_pt, t=1.0, tp=tp_pt, mu=mu_pt, Delta_max=8.0, n_k=24, tol=1e-6, max_iter=60
            )
            print(f"    点 {idx+1}: U={U_pt:.2f}, T={T_pt:.3f}, t'={tp_pt:.2f}, μ={mu_pt:.2f} -> Δ={d_pt:.4f}")
        except Exception as e:
            print(f"    点 {idx+1}: 计算失败 ({e})")

    return Delta_sc, Tc


def run_boundary_word_demo():
    print_section("任务 8：边界词几何")


    word = "rrrdddllluuu"
    ok, msg = utils.boundary_word_check(word)
    print(f"  边界词 '{word}': 有效性={ok}, 消息={msg}")
    area = utils.boundary_word_area(word)
    perim = utils.boundary_word_perimeter(word)
    cx, cy = utils.boundary_word_centroid(word)
    moment = utils.boundary_word_moment(word)
    print(f"  面积={area:.1f}, 周长={perim}, 质心=({cx:.2f}, {cy:.2f}), 转动惯量={moment:.2f}")


def main():
    print("高温超导微观机制 —— 多尺度计算框架")
    print("=" * 70)
    print("物理模型: 2D 紧束缚 + d-波 BCS + TDGL + 渗流相干性")
    print("=" * 70)

    np.random.seed(42)


    run_tight_binding_and_validate()
    run_optimal_sampling_and_bz()
    eps_grid, kx, ky = run_fermi_surface_analysis()
    Delta_field = run_spectral_analysis()
    Delta_tdgl = run_tdgl_and_dynamics()
    run_percolation_analysis(Delta_tdgl)
    run_bcs_gap_and_phase_diagram()
    run_boundary_word_demo()

    print("\n" + "=" * 70)
    print("  全部计算完成，无报错。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
