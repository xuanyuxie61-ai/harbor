#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
-------
高温超导微观机制多尺度计算框架 —— 统一入口。

项目：凝聚态物理：高温超导微观机制
问题：基于二维紧束缚模型 + d-波 BCS 理论 + 时间依赖 Ginzburg-Landau 方程 + 渗流相干性分析，
      自洽计算能隙、相图、序参量动力学及超导连通性。

运行方式：
    python main.py
无需任何命令行参数。
"""

import sys
import numpy as np

# 导入各子模块
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
    """任务 1：构建紧束缚哈密顿量，验证稀疏格式与本征值求解器精度。"""
    print_section("任务 1：紧束缚哈密顿量构建与数值验证")

    Nx, Ny = 8, 8
    t, tp, mu = 1.0, 0.3, 0.0
    H = tight_binding.build_tight_binding_hamiltonian(Nx, Ny, t=t, tp=tp, mu=mu, open_boundary=False)
    print(f"  晶格尺寸: {Nx}x{Ny}, t={t}, t'={tp}, μ={mu}")
    print(f"  哈密顿量维度: {H.shape}, 非零元数: {H.nnz}")

    # 验证厄米性
    is_herm = tight_binding.validate_hamiltonian_hermiticity(H, tol=1e-10)
    print(f"  厄米性验证: {'通过' if is_herm else '失败'}")

    # 转换为 CCS 格式并转回（验证 457_ge_to_ccs 算法）
    H_dense = H.toarray()
    nz_num, colptr, rowind, vals = tight_binding.dense_to_ccs(H_dense)
    H_back = tight_binding.ccs_to_csr(colptr, rowind, vals, H_dense.shape[0], H_dense.shape[1])
    diff = np.linalg.norm(H_dense - H_back.toarray(), ord='fro')
    print(f"  CCS 转换往返误差 (Frobenius): {diff:.3e}")

    # 测试本征值求解器精度（验证 1206_test_eigen）
    A_test, Q_test, lam_exact = test_eigen_validation.generate_symmetric_test_matrix(20, lambda_mean=0.0, lambda_std=1.0)
    val_res = test_eigen_validation.validate_eigensolver(A_test, lam_exact)
    print(f"  测试矩阵本征值最大绝对误差: {val_res['max_abs_err']:.3e}")
    print(f"  测试矩阵本征值平均相对误差: {val_res['mean_rel_err']:.3e}")

    # 对角化紧束缚哈密顿量
    eigvals, eigvecs = tight_binding.diagonalize_sparse_hamiltonian(H, k=6, sigma=0.0)
    print(f"  费米面附近 6 个本征值: {np.round(eigvals, 4)}")

    # 能带插值（验证 593_interp_ncs）
    k_path = np.linspace(-np.pi, np.pi, 20)
    # 沿 (k,0) 方向的色散
    energies = tight_binding.dispersion_2d(k_path, np.zeros_like(k_path), t=t, tp=tp, mu=mu)
    k_fine = np.linspace(-np.pi, np.pi, 100)
    energies_fine = tight_binding.band_structure_interpolation(k_path, energies, k_fine)
    print(f"  能带插值: 粗网格 {len(k_path)} 点 -> 细网格 {len(k_fine)} 点，边界二阶导数=0（自然样条）")

    return H, eigvals, k_fine, energies_fine


def run_optimal_sampling_and_bz():
    """任务 2：最优 k 点采样与布里渊区积分。"""
    print_section("任务 2：最优采样与 BZ 积分")

    # CVT 最优采样（验证 263_cvtm_1d）
    g, e_hist, m_hist = optimal_sampling.cvtm_1d(g_num=16, it_num=30, s_num=3000, seed=42)
    k_opt = optimal_sampling.optimal_k_path_sampling(-np.pi, np.pi, n_points=16, it_num=30, s_num=3000)
    print(f"  CVT 生成 {len(g)} 个最优 k 点，最终能量泛函={e_hist[-1]:.4f}, 平均移动量={m_hist[-1]:.3e}")

    # 楔形积分验证（验证 1409_wedge_integrals）
    e_vec = np.array([2, 1, 0])
    exact = bz_integration.wedge01_monomial_integral(e_vec)
    # Monte Carlo 验证
    pts = bz_integration.wedge01_sample(50000)
    vals = bz_integration.monomial_value(3, pts.shape[1], e_vec, pts)
    mc_est = np.mean(vals) * bz_integration.wedge01_volume()
    print(f"  楔形单项式积分 (e=[2,1,0]): 精确值={exact:.6f}, MC={mc_est:.6f}")

    # 四面体高斯求积（验证 1244_tetrahedron_arbq_rule）
    # 测试积分 f(x,y,z)=1 在标准单形上 -> 结果应为 1/6
    def f_unit(p):
        return np.ones(p.shape[0])
    tet_int = bz_integration.integrate_tetrahedron(f_unit, degree=3)
    print(f"  四面体求积 (f=1): 结果={tet_int:.6f}, 理论=0.166667")

    # Jacobi 多项式与 Gauss-Jacobi 求积（验证 607_jacobi_polynomial）
    x_nodes, w_nodes = bz_integration.gauss_jacobi_quadrature(8, 0.0, 0.0)
    # 积分 f(x)=x^2 在 [-1,1] 上 -> 2/3
    int_x2 = np.sum(w_nodes * x_nodes ** 2)
    print(f"  Gauss-Jacobi (α=β=0, n=8) 积分 x^2: {int_x2:.6f}, 理论=0.666667")

    # BZ 积分示例：态密度
    t, tp, mu = 1.0, 0.3, 0.0
    def dos_integrand(kpts):
        kx = kpts[:, 0]
        ky = kpts[:, 1]
        eps = -2.0 * t * (np.cos(kx) + np.cos(ky)) + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu
        # Lorentzian 近似 delta 函数
        eta = 0.05
        return (1.0 / np.pi) * eta / (eps ** 2 + eta ** 2)

    dos = bz_integration.integrate_bz_gauss_legendre_2d(dos_integrand, n_per_dim=40)
    print(f"  BZ 态密度 (η=0.05): N(0) ≈ {dos:.4f}")

    return k_opt


def run_fermi_surface_analysis():
    """任务 3：费米面几何与嵌套向量分析。"""
    print_section("任务 3：费米面几何分析")

    # 构建色散网格
    nk = 64
    kx = np.linspace(-np.pi, np.pi, nk)
    ky = np.linspace(-np.pi, np.pi, nk)
    KX, KY = np.meshgrid(kx, ky)
    eps_grid = tight_binding.dispersion_2d(KX, KY, t=1.0, tp=0.3, mu=0.0)

    # 环形距离统计（验证 007_annulus_distance）
    stats = fermi_surface.fermi_surface_annulus_stats(kx, ky, eps_grid, mu=0.0, delta_e=0.15, n_mc=3000)
    print(f"  费米环质心: ({stats['centroid'][0]:.3f}, {stats['centroid'][1]:.3f})")
    print(f"  等效内外半径: r1={stats['r1']:.3f}, r2={stats['r2']:.3f}")
    print(f"  环上动量距离均值={stats['mean_distance']:.3f}, 方差={stats['variance']:.3f}")

    # 边界追踪（验证 110_boundary_word_square 思想）
    boundary_info = fermi_surface.trace_fermi_surface_boundary(eps_grid, kx, ky, mu=0.0)
    print(f"  费米面边界多边形: 点数={boundary_info['boundary_points'].shape[0]}")
    print(f"  近似面积={boundary_info['area_approx']:.4f}, 近似质心=({boundary_info['centroid_approx'][0]:.3f}, {boundary_info['centroid_approx'][1]:.3f})")
    print(f"  近似转动惯量={boundary_info['moment_approx']:.4f}")

    # 嵌套向量
    q_nest, overlap = fermi_surface.fermi_surface_nesting_vector(kx, ky, eps_grid, mu=0.0, delta_e=0.1)
    print(f"  最优嵌套向量 Q=({q_nest[0]:.3f}, {q_nest[1]:.3f}), 重叠度={overlap:.3f}")

    return eps_grid, kx, ky


def run_spectral_analysis():
    """任务 4：Haar 小波多分辨率谱分析。"""
    print_section("任务 4：Haar 小波谱分析")

    # 构造一个具有多尺度结构的序参量场（模拟 CDW+PDW 共存）
    N = 128
    x = np.linspace(0, 4 * np.pi, N)
    y = np.linspace(0, 4 * np.pi, N)
    X, Y = np.meshgrid(x, y)
    # 叠加长波长（配对）+ 短波长（电荷密度波）
    Delta_field = (1.0 * np.sin(X) * np.sin(Y)
                   + 0.3 * np.sin(4 * X) * np.sin(4 * Y)
                   + 0.1 * np.random.randn(N, N))

    # 完全 2D Haar 变换（验证 496_haar_transform）
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
    """任务 5：TDGL 动力学、波动方程与混沌分析。"""
    print_section("任务 5：时间依赖动力学")

    # 波动方程 Goldstone 模式（验证 1402_wave_pde）
    t_wave, u_wave, v_wave, energy_wave = tdgl_dynamics.solve_wave_equation_mol(
        nx=32, c=1.0, t_span=(0.0, 2.0 * np.pi), nt_eval=50
    )
    E_conservation = np.max(np.abs(energy_wave - energy_wave[0]))
    print(f"  波动方程能量守恒偏差: {E_conservation:.3e}")

    # Chen 混沌系统（验证 168_chen_ode）
    t_chen, y_chen, lyap = tdgl_dynamics.solve_chen_system(
        t_span=(0.0, 10.0), nt_eval=1000
    )
    print(f"  Chen 系统 Lyapunov 指数估计: {lyap:.4f}")
    print(f"  终态: x={y_chen[-1, 0]:.3f}, y={y_chen[-1, 1]:.3f}, z={y_chen[-1, 2]:.3f}")

    # TDGL 演化
    t_tdgl, Delta_tdgl, max_amp = tdgl_dynamics.tdgl_evolution_1d(
        Nx=64, L=10.0, T=1.0, Tc=1.5, tau=1.0, xi=1.0,
        b_coeff=1.0, t_span=(0.0, 15.0), nt=300
    )
    rho_s = tdgl_dynamics.compute_superfluid_stiffness(Delta_tdgl, dx=10.0 / 64, dt=t_tdgl[1] - t_tdgl[0])
    print(f"  TDGL 稳态最大振幅: {max_amp[-1]:.4f}")
    print(f"  估算超流刚度 ρ_s: {rho_s:.4f}")

    return Delta_tdgl


def run_percolation_analysis(Delta_field):
    """任务 6：渗流相干性分析。"""
    print_section("任务 6：渗流与超导相干性")

    # 对 TDGL 终态做渗流分析（验证 865_percolation_simulation）
    # Delta_field 形状为 (nt, Nx)，需要 reshape 为二维空间场
    nt, nx = Delta_field.shape
    ny = int(np.sqrt(nx))
    if ny * ny == nx:
        field_2d = Delta_field[-1, :].reshape(ny, ny)
    else:
        # 不能完美开方，则用 tile 构造一个合理的二维场
        ny = max(8, nx // 8)
        if nx % ny == 0:
            field_2d = Delta_field[-1, :].reshape(ny, nx // ny)
        else:
            # 截取可整除部分
            nx_use = (nx // ny) * ny
            field_2d = Delta_field[-1, :nx_use].reshape(ny, nx_use // ny)

    analysis = percolation_coherence.superconducting_percolation_analysis(field_2d, threshold=0.3)

    print(f"  超导占据填充率: {analysis['filling_fraction']:.3f}")
    print(f"  连通团簇数: {analysis['component_num']}")
    print(f"  平均团簇尺寸: {analysis['mean_cluster_size']:.1f}")
    print(f"  最大团簇尺寸: {analysis['largest_cluster_size']}")
    print(f"  是否存在跨越团簇: {'是' if analysis['spanning'] else '否'}")

    # 纯渗流阈值估计
    p_c_est = percolation_coherence.find_percolation_threshold(m=50, n=50, n_trials=12)
    print(f"  数值估计渗流阈值 p_c ≈ {p_c_est:.4f} (理论≈0.5927)")

    return analysis


def run_bcs_gap_and_phase_diagram():
    """任务 7：d-波 BCS 能隙自洽求解与相图。"""
    print_section("任务 7：d-波 BCS 能隙方程")

    U = 8.0
    T = 0.05
    beta = 1.0 / max(T, 1e-6)
    t, tp, mu = 1.0, 0.3, 0.0

    # 自洽求解能隙（验证 897_polynomial_root_bound 的 bracket 思想）
    Delta_sc, history, converged = bcs_gap_solver.solve_gap_self_consistent(
        U=U, beta=beta, t=t, tp=tp, mu=mu,
        Delta_max=5.0, n_k=32, tol=1e-7, max_iter=80
    )
    print(f"  参数: U={U}, T={T}, t'={tp}, μ={mu}")
    print(f"  自洽 d-波能隙 Δ_0 = {Delta_sc:.6f}, 收敛={'是' if converged else '否'}, 迭代次数={len(history)}")

    # 计算临界温度
    Tc = bcs_gap_solver.compute_critical_temperature(
        U=U, t=t, tp=tp, mu=mu, beta_max=200.0, n_k=32
    )
    print(f"  估算临界温度 T_c = {Tc:.4f}")

    # 自由能
    F_sc = bcs_gap_solver.compute_free_energy(Delta_sc, U, beta, t=t, tp=tp, mu=mu, n_k=24)
    F_n = bcs_gap_solver.compute_free_energy(0.0, U, beta, t=t, tp=tp, mu=mu, n_k=24)
    print(f"  超导态自由能 F_s = {F_sc:.6f}")
    print(f"  正常态自由能 F_n = {F_n:.6f}")
    print(f"  自由能差 ΔF = F_s - F_n = {F_sc - F_n:.6f} (负值表示超导稳定)")

    # Box-Behnken 参数探索（验证 111_box_behnken）
    design = bcs_gap_solver.box_behnken_parameter_sweep(
        U_range=[1.0, 4.0],
        T_range=[0.05, 0.5],
        tp_range=[0.0, 0.5],
        mu_range=[-1.0, 1.0]
    )
    print(f"  Box-Behnken 4 因子实验设计: 生成 {design.shape[1]} 个参数组合点")
    # 对前 3 个点快速计算能隙
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
    """任务 8：边界词多连方几何演示（验证 110_boundary_word_square）。"""
    print_section("任务 8：边界词几何")

    # 构造一个 3x3 正方形的边界词
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

    # 执行全部任务管线
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
    main()

# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: polynomial_root_bound 对 z^2-3z+2 根界覆盖最大根 ----
bound = utils.polynomial_root_bound([1.0, -3.0, 2.0])
assert bound >= 2.0, '[TC01] polynomial_root_bound 根界 FAILED'

# ---- TC02: box_behnken_size 三维公式验证 ----
size_3 = utils.box_behnken_size(3)
assert size_3 == 1 + 3 * 2 ** 2, '[TC02] box_behnken_size dim=3 FAILED'

# ---- TC03: boundary_word_area 3x3 正方形面积为 9 ----
area = utils.boundary_word_area("rrrdddllluuu")
assert abs(area - 9.0) < 1e-12, '[TC03] boundary_word_area 3x3 FAILED'

# ---- TC04: safe_sqrt 负输入返回非负有限值 ----
val = utils.safe_sqrt(-1.0)
assert val >= 0.0 and np.isfinite(val), '[TC04] safe_sqrt 负输入 FAILED'

# ---- TC05: fermi_dirac 零能高温极限近似 0.5 ----
fd = utils.fermi_dirac(0.0, 1e-10, mu=0.0)
assert abs(fd - 0.5) < 1e-6, '[TC05] fermi_dirac 高温极限 FAILED'

# ---- TC06: 紧束缚哈密顿量维度正确且厄米 ----
H = tight_binding.build_tight_binding_hamiltonian(4, 4, t=1.0, tp=0.3, mu=0.0)
assert H.shape == (16, 16), '[TC06] 哈密顿量维度 FAILED'
assert tight_binding.validate_hamiltonian_hermiticity(H, tol=1e-10), '[TC06] 哈密顿量厄米性 FAILED'

# ---- TC07: 色散关系在 Gamma 点解析值验证 ----
eps0 = tight_binding.dispersion_2d(0.0, 0.0, t=1.0, tp=0.3, mu=0.0)
assert abs(eps0 - (-4.0 + 1.2)) < 1e-12, '[TC07] dispersion_2d Gamma点 FAILED'

# ---- TC08: CCS 稠密-稀疏往返转换精度 ----
A_test = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
nz, cp, ri, va = tight_binding.dense_to_ccs(A_test)
H_back = tight_binding.ccs_to_csr(cp, ri, va, 3, 3)
assert np.linalg.norm(A_test - H_back.toarray()) < 1e-12, '[TC08] CCS往返转换 FAILED'

# ---- TC09: 自然三次样条在节点处精确插值 ----
xd = np.array([0.0, 1.0, 2.0, 3.0])
yd = np.array([1.0, 2.0, 1.5, 0.0])
ys = utils.natural_cubic_spline(xd, yd, xd)
assert np.max(np.abs(ys - yd)) < 1e-10, '[TC09] 自然三次样条节点精度 FAILED'

# ---- TC10: 楔形单项式积分 e=[0,0,0] 等于体积 1 ----
val = bz_integration.wedge01_monomial_integral([0, 0, 0])
assert abs(val - 1.0) < 1e-12, '[TC10] wedge01_monomial_integral 体积 FAILED'

# ---- TC11: Gauss-Jacobi alpha=beta=0 n=2 积分常数精确为 2 ----
x, w = bz_integration.gauss_jacobi_quadrature(2, 0.0, 0.0)
assert abs(np.sum(w) - 2.0) < 1e-12, '[TC11] Gauss-Jacobi 权重和 FAILED'

# ---- TC12: Jacobi 多项式 P0 在所有点恒为 1 ----
p0 = bz_integration.jacobi_polynomial_eval(3, 0, 0.0, 0.0, np.array([-0.5, 0.0, 0.5]))
assert np.allclose(p0[:, 0], 1.0), '[TC12] Jacobi P0 FAILED'

# ---- TC13: 四面体高斯求积 f=1 等于标准单形体积 1/6 ----
def f_one(p): return np.ones(p.shape[0])
tet_val = bz_integration.integrate_tetrahedron(f_one, degree=3)
assert abs(tet_val - 1.0 / 6.0) < 1e-12, '[TC13] 四面体积分体积 FAILED'

# ---- TC14: 一维 Haar 正逆变换往返误差为零 ----
u = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
c = spectral_analysis.haar_1d(u)
u_back = spectral_analysis.haar_1d_inverse(c)
assert np.max(np.abs(u - u_back)) < 1e-12, '[TC14] 1D Haar 正逆变换 FAILED'

# ---- TC15: 二维 Haar 正逆变换往返误差为零 ----
A = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
              [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
C = spectral_analysis.haar_2d(A)
A_back = spectral_analysis.haar_2d_inverse(C)
assert np.max(np.abs(A - A_back)) < 1e-12, '[TC15] 2D Haar 正逆变换 FAILED'

# ---- TC16: 小波能量谱非负且总能量守恒 ----
A = np.ones((4, 4))
C = spectral_analysis.haar_2d(A)
spectrum = spectral_analysis.wavelet_energy_spectrum(C)
assert np.all(spectrum >= 0), '[TC16] 能量谱非负 FAILED'
assert abs(np.sum(spectrum) - np.sum(A ** 2)) < 1e-10, '[TC16] 能量谱守恒 FAILED'

# ---- TC17: 渗流 p=0 时无占据团簇 ----
np.random.seed(42)
res = percolation_coherence.percolation_simulation(10, 10, 0.0, seed=42)
assert res['component_num'] == 0, '[TC17] 渗流 p=0 团簇数 FAILED'

# ---- TC18: 渗流 p=1 时单一团簇且存在跨越 ----
np.random.seed(42)
res = percolation_coherence.percolation_simulation(5, 5, 1.0, seed=42)
assert res['component_num'] == 1, '[TC18] 渗流 p=1 团簇数 FAILED'
assert res['spanning'] is True, '[TC18] 渗流 p=1 跨越 FAILED'

# ---- TC19: d-wave 形状因子在 (pi,0) 处为 -1 ----
phi = bcs_gap_solver.d_wave_form_factor(np.pi, 0.0)
assert abs(phi - (-1.0)) < 1e-12, '[TC19] d-wave 形状因子 FAILED'

# ---- TC20: CVTM 生成元数量正确且位于单位区间内 ----
np.random.seed(42)
g, e_hist, m_hist = optimal_sampling.cvtm_1d(g_num=8, it_num=5, s_num=1000, seed=42)
assert g.size == 8, '[TC20] CVTM 生成元数量 FAILED'
assert np.all((g >= 0.0) & (g <= 1.0)), '[TC20] CVTM 生成元范围 FAILED'

# ---- TC21: 波动方程能量守恒偏差在可接受范围 ----
np.random.seed(42)
t_w, u_w, v_w, E_w = tdgl_dynamics.solve_wave_equation_mol(nx=16, c=1.0, t_span=(0.0, 1.0), nt_eval=20)
assert np.max(np.abs(E_w - E_w[0])) < 1.0, '[TC21] 波动方程能量守恒 FAILED'

# ---- TC22: Chen 系统在原点处导数为零向量 ----
rhs = tdgl_dynamics.chen_system_rhs(0.0, np.array([0.0, 0.0, 0.0]))
assert np.allclose(rhs, 0.0), '[TC22] Chen 系统原点导数 FAILED'

# ---- TC23: 准粒子能量 Delta=0 时退化为裸能谱绝对值 ----
eps_test = bcs_gap_solver.quasiparticle_energy(0.0, 0.0, 0.0, t=1.0, tp=0.3, mu=0.0)
expected = abs(-4.0 + 1.2)
assert abs(eps_test - expected) < 1e-10, '[TC23] 准粒子能量 Delta=0 FAILED'

# ---- TC24: 对称测试矩阵本征值求解器最大误差足够小 ----
np.random.seed(42)
A_test, Q_test, lam_exact = test_eigen_validation.generate_symmetric_test_matrix(10, lambda_mean=0.0, lambda_std=1.0)
val_res = test_eigen_validation.validate_eigensolver(A_test, lam_exact)
assert val_res['max_abs_err'] < 1e-8, '[TC24] 本征值求解精度 FAILED'

# ---- TC25: 直方图分箱计数总和等于数据长度 ----
data = np.array([0.1, 0.5, 0.9, 1.2, 1.8])
centers, counts = test_eigen_validation.r8vec_bin(data, 3, a_min=0.0, a_max=2.0)
assert np.sum(counts) == data.size, '[TC25] r8vec_bin 计数总和 FAILED'

print('\n全部 25 个测试通过!\n')
