#!/usr/bin/env python3
"""
main.py — PROJECT 266: 1D DFT 能带计算的高阶有限差分与稳定性分析
===================================================================

统一入口: 零参数运行, 完成完整的计算凝聚态科研流程。

    python main.py

本程序实现以下完整的计算流程:

  Phase 1: 物理常数与晶格设置
    - 加载原子单位常数
    - 设置 1D Mathieu 晶体模型参数
    - 生成 Monkhorst-Pack k 点网格

  Phase 2: 高阶有限差分算子构建
    - 构建 2p 阶精度 Laplacian FD 模板
    - 分析色散关系误差
    - 确定稳定性极限

  Phase 3: 势场构造
    - Mathieu 型周期势 V(x) = V₁cos(Gx) + V₂cos(2Gx) + V₃cos(3Gx)
    - Fourier 分析
    - Hartree 势与 XC 势初始化

  Phase 4: Kohn-Sham 方程求解 (直接求解, 非 SCF)
    - 在每个 k 点构建 H(k) = T_FD(k) + V_ext
    - 求解特征值问题 → 能带结构 ε_n(k)

  Phase 5: 自洽场迭代 (DFT)
    - 初始密度猜测
    - KS 求解 → 新密度
    - 密度混合 (线性/Pulay)
    - 收敛判断

  Phase 6: 能带分析与态密度
    - 能隙提取
    - 有效质量计算
    - 群速度
    - Gaussian 展宽态密度

  Phase 7: van Hove 奇点检测 (Chebyshev 代理求根)
    - 构建 dε/dk 的 Chebyshev 展开
    - 伴随矩阵特征值 → 精确根

  Phase 8: Monte Carlo BZ 积分验证
    - 随机/遍历采样对比
    - 与精确参考值比较

  Phase 9: 稳定性分析
    - FD 色散误差
    - 网格收敛性 (Richardson 外推)
    - 条件数分析
    - 概率稳定性 (Monte Carlo)

  Phase 10: 贝叶斯反演
    - 从观测能带重建势场参数
    - MLE 优化
    - Laplace 后验近似

融合 15 个种子项目的核心算法:
  681_line_integrals → BZ 精确积分参考
  040_asa121 → trigamma 函数 (Fermi 热力学)
  350_fd_predator_prey → SCF 动力学稳定性
  302_disk01_rule → Gauss 求积节点
  549_humps → Lorentzian 赝势
  683_line_monte_carlo → BZ Monte Carlo 积分
  559_hypercube_integrals → 高维 BZ 推广
  271_dg1d_advection → 谱 FD 算子, Vandermonde
  742_mcnuggets → 能带占据组合计数
  596_interp_trig → 能带三角插值
  225_cpr → Chebyshev 求根 (van Hove)
  362_fd1d_heat_steady → Hartree/Poisson 求解
  1415_will_you_be_alive → 概率稳定性分析
  1247_vcasasmo_BayRad3D → 贝叶斯反演
  054_asa299 → k 点星枚举
"""

import sys
import os
import numpy as np
import time

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header():
    """打印项目标题"""
    print("=" * 72)
    print("  PROJECT 266: 计算凝聚态 DFT 能带计算")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 72)
    print()


def print_phase(phase_num: int, title: str):
    """打印阶段标题"""
    print(f"\n{'─' * 60}")
    print(f"  Phase {phase_num}: {title}")
    print(f"{'─' * 60}")


def phase1_constants_and_lattice():
    """Phase 1: 物理常数与晶格设置"""
    print_phase(1, "物理常数与晶格设置")

    from physical_constants import (EV_PER_HARTREE, BOHR_RADIUS,
                                     get_default_crystal_params,
                                     fd_coefficients_2nd, PI)
    from lattice import (Lattice1D, monkhorst_pack_grid,
                          kpoint_weights_uniform, enumerate_kpoint_stars,
                          band_occupation_count)

    params = get_default_crystal_params()
    a = params['a_lattice']
    lattice = Lattice1D(a)

    print(f"  晶格常数 a = {a:.4f} Bohr = {a * 0.5292:.4f} Å")
    print(f"  倒格矢 G = {lattice.reciprocal_lattice_vector:.6f} 1/Bohr")
    print(f"  BZ 边界 k_max = {lattice.bz_boundary:.6f} 1/Bohr")
    print(f"  1 Hartree = {EV_PER_HARTREE:.6f} eV")

    # k 点网格
    n_kpoints = params['n_kpoints']
    kpoints = monkhorst_pack_grid(n_kpoints, a)
    weights = kpoint_weights_uniform(n_kpoints)
    print(f"  k 点数: {n_kpoints}")
    print(f"  k 范围: [{kpoints[0]:.4f}, {kpoints[-1]:.4f}] 1/Bohr")

    # k 点星枚举 (源自 054_asa299)
    stars = enumerate_kpoint_stars(4)
    print(f"  k 点星: {stars}")

    # 能带占据计数 (源自 742_mcnuggets)
    n_occ_count = band_occupation_count(
        params['n_electrons'], params['n_bands'], n_kpoints)
    print(f"  能带占据构型数: {n_occ_count}")

    # FD 系数预览
    for order in [1, 2, 3, 4]:
        c = fd_coefficients_2nd(order)
        print(f"  FD {2*order}阶系数: {np.array2string(c, precision=6)}")

    return params, lattice, kpoints, weights


def phase2_fd_operators(fd_order: int, n_grid: int, a: float):
    """Phase 2: 有限差分算子与色散分析"""
    print_phase(2, "高阶有限差分算子与色散分析")

    from fd_operators import (compute_fd_dispersion, fd_stability_limit,
                                vandermonde_1d, gauss_lobatto_nodes)
    from stability_analysis import FDStabilityAnalyzer

    # 色散关系
    kdx, k2_mod, k2_exact = compute_fd_dispersion(fd_order)
    print(f"  FD 半带宽 p = {fd_order}, 精度 = O(dx^{2*fd_order})")
    print(f"  修正波数范围: k²_mod·dx² ∈ [{k2_mod.min():.4f}, {k2_mod.max():.4f}]")

    # 稳定性分析
    analyzer = FDStabilityAnalyzer(fd_order)
    disp = analyzer.dispersion_error_analysis()
    print(f"  1% 误差阈值: kdx = {disp['kdx_max_1pct']:.4f}")
    print(f"  10% 误差阈值: kdx = {disp['kdx_max_10pct']:.4f}")
    print(f"  Nyquist 误差: {disp['error_at_nyquist']:.4e}")
    print(f"  收敛阶数 (实测): {disp['convergence_order']:.1f}")
    print(f"  收敛阶数 (理论): {disp['expected_order']:.1f}")

    # 稳定性极限
    kdx_stable = fd_stability_limit(fd_order)
    print(f"  稳定性极限: kdx_max = {kdx_stable:.4f}")

    # Gauss-Lobatto 节点 (源自 271_dg1d_advection)
    gl_nodes = gauss_lobatto_nodes(5)
    print(f"  5点 Gauss-Lobatto: {np.array2string(gl_nodes, precision=4)}")

    # Vandermonde 矩阵 (源自 271_dg1d_advection)
    V = vandermonde_1d(3, gl_nodes[:4])
    print(f"  Vandermonde 矩阵 (4×4): 条件数 = "
          f"{np.linalg.cond(V):.2e}")

    return disp


def phase3_potential(params: dict, n_grid: int):
    """Phase 3: 势场构造与 Fourier 分析"""
    print_phase(3, "周期势场构造与 Fourier 分析")

    from potential import (MathieuPotential, PseudoPotential,
                            fourier_analysis_potential)

    a = params['a_lattice']

    # Mathieu 势
    pot = MathieuPotential(a, params['V1'], params['V2'], params['V3'])
    dx = a / n_grid
    x_grid = np.arange(n_grid) * dx
    Vx = pot.V(x_grid)

    print(f"  Mathieu 势: V(x) = {params['V1']}·cos(Gx) "
          f"+ {params['V2']}·cos(2Gx) + {params['V3']}·cos(3Gx)")
    print(f"  V_min = {Vx.min():.6f} Ha, V_max = {Vx.max():.6f} Ha")
    print(f"  ⟨V⟩ = {np.mean(Vx):.6f} Ha")

    # Mathieu 参数
    mq = pot.mathieu_parameters()
    print(f"  Mathieu q₁ = {mq['q1']:.6f}, q₂ = {mq['q2']:.6f}")

    # Fourier 分析
    n_indices, V_n = fourier_analysis_potential(pot, 5)
    print(f"  Fourier 系数:")
    for i, (n, v) in enumerate(zip(n_indices, V_n)):
        if abs(v) > 1e-10:
            print(f"    V_{{{n}}} = {v:.6f} Ha")

    # 赝势测试 (源自 549_humps)
    pseudo = PseudoPotential(a, np.array([a / 2]), np.array([1.0]),
                              sigma=0.5, n_images=3)
    V_pseudo = pseudo.V(x_grid)
    print(f"  Lorentzian 赝势: V_max = {V_pseudo.max():.4f} Ha")

    return pot, Vx, x_grid, dx


def phase4_direct_ks_solve(Vx: np.ndarray, kpoints: np.ndarray,
                             params: dict):
    """Phase 4: 直接 KS 求解 (非自洽, 固定外势)"""
    print_phase(4, "Kohn-Sham 方程直接求解 (非自洽)")

    from kohn_sham import solve_all_bands
    from band_structure import BandStructure, analyze_band_structure

    a = params['a_lattice']
    n_grid = params['n_grid']
    fd_order = params['fd_order']
    n_bands = params['n_bands']
    dx = a / n_grid

    t0 = time.time()
    eigenvalues, eigenvectors = solve_all_bands(
        Vx, kpoints, n_grid, dx, fd_order, a, n_bands)
    t_solve = time.time() - t0

    print(f"  网格: {n_grid} 点, FD {2*fd_order} 阶")
    print(f"  k 点: {len(kpoints)}, 能带: {n_bands}")
    print(f"  求解时间: {t_solve:.3f} s")

    # 能带结构
    bs = BandStructure(kpoints, eigenvalues, eigenvectors, a)

    # 分析
    analysis = analyze_band_structure(bs)
    print(f"  能带分析:")
    for ib in range(min(3, len(analysis['band_gaps_X']))):
        print(f"    能带 {ib}→{ib+1} 能隙:")
        print(f"      X 点: {analysis['band_gaps_X'][ib]*1000:.2f} meV")
        print(f"      Γ 点: {analysis['band_gaps_Gamma'][ib]*1000:.2f} meV")
    for ib in range(min(3, len(analysis['bandwidths']))):
        print(f"    能带 {ib} 宽度: "
              f"{analysis['bandwidths'][ib]*1000:.2f} meV")

    return bs, eigenvalues, eigenvectors


def phase5_scf_iteration(pot, params: dict, kpoints: np.ndarray,
                           weights: np.ndarray):
    """Phase 5: 自洽场迭代"""
    print_phase(5, "自洽场 (SCF) 迭代")

    from scf_solver import SCFSolver
    from kohn_sham import solve_all_bands
    from xc_functionals import LDATotal1D

    a = params['a_lattice']
    n_grid = params['n_grid']
    fd_order = params['fd_order']
    n_bands = min(4, params['n_bands'])
    dx = a / n_grid
    x_grid = np.arange(n_grid) * dx
    v_ext = pot.V(x_grid)

    xc = LDATotal1D()

    solver = SCFSolver(
        mixing_alpha=params['mixing_alpha'],
        max_iterations=20,
        convergence_tol=params['scf_tol'],
        mixing_scheme='linear')

    def ks_func(v_eff, kpts):
        return solve_all_bands(
            v_eff, kpts, n_grid, dx, fd_order, a, n_bands)

    result = solver.run_scf(
        ks_func, v_ext, n_grid, dx, fd_order, a,
        params['n_electrons'], n_bands,
        kpoints[:8], weights[:8],
        temperature=params['temperature'],
        hartree_strength=params['hartree_strength'],
        xc_functional=xc,
        verbose=True)

    print(f"  SCF 收敛: {result['converged']}")
    print(f"  迭代次数: {result['n_iterations']}")
    print(f"  Fermi 能: {result['fermi_energy']:.6f} Ha = "
          f"{result['fermi_energy']*27.2114:.4f} eV")

    return result


def phase6_band_analysis(bs, eigenvalues: np.ndarray,
                           params: dict, weights: np.ndarray):
    """Phase 6: 能带分析与态密度"""
    print_phase(6, "能带分析与态密度计算")

    from band_structure import interpolate_band_trig
    from dos import dos_gaussian_broadening, integrated_dos

    a = params['a_lattice']

    # 能带三角插值 (源自 596_interp_trig)
    k_fine = np.linspace(bs.kpoints[0], bs.kpoints[-1], 200)
    for ib in range(min(3, bs.n_bands)):
        e_fine = interpolate_band_trig(
            bs.kpoints, bs.eigenvalues[:, ib], k_fine, a)
        print(f"  能带 {ib} 插值: "
              f"ε ∈ [{e_fine.min():.4f}, {e_fine.max():.4f}] Ha")

    # 态密度
    energy_grid = np.linspace(
        eigenvalues.min() - 0.2, eigenvalues.max() + 0.2, 200)
    dos = dos_gaussian_broadening(eigenvalues, weights, energy_grid,
                                    sigma=0.05)
    idos = integrated_dos(dos, energy_grid)

    # Fermi 能位置
    from xc_functionals import find_fermi_energy
    mu = find_fermi_energy(eigenvalues, weights,
                            params['n_electrons'], 0.001)
    print(f"  Fermi 能 μ = {mu:.6f} Ha")
    print(f"  DOS(E_F) = {np.interp(mu, energy_grid, dos):.4f} "
          f"states/Ha")
    print(f"  IDOS(E_F) = {np.interp(mu, energy_grid, idos):.4f}")

    return dos, energy_grid, mu


def phase7_van_hove_detection(bs, params: dict):
    """Phase 7: van Hove 奇点检测"""
    print_phase(7, "van Hove 奇点检测 (Chebyshev 代理求根)")

    from chebyshev_rootfinder import find_van_hove_singularities
    from band_structure import BandStructure

    a = params['a_lattice']

    # 构造能带函数
    def band_func(band_idx):
        def f(k):
            # 线性插值
            idx = np.searchsorted(bs.kpoints, k)
            if idx == 0:
                return bs.eigenvalues[0, band_idx]
            if idx >= len(bs.kpoints):
                return bs.eigenvalues[-1, band_idx]
            # 线性插值
            k0, k1 = bs.kpoints[idx - 1], bs.kpoints[idx]
            e0 = bs.eigenvalues[idx - 1, band_idx]
            e1 = bs.eigenvalues[idx, band_idx]
            t = (k - k0) / (k1 - k0) if k1 != k0 else 0
            return e0 + t * (e1 - e0)
        return f

    for ib in range(min(3, bs.n_bands)):
        k_vh, e_vh = find_van_hove_singularities(
            band_func(ib), a, ib, n_chebyshev=32)
        print(f"  能带 {ib}: 找到 {len(k_vh)} 个 van Hove 奇点")
        for k, e in zip(k_vh, e_vh):
            print(f"    k = {k:.4f} 1/Bohr, ε = {e:.6f} Ha")


def phase8_monte_carlo_bz(pot, params: dict):
    """Phase 8: Monte Carlo BZ 积分"""
    print_phase(8, "Monte Carlo BZ 积分验证")

    from monte_carlo_bz import (monte_carlo_bz_integral,
                                  sample_bz_random, sample_bz_ergodic,
                                  bz_monomial_reference)
    from kohn_sham import solve_kohn_sham_at_kpoint

    a = params['a_lattice']
    n_grid = 48
    fd_order = 2
    dx = a / n_grid
    x_grid = np.arange(n_grid) * dx
    v_ext = pot.V(x_grid)

    # 测试: BZ 上 ε₁(k) 的积分
    def band_energy_func(kpoints):
        results = []
        for k in kpoints:
            evals, _ = solve_kohn_sham_at_kpoint(
                v_ext, k, n_grid, dx, fd_order, a, 2)
            results.append(evals[0])
        return np.array(results)

    # 精确参考 (使用密集网格)
    k_dense = np.linspace(-np.pi / a, np.pi / a, 64)
    e_dense = band_energy_func(k_dense)
    exact_integral = np.trapz(e_dense, k_dense)
    print(f"  精确积分 (密集网格): {exact_integral:.8f} Ha·Bohr")

    # Monte Carlo
    for method in ['random', 'ergodic']:
        for n_mc in [16, 64, 128]:
            val, err = monte_carlo_bz_integral(
                band_energy_func, n_mc, a, method=method)
            print(f"  MC ({method:8s}, N={n_mc:5d}): "
                  f"I = {val:.6f} ± {err:.6f}")

    # BZ 单式积分参考 (源自 681_line_integrals)
    ref_0 = bz_monomial_reference(0, a)
    ref_2 = bz_monomial_reference(2, a)
    print(f"  BZ 精确参考: ∫dk = {ref_0:.6f}, ∫k²dk = {ref_2:.6f}")


def phase9_stability(params: dict):
    """Phase 9: 稳定性分析"""
    print_phase(9, "数值稳定性与收敛性分析")

    from stability_analysis import (FDStabilityAnalyzer,
                                     grid_convergence_analysis,
                                     hamiltonian_condition_number,
                                     stability_probability_analysis)

    a = params['a_lattice']

    # FD 精度对比
    print("  FD 精度对比:")
    for order in [1, 2, 3, 4]:
        analyzer = FDStabilityAnalyzer(order)
        info = analyzer.dispersion_error_analysis()
        dx_max = analyzer.required_grid_spacing(1.0, 1.0)
        print(f"    {2*order}阶: kdx_1%={info['kdx_max_1pct']:.3f}, "
              f"Nyquist误差={info['error_at_nyquist']:.2e}, "
              f"dx_max(E<1Ha)={dx_max:.4f}")

    # 网格收敛
    print("  网格收敛性:")
    conv = grid_convergence_analysis(
        [32, 64, 128, 256], fd_order=2, a=a, n_bands=2)
    for N, dx, E in zip(conv['n_grids'], conv['dx_values'],
                          conv['energies']):
        print(f"    N={N:4d}, dx={dx:.4f}: E₁ = {E:.8f} Ha")
    print(f"    Richardson 外推: E_∞ = {conv['e_extrapolated']:.8f} Ha")
    print(f"    收敛阶数 (估计): {conv['estimated_order']:.1f}")

    # 条件数
    from potential import MathieuPotential
    from lattice import monkhorst_pack_grid
    pot = MathieuPotential(a)
    n_grid = 64
    dx_grid = a / n_grid
    x_grid = np.arange(n_grid) * dx_grid
    v_ext = pot.V(x_grid)

    cond = hamiltonian_condition_number(
        n_grid, dx_grid, 2, v_ext, 0.0, a)
    print(f"  哈密顿量条件数: κ(H) = {cond['condition_number']:.2e}")
    print(f"  谱范围: [{cond['lambda_min']:.4f}, {cond['lambda_max']:.4f}]")

    # 概率稳定性 (快速版)
    print("  概率稳定性 (3次试验):")
    prob = stability_probability_analysis(
        n_trials=3, n_grid=32, fd_order=2, a=a)
    print(f"    收敛概率: {prob['convergence_probability']:.2f}")
    print(f"    平均迭代: {prob['mean_iterations']:.1f} ± "
          f"{prob['std_iterations']:.1f}")


def phase10_bayesian_inverse(params: dict):
    """Phase 10: 贝叶斯反演"""
    print_phase(10, "贝叶斯反演: 从能带重建势场")

    from bayesian_inverse import run_bayesian_inverse
    from kohn_sham import solve_all_bands
    from lattice import monkhorst_pack_grid

    a = params['a_lattice']
    n_grid = 48
    fd_order = 2
    n_bands = 3

    # "真实" 参数
    true_params = np.array([a, 0.5, 0.15, 0.03])

    # 生成 "观测" 数据
    from potential import MathieuPotential
    pot_true = MathieuPotential(*true_params)
    dx = a / n_grid
    x_grid = np.arange(n_grid) * dx
    v_true = pot_true.V(x_grid)
    kpoints = monkhorst_pack_grid(8, a)
    obs_eigenvalues, _ = solve_all_bands(
        v_true, kpoints, n_grid, dx, fd_order, a, n_bands)

    # 加噪声
    np.random.seed(123)
    obs_eigenvalues += 0.01 * np.random.randn(*obs_eigenvalues.shape)

    print(f"  真实参数: a={true_params[0]:.2f}, "
          f"V₁={true_params[1]:.3f}, "
          f"V₂={true_params[2]:.3f}, "
          f"V₃={true_params[3]:.3f}")

    # 反演
    result = run_bayesian_inverse(
        obs_eigenvalues, true_params,
        n_grid=n_grid, fd_order=fd_order,
        n_kpoints=8, n_bands=n_bands, sigma=0.01)

    est = result['estimated_params']
    err = result['param_error_relative']
    print(f"  反演参数: a={est[0]:.2f}, V₁={est[1]:.3f}, "
          f"V₂={est[2]:.3f}, V₃={est[3]:.3f}")
    print(f"  相对误差: a={err[0]*100:.1f}%, V₁={err[1]*100:.1f}%, "
          f"V₂={err[2]*100:.1f}%, V₃={err[3]*100:.1f}%")
    print(f"  后验标准差: {result['posterior_std']}")
    print(f"  最终对数似然: {result['final_log_likelihood']:.4f}")


def main():
    """主函数: 执行完整的 DFT 能带计算流程"""
    print_header()
    t_start = time.time()

    # Phase 1
    params, lattice, kpoints, weights = phase1_constants_and_lattice()

    # Phase 2
    fd_info = phase2_fd_operators(
        params['fd_order'], params['n_grid'], params['a_lattice'])

    # Phase 3
    pot, Vx, x_grid, dx = phase3_potential(params, params['n_grid'])

    # Phase 4
    bs, eigenvalues, eigenvectors = phase4_direct_ks_solve(
        Vx, kpoints, params)

    # Phase 5
    scf_result = phase5_scf_iteration(pot, params, kpoints, weights)

    # Phase 6
    dos, energy_grid, mu = phase6_band_analysis(
        bs, eigenvalues, params, weights)

    # Phase 7
    phase7_van_hove_detection(bs, params)

    # Phase 8
    phase8_monte_carlo_bz(pot, params)

    # Phase 9
    phase9_stability(params)

    # Phase 10
    phase10_bayesian_inverse(params)

    t_total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"  全部计算完成! 总耗时: {t_total:.2f} s")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
