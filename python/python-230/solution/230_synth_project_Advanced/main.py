"""
main.py
=======

统一入口 —— 计算高能物理 Profile Likelihood 与 CLs 上限设定:
高阶有限差分与稳定性分析 (小规模可复现实验)。

本项目融合了 15 个科研种子项目的核心算法, 构建了一个完整的
高能物理统计推断框架, 包括:
- 物理模型 (H → γγ 双光子搜索)
- Profile Likelihood 构造与 Profiling 优化
- 高阶有限差分与 Richardson 外推
- Lebesgue 常数与数值稳定性分析
- CLs 上限设定 (渐近 + 蒙特卡洛)
- μ 扫描与期望限 Band

零参数运行: python main.py

种子项目融合清单:
    [1220] Unruh 热力学 → 探测器效率温度修正
    [846]  参数化热方程 FEM → nuisance 参数空间积分
    [1171] MAGNet 网格采样 → profiling 初始化策略
    [1297] 形式化配置空间 → CLs 离散扫描
    [1020] SoleFlip 量化 → 自适应采样收敛控制
    [752]  网格带宽 → profiling 依赖图稀疏性
    [285]  有向图邻接 → nuisance 依赖图分解
    [658]  Lebesgue 常数 → 差分稳定性分析
    [1435] Zoomin 高阶求根 → Halley/Brent profiling
    [807]  不动点迭代 → profiling 鲁棒化
    [671]  Game of Life → 离散初始化扫描
    [1225] VSC HEOM → 谱分解/质量分辨率展宽
    [229]  立方体求积 → nuisance 空间高维积分
    [166]  Chebyshev Type 2 精确性 → 求积精度验证
    [159]  Chebyshev 插值 → profile likelihood 插值加速
"""

from __future__ import annotations
import sys
import time
import math
import numpy as np


def banner(title: str) -> None:
    """打印章节标题。"""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def sub_banner(title: str) -> None:
    width = 72
    print()
    print("-" * width)
    print(f"  {title}")
    print("-" * width)


# =========================================================================
# 模块 1: 物理模型构建
# =========================================================================
def run_physical_model():
    banner("模块 1: 物理模型构建 (H -> γγ 双光子搜索)")
    from physical_model import (
        make_default_binmodel,
        generate_observed_data,
        unruh_temperature,
        signal_efficiency_unruh,
        effective_mass_resolution,
    )

    model = make_default_binmodel()
    print(f"区间数: {model.n_bins}")
    print(f"Nuisance 参数数: {model.n_nuis}")
    print(f"标称信号 s_nom: {model.s_nom}")
    print(f"标称背景 b_nom: {model.b_nom}")
    print(f"Unruh 加速度: a = {model.accel}")

    # Unruh 温度
    TU = unruh_temperature(model.accel)
    print(f"Unruh 等效温度: T_U = {TU:.6e} GeV")

    # Unruh 修正效率
    eps = signal_efficiency_unruh(
        np.ones(model.n_bins), model.accel, model.c2, model.c4, model.T_beam,
    )
    print(f"Unruh 修正后效率: {np.round(eps, 6)}")

    # 质量分辨率
    sigma_stat = 1.5 * np.ones(model.n_bins)
    sigma_JES = 0.01 * np.ones(model.n_bins)
    theta_JES = 0.5
    sigma_eff = effective_mass_resolution(sigma_stat, sigma_JES, model.masses, theta_JES)
    print(f"有效质量分辨率 (theta_JES=0.5): {np.round(sigma_eff, 4)}")

    # 期望事件率
    theta0 = np.zeros(model.n_nuis)
    nu_sig = model.expected_rate(1.0, theta0)
    nu_bkg = model.expected_rate(0.0, theta0)
    print(f"\nmu=1 期望事件率: {np.round(nu_sig, 2)}")
    print(f"mu=0 期望事件率: {np.round(nu_bkg, 2)}")

    # 生成伪数据
    data = generate_observed_data(model, mu_true=0.0, seed=230)
    print(f"\n伪观测数据 (b-only, seed=230): {data.astype(int)}")

    # Renyi 噪声下界
    S2 = model.renyi_entropy_floor(1.0, theta0, alpha=2.0)
    S1 = model.renyi_entropy_floor(1.0, theta0, alpha=1.0)
    print(f"\nRenyi 噪声下界 (alpha=1, Shannon): {S1:.4f}")
    print(f"Renyi 噪声下界 (alpha=2, collision): {S2:.4f}")

    return model, data


# =========================================================================
# 模块 2: 似然函数与 Profiling
# =========================================================================
def run_likelihood(model, data):
    banner("模块 2: 似然函数与全局拟合")
    from likelihood import (
        nll_poisson_gaussian,
        nll_gradient_theta,
        nll_hessian_theta_diag,
        global_best_fit,
        profile_likelihood_ratio,
    )

    theta0 = np.zeros(model.n_nuis)
    nll_bg = nll_poisson_gaussian(model, data, 0.0, theta0)
    nll_sig = nll_poisson_gaussian(model, data, 1.0, theta0)
    print(f"NLL(mu=0, theta=0): {nll_bg:.4f}")
    print(f"NLL(mu=1, theta=0): {nll_sig:.4f}")

    grad = nll_gradient_theta(model, data, 1.0, theta0)
    print(f"grad_theta NLL at (mu=1, theta=0): {np.round(grad, 6)}")

    Hdiag = nll_hessian_theta_diag(model, data, 1.0, theta0)
    print(f"diag(H) at (mu=1, theta=0): {np.round(Hdiag, 4)}")

    # 全局拟合
    mu_hat, nll_min, theta_hat = global_best_fit(model, data)
    print(f"\n全局最优拟合:")
    print(f"  mu_hat = {mu_hat:.6f}")
    print(f"  theta_hat = {np.round(theta_hat, 6)}")
    print(f"  NLL_min = {nll_min:.6f}")

    # Profile likelihood ratio
    for mu_test in [0.5, 1.0, 2.0, 5.0]:
        lam, qmu = profile_likelihood_ratio(
            model, data, mu_test, mu_hat, nll_min,
        )
        print(f"  lambda(mu={mu_test:.1f}) = {lam:.6f}, q_mu = {qmu:.4f}")

    return mu_hat, nll_min, theta_hat


# =========================================================================
# 模块 3: 高阶有限差分与 Richardson 外推
# =========================================================================
def run_finite_diff(model, data, mu_hat, nll_min):
    banner("模块 3: 高阶有限差分与 Richardson 外推")
    from finite_diff import FiniteDiff, fornberg_weights, complex_step_derivative
    from richardson import RichardsonExtrapolator, richardson_fd, optimal_step_size

    # Profile NLL 作为 mu 的函数
    from likelihood import profile_nll
    def profiled_nll_mu(mu_val):
        nll_val, _ = profile_nll(model, data, mu_val)
        return nll_val

    # 有限差分求 d(NLL)/dmu
    mu_test = 1.0

    for fd_order in [2, 4, 6]:
        fd = FiniteDiff(m=1, fd_order=fd_order)
        h = 0.01
        deriv = fd(profiled_nll_mu, mu_test, h)
        print(f"  O(h^{fd_order}) 中心差分: d(NLL)/dmu|_{{mu=1}} = {deriv:.8f}")

    # Richardson 外推
    best, err, conv = richardson_fd(
        profiled_nll_mu, mu_test, m=1, h0=0.1, max_level=8, tol=1e-12,
    )
    print(f"\nRichardson 外推结果: d(NLL)/dmu = {best:.12f}")
    print(f"  估计误差: {err:.2e}")
    print(f"  收敛: {conv}")

    # 最优步长
    h_opt = optimal_step_size(profiled_nll_mu, mu_test, m=1)
    print(f"  理论最优步长: h* = {h_opt:.2e}")

    # Fornberg 权重 (非等距)
    nodes = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    w = fornberg_weights(nodes, 0.0, 1)
    print(f"\nFornberg 权重 (非等距 5 节点): {np.round(w, 6)}")

    # 复步微分 (解析延拓)
    df_cs = complex_step_derivative(lambda z: np.sin(z), 0.5)
    print(f"\n复步微分 d/dmu sin(mu)|_{{mu=0.5}} = {df_cs:.15f}")
    print(f"精确值 cos(0.5) = {math.cos(0.5):.15f}")


# =========================================================================
# 模块 4: 稳定性分析
# =========================================================================
def run_stability():
    banner("模块 4: 数值稳定性分析 (Lebesgue 常数与条件数)")
    from stability import (
        lebesgue_constant,
        chebyshev_nodes,
        equidistant_nodes,
        fd_condition_number,
        StabilityAnalyzer,
        stability_score,
    )
    from finite_diff import FiniteDiff

    # Lebesgue 常数对比
    print("Lebesgue 常数 Lambda_n (节点稳定性度量):")
    for n in [5, 10, 15, 20]:
        nodes_eq = equidistant_nodes(n)
        nodes_ch = chebyshev_nodes(n, kind=2)
        lam_eq = lebesgue_constant(nodes_eq)
        lam_ch = lebesgue_constant(nodes_ch)
        print(f"  n={n:2d}: 等距 Lambda={lam_eq:10.2f}, Chebyshev Lambda={lam_ch:.4f}, "
              f"比值={lam_eq / lam_ch:.1f}")

    # 差分条件数
    print("\n有限差分权重条件数 kappa = Sum|w_j|:")
    for fd_order in [2, 4, 6]:
        fd = FiniteDiff(m=1, fd_order=fd_order)
        kappa = fd_condition_number(fd.weights, 1)
        print(f"  O(h^{fd_order}) 一阶差分: kappa = {kappa:.4f}")
        fd2 = FiniteDiff(m=2, fd_order=fd_order)
        kappa2 = fd_condition_number(fd2.weights, 2)
        print(f"  O(h^{fd_order}) 二阶差分: kappa = {kappa2:.4f}")

    # 稳定性扫描
    f_test = lambda x: math.sin(x)
    x0 = 0.5
    exact = math.cos(x0)
    sa = StabilityAnalyzer(fd_order=4, deriv_order=1)
    results = sa.scan(f_test, x0, exact)
    h_opt, min_err, width = sa.find_stable_region(
        results["rel_error"], results["h"],
    )
    score = stability_score(results["rel_error"], results["h"])
    print(f"\n稳定性扫描结果 (sin(x) 在 x=0.5):")
    print(f"  最优 h = {h_opt:.2e}")
    print(f"  最小相对误差 = {min_err:.2e}")
    print(f"  稳定平台宽度 = {width:.2f} decades")
    print(f"  稳定性评分 = {score:.1f}/100")


# =========================================================================
# 模块 5: Profiling 优化器
# =========================================================================
def run_profiling(model, data):
    banner("模块 5: Nuisance Profiling 优化器")
    from profiling_optimizer import (
        build_nuisance_dependency_graph,
        adjacency_to_blocks,
        discrete_initialization_scan,
        halley_profiling_step,
        fixed_point_profiling,
        profile_theta,
    )

    mu_test = 1.0

    # 依赖图
    A = build_nuisance_dependency_graph(model)
    print(f"Nuisance 依赖图邻接矩阵:")
    print(f"{A}")
    blocks = adjacency_to_blocks(A)
    print(f"连通分量 (独立子问题): {blocks}")
    print(f"  可将 profiling 分解为 {len(blocks)} 个独立子问题")

    # 离散初始化扫描
    theta_scan = discrete_initialization_scan(model, data, mu_test)
    print(f"\n离散初始化扫描 (seed 671 -> CA 式): theta_init = {theta_scan}")

    # Halley 迭代
    theta_halley = theta_scan.copy()
    for step in range(5):
        theta_halley, max_upd = halley_profiling_step(
            model, data, mu_test, theta_halley,
        )
        if max_upd < 1e-10:
            print(f"  Halley 收敛于第 {step + 1} 步 (max_upd={max_upd:.2e})")
            break
    print(f"  Halley 结果: theta = {np.round(theta_halley, 8)}")

    # 不动点迭代
    theta_fp, fp_iters, fp_res = fixed_point_profiling(
        model, data, mu_test, theta_halley,
    )
    print(f"  不动点迭代: {fp_iters} 步, 残差 = {fp_res:.2e}")
    print(f"  FP 结果: theta = {np.round(theta_fp, 8)}")

    # 混合 profiling
    theta_final, nll_final, info = profile_theta(model, data, mu_test)
    print(f"\n混合 Profiling 最终结果 (mu={mu_test}):")
    print(f"  theta_hat_mu = {np.round(theta_final, 8)}")
    print(f"  Profiled NLL = {nll_final:.8f}")
    print(f"  Halley 步数: {info['halley_iters']}")
    print(f"  FP 步数: {info['fp_iters']}")


# =========================================================================
# 模块 6: 渐近 CLs 与 mu 扫描
# =========================================================================
def run_cls_scan(model, data, mu_hat, nll_min):
    banner("模块 6: 渐近 CLs 与 mu 扫描")
    from scan import (
        scan_profile_likelihood,
        scan_cls_asymptotic,
        find_cls_crossing,
        compute_expected_band,
    )
    from asymptotic import asymptotic_cls, estimate_sigma, asimov_q_mu

    # sigma 估计
    sigma = estimate_sigma(model, mu=1.0)
    q_A_1 = asimov_q_mu(model, mu=1.0)
    print(f"Asimov q_{{mu=1,A}} = {q_A_1:.4f}")
    print(f"估计 sigma = {sigma:.4f}")

    # mu 扫描
    mu_grid = np.linspace(0.0, 8.0, 40)
    mu_vals, cls_vals, p_sb, p_b, scan_info = scan_cls_asymptotic(
        model, data, mu_grid,
    )

    print(f"\nCLs 扫描结果 (部分):")
    print(f"  {'mu':>6s}  {'CLs':>10s}  {'p_sb':>12s}  {'1-p_b':>12s}")
    for i in range(0, len(mu_vals), 5):
        print(f"  {mu_vals[i]:6.2f}  {cls_vals[i]:10.6f}  "
              f"{p_sb[i]:12.4e}  {1 - p_b[i]:12.6f}")

    # 95% CL 上限
    mu_up, up_info = find_cls_crossing(model, data, alpha=0.05)
    print(f"\n95% CL 观测上限: mu_up = {mu_up:.4f} ({up_info['status']})")

    # 期望限 Band
    band = compute_expected_band(model, mu_grid)
    print(f"\n期望限 (Asimov b-only):")
    print(f"  Median: {band['mu_up_median']:.4f}")
    if "mu_up_plus_1sigma" in band:
        print(f"  +1sigma band: {band['mu_up_plus_1sigma']:.4f}")
        print(f"  -1sigma band: {band['mu_up_minus_1sigma']:.4f}")
        print(f"  +2sigma band: {band['mu_up_plus_2sigma']:.4f}")
        print(f"  -2sigma band: {band['mu_up_minus_2sigma']:.4f}")


# =========================================================================
# 模块 7: 蒙特卡洛 CLs (小规模验证)
# =========================================================================
def run_toy_mc(model, data):
    banner("模块 7: 蒙特卡洛 CLs (小规模可复现)")
    from cls_calculator import ToyMCCLs
    from asymptotic import asymptotic_cls, estimate_sigma, asimov_q_mu

    sigma = estimate_sigma(model, mu=1.0)

    # 单点 MC
    mu_test = 2.0
    toy = ToyMCCLs(model, data, n_toys_sb=1000, n_toys_b=1000, seed=230)
    cls_mc, p_sb_mc, p_b_mc, info_mc = toy.compute_cls(mu_test, quick=True)
    print(f"MC CLs(mu={mu_test}) = {cls_mc:.6f}")
    print(f"  p_sb (MC) = {p_sb_mc:.6e}")
    print(f"  1 - p_b (MC) = {1 - p_b_mc:.6f}")

    # 渐近对照
    q_A = asimov_q_mu(model, mu=mu_test)
    cls_asy, p_sb_asy, p_b_asy = asymptotic_cls(q_A, mu_test, sigma)
    print(f"\n渐近 CLs(mu={mu_test}) = {cls_asy:.6f} (对照)")
    print(f"  p_sb (渐近) = {p_sb_asy:.6e}")
    print(f"  1 - p_b (渐近) = {1 - p_b_asy:.6f}")
    print(f"\nMC / 渐近 偏差: |Delta CLs| = {abs(cls_mc - cls_asy):.6f}")


# =========================================================================
# 模块 8: 数值工具验证
# =========================================================================
def run_numerical_tools():
    banner("模块 8: 数值工具验证 (Chebyshev, 求积, FEM)")
    from numerical_tools import (
        chebyshev_interpolant,
        chebyshev2_exactness_test,
        cube_quadrature_3d,
        fem_parameter_integral,
    )

    # Chebyshev 插值 profile likelihood
    f_test = lambda x: math.exp(-0.5 * x * x)
    interp = chebyshev_interpolant(f_test, n=20, a=-4.0, b=4.0)
    x_test_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    print("Chebyshev 插值精度 (n=20, exp(-x^2/2)):")
    for x in x_test_vals:
        exact = math.exp(-0.5 * x * x)
        approx = interp(x)
        print(f"  x={x:5.1f}: 精确={exact:.10f}, 插值={approx:.10f}, "
              f"误差={abs(approx - exact):.2e}")

    # Gauss-Chebyshev Type 2 精确度
    results = chebyshev2_exactness_test(n=10, max_degree=25)
    max_exact_degree = 0
    for k, exact, quad, rel_err in results:
        if rel_err < 1e-10:
            max_exact_degree = k
    print(f"\nGauss-Chebyshev Type 2 (n=10) 精确至 {max_exact_degree} 阶多项式")
    print(f"  (理论: 2n-1 = 19)")

    # 3D 求积
    f_3d = lambda p: p[0]**2 + p[1]**2 + p[2]**2
    val, npts = cube_quadrature_3d(f_3d, degree=3)
    print(f"\n3D 立方体求积 Integral(x^2+y^2+z^2) dV = {val:.8f}")
    print(f"  精确值 = 8.0 (3 * (2/3) * 4 = 8)")

    # FEM 参数积分
    g_gauss = lambda th: math.exp(-0.5 * th[0]**2) / math.sqrt(2 * math.pi)
    val_fem = fem_parameter_integral(g_gauss, n_elements=100, n_nuis=1)
    erf_exact = math.erf(5.0 / math.sqrt(2))
    print(f"\nFEM 1D Gaussian 积分 [-5,5]:")
    print(f"  数值: {val_fem:.10f}")
    print(f"  精确 (erf): {erf_exact:.10f}")


# =========================================================================
# 主程序
# =========================================================================
def main():
    t_start = time.time()
    print()
    print("#" * 72)
    print("#")
    print("#  PROJECT 230: 计算高能物理 Profile Likelihood 与 CLs 上限设定")
    print("#  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("#")
    print("#  融合 15 个科研种子项目的核心算法")
    print("#")
    print("#" * 72)

    # 设置随机种子 (可复现)
    np.random.seed(230)

    # 模块 1: 物理模型
    model, data = run_physical_model()

    # 模块 2: 似然函数
    mu_hat, nll_min, theta_hat = run_likelihood(model, data)

    # 模块 3: 有限差分
    run_finite_diff(model, data, mu_hat, nll_min)

    # 模块 4: 稳定性分析
    run_stability()

    # 模块 5: Profiling 优化器
    run_profiling(model, data)

    # 模块 6: CLs 扫描
    run_cls_scan(model, data, mu_hat, nll_min)

    # 模块 7: 蒙特卡洛验证
    run_toy_mc(model, data)

    # 模块 8: 数值工具
    run_numerical_tools()

    t_end = time.time()
    banner("计算完成")
    print(f"总耗时: {t_end - t_start:.2f} 秒")
    print(f"\n本项目融合了以下 15 个种子项目的核心算法:")
    print(f"  [1220] Unruh 热力学 -> 探测器效率温度修正")
    print(f"  [846]  参数化热方程 FEM -> nuisance 参数空间积分")
    print(f"  [1171] MAGNet 网格采样 -> profiling 初始化")
    print(f"  [1297] 形式化配置空间 -> CLs 离散扫描")
    print(f"  [1020] SoleFlip 量化 -> 自适应采样")
    print(f"  [752]  网格带宽 -> 依赖图稀疏性")
    print(f"  [285]  有向图邻接 -> nuisance 分解")
    print(f"  [658]  Lebesgue 常数 -> 差分稳定性")
    print(f"  [1435] Zoomin 求根 -> Halley/Brent profiling")
    print(f"  [807]  不动点迭代 -> profiling 鲁棒化")
    print(f"  [671]  Game of Life -> 离散初始化扫描")
    print(f"  [1225] VSC HEOM -> 谱分解/分辨率展宽")
    print(f"  [229]  立方体求积 -> nuisance 高维积分")
    print(f"  [166]  Chebyshev Type 2 -> 求积精度验证")
    print(f"  [159]  Chebyshev 插值 -> profile likelihood 加速")
    print()
    print("ALL DONE.")


if __name__ == "__main__":
    main()
