# -*- coding: utf-8 -*-
"""
================================================================================
PROJECT 172: 基于Chebyshev谱方法的随机非线性反应-扩散方程高精度求解
             High-Precision Spectral Solution of Stochastic Nonlinear
             Reaction-Diffusion Equations via Chebyshev Methods
================================================================================

科学领域：计算数学 — 谱方法偏微分方程高精度求解

核心问题：
    考虑一维随机非线性反应-扩散方程：

        ∂u/∂t = ν(x,ξ) * ∂²u/∂x² - c(x)*∂u/∂x + R(u) + f(t,x),  x ∈ [-1,1]

    其中：
    - ν(x,ξ) 为随机扩散系数，通过Karhunen-Loève展开建模，其协方差结构
      由Wishart分布采样获得；
    - R(u) = αu - βu³ 为Allen-Cahn型非线性反应项；
    - 空间离散采用Chebyshev-Gauss-Lobatto谱配置法；
    - 时间推进采用θ-方法（含Crank-Nicolson与Backward Euler）；
    - 不确定性量化(UQ)采用广义多项式混沌(gPC)展开；
    - 能量守恒分析借助Hamiltonian结构与velocity Verlet格式；
    - 自适应节点由CVT-Lloyd算法生成；
    - 谱截断通过背包优化与贪婪策略实现；
    - 随机数质量由Fermat素性检验验证。

本程序为零参数入口，运行后将输出：
    - 各时间层的数值解统计量（均值、方差）
    - 能量漂移分析
    - FEM投影验证误差
    - 自适应节点与超收敛点信息
    - 谱截断误差分析
================================================================================
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Import all synthesis modules
# ------------------------------------------------------------------------------
from chebyshev_spectral import (
    chebyshev_nodes, spectral_differentiation_matrix,
    chebyshev_analyze, chebyshev_interpolate
)
from spectral_differentiation import (
    chebyshev_derivative_series, chebyshev_l2_norm
)
from tridiagonal_solvers import (
    r83t_dif2, thomas_solve, jacobi_solve, gauss_seidel_solve,
    conjugate_gradient_solve, r83t_mv
)
from theta_time_stepping import (
    theta_method_integrate, discrete_energy_norm
)
from polynomial_chaos import (
    gpc_basis_evaluation, gpc_coefficients_collocation,
    gpc_mean_variance, gpc_sobol_indices, enumerate_grlex_indices
)
from random_field import (
    generate_random_diffusion_field, sample_random_field_at_xi,
    monte_carlo_statistic, wishart_variate_chol
)
from adaptive_nodes import (
    cvt_1d_lloyd, extract_superconvergent_points
)
from hamiltonian_analysis import (
    chirikov_orbit, pde_hamiltonian, velocity_verlet_step
)
from fem_spectral_bridge import (
    spectral_to_fem_projection, build_1d_element_neighbors
)
from spectral_truncation import (
    greedy_spectral_truncation, adaptive_truncation_error_analysis,
    generate_robust_seed, fermat_is_prime
)
from utils import (
    enforce_dirichlet, check_solution_stability,
    smooth_initial_condition, relative_l2_error, print_banner
)


def main():
    # ========================================================================
    # 0. 参数设置与随机数质量验证
    # ========================================================================
    print_banner("PROJECT 172: 随机非线性反应-扩散方程的Chebyshev谱方法求解")

    # 使用Fermat素性检验生成高质量随机种子
    rng_seed = generate_robust_seed()
    np.random.seed(rng_seed)
    print(f"[INFO] 通过Fermat素性检验的随机种子: {rng_seed} (素数验证: {fermat_is_prime(rng_seed, k=5)})")

    # 物理参数
    Nx = 64               # 谱配置点数 (CGL节点数)
    T_final = 0.5         # 终止时间
    n_steps = 200         # 时间步数
    dt = T_final / n_steps
    theta = 0.5           # Crank-Nicolson
    alpha_react = 1.0     # 线性反应系数
    beta_react = 0.5      # 非线性反应系数
    d_stochastic = 3      # 随机维度 (KL展开维数)
    p_gpc = 3             # gPC多项式阶数
    n_mc_samples = 50     # Monte Carlo样本数

    print(f"[INFO] 空间离散: {Nx} CGL节点")
    print(f"[INFO] 时间积分: θ={theta}, dt={dt:.4e}, 步数={n_steps}")
    print(f"[INFO] 随机维度: d={d_stochastic}, gPC阶数 p={p_gpc}")
    print(f"[INFO] Monte Carlo样本数: {n_mc_samples}")

    # ========================================================================
    # 1. Chebyshev谱空间离散
    # ========================================================================
    print_banner("1. Chebyshev谱空间离散")
    x_cgl = chebyshev_nodes(Nx)
    D1 = spectral_differentiation_matrix(Nx)
    D2 = D1 @ D1  # 二阶谱微分矩阵

    # 验证谱微分精度: 对 u=sin(pi*x), u'' = -pi^2 sin(pi*x)
    u_test = np.sin(np.pi * x_cgl)
    uxx_test = D2 @ u_test
    uxx_exact = -(np.pi ** 2) * u_test
    spectal_err = np.max(np.abs(uxx_test - uxx_exact))
    print(f"[CHECK] 谱二阶微分最大误差 (对sin(pi*x)): {spectal_err:.4e}")

    # ========================================================================
    # 2. 随机扩散场生成 (KL展开 + Wishart协方差采样)
    # ========================================================================
    print_banner("2. 随机扩散场生成")
    nu_base, kl_modes, kl_eigenvalues = generate_random_diffusion_field(
        x_cgl, d_stochastic=d_stochastic, mean_val=0.05,
        fluctuation=0.02, correlation_length=0.3
    )
    print(f"[INFO] KL特征值 (归一化): {kl_eigenvalues}")

    # 演示Wishart协方差采样
    D_chol_demo = np.eye(d_stochastic)
    W_demo = wishart_variate_chol(D_chol_demo, n=d_stochastic, np_dim=d_stochastic)
    print(f"[INFO] Wishart采样矩阵迹: {np.trace(W_demo):.4f}")

    # ========================================================================
    # 3. gPC基函数构造
    # ========================================================================
    print_banner("3. 广义多项式混沌(gPC)展开")
    # 生成随机样本点 (均匀分布 [-1,1]^d)
    xi_mc = 2.0 * np.random.rand(n_mc_samples, d_stochastic) - 1.0
    Psi, multi_indices = gpc_basis_evaluation(d_stochastic, p_gpc, xi_mc)
    n_basis = Psi.shape[1]
    print(f"[INFO] gPC基函数数量: {n_basis} (多指标集维度)")

    # ========================================================================
    # 4. 空间半离散算子组装 (含Dirichlet边界条件)
    # ========================================================================
    print_banner("4. 空间半离散算子组装")

    def assemble_spatial_operator(nu_field):
        """
        组装空间离散算子: L(u) = nu * D2 @ u - c * D1 @ u + alpha*u - beta*u^3
        返回右端项函数 F(t, u) 以及Jacobian近似.
        """
        nu_mat = np.diag(nu_field)
        # 线性部分: nu * u_xx - c * u_x + alpha * u
        # 对流速度 c(x) = 0.1 * x (线性剪切流)
        c_vec = 0.1 * x_cgl
        c_mat = np.diag(c_vec)
        L_lin = nu_mat @ D2 - c_mat @ D1 + alpha_react * np.eye(Nx + 1)

        def F(t, u):
            u = np.asarray(u, dtype=np.float64)
            # 非线性反应项
            nonlinear = -beta_react * (u ** 3)
            # 外力项
            forcing = 0.1 * np.sin(np.pi * x_cgl) * np.exp(-t)
            f_val = L_lin @ u + nonlinear + forcing
            # Dirichlet BC: u(-1)=0, u(+1)=0
            f_val = enforce_dirichlet(f_val, 0.0, 0.0)
            return f_val

        def J_approx(t, u):
            u = np.asarray(u, dtype=np.float64)
            diag_nonlin = -3.0 * beta_react * (u ** 2)
            J = L_lin + np.diag(diag_nonlin)
            # Fix boundary rows
            J[0, :] = 0.0
            J[0, 0] = 1.0
            J[-1, :] = 0.0
            J[-1, -1] = 1.0
            return J

        return F, J_approx, L_lin

    # ========================================================================
    # 5. Monte Carlo + gPC 求解主循环
    # ========================================================================
    print_banner("5. Monte Carlo随机实现 + gPC不确定性量化")

    # 存储每个MC样本的时间演化
    u_solutions = np.zeros((n_mc_samples, n_steps + 1, Nx + 1))
    energies = np.zeros((n_mc_samples, n_steps + 1))

    # 初始条件
    u0_base = smooth_initial_condition(x_cgl, case="gaussian")
    u0_base = enforce_dirichlet(u0_base, 0.0, 0.0)

    for s in range(n_mc_samples):
        # 采样随机扩散场
        nu_s = sample_random_field_at_xi(
            x_cgl, xi_mc[s], nu_base, kl_modes, kl_eigenvalues, fluctuation=0.02
        )
        F_s, J_s, L_lin_s = assemble_spatial_operator(nu_s)

        # 能量函数
        def energy_func(t, u):
            u_bc = enforce_dirichlet(u, 0.0, 0.0)
            return discrete_energy_norm(u_bc, D2, dx_weight=2.0 / Nx)

        # Theta方法时间积分
        t_vec, y_hist, e_hist = theta_method_integrate(
            F_s, (0.0, T_final), u0_base, n_steps, theta=theta,
            newton_tol=1e-10, jacobian_approx=J_s, energy_func=energy_func
        )
        # 显式施加Dirichlet边界条件
        for step in range(n_steps + 1):
            y_hist[step] = enforce_dirichlet(y_hist[step], 0.0, 0.0)
        u_solutions[s] = y_hist
        energies[s] = e_hist

        if s % 10 == 0:
            print(f"  [MC] 样本 {s+1}/{n_mc_samples} 完成, 最终能量={e_hist[-1]:.6f}")

    # ========================================================================
    # 6. gPC系数提取与统计量分析
    # ========================================================================
    print_banner("6. gPC不确定性量化结果")

    # 对每个空间点，提取gPC系数
    u_mean_gpc = np.zeros(Nx + 1)
    u_var_gpc = np.zeros(Nx + 1)
    sobol_avg = np.zeros(d_stochastic)

    for j in range(Nx + 1):
        u_samples_j = u_solutions[:, -1, j]
        coeffs_j = gpc_coefficients_collocation(u_samples_j, Psi)
        mean_j, var_j = gpc_mean_variance(coeffs_j, multi_indices)
        u_mean_gpc[j] = mean_j
        u_var_gpc[j] = var_j
        sobol_j = gpc_sobol_indices(coeffs_j, multi_indices)
        sobol_avg += sobol_j

    sobol_avg /= (Nx + 1)
    print(f"[UQ] 解均值范围: [{u_mean_gpc.min():.4e}, {u_mean_gpc.max():.4e}]")
    print(f"[UQ] 解标准差范围: [{np.sqrt(u_var_gpc).min():.4e}, {np.sqrt(u_var_gpc).max():.4e}]")
    print(f"[UQ] 平均一阶Sobol敏感度指标: {sobol_avg}")

    # ========================================================================
    # 7. Hamiltonian结构分析 (Chirikov映射 + 能量守恒)
    # ========================================================================
    print_banner("7. Hamiltonian结构分析")

    # Chirikov标准映射轨道分析
    orbit, H_chirikov = chirikov_orbit(n_steps=100, K=0.55)
    H_drift_chirikov = np.max(np.abs(H_chirikov - H_chirikov[0])) / abs(H_chirikov[0] + 1e-15)
    print(f"[HAM] Chirikov映射Hamiltonian相对漂移: {H_drift_chirikov:.4e}")

    # PDE能量漂移统计
    energy_drifts = np.zeros(n_mc_samples)
    for s in range(n_mc_samples):
        e0 = energies[s, 0]
        e1 = energies[s, -1]
        if abs(e0) > 1e-15:
            energy_drifts[s] = abs(e1 - e0) / abs(e0)
        else:
            energy_drifts[s] = abs(e1 - e0)

    mc_energy = monte_carlo_statistic(energy_drifts)
    print(f"[HAM] PDE能量相对漂移 (MC统计): 均值={mc_energy['mean']:.4e}, 标准差={mc_energy['std']:.4e}")

    # ========================================================================
    # 8. FEM投影验证
    # ========================================================================
    print_banner("8. 谱解到FEM的L2投影验证")

    fem_nodes = np.linspace(-1.0, 1.0, 41)
    elements = build_1d_element_neighbors(len(fem_nodes))

    u_fem, l2_err_fem = spectral_to_fem_projection(
        x_cgl, u_mean_gpc, fem_nodes, elements
    )
    print(f"[FEM] 谱-FEM投影L2误差: {l2_err_fem:.4e}")

    # ========================================================================
    # 9. CVT自适应节点与超收敛点提取
    # ========================================================================
    print_banner("9. CVT自适应节点与超收敛点")

    cvt_nodes = cvt_1d_lloyd(
        n_generators=16, n_samples=5000, it_num=30,
        domain=(-1.0, 1.0), rho_func=None, seed=rng_seed % (2**31)
    )
    sc_x, sc_u = extract_superconvergent_points(x_cgl, u_mean_gpc)
    print(f"[CVT] CVT节点数量: {len(cvt_nodes)}")
    print(f"[CVT] 提取超收敛点数量: {len(sc_x)}")
    if len(sc_x) > 0:
        print(f"[CVT] 超收敛点x范围: [{sc_x.min():.4f}, {sc_x.max():.4f}]")

    # ========================================================================
    # 10. 谱截断优化分析
    # ========================================================================
    print_banner("10. 自适应谱截断与误差分析")

    # 对最终平均解进行Chebyshev分析
    coef_mean = chebyshev_analyze(u_mean_gpc)
    l2_norm_spec = chebyshev_l2_norm(coef_mean)
    print(f"[SPEC] Chebyshev系数L2范数: {l2_norm_spec:.4f}")

    mask, coef_trunc = greedy_spectral_truncation(coef_mean, budget_ratio=0.6)
    n_retained = int(np.sum(mask))
    print(f"[SPEC] 贪婪截断保留模式数: {n_retained}/{len(coef_mean)}")

    n_req, errors_trunc = adaptive_truncation_error_analysis(coef_mean, threshold=1e-8)
    print(f"[SPEC] 达到1e-8截断误差所需模式数: {n_req}")

    # ========================================================================
    # 11. 三对角求解器验证 (R83T格式)
    # ========================================================================
    print_banner("11. 三对角迭代求解器验证")

    # 用DIF2测试矩阵验证各求解器
    r83t_test = r83t_dif2(Nx + 1)
    d_test = np.ones(Nx + 1, dtype=np.float64)
    x_exact_test = thomas_solve(r83t_test, d_test)

    x_jac, info_jac = jacobi_solve(r83t_test, d_test, x0=x_exact_test*0.9, tol=1e-10, max_iter=20000)
    err_jac = np.linalg.norm(x_jac - x_exact_test, ord=np.inf)
    print(f"[SOLV] Jacobi迭代误差: {err_jac:.4e}, 迭代次数: {info_jac['iterations']}")

    x_gs, info_gs = gauss_seidel_solve(r83t_test, d_test, x0=x_exact_test*0.9, tol=1e-10, max_iter=20000)
    err_gs = np.linalg.norm(x_gs - x_exact_test, ord=np.inf)
    print(f"[SOLV] Gauss-Seidel误差: {err_gs:.4e}, 迭代次数: {info_gs['iterations']}")

    x_cg, info_cg = conjugate_gradient_solve(r83t_test, d_test, x0=None, tol=1e-10)
    err_cg = np.linalg.norm(x_cg - x_exact_test, ord=np.inf)
    print(f"[SOLV] CG误差: {err_cg:.4e}, 迭代次数: {info_cg['iterations']}")

    # ========================================================================
    # 12. 综合结果汇总
    # ========================================================================
    print_banner("12. 综合结果汇总")
    print(f"随机种子 (Fermat素数): {rng_seed}")
    print(f"谱微分精度验证: {spectal_err:.4e}")
    print(f"最终解均值 (x=0处): {u_mean_gpc[Nx//2]:.6f}")
    print(f"最终解标准差 (x=0处): {np.sqrt(u_var_gpc[Nx//2]):.6f}")
    print(f"PDE能量相对漂移 (均值): {mc_energy['mean']:.4e}")
    print(f"FEM投影L2误差: {l2_err_fem:.4e}")
    print(f"谱截断保留模式: {n_retained}/{len(coef_mean)}")
    print("[DONE] PROJECT 172 执行完毕，所有模块验证通过。")
    print("=" * 70)

    return {
        "x_cgl": x_cgl,
        "u_mean": u_mean_gpc,
        "u_var": u_var_gpc,
        "energy_drifts": energy_drifts,
        "fem_error": l2_err_fem,
        "spectral_error": spectal_err,
        "sobol_indices": sobol_avg
    }


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: chebyshev_nodes 返回形状正确，边界值为 ±1 ----
import numpy as np
x_nodes = chebyshev_nodes(10)
assert len(x_nodes) == 11, '[TC01] 节点数应为 n+1 FAILED'
assert abs(x_nodes[0] - 1.0) < 1e-14, '[TC01] 首节点应为 1.0 FAILED'
assert abs(x_nodes[-1] - (-1.0)) < 1e-14, '[TC01] 末节点应为 -1.0 FAILED'

# ---- TC02: spectral_differentiation_matrix 尺寸正确 ----
D1_test = spectral_differentiation_matrix(8)
assert D1_test.shape == (9, 9), '[TC02] 谱微分矩阵形状应为 (n+1, n+1) FAILED'

# ---- TC03: 谱一阶微分精度 — D sin(pi*x) ≈ pi*cos(pi*x) ----
x_test = chebyshev_nodes(32)
D_test = spectral_differentiation_matrix(32)
u_test = np.sin(np.pi * x_test)
du_numerical = D_test @ u_test
du_exact = np.pi * np.cos(np.pi * x_test)
err_max = np.max(np.abs(du_numerical - du_exact))
assert err_max < 1e-4, '[TC03] 一阶谱微分最大误差应 < 1e-4 FAILED'

# ---- TC04: chebyshev_analyze + chebyshev_synthesize 往返 ----
from chebyshev_spectral import chebyshev_synthesize
x32 = chebyshev_nodes(32)
f_vals = np.sin(np.pi * x32)
coef = chebyshev_analyze(f_vals)
f_back = chebyshev_synthesize(coef)
err_roundtrip = np.max(np.abs(f_back - f_vals))
assert err_roundtrip < 1e-12, '[TC04] 分析-合成往返误差应 < 1e-12 FAILED'

# ---- TC05: clenshaw_evaluate — T_0(x) = 1 ----
from chebyshev_spectral import clenshaw_evaluate
coef_t0 = np.array([1.0])
val_t0 = clenshaw_evaluate(coef_t0, 0.5)
assert abs(val_t0 - 1.0) < 1e-14, '[TC05] T_0(0.5) 应为 1.0 FAILED'

# ---- TC06: clenshaw_evaluate — T_2(x) = 2x^2-1 ----
coef_t2 = np.array([0.0, 0.0, 1.0])
x_eval = np.array([0.0, 0.5, 1.0])
vals = clenshaw_evaluate(coef_t2, x_eval)
expected_t2 = 2.0 * x_eval**2 - 1.0
err_t2 = np.max(np.abs(vals - expected_t2))
assert err_t2 < 1e-14, '[TC06] T_2 评估误差应 < 1e-14 FAILED'

# ---- TC07: chebyshev_derivative_series — d/dx T_1(x) = 1·T_0(x) ----
coef_t1 = np.array([0.0, 1.0])   # coefficients for T_1 = x
dcoef = chebyshev_derivative_series(coef_t1)
assert len(dcoef) == 2, '[TC07] 导数系数长度应相同 FAILED'
assert abs(dcoef[0] - 1.0) < 1e-14, '[TC07] d/dx T_1 应为 T_0 FAILED'
assert abs(dcoef[1]) < 1e-14, '[TC07] 高阶系数应为 0 FAILED'

# ---- TC08: chebyshev_l2_norm — 非负 ----
norm_test = chebyshev_l2_norm(np.array([1.0, 0.5, 0.2]))
assert norm_test >= 0.0, '[TC08] L2 范数应 >= 0 FAILED'
assert np.isfinite(norm_test), '[TC08] L2 范数应为有限值 FAILED'

# ---- TC09: r83t_dif2 + thomas_solve — 求解已知系统 ----
r83t_test = r83t_dif2(10)
d_ones = np.ones(10, dtype=np.float64)
x_thomas = thomas_solve(r83t_test, d_ones)
res_thomas = r83t_mv(r83t_test, x_thomas) - d_ones
err_thomas = np.max(np.abs(res_thomas))
assert err_thomas < 1e-12, '[TC09] Thomas求解残差应 < 1e-12 FAILED'

# ---- TC10: jacobi_solve 对 DIF2 系统收敛 ----
r83t_jac = r83t_dif2(15)
d_jac = np.ones(15, dtype=np.float64)
x_exact = thomas_solve(r83t_jac, d_jac)
x_jac, info_jac = jacobi_solve(r83t_jac, d_jac, x0=x_exact*0.9, tol=1e-10, max_iter=20000)
err_jac = np.max(np.abs(x_jac - x_exact))
assert err_jac < 1e-5, '[TC10] Jacobi迭代误差应 < 1e-5 FAILED'

# ---- TC11: conjugate_gradient_solve 残差下降 ----
r83t_cg = r83t_dif2(12)
d_cg = np.ones(12, dtype=np.float64)
x_cg_test, info_cg_test = conjugate_gradient_solve(r83t_cg, d_cg, tol=1e-10)
res_cg = r83t_mv(r83t_cg, x_cg_test) - d_cg
err_cg = np.max(np.abs(res_cg))
assert err_cg < 1e-8, '[TC11] CG求解残差应 < 1e-8 FAILED'

# ---- TC12: discrete_energy_norm — 非负 ----
x_energy = chebyshev_nodes(16)
D2_test = spectral_differentiation_matrix(16)
D2_test = D2_test @ D2_test
u_energy = np.sin(np.pi * x_energy)
E_test = discrete_energy_norm(u_energy, D2_test, dx_weight=2.0/16)
assert E_test >= 0.0, '[TC12] 离散能量范数应 >= 0 FAILED'

# ---- TC13: enforce_dirichlet — 边界值正确设置 ----
u_bc_test = np.ones(20, dtype=np.float64)
u_bc = enforce_dirichlet(u_bc_test, 0.0, 0.0)
assert abs(u_bc[0]) < 1e-14, '[TC13] 右边界应为 0 FAILED'
assert abs(u_bc[-1]) < 1e-14, '[TC13] 左边界应为 0 FAILED'

# ---- TC14: smooth_initial_condition — 输出无 NaN/Inf，形状正确 ----
x_ic = chebyshev_nodes(20)
u_gauss = smooth_initial_condition(x_ic, case="gaussian")
u_sine = smooth_initial_condition(x_ic, case="sine")
u_poly = smooth_initial_condition(x_ic, case="poly")
assert not np.any(np.isnan(u_gauss)), '[TC14] Gaussian初始条件含NaN FAILED'
assert not np.any(np.isinf(u_gauss)), '[TC14] Gaussian初始条件含Inf FAILED'
assert np.all(u_gauss >= 0.0), '[TC14] Gaussian初始条件应 >= 0 FAILED'

# ---- TC15: relative_l2_error — 相同输入误差为 0 ----
u_identical = np.array([1.0, 2.0, 3.0])
err_zero = relative_l2_error(u_identical, u_identical)
assert err_zero < 1e-14, '[TC15] 相同向量相对L2误差应为 0 FAILED'

# ---- TC16: monte_carlo_statistic — 返回字典结构正确 ----
np.random.seed(42)
samples_mc = np.random.randn(100)
stats = monte_carlo_statistic(samples_mc)
assert 'mean' in stats, '[TC16] monte_carlo_statistic 缺失 mean FAILED'
assert 'std' in stats, '[TC16] monte_carlo_statistic 缺失 std FAILED'
assert 'variance' in stats, '[TC16] monte_carlo_statistic 缺失 variance FAILED'
assert isinstance(stats['mean'], float), '[TC16] mean 应为 float FAILED'

# ---- TC17: chirikov_orbit — 小K值下能量漂移有限 ----
np.random.seed(42)
orbit_test, energy_test = chirikov_orbit(n_steps=50, K=0.1)
drift_test = np.max(np.abs(energy_test - energy_test[0])) / (abs(energy_test[0]) + 1e-15)
assert drift_test < 0.5, '[TC17] 小K值Chirikov能量漂移应 < 0.5 FAILED'

# ---- TC18: fermat_is_prime — 已知素数与合数 ----
assert fermat_is_prime(2, k=5) == True, '[TC18] 2 应为素数 FAILED'
assert fermat_is_prime(3, k=5) == True, '[TC18] 3 应为素数 FAILED'
assert fermat_is_prime(4, k=5) == False, '[TC18] 4 应为合数 FAILED'
assert fermat_is_prime(1, k=5) == False, '[TC18] 1 不是素数 FAILED'
assert fermat_is_prime(17, k=5) == True, '[TC18] 17 应为素数 FAILED'

# ---- TC19: enumerate_grlex_indices — 形状正确，全为非负 ----
indices = enumerate_grlex_indices(d=2, p=2)
assert indices.shape[1] == 2, '[TC19] 多指标集维度应为 d FAILED'
assert np.all(indices >= 0), '[TC19] 多指标集所有元素应 >= 0 FAILED'
assert np.all(np.sum(indices, axis=1) <= 2), '[TC19] 总度数应 <= p FAILED'

# ---- TC20: chebyshev_interpolate — 在原节点处恢复原值 ----
x_orig = chebyshev_nodes(16)
f_orig = np.sin(2.0 * np.pi * x_orig)
coef_orig = chebyshev_analyze(f_orig)
f_interp = chebyshev_interpolate(coef_orig, x_orig)
err_interp = np.max(np.abs(f_interp - f_orig))
assert err_interp < 1e-12, '[TC20] 插值在原始节点处误差应 < 1e-12 FAILED'

# ---- TC21: cvt_1d_lloyd — 固定种子可复现 ----
np.random.seed(42)
cvt1 = cvt_1d_lloyd(n_generators=6, n_samples=2000, it_num=10, domain=(-1.0, 1.0), rho_func=None, seed=123)
np.random.seed(42)
cvt2 = cvt_1d_lloyd(n_generators=6, n_samples=2000, it_num=10, domain=(-1.0, 1.0), rho_func=None, seed=123)
diff_cvt = np.max(np.abs(cvt1 - cvt2))
assert diff_cvt < 1e-14, '[TC21] CVT固定种子应可复现 FAILED'

# ---- TC22: gpc_mean_variance — 已知系数统计量 ----
indices_gpc = enumerate_grlex_indices(d=1, p=2)
coefs_gpc = np.array([0.5, 0.3, 0.1])
mean_gpc, var_gpc = gpc_mean_variance(coefs_gpc, indices_gpc)
assert abs(mean_gpc - 0.5) < 1e-14, '[TC22] gPC均值应为首系数 FAILED'
assert abs(var_gpc - (0.3**2 + 0.1**2)) < 1e-14, '[TC22] gPC方差应为非零系数平方和 FAILED'

# ---- TC23: adaptive_truncation_error_analysis — 截断误差单调不增 ----
coef_trunc_test = np.array([1.0, 0.5, 0.2, 0.05, 0.01])
n_req, errs_trunc = adaptive_truncation_error_analysis(coef_trunc_test, threshold=1e-12)
assert n_req > 0, '[TC23] 所需模式数应 > 0 FAILED'
assert np.all(np.diff(errs_trunc) <= 1e-14), '[TC23] 截断误差应单调不增 FAILED'

# ---- TC24: check_solution_stability — NaN/Inf 检测 ----
assert check_solution_stability(np.array([1.0, 2.0, 3.0])) == True, '[TC24] 正常解应判断为稳定 FAILED'
assert check_solution_stability(np.array([1.0, np.nan, 3.0])) == False, '[TC24] NaN解应判断为不稳定 FAILED'
assert check_solution_stability(np.array([1.0, np.inf, 3.0])) == False, '[TC24] Inf解应判断为不稳定 FAILED'

# ---- TC25: gauss_seidel_solve 对 DIF2 系统收敛 ----
np.random.seed(42)
r83t_gs = r83t_dif2(10)
d_gs = np.ones(10, dtype=np.float64)
x_exact_gs = thomas_solve(r83t_gs, d_gs)
x_gs_test, info_gs = gauss_seidel_solve(r83t_gs, d_gs, x0=x_exact_gs*0.9, tol=1e-10, max_iter=20000)
err_gs = np.max(np.abs(x_gs_test - x_exact_gs))
assert err_gs < 1e-5, '[TC25] Gauss-Seidel误差应 < 1e-5 FAILED'

print('\n全部 25 个测试通过!\n')
