#!/usr/bin/env python3
"""
main.py
-------
鲁棒优化与不确定约束 —— 统一入口
项目 217 合成项目：数学优化方向的博士级科学计算问题

科学问题：
    模拟移动床 (SMB) 色谱过程的鲁棒优化，考虑：
      - 参数不确定性 (扩散系数、流速、吸附等温线参数)
      - 机会约束 (产品纯度概率保证)
      - PDE 约束 (对流-扩散-吸附方程)
      - 延迟切换动力学 (双稳态)
      - 谱鲁棒性 (特征值灵敏度)

算法流程：
    1. 构造不确定性集合 (椭球/超球)
    2. 定义 SMB 过程模型与目标函数
    3. 求解最坏情况鲁棒优化
    4. 求解机会约束鲁棒优化
    5. 求解均值-方差鲁棒优化
    6. 分析谱鲁棒性
    7. 自适应 ESN 控制验证
    8. 输出结果

运行方式：
    python main.py
"""

from __future__ import annotations
import sys
import os
import numpy as np

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uncertainty_set import (
    EllipsoidalUncertaintySet,
    hyperball_positive_sample,
    noncentral_t_cdf,
    toeplitz_covariance,
    toeplitz_solve,
)
from circulant_kkt import (
    CirculantKKTSystem,
    circulant_solve,
    circulant_mv,
)
from dg_pde_constraint import DG1DHeat
from fem3d_mesh import (
    tet_mesh_generate,
    fem3d_project,
    fem3d_evaluate,
    local_mass_matrix,
    tet_volume,
)
from chance_constraints import (
    ChanceConstraint,
    JointChanceConstraint,
    norm_ppf,
)
from lmc_robust_sampler import HigherOrderLangevin, RobustObjectiveSampler
from delay_bistability import (
    DDAESolver,
    delay_bist_2d_cubic,
    find_equilibrium,
)
from adaptive_esn_controller import (
    AdaptiveRobustController,
    NonlinearPlant,
)
from spectral_robustness import (
    SpectralRobustnessAnalyzer,
    laplacian_2d,
    chladni_eigenmodes,
)
from smb_process import (
    SimulatedMovingBed,
    compute_purity,
    compute_productivity,
)
from uncertain_field import (
    UncertainField,
    KarhunenLoeveExpansion,
    divdif,
    eval_newton,
)
from adjoint_sensitivity import (
    gradient_forward,
    gradient_central,
    gradient_richardson,
    robust_objective_gradient,
    solve_adjoint,
)
from robust_optimizer import (
    RobustOptimizationProblem,
    worst_case_robust_optimize,
    chance_constrained_robust_optimize,
    mean_variance_robust_optimize,
)


# =============================================================================
# 1. 不确定性集合构造
# =============================================================================
def step1_uncertainty_sets():
    print("=" * 70)
    print("步骤 1: 构造不确定性集合")
    print("=" * 70)

    # 椭球集合
    dim = 4
    center = np.zeros(dim)
    E = EllipsoidalUncertaintySet(
        dim=dim, center=center, rho=1.0, toeplitz_rho=0.3, seed=42
    )
    samples = E.sample(20)
    print(f"  椭球集合维度: {dim}")
    print(f"  样本数量: {samples.shape[0]}")
    print(f"  样本均值: {samples.mean(axis=0)}")
    print(f"  最坏情况线性目标: {E.worst_case_linear(np.ones(dim)):.4f}")

    # 超球正象限采样
    u = hyperball_positive_sample(dim)
    print(f"  超球正象限样本: {u}")

    # 非中心 t 分布
    p, ifl = noncentral_t_cdf(2.0, 10.0, 0.5)
    print(f"  非中心 t CDF(2.0, df=10, delta=0.5) = {p:.4f}, ifault={ifl}")

    # Toeplitz 求解
    n = 5
    rho = 0.5
    Sigma = toeplitz_covariance(n, rho)
    b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    first_row = Sigma[0, :]
    x = toeplitz_solve(first_row, b)
    print(f"  Toeplitz 求解: ||x|| = {np.linalg.norm(x):.4f}")

    return E


# =============================================================================
# 2. 循环预条件 KKT 系统
# =============================================================================
def step2_circulant_kkt():
    print("\n" + "=" * 70)
    print("步骤 2: 循环预条件 KKT 系统")
    print("=" * 70)

    n = 16
    first_row = np.zeros(n)
    first_row[0] = 4.0
    first_row[1] = 1.0
    first_row[-1] = 1.0
    b = np.random.default_rng(42).standard_normal(n)

    # FFT 求解
    x_fft = circulant_solve(first_row, b)
    print(f"  循环系统维度: {n}")
    print(f"  FFT 求解 ||x|| = {np.linalg.norm(x_fft):.4f}")

    # KKT 系统
    H_diag = 2.0 * np.ones(n)
    mu = 0.1
    kkt = CirculantKKTSystem(first_row, H_diag, mu)
    r1 = np.random.default_rng(0).standard_normal(n)
    r2 = np.random.default_rng(1).standard_normal(n)
    x_kkt, y_kkt, its = kkt.pcg_solve(r1, r2)
    print(f"  PCG 迭代次数: {its}")
    print(f"  KKT 解 ||x|| = {np.linalg.norm(x_kkt):.4f}")

    return x_fft


# =============================================================================
# 3. DG PDE 约束求解
# =============================================================================
def step3_dg_pde():
    print("\n" + "=" * 70)
    print("步骤 3: DG 离散对流-扩散方程")
    print("=" * 70)

    dg = DG1DHeat(K=8, Np=3, L=1.0, velocity=1.0, diffusion=0.05)
    u0 = np.zeros((dg.Np, dg.K))
    u_final, x, t = dg.solve(u0, T_final=0.1, u_in=1.0)
    print(f"  单元数: {dg.K}, 阶数: {dg.Np - 1}")
    print(f"  最终时间: {t[-1]:.4f}")
    print(f"  max(u) = {np.max(u_final):.4f}, min(u) = {np.min(u_final):.4f}")

    return u_final


# =============================================================================
# 4. 3D FEM 网格与投影
# =============================================================================
def step4_fem3d():
    print("\n" + "=" * 70)
    print("步骤 4: 3D FEM 网格与 L2 投影")
    print("=" * 70)

    nodes, elems = tet_mesh_generate(3, 3, 3, 1.0, 1.0, 1.0)
    print(f"  节点数: {nodes.shape[0]}, 单元数: {elems.shape[0]}")

    # 体积检查
    vols = [tet_volume(nodes[e]) for e in elems]
    print(f"  总体积: {sum(vols):.4f} (期望 1.0)")

    # L2 投影
    rng = np.random.default_rng(0)
    sample_pts = rng.random((30, 3))
    sample_vals = np.sin(np.pi * sample_pts[:, 0]) * np.cos(np.pi * sample_pts[:, 1])
    c = fem3d_project(nodes, elems, sample_pts, sample_vals, reg=1e-6)
    print(f"  FEM 系数 ||c|| = {np.linalg.norm(c):.4f}")

    return c


# =============================================================================
# 5. 机会约束处理
# =============================================================================
def step5_chance_constraints():
    print("\n" + "=" * 70)
    print("步骤 5: 机会约束处理")
    print("=" * 70)

    # 正态分位数
    for p in [0.9, 0.95, 0.99]:
        print(f"  z_{p} = {norm_ppf(p):.4f}")

    # 机会约束
    cc = ChanceConstraint(
        a=np.array([1.0, 2.0]),
        b=5.0,
        w_mean=np.array([0.1, -0.1]),
        w_cov=np.array([[0.1, 0.02], [0.02, 0.2]]),
        alpha=0.05,
    )
    x = np.array([1.0, 1.0])
    viol = cc.evaluate(x)
    print(f"  机会约束违反: {viol:.4f}")
    print(f"  可行: {cc.is_feasible(x)}")

    # 联合机会约束
    ccs = [
        ChanceConstraint(
            a=np.array([1.0, 0.0]),
            b=3.0,
            w_mean=np.array([0.1, 0.0]),
            w_cov=0.1 * np.eye(2),
            alpha=0.05,
        ),
        ChanceConstraint(
            a=np.array([0.0, 1.0]),
            b=4.0,
            w_mean=np.array([0.0, 0.1]),
            w_cov=0.1 * np.eye(2),
            alpha=0.05,
        ),
    ]
    jcc = JointChanceConstraint(ccs, alpha=0.1)
    print(f"  联合最大违反: {jcc.max_violation(x):.4f}")
    print(f"  联合可行: {jcc.is_feasible(x)}")

    return cc


# =============================================================================
# 6. LMC 鲁棒采样
# =============================================================================
def step6_lmc_sampler():
    print("\n" + "=" * 70)
    print("步骤 6: 高阶 Langevin Monte Carlo 采样")
    print("=" * 70)

    def grad_U(w):
        return w

    sampler = HigherOrderLangevin(
        K=3, d=2, h=0.01, gamma=1.0, grad_U_fn=grad_U, rng=np.random.default_rng(42)
    )
    samples = sampler.sample(x0=np.array([1.0, 1.0]), n_samples=300, burn_in=50)
    print(f"  样本形状: {samples.shape}")
    print(f"  样本均值: {samples.mean(axis=0)}")
    print(f"  样本标准差: {samples.std(axis=0)}")

    # 鲁棒目标估计
    def f_obj(x, w):
        return np.sum((x - w) ** 2)

    ros = RobustObjectiveSampler(
        f_fn=f_obj, w_dim=2, potential_grad=grad_U, K=3, seed=0
    )
    mu, se = ros.estimate(np.array([0.5, 0.5]), n_samples=100)
    print(f"  鲁棒目标估计: {mu:.4f} +/- {se:.4f}")

    return samples


# =============================================================================
# 7. 延迟双稳态动力学
# =============================================================================
def step7_delay_bistability():
    print("\n" + "=" * 70)
    print("步骤 7: 延迟双稳态 DDAE 求解")
    print("=" * 70)

    params = {"c": 0.5, "eps": 0.1, "n": 4.0, "a": 1.0, "r": 0.5}
    solver = DDAESolver(delay_bist_2d_cubic, tau=0.5, params=params, dim=2)
    y0 = np.array([0.5, 0.3])
    y_arr, t_arr = solver.solve(y0, T_final=3.0, dt=0.01)
    print(f"  时间范围: [0, {t_arr[-1]:.2f}]")
    print(f"  最终状态: Cdk1={y_arr[-1, 0]:.4f}, Apc={y_arr[-1, 1]:.4f}")

    # 平衡点
    y_eq, conv = find_equilibrium(params)
    print(f"  平衡点: {y_eq}, 收敛: {conv}")

    return y_arr


# =============================================================================
# 8. 谱鲁棒性分析
# =============================================================================
def step8_spectral_robustness():
    print("\n" + "=" * 70)
    print("步骤 8: 谱鲁棒性分析 (Chladni 板)")
    print("=" * 70)

    nx, ny = 8, 8
    vals_c, vecs_c = chladni_eigenmodes(nx, ny, mu=0.225, k=3)
    print(f"  Chladni 特征值 (前 3): {vals_c}")

    # Laplacian
    L = laplacian_2d(nx, ny)
    vals_L = np.linalg.eigvalsh(L)
    print(f"  Laplacian 特征值范围: [{vals_L[0]:.4f}, {vals_L[-1]:.4f}]")

    # 鲁棒性分析
    def L_fn(x):
        return x[0] * L

    analyzer = SpectralRobustnessAnalyzer(L_fn, np.array([1.0]), n_modes=3)
    print(f"  名义特征值: {analyzer.nominal_eigenvalues}")
    rho = analyzer.robustness_ratio(np.array([0.9]))
    print(f"  鲁棒性比 (x=0.9): {rho:.4f}")

    # 脆弱性指标
    rng = np.random.default_rng(0)
    perturbations = 0.8 + 0.4 * rng.random((10, 1))
    F_mean, F_std = analyzer.fragility_index(perturbations)
    print(f"  脆弱性指标: mean={F_mean:.4f}, std={F_std:.4f}")

    return analyzer


# =============================================================================
# 9. 自适应 ESN 控制
# =============================================================================
def step9_esn_control():
    print("\n" + "=" * 70)
    print("步骤 9: 自适应 ESN 鲁棒控制")
    print("=" * 70)

    plant = NonlinearPlant()

    def ref_fn(t):
        return 1.0 if t > 1.0 else 0.0

    ctrl = AdaptiveRobustController(
        plant_fn=plant.forward,
        reservoir_size=30,
        spectral_radius=0.8,
        leaky=0.8,
        Kp=1e-3,
        Kd=1e-5,
        seed=42,
    )
    t, y, u = ctrl.run_simulation(ref_fn, T=3.0, dt=0.1, noise_std=0.01)
    ref_vals = np.array([ref_fn(ti) for ti in t])
    mean_err = np.mean(np.abs(y - ref_vals))
    print(f"  仿真时间: [0, {t[-1]:.2f}]")
    print(f"  平均跟踪误差: {mean_err:.4f}")
    print(f"  最终输出: {y[-1]:.4f}, 参考: {ref_fn(t[-1]):.4f}")

    return mean_err


# =============================================================================
# 10. SMB 过程仿真
# =============================================================================
def step10_smb_process():
    print("\n" + "=" * 70)
    print("步骤 10: SMB 色谱过程仿真")
    print("=" * 70)

    smb = SimulatedMovingBed(n_columns=4, n_cells=8, L=1.0)
    smb.set_flow_rates(Q_extract=1.0, Q_feed=0.5, Q_desorb=1.0, Q_raff=0.5)
    result = smb.run(C_feed=1.0, C_desorb=0.0, T=2.0, t_switch=0.5, dt=0.05)
    print(f"  仿真时间: [0, {result['t'][-1]:.2f}]")
    ext_p, raff_p = compute_purity(result["extract"], result["raffinate"])
    prod = compute_productivity(result["extract"], 2.0)
    print(f"  Extract 纯度: {ext_p:.4f}")
    print(f"  Raffinate 纯度: {raff_p:.4f}")
    print(f"  生产率: {prod:.4f}")

    return result


# =============================================================================
# 11. 不确定性场构造
# =============================================================================
def step11_uncertain_field():
    print("\n" + "=" * 70)
    print("步骤 11: 不确定性随机场构造")
    print("=" * 70)

    # Newton 插值
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.sin(x)
    yd = divdif(x, y)
    xp = np.linspace(0, 3, 20)
    yp = eval_newton(x, yd, xp)
    err = np.max(np.abs(yp - np.sin(xp)))
    print(f"  Newton 插值最大误差: {err:.4e}")

    # 随机场
    uf = UncertainField(n_nodes=10, sigma=0.1, seed=42)
    x_n, D_n = uf.sample_field()
    print(f"  随机场均值: {D_n.mean():.4f}, 标准差: {D_n.std():.4f}")

    # KL 展开
    kl = KarhunenLoeveExpansion(n_terms=5, correlation_length=0.3)
    coeffs = kl.sample(1)[0]
    x_q = np.linspace(0, 1, 50)
    xi = kl.evaluate(coeffs, x_q)
    print(f"  KL 展开: min={xi.min():.4f}, max={xi.max():.4f}")

    return uf


# =============================================================================
# 12. 鲁棒优化求解
# =============================================================================
def step12_robust_optimization(E: EllipsoidalUncertaintySet):
    print("\n" + "=" * 70)
    print("步骤 12: 鲁棒优化求解")
    print("=" * 70)

    dim = 3
    # 目标函数 (二次)
    A = np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.3], [0.0, 0.3, 1.5]])
    b = np.array([1.0, -1.0, 0.5])

    def f_obj(x):
        return 0.5 * x @ A @ x + b @ x

    def f_wc(x_full):
        x = x_full[:dim]
        w = x_full[dim:]
        # 参数扰动影响 A
        A_pert = A + 0.1 * np.diag(w[:dim] if w.size >= dim else np.pad(w, (0, dim - w.size)))
        return 0.5 * x @ A_pert @ x + b @ x

    problem = RobustOptimizationProblem(
        dim=dim,
        objective_fn=f_obj,
        worst_case_fn=f_wc,
        bounds=(-5.0 * np.ones(dim), 5.0 * np.ones(dim)),
    )

    # 最坏情况优化
    print("  最坏情况鲁棒优化:")
    x0 = np.zeros(dim)
    E_wc = EllipsoidalUncertaintySet(
        dim=dim, center=np.zeros(dim), rho=0.5, toeplitz_rho=0.3, seed=0
    )
    res_wc = worst_case_robust_optimize(
        problem, E_wc, x0, lr=0.05, n_iter=100, n_samples=10
    )
    print(f"    最优值: {res_wc['f_opt']:.4f}")
    print(f"    收敛: {res_wc['converged']}")

    # 均值-方差优化
    print("  均值-方差鲁棒优化:")
    res_mv = mean_variance_robust_optimize(
        problem, E_wc, x0, risk_weight=0.5, lr=0.05, n_iter=100, n_samples=10
    )
    print(f"    最优值: {res_mv['f_opt']:.4f}")

    # 机会约束优化
    print("  机会约束鲁棒优化:")
    cc = ChanceConstraint(
        a=np.array([1.0, 0.0, 0.0]),
        b=2.0,
        w_mean=np.array([0.1, 0.0, 0.0]),
        w_cov=0.1 * np.eye(dim),
        alpha=0.05,
    )
    jcc = JointChanceConstraint([cc], alpha=0.05)
    res_cc = chance_constrained_robust_optimize(
        problem, jcc, x0, lr=0.05, n_iter=100, penalty_weight=10.0
    )
    print(f"    最优值: {res_cc['f_opt']:.4f}")
    print(f"    可行: {res_cc['feasible']}")

    return res_wc


# =============================================================================
# 主程序
# =============================================================================
def main():
    print("*" * 70)
    print("*  鲁棒优化与不确定约束 —— 博士级科学计算项目 217")
    print("*  应用：模拟移动床色谱过程的鲁棒优化")
    print("*" * 70)

    # 执行所有步骤
    E = step1_uncertainty_sets()
    step2_circulant_kkt()
    step3_dg_pde()
    step4_fem3d()
    step5_chance_constraints()
    step6_lmc_sampler()
    step7_delay_bistability()
    step8_spectral_robustness()
    step9_esn_control()
    step10_smb_process()
    step11_uncertain_field()
    step12_robust_optimization(E)

    print("\n" + "=" * 70)
    print("所有步骤执行完毕！")
    print("=" * 70)
    print("项目成功展示了：")
    print("  - 不确定性集合构造 (椭球、超球、非中心 t)")
    print("  - 循环预条件 KKT 系统求解")
    print("  - DG 离散 PDE 约束")
    print("  - 3D FEM 网格与投影")
    print("  - 机会约束处理")
    print("  - LMC 鲁棒采样")
    print("  - 延迟双稳态动力学")
    print("  - 谱鲁棒性分析")
    print("  - 自适应 ESN 控制")
    print("  - SMB 色谱过程仿真")
    print("  - 不确定性随机场")
    print("  - 多种鲁棒优化算法")
    print("=" * 70)


if __name__ == "__main__":
    main()
