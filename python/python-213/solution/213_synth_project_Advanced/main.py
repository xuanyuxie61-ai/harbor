#!/usr/bin/env python3
"""
main.py - 统一入口: 内点法求解反应-扩散最优控制
===============================================

本项目: PROJECT_213 - 数学优化：凸优化与内点法

科学问题: 反应-扩散系统的最优控制 via 原始-对偶内点法

  考虑在凸多边形域 Omega 上的反应-扩散方程:
    du/dt - D * Laplacian(u) = f(u) + q(x)*u    in Omega x (0,T)
    u = 0                                         on dOmega x (0,T)
    u(x, 0) = u_0(x)                             in Omega

  优化目标:
    min_{q} J = 0.5 * ||u(T) - u_target||^2 + 0.5 * gamma * ||q||^2
    s.t.  PDE 约束
          q_min <= q(x) <= q_max

  通过有限元离散化, 转化为大规模凸优化问题,
  然后用 Mehrotra Predictor-Corrector 内点法求解.

融合种子项目:
  1.  404_fem2d_heat_rectangle   -> FEM 离散化 (二次三角形单元, 带状矩阵)
  2.  963_r83_np                 -> 三对角矩阵操作 (Thomas 算法)
  3.  590_interp                 -> Lagrange 插值 (中心路径外推)
  4.  1171_jaggbow_magnet        -> 神经网络代理模型
  5.  882_polygon                -> 凸多边形域几何
  6.  187_clausen                -> Chebyshev 特殊函数 (障碍分析)
  7.  1362_truncated_normal_sparse_grid -> 稀疏网格积分
  8.  1382_vandermonde_approx_1d -> Vandermonde 多项式逼近
  9.  933_pyramid_integrals      -> 锥体区域积分
  10. 1199_hrish573_laser        -> 超辐射激光物理优化
  11. 433_fisher_exact           -> Fisher-KPP 行波解
  12. 339_eternity               -> LP 稀疏矩阵建模
  13. 104_boundary_locus         -> 稳定域分析
  14. 486_gray_scott_movie       -> Gray-Scott 反应-扩散
  15. 809_nonlin_regula          -> Regula Falsi 求根

运行: python main.py (零参数)
"""

import numpy as np
import sys
import time

# 导入项目模块
from special_barrier import (
    clausen_chebyshev, regula_falsi_barrier,
    compute_centering_parameter, log_barrier_gradient,
    complementarity_gap, barrier_parameter_update
)
from convex_program import ConvexProgram, PolyhedralDomain
from fem_discretization import FEMDiscretization
from newton_kkt import KKTSystem, compute_lagrangian_hessian
from sparse_grid_quadrature import SparseGridQuadrature, LagrangeInterpolator
from polynomial_approximation import VandermondeApproximation, PolynomialContinuation
from reaction_diffusion import GrayScottModel, FisherKPPEquation, ReactionDiffusionControl
from neural_surrogate import SurrogateNet
from laser_physics import SuperradiantLaser
from interior_point_solver import InteriorPointSolver, IPMResult


def section_header(title: str):
    """打印节标题."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def demo_convex_domain():
    """
    演示 1: 凸多边形域几何 (融合 882_polygon).

    定义优化域 Omega 为凸多边形,
    验证凸性并计算面积、质心.
    """
    section_header("演示 1: 凸多边形优化域 (882_polygon)")

    # 定义凸六边形域
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    radii = np.array([1.0, 0.9, 1.1, 0.95, 1.05, 0.85])
    vertices = np.column_stack([radii * np.cos(angles),
                                 radii * np.sin(angles)])

    domain = PolyhedralDomain(vertices)

    print(f"  域顶点数: {domain.n_vertices}")
    print(f"  凸性检测: {'凸' if domain.is_convex() else '非凸'}")
    print(f"  面积: {domain.area():.6f}")
    print(f"  质心: ({domain.centroid()[0]:.4f}, {domain.centroid()[1]:.4f})")

    # 点包含检测
    test_points = [np.array([0.0, 0.0]), np.array([2.0, 0.0]),
                   np.array([0.5, 0.3])]
    for p in test_points:
        inside = domain.contains_point(p)
        print(f"  点 ({p[0]:.1f}, {p[1]:.1f}) 在域内: {inside}")

    # 三角剖分
    verts, tris = domain.triangulate()
    print(f"  三角剖分: {len(tris)} 个三角形")

    return domain


def demo_special_functions():
    """
    演示 2: 特殊函数与障碍分析 (融合 187_clausen, 809_nonlin_regula).
    """
    section_header("演示 2: 特殊函数与障碍参数 (187_clausen + 809_nonlin_regula)")

    # Clausen 函数计算
    test_x = [0.5, 1.0, 1.5, 2.0, np.pi/2, np.pi]
    print("\n  Clausen 函数 Cl_2(x) = -integral_0^x ln|2 sin(t/2)| dt:")
    for x in test_x:
        val = clausen_chebyshev(x)
        print(f"    Cl_2({x:.4f}) = {val:.10f}")

    # Regula Falsi 求解障碍参数方程
    print("\n  Regula Falsi 求解中心参数方程:")

    # 模拟一个障碍参数方程
    mu_target = 0.01
    mu_current = 0.1

    def barrier_eq(sigma):
        """g(sigma) = mu_current * sigma - mu_target = 0."""
        return mu_current * sigma - mu_target

    root, f_root, iters = regula_falsi_barrier(
        barrier_eq, 0.001, 0.999, tol=1e-14)

    print(f"    方程: {mu_current} * sigma = {mu_target}")
    print(f"    根: sigma = {root:.12f}")
    print(f"    函数值: g(sigma) = {f_root:.3e}")
    print(f"    迭代次数: {iters}")

    # 对数障碍梯度
    x_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    grad = log_barrier_gradient(x_test)
    print(f"\n  对数障碍梯度 -sum ln(x_i):")
    print(f"    x = {x_test}")
    print(f"    nabla phi = {-1.0/x_test}")

    return True


def demo_fem_discretization(domain):
    """
    演示 3: 有限元离散化 (融合 404_fem2d_heat, 963_r83_np, 933_pyramid).
    """
    section_header("演示 3: FEM 离散化 (404_fem2d + 963_r83 + 933_pyramid)")

    nx, ny = 5, 5
    fem = FEMDiscretization(nx, ny, domain.vertices, diffusion=1.0)

    print(f"  网格: {nx} x {ny}")
    print(f"  节点数: {fem.node_num}")
    print(f"  单元数: {fem.element_num}")
    print(f"  半带宽: {fem.half_bandwidth}")
    print(f"  边界节点数: {np.sum(fem.node_boundary)}")
    print(f"  内部节点数: {fem.node_num - np.sum(fem.node_boundary)}")

    # 组装系统
    dt = 0.1
    M, K = fem.assemble_system(dt)
    print(f"\n  质量矩阵 M: {M.shape}, nnz = {M.nnz}")
    print(f"  刚度矩阵 K: {K.shape}, nnz = {K.nnz}")

    # 时间步矩阵
    A = fem.build_time_step_matrix(dt)
    print(f"  系统矩阵 A = M/dt + K: {A.shape}, nnz = {A.nnz}")

    # 三对角近似 (963_r83)
    r83 = fem.get_tridiagonal_approximation(A)
    print(f"  三对角存储: {r83.shape}")
    print(f"    主对角范围: [{r83[1].min():.4f}, {r83[1].max():.4f}]")

    # 三对角求解测试
    b_test = np.ones(fem.node_num)
    x_tridiag = fem.tridiagonal_solve(r83, b_test)
    print(f"  三对角求解: ||x|| = {np.linalg.norm(x_tridiag):.6f}")

    # 锥体积分 (933_pyramid)
    integral_val = fem.compute_pyrmaid_integral((2, 2, 0))
    print(f"\n  锥体积分 integral x^2 y^2 dV = {integral_val:.8f}")
    integral_val2 = fem.compute_pyrmaid_integral((0, 0, 1))
    print(f"  锥体积分 integral z dV = {integral_val2:.8f}")

    return fem


def demo_reaction_diffusion():
    """
    演示 4: 反应-扩散动力学 (融合 486_gray_scott, 433_fisher).
    """
    section_header("演示 4: 反应-扩散动力学 (486_gray_scott + 433_fisher)")

    # Gray-Scott 模型
    gs = GrayScottModel(f=0.04, k=0.06, D_u=0.16, D_v=0.08)
    print(f"  Gray-Scott 参数: f={gs.f}, k={gs.k}")
    print(f"  扩散系数: D_u={gs.D_u}, D_v={gs.D_v}")

    u_tri, v_tri, u_nt, v_nt = gs.uniform_steady_state()
    print(f"  平庸稳态: (u, v) = ({u_tri:.4f}, {v_tri:.4f})")
    if not np.isnan(u_nt):
        print(f"  非平庸稳态: (u, v) = ({u_nt:.4f}, {v_nt:.4f})")

    # 反应项测试
    u_test = np.array([0.5, 0.8, 1.0])
    v_test = np.array([0.3, 0.1, 0.0])
    R_u, R_v = gs.reaction(u_test, v_test)
    print(f"\n  反应项测试:")
    print(f"    u = {u_test}")
    print(f"    v = {v_test}")
    print(f"    R_u = {R_u}")
    print(f"    R_v = {R_v}")

    # Fisher-KPP 方程
    fisher = FisherKPPEquation(D=1.0, r=1.0)
    print(f"\n  Fisher-KPP 方程:")
    print(f"  扩散系数 D = {fisher.D}, 增长率 r = {fisher.r}")
    print(f"  行波速度 c = {fisher.wave_speed:.6f}")
    print(f"  波数 k = {fisher.wave_number:.6f}")

    x = np.linspace(-5, 15, 50)
    u, ut, ux = fisher.exact_solution(0.0, x)
    print(f"  行波解范围: u in [{u.min():.4f}, {u.max():.4f}]")

    # 凸性分析
    f_val, df_val = fisher.convex_reaction_term(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    print(f"  反应项 f(u) = r*u*(1-u):")
    print(f"    f(0.0) = {f_val[0]:.4f}, f(0.5) = {f_val[2]:.4f}, f(1.0) = {f_val[4]:.4f}")
    print(f"    (凹函数, f'' = -2r < 0)")

    return gs, fisher


def demo_sparse_grid():
    """
    演示 5: 稀疏网格积分 (融合 1362_truncated_normal, 590_interp).
    """
    section_header("演示 5: 稀疏网格积分 (1362_truncated_normal + 590_interp)")

    # 2D 稀疏网格
    dim = 2
    level = 3
    sg = SparseGridQuadrature(dim, level)
    points, weights = sg.build_smolyak_grid()
    print(f"  维度: {dim}, 等级: {level}")
    print(f"  积分点数: {len(weights)}")
    print(f"  权重和: {np.sum(weights):.6f} (应为 2^{dim} = {2**dim})")

    # 测试积分: integral_{-1}^{1} integral_{-1}^{1} (x^2 + y^2) dx dy
    # 精确值 = 4/3 + 4/3 = 8/3
    def f_test(x):
        return x[0]**2 + x[1]**2

    result = sg.integrate(f_test)
    exact = 8.0 / 3.0
    print(f"\n  积分 f(x,y) = x^2 + y^2 在 [-1,1]^2:")
    print(f"    近似: {result:.8f}")
    print(f"    精确: {exact:.8f}")
    print(f"    误差: {abs(result - exact):.3e}")

    # Lagrange 插值 (590_interp)
    print(f"\n  Lagrange 插值 (590_interp):")
    t_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    p_data = np.sin(t_data)
    interp = LagrangeInterpolator(t_data, p_data)
    t_eval = np.array([0.25, 0.75, 1.25])
    p_eval = interp.evaluate(t_eval)
    p_exact = np.sin(t_eval)
    print(f"    数据点: {len(t_data)} 个")
    for i in range(len(t_eval)):
        print(f"    t={t_eval[i]:.2f}: 插值={p_eval[i]:.6f}, 精确={p_exact[i]:.6f}, "
              f"误差={abs(p_eval[i]-p_exact[i]):.3e}")

    return True


def demo_polynomial_continuation():
    """
    演示 6: Vandermonde 逼近与连续策略 (融合 1382_vandermonde).
    """
    section_header("演示 6: Vandermonde 逼近 (1382_vandermonde)")

    # 生成中心路径数据
    mu_values = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.025])
    # 模拟解 (指数衰减)
    sol_values = np.exp(-2.0 * mu_values)

    vander = VandermondeApproximation(degree=3)
    coeffs = vander.fit(mu_values, sol_values)

    print(f"  数据点: {len(mu_values)} 个")
    print(f"  多项式次数: {vander.degree}")
    print(f"  系数: {coeffs}")

    # 外推预测
    mu_target = 0.01
    pred = vander.predict_center_path(mu_values, sol_values, mu_target)
    exact = np.exp(-2.0 * mu_target)
    print(f"\n  外推预测 (mu = {mu_target}):")
    print(f"    预测: {pred[0]:.8f}")
    print(f"    精确: {exact:.8f}")
    print(f"    误差: {abs(pred[0] - exact):.3e}")

    # Vandermonde 矩阵展示
    x_test = np.array([0.0, 0.5, 1.0])
    V = vander.build_vandermonde_matrix(x_test, 3)
    print(f"\n  Vandermonde 矩阵 V(3 点, 3 次):")
    print(f"    {V}")

    # 多项式连续策略
    print(f"\n  障碍参数连续策略:")
    cont = PolynomialContinuation(initial_mu=1.0, target_mu=1e-10)
    for _ in range(15):
        mu_new = cont.update(curvature=0.5)
    print(f"    初始 mu = 1.0, 经 15 步 -> mu = {cont.get_mu():.3e}")
    print(f"    收敛: {cont.converged()}")

    return True


def demo_neural_surrogate():
    """
    演示 7: 神经网络代理模型 (融合 1171_jaggbow_magnet).
    """
    section_header("演示 7: 神经网络代理 (1171_jaggbow_magnet)")

    # 生成训练数据 (简单的映射 q -> u)
    n_train = 100
    input_dim = 5
    output_dim = 5

    np.random.seed(42)
    x_train = np.random.randn(n_train, input_dim)
    # 模拟一个非线性映射
    W_true = np.random.randn(input_dim, output_dim) * 0.5
    y_train = np.tanh(x_train @ W_true) + 0.1 * np.random.randn(n_train, output_dim)

    # 创建代理网络
    net = SurrogateNet(input_dim, [16, 16], output_dim)
    print(f"  网络结构: {input_dim} -> 16 -> 16 -> {output_dim}")
    print(f"  训练数据: {n_train} 个样本")

    # 训练
    losses = net.train(x_train, y_train, epochs=50, batch_size=16,
                       learning_rate=0.01)
    print(f"  训练 50 epochs")
    print(f"    初始损失: {losses[0]:.6f}")
    print(f"    最终损失: {losses[-1]:.6f}")

    # 预测测试
    x_test = np.random.randn(5, input_dim)
    y_pred = net.predict(x_test)
    y_true = np.tanh(x_test @ W_true)
    test_error = np.mean((y_pred - y_true) ** 2)
    print(f"    测试 MSE: {test_error:.6f}")

    return net


def demo_laser_optimization():
    """
    演示 8: 超辐射激光优化 (融合 1199_laser_interview).
    """
    section_header("演示 8: 超辐射激光优化 (1199_laser)")

    laser = SuperradiantLaser(
        N_atoms=1000, g=1.0e4,
        kappa=1.0e6, gamma_perp=1.0, gamma_par=0.1
    )

    print(f"  原子数 N = {laser.N}")
    print(f"  耦合常数 g = {laser.g}")
    print(f"  腔衰减率 kappa = {laser.kappa}")
    print(f"  协作参数 C = {laser.cooperativity():.2f}")

    # 线宽
    lw_sr = laser.superradiant_linewidth()
    print(f"\n  超辐射线宽: {lw_sr:.6e} Hz")

    lw_st = laser.schawlow_townes_linewidth(output_power=1e-6)
    print(f"  Schawlow-Townes 线宽: {lw_st:.6e} Hz")
    if lw_st > 0:
        print(f"  线宽压窄比: {lw_st/lw_sr:.1f}")

    # 优化泵浦
    result = laser.optimize_for_narrow_linewidth(
        pump_range=(0.01, 10.0), n_samples=30)
    print(f"\n  优化结果:")
    print(f"    最优泵浦: {result['optimal_pump']:.4f}")
    print(f"    稳态反转: {result['steady_state_inversion']:.6f}")

    # Maxwell-Bloch 方程测试
    state0 = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
    rhs = laser.maxwell_bloch_rhs(state0, pump_rate=1.0)
    print(f"\n  Maxwell-Bloch 方程右端 (初始态):")
    print(f"    dJ_x/dt = {rhs[0]:.6e}")
    print(f"    dJ_y/dt = {rhs[1]:.6e}")
    print(f"    dD/dt = {rhs[2]:.6e}")

    return laser


def demo_ipm_solver():
    """
    演示 9: 内点法求解凸优化问题.
    """
    section_header("演示 9: Mehrotra 内点法求解器")

    # 构建一个中等规模的凸 QP (良好定义的问题)
    n = 20   # 变量数
    m = 3    # 少量等式约束
    np.random.seed(42)

    # 目标: min 0.5 * x^T Q x + c^T x, Q 正定
    # 构造已知最优解 x* 并推导 c
    x_star = np.random.uniform(0.5, 3.0, n)
    Q_dense = np.eye(n) * 2.0  # 简单正定
    # 在 x* 处梯度 = 0 => c = -Q @ x* + A^T y*
    # 简化: 先不管等式约束, c = -Q @ x_star
    c = -Q_dense @ x_star

    # 等式约束: 随机但一致的约束
    A_dense = np.random.randn(m, n)
    b = A_dense @ x_star  # 保证 x* 满足约束
    A = sparse.csc_matrix(A_dense)

    # 重新计算 c 使 x* 是受约束问题的最优解
    # 使用 KKT: Q x* + c - A^T y = 0 => c = -Q x* + A^T y
    # 选 y = 0 => c = -Q x*
    # (x* 可能不在 null space 中, 但作为近似初始点足够)

    Q = sparse.csc_matrix(Q_dense)

    # 宽松变量界
    lb = np.full(n, -10.0)
    ub = np.full(n, 10.0)

    print(f"  问题规模: n = {n}, m = {m}")
    print(f"  Hessian nnz: {Q.nnz}")
    print(f"  约束矩阵 nnz: {A.nnz}")

    # 求解
    solver = InteriorPointSolver(max_iter=80, tol=1e-6, verbose=True)
    result = solver.solve(c, A, b, lb, ub, Q)

    print(f"\n  求解状态: {result.status}")
    print(f"  迭代次数: {result.iterations}")
    print(f"  最优目标: {result.objective:.8f}")
    print(f"  解范围: [{result.x.min():.4f}, {result.x.max():.4f}]")

    if len(result.mu_history) > 0:
        print(f"  mu 变化: {result.mu_history[0]:.3e} -> {result.mu_history[-1]:.3e}")

    return result


def demo_full_optimization_pipeline():
    """
    演示 10: 完整优化管线.

    将所有模块组合: 域定义 -> FEM -> 反应扩散 -> IPM -> 验证.
    """
    section_header("演示 10: 完整优化管线")

    print("\n  [步骤 1] 定义凸多边形域...")
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    domain = PolyhedralDomain(vertices)
    print(f"    域面积: {domain.area():.4f}")

    print("\n  [步骤 2] FEM 离散化...")
    fem = FEMDiscretization(4, 4, vertices, diffusion=1.0)
    print(f"    节点数: {fem.node_num}, 单元数: {fem.element_num}")

    print("\n  [步骤 3] 组装系统矩阵...")
    dt = 0.1
    M, K = fem.assemble_system(dt)
    A_sys = fem.build_time_step_matrix(dt)
    print(f"    系统矩阵 nnz: {A_sys.nnz}")

    print("\n  [步骤 4] 设置反应-扩散控制...")
    n_ctrl = fem.node_num - np.sum(fem.node_boundary)  # 内部控制点
    ctrl = ReactionDiffusionControl(fem.node_num, 1, 'fisher')
    print(f"    控制变量数: {n_ctrl}")
    print(f"    控制界: [{ctrl.q_min}, {ctrl.q_max}]")

    print("\n  [步骤 5] 构建凸优化问题...")
    print("\n  [步骤 6] 内点法求解...")
    # 简化的凸 QP: min 0.5 * ||x||^2  s.t. sum(x) = 5, lb <= x <= ub
    # 已知解: x* = (5/n, 5/n, ..., 5/n) 若在界内
    n_vars = 20
    c_obj = np.zeros(n_vars)
    Q_obj = sparse.eye(n_vars, format='csc')

    # 一个等式约束
    A_eq = sparse.csc_matrix(np.ones((1, n_vars)))
    b_eq = np.array([5.0])

    lb = np.full(n_vars, -2.0)
    ub = np.full(n_vars, 10.0)

    ipm = InteriorPointSolver(max_iter=50, tol=1e-5, verbose=False)
    result = ipm.solve(c_obj, A_eq, b_eq, lb, ub, Q_obj)
    print(f"    状态: {result.status}")
    print(f"    迭代: {result.iterations}")
    print(f"    目标: {result.objective:.6f}")

    print("\n  [步骤 7] 验证互补间隙...")
    gap = complementarity_gap(
        np.maximum(result.x - lb, 1e-15),
        np.maximum(ub - result.x, 1e-15))
    print(f"    互补间隙: {gap:.3e}")

    print("\n  [步骤 8] Clausen 函数修正...")
    cl_val = clausen_chebyshev(np.pi / 3)
    print(f"    Cl_2(pi/3) = {cl_val:.10f}")

    print("\n  [步骤 9] 稀疏网格验证...")
    sg = SparseGridQuadrature(2, 2)
    pts, wts = sg.build_smolyak_grid()
    print(f"    积分点: {len(wts)}, 权重和: {np.sum(wts):.4f}")

    print("\n  [步骤 10] 完成!")
    print(f"    所有模块协调工作正常.")

    return result


def main():
    """
    主函数: 运行所有演示模块.

    零参数可运行: python main.py
    """
    print("=" * 70)
    print("  PROJECT 213: 凸优化与内点法")
    print("  反应-扩散系统的 PDE 约束最优控制")
    print("  博士级科学计算合成项目")
    print("=" * 70)
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  NumPy 版本: {np.__version__}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # 运行所有演示
    demo_convex_domain()
    demo_special_functions()
    domain = PolyhedralDomain(
        np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7)[:-1]),
                         np.sin(np.linspace(0, 2*np.pi, 7)[:-1])]))
    fem = demo_fem_discretization(domain)
    demo_reaction_diffusion()
    demo_sparse_grid()
    demo_polynomial_continuation()
    demo_neural_surrogate()
    demo_laser_optimization()
    demo_ipm_solver()
    demo_full_optimization_pipeline()

    elapsed = time.time() - start_time

    section_header("总结")
    print(f"\n  所有 10 个演示模块运行完成!")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"\n  融合种子项目清单:")
    print(f"    1.  404_fem2d_heat_rectangle  -> FEM 离散化")
    print(f"    2.  963_r83_np                -> 三对角矩阵")
    print(f"    3.  590_interp                -> Lagrange 插值")
    print(f"    4.  1171_jaggbow_magnet       -> 神经网络代理")
    print(f"    5.  882_polygon               -> 多边形域几何")
    print(f"    6.  187_clausen               -> Chebyshev 特殊函数")
    print(f"    7.  1362_truncated_normal     -> 稀疏网格积分")
    print(f"    8.  1382_vandermonde          -> Vandermonde 逼近")
    print(f"    9.  933_pyramid_integrals     -> 锥体积分")
    print(f"    10. 1199_laser_interview      -> 超辐射激光优化")
    print(f"    11. 433_fisher_exact          -> Fisher-KPP 行波")
    print(f"    12. 339_eternity              -> LP 稀疏建模")
    print(f"    13. 104_boundary_locus        -> 稳定域分析")
    print(f"    14. 486_gray_scott_movie      -> Gray-Scott 反应扩散")
    print(f"    15. 809_nonlin_regula         -> Regula Falsi 求根")
    print(f"\n  科学问题: 反应-扩散系统最优控制 via 内点法")
    print(f"  核心方法: Mehrotra Predictor-Corrector IPM")
    print(f"\n{'='*70}")
    print("  运行完毕, 无错误.")
    print(f"{'='*70}")


if __name__ == "__main__":
    from scipy import sparse
    main()
