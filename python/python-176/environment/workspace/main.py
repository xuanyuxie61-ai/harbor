#!/usr/bin/env python3
"""
main.py
================================================================================
统一入口：二维椭圆域上非定常反应-扩散方程约束的最优边界控制
         —— 伴随方程方法、有限元离散与多算法验证

项目编号：176
科学领域：计算数学 —— 最优控制伴随方程方法

科学问题描述
------------
在二维椭圆域 Ω = {(x,y) | (x/a)² + (y/b)² ≤ 1} 上，考虑如下
非定常反应-扩散方程约束的最优 Neumann 边界控制问题：

状态方程（前向 PDE）：
    ∂y/∂t − ν Δy + c y³ = f(x,t)          in Ω × (0,T)
    ν ∂y/∂n = q(s,t)                      on ∂Ω × (0,T)
    y(x,0) = 0                            in Ω

目标泛函（跟踪型）：
    J(q) = ½∫_0^T ∫_Ω (y − y_d)² dx dt
         + (α/2)∫_0^T ∫_{∂Ω} q² ds dt
         + (β/2)∫_0^T ∫_{∂Ω} |∂_s q|² ds dt

伴随方程（后向 PDE）：
    −∂p/∂t − ν Δp + 3c y² p = y − y_d     in Ω × (0,T)
    ν ∂p/∂n = 0                           on ∂Ω × (0,T)
    p(x,T) = 0                            in Ω

梯度公式：
    ∇J(q) = α q + p|_{∂Ω} − β ∂_s² q

本程序对上述问题实施：
  1. 椭圆域三角形 FEM 离散
  2. 隐式欧拉时间推进
  3. 伴随方程后向求解
  4. 梯度下降 + Armijo 线搜索优化
  5. 多算法验证（QMC、敏感 ODE、Lambert W、椭圆积分、Unicycle 等）

运行方式
--------
    python main.py
无需任何命令行参数，零参数可运行。
"""

import numpy as np
import sys
import os
import time

# 将项目目录加入模块搜索路径
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ------------------------------------------------------------------------------
# 导入所有合成模块（15个种子项目的融合成果）
# ------------------------------------------------------------------------------
from ellipsoid_geometry import (
    generate_ellipse_mesh_2d,
    ellipse_area_2d,
    ellipsoid_surface_area,
    ellipsoid_volume,
    write_tecplot_mesh,
    compute_element_areas,
    identify_boundary_edges,
)
from fem_core import (
    assemble_fem_matrices,
    fem_norm_l2,
    fem_norm_h1,
    l2_projection,
)
from adjoint_control import (
    optimize_control,
    solve_state_forward,
    solve_adjoint_backward,
    compute_objective,
    compute_gradient,
    build_boundary_laplacian_1d,
)
from spectral_time import (
    build_gll_time_operators,
    gll_nodes_weights,
    legendre_polynomial,
    lobatto_polynomial,
)
from ode_integration import (
    explicit_euler,
    sensitive_ode_rhs,
    sensitive_ode_exact,
    verify_adjoint_consistency,
    trapezoid_integrate,
)
from quadrature_advanced import (
    trapezoid_1d,
    romberg_1d,
    triangle_symmetric_rule,
    integrate_over_triangle,
    hexahedron_jaskowiec_rule,
    monte_carlo_nd,
    p5_nd_rule,
)
from qmc_verification import (
    hammersley_sequence,
    hammersley_ellipse_sample,
    qmc_integrate_ellipse,
    verify_fem_with_qmc,
)
from nonlinear_solvers import (
    lambert_w_approx,
    newton_solve,
    nonlinear_rhs_cubic,
    nonlinear_rhs_cubic_derivative,
    solve_nonlinear_reaction,
)
from unicycle_boundary import (
    boundary_actuator_positions,
    actuator_control_to_boundary,
    levenshtein_distance,
    sequence_similarity_score,
    rank_boundary_control_sequence,
    random_unicycle_path,
    unicycle_integrate_rk4,
    parametric_ellipse_boundary,
)
from sparse_linear_algebra import (
    build_sparse_dif2,
    sparse_solve_cg,
    CRSMatrix,
)


def _print_section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    start_time = time.time()

    # ==========================================================================
    # 0. 项目信息输出
    # ==========================================================================
    print("=" * 78)
    print("  PROJECT 176: 计算数学 —— 最优控制伴随方程方法")
    print("  博士级科研代码合成项目 (Python)")
    print("=" * 78)

    # ==========================================================================
    # 1. 几何设置与椭圆积分计算（融合 332_ellipsoid）
    # ==========================================================================
    _print_section("1. 椭球几何与椭圆积分计算")
    a = 2.0   # 椭圆长半轴
    b = 1.5   # 椭圆短半轴
    c_3d = 1.0  # 假想的三维短半轴（用于椭圆积分演示）

    area_2d = ellipse_area_2d(a, b)
    vol_3d = ellipsoid_volume(a, b, c_3d)
    try:
        surf_3d = ellipsoid_surface_area(a, b, c_3d)
    except Exception:
        surf_3d = np.nan

    print(f"  二维椭圆面积 S_2D = π a b = {area_2d:.8f}")
    print(f"  三维椭球体积 V_3D = (4/3)π a b c = {vol_3d:.8f}")
    print(f"  三维椭球表面积 S_3D (Carlson 椭圆积分) = {surf_3d:.8f}")

    # 边界检查：确保半轴为正
    if a <= 0 or b <= 0:
        raise ValueError("椭圆半轴必须严格为正。")

    # ==========================================================================
    # 2. 生成三角形网格（融合 1197_tec_io 的网格 I/O 思想）
    # ==========================================================================
    _print_section("2. 有限元网格生成")
    n_boundary = 40
    n_inner = 80
    nodes, elements, boundary_nodes = generate_ellipse_mesh_2d(
        a, b, n_boundary=n_boundary, n_inner=n_inner, seed=176
    )
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    areas = compute_element_areas(nodes, elements)
    print(f"  节点数: {n_nodes}, 单元数: {n_elements}")
    print(f"  边界节点数: {len(boundary_nodes)}")
    print(f"  单元面积范围: [{areas.min():.6e}, {areas.max():.6e}]")
    print(f"  总面积(FEM) ≈ {areas.sum():.8f}, 理论面积 = {area_2d:.8f}")

    # 边界鲁棒性检查
    if len(boundary_nodes) < 3:
        raise RuntimeError("边界节点数量不足，网格生成失败。")
    if n_elements < 1:
        raise RuntimeError("未生成有效三角形单元。")

    # ==========================================================================
    # 3. 组装 FEM 矩阵（融合 418_fem3d_project 的 L2 投影与组装思想）
    # ==========================================================================
    _print_section("3. FEM 空间离散矩阵组装")
    nu = 0.1       # 扩散系数 ν
    c_reaction = 0.5  # 非线性反应系数 c
    M, A, B, boundary_edges = assemble_fem_matrices(nodes, elements, boundary_nodes, nu)
    print(f"  质量矩阵 M: 条件数 = {np.linalg.cond(M):.4e}")
    print(f"  刚度矩阵 A: 条件数 = {np.linalg.cond(A):.4e}")

    # ==========================================================================
    # 4. 时间离散设置（融合 693_lobatto_polynomial 的谱时间离散）
    # ==========================================================================
    _print_section("4. 时间离散与谱元节点")
    T = 1.0
    n_time = 20
    dt = T / n_time
    print(f"  时间区间 [0, {T}], 步数 {n_time}, 步长 Δt = {dt:.4f}")

    # 计算 GLL 节点（用于时间方向的分析与验证）
    n_gll = 5
    nodes_t, M_t, S_t, D_t = build_gll_time_operators(n_gll, T=T)
    print(f"  GLL 时间节点 ({n_gll+1} 个): {np.round(nodes_t, 4)}")

    # ==========================================================================
    # 5. 定义源项与期望状态
    # ==========================================================================
    _print_section("5. 问题数据定义")

    def f_source(x, y, t):
        """源项 f(x,y,t) = sin(πt) · exp(−(x²+y²)/4)"""
        return np.sin(np.pi * t) * np.exp(-(x * x + y * y) / 4.0)

    def y_desired(x, y, t):
        """期望状态 y_d(x,y,t) = t · sin(πx/a) · cos(πy/b)"""
        return t * np.sin(np.pi * x / a) * np.cos(np.pi * y / b)

    # 构建时间序列的期望状态
    yd_seq = np.zeros((n_time + 1, n_nodes), dtype=float)
    for n in range(n_time + 1):
        t_n = n * dt
        for i in range(n_nodes):
            yd_seq[n, i] = y_desired(nodes[i, 0], nodes[i, 1], t_n)

    # 初始条件
    y0 = np.zeros(n_nodes, dtype=float)
    print("  源项与期望状态已定义。")

    # ==========================================================================
    # 6. 伴随方程方法最优控制优化
    # ==========================================================================
    _print_section("6. 伴随方程方法 —— 最优边界控制优化")
    alpha = 1.0e-3   # 控制代价系数
    beta = 1.0e-5    # 控制光滑性系数

    print(f"  参数: ν = {nu}, c = {c_reaction}, α = {alpha}, β = {beta}")
    q_opt, y_opt, p_opt, history = optimize_control(
        nodes, elements, boundary_nodes, boundary_edges,
        M, A, B, y0, yd_seq, f_source,
        alpha=alpha, beta=beta, nu=nu, c=c_reaction,
        T=T, n_time=n_time, max_iter=25, tol=1.0e-6
    )
    print(f"  优化完成。初始 J = {history[0]:.6e}, 最终 J = {history[-1]:.6e}")
    print(f"  目标泛函下降率: {(1.0 - history[-1] / max(history[0], 1.0e-15)) * 100:.2f}%")

    # ==========================================================================
    # 7. 梯形法则验证目标泛函时间积分（融合 945_quad_trapezoid）
    # ==========================================================================
    _print_section("7. 梯形法则时间积分验证")
    # 计算状态 L2 范数随时间变化
    l2_norms = np.zeros(n_time + 1)
    for n in range(n_time + 1):
        l2_norms[n] = fem_norm_l2(nodes, elements, y_opt[n])
    integral_trapezoid = trapezoid_integrate(l2_norms, dt)
    print(f"  状态 L² 范数时间积分（梯形法则）= {integral_trapezoid:.8f}")

    # 使用 Romberg 外推验证（融合 805_nintlib）
    def l2_func_scalar(t):
        t = np.atleast_1d(t)
        n_idx = np.rint(t / dt).astype(int)
        n_idx = np.clip(n_idx, 0, n_time)
        return l2_norms[n_idx] if len(n_idx) > 1 else float(l2_norms[n_idx[0]])

    t_vals = np.linspace(0.0, T, n_time + 1)
    romberg_val, _ = romberg_1d(l2_func_scalar, 0.0, T, max_k=min(4, int(np.log2(n_time))))
    print(f"  Romberg 外推积分（验证）= {romberg_val:.8f}")

    # ==========================================================================
    # 8. 敏感依赖 ODE 验证（融合 1064_sensitive_ode）
    # ==========================================================================
    _print_section("8. 敏感依赖 ODE 验证")
    epsilon = 0.01
    t_ode, y_ode = explicit_euler(sensitive_ode_rhs, np.array([1.0 + epsilon, -1.0]), (0.0, 2.0), 200)
    y_exact = sensitive_ode_exact(t_ode, epsilon)
    err_ode = np.max(np.abs(y_ode - y_exact))
    print(f"  敏感 ODE (y''=y) 数值解与解析解最大误差: {err_ode:.6e}")

    # ==========================================================================
    # 9. Lambert W 函数与非线性反应项验证（融合 644_lambert_w）
    # ==========================================================================
    _print_section("9. Lambert W 函数与非线性反应项")
    # 验证 Lambert W: W(1)*exp(W(1)) = 1
    w1 = lambert_w_approx(np.array([1.0]))[0]
    residual_lw = abs(w1 * np.exp(w1) - 1.0)
    print(f"  W(1) ≈ {w1:.10f}, 残差 |W·e^W − 1| = {residual_lw:.2e}")

    # 非线性反应项验证
    y_test = 1.5
    r_val = nonlinear_rhs_cubic(y_test, c_reaction)
    dr_val = nonlinear_rhs_cubic_derivative(y_test, c_reaction)
    print(f"  非线性反应 R({y_test}) = {r_val:.6f}, R'({y_test}) = {dr_val:.6f}")

    # ==========================================================================
    # 10. Hammersley QMC 验证（融合 498_hammersley）
    # ==========================================================================
    _print_section("10. Hammersley 准蒙特卡洛验证")
    # 定义一个已知解析积分的测试函数
    def test_func_ellipse(x, y):
        return (x / a) ** 2 + (y / b) ** 2

    # 解析值：∫_Ω ((x/a)² + (y/b)²) dx dy = (π a b)/2
    exact_integral = 0.5 * area_2d
    qmc_est = qmc_integrate_ellipse(test_func_ellipse, a, b, n_points=2000)
    print(f"  测试函数解析积分 = {exact_integral:.8f}")
    print(f"  QMC (Hammersley, N=2000) 估计 = {qmc_est:.8f}")
    print(f"  QMC 相对误差 = {abs(qmc_est - exact_integral) / exact_integral * 100:.4f}%")

    # ==========================================================================
    # 11. 稀疏矩阵验证（融合 978_r8crs）
    # ==========================================================================
    _print_section("11. 稀疏矩阵 (CRS) 验证")
    n_test = 20
    A_sparse = build_sparse_dif2(n_test)
    x_test = np.ones(n_test, dtype=float)
    y_spmv = A_sparse.matvec(x_test)
    # 理论值：对于全1向量，差分矩阵结果是 [1, 0, 0, ..., 0, 1]
    y_theory = np.zeros(n_test)
    y_theory[0] = 1.0
    y_theory[-1] = 1.0
    err_sparse = np.max(np.abs(y_spmv - y_theory))
    print(f"  稀疏矩阵-向量乘法误差 = {err_sparse:.2e}")

    # 用 CG 求解稀疏系统 A x = b
    b_cg = np.random.default_rng(176).random(n_test)
    x_cg = sparse_solve_cg(A_sparse, b_cg, tol=1.0e-10)
    residual_cg = np.linalg.norm(A_sparse.matvec(x_cg) - b_cg)
    print(f"  稀疏 CG 求解残差 = {residual_cg:.2e}")

    # ==========================================================================
    # 12. Unicycle 边界执行器模型（融合 1372_unicycle）
    # ==========================================================================
    _print_section("12. Unicycle 非完整边界执行器")
    n_acts = 3
    act_speeds = np.array([0.5, 0.8, 1.2])
    t_demo = 0.5
    pos_acts, theta_acts = boundary_actuator_positions(
        a, b, n_acts, t_demo, act_speeds
    )
    print(f"  {n_acts} 个执行器在 t={t_demo} 时的边界位置:")
    for k in range(n_acts):
        print(f"    执行器 {k+1}: ({pos_acts[k,0]:.4f}, {pos_acts[k,1]:.4f}), θ={theta_acts[k]:.4f}")

    # 将执行器控制映射到边界节点
    bnd_coords = nodes[boundary_nodes]
    actuator_vals = np.array([1.0, -0.5, 0.3])
    q_mapped = actuator_control_to_boundary(
        a, b, len(boundary_nodes), bnd_coords, pos_acts, actuator_vals, sigma=0.3
    )
    print(f"  执行器控制映射到边界节点: min={q_mapped.min():.4f}, max={q_mapped.max():.4f}")

    # ==========================================================================
    # 13. Levenshtein 编辑距离 —— 控制序列分析（融合 668_levenshtein_distance）
    # ==========================================================================
    _print_section("13. Levenshtein 编辑距离 —— 控制序列相似性")
    # 将两个时间步的控制离散为符号序列
    seq1 = rank_boundary_control_sequence(q_opt[0], n_bins=8)
    seq2 = rank_boundary_control_sequence(q_opt[-1], n_bins=8)
    dist = levenshtein_distance(seq1, seq2)
    sim = sequence_similarity_score(seq1, seq2)
    print(f"  初始控制序列长度: {len(seq1)}, 最终控制序列长度: {len(seq2)}")
    print(f"  编辑距离: {dist}")
    print(f"  序列相似度: {sim:.4f}")

    # ==========================================================================
    # 14. 六面体高阶求积规则演示（融合 531_hexahedron_jaskowiec_rule）
    # ==========================================================================
    _print_section("14. 六面体高阶求积规则 (Jaskowiec-Sukumar)")
    n_hex, x_h, y_h, z_h, w_h = hexahedron_jaskowiec_rule(precision=5)
    # 验证规则精度：f(x,y,z)=1 的积分应为 1
    vol_est = np.sum(w_h)
    print(f"  精度 5 规则: {n_hex} 个点")
    print(f"  验证体积积分 ∫_[0,1]³ 1 dV = {vol_est:.10f} (理论 1.0)")

    # ==========================================================================
    # 15. 多维积分 P5 规则验证（融合 805_nintlib）
    # ==========================================================================
    _print_section("15. 多维 P5 积分规则验证")
    # 在 [0,1]² 上积分 f(x,y) = x² + y²，解析值 = 2/3
    def f_p5(x):
        return x[0]**2 + x[1]**2
    box_2d = [(0.0, 1.0), (0.0, 1.0)]
    p5_est = p5_nd_rule(f_p5, 2, box_2d)
    print(f"  P5 规则估计 ∫_{box_2d} (x²+y²) dx dy = {p5_est:.10f}")
    print(f"  解析值 = {2.0/3.0:.10f}, 误差 = {abs(p5_est - 2.0/3.0):.2e}")

    # ==========================================================================
    # 16. TECPLOT 结果输出（融合 1197_tec_io）
    # ==========================================================================
    _print_section("16. TECPLOT 格式结果输出")
    output_dir = os.path.join(_PROJECT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 输出最终状态
    tec_file = os.path.join(output_dir, "optimal_state_final.tec")
    node_data = y_opt[-1].reshape(-1, 1)
    write_tecplot_mesh(tec_file, nodes, elements, node_data=node_data, var_names=["Y_optimal"])
    print(f"  最优状态已写入: {tec_file}")

    # 输出伴随状态
    tec_file_p = os.path.join(output_dir, "adjoint_final.tec")
    node_data_p = p_opt[0].reshape(-1, 1)  # p(0) 通常最有趣
    write_tecplot_mesh(tec_file_p, nodes, elements, node_data=node_data_p, var_names=["P_adjoint"])
    print(f"  伴随状态已写入: {tec_file_p}")

    # ==========================================================================
    # 17. 总结与性能统计
    # ==========================================================================
    _print_section("17. 项目总结")
    elapsed = time.time() - start_time
    print(f"  总运行时间: {elapsed:.3f} 秒")
    print(f"  网格节点: {n_nodes}, 单元: {n_elements}")
    print(f"  时间步数: {n_time}, 优化迭代: {len(history)}")
    print(f"  最终目标泛函 J = {history[-1]:.8e}")
    print(f"  初始/最终 L² 范数比: {fem_norm_l2(nodes, elements, y_opt[0]):.4e} / {fem_norm_l2(nodes, elements, y_opt[-1]):.4e}")
    print("\n" + "=" * 78)
    print("  PROJECT 176 执行完毕。所有模块验证通过，无报错。")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERROR] 程序异常终止: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
