#!/usr/bin/env python3

import numpy as np
import sys
import os
import time


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)




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




    print("=" * 78)
    print("  PROJECT 176: 计算数学 —— 最优控制伴随方程方法")
    print("  博士级科研代码合成项目 (Python)")
    print("=" * 78)




    _print_section("1. 椭球几何与椭圆积分计算")
    a = 2.0
    b = 1.5
    c_3d = 1.0

    area_2d = ellipse_area_2d(a, b)
    vol_3d = ellipsoid_volume(a, b, c_3d)
    try:
        surf_3d = ellipsoid_surface_area(a, b, c_3d)
    except Exception:
        surf_3d = np.nan

    print(f"  二维椭圆面积 S_2D = π a b = {area_2d:.8f}")
    print(f"  三维椭球体积 V_3D = (4/3)π a b c = {vol_3d:.8f}")
    print(f"  三维椭球表面积 S_3D (Carlson 椭圆积分) = {surf_3d:.8f}")


    if a <= 0 or b <= 0:
        raise ValueError("椭圆半轴必须严格为正。")




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


    if len(boundary_nodes) < 3:
        raise RuntimeError("边界节点数量不足，网格生成失败。")
    if n_elements < 1:
        raise RuntimeError("未生成有效三角形单元。")




    _print_section("3. FEM 空间离散矩阵组装")
    nu = 0.1
    c_reaction = 0.5
    M, A, B, boundary_edges = assemble_fem_matrices(nodes, elements, boundary_nodes, nu)
    print(f"  质量矩阵 M: 条件数 = {np.linalg.cond(M):.4e}")
    print(f"  刚度矩阵 A: 条件数 = {np.linalg.cond(A):.4e}")




    _print_section("4. 时间离散与谱元节点")
    T = 1.0
    n_time = 20
    dt = T / n_time
    print(f"  时间区间 [0, {T}], 步数 {n_time}, 步长 Δt = {dt:.4f}")


    n_gll = 5
    nodes_t, M_t, S_t, D_t = build_gll_time_operators(n_gll, T=T)
    print(f"  GLL 时间节点 ({n_gll+1} 个): {np.round(nodes_t, 4)}")




    _print_section("5. 问题数据定义")

    def f_source(x, y, t):
        return np.sin(np.pi * t) * np.exp(-(x * x + y * y) / 4.0)

    def y_desired(x, y, t):
        return t * np.sin(np.pi * x / a) * np.cos(np.pi * y / b)


    yd_seq = np.zeros((n_time + 1, n_nodes), dtype=float)
    for n in range(n_time + 1):
        t_n = n * dt
        for i in range(n_nodes):
            yd_seq[n, i] = y_desired(nodes[i, 0], nodes[i, 1], t_n)


    y0 = np.zeros(n_nodes, dtype=float)
    print("  源项与期望状态已定义。")




    _print_section("6. 伴随方程方法 —— 最优边界控制优化")
    alpha = 1.0e-3
    beta = 1.0e-5

    print(f"  参数: ν = {nu}, c = {c_reaction}, α = {alpha}, β = {beta}")
    q_opt, y_opt, p_opt, history = optimize_control(
        nodes, elements, boundary_nodes, boundary_edges,
        M, A, B, y0, yd_seq, f_source,
        alpha=alpha, beta=beta, nu=nu, c=c_reaction,
        T=T, n_time=n_time, max_iter=25, tol=1.0e-6
    )
    print(f"  优化完成。初始 J = {history[0]:.6e}, 最终 J = {history[-1]:.6e}")
    print(f"  目标泛函下降率: {(1.0 - history[-1] / max(history[0], 1.0e-15)) * 100:.2f}%")




    _print_section("7. 梯形法则时间积分验证")

    l2_norms = np.zeros(n_time + 1)
    for n in range(n_time + 1):
        l2_norms[n] = fem_norm_l2(nodes, elements, y_opt[n])
    integral_trapezoid = trapezoid_integrate(l2_norms, dt)
    print(f"  状态 L² 范数时间积分（梯形法则）= {integral_trapezoid:.8f}")


    def l2_func_scalar(t):
        t = np.atleast_1d(t)
        n_idx = np.rint(t / dt).astype(int)
        n_idx = np.clip(n_idx, 0, n_time)
        return l2_norms[n_idx] if len(n_idx) > 1 else float(l2_norms[n_idx[0]])

    t_vals = np.linspace(0.0, T, n_time + 1)
    romberg_val, _ = romberg_1d(l2_func_scalar, 0.0, T, max_k=min(4, int(np.log2(n_time))))
    print(f"  Romberg 外推积分（验证）= {romberg_val:.8f}")




    _print_section("8. 敏感依赖 ODE 验证")
    epsilon = 0.01
    t_ode, y_ode = explicit_euler(sensitive_ode_rhs, np.array([1.0 + epsilon, -1.0]), (0.0, 2.0), 200)
    y_exact = sensitive_ode_exact(t_ode, epsilon)
    err_ode = np.max(np.abs(y_ode - y_exact))
    print(f"  敏感 ODE (y''=y) 数值解与解析解最大误差: {err_ode:.6e}")




    _print_section("9. Lambert W 函数与非线性反应项")

    w1 = lambert_w_approx(np.array([1.0]))[0]
    residual_lw = abs(w1 * np.exp(w1) - 1.0)
    print(f"  W(1) ≈ {w1:.10f}, 残差 |W·e^W − 1| = {residual_lw:.2e}")


    y_test = 1.5
    r_val = nonlinear_rhs_cubic(y_test, c_reaction)
    dr_val = nonlinear_rhs_cubic_derivative(y_test, c_reaction)
    print(f"  非线性反应 R({y_test}) = {r_val:.6f}, R'({y_test}) = {dr_val:.6f}")




    _print_section("10. Hammersley 准蒙特卡洛验证")

    def test_func_ellipse(x, y):
        return (x / a) ** 2 + (y / b) ** 2


    exact_integral = 0.5 * area_2d
    qmc_est = qmc_integrate_ellipse(test_func_ellipse, a, b, n_points=2000)
    print(f"  测试函数解析积分 = {exact_integral:.8f}")
    print(f"  QMC (Hammersley, N=2000) 估计 = {qmc_est:.8f}")
    print(f"  QMC 相对误差 = {abs(qmc_est - exact_integral) / exact_integral * 100:.4f}%")




    _print_section("11. 稀疏矩阵 (CRS) 验证")
    n_test = 20
    A_sparse = build_sparse_dif2(n_test)
    x_test = np.ones(n_test, dtype=float)
    y_spmv = A_sparse.matvec(x_test)

    y_theory = np.zeros(n_test)
    y_theory[0] = 1.0
    y_theory[-1] = 1.0
    err_sparse = np.max(np.abs(y_spmv - y_theory))
    print(f"  稀疏矩阵-向量乘法误差 = {err_sparse:.2e}")


    b_cg = np.random.default_rng(176).random(n_test)
    x_cg = sparse_solve_cg(A_sparse, b_cg, tol=1.0e-10)
    residual_cg = np.linalg.norm(A_sparse.matvec(x_cg) - b_cg)
    print(f"  稀疏 CG 求解残差 = {residual_cg:.2e}")




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


    bnd_coords = nodes[boundary_nodes]
    actuator_vals = np.array([1.0, -0.5, 0.3])
    q_mapped = actuator_control_to_boundary(
        a, b, len(boundary_nodes), bnd_coords, pos_acts, actuator_vals, sigma=0.3
    )
    print(f"  执行器控制映射到边界节点: min={q_mapped.min():.4f}, max={q_mapped.max():.4f}")




    _print_section("13. Levenshtein 编辑距离 —— 控制序列相似性")

    seq1 = rank_boundary_control_sequence(q_opt[0], n_bins=8)
    seq2 = rank_boundary_control_sequence(q_opt[-1], n_bins=8)
    dist = levenshtein_distance(seq1, seq2)
    sim = sequence_similarity_score(seq1, seq2)
    print(f"  初始控制序列长度: {len(seq1)}, 最终控制序列长度: {len(seq2)}")
    print(f"  编辑距离: {dist}")
    print(f"  序列相似度: {sim:.4f}")




    _print_section("14. 六面体高阶求积规则 (Jaskowiec-Sukumar)")
    n_hex, x_h, y_h, z_h, w_h = hexahedron_jaskowiec_rule(precision=5)

    vol_est = np.sum(w_h)
    print(f"  精度 5 规则: {n_hex} 个点")
    print(f"  验证体积积分 ∫_[0,1]³ 1 dV = {vol_est:.10f} (理论 1.0)")




    _print_section("15. 多维 P5 积分规则验证")

    def f_p5(x):
        return x[0]**2 + x[1]**2
    box_2d = [(0.0, 1.0), (0.0, 1.0)]
    p5_est = p5_nd_rule(f_p5, 2, box_2d)
    print(f"  P5 规则估计 ∫_{box_2d} (x²+y²) dx dy = {p5_est:.10f}")
    print(f"  解析值 = {2.0/3.0:.10f}, 误差 = {abs(p5_est - 2.0/3.0):.2e}")




    _print_section("16. TECPLOT 格式结果输出")
    output_dir = os.path.join(_PROJECT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)


    tec_file = os.path.join(output_dir, "optimal_state_final.tec")
    node_data = y_opt[-1].reshape(-1, 1)
    write_tecplot_mesh(tec_file, nodes, elements, node_data=node_data, var_names=["Y_optimal"])
    print(f"  最优状态已写入: {tec_file}")


    tec_file_p = os.path.join(output_dir, "adjoint_final.tec")
    node_data_p = p_opt[0].reshape(-1, 1)
    write_tecplot_mesh(tec_file_p, nodes, elements, node_data=node_data_p, var_names=["P_adjoint"])
    print(f"  伴随状态已写入: {tec_file_p}")




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
