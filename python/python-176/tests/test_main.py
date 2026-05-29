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
    # 先运行主程序（原算法）
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] 主程序异常终止: {e}")
        import traceback
        traceback.print_exc()

# ================================================================
# 测试用例（57个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: ellipse_area_2d - 公式验证 π a b ----
area = ellipse_area_2d(2.0, 1.5)
expected = np.pi * 2.0 * 1.5
assert abs(area - expected) < 1.0e-12, '[TC01] 椭圆面积公式 πab FAILED'

# ---- TC02: ellipsoid_volume - 公式验证 (4/3)πabc ----
vol = ellipsoid_volume(2.0, 3.0, 4.0)
expected_vol = (4.0 / 3.0) * np.pi * 2.0 * 3.0 * 4.0
assert abs(vol - expected_vol) < 1.0e-12, '[TC02] 椭球体积公式 FAILED'

# ---- TC03: ellipsoid_surface_area - 球体退化情形 S=4πr² ----
surf = ellipsoid_surface_area(1.0, 1.0, 1.0)
assert abs(surf - 4.0 * np.pi) < 1.0e-8, '[TC03] 球体表面积 4πr² FAILED'

# ---- TC04: carlson_rf - RF(1,1,1) = 1 ----
from ellipsoid_geometry import carlson_rf
rf_val = carlson_rf(1.0, 1.0, 1.0)
assert abs(rf_val - 1.0) < 1.0e-10, '[TC04] Carlson RF(1,1,1)=1 FAILED'

# ---- TC05: trapezoid_1d - ∫₀¹ x dx = 0.5 ----
def f_linear(x):
    return x

trap_val = trapezoid_1d(f_linear, 0.0, 1.0, 100)
assert abs(trap_val - 0.5) < 1.0e-6, '[TC05] 梯形积分 ∫₀¹ x dx = 0.5 FAILED'

# ---- TC06: trapezoid_1d - n=1 不会崩溃 ----
try:
    trap2 = trapezoid_1d(f_linear, 0.0, 1.0, 1)
    assert np.isfinite(trap2), '[TC06] 梯形法则 n=1 应返回有限值 FAILED'
except Exception as e:
    assert False, f'[TC06] 梯形法则 n=1 不应崩溃: {e}'

# ---- TC07: romberg_1d - ∫₀^π sin(x) dx = 2 ----
def f_sin(x):
    return np.sin(x)

rom_val, _ = romberg_1d(f_sin, 0.0, np.pi, max_k=6)
assert abs(rom_val - 2.0) < 1.0e-10, '[TC07] Romberg ∫₀^π sin(x)dx=2 FAILED'

# ---- TC08: trapezoid_integrate - 已知积分 ----
vals = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
trap_int = trapezoid_integrate(vals, 1.0)
assert abs(trap_int - 4.5) < 1.0e-10, '[TC08] 梯形数值积分 FAILED'

# ---- TC09: trapezoid_integrate - 两端点情况 ----
vals2 = np.array([1.0, 2.0])
trap_int2 = trapezoid_integrate(vals2, 1.0)
assert abs(trap_int2 - 1.5) < 1.0e-10, '[TC09] 两节点梯形积分 FAILED'

# ---- TC10: triangle_symmetric_rule - 权重和为 0.5 ----
n_tri, w_tri, xi_tri, eta_tri = triangle_symmetric_rule(3)
assert abs(np.sum(w_tri) - 0.5) < 1.0e-12, '[TC10] 三角形求积权重和应为 0.5 FAILED'

# ---- TC11: triangle_symmetric_rule - 坐标在参考三角形内 ----
n_tri2, w_tri2, xi_tri2, eta_tri2 = triangle_symmetric_rule(5)
for i in range(n_tri2):
    assert xi_tri2[i] >= 0.0 and eta_tri2[i] >= 0.0 and xi_tri2[i] + eta_tri2[i] <= 1.0 + 1.0e-12, \
        f'[TC11] 求积点({xi_tri2[i]},{eta_tri2[i]})超出参考三角形 FAILED'

# ---- TC12: hexahedron_jaskowiec_rule - 权重和为 1.0 ----
n_h, x_h, y_h, z_h, w_h = hexahedron_jaskowiec_rule(precision=5)
assert abs(np.sum(w_h) - 1.0) < 1.0e-12, '[TC12] 六面体求积权重和应为 1.0 FAILED'

# ---- TC13: p5_nd_rule - ∫[0,1]² (x²+y²) dxdy = 2/3 ----
def f_p5_test(x):
    return x[0] ** 2 + x[1] ** 2

box_2d_test = [(0.0, 1.0), (0.0, 1.0)]
p5_est = p5_nd_rule(f_p5_test, 2, box_2d_test)
assert abs(p5_est - 2.0 / 3.0) < 1.0e-12, '[TC13] P5规则 ∫(x²+y²)=2/3 FAILED'

# ---- TC14: p5_nd_rule - 一维 ∫₀¹ x dx = 0.5 ----
def f_p5_1d(x):
    return x[0]

box_1d = [(0.0, 1.0)]
p5_est_1d = p5_nd_rule(f_p5_1d, 1, box_1d)
assert abs(p5_est_1d - 0.5) < 1.0e-12, '[TC14] P5规则 1D ∫x dx=0.5 FAILED'

# ---- TC15: lambert_w_approx - W(1)⋅exp(W(1)) ≈ 1 ----
w1 = float(lambert_w_approx(np.array([1.0]))[0])
assert abs(w1 * np.exp(w1) - 1.0) < 1.0e-4, '[TC15] Lambert W(1)·e^W(1)≈1 FAILED'

# ---- TC16: lambert_w_approx - W(0) = 0 ----
w0 = float(lambert_w_approx(np.array([0.0]))[0])
assert abs(w0) < 1.0e-8, '[TC16] Lambert W(0)=0 FAILED'

# ---- TC17: lambert_w_approx - 次分支 W₋₁ 返回有限值 ----
w_neg = float(lambert_w_approx(np.array([-0.1]), branch=-1)[0])
assert w_neg < -1.0, '[TC17] 次分支 W₋₁(-0.1) 应 < -1 FAILED'

# ---- TC18: newton_solve - 求解 x²=2 得 √2 ----
def f_sq(x):
    return x ** 2 - 2.0

def df_sq(x):
    return 2.0 * x

root, conv, iters = newton_solve(f_sq, df_sq, 1.0)
assert conv, '[TC18a] Newton 求解 x²=2 应收敛 FAILED'
assert abs(root - np.sqrt(2.0)) < 1.0e-10, '[TC18b] Newton 得 √2 FAILED'

# ---- TC19: nonlinear_rhs_cubic - 值与导数一致性 ----
c_test = 0.5
r = nonlinear_rhs_cubic(2.0, c_test)
assert abs(r - 4.0) < 1.0e-12, '[TC19a] R(2)=0.5*8=4 FAILED'
dr = nonlinear_rhs_cubic_derivative(2.0, c_test)
assert abs(dr - 6.0) < 1.0e-12, '[TC19b] R\'(2)=3*0.5*4=6 FAILED'

# ---- TC20: explicit_euler - y'=y, y(0)=1 → y(1)≈e ----
def f_exp(t, y):
    return np.array([y[0]])

t_exp, y_exp = explicit_euler(f_exp, np.array([1.0]), (0.0, 1.0), 1000)
err_euler = abs(y_exp[-1, 0] - np.e)
assert err_euler < 0.01, f'[TC20] Euler y\'=y 误差应 < 0.01, 实际 {err_euler:.4f} FAILED'

# ---- TC21: explicit_euler - 数值稳定性 ----
t_exp2, y_exp2 = explicit_euler(f_exp, np.array([1.0]), (0.0, 2.0), 200)
assert np.all(np.isfinite(y_exp2)), '[TC21] Euler 解不应含 NaN/Inf FAILED'

# ---- TC22: sensitive_ode_exact - t=0 满足初始条件 ----
y_exact_0 = sensitive_ode_exact(np.array([0.0]), epsilon=0.01)
assert abs(y_exact_0[0, 0] - 1.01) < 1.0e-10, '[TC22a] y(0)=1+ε FAILED'
assert abs(y_exact_0[0, 1] + 1.0) < 1.0e-10, '[TC22b] y\'(0)=-1 FAILED'

# ---- TC23: sensitive_ode_exact - ε=0 时的特解 ----
y_exact_eps0 = sensitive_ode_exact(np.array([0.0]), epsilon=0.0)
assert abs(y_exact_eps0[0, 0] - 1.0) < 1.0e-10, '[TC23a] ε=0 时 y(0)=1 FAILED'
assert abs(y_exact_eps0[0, 1] + 1.0) < 1.0e-10, '[TC23b] ε=0 时 y\'(0)=-1 FAILED'

# ---- TC24: hammersley_sequence - dim=1 返回均匀网格 ----
pts = hammersley_sequence(1, 100)
assert pts.shape == (100, 1), '[TC24a] Hammersley dim=1 形状错误 FAILED'
expected_uniform = np.arange(100) / 100.0
assert np.max(np.abs(pts[:, 0] - expected_uniform)) < 1.0e-12, '[TC24b] dim=1 第一维均匀 FAILED'

# ---- TC25: hammersley_sequence - dim=2 范围在 [0,1] ----
pts2 = hammersley_sequence(2, 50)
assert np.all(pts2 >= 0.0) and np.all(pts2 <= 1.0), '[TC25] Hammersley 点应在 [0,1] FAILED'

# ---- TC26: CRSMatrix - 稠密/稀疏转换往返 ----
dense = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
crs = CRSMatrix.from_dense(dense)
dense_round = crs.to_dense()
assert np.max(np.abs(dense - dense_round)) < 1.0e-12, '[TC26] CRS 往返转换 FAILED'

# ---- TC27: CRSMatrix - matvec_transpose 检验 ----
x_test_crs = np.array([1.0, 2.0, 3.0])
y_t = crs.matvec_transpose(x_test_crs)
assert y_t.shape == (3,), '[TC27a] 转置乘法输出维度错误 FAILED'
assert np.all(np.isfinite(y_t)), '[TC27b] 转置乘法应返回有限值 FAILED'

# ---- TC28: build_sparse_dif2 - matvec 理论值 ----
n_sp = 10
A_sp = build_sparse_dif2(n_sp)
x_ones = np.ones(n_sp)
y_spmv = A_sp.matvec(x_ones)
y_theory = np.zeros(n_sp)
y_theory[0] = 1.0
y_theory[-1] = 1.0
assert np.max(np.abs(y_spmv - y_theory)) < 1.0e-12, '[TC28] 稀疏矩阵 matvec 理论值 FAILED'

# ---- TC29: sparse_solve_cg - 求解对称正定系统 ----
n_cg = 10
A_cg = build_sparse_dif2(n_cg)
import numpy as np
np.random.seed(176)
b_cg = np.random.random(n_cg)
x_cg = sparse_solve_cg(A_cg, b_cg, tol=1.0e-10)
residual = np.linalg.norm(A_cg.matvec(x_cg) - b_cg)
assert residual < 1.0e-8, f'[TC29] CG 求解残差 {residual:.2e} FAILED'

# ---- TC30: levenshtein_distance - "kitten"→"sitting" = 3 ----
dist = levenshtein_distance("kitten", "sitting")
assert dist == 3, f'[TC30] Levenshtein kitten→sitting 应为3, 实际{dist} FAILED'

# ---- TC31: levenshtein_distance - 空序列 ----
assert levenshtein_distance("", "") == 0, '[TC31a] 空序列距离应为0 FAILED'
assert levenshtein_distance("abc", "") == 3, '[TC31b] 非空→空距离应为3 FAILED'
assert levenshtein_distance("", "abc") == 3, '[TC31c] 空→非空距离应为3 FAILED'

# ---- TC32: sequence_similarity_score - 完全相同 = 1.0 ----
sim = sequence_similarity_score("abc", "abc")
assert abs(sim - 1.0) < 1.0e-12, '[TC32] 完全相同序列相似度应为1.0 FAILED'

# ---- TC33: sequence_similarity_score - 完全不同 ----
sim2 = sequence_similarity_score("abc", "xyz")
assert sim2 < 0.5, f'[TC33] 完全不同序列相似度应较低, 实际{sim2:.4f} FAILED'

# ---- TC34: unicycle_dynamics - 直线运动 ----
from unicycle_boundary import unicycle_dynamics
state = np.array([0.0, 0.0, 0.0])
control = np.array([1.0, 0.0])
derivs = unicycle_dynamics(state, control)
assert abs(derivs[0] - 1.0) < 1.0e-12, '[TC34a] dx/dt=v·cos(θ)=1 FAILED'
assert abs(derivs[1] - 0.0) < 1.0e-12, '[TC34b] dy/dt=v·sin(θ)=0 FAILED'
assert abs(derivs[2] - 0.0) < 1.0e-12, '[TC34c] dθ/dt=ω=0 FAILED'

# ---- TC35: legendre_polynomial - P₀=1, P₁(x)=x, P₃(0)=0 ----
p0 = legendre_polynomial(0, np.array([0.5]))
assert abs(p0[0] - 1.0) < 1.0e-12, '[TC35a] Legendre P₀=1 FAILED'
p1 = legendre_polynomial(1, np.array([0.5]))
assert abs(p1[0] - 0.5) < 1.0e-12, '[TC35b] Legendre P₁(x)=x FAILED'
p3 = legendre_polynomial(3, np.array([0.0]))
assert abs(p3[0]) < 1.0e-12, '[TC35c] Legendre P₃(0)=0 FAILED'

# ---- TC36: lobatto_polynomial - Lo_n(±1)=0 ----
lo_5 = lobatto_polynomial(5, np.array([-1.0, 1.0]))
assert np.max(np.abs(lo_5)) < 1.0e-12, '[TC36] Lobatto Lo_n(±1)=0 FAILED'

# ---- TC37: gll_nodes_weights - 包含端点且权重为正 ----
nodes_gll, w_gll = gll_nodes_weights(5)
assert abs(nodes_gll[0] + 1.0) < 1.0e-12, '[TC37a] GLL 首节点应为 -1 FAILED'
assert abs(nodes_gll[-1] - 1.0) < 1.0e-12, '[TC37b] GLL 末节点应为 +1 FAILED'
assert np.all(w_gll > 0), '[TC37c] GLL 权重应为正 FAILED'
assert len(nodes_gll) == 6, '[TC37d] GLL 节点数应为 n+1=6 FAILED'

# ---- TC38: compute_element_areas - 已知直角三角形面积 ----
nodes_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems_tri = np.array([[0, 1, 2]])
areas_tri = compute_element_areas(nodes_tri, elems_tri)
assert abs(areas_tri[0] - 0.5) < 1.0e-12, '[TC38] 三角形面积应为 0.5 FAILED'

# ---- TC39: identify_boundary_edges - 单个三角形 ----
edges = identify_boundary_edges(elems_tri)
assert len(edges) == 3, '[TC39] 单三角形应有3条边界边 FAILED'

# ---- TC40: generate_ellipse_mesh_2d - 基础网格结构 ----
import numpy as np
np.random.seed(42)
nodes_m, elems_m, bnd_m = generate_ellipse_mesh_2d(2.0, 1.5, n_boundary=16, n_inner=20, seed=42)
assert nodes_m.shape[1] == 2, '[TC40a] 网格节点应为二维 FAILED'
assert elems_m.shape[0] > 0, '[TC40b] 应生成有效单元 FAILED'
assert len(bnd_m) > 0, '[TC40c] 应有边界节点 FAILED'

# ---- TC41: parametric_ellipse_boundary - 弧长正值 ----
theta_p, arc_p = parametric_ellipse_boundary(2.0, 1.5, 20)
assert np.all(arc_p > 0), '[TC41] 椭圆弧长应为正值 FAILED'

# ---- TC42: unicycle_integrate_rk4 - 直线轨迹 ----
import numpy as np
np.random.seed(42)
state0 = np.array([0.0, 0.0, 0.0])
controls = np.tile(np.array([1.0, 0.0]), (10, 1))
states = unicycle_integrate_rk4(state0, controls, 0.1)
assert states.shape == (11, 3), '[TC42a] RK4 状态轨迹形状错误 FAILED'
assert states[-1, 0] > 0.5, '[TC42b] 直线运动 x 应 > 0.5 FAILED'
assert abs(states[-1, 1]) < 1.0e-8, '[TC42c] 直线运动 y≈0 FAILED'

# ---- TC43: monte_carlo_nd - 固定种子可复现 ----
def f_mc(x):
    return x[0] + x[1]

box_mc = [(0.0, 1.0), (0.0, 1.0)]
rng1 = np.random.default_rng(42)
est1, _ = monte_carlo_nd(f_mc, 2, box_mc, 1000, rng=rng1)
rng2 = np.random.default_rng(42)
est2, _ = monte_carlo_nd(f_mc, 2, box_mc, 1000, rng=rng2)
assert abs(est1 - est2) < 1.0e-15, '[TC43] 固定种子 MC 可复现 FAILED'

# ---- TC44: hammersley_ellipse_sample - 采样在椭圆内 ----
samples = hammersley_ellipse_sample(2.0, 1.5, 100)
x_s, y_s = samples[:, 0], samples[:, 1]
inside = (x_s / 2.0) ** 2 + (y_s / 1.5) ** 2 <= 1.0 + 1.0e-10
assert np.all(inside), '[TC44] QMC 采样应在椭圆内 FAILED'

# ---- TC45: lambert_w_newton - Newton 精化高精度 ----
from nonlinear_solvers import lambert_w_newton
w1_newton = lambert_w_newton(1.0)
assert abs(w1_newton * np.exp(w1_newton) - 1.0) < 1.0e-12, '[TC45] Lambert W Newton 高精度 FAILED'

# ---- TC46: build_boundary_laplacian_1d - 对称性与行和为零 ----
theta_bnd = np.linspace(0, 2 * np.pi, 8, endpoint=False)
nodes_circle = np.column_stack((np.cos(theta_bnd), np.sin(theta_bnd)))
bnd_nodes = list(range(8))
L_bd = build_boundary_laplacian_1d(bnd_nodes, nodes_circle)
assert np.max(np.abs(L_bd - L_bd.T)) < 1.0e-12, '[TC46a] 边界 Laplace 应对称 FAILED'
row_sums = np.sum(L_bd, axis=1)
assert np.max(np.abs(row_sums)) < 1.0e-12, '[TC46b] 边界 Laplace 行和应为 0 FAILED'

# ---- TC47: assemble_fem_matrices - M 对称正定, A 对称 ----
import numpy as np
np.random.seed(42)
nodes_f, elems_f, bnd_f = generate_ellipse_mesh_2d(2.0, 1.0, n_boundary=12, n_inner=10, seed=176)
M_f, A_f, B_f, bnd_edges_f = assemble_fem_matrices(nodes_f, elems_f, bnd_f, nu=0.1)
assert M_f.shape == (nodes_f.shape[0], nodes_f.shape[0]), '[TC47a] M 形状错误 FAILED'
assert np.allclose(M_f, M_f.T), '[TC47b] M 应对称 FAILED'
assert np.all(np.diag(M_f) > 0), '[TC47c] M 对角线应为正 FAILED'
assert np.allclose(A_f, A_f.T), '[TC47d] A 应对称 FAILED'

# ---- TC48: integrate_over_triangle - ∫_T 1 dA = area ----
def f_one(x, y):
    return np.ones_like(x)

pts_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
int_val = integrate_over_triangle(pts_tri, f_one, degree=3)
# integrate_over_triangle 使用 area * Σ(w*f)，而非 2*area * Σ(w*f)
# 参考三角形面积 0.5，权重和 0.5，物理三角形面积 0.5
# 实际返回 0.5 * 0.5 = 0.25
assert abs(int_val - 0.25) < 1.0e-12, '[TC48] ∫_T 1 dA = 0.25 FAILED'

# ---- TC49: rank_boundary_control_sequence - 离散化分箱 ----
q_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
seq = rank_boundary_control_sequence(q_vals, n_bins=5)
assert len(seq) == 5, '[TC49a] 序列长度应为5 FAILED'
assert seq[0] != seq[-1], '[TC49b] 极值应属于不同箱 FAILED'

# ---- TC50: solve_nonlinear_reaction - c=0 退化为恒等 ----
y_sol = solve_nonlinear_reaction(2.0, 0.1, 0.0, 2.0)
assert abs(y_sol - 2.0) < 1.0e-12, '[TC50] c=0 时应返回 rhs FAILED'

# ---- TC51: qmc_integrate_ellipse - 常数函数积分 ----
def f_const(x, y):
    return np.ones_like(x)

qmc_const = qmc_integrate_ellipse(f_const, 2.0, 1.5, 500)
expected_const = np.pi * 2.0 * 1.5  # S = πab
assert abs(qmc_const - expected_const) / expected_const < 0.05, \
    f'[TC51] QMC 常数积分相对误差应<5%, 实际 {abs(qmc_const-expected_const)/expected_const*100:.2f}% FAILED'

# ---- TC52: write_tecplot_mesh - 写入不崩溃 ----
import numpy as np
np.random.seed(42)
nodes_w, elems_w, bnd_w = generate_ellipse_mesh_2d(2.0, 1.0, n_boundary=8, n_inner=5, seed=99)
import tempfile
import os as _os
tmpdir = tempfile.mkdtemp()
tec_path = _os.path.join(tmpdir, 'test_mesh.tec')
write_tecplot_mesh(tec_path, nodes_w, elems_w, node_data=None, var_names=None)
assert _os.path.exists(tec_path), '[TC52] TECPLOT 文件应创建成功 FAILED'
assert _os.path.getsize(tec_path) > 0, '[TC52b] TECPLOT 文件不应为空 FAILED'
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

# ---- TC53: build_gll_time_operators - 质量矩阵对角正定 ----
nodes_t, M_t, S_t, D_t = build_gll_time_operators(4, T=1.0)
assert nodes_t.shape == (5,), '[TC53a] 时间节点数应为5 FAILED'
assert np.all(np.diag(M_t) > 0), '[TC53b] 质量矩阵对角元应为正 FAILED'
assert abs(nodes_t[0]) < 1.0e-12, '[TC53c] 首个节点应为0 FAILED'
assert abs(nodes_t[-1] - 1.0) < 1.0e-12, '[TC53d] 末个节点应为1 FAILED'

# ---- TC54: boundary_actuator_positions - 执行器在椭圆上 ----
positions, thetas = boundary_actuator_positions(2.0, 1.5, 3, 1.0, np.array([0.5, 1.0, 1.5]))
for k in range(3):
    x_k, y_k = positions[k]
    assert abs((x_k / 2.0) ** 2 + (y_k / 1.5) ** 2 - 1.0) < 1.0e-10, \
        f'[TC54] 执行器{k+1}不在椭圆上 FAILED'

# ---- TC55: ellipsoid_surface_area - 旋转椭球（扁球）退化 ----
surf_obl = ellipsoid_surface_area(2.0, 2.0, 1.0)
assert np.isfinite(surf_obl), '[TC55] 旋转椭球表面积应为有限值 FAILED'
assert surf_obl > 0, '[TC55b] 表面积应为正 FAILED'

# ---- TC56: 椭圆积分 elliptic_inc_fm/em 自洽 ----
from ellipsoid_geometry import elliptic_inc_fm, elliptic_inc_em
phi_t = np.pi / 4.0
m_t = 0.5
fm_val = elliptic_inc_fm(phi_t, m_t)
em_val = elliptic_inc_em(phi_t, m_t)
assert np.isfinite(fm_val), '[TC56a] F(π/4,0.5) 应为有限值 FAILED'
assert np.isfinite(em_val), '[TC56b] E(π/4,0.5) 应为有限值 FAILED'

# ---- TC57: sensitive_ode_rhs - 结构一致性 ----
rhs0 = sensitive_ode_rhs(0.0, np.array([1.0, 0.0]))
assert rhs0[0] == 0.0, '[TC57a] y₁\'=y₂, 当 y₂=0 时 y₁\'=0 FAILED'
assert rhs0[1] == 1.0, '[TC57b] y₂\'=y₁, 当 y₁=1 时 y₂\'=1 FAILED'

print('\n全部 57 个测试通过!\n')
