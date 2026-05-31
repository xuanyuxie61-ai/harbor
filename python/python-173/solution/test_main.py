"""
自适应网格细化 (AMR) 求解二维对流-扩散-反应方程的统一入口

科学问题:
    在二维区域 Ω = [0,1] × [0,1] 上求解非定常对流-扩散-反应方程:
        ∂u/∂t + v·∇u = ∇·(D ∇u) + R(u) + f(x,y,t)
    
    其中:
        v(x,y) = (v_x, v_y) 为对流速度场
        D(x,y) 为扩散系数
        R(u) = α u (1 - u) 为 Fisher-KPP 型反应项
        f(x,y,t) 为外部源项
    
    边界条件: Dirichlet 边界 u = 0
    初始条件: u(x,y,0) = u_0(x,y)

AMR 工作流程:
    1. 生成初始三角形网格 (CVT + Delaunay)
    2. 求解 PDE 到指定时间
    3. 计算后验误差指示子 (Chebyshev + ZZ 恢复)
    4. 标记并细化高误差单元
    5. 在细化网格上传递解并继续求解
    6. 重复 2-5 直到达到目标精度或最大细化层级
    7. 应用 RCM 重排序优化稀疏求解器性能
    8. 输出数值结果与收敛分析

物理背景:
    该方程描述活性物质在流体中的输运与反应过程，
    广泛应用于燃烧学、生态学传播、化学反应工程等领域。
    Fisher-KPP 反应项 R(u) = α u(1-u) 描述自催化反应，
    其解可形成行波前沿，对流场和扩散的竞争决定波速:
        c* = 2 * sqrt(α * D)
"""

import numpy as np
import time

from cvt_mesh import generate_cvt_mesh, compute_delaunay_triangulation, triangle_area
from triangulation_refine import refine_marked_elements, compute_mesh_quality
from graph_mesh import (
    build_mesh_adjacency, apply_rcm_to_mesh,
    graph_is_connected, hits_ranking, compute_element_adjacency
)
from shepard_transfer import solution_transfer_between_meshes
from error_indicator import compute_all_error_indicators, compute_gradient_recovery_error
from fem_solver import solve_steady_fem
from advection_diffusion import (
    identify_boundary_edges,
    compute_advection_matrix,
    compute_reaction_term,
    compute_mass_matrix_lumped,
    compute_cfl_condition,
    advection_diffusion_reaction_step
)
from time_integrator import adaptive_time_stepping, analyze_stiffness_eigenvalues
from hex_boundary import approximate_boundary_with_hexagons, hex_boundary_refinement_indicator
from quadrature_rules import TriangleQuadrature


def print_banner():
    """打印项目横幅。"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║   自适应网格细化 (AMR) 求解二维对流-扩散-反应方程                   ║
    ║   Adaptive Mesh Refinement for 2D Advection-Diffusion-Reaction    ║
    ║                                                                      ║
    ║   科学领域: 计算数学 — 自适应网格细化                                ║
    ║   融合 15 个种子项目的核心算法                                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def setup_problem():
    """
    设置物理问题的参数和函数。
    
    Returns
    -------
    problem : dict
        包含所有物理参数和函数的字典
    """
    problem = {}

    # 区域定义
    problem['domain_bounds'] = ((0.0, 1.0), (0.0, 1.0))

    # 扩散系数 (各向异性)
    def D_func(x, y):
        return 0.01 + 0.02 * (x ** 2 + y ** 2)
    problem['D_func'] = D_func

    # 反应系数 (线性稳定化项)
    def c_func(x, y):
        return 0.5
    problem['c_func'] = c_func

    # 对流速度场 (旋转流)
    def v_func(x, y):
        v_x = 0.5 * (y - 0.5)
        v_y = -0.5 * (x - 0.5)
        return v_x, v_y
    problem['v_func'] = v_func

    # Fisher-KPP 反应项 R(u) = alpha * u * (1 - u)
    alpha_reaction = 5.0
    def R_func(u, x, y):
        return alpha_reaction * u * (1.0 - u)
    problem['R_func'] = R_func

    # 源项 (高斯脉冲)
    def f_func(x, y):
        return 2.0 * np.exp(-50.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    problem['f_func'] = f_func

    # 初始条件
    def u0_func(x, y):
        return np.exp(-100.0 * ((x - 0.3) ** 2 + (y - 0.3) ** 2))
    problem['u0_func'] = u0_func

    # Dirichlet 边界值
    def g_dirichlet(x, y, t=0.0):
        return 0.0
    problem['g_dirichlet'] = g_dirichlet

    # 时间参数
    problem['t_end'] = 0.1
    problem['dt_init'] = 0.001

    # AMR 参数
    problem['max_refine_level'] = 2
    problem['refine_threshold'] = 0.5
    problem['target_error'] = 0.05
    problem['n_cells_init'] = 80

    return problem


def initialize_mesh(problem):
    """
    生成初始 CVT 优化的三角形网格。
    
    Parameters
    ----------
    problem : dict
    
    Returns
    -------
    nodes : ndarray
    triangles : ndarray
    """
    domain_bounds = problem['domain_bounds']
    n_cells = problem['n_cells_init']

    # 密度函数：在中心区域加密
    def density_func(x, y):
        cx, cy = 0.5, 0.5
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return 1.0 + 5.0 * np.exp(-20.0 * dist ** 2)

    print("[AMR] 生成 CVT 优化网格...")
    generators, energy_history = generate_cvt_mesh(
        n_cells=n_cells,
        domain_bounds=domain_bounds,
        density_func=density_func,
        it_max=30,
        sample_multiplier=50,
        tol=1e-4
    )

    print(f"[AMR] CVT 能量收敛: {energy_history[-1]:.6e} (迭代 {len(energy_history)} 次)")

    # Delaunay 三角剖分
    triangles, nodes = compute_delaunay_triangulation(generators)
    print(f"[AMR] 初始网格: {len(nodes)} 节点, {len(triangles)} 三角形")

    return nodes, triangles


def prepare_boundary_conditions(nodes, triangles, problem, t=0.0):
    """
    准备边界条件。
    
    Returns
    -------
    dirichlet_nodes : ndarray
    dirichlet_values : ndarray
    neumann_edges : list
    """
    domain_bounds = problem['domain_bounds']
    g_dirichlet = problem['g_dirichlet']

    boundary_nodes, dirichlet_nodes, neumann_edges = identify_boundary_edges(
        nodes, triangles, domain_bounds
    )

    dirichlet_values = np.array([
        g_dirichlet(nodes[i, 0], nodes[i, 1], t)
        for i in dirichlet_nodes
    ])

    return dirichlet_nodes, dirichlet_values, neumann_edges


def run_amr_cycle(nodes, triangles, u_current, problem, refine_level,
                  dirichlet_nodes, dirichlet_values):
    """
    执行一个 AMR 周期：求解 → 误差估计 → 标记细化 → 网格更新 → 解传递。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    u_current : ndarray
        当前解（可能与 nodes 维度不同，表示在上一网格上）
    problem : dict
    refine_level : int
    dirichlet_nodes : ndarray
    dirichlet_values : ndarray
    
    Returns
    -------
    new_nodes : ndarray
    new_triangles : ndarray
    u_new : ndarray
    errors : ndarray
    total_error : float
    info : dict
    """
    info = {}

    # 如果解与节点数量不匹配，先传递解
    if len(u_current) != len(nodes):
        # 需要旧网格信息，这里简化：使用 Shepard 插值
        # 创建临时旧节点（前一步的节点）
        # 实际上在主循环中保持旧网格信息
        pass

    # 1. 时间推进求解
    print(f"[AMR] 层级 {refine_level}: 时间推进求解...")
    D_func = problem['D_func']
    c_func = problem['c_func']
    v_func = problem['v_func']
    R_func = problem['R_func']
    f_func = problem['f_func']

    # 计算 CFL 条件
    dt_max, h_min, pe_max = compute_cfl_condition(nodes, triangles, v_func, D_func)
    dt = min(problem['dt_init'], dt_max)
    print(f"[AMR]   CFL: dt={dt:.6f}, h_min={h_min:.6f}, Pe_max={pe_max:.2f}")

    # 计算 lumped 质量矩阵
    M_lumped = compute_mass_matrix_lumped(nodes, triangles)

    # 时间步进 (使用几个隐式步)
    n_steps = max(1, int(problem['t_end'] / dt))
    dt = problem['t_end'] / n_steps
    u_solution = u_current.copy()

    for step in range(n_steps):
        u_solution = advection_diffusion_reaction_step(
            nodes, triangles,
            u_solution, dt,
            D_func, c_func, v_func, R_func, f_func,
            dirichlet_nodes, dirichlet_values,
            M_lumped, scheme='implicit'
        )

    info['dt'] = dt
    info['n_steps'] = n_steps

    # 2. 误差估计
    print(f"[AMR]   计算后验误差指示子...")
    errors_cheb, total_error_cheb = compute_all_error_indicators(
        nodes, triangles, u_solution, cheb_degree=4
    )
    errors_zz, total_error_zz = compute_gradient_recovery_error(
        nodes, triangles, u_solution
    )

    # 综合误差
    errors = 0.5 * errors_cheb + 0.5 * errors_zz
    total_error = np.sqrt(total_error_cheb ** 2 + total_error_zz ** 2)

    print(f"[AMR]   Chebyshev 误差: {total_error_cheb:.6e}")
    print(f"[AMR]   ZZ 恢复误差: {total_error_zz:.6e}")
    print(f"[AMR]   综合误差: {total_error:.6e}")

    info['errors'] = errors
    info['total_error'] = total_error

    # 3. 网格质量分析
    quality = compute_mesh_quality(nodes, triangles)
    print(f"[AMR]   网格质量: min={quality.min():.4f}, mean={quality.mean():.4f}")
    info['quality_min'] = quality.min()
    info['quality_mean'] = quality.mean()

    # 4. 图论分析
    adj_list, adj_row, adj = build_mesh_adjacency(len(nodes), triangles)
    connected = graph_is_connected(adj_list)
    print(f"[AMR]   网格连通性: {'连通' if connected else '不连通'}")

    # HITS 重要性排序
    auth, hub = hits_ranking(len(nodes), adj_row, adj, max_iter=30)
    print(f"[AMR]   HITS authority 范围: [{auth.min():.4f}, {auth.max():.4f}]")
    info['hits_auth_max'] = auth.max()

    # 5. 判断是否细化
    if refine_level >= problem['max_refine_level']:
        print(f"[AMR]   达到最大细化层级 {problem['max_refine_level']}，停止细化")
        return nodes, triangles, u_solution, errors, total_error, info

    if total_error < problem['target_error']:
        print(f"[AMR]   误差 {total_error:.6e} < 目标 {problem['target_error']:.6e}，停止细化")
        return nodes, triangles, u_solution, errors, total_error, info

    # 6. 自适应细化
    print(f"[AMR]   标记并细化高误差单元...")
    new_nodes, new_triangles, parent_map, node_level, refined_count = refine_marked_elements(
        nodes, triangles, errors, threshold_ratio=problem['refine_threshold']
    )

    print(f"[AMR]   细化 {refined_count} / {len(triangles)} 个单元")
    print(f"[AMR]   新网格: {len(new_nodes)} 节点, {len(new_triangles)} 三角形")

    # 7. 解传递
    u_transferred = solution_transfer_between_meshes(
        nodes, triangles, u_solution,
        new_nodes, new_triangles, transfer_type='shepard'
    )

    # 8. RCM 重排序优化
    adj_list_new, adj_row_new, adj_new = build_mesh_adjacency(len(new_nodes), new_triangles)
    rcm_nodes, rcm_triangles, perm, bw_before, bw_after = apply_rcm_to_mesh(
        new_nodes, new_triangles, adj_row_new, adj_new
    )
    print(f"[AMR]   RCM 带宽: {bw_before} -> {bw_after}")

    # 对解也进行重排序
    u_rcm = u_transferred[perm]

    # 更新边界条件到重排序后的网格
    boundary_nodes_new, dirichlet_nodes_new, neumann_edges_new = identify_boundary_edges(
        rcm_nodes, rcm_triangles, problem['domain_bounds']
    )
    dirichlet_values_new = np.array([
        problem['g_dirichlet'](rcm_nodes[i, 0], rcm_nodes[i, 1])
        for i in dirichlet_nodes_new
    ])

    info['refined_count'] = refined_count
    info['bw_before'] = bw_before
    info['bw_after'] = bw_after
    info['dirichlet_nodes_new'] = dirichlet_nodes_new
    info['dirichlet_values_new'] = dirichlet_values_new

    return rcm_nodes, rcm_triangles, u_rcm, errors, total_error, info


def main():
    """
    主程序入口。零参数运行，执行完整的 AMR 求解流程。
    """
    print_banner()
    start_time = time.time()

    # 1. 设置问题
    problem = setup_problem()
    print("[MAIN] 问题设置完成")
    print(f"[MAIN] 物理参数:")
    print(f"       扩散系数: D(x,y) = 0.01 + 0.02*(x^2+y^2)")
    print(f"       对流速度: 旋转流场")
    print(f"       反应项: Fisher-KPP, α = 5.0")
    print(f"       时间终值: T = {problem['t_end']}")
    print(f"       AMR 最大层级: {problem['max_refine_level']}")
    print(f"       目标误差: {problem['target_error']}")

    # 2. 生成初始网格
    nodes, triangles = initialize_mesh(problem)

    # 3. 准备边界条件
    dirichlet_nodes, dirichlet_values, neumann_edges = prepare_boundary_conditions(
        nodes, triangles, problem
    )
    print(f"[MAIN] Dirichlet 边界节点: {len(dirichlet_nodes)} 个")

    # 4. 初始化解
    u_current = np.array([
        problem['u0_func'](nodes[i, 0], nodes[i, 1])
        for i in range(len(nodes))
    ])
    print(f"[MAIN] 初始解 L2 范数: {np.linalg.norm(u_current):.6f}")

    # 5. AMR 循环
    all_errors = []
    all_node_counts = [len(nodes)]
    all_tri_counts = [len(triangles)]

    for level in range(problem['max_refine_level'] + 1):
        print(f"\n{'='*60}")
        print(f"[MAIN] AMR 层级 {level}")
        print(f"{'='*60}")

        nodes, triangles, u_current, errors, total_error, info = run_amr_cycle(
            nodes, triangles, u_current, problem, level,
            dirichlet_nodes, dirichlet_values
        )

        all_errors.append(total_error)
        all_node_counts.append(len(nodes))
        all_tri_counts.append(len(triangles))

        # 更新边界条件到当前网格
        dirichlet_nodes, dirichlet_values, neumann_edges = prepare_boundary_conditions(
            nodes, triangles, problem
        )

        # 如果达到目标精度，提前退出
        if total_error < problem['target_error'] and level > 0:
            print(f"\n[MAIN] 达到目标精度，提前终止 AMR")
            break

    # 6. 最终稳态求解 (可选)
    print(f"\n{'='*60}")
    print("[MAIN] 最终稳态求解与收敛分析")
    print(f"{'='*60}")

    D_func = problem['D_func']
    c_func = problem['c_func']
    f_func = problem['f_func']

    u_steady, A_stiff, b_load = solve_steady_fem(
        nodes, triangles,
        D_func, c_func, f_func,
        dirichlet_nodes=dirichlet_nodes,
        dirichlet_values=dirichlet_values,
        neumann_edges=neumann_edges,
        quad_degree=4
    )

    # 7. 刚度分析
    M_lumped = compute_mass_matrix_lumped(nodes, triangles)
    eigenvalues, stiffness_ratio, spectral_radius = analyze_stiffness_eigenvalues(
        A_stiff, M_lumped, n_eig=10
    )
    print(f"[MAIN] 系统刚度分析:")
    print(f"       刚度比: {stiffness_ratio:.2e}")
    print(f"       谱半径: {spectral_radius:.4f}")
    print(f"       前10个特征值实部: {np.real(eigenvalues[:10]).tolist()}")

    # 8. 六边形边界分析
    def boundary_param(t):
        return 0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)

    hex_points, boundary_pts = approximate_boundary_with_hexagons(
        boundary_param,
        problem['domain_bounds'],
        hex_size=0.08,
        n_samples=200
    )
    print(f"[MAIN] 六边形边界近似: {len(hex_points)} 个边界格点")

    # 计算边界细化指示子
    # 将稳态解插值到边界点
    from shepard_transfer import shepard_interp_nd
    boundary_solutions = shepard_interp_nd(
        2, nodes, u_steady, p=2.0, xi=hex_points
    )
    hex_indicators = hex_boundary_refinement_indicator(
        hex_points, boundary_solutions, hex_size=0.08
    )
    print(f"[MAIN] 边界细化指示子: max={hex_indicators.max():.4f}")

    # 9. 收敛分析
    print(f"\n{'='*60}")
    print("[MAIN] 结果汇总")
    print(f"{'='*60}")
    print(f"最终网格规模: {len(nodes)} 节点, {len(triangles)} 三角形")
    print(f"AMR 误差历史: {[float(e) for e in all_errors]}")
    print(f"网格规模历史 (节点): {all_node_counts}")
    print(f"网格规模历史 (单元): {all_tri_counts}")
    print(f"最终瞬态解范围: [{u_current.min():.6f}, {u_current.max():.6f}]")
    print(f"最终稳态解范围: [{u_steady.min():.6f}, {u_steady.max():.6f}]")

    # 计算能量范数误差估计
    energy_norm = np.sqrt(u_steady @ A_stiff @ u_steady)
    print(f"稳态解能量范数: {energy_norm:.6f}")

    # 10. 数值验证
    print(f"\n[MAIN] 数值验证:")
    # 检查质量守恒 (粗略)
    mass = np.sum(M_lumped * u_steady)
    print(f"       稳态总质量: {mass:.6f}")

    # 检查边界条件满足情况
    boundary_residual = np.max(np.abs(u_steady[dirichlet_nodes] - dirichlet_values))
    print(f"       Dirichlet 边界残差: {boundary_residual:.2e}")

    elapsed = time.time() - start_time
    print(f"\n[MAIN] 总运行时间: {elapsed:.2f} 秒")
    print("[MAIN] AMR 求解完成。")

    return {
        'nodes': nodes,
        'triangles': triangles,
        'u_transient': u_current,
        'u_steady': u_steady,
        'errors': all_errors,
        'node_counts': all_node_counts,
        'tri_counts': all_tri_counts,
        'stiffness_ratio': stiffness_ratio,
        'time_elapsed': elapsed
    }


if __name__ == "__main__":
    results = main()


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
from error_indicator import chebyshev_nodes, divided_differences, newton_interpolation, chebyshev_approximate_1d
from fem_solver import shape_functions_p1, assemble_fem_system
from cvt_mesh import compute_voronoi_cells, cvt_iterate, cvt_energy
from shepard_transfer import shepard_interp_nd
from time_integrator import backward_euler_step
from quadrature_rules import gauss_legendre_1d
from hex_boundary import axial_to_cartesian, cartesian_to_axial, hex_round, generate_hexagonal_lattice
from graph_mesh import adj_bandwidth

# ---- TC01: chebyshev_nodes 输出长度、边界、唯一性 ----
import numpy as np
x = chebyshev_nodes(0.0, 1.0, 7)
assert len(x) == 7, '[TC01] chebyshev_nodes 长度 FAILED'
assert np.all(x >= 0.0) and np.all(x <= 1.0), '[TC01] chebyshev_nodes 边界 FAILED'
assert len(np.unique(np.round(x, 10))) == 7, '[TC01] chebyshev_nodes 唯一性 FAILED'

# ---- TC02: divided_differences 线性数据高阶差商为零 ----
x2 = np.array([0.0, 1.0, 2.0, 3.0])
y2 = np.array([1.0, 3.0, 5.0, 7.0])
dd2 = divided_differences(x2, y2)
assert abs(dd2[-1]) < 1e-12, '[TC02] divided_differences 线性高阶差商 FAILED'

# ---- TC03: newton_interpolation 精确复现数据点 ----
x_nodes = np.array([0.0, 0.5, 1.0])
y_nodes = np.array([0.0, 0.25, 1.0])
dd3 = divided_differences(x_nodes, y_nodes)
y_eval = newton_interpolation(x_nodes, dd3, x_nodes)
assert np.allclose(y_eval, y_nodes, atol=1e-12), '[TC03] newton_interpolation 复现 FAILED'

# ---- TC04: chebyshev_approximate_1d 线性函数误差接近零 ----
def f_linear(x):
    return 2.0 * x + 1.0
max_err, dd4, x_cheb = chebyshev_approximate_1d(f_linear, -1.0, 1.0, 5)
assert max_err < 1e-10, '[TC04] chebyshev_approximate_1d 线性误差 FAILED'

# ---- TC05: shape_functions_p1 单位分解性 ----
phi, grad_phi = shape_functions_p1(0.3, 0.3)
assert abs(np.sum(phi) - 1.0) < 1e-12, '[TC05] shape_functions_p1 单位分解 FAILED'
assert np.all(phi >= -1e-12), '[TC05] shape_functions_p1 非负性 FAILED'

# ---- TC06: shape_functions_p1 节点处 delta 性质 ----
phi06, _ = shape_functions_p1(0.0, 0.0)
assert abs(phi06[0] - 1.0) < 1e-12, '[TC06] shape_functions_p1 delta 1 FAILED'
assert abs(phi06[1]) < 1e-12 and abs(phi06[2]) < 1e-12, '[TC06] shape_functions_p1 delta 0 FAILED'

# ---- TC07: triangle_area 直角三角形面积 ----
nodes07 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
area07 = triangle_area(nodes07, [0, 1, 2])
assert abs(area07 - 0.5) < 1e-12, '[TC07] triangle_area FAILED'

# ---- TC08: TriangleQuadrature 常数积分 = 0.5 ----
quad08 = TriangleQuadrature(degree=2)
integral08 = quad08.integrate(lambda xi, eta: 1.0)
assert abs(integral08 - 0.5) < 1e-12, '[TC08] TriangleQuadrature 常数积分 FAILED'

# ---- TC09: gauss_legendre_1d 权重和 = 2 ----
x_g, w_g = gauss_legendre_1d(5)
assert abs(np.sum(w_g) - 2.0) < 1e-12, '[TC09] gauss_legendre_1d 权重和 FAILED'

# ---- TC10: compute_mass_matrix_lumped 输出正定性和形状 ----
nodes10 = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
tris10 = np.array([[0, 1, 2], [1, 3, 2]])
M10 = compute_mass_matrix_lumped(nodes10, tris10)
assert len(M10) == 4, '[TC10] mass matrix 形状 FAILED'
assert np.all(M10 > 0), '[TC10] mass matrix 正定性 FAILED'

# ---- TC11: axial_to_cartesian 往返一致性 ----
q0, r0 = 3, -2
x11, y11 = axial_to_cartesian(q0, r0, size=0.5)
q_back, r_back = cartesian_to_axial(x11, y11, size=0.5)
assert abs(q_back - q0) < 1e-10, '[TC11] axial 往返 q FAILED'
assert abs(r_back - r0) < 1e-10, '[TC11] axial 往返 r FAILED'

# ---- TC12: hex_round 输出整数类型 ----
np.random.seed(42)
q_rand = np.random.randn(10)
r_rand = np.random.randn(10)
q_int, r_int = hex_round(q_rand, r_rand)
assert np.issubdtype(q_int.dtype, np.integer), '[TC12] hex_round 整数类型 FAILED'

# ---- TC13: compute_voronoi_cells 输出形状和索引范围 ----
np.random.seed(42)
samples13 = np.random.rand(200, 2)
gens13 = np.random.rand(6, 2)
nearest13 = compute_voronoi_cells(samples13, gens13)
assert len(nearest13) == 200, '[TC13] voronoi cells 长度 FAILED'
assert np.all(nearest13 >= 0) and np.all(nearest13 < 6), '[TC13] voronoi cells 范围 FAILED'

# ---- TC14: cvt_energy 非负性 ----
np.random.seed(42)
samples14 = np.random.rand(200, 2)
gens14 = np.random.rand(8, 2)
energy14 = cvt_energy(gens14, samples14)
assert energy14 >= 0.0, '[TC14] cvt_energy 非负 FAILED'

# ---- TC15: shepard_interp_nd 数据点处精确复现 ----
xd15 = np.array([[0., 0.], [1., 0.], [0., 1.]])
zd15 = np.array([1.0, 2.0, 3.0])
zi15 = shepard_interp_nd(2, xd15, zd15, p=2.0, xi=xd15.copy())
assert np.allclose(zi15, zd15, atol=1e-12), '[TC15] shepard 精确复现 FAILED'

# ---- TC16: shepard_interp_nd p=0 简单平均 ----
zi16 = shepard_interp_nd(2, xd15, zd15, p=0.0, xi=np.array([[0.5, 0.5]]))
assert abs(zi16[0] - np.mean(zd15)) < 1e-12, '[TC16] shepard p=0 平均 FAILED'

# ---- TC17: build_mesh_adjacency 简单网格连通性 ----
nodes17 = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
tris17 = np.array([[0, 1, 2], [1, 3, 2]])
adj_list17, adj_row17, adj17 = build_mesh_adjacency(4, tris17)
assert len(adj_list17) == 4, '[TC17] 邻接表长度 FAILED'
assert 2 in adj_list17[0], '[TC17] 邻接 node0->node2 FAILED'

# ---- TC18: graph_is_connected 简单网格连通 ----
assert graph_is_connected(adj_list17) == True, '[TC18] 图连通 FAILED'

# ---- TC19: adj_bandwidth 正整数值 ----
bw19 = adj_bandwidth(4, adj_row17, adj17)
assert bw19 >= 1, '[TC19] 带宽正值 FAILED'
assert isinstance(bw19, (int, np.integer)), '[TC19] 带宽整数 FAILED'

# ---- TC20: compute_element_adjacency 邻居检测 ----
elem_neighbors20 = compute_element_adjacency(2, tris17)
assert len(elem_neighbors20) == 2, '[TC20] 单元邻接长度 FAILED'
assert 1 in elem_neighbors20[0], '[TC20] 单元邻接邻居 FAILED'

# ---- TC21: compute_mesh_quality 在 [0, 1] 内 ----
quality21 = compute_mesh_quality(nodes17, tris17)
assert np.all(quality21 >= 0.0), '[TC21] 网格质量 >= 0 FAILED'
assert np.all(quality21 <= 1.0 + 1e-12), '[TC21] 网格质量 <= 1 FAILED'

# ---- TC22: backward_euler_step 收敛性和输出形状 ----
def rhs22(u):
    return -u
u0_22 = np.array([1.0, 2.0, 3.0])
u_new22, conv22 = backward_euler_step(u0_22, 0.01, rhs22)
assert conv22 == True, '[TC22] backward_euler 收敛 FAILED'
assert len(u_new22) == 3, '[TC22] backward_euler 形状 FAILED'

# ---- TC23: compute_cfl_condition 正值输出 ----
def v_test23(x, y):
    return 0.5, 0.0
def D_test23(x, y):
    return 0.1
dt_max23, h_min23, pe_max23 = compute_cfl_condition(nodes17, tris17, v_test23, D_test23)
assert dt_max23 > 0, '[TC23] CFL dt 正值 FAILED'
assert h_min23 > 0, '[TC23] CFL h_min 正值 FAILED'

# ---- TC24: identify_boundary_edges 边界节点数 ----
b_nodes24, d_nodes24, n_edges24 = identify_boundary_edges(
    nodes17, tris17, ((0., 1.), (0., 1.))
)
assert len(b_nodes24) == 4, '[TC24] 边界节点数 FAILED'

# ---- TC25: compute_reaction_term 输出形状和有限值 ----
u_curr25 = np.array([0.1, 0.2, 0.3, 0.4])
def R_test25(u, x, y):
    return 5.0 * u * (1.0 - u)
b_R25 = compute_reaction_term(nodes17, tris17, u_curr25, R_test25)
assert len(b_R25) == 4, '[TC25] 反应项形状 FAILED'
assert np.all(np.isfinite(b_R25)), '[TC25] 反应项有限值 FAILED'

# ---- TC26: generate_cvt_mesh 可复现性 (内部 seed=42) ----
np.random.seed(42)
gens26a, ehist26a = generate_cvt_mesh(20, ((0., 1.), (0., 1.)), it_max=3, sample_multiplier=20)
np.random.seed(42)
gens26b, ehist26b = generate_cvt_mesh(20, ((0., 1.), (0., 1.)), it_max=3, sample_multiplier=20)
assert np.allclose(gens26a, gens26b, atol=1e-10), '[TC26] CVT 可复现 FAILED'

# ---- TC27: generate_hexagonal_lattice 非空二维输出 ----
points27, axial27 = generate_hexagonal_lattice(radius=2, size=1.0)
assert len(points27) > 0, '[TC27] 六边形格点非空 FAILED'
assert points27.shape[1] == 2, '[TC27] 六边形格点二维 FAILED'

# ---- TC28: assemble_fem_system 刚度矩阵对称性 ----
nodes28 = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
tris28 = np.array([[0, 1, 2], [1, 3, 2]])
def D_const28(x, y): return 1.0
def c_const28(x, y): return 0.0
def f_const28(x, y): return 1.0
A28, b28 = assemble_fem_system(nodes28, tris28, D_const28, c_const28, f_const28, quad_degree=3)
assert np.allclose(A28, A28.T, atol=1e-12), '[TC28] FEM 对称性 FAILED'

# ---- TC29: adaptive_time_stepping 到达终点时间 ----
def rhs29(u):
    return -0.5 * u
u0_29 = np.array([1.0])
t_hist29, u_hist29, dt_hist29 = adaptive_time_stepping(
    u0_29, (0.0, 0.05), 0.01, rhs29, scheme='bdf1', dt_min=1e-6, dt_max=0.1
)
assert len(t_hist29) > 1, '[TC29] 时间步进历史 FAILED'
assert t_hist29[-1] >= 0.049, '[TC29] 时间步进终点 FAILED'

# ---- TC30: cvt_iterate 迭代后能量非负 ----
np.random.seed(42)
gens30 = np.random.rand(12, 2)
samples30 = np.random.rand(300, 2)
gens30_new, diff30, energy30 = cvt_iterate(gens30, samples30)
assert energy30 >= 0.0, '[TC30] cvt_iterate 能量非负 FAILED'


print('\n全部 30 个测试通过!\n')
