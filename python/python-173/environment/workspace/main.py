
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
    problem = {}


    problem['domain_bounds'] = ((0.0, 1.0), (0.0, 1.0))


    def D_func(x, y):
        return 0.01 + 0.02 * (x ** 2 + y ** 2)
    problem['D_func'] = D_func


    def c_func(x, y):
        return 0.5
    problem['c_func'] = c_func


    def v_func(x, y):
        v_x = 0.5 * (y - 0.5)
        v_y = -0.5 * (x - 0.5)
        return v_x, v_y
    problem['v_func'] = v_func


    alpha_reaction = 5.0
    def R_func(u, x, y):
        return alpha_reaction * u * (1.0 - u)
    problem['R_func'] = R_func


    def f_func(x, y):
        return 2.0 * np.exp(-50.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    problem['f_func'] = f_func


    def u0_func(x, y):
        return np.exp(-100.0 * ((x - 0.3) ** 2 + (y - 0.3) ** 2))
    problem['u0_func'] = u0_func


    def g_dirichlet(x, y, t=0.0):
        return 0.0
    problem['g_dirichlet'] = g_dirichlet


    problem['t_end'] = 0.1
    problem['dt_init'] = 0.001


    problem['max_refine_level'] = 2
    problem['refine_threshold'] = 0.5
    problem['target_error'] = 0.05
    problem['n_cells_init'] = 80

    return problem


def initialize_mesh(problem):
    domain_bounds = problem['domain_bounds']
    n_cells = problem['n_cells_init']


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


    triangles, nodes = compute_delaunay_triangulation(generators)
    print(f"[AMR] 初始网格: {len(nodes)} 节点, {len(triangles)} 三角形")

    return nodes, triangles


def prepare_boundary_conditions(nodes, triangles, problem, t=0.0):
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
    info = {}


    if len(u_current) != len(nodes):



        pass


    print(f"[AMR] 层级 {refine_level}: 时间推进求解...")
    D_func = problem['D_func']
    c_func = problem['c_func']
    v_func = problem['v_func']
    R_func = problem['R_func']
    f_func = problem['f_func']


    dt_max, h_min, pe_max = compute_cfl_condition(nodes, triangles, v_func, D_func)
    dt = min(problem['dt_init'], dt_max)
    print(f"[AMR]   CFL: dt={dt:.6f}, h_min={h_min:.6f}, Pe_max={pe_max:.2f}")


    M_lumped = compute_mass_matrix_lumped(nodes, triangles)


    n_steps = max(1, int(problem['t_end'] / dt))
    dt = problem['t_end'] / n_steps
    u_solution = u_current.copy()




    for step in range(n_steps):
        u_solution = advection_diffusion_reaction_step(
            nodes, triangles,
            u_solution, dt,
            D_func, c_func, v_func, R_func, f_func,
            dirichlet_nodes, dirichlet_values,
            M_lumped, scheme='FIXME'
        )

    info['dt'] = dt
    info['n_steps'] = n_steps


    print(f"[AMR]   计算后验误差指示子...")
    errors_cheb, total_error_cheb = compute_all_error_indicators(
        nodes, triangles, u_solution, cheb_degree=4
    )
    errors_zz, total_error_zz = compute_gradient_recovery_error(
        nodes, triangles, u_solution
    )


    errors = 0.5 * errors_cheb + 0.5 * errors_zz
    total_error = np.sqrt(total_error_cheb ** 2 + total_error_zz ** 2)

    print(f"[AMR]   Chebyshev 误差: {total_error_cheb:.6e}")
    print(f"[AMR]   ZZ 恢复误差: {total_error_zz:.6e}")
    print(f"[AMR]   综合误差: {total_error:.6e}")

    info['errors'] = errors
    info['total_error'] = total_error


    quality = compute_mesh_quality(nodes, triangles)
    print(f"[AMR]   网格质量: min={quality.min():.4f}, mean={quality.mean():.4f}")
    info['quality_min'] = quality.min()
    info['quality_mean'] = quality.mean()


    adj_list, adj_row, adj = build_mesh_adjacency(len(nodes), triangles)
    connected = graph_is_connected(adj_list)
    print(f"[AMR]   网格连通性: {'连通' if connected else '不连通'}")


    auth, hub = hits_ranking(len(nodes), adj_row, adj, max_iter=30)
    print(f"[AMR]   HITS authority 范围: [{auth.min():.4f}, {auth.max():.4f}]")
    info['hits_auth_max'] = auth.max()


    if refine_level >= problem['max_refine_level']:
        print(f"[AMR]   达到最大细化层级 {problem['max_refine_level']}，停止细化")
        return nodes, triangles, u_solution, errors, total_error, info

    if total_error < problem['target_error']:
        print(f"[AMR]   误差 {total_error:.6e} < 目标 {problem['target_error']:.6e}，停止细化")
        return nodes, triangles, u_solution, errors, total_error, info


    print(f"[AMR]   标记并细化高误差单元...")
    new_nodes, new_triangles, parent_map, node_level, refined_count = refine_marked_elements(
        nodes, triangles, errors, threshold_ratio=problem['refine_threshold']
    )

    print(f"[AMR]   细化 {refined_count} / {len(triangles)} 个单元")
    print(f"[AMR]   新网格: {len(new_nodes)} 节点, {len(new_triangles)} 三角形")


    u_transferred = solution_transfer_between_meshes(
        nodes, triangles, u_solution,
        new_nodes, new_triangles, transfer_type='shepard'
    )


    adj_list_new, adj_row_new, adj_new = build_mesh_adjacency(len(new_nodes), new_triangles)
    rcm_nodes, rcm_triangles, perm, bw_before, bw_after = apply_rcm_to_mesh(
        new_nodes, new_triangles, adj_row_new, adj_new
    )
    print(f"[AMR]   RCM 带宽: {bw_before} -> {bw_after}")


    u_rcm = u_transferred[perm]


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
    print_banner()
    start_time = time.time()


    problem = setup_problem()
    print("[MAIN] 问题设置完成")
    print(f"[MAIN] 物理参数:")
    print(f"       扩散系数: D(x,y) = 0.01 + 0.02*(x^2+y^2)")
    print(f"       对流速度: 旋转流场")
    print(f"       反应项: Fisher-KPP, α = 5.0")
    print(f"       时间终值: T = {problem['t_end']}")
    print(f"       AMR 最大层级: {problem['max_refine_level']}")
    print(f"       目标误差: {problem['target_error']}")


    nodes, triangles = initialize_mesh(problem)


    dirichlet_nodes, dirichlet_values, neumann_edges = prepare_boundary_conditions(
        nodes, triangles, problem
    )
    print(f"[MAIN] Dirichlet 边界节点: {len(dirichlet_nodes)} 个")


    u_current = np.array([
        problem['u0_func'](nodes[i, 0], nodes[i, 1])
        for i in range(len(nodes))
    ])
    print(f"[MAIN] 初始解 L2 范数: {np.linalg.norm(u_current):.6f}")


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


        dirichlet_nodes, dirichlet_values, neumann_edges = prepare_boundary_conditions(
            nodes, triangles, problem
        )


        if total_error < problem['target_error'] and level > 0:
            print(f"\n[MAIN] 达到目标精度，提前终止 AMR")
            break


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


    M_lumped = compute_mass_matrix_lumped(nodes, triangles)
    eigenvalues, stiffness_ratio, spectral_radius = analyze_stiffness_eigenvalues(
        A_stiff, M_lumped, n_eig=10
    )
    print(f"[MAIN] 系统刚度分析:")
    print(f"       刚度比: {stiffness_ratio:.2e}")
    print(f"       谱半径: {spectral_radius:.4f}")
    print(f"       前10个特征值实部: {np.real(eigenvalues[:10]).tolist()}")


    def boundary_param(t):
        return 0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)

    hex_points, boundary_pts = approximate_boundary_with_hexagons(
        boundary_param,
        problem['domain_bounds'],
        hex_size=0.08,
        n_samples=200
    )
    print(f"[MAIN] 六边形边界近似: {len(hex_points)} 个边界格点")



    from shepard_transfer import shepard_interp_nd
    boundary_solutions = shepard_interp_nd(
        2, nodes, u_steady, p=2.0, xi=hex_points
    )
    hex_indicators = hex_boundary_refinement_indicator(
        hex_points, boundary_solutions, hex_size=0.08
    )
    print(f"[MAIN] 边界细化指示子: max={hex_indicators.max():.4f}")


    print(f"\n{'='*60}")
    print("[MAIN] 结果汇总")
    print(f"{'='*60}")
    print(f"最终网格规模: {len(nodes)} 节点, {len(triangles)} 三角形")
    print(f"AMR 误差历史: {[float(e) for e in all_errors]}")
    print(f"网格规模历史 (节点): {all_node_counts}")
    print(f"网格规模历史 (单元): {all_tri_counts}")
    print(f"最终瞬态解范围: [{u_current.min():.6f}, {u_current.max():.6f}]")
    print(f"最终稳态解范围: [{u_steady.min():.6f}, {u_steady.max():.6f}]")


    energy_norm = np.sqrt(u_steady @ A_stiff @ u_steady)
    print(f"稳态解能量范数: {energy_norm:.6f}")


    print(f"\n[MAIN] 数值验证:")

    mass = np.sum(M_lumped * u_steady)
    print(f"       稳态总质量: {mass:.6f}")


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
