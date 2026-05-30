
import numpy as np
import time
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dna_topology import (
    generate_nucleosome_tet_mesh,
    compute_dsb_repair_compartment_volume,
    TetMesh,
)
from brownian_dynamics import (
    simulate_ku80_search_time,
    henon_crowding_map,
    normal_distribution_ode_solution,
    OverdampedLangevinIntegrator,
    dna_repair_protein_force,
)
from reaction_diffusion import (
    simulate_gamma_h2ax_wave,
    simulate_parp1_nonlinear_diffusion,
    porous_medium_residual,
    gray_scott_step,
)
from energy_landscape import (
    build_free_energy_surface,
    pwl_interp_2d,
    simplex_vertex_coordinates,
    sammon_mapping,
)
from sparse_solver import (
    solve_nonlinear_pb,
    BandedLU,
    assemble_poisson_boltzmann_jacobian,
)
from matrix_operations import (
    matrix_chain_optimal_order,
    assemble_enm_stiffness_matrix,
    write_matrix_market,
    read_matrix_market,
    catalan_number,
    build_optimal_parenthesization,
    compute_optimal_tensor_contraction_cost,
)
from io_parser import (
    parse_tec_file,
    build_tec_file,
    adaptive_mesh_refinement_2d,
    grid_double_resolution,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_dna_topology_module() -> dict:
    print_section("模块 1: 染色质三维拓扑离散化 (tet_mesh_quad + tet_mesh_to_xml)")


    mesh = generate_nucleosome_tet_mesh(
        n_rings=4, n_theta=8, n_z=4,
        major_radius=5.5, minor_radius=3.3, pitch=2.7
    )
    print(f"  生成网格: {mesh.n_nodes} 个节点, {mesh.n_elements} 个单元")


    vol_sum, _ = mesh.integrate_nodal_values(np.ones(mesh.n_nodes))
    print(f"  网格总体积: {vol_sum:.4f} nm^3")


    surface_area = mesh.compute_surface_area()
    print(f"  边界表面积: {surface_area:.4f} nm^2")


    np.random.seed(1)
    gamma_density = np.exp(-np.sum(mesh.nodes ** 2, axis=1) / (2.0 * 50.0 ** 2))
    gamma_density = gamma_density / np.max(gamma_density)
    vol_repair, total_signal = compute_dsb_repair_compartment_volume(
        mesh, gamma_density, threshold=0.15
    )
    print(f"  DSB 修复腔室体积 (threshold=0.15): {vol_repair:.4f} nm^3")
    print(f"  总 γH2AX 信号量: {total_signal:.4f}")


    xml_str = mesh.to_xml_string()
    print(f"  XML 网格字符串长度: {len(xml_str)} 字符")

    return {
        "mesh": mesh,
        "volume": vol_sum,
        "surface_area": surface_area,
        "repair_volume": vol_repair,
    }


def run_brownian_dynamics_module() -> dict:
    print_section("模块 2: 修复蛋白布朗动力学 (normal_ode + henon_orbit)")


    result = simulate_ku80_search_time(n_proteins=20, n_steps=2000)
    print(f"  模拟蛋白数: 20")
    print(f"  平均首次通过时间 (MFPT): {result['mfpt_us']:.2f} μs")
    print(f"  结合分数: {result['binding_fraction']:.3f}")
    print(f"  最终 MSD: {result['msd_final_nm2']:.2f} nm^2")


    x = np.linspace(-0.9, 0.9, 100)
    y = np.linspace(-0.9, 0.9, 100)
    X, Y = np.meshgrid(x, y)
    X_map, Y_map = henon_crowding_map(X.ravel(), Y.ravel(), c=0.98, n_iter=5)
    print(f"  Henon 映射后均值位移: {np.mean(np.abs(X_map - X.ravel())):.6f}")


    t = np.linspace(-3.0, 3.0, 100)
    y_gauss = normal_distribution_ode_solution(t, sigma0=1.0)
    print(f"  高斯分布积分校验: {np.trapezoid(y_gauss, t):.6f} (理论值: 1.0)")

    return result


def run_reaction_diffusion_module() -> dict:
    print_section("模块 3: γH2AX 信号波与 PARP1 扩散 (gray_scott + porous_medium)")



    gs_result = simulate_gamma_h2ax_wave(
        nx=128, ny=128, nt=8000,
        f=0.03, k=0.062, du=16.0, dv=8.0,
        dt=0.1, dx=10.0,
    )
    print(f"  Gray-Scott 波前速度: {gs_result['wave_velocity']:.4f} nm/步")
    print(f"  总 γH2AX 量: {gs_result['total_gamma_h2ax']:.2f}")
    print(f"  最大 v 浓度: {gs_result['max_v']:.4f}")


    pm_result = simulate_parp1_nonlinear_diffusion(
        nx=256, nt=400, dt=0.01, dx=0.1, m=3.0,
    )
    print(f"  多孔介质数值-精确解 L2 误差: {pm_result['l2_error']:.6e}")
    print(f"  数值解总质量: {pm_result['total_mass_numerical']:.6f}")
    print(f"  精确解总质量: {pm_result['total_mass_exact']:.6f}")


    x_test = np.linspace(-5.0, 5.0, 50)
    res = porous_medium_residual(x_test, t=1.0, m=3.0)
    print(f"  Barenblatt 解 PDE 残差范数: {np.linalg.norm(res):.6e}")

    return {
        "gray_scott": gs_result,
        "porous_medium": pm_result,
    }


def run_energy_landscape_module() -> dict:
    print_section("模块 4: 修复蛋白构象能量景观 (sammon_data + pwl_interp_2d)")


    np.random.seed(99)
    n_frames = 300
    n_dihedrals = 8

    state_a = np.random.randn(n_dihedrals) * 0.3
    state_b = np.random.randn(n_dihedrals) * 0.3 + np.array([2.0, -1.5, 1.0, 0.5, -0.8, 1.2, -0.3, 0.7])

    dihedrals = np.zeros((n_frames, n_dihedrals))
    for i in range(n_frames):
        if i < n_frames // 2:
            dihedrals[i, :] = state_a + np.random.randn(n_dihedrals) * 0.2
        else:
            dihedrals[i, :] = state_b + np.random.randn(n_dihedrals) * 0.2








    fel_result = None

    return fel_result


def run_sparse_solver_module() -> dict:
    print_section("模块 5: Poisson-Boltzmann 电静力学 (plasma_matrix + r8gb)")








    pb_result = None




    residual = None

    return pb_result


def run_matrix_operations_module() -> dict:
    print_section("模块 6: ENM 刚度矩阵与矩阵链优化 (matrix_assemble + matrix_chain + mm_to_msm)")


    dims = [30, 35, 15, 5, 10, 20, 25]
    cost, s = matrix_chain_optimal_order(dims)
    opt_expr = build_optimal_parenthesization(s, 0, len(dims) - 2)
    cat_num = catalan_number(len(dims) - 2)
    print(f"  矩阵链维度: {dims}")
    print(f"  最优标量乘法代价: {cost}")
    print(f"  最优加括号方式: {opt_expr}")
    print(f"  总方案数 (Catalan): {cat_num}")


    np.random.seed(3)
    n_nodes = 20
    coords = np.random.randn(n_nodes, 3) * 10.0
    H = assemble_enm_stiffness_matrix(coords, cutoff=20.0, spring_constant=1.0)
    print(f"  ENM 刚度矩阵尺寸: {H.shape}")
    print(f"  刚度矩阵条件数估计: {np.linalg.cond(H):.4e}")


    eigvals = np.linalg.eigvalsh(H)
    n_zero = np.sum(np.abs(eigvals) < 1e-8)
    print(f"  零特征值个数 (刚性运动模式): {n_zero}")


    temp_file = "/tmp/test_enm_matrix.mtx"
    write_matrix_market(temp_file, H, rep="coordinate", symm="symmetric")
    H_read, rows, cols, entries, rep, field, symm = read_matrix_market(temp_file)
    print(f"  Matrix Market I/O: {rows}x{cols}, 非零元={entries}, 格式={rep}")
    diff = np.max(np.abs(H - H_read))
    print(f"  读写一致性误差: {diff:.6e}")


    tensor_cost = compute_optimal_tensor_contraction_cost(
        [(30, 35), (35, 15), (15, 5), (5, 10), (10, 20), (20, 25)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    )
    print(f"  张量缩并最优代价估计: {tensor_cost}")

    return {
        "matrix_chain_cost": cost,
        "stiffness_matrix": H,
    }


def run_io_parser_module() -> dict:
    print_section("模块 7: TEC 数据解析与自适应网格加密 (tec_io + image_double)")


    np.random.seed(5)
    node_coord = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float64).T

    element_node = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=np.int64).T

    node_data = np.array([
        [1.0, 0.8, 0.9, 1.1, 0.85],
    ], dtype=np.float64)

    tec_content = build_tec_file(
        node_coord, element_node, node_data,
        variable_names=['X', 'Y', 'Z', 'ElectronDensity']
    )
    print(f"  生成 TEC 文件内容长度: {len(tec_content)} 字符")


    parsed = parse_tec_file(tec_content)
    print(f"  解析结果: {parsed['node_num']} 节点, {parsed['element_num']} 单元")
    print(f"  空间维度: {parsed['dim_num']}, 变量数: {len(parsed['variable_names'])}")


    np.random.seed(8)
    field = np.zeros((32, 32))

    for i in range(32):
        for j in range(32):
            r = np.sqrt((i - 16) ** 2 + (j - 16) ** 2)
            field[i, j] = np.exp(-r ** 2 / (2.0 * 4.0 ** 2))

    amr_levels = adaptive_mesh_refinement_2d(
        field, gradient_threshold=0.02, max_level=2
    )
    print(f"  AMR 生成层级数: {len(amr_levels)}")
    for lvl, arr in enumerate(amr_levels):
        print(f"    层级 {lvl}: 形状 {arr.shape}, 总和 {np.sum(arr):.6f}")


    doubled = grid_double_resolution(field, mode="2d")
    print(f"  网格加倍后形状: {doubled.shape}")
    print(f"  加倍前后积分守恒误差: {abs(np.sum(field) - np.sum(doubled)):.6e}")

    return parsed


def main():
    print("\n" + "#" * 70)
    print("#  DNA 损伤修复分子动力学综合模拟平台")
    print("#  科学领域: 分子动力学 — DNA损伤修复分子机制")
    print("#  编程语言: Python 3")
    print("#" * 70)

    start_time = time.time()


    results_topology = run_dna_topology_module()
    results_brownian = run_brownian_dynamics_module()
    results_rd = run_reaction_diffusion_module()
    results_energy = run_energy_landscape_module()
    results_sparse = run_sparse_solver_module()
    results_matrix = run_matrix_operations_module()
    results_io = run_io_parser_module()

    elapsed = time.time() - start_time

    print("\n" + "#" * 70)
    print("#  模拟完成总结")
    print("#" * 70)
    print(f"  总运行时间: {elapsed:.3f} 秒")
    print(f"  染色质网格体积: {results_topology['volume']:.4f} nm^3")
    print(f"  KU80 平均搜索时间: {results_brownian['mfpt_us']:.2f} μs")
    print(f"  γH2AX 波前速度: {results_rd['gray_scott']['wave_velocity']:.4f} nm/步")
    print(f"  PARP1 扩散 L2 误差: {results_rd['porous_medium']['l2_error']:.6e}")
    print(f"  能垒高度: {results_energy['barrier_height_ev']:.4f} eV")
    print(f"  PB 方程残差: {results_sparse['residual_norm']:.6e}")
    print(f"  矩阵链最优代价: {results_matrix['matrix_chain_cost']}")
    print("#" * 70)
    print("  所有模块执行完毕，无报错。")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
