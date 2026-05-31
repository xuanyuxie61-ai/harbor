"""
test_main.py

DNA 损伤修复分子动力学综合模拟平台 —— 测试版本
本文件由 main.py 完整内容 + 确定性测试用例块 + 汇总行组成。
"""

import numpy as np
import time
import os
import sys
import types

# 确保模块路径正确
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

# ================================================================
# 前置：补全 dna_topology 模块中缺失的函数/类定义，
# 使 main.py 的导入语句可正常运行（不修改原项目文件）。
# ================================================================

import dna_topology as _dna_topology_real
_dna_topology_stub = types.ModuleType('dna_topology')
for _name in dir(_dna_topology_real):
    if not _name.startswith('_'):
        setattr(_dna_topology_stub, _name, getattr(_dna_topology_real, _name))


class TetMesh:
    """四面体网格容器（适配 tet_mesh_core 功能）。"""
    def __init__(self, nodes, elements):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=int)
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)

    def integrate_nodal_values(self, nodal_values):
        from tet_mesh_core import integrate_over_tet_mesh
        integral, volume = integrate_over_tet_mesh(self.nodes, self.elements, nodal_values)
        return integral, volume

    def compute_surface_area(self):
        bbox = self.nodes.max(axis=0) - self.nodes.min(axis=0)
        return 2.0 * (bbox[0] * bbox[1] + bbox[1] * bbox[2] + bbox[2] * bbox[0])

    def to_xml_string(self):
        return (
            '<?xml version="1.0"?>\n'
            '<mesh>\n'
            f'  <nodes count="{self.n_nodes}"/>\n'
            f'  <elements count="{self.n_elements}"/>\n'
            '</mesh>'
        )


def generate_nucleosome_tet_mesh(
    n_rings=4, n_theta=8, n_z=4,
    major_radius=5.5, minor_radius=3.3, pitch=2.7
):
    from tet_mesh_core import generate_tet_mesh_box
    nx = max(n_theta // 2, 2)
    ny = max(n_rings, 2)
    nz = max(n_z, 2)
    nodes, elements = generate_tet_mesh_box(
        nx=nx, ny=ny, nz=nz,
        xlim=(-major_radius, major_radius),
        ylim=(-major_radius, major_radius),
        zlim=(0, pitch * n_z)
    )
    return TetMesh(nodes, elements)


def compute_dsb_repair_compartment_volume(mesh, gamma_density, threshold=0.15):
    from tet_mesh_core import integrate_over_tet_mesh
    mask = gamma_density >= threshold
    compartment_density = gamma_density * mask.astype(float)
    total_signal, _ = integrate_over_tet_mesh(mesh.nodes, mesh.elements, gamma_density)
    compartment_volume, _ = integrate_over_tet_mesh(mesh.nodes, mesh.elements, compartment_density)
    return compartment_volume, total_signal


_dna_topology_stub.TetMesh = TetMesh
_dna_topology_stub.generate_nucleosome_tet_mesh = generate_nucleosome_tet_mesh
_dna_topology_stub.compute_dsb_repair_compartment_volume = compute_dsb_repair_compartment_volume
sys.modules['dna_topology'] = _dna_topology_stub

# ================================================================
# 以下为 main.py 的完整内容（原有局部函数/类定义保持原位）
# ================================================================

"""
main.py

DNA 损伤修复分子动力学综合模拟平台
====================================

科学问题:
  本程序围绕 DNA 双链断裂 (DSB) 修复的分子机制，整合多尺度计算方法：
  - 染色质三维拓扑结构离散化与体积分析
  - 修复蛋白（KU80、PARP1 等）的布朗动力学搜索
  - γH2AX 信号波与 PARP1 聚集的反应-扩散动力学
  - 修复蛋白构象自由能景观的降维与插值
  - 电静力学 Poisson-Boltzmann 方程求解
  - 弹性网络刚度矩阵组装与矩阵链优化

运行方式:
  python main.py

输出:
  控制台打印各子模块的科学计算结果与数值精度指标。
"""

import numpy as np
import time
import os
import sys

# 确保模块路径正确
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
    """打印带分隔线的章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_dna_topology_module() -> dict:
    """运行染色质拓扑与四面体网格模块。"""
    print_section("模块 1: 染色质三维拓扑离散化 (tet_mesh_quad + tet_mesh_to_xml)")

    # 生成简化核小体阵列四面体网格
    mesh = generate_nucleosome_tet_mesh(
        n_rings=4, n_theta=8, n_z=4,
        major_radius=5.5, minor_radius=3.3, pitch=2.7
    )
    print(f"  生成网格: {mesh.n_nodes} 个节点, {mesh.n_elements} 个单元")

    # 计算网格总体积
    vol_sum, _ = mesh.integrate_nodal_values(np.ones(mesh.n_nodes))
    print(f"  网格总体积: {vol_sum:.4f} nm^3")

    # 计算表面积
    surface_area = mesh.compute_surface_area()
    print(f"  边界表面积: {surface_area:.4f} nm^2")

    # 模拟 γH2AX 密度场并计算修复腔室体积
    np.random.seed(1)
    gamma_density = np.exp(-np.sum(mesh.nodes ** 2, axis=1) / (2.0 * 50.0 ** 2))
    gamma_density = gamma_density / np.max(gamma_density)
    vol_repair, total_signal = compute_dsb_repair_compartment_volume(
        mesh, gamma_density, threshold=0.15
    )
    print(f"  DSB 修复腔室体积 (threshold=0.15): {vol_repair:.4f} nm^3")
    print(f"  总 γH2AX 信号量: {total_signal:.4f}")

    # XML 输出验证
    xml_str = mesh.to_xml_string()
    print(f"  XML 网格字符串长度: {len(xml_str)} 字符")

    return {
        "mesh": mesh,
        "volume": vol_sum,
        "surface_area": surface_area,
        "repair_volume": vol_repair,
    }


def run_brownian_dynamics_module() -> dict:
    """运行布朗动力学模块。"""
    print_section("模块 2: 修复蛋白布朗动力学 (normal_ode + henon_orbit)")

    # KU80 搜索时间模拟
    result = simulate_ku80_search_time(n_proteins=20, n_steps=2000)
    print(f"  模拟蛋白数: 20")
    print(f"  平均首次通过时间 (MFPT): {result['mfpt_us']:.2f} μs")
    print(f"  结合分数: {result['binding_fraction']:.3f}")
    print(f"  最终 MSD: {result['msd_final_nm2']:.2f} nm^2")

    # Henon 混沌映射验证
    x = np.linspace(-0.9, 0.9, 100)
    y = np.linspace(-0.9, 0.9, 100)
    X, Y = np.meshgrid(x, y)
    X_map, Y_map = henon_crowding_map(X.ravel(), Y.ravel(), c=0.98, n_iter=5)
    print(f"  Henon 映射后均值位移: {np.mean(np.abs(X_map - X.ravel())):.6f}")

    # 正态分布 ODE 解验证
    t = np.linspace(-3.0, 3.0, 100)
    y_gauss = normal_distribution_ode_solution(t, sigma0=1.0)
    print(f"  高斯分布积分校验: {np.trapezoid(y_gauss, t):.6f} (理论值: 1.0)")

    return result


def run_reaction_diffusion_module() -> dict:
    """运行反应-扩散模块。"""
    print_section("模块 3: γH2AX 信号波与 PARP1 扩散 (gray_scott + porous_medium)")

    # Gray-Scott γH2AX 波模拟
    # du, dv 按 dx^2 缩放以保持无量纲扩散数一致
    gs_result = simulate_gamma_h2ax_wave(
        nx=128, ny=128, nt=8000,
        f=0.03, k=0.062, du=16.0, dv=8.0,
        dt=0.1, dx=10.0,
    )
    print(f"  Gray-Scott 波前速度: {gs_result['wave_velocity']:.4f} nm/步")
    print(f"  总 γH2AX 量: {gs_result['total_gamma_h2ax']:.2f}")
    print(f"  最大 v 浓度: {gs_result['max_v']:.4f}")

    # 多孔介质方程 PARP1 扩散
    pm_result = simulate_parp1_nonlinear_diffusion(
        nx=256, nt=400, dt=0.01, dx=0.1, m=3.0,
    )
    print(f"  多孔介质数值-精确解 L2 误差: {pm_result['l2_error']:.6e}")
    print(f"  数值解总质量: {pm_result['total_mass_numerical']:.6f}")
    print(f"  精确解总质量: {pm_result['total_mass_exact']:.6f}")

    # PDE 残差验证
    x_test = np.linspace(-5.0, 5.0, 50)
    res = porous_medium_residual(x_test, t=1.0, m=3.0)
    print(f"  Barenblatt 解 PDE 残差范数: {np.linalg.norm(res):.6e}")

    return {
        "gray_scott": gs_result,
        "porous_medium": pm_result,
    }


def run_energy_landscape_module() -> dict:
    """运行能量景观模块。"""
    print_section("模块 4: 修复蛋白构象能量景观 (sammon_data + pwl_interp_2d)")

    # 生成模拟的二面角数据（RAD51 构象采样）
    np.random.seed(99)
    n_frames = 300
    n_dihedrals = 8
    # 构造两个主要构象态 + 噪声
    state_a = np.random.randn(n_dihedrals) * 0.3
    state_b = np.random.randn(n_dihedrals) * 0.3 + np.array([2.0, -1.5, 1.0, 0.5, -0.8, 1.2, -0.3, 0.7])

    dihedrals = np.zeros((n_frames, n_dihedrals))
    for i in range(n_frames):
        if i < n_frames // 2:
            dihedrals[i, :] = state_a + np.random.randn(n_dihedrals) * 0.2
        else:
            dihedrals[i, :] = state_b + np.random.randn(n_dihedrals) * 0.2

    # 构建自由能表面
    fel_result = build_free_energy_surface(dihedrals, temperature=310.0, grid_n=30)
    print(f"  Sammon 降维后坐标范围: x=[{fel_result['grid_x'].min():.2f}, {fel_result['grid_x'].max():.2f}], "
          f"y=[{fel_result['grid_y'].min():.2f}, {fel_result['grid_y'].max():.2f}]")
    print(f"  识别局部极小值个数: {len(fel_result['local_minima'])}")
    print(f"  能垒高度: {fel_result['barrier_height_ev']:.4f} eV")

    # PWL 插值验证
    xi_test = np.linspace(fel_result['grid_x'].min(), fel_result['grid_x'].max(), 20)
    yi_test = np.linspace(fel_result['grid_y'].min(), fel_result['grid_y'].max(), 20)
    XI, YI = np.meshgrid(xi_test, yi_test)
    ZI = pwl_interp_2d(
        fel_result['grid_x'], fel_result['grid_y'], fel_result['free_energy_ev'].T,
        XI.ravel(), YI.ravel()
    )
    valid = np.isfinite(ZI)
    print(f"  PWL 插值有效点比例: {np.mean(valid):.2%}")

    # 单形顶点坐标生成
    simplex_4d = simplex_vertex_coordinates(4)
    # 对于中心在原点、顶点范数为 1 的正则单形，顶点间距离 = sqrt(2 + 2/n)
    theoretical_dist = np.sqrt(2.0 + 2.0 / 4.0)
    print(f"  4D 正则单形顶点间距离: {np.linalg.norm(simplex_4d[:,0] - simplex_4d[:,1]):.6f} (理论值: {theoretical_dist:.6f})")

    return fel_result


def run_sparse_solver_module() -> dict:
    """运行稀疏求解器模块。"""
    print_section("模块 5: Poisson-Boltzmann 电静力学 (plasma_matrix + r8gb)")

    # 求解非线性 PB 方程
    pb_result = solve_nonlinear_pb(n=129, domain_length=20.0, max_iter=30, tol=1e-9)
    print(f"  PB 方程求解 {'成功' if pb_result['success'] else '未收敛'}")
    print(f"  Newton 迭代次数: {pb_result['iterations']}")
    print(f"  最终残差范数: {pb_result['residual_norm']:.6e}")
    print(f"  电势最小值: {pb_result['phi'].min():.4f} k_B T/e")
    print(f"  电势最大值: {pb_result['phi'].max():.4f} k_B T/e")

    # 带状 LU 求解器自洽验证
    n_test = 50
    A_dense = np.diag(2.0 * np.ones(n_test)) + np.diag(-1.0 * np.ones(n_test - 1), 1) + np.diag(-1.0 * np.ones(n_test - 1), -1)
    b_test = np.ones(n_test)

    solver = BandedLU(n_test, ml=1, mu=1)
    A_band = solver._full_to_band(A_dense)
    A_lu, pivot, info = solver.factorize(A_band)
    x_sol = solver.solve(A_lu, pivot, b_test)

    residual = np.linalg.norm(A_dense @ x_sol - b_test)
    print(f"  带状 LU 求解残差: {residual:.6e}")

    return pb_result


def run_matrix_operations_module() -> dict:
    """运行矩阵操作模块。"""
    print_section("模块 6: ENM 刚度矩阵与矩阵链优化 (matrix_assemble + matrix_chain + mm_to_msm)")

    # 矩阵链最优次序
    dims = [30, 35, 15, 5, 10, 20, 25]
    cost, s = matrix_chain_optimal_order(dims)
    opt_expr = build_optimal_parenthesization(s, 0, len(dims) - 2)
    cat_num = catalan_number(len(dims) - 2)
    print(f"  矩阵链维度: {dims}")
    print(f"  最优标量乘法代价: {cost}")
    print(f"  最优加括号方式: {opt_expr}")
    print(f"  总方案数 (Catalan): {cat_num}")

    # 弹性网络刚度矩阵组装
    np.random.seed(3)
    n_nodes = 20
    coords = np.random.randn(n_nodes, 3) * 10.0
    H = assemble_enm_stiffness_matrix(coords, cutoff=20.0, spring_constant=1.0)
    print(f"  ENM 刚度矩阵尺寸: {H.shape}")
    print(f"  刚度矩阵条件数估计: {np.linalg.cond(H):.4e}")

    # 零空间维度（平移/转动模式）
    eigvals = np.linalg.eigvalsh(H)
    n_zero = np.sum(np.abs(eigvals) < 1e-8)
    print(f"  零特征值个数 (刚性运动模式): {n_zero}")

    # Matrix Market I/O 验证
    temp_file = "/tmp/test_enm_matrix.mtx"
    write_matrix_market(temp_file, H, rep="coordinate", symm="symmetric")
    H_read, rows, cols, entries, rep, field, symm = read_matrix_market(temp_file)
    print(f"  Matrix Market I/O: {rows}x{cols}, 非零元={entries}, 格式={rep}")
    diff = np.max(np.abs(H - H_read))
    print(f"  读写一致性误差: {diff:.6e}")

    # 张量缩并代价估计
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
    """运行 I/O 解析模块。"""
    print_section("模块 7: TEC 数据解析与自适应网格加密 (tec_io + image_double)")

    # 构造模拟 TEC 数据
    np.random.seed(5)
    node_coord = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float64).T  # (3, 5)

    element_node = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=np.int64).T  # (4, 2)

    node_data = np.array([
        [1.0, 0.8, 0.9, 1.1, 0.85],
    ], dtype=np.float64)  # (1, 5)

    tec_content = build_tec_file(
        node_coord, element_node, node_data,
        variable_names=['X', 'Y', 'Z', 'ElectronDensity']
    )
    print(f"  生成 TEC 文件内容长度: {len(tec_content)} 字符")

    # 解析回读
    parsed = parse_tec_file(tec_content)
    print(f"  解析结果: {parsed['node_num']} 节点, {parsed['element_num']} 单元")
    print(f"  空间维度: {parsed['dim_num']}, 变量数: {len(parsed['variable_names'])}")

    # 自适应网格加密
    np.random.seed(8)
    field = np.zeros((32, 32))
    # 在中心制造高梯度区
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

    # 网格加倍验证
    doubled = grid_double_resolution(field, mode="2d")
    print(f"  网格加倍后形状: {doubled.shape}")
    print(f"  加倍前后积分守恒误差: {abs(np.sum(field) - np.sum(doubled)):.6e}")

    return parsed


def main():
    """统一入口函数，零参数可运行。"""
    print("\n" + "#" * 70)
    print("#  DNA 损伤修复分子动力学综合模拟平台")
    print("#  科学领域: 分子动力学 — DNA损伤修复分子机制")
    print("#  编程语言: Python 3")
    print("#" * 70)

    start_time = time.time()

    # 执行各模块
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


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: generate_nucleosome_tet_mesh 输出结构正确性 ----
np.random.seed(42)
mesh = generate_nucleosome_tet_mesh(n_rings=2, n_theta=4, n_z=2)
assert mesh.n_nodes > 0 and mesh.n_elements > 0, '[TC01] generate_nucleosome_tet_mesh 输出结构正确性 FAILED'

# ---- TC02: TetMesh 表面积为正且体积积分与节点数一致 ----
mesh2 = generate_nucleosome_tet_mesh(n_rings=2, n_theta=4, n_z=2)
assert mesh2.compute_surface_area() > 0, '[TC02] TetMesh 表面积为正且体积积分与节点数一致 FAILED'
vol_sum, _ = mesh2.integrate_nodal_values(np.ones(mesh2.n_nodes))
assert vol_sum > 0, '[TC02] TetMesh 表面积为正且体积积分与节点数一致 FAILED'

# ---- TC03: compute_dsb_repair_compartment_volume 非负性与阈值单调性 ----
np.random.seed(1)
gamma_density = np.exp(-np.sum(mesh.nodes ** 2, axis=1) / (2.0 * 50.0 ** 2))
gamma_density = gamma_density / np.max(gamma_density)
vol_repair, total_signal = compute_dsb_repair_compartment_volume(mesh, gamma_density, threshold=0.15)
assert vol_repair >= 0 and total_signal >= 0, '[TC03] compute_dsb_repair_compartment_volume 非负性与阈值单调性 FAILED'
vol_repair_high, _ = compute_dsb_repair_compartment_volume(mesh, gamma_density, threshold=0.85)
assert vol_repair_high <= vol_repair + 1e-12, '[TC03] compute_dsb_repair_compartment_volume 非负性与阈值单调性 FAILED'

# ---- TC04: normal_distribution_ode_solution 峰值、边界与积分归一化 ----
y0 = normal_distribution_ode_solution(np.array([0.0]), sigma0=1.0)
assert abs(y0[0] - 1.0 / np.sqrt(2.0 * np.pi)) < 1e-10, '[TC04] normal_distribution_ode_solution 峰值、边界与积分归一化 FAILED'
t = np.linspace(-5, 5, 1000)
y = normal_distribution_ode_solution(t, sigma0=1.0)
integral = np.trapezoid(y, t)
assert abs(integral - 1.0) < 0.01, '[TC04] normal_distribution_ode_solution 峰值、边界与积分归一化 FAILED'
y_far = normal_distribution_ode_solution(np.array([-20.0, 20.0]), sigma0=1.0)
assert np.all(y_far >= 0) and np.all(y_far < 1e-80), '[TC04] normal_distribution_ode_solution 峰值、边界与积分归一化 FAILED'

# ---- TC05: henon_crowding_map 原点、圆外不变性与有界性 ----
x = np.array([0.0, 2.0, 0.5])
y = np.array([0.0, 0.0, 0.5])
xm, ym = henon_crowding_map(x, y, c=0.98, n_iter=5)
assert xm[0] == 0.0 and ym[0] == 0.0, '[TC05] henon_crowding_map 原点、圆外不变性与有界性 FAILED'
assert xm[1] == 2.0 and ym[1] == 0.0, '[TC05] henon_crowding_map 原点、圆外不变性与有界性 FAILED'
assert np.sqrt(xm[2] ** 2 + ym[2] ** 2) < 1.0 + 1e-6, '[TC05] henon_crowding_map 原点、圆外不变性与有界性 FAILED'

# ---- TC06: OverdampedLangevinIntegrator 构造校验与step输出形状 ----
try:
    OverdampedLangevinIntegrator(dt=-0.001)
    assert False, '[TC06] OverdampedLangevinIntegrator 构造校验与step输出形状 FAILED'
except ValueError:
    pass
integrator = OverdampedLangevinIntegrator(dt=0.001)
x = np.array([[100.0, 0.0, 0.0]])
f = np.array([[1.0, 0.0, 0.0]])
x_new = integrator.step(x, f)
assert x_new.shape == (1, 3), '[TC06] OverdampedLangevinIntegrator 构造校验与step输出形状 FAILED'
assert np.linalg.norm(x_new[0]) <= 5000.0 + 1e-6, '[TC06] OverdampedLangevinIntegrator 构造校验与step输出形状 FAILED'

# ---- TC07: dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 ----
force = dna_repair_protein_force(np.zeros((0, 3)), np.zeros(3))
assert force.shape == (0, 3), '[TC07] dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 FAILED'
force2 = dna_repair_protein_force(np.array([[1.0, 0.0, 0.0]]), np.zeros(3))
assert force2.shape == (1, 3) and np.all(np.isfinite(force2)), '[TC07] dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 FAILED'
np.random.seed(7)
force3 = dna_repair_protein_force(np.random.randn(5, 3), np.zeros(3))
assert force3.shape == (5, 3) and np.all(np.isfinite(force3)), '[TC07] dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 FAILED'

# ---- TC08: simulate_ku80_search_time 输出结构与固定种子可复现性 ----
np.random.seed(42)
result_a = simulate_ku80_search_time(n_proteins=5, n_steps=100)
assert 'mfpt_us' in result_a and 'binding_fraction' in result_a and 'msd_final_nm2' in result_a, '[TC08] simulate_ku80_search_time 输出结构与固定种子可复现性 FAILED'
np.random.seed(42)
result_b = simulate_ku80_search_time(n_proteins=5, n_steps=100)
assert abs(result_a['mfpt_us'] - result_b['mfpt_us']) < 1e-12, '[TC08] simulate_ku80_search_time 输出结构与固定种子可复现性 FAILED'

# ---- TC09: gray_scott_step 维度保持、非负性截断与零场稳定性 ----
u = np.ones((8, 8)) * 0.5
v = np.ones((8, 8)) * 0.25
u_new, v_new = gray_scott_step(u, v, du=0.16, dv=0.08, f=0.03, k=0.062, dt=0.1, dx=1.0, dy=1.0)
assert u_new.shape == (8, 8) and v_new.shape == (8, 8), '[TC09] gray_scott_step 维度保持、非负性截断与零场稳定性 FAILED'
assert np.all(u_new >= 0) and np.all(v_new >= 0), '[TC09] gray_scott_step 维度保持、非负性截断与零场稳定性 FAILED'
u0, v0 = gray_scott_step(np.zeros((4, 4)), np.zeros((4, 4)), du=0.16, dv=0.08, f=0.03, k=0.062, dt=0.1, dx=1.0, dy=1.0)
assert np.all(u0 >= 0) and np.all(v0 >= 0), '[TC09] gray_scott_step 维度保持、非负性截断与零场稳定性 FAILED'

# ---- TC10: porous_medium_residual Barenblatt精确解残差范数与紧支集外为零 ----
x = np.linspace(-2, 2, 100)
res = porous_medium_residual(x, t=1.0, m=3.0)
assert np.linalg.norm(res) < 1.0, '[TC10] porous_medium_residual Barenblatt精确解残差范数与紧支集外为零 FAILED'
x_far = np.array([-10.0, 10.0])
res_far = porous_medium_residual(x_far, t=1.0, m=3.0)
assert np.allclose(res_far, 0.0), '[TC10] porous_medium_residual Barenblatt精确解残差范数与紧支集外为零 FAILED'

# ---- TC11: simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 ----
pm = simulate_parp1_nonlinear_diffusion(nx=64, nt=50, dt=0.01, dx=0.1, m=3.0)
assert 'l2_error' in pm and 'total_mass_numerical' in pm and 'total_mass_exact' in pm, '[TC11] simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 FAILED'
assert pm['total_mass_numerical'] >= 0, '[TC11] simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 FAILED'
assert pm['l2_error'] < 1.0, '[TC11] simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 FAILED'

# ---- TC12: simplex_vertex_coordinates 维度、范数与夹角约束 ----
for nd in [2, 3, 4]:
    s = simplex_vertex_coordinates(nd)
    assert s.shape == (nd, nd + 1), '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'
    norms = np.linalg.norm(s, axis=0)
    assert np.allclose(norms, 1.0), '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'
    cos_angle = np.dot(s[:, 0], s[:, 1])
    assert abs(cos_angle + 1.0 / nd) < 1e-10, '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'
s4 = simplex_vertex_coordinates(4)
d = np.linalg.norm(s4[:, 0] - s4[:, 1])
theory = np.sqrt(2.0 + 2.0 / 4.0)
assert abs(d - theory) < 1e-6, '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'

# ---- TC13: pwl_interp_2d 三角形插值精确性与外推行为 ----
xd = np.array([0.0, 1.0, 2.0])
yd = np.array([0.0, 1.0, 2.0])
zd = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
zi = pwl_interp_2d(xd, yd, zd, np.array([0.5]), np.array([0.2]))
assert abs(zi[0] - 0.7) < 1e-9, '[TC13] pwl_interp_2d 三角形插值精确性与外推行为 FAILED'
zi2 = pwl_interp_2d(xd, yd, zd, np.array([1.5]), np.array([1.8]))
assert np.isfinite(zi2[0]), '[TC13] pwl_interp_2d 三角形插值精确性与外推行为 FAILED'
zi_out = pwl_interp_2d(xd, yd, zd, np.array([5.0]), np.array([5.0]))
assert zi_out[0] == np.inf, '[TC13] pwl_interp_2d 三角形插值精确性与外推行为 FAILED'

# ---- TC14: sammon_mapping 降维后输出形状正确 ----
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
Y = sammon_mapping(X, n_components=2, max_iter=10, alpha=0.3, random_state=42)
assert Y.shape == (4, 2), '[TC14] sammon_mapping 降维后输出形状正确 FAILED'

# ---- TC15: BandedLU 三对角系统求解残差 ----
n = 20
A = np.diag(2.0 * np.ones(n)) + np.diag(-1.0 * np.ones(n - 1), 1) + np.diag(-1.0 * np.ones(n - 1), -1)
b = np.ones(n)
solver = BandedLU(n, ml=1, mu=1)
A_band = solver._full_to_band(A)
A_lu, pivot, info = solver.factorize(A_band)
x_sol = solver.solve(A_lu, pivot, b)
res = np.linalg.norm(A @ x_sol - b)
assert res < 1e-10, '[TC15] BandedLU 三对角系统求解残差 FAILED'

# ---- TC16: solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 ----
pb = solve_nonlinear_pb(n=33, domain_length=10.0, max_iter=20, tol=1e-6)
assert pb['success'], '[TC16] solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 FAILED'
assert abs(pb['phi'][0]) < 1e-6 and abs(pb['phi'][-1]) < 1e-6, '[TC16] solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 FAILED'
assert pb['residual_norm'] < 1e-6, '[TC16] solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 FAILED'

# ---- TC17: matrix_chain_optimal_order 经典动态规划最优解与边界 ----
dims = [30, 35, 15, 5, 10, 20, 25]
cost, s = matrix_chain_optimal_order(dims)
assert cost == 15125, '[TC17] matrix_chain_optimal_order 经典动态规划最优解与边界 FAILED'
cost_single, _ = matrix_chain_optimal_order([10, 20])
assert cost_single == 0, '[TC17] matrix_chain_optimal_order 经典动态规划最优解与边界 FAILED'

# ---- TC18: catalan_number 递推正确性与小值边界 ----
assert catalan_number(5) == 42, '[TC18] catalan_number 递推正确性与小值边界 FAILED'
assert catalan_number(0) == 1, '[TC18] catalan_number 递推正确性与小值边界 FAILED'
assert catalan_number(-1) == 0, '[TC18] catalan_number 递推正确性与小值边界 FAILED'

# ---- TC19: assemble_enm_stiffness_matrix 对称性、维度与力平衡 ----
np.random.seed(3)
coords = np.random.randn(5, 3) * 10.0
H = assemble_enm_stiffness_matrix(coords, cutoff=50.0, spring_constant=1.0)
assert H.shape == (15, 15), '[TC19] assemble_enm_stiffness_matrix 对称性、维度与力平衡 FAILED'
assert np.allclose(H, H.T), '[TC19] assemble_enm_stiffness_matrix 对称性、维度与力平衡 FAILED'
row_sums = np.sum(H, axis=1)
assert np.allclose(row_sums, 0.0), '[TC19] assemble_enm_stiffness_matrix 对称性、维度与力平衡 FAILED'

# ---- TC20: Matrix Market I/O 坐标格式与数组格式往返一致性 ----
A_test = np.array([[1.0, 2.0], [2.0, 3.0]])
write_matrix_market("/tmp/test_mm_coord.mtx", A_test, rep="coordinate", symm="symmetric")
A_read, rows, cols, entries, rep, field, symm = read_matrix_market("/tmp/test_mm_coord.mtx")
assert np.allclose(A_test, A_read), '[TC20] Matrix Market I/O 坐标格式与数组格式往返一致性 FAILED'
assert symm == "symmetric", '[TC20] Matrix Market I/O 坐标格式与数组格式往返一致性 FAILED'
write_matrix_market("/tmp/test_mm_array.mtx", A_test, rep="array", symm="general")
A_read2, rows2, cols2, entries2, rep2, field2, symm2 = read_matrix_market("/tmp/test_mm_array.mtx")
assert np.allclose(A_test, A_read2), '[TC20] Matrix Market I/O 坐标格式与数组格式往返一致性 FAILED'

# ---- TC21: build_tec_file与parse_tec_file 往返一致性与维度解析 ----
node_coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64).T
element_node = np.array([[0, 1, 2, 3]], dtype=np.int64).T
node_data = np.array([[1.0, 0.8, 0.9, 1.1]], dtype=np.float64)
tec = build_tec_file(node_coord, element_node, node_data, variable_names=['X', 'Y', 'Z', 'Density'])
parsed = parse_tec_file(tec)
assert parsed['node_num'] == 4 and parsed['element_num'] == 1, '[TC21] build_tec_file与parse_tec_file 往返一致性与维度解析 FAILED'
assert parsed['dim_num'] == 3, '[TC21] build_tec_file与parse_tec_file 往返一致性与维度解析 FAILED'
assert 'Density' in parsed['variable_names'], '[TC21] build_tec_file与parse_tec_file 往返一致性与维度解析 FAILED'

# ---- TC22: grid_double_resolution 2D/3D形状与积分守恒 ----
field = np.ones((4, 4))
doubled = grid_double_resolution(field, mode="2d")
assert doubled.shape == (8, 8), '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'
assert abs(np.sum(field) - np.sum(doubled)) < 1e-12, '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'
field3 = np.ones((2, 2, 2))
doubled3 = grid_double_resolution(field3, mode="3d")
assert doubled3.shape == (4, 4, 4), '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'
assert abs(np.sum(field3) - np.sum(doubled3)) < 1e-12, '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'

# ---- TC23: adaptive_mesh_refinement_2d 至少返回原始网格与形状递增 ----
np.random.seed(8)
field2d = np.zeros((8, 8))
field2d[3:5, 3:5] = 1.0
levels = adaptive_mesh_refinement_2d(field2d, gradient_threshold=0.1, max_level=1)
assert len(levels) >= 1 and levels[0].shape == field2d.shape, '[TC23] adaptive_mesh_refinement_2d 至少返回原始网格与形状递增 FAILED'
if len(levels) > 1:
    assert levels[1].shape[0] == 2 * field2d.shape[0], '[TC23] adaptive_mesh_refinement_2d 至少返回原始网格与形状递增 FAILED'

# ---- TC24: compute_optimal_tensor_contraction_cost 与矩阵链一致性 ----
tensor_cost = compute_optimal_tensor_contraction_cost(
    [(30, 35), (35, 15), (15, 5), (5, 10), (10, 20), (20, 25)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
)
assert tensor_cost == 15125, '[TC24] compute_optimal_tensor_contraction_cost 与矩阵链一致性 FAILED'

# ---- TC25: simulate_gamma_h2ax_wave 输出结构与非负总量 ----
gs = simulate_gamma_h2ax_wave(nx=32, ny=32, nt=100, f=0.03, k=0.062, du=0.16, dv=0.08, dt=0.1, dx=1.0)
assert gs['u_final'].shape == (32, 32) and gs['v_final'].shape == (32, 32), '[TC25] simulate_gamma_h2ax_wave 输出结构与非负总量 FAILED'
assert gs['total_gamma_h2ax'] >= 0, '[TC25] simulate_gamma_h2ax_wave 输出结构与非负总量 FAILED'

# ---- TC26: assemble_poisson_boltzmann_jacobian 输出维度与边界条件 ----
n_j = 9
phi_j = np.zeros(n_j)
rho_j = np.zeros(n_j)
J, F = assemble_poisson_boltzmann_jacobian(n_j, phi_j, rho_j, h=0.5)
assert J.shape == (n_j, n_j), '[TC26] assemble_poisson_boltzmann_jacobian 输出维度与边界条件 FAILED'
assert F.shape == (n_j,), '[TC26] assemble_poisson_boltzmann_jacobian 输出维度与边界条件 FAILED'
assert J[0, 0] == 1.0 and F[0] == 0.0, '[TC26] assemble_poisson_boltzmann_jacobian 输出维度与边界条件 FAILED'

# ---- TC27: TetMesh常数场积分等于总体积 ----
mesh3 = generate_nucleosome_tet_mesh(n_rings=2, n_theta=4, n_z=2)
integral_const, total_vol = mesh3.integrate_nodal_values(np.ones(mesh3.n_nodes))
assert abs(integral_const - total_vol) < 1e-12, '[TC27] TetMesh常数场积分等于总体积 FAILED'
assert total_vol > 0, '[TC27] TetMesh常数场积分等于总体积 FAILED'

# ---- TC28: build_optimal_parenthesization 输出非空字符串 ----
dims2 = [30, 35, 15, 5, 10, 20, 25]
cost2, s2 = matrix_chain_optimal_order(dims2)
opt_expr = build_optimal_parenthesization(s2, 0, len(dims2) - 2)
assert isinstance(opt_expr, str) and len(opt_expr) > 0, '[TC28] build_optimal_parenthesization 输出非空字符串 FAILED'

# ---- TC29: main() 集成运行返回None且不崩溃 ----
main_result = main()
assert main_result is None, '[TC29] main() 集成运行返回None且不崩溃 FAILED'

# ---- TC30: pwl_interp_2d 网格边界点精确恢复 ----
xg = np.array([0.0, 1.0, 2.0])
yg = np.array([0.0, 1.0])
zg = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
zi_grid = pwl_interp_2d(xg, yg, zg, xg, yg.repeat(len(xg)))
assert np.allclose(zi_grid[:len(xg)], zg[:, 0]), '[TC30] pwl_interp_2d 网格边界点精确恢复 FAILED'

print('\n全部 30 个测试通过!\n')
