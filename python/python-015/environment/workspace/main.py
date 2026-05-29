"""
main.py
拓扑半金属Weyl节点数值研究系统 - 统一入口

凝聚态物理：拓扑半金属Weyl节点的多尺度数值研究

本程序执行以下博士级科学计算流程：
1. Weyl哈密顿量建模（线性模型与紧束缚模型）
2. Weyl节点定位与拓扑分类
3. 布里渊区多方法采样（MC/CVT/均匀）
4. Berry曲率与Berry相位计算
5. 拓扑不变量（Chern数、Weyl荷）
6. Fermi面三角剖分与面积计算
7. 态密度计算（直方图/高斯展宽/Hermite求积）
8. 稀疏网格高维积分
9. 半经典运动方程演化
10. 输运通道优化（背包问题）
11. 统计显著性检验

所有计算零参数自动运行，使用内置默认物理参数。
"""

import numpy as np
import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weyl_hamiltonian import WeylHamiltonian, band_gap, velocity_operator
from berry_curvature import (
    berry_connection_numeric, berry_curvature_numeric,
    berry_curvature_analytic_linear, berry_phase_1d,
    chern_number_2d_slice, weyl_charge_surface_integral
)
from bzone_sampler import (
    ellipse_sample, ellipse_area, cvt_sampler_nonuniform,
    uniform_kpoint_grid, adaptive_weyl_node_sampler
)
from sparse_integrator import sparse_grid_quadrature, integrate_sparse_grid
from triangulation_mesh import (
    delaunay_triangulate_2d, triangle_area_2d,
    triangulation_boundary_nodes, triangulation_total_area,
    fermi_surface_2d_slice, node_values_to_element_average
)
from ode_evolver import (
    rk12_adaptive, periodic_lattice_dynamics,
    evolve_berry_phase_along_path
)
from density_of_states import (
    dos_histogram, dos_gaussian_broadening,
    dos_weyl_semimetal_analytic, test_hermite_exactness
)
from topological_invariant import (
    compute_chern_numbers_vs_kz, locate_weyl_nodes_from_chern_jump,
    compute_weyl_charges_spherical, nielsen_ninomiya_theorem_check,
    berry_phase_wilson_loop
)
from transport_optimizer import (
    knapsack_rational, transport_channel_selection,
    chiral_anomaly_conductance
)
from statistical_tests import (
    alnorm, chyper, weyl_node_pairing_test, berry_curvature_significance
)
from mesh_extractor import (
    mesh2d_extract, mesh3d_extract, node_values_to_elements,
    create_fermi_surface_mesh, mesh_to_xml_string
)
from utils import (
    safe_normalize, finite_difference_gradient,
    rotation_matrix_from_axis_angle, fermi_dirac
)


def print_section(title):
    """打印章节分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """主程序：零参数可运行"""
    
    print("=" * 70)
    print("  拓扑半金属Weyl节点数值研究系统")
    print("  Topological Weyl Semimetal Numerical Research System")
    print("=" * 70)
    
    # ================================================================
    # 1. Weyl哈密顿量初始化
    # ================================================================
    print_section("1. Weyl哈密顿量建模")
    
    # 线性Weyl模型：H = hbar*v_F * k·sigma
    ham_linear = WeylHamiltonian(model_type="linear", hbar=1.0, v_f=1.0)
    print("  [线性Weyl模型] H = hbar*v_F * (k_x*sigma_x + k_y*sigma_y + k_z*sigma_z)")
    
    # 紧束缚双Weyl模型
    ham_tb = WeylHamiltonian(model_type="tight_binding", hbar=1.0, v_f=1.0)
    print("  [紧束缚模型] 模拟TaAs类材料的典型参数")
    
    # 测试单点本征值
    k_test = np.array([0.1, 0.2, 0.3])
    e_lin, v_lin = ham_linear.eigenproblem(k_test)
    print(f"  测试k点 {k_test}: E = {e_lin}")
    print(f"  能带间隙: {band_gap(e_lin.reshape(1, -1))[0]:.6f}")
    
    # ================================================================
    # 2. Weyl节点定位
    # ================================================================
    print_section("2. Weyl节点定位")
    
    # 线性模型：节点在原点
    node_linear = ham_linear.weyl_node_position_linear()
    print(f"  线性模型Weyl节点位置: {node_linear}")
    
    # 紧束缚模型：数值搜索
    nodes_tb = ham_tb.find_weyl_nodes_tight_binding(grid_size=32)
    print(f"  紧束缚模型发现 {len(nodes_tb)} 个Weyl节点:")
    for i, node in enumerate(nodes_tb):
        print(f"    节点 {i+1}: k = ({node[0]:.4f}, {node[1]:.4f}, {node[2]:.4f})")
    
    # ================================================================
    # 3. 布里渊区多方法采样
    # ================================================================
    print_section("3. 布里渊区多方法采样")
    
    bz_bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
    
    # 3a. 椭圆/椭球蒙特卡洛采样（Weyl节点附近）
    n_mc = 200
    a_ellipsoid = np.eye(3)
    mc_samples = ellipse_sample(n_mc, a_ellipsoid, 0.5).T + node_linear
    vol_ellipsoid = ellipse_area(a_ellipsoid, 0.5)
    print(f"  [椭球MC采样] 采样点数: {n_mc}, 椭球体积: {vol_ellipsoid:.6f}")
    
    # 3b. CVT非均匀密度采样
    def density_func(pts):
        # 密度与能隙倒数相关（Weyl节点附近密度高）
        dens = np.zeros(len(pts))
        for i in range(len(pts)):
            e, _ = ham_linear.eigenproblem(pts[i])
            gap = abs(e[1] - e[0])
            dens[i] = 1.0 / (gap + 0.01)
        return dens
    
    cvt_samples = cvt_sampler_nonuniform(
        n_generators=50, sample_num=500, it_num=10,
        density_func=density_func, bounds=bz_bounds
    )
    print(f"  [CVT采样] 生成器数: 50, 迭代次数: 10")
    
    # 3c. 均匀k点网格
    uniform_samples = uniform_kpoint_grid(bz_bounds, grid_size=8)
    print(f"  [均匀网格] 网格尺寸: 8x8x8, 总点数: {len(uniform_samples)}")
    
    # 3d. 自适应Weyl节点采样
    if len(nodes_tb) > 0:
        adaptive_samples = adaptive_weyl_node_sampler(
            n_points=150, weyl_nodes=nodes_tb, node_radius=0.4, bz_bounds=bz_bounds
        )
        print(f"  [自适应采样] 总点数: {len(adaptive_samples)} (节点附近密集)")
    else:
        adaptive_samples = uniform_samples
        print(f"  [自适应采样] 退化为均匀采样（未发现节点）")
    
    # ================================================================
    # 4. Berry曲率与Berry相位
    # ================================================================
    print_section("4. Berry曲率与Berry相位计算")
    
    # 在线性模型Weyl节点附近计算Berry曲率
    k_near = np.array([0.2, 0.1, 0.15])
    omega_num = berry_curvature_numeric(ham_linear, k_near, band_index=0)
    omega_ana = berry_curvature_analytic_linear(k_near, chirality=1)
    
    print(f"  测试k点: {k_near}")
    print(f"  Berry曲率(数值): Omega_xy = {omega_num[0,1]:.6f}")
    print(f"  Berry曲率(解析): Omega_xy = {omega_ana[0,1]:.6f}")
    
    # Berry相位沿闭合路径
    theta = np.linspace(0, 2 * np.pi, 50)
    radius = 0.3
    path = np.array([
        [radius * np.cos(t), radius * np.sin(t), 0.0]
        for t in theta
    ])
    phase = berry_phase_1d(ham_linear, path, band_index=0)
    print(f"  Berry相位(半径{radius}圆环): gamma = {phase:.6f} rad = {phase/np.pi:.4f}*pi")
    
    # ================================================================
    # 5. 拓扑不变量
    # ================================================================
    print_section("5. 拓扑不变量计算")
    
    # 5a. Chern数随kz变化
    kz_values = np.linspace(-1.0, 1.0, 15)
    chern_numbers = compute_chern_numbers_vs_kz(
        ham_linear, kz_values,
        kx_range=(-1.0, 1.0), ky_range=(-1.0, 1.0),
        grid_size=20, band_index=0
    )
    print("  Chern数随kz变化:")
    for i in range(len(kz_values)):
        print(f"    kz = {kz_values[i]:.3f}: C = {chern_numbers[i]:.1f}")
    
    # 5b. Weyl荷计算
    if len(nodes_tb) > 0:
        charges = compute_weyl_charges_spherical(ham_tb, nodes_tb, radius=0.2)
        print(f"  Weyl荷(紧束缚模型): {charges}")
        nn_ok = nielsen_ninomiya_theorem_check(charges)
        print(f"  Nielsen-Ninomiya定理验证: {'通过' if nn_ok else '未通过'} (sum Q = {np.sum(charges):.1f})")
    else:
        # 线性模型中手动测试
        charge = weyl_charge_surface_integral(ham_linear, node_linear, radius=0.3)
        print(f"  Weyl荷(线性模型): Q = {charge:.3f} (理论值: +1)")
    
    # 5c. Wilson loop Berry相位
    kx_line = np.linspace(-np.pi, np.pi, 40)
    wl_phase = berry_phase_wilson_loop(ham_linear, kx_line, ky_fixed=0.0, kz_fixed=0.0)
    print(f"  Wilson loop Berry相位: {wl_phase:.6f}")
    
    # ================================================================
    # 6. Fermi面三角剖分
    # ================================================================
    print_section("6. Fermi面三角剖分")
    
    # 构造二维截面的Fermi面
    def energy_func_2d(kpts):
        e, _ = ham_linear.eigenproblem(kpts)
        return e[:, 0] if e.ndim > 1 else e
    
    fs_points, fs_triangles = fermi_surface_2d_slice(
        energy_func_2d,
        kx_range=(-1.0, 1.0), ky_range=(-1.0, 1.0),
        kz_fixed=0.0, e_fermi=0.0, grid_size=24
    )
    print(f"  Fermi面采样点数: {len(fs_points)}")
    print(f"  三角形数量: {len(fs_triangles)}")
    
    fs_area = 0.0
    if len(fs_triangles) > 0:
        fs_area = triangulation_total_area(fs_points, fs_triangles)
        print(f"  Fermi面总面积: {fs_area:.6f}")
        
        # 边界节点
        is_bdy = triangulation_boundary_nodes(len(fs_points), fs_triangles)
        print(f"  边界节点数: {np.sum(is_bdy)}")
        
        # 节点值到元素平均
        node_vals = np.random.rand(len(fs_points))  # 示例数据
        elem_vals = node_values_to_element_average(node_vals, fs_triangles)
        print(f"  节点到元素平均完成，元素值范围: [{elem_vals.min():.4f}, {elem_vals.max():.4f}]")
    
    # ================================================================
    # 7. 态密度计算
    # ================================================================
    print_section("7. 态密度(DOS)计算")
    
    # 7a. 在k空间网格上计算能量
    k_grid = uniform_kpoint_grid(bz_bounds, grid_size=12)
    energies_all, _ = ham_linear.eigenproblem(k_grid)
    energies_flat = energies_all.flatten()
    
    # 7b. 直方图法
    e_bins, dos_hist = dos_histogram(energies_flat, -2.0, 2.0, n_bins=40)
    print(f"  直方图法DOS峰值位置: E = {e_bins[np.argmax(dos_hist)]:.3f}")
    
    # 7c. 高斯展宽法
    e_grid = np.linspace(-2.0, 2.0, 100)
    dos_gauss = dos_gaussian_broadening(energies_flat, e_grid, sigma=0.08)
    print(f"  高斯展宽法DOS峰值: E = {e_grid[np.argmax(dos_gauss)]:.3f}")
    
    # 7d. 解析Weyl半金属DOS
    dos_analytic = dos_weyl_semimetal_analytic(e_grid, v_f=1.0)
    print(f"  解析DOS在E=0.5: D(E) = {dos_weyl_semimetal_analytic(np.array([0.5]))[0]:.6f}")
    
    # 7e. Hermite求积精确性检验
    errors = test_hermite_exactness(alpha=0.0, max_degree=7, n_points=5)
    print("  Gauss-Hermite求积精确性检验 (5点规则应精确到9次):")
    for deg in range(0, 8, 2):
        print(f"    次数 {deg}: 相对误差 = {errors[deg]:.2e}")
    
    # ================================================================
    # 8. 稀疏网格高维积分
    # ================================================================
    print_section("8. 稀疏网格高维积分")
    
    # 计算Berry曲率在BZ上的平均值
    def berry_curvature_integrand(kpts):
        vals = np.zeros(len(kpts))
        for i in range(len(kpts)):
            omega = berry_curvature_numeric(ham_linear, kpts[i], band_index=0)
            vals[i] = omega[0, 1]  # Omega_xy分量
        return vals
    
    sg_points, sg_weights = sparse_grid_quadrature(dim=3, max_level=3, bounds=bz_bounds)
    print(f"  稀疏网格(层级3): 节点数 = {len(sg_points)}")
    
    if len(sg_points) > 0:
        sg_integral = integrate_sparse_grid(3, 3, berry_curvature_integrand, bz_bounds)
        print(f"  Omega_xy在BZ积分: {sg_integral:.6f}")
    
    # ================================================================
    # 9. 半经典运动方程演化
    # ================================================================
    print_section("9. 半经典运动方程演化")
    
    # 周期性晶格动力学测试
    y0_l96 = np.array([8.0, 8.0, 8.01, 8.0])
    t_l96, y_l96, e_l96 = rk12_adaptive(
        lambda t, y: periodic_lattice_dynamics(y, force=8.0),
        tspan=(0.0, 5.0), y0=y0_l96, dt=0.01, tol=1e-5
    )
    print(f"  Lorenz96-like周期性动力学: 演化到 t = {t_l96[-1]:.2f}")
    print(f"  最终状态: {y_l96[-1]}")
    print(f"  最终误差估计: {e_l96[-1]:.2e}")
    
    # ================================================================
    # 10. 输运通道优化
    # ================================================================
    print_section("10. 输运通道优化")
    
    n_channels = 20
    gains = np.random.rand(n_channels) * 2.0 + 0.5
    costs = np.random.rand(n_channels) * 1.0 + 0.2
    budget = 8.0
    
    # 按增益密度排序
    from transport_optimizer import sort_by_profit_density
    gains_s, costs_s = sort_by_profit_density(gains, costs)
    x_opt, total_cost, total_gain = knapsack_rational(n_channels, budget, gains_s, costs_s)
    
    print(f"  通道数: {n_channels}, 总预算: {budget}")
    print(f"  最优选择比例: {x_opt}")
    print(f"  总成本: {total_cost:.4f}, 总增益: {total_gain:.4f}")
    print(f"  增益密度排序后的最优策略完成")
    
    # 手征反常电导
    k_test_transport = np.array([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0]])
    e_field = np.array([1.0, 0.0, 0.0])
    b_field = np.array([1.0, 0.0, 0.0])
    cond = chiral_anomaly_conductance(ham_linear, k_test_transport, e_field, b_field)
    print(f"  手征反常修正电导: {cond}")
    
    # ================================================================
    # 11. 统计检验
    # ================================================================
    print_section("11. 统计显著性检验")
    
    # 11a. Weyl节点配对检验
    p_pair, expected = weyl_node_pairing_test(
        n_total_kpoints=10000, n_candidate_nodes=20,
        n_tested=10, n_confirmed_pairs=6
    )
    print(f"  Weyl节点配对检验: p值 = {p_pair:.4f}, 期望配对数 = {expected:.2f}")
    
    # 11b. Berry曲率显著性
    omega_samples = np.random.randn(100) * 0.5 + 0.3
    z_score, p_berry = berry_curvature_significance(omega_samples)
    print(f"  Berry曲率显著性: Z = {z_score:.3f}, p = {p_berry:.4f}")
    
    # 11c. 正态累积概率
    p_norm = alnorm(1.96, upper=True)
    print(f"  标准正态P(Z > 1.96) = {p_norm:.4f}")
    
    # 11d. 超几何分布
    hg_prob, ifault = chyper(point=True, kk=10, ll=3, mm=100, nn=30)
    print(f"  超几何分布P(X=3): {hg_prob:.6f} (ifault={ifault})")
    
    # ================================================================
    # 12. 网格提取与格式转换
    # ================================================================
    print_section("12. 网格提取与格式转换")
    
    # 2D Fermi面网格
    if len(fs_points) > 0 and len(fs_triangles) > 0:
        mesh_2d = mesh2d_extract(fs_points, fs_triangles)
        print(f"  2D网格: {mesh_2d['node_num']} 节点, {mesh_2d['element_num']} 元素")
        
        # 节点值到元素
        elem_data = node_values_to_elements(node_vals, fs_triangles)
        print(f"  节点值→元素值平均完成")
        
        # XML格式输出（仅打印信息，不写文件）
        xml_info = mesh_to_xml_string(mesh_2d)
        print(f"  XML格式字符串长度: {len(xml_info)} 字符")
    
    # 3D网格示例
    nodeco_3d = np.random.rand(8, 3)
    elnode_3d = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    mesh_3d = mesh3d_extract(nodeco_3d, elnode_3d)
    print(f"  3D网格示例: {mesh_3d['node_num']} 节点, {mesh_3d['element_num']} 四面体")
    
    # ================================================================
    # 13. 总结
    # ================================================================
    print_section("计算完成 - 结果总结")
    print(f"  Weyl节点数 (紧束缚模型): {len(nodes_tb)}")
    print(f"  Weyl节点数 (线性模型): 1 (位于原点)")
    if len(nodes_tb) > 0:
        print(f"  Nielsen-Ninomiya定理: {'满足' if nielsen_ninomiya_theorem_check(compute_weyl_charges_spherical(ham_tb, nodes_tb, radius=0.2)) else '需更多网格细化'}")
    print(f"  Berry相位(圆环): {phase:.4f} rad")
    print(f"  Chern数范围: [{chern_numbers.min():.0f}, {chern_numbers.max():.0f}]")
    print(f"  Fermi面面积: {fs_area if len(fs_triangles) > 0 else 0:.4f}")
    print(f"  输运最优增益: {total_gain:.4f}")
    print(f"  所有模块运行正常，无报错。")
    print("=" * 70)


if __name__ == "__main__":
    main()
