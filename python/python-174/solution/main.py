"""
main.py
快速多极子方法(FMM)三维N体问题统一入口

科学问题:
    计算N个带电粒子在三维空间中的库仑相互作用势能,
    利用自适应快速多极子方法将复杂度从O(N^2)降低到O(N),
    并通过蒙特卡洛采样、误差分析、自适应网格细化等博士级技术
    确保数值结果的高精度与鲁棒性。

核心物理模型:
    势能: Phi(r_i) = sum_{j != i} q_j / |r_i - r_j|
    力:   F(r_i) = sum_{j != i} q_j * (r_i - r_j) / |r_i - r_j|^3

数学基础:
    - 球谐函数展开: Y_l^m(theta, phi) = N_l^m * P_l^m(cos theta) * e^{i*m*phi}
    - 多极展开: 1/|x-x_j| = sum_{l,m} (r_j^l / r^{l+1}) * Y_l^m(theta_j,phi_j) * Y_l^{m*}(theta,phi)
    - 局部展开: 1/|x-x_j| = sum_{l,m} (r^l / r_j^{l+1}) * Y_l^m(theta,phi) * Y_l^{m*}(theta_j,phi_j)
    - M2M/M2L/L2L转换算子基于球谐加法定理

运行方式:
    python main.py
    (零参数, 自动生成测试粒子并运行完整实验)
"""

import numpy as np
import time
import sys

from fmm_solver import FMMSolver
from nbody_kernel import coulomb_potential_direct
from spherical_geometry import compute_bounding_sphere, uniform_sphere_sample
from monte_carlo_sampler import monte_carlo_fmm_verification, nonuniform_particle_sample
from error_analysis import (
    relative_l2_error, relative_inf_error, fmm_error_budget,
    estimate_truncation_order, convergence_order
)
from adaptive_mesh import AdaptiveTriMesh, project_3d_to_2d, triangle_area


def generate_test_particles(n_particles=200, charge_distribution="uniform"):
    """
    生成测试粒子
    
    参数:
        n_particles: int, 粒子数
        charge_distribution: str, "uniform", "gaussian", 或 "dipole"
    
    返回:
        points: ndarray (N, 3)
        charges: ndarray (N,)
    """
    np.random.seed(42)
    if charge_distribution == "uniform":
        points = np.random.uniform(-1.0, 1.0, size=(n_particles, 3))
        charges = np.random.uniform(-1.0, 1.0, size=n_particles)
    elif charge_distribution == "gaussian":
        points = np.random.normal(size=(n_particles, 3)) * 0.5
        charges = np.random.normal(size=n_particles)
    elif charge_distribution == "dipole":
        # 创建偶极子分布
        n_half = n_particles // 2
        points_pos = np.random.normal(size=(n_half, 3)) * 0.3 + np.array([0.5, 0.0, 0.0])
        points_neg = np.random.normal(size=(n_half, 3)) * 0.3 - np.array([0.5, 0.0, 0.0])
        points = np.vstack([points_pos, points_neg])
        charges = np.hstack([np.ones(n_half), -np.ones(n_half)])
        if n_particles % 2 == 1:
            points = np.vstack([points, np.zeros((1, 3))])
            charges = np.hstack([charges, [0.0]])
    else:
        points = np.random.uniform(-1.0, 1.0, size=(n_particles, 3))
        charges = np.random.uniform(-1.0, 1.0, size=n_particles)
    return points, charges


def run_fmm_experiment(n_particles=200, expansion_order=4, max_depth=5):
    """
    运行单次FMM实验
    
    返回:
        dict: 包含points, charges, fmm_potential, direct_potential, timing, error
    """
    print(f"\n{'='*60}")
    print(f"FMM实验: N={n_particles}, L={expansion_order}, max_depth={max_depth}")
    print(f"{'='*60}")

    # 生成粒子
    points, charges = generate_test_particles(n_particles, "dipole")
    print(f"粒子范围: x=[{points[:,0].min():.3f},{points[:,0].max():.3f}], "
          f"y=[{points[:,1].min():.3f},{points[:,1].max():.3f}], "
          f"z=[{points[:,2].min():.3f},{points[:,2].max():.3f}]")
    print(f"总电荷: {charges.sum():.6f}")

    # FMM计算
    t0 = time.time()
    solver = FMMSolver(points, charges, order=expansion_order, max_depth=max_depth, max_particles=20)
    fmm_potential = solver.compute_potential()
    t_fmm = time.time() - t0
    print(f"FMM计算时间: {t_fmm:.4f} s")

    # 直接求和 (仅对小规模)
    t0 = time.time()
    if n_particles <= 1000:
        direct_potential = coulomb_potential_direct(points, charges)
        t_direct = time.time() - t0
        print(f"直接求和时间: {t_direct:.4f} s")

        # 误差分析
        l2_err = relative_l2_error(fmm_potential, direct_potential)
        inf_err = relative_inf_error(fmm_potential, direct_potential)
        print(f"相对L2误差: {l2_err:.6e}")
        print(f"相对L_inf误差: {inf_err:.6e}")
        if t_fmm > 1e-6 and t_direct > 1e-6:
            speedup = t_direct / t_fmm
            print(f"加速比: {speedup:.2f}x")
    else:
        direct_potential = None
        t_direct = None
        l2_err = None
        inf_err = None
        print("粒子数过大, 跳过直接求和")

    # 树统计
    stats = solver.get_tree_statistics()
    print(f"八叉树统计: 节点数={stats['total_nodes']}, 叶子数={stats['total_leaves']}, "
          f"最大深度={stats['max_depth']}, 平均每叶粒子数={stats['avg_particles_per_leaf']:.1f}")

    # 误差预算
    budget = fmm_error_budget(n_particles, expansion_order)
    print(f"误差预算: 截断={budget['truncation_error']:.2e}, 转换={budget['translation_error']:.2e}, "
          f"舍入={budget['roundoff_error']:.2e}, 总估计={budget['total_error_estimate']:.2e}")

    return {
        "points": points,
        "charges": charges,
        "fmm_potential": fmm_potential,
        "direct_potential": direct_potential,
        "t_fmm": t_fmm,
        "t_direct": t_direct,
        "l2_error": l2_err,
        "inf_error": inf_err,
        "tree_stats": stats,
        "error_budget": budget
    }


def run_convergence_study():
    """收敛性研究: 改变展开阶数, 观察误差衰减"""
    print(f"\n{'='*60}")
    print("FMM收敛性研究")
    print(f"{'='*60}")

    n_particles = 100
    orders = [2, 4, 6, 8]
    errors_l2 = []
    errors_inf = []

    points, charges = generate_test_particles(n_particles, "dipole")
    direct_potential = coulomb_potential_direct(points, charges)

    for L in orders:
        solver = FMMSolver(points, charges, order=L, max_depth=5, max_particles=20)
        fmm_potential = solver.compute_potential()
        l2_err = relative_l2_error(fmm_potential, direct_potential)
        inf_err = relative_inf_error(fmm_potential, direct_potential)
        errors_l2.append(l2_err)
        errors_inf.append(inf_err)
        print(f"L={L}: L2误差={l2_err:.6e}, L_inf误差={inf_err:.6e}")

    # 估计收敛阶数
    p_l2 = convergence_order(errors_l2, np.array(orders, dtype=float))
    print(f"\n收敛阶数估计 (L2): {p_l2}")

    # 截断误差指数拟合
    machine_eps = 2.2e-16
    fit = estimate_truncation_order(errors_l2, machine_eps, orders)
    print(f"截断误差衰减率: {fit['rate']:.4f}")


def run_monte_carlo_validation():
    """蒙特卡洛验证FMM结果"""
    print(f"\n{'='*60}")
    print("蒙特卡洛FMM验证")
    print(f"{'='*60}")

    points, charges = generate_test_particles(150, "gaussian")
    solver = FMMSolver(points, charges, order=5, max_depth=5)
    fmm_potential = solver.compute_potential()
    direct_potential = coulomb_potential_direct(points, charges)

    result = monte_carlo_fmm_verification(
        points, charges, fmm_potential, direct_potential,
        n_sample_pairs=100, confidence=0.95
    )
    print(f"平均相对误差: {result['mean_relative_error']:.6e}")
    print(f"标准误差: {result['std_error']:.6e}")
    print(f"95%置信区间: [{result['confidence_interval'][0]:.6e}, {result['confidence_interval'][1]:.6e}]")
    print(f"Z分数: {result['z_score']:.4f}")

    # 非均匀采样验证
    def density_func(v):
        return np.exp(-2.0 * np.linalg.norm(v))
    samples = nonuniform_particle_sample(density_func, 50, domain="sphere")
    print(f"非均匀采样粒子数: {samples.shape[0]}")


def run_adaptive_mesh_demo():
    """自适应网格细化演示"""
    print(f"\n{'='*60}")
    print("自适应网格细化")
    print(f"{'='*60}")

    points, charges = generate_test_particles(80, "uniform")
    # 投影到2D
    proj = project_3d_to_2d(points)

    # 构建初始三角网格 (Delaunay三角剖分的简化版)
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(proj)
        elements = [tuple(tri.simplices[i]) for i in range(tri.simplices.shape[0])]
        mesh = AdaptiveTriMesh(proj, elements)
        print(f"初始网格: {mesh.points.shape[0]} 节点, {len(mesh.elements)} 三角形")

        # 计算各单元面积
        areas = np.array([mesh.element_area(i) for i in range(len(mesh.elements))])
        print(f"三角形面积范围: [{areas.min():.6f}, {areas.max():.6f}], 平均: {areas.mean():.6f}")

        # 模拟误差指示子 (面积越大误差越大)
        indicators = areas / (areas.max() + 1e-15)
        refined_mesh = mesh.refine_by_indicator(indicators, theta=0.8)
        print(f"细化后网格: {refined_mesh.points.shape[0]} 节点, {len(refined_mesh.elements)} 三角形")

        # 输出网格数据
        data = refined_mesh.to_mesh_data()
        print(f"网格数据: dim={data['dim']}, vertices={data['vertices']}, triangles={data['triangles']}")
    except Exception as e:
        print(f"网格演示跳过 (缺少scipy或异常): {e}")


def run_spherical_harmonic_test():
    """球谐函数计算测试"""
    print(f"\n{'='*60}")
    print("球谐函数计算测试")
    print(f"{'='*60}")

    from spherical_geometry import legendre_associated_normalized, spherical_harmonic_basis
    theta = np.pi / 3.0
    phi = np.pi / 4.0
    l_max = 4
    c_all, s_all = spherical_harmonic_basis(l_max, theta, phi)
    print(f"球谐基计算: l_max={l_max}")
    for m in range(l_max + 1):
        print(f"  m={m}: Y_{l_max}^{m},c = {c_all[m][l_max]:.6f}")

    # 测试归一化Legendre
    cx = legendre_associated_normalized(5, 2, 0.5)
    print(f"P_5^2(0.5) 归一化值: {cx[5]:.6f}")


def run_kernel_tests():
    """核函数与转换算子测试"""
    print(f"\n{'='*60}")
    print("核函数与转换算子测试")
    print(f"{'='*60}")

    from nbody_kernel import monomial_value, pwc_kernel_approx, evaluate_pwc_kernel
    from translation_operators import m2m_translate, m2l_translate, l2l_translate

    # 单项式测试
    pts = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])
    val = monomial_value([1, 2, 0], pts)
    print(f"单项式 x*y^2 在 (1,2,3) 和 (2,1,0.5) 的值: {val}")

    # 分段常数核
    breaks, values = pwc_kernel_approx(0.1, 2.0, 10)
    k_val = evaluate_pwc_kernel(0.5, breaks, values)
    print(f"分段常数核在 r=0.5 的值: {k_val:.6f} (精确值: {1.0/0.5:.6f})")

    # 转换算子测试 (小规模)
    child_center = np.zeros(3)
    parent_center = np.array([1.0, 0.0, 0.0])
    order = 2
    child_real = [np.array([1.0]), np.array([0.5, 0.3]), np.array([0.1, 0.05, 0.02])]
    child_imag = [np.zeros(1), np.zeros(2), np.zeros(3)]
    p_real, p_imag = m2m_translate(child_real, child_imag, child_center, parent_center, order)
    print(f"M2M转换测试: 子节点中心 {child_center}, 父节点中心 {parent_center}")
    print(f"  父节点多极矩 (l=0): {p_real[0][0]:.6f}")


def main():
    """主入口函数"""
    print("=" * 70)
    print("快速多极子方法(FMM) - 三维N体问题求解器")
    print("计算数学: 快速多极子方法N体问题")
    print("=" * 70)

    # 1. 基础FMM实验
    result = run_fmm_experiment(n_particles=200, expansion_order=5, max_depth=5)

    # 2. 收敛性研究
    run_convergence_study()

    # 3. 蒙特卡洛验证
    run_monte_carlo_validation()

    # 4. 自适应网格
    run_adaptive_mesh_demo()

    # 5. 球谐函数测试
    run_spherical_harmonic_test()

    # 6. 核函数与转换算子
    run_kernel_tests()

    print(f"\n{'='*70}")
    print("所有实验完成!")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
