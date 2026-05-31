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
    main()

# ================================================================
# 测试用例（51个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# 测试所需额外导入 (部分函数仅在main.py函数体内导入)
from nbody_kernel import monomial_value, coulomb_force_direct, pwc_kernel_approx, evaluate_pwc_kernel, build_transition_matrix_from_neighbors, kernel_gradient_laplacian
from spherical_geometry import cartesian_to_spherical, spherical_to_cartesian, great_circle_distance, spherical_cap_area, solid_angle, quadrilateral_area_2d, legendre_associated_normalized, spherical_harmonic_basis, circle_points_on_plane
from monte_carlo_sampler import walker_build, walker_sampler, disk01_positive_sample, coin_biased, alnorm, mc_estimate_integral
from error_analysis import kolmogorov_smirnov_statistic, markov_chain_steady_state, second_eigenvalue_rate
from translation_operators import m2m_translate, l2l_translate, compute_translation_matrix
from adaptive_mesh import refine_triangle_midpoint

# ---- TC01: monomial_value 返回标量ndarray ----
import numpy as np
x = monomial_value([1, 0, 0], np.array([[2.0, 0.0, 0.0]]))
assert isinstance(x, np.ndarray), '[TC01] monomial_value 应返回 ndarray FAILED'
assert abs(x[0] - 2.0) < 1e-12, '[TC01] monomial_value x^1 在 (2,0,0) 应为 2.0 FAILED'

# ---- TC02: monomial_value 多指数 x^2*y^1 ----
import numpy as np
x = monomial_value([2, 1, 0], np.array([[3.0, 2.0, 1.0]]))
assert abs(x[0] - 18.0) < 1e-12, '[TC02] monomial_value x^2*y 在 (3,2,1) 应为 18 FAILED'

# ---- TC03: coulomb_potential_direct 两粒子解析验证 ----
import numpy as np
np.random.seed(42)
pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
chg = np.array([1.0, -1.0])
pot = coulomb_potential_direct(pts, chg)
assert abs(pot[0] + 1.0) < 1e-10, '[TC03] 两粒子势能点0应为 -1.0 FAILED'
assert abs(pot[1] - 1.0) < 1e-10, '[TC03] 两粒子势能点1应为 1.0 FAILED'

# ---- TC04: coulomb_potential_direct 软化参数 ----
import numpy as np
np.random.seed(42)
pts_eq = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
chg_eq = np.array([1.0, 1.0])
pot_eq = coulomb_potential_direct(pts_eq, chg_eq, epsilon=0.5)
assert abs(pot_eq[0] - 2.0) < 1e-10, '[TC04] 软化参数0.5下势能应为 2.0 FAILED'

# ---- TC05: coulomb_potential_direct 有限性 ----
import numpy as np
np.random.seed(42)
pts_10 = np.random.uniform(-1, 1, size=(10, 3))
chg_10 = np.random.uniform(-1, 1, size=10)
pot_10 = coulomb_potential_direct(pts_10, chg_10)
assert np.all(np.isfinite(pot_10)), '[TC05] 势能应全为有限值 FAILED'

# ---- TC06: coulomb_force_direct 输出形状 ----
import numpy as np
np.random.seed(42)
pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
chg = np.array([1.0, -1.0])
f = coulomb_force_direct(pts, chg)
assert f.shape == (2, 3), '[TC06] 力输出形状应为 (2,3) FAILED'
assert np.all(np.isfinite(f)), '[TC06] 力应全为有限值 FAILED'

# ---- TC07: pwc_kernel_approx 输出形状与单调性 ----
import numpy as np
nb, nv = pwc_kernel_approx(0.1, 2.0, 10)
assert len(nb) == 11, '[TC07] 断点数应为 n_segments+1=11 FAILED'
assert len(nv) == 10, '[TC07] 常数值数应为 10 FAILED'
assert nb[0] == 0.1, '[TC07] 第一个断点应为 r_min=0.1 FAILED'
assert nb[-1] == 2.0, '[TC07] 最后一个断点应为 r_max=2.0 FAILED'

# ---- TC08: evaluate_pwc_kernel 单值 ----
import numpy as np
nb, nv = pwc_kernel_approx(0.5, 2.0, 3)
val = evaluate_pwc_kernel(0.6, nb, nv)
assert float(val) > 0, '[TC08] 分段核函数值应为正 FAILED'

# ---- TC09: build_transition_matrix_from_neighbors 行和为1 ----
import numpy as np
np.random.seed(42)
nc = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
T = build_transition_matrix_from_neighbors(nc, 3)
row_sums = np.sum(T, axis=1)
assert np.all(np.abs(row_sums - 1.0) < 1e-12), '[TC09] 转移矩阵行和应为 1 FAILED'

# ---- TC10: kernel_gradient_laplacian 源外拉普拉斯量为0 ----
import numpy as np
np.random.seed(42)
pts_k = np.array([[0.0, 0.0, 0.0]])
chg_k = np.array([1.0])
target_k = np.array([1.0, 0.0, 0.0])
pot_k, grad_k, lap_k = kernel_gradient_laplacian(pts_k, chg_k, target_k)
assert abs(lap_k - 0.0) < 1e-12, '[TC10] 源外拉普拉斯量应为 0 FAILED'
assert abs(pot_k - 1.0) < 1e-10, '[TC10] 1/r势能在r=1处应为 1.0 FAILED'

# ---- TC11: cartesian_to_spherical 基本值 ----
import numpy as np
np.random.seed(42)
xyz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
r, theta, phi = cartesian_to_spherical(xyz)
assert abs(r[0] - 1.0) < 1e-12, '[TC11] (1,0,0) 半径应为 1 FAILED'
assert abs(theta[1]) < 1e-12, '[TC11] (0,0,1) theta 应为 0 FAILED'
assert abs(r[1] - 1.0) < 1e-12, '[TC11] (0,0,1) 半径应为 1 FAILED'

# ---- TC12: spherical_to_cartesian 往返 ----
import numpy as np
np.random.seed(42)
xyz2 = spherical_to_cartesian(1.0, np.pi / 2, 0.0)
assert abs(xyz2[0, 0] - 1.0) < 1e-10, '[TC12] sin(pi/2)*cos(0) 应为 1.0 FAILED'
assert abs(xyz2[0, 2]) < 1e-10, '[TC12] cos(pi/2) 应为 0 FAILED'

# ---- TC13: great_circle_distance 赤道对跖点 ----
import numpy as np
d = great_circle_distance(1.0, np.pi / 2, 0.0, np.pi / 2, np.pi)
assert abs(d - np.pi) < 1e-8, '[TC13] 赤道对跖点大圆距离应为 pi FAILED'

# ---- TC14: spherical_cap_area 半球 ----
import numpy as np
np.random.seed(42)
area = spherical_cap_area(1.0, np.pi / 2)
assert abs(area - 2.0 * np.pi) < 1e-10, '[TC14] 半球冠面积应为 2*pi FAILED'

# ---- TC15: solid_angle 全空间 ----
import numpy as np
omega = solid_angle(1.0, 4.0 * np.pi)
assert abs(omega - 4.0 * np.pi) < 1e-10, '[TC15] 全空间立体角应为 4*pi FAILED'

# ---- TC16: quadrilateral_area_2d 单位正方形 ----
import numpy as np
np.random.seed(42)
quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
area_q = quadrilateral_area_2d(quad)
assert abs(area_q - 1.0) < 1e-10, '[TC16] 单位正方形面积应为 1.0 FAILED'

# ---- TC17: legendre_associated_normalized P_0^0 归一化 ----
import numpy as np
cx = legendre_associated_normalized(0, 0, 0.0)
assert abs(cx[0] - 1.0 / np.sqrt(4.0 * np.pi)) < 1e-12, '[TC17] P_0^0(0) 归一化值应为 1/sqrt(4pi) FAILED'

# ---- TC18: legendre_associated_normalized 高阶有限性 ----
import numpy as np
cx = legendre_associated_normalized(5, 2, 0.5)
assert np.all(np.isfinite(cx)), '[TC18] 归一化连带Legendre应全有限 FAILED'

# ---- TC19: spherical_harmonic_basis 输出结构 ----
import numpy as np
np.random.seed(42)
c, s = spherical_harmonic_basis(3, np.pi / 2, 0.0)
assert len(c) == 4, '[TC19] 球谐基 c_all 长度应为 L+1=4 FAILED'
assert len(s) == 4, '[TC19] 球谐基 s_all 长度应为 L+1=4 FAILED'

# ---- TC20: uniform_sphere_sample 单位球面 ----
import numpy as np
np.random.seed(42)
samples = uniform_sphere_sample(100)
norms = np.linalg.norm(samples, axis=1)
assert np.all(np.abs(norms - 1.0) < 1e-12), '[TC20] 球面采样点应在单位球面上 FAILED'

# ---- TC21: circle_points_on_plane 输出形状 ----
import numpy as np
np.random.seed(42)
pts_circ = circle_points_on_plane(np.array([0.0, 0.0, 0.0]), 1.0, np.array([0.0, 0.0, 1.0]), num_points=16)
assert pts_circ.shape == (16, 3), '[TC21] 圆周点形状应为 (16,3) FAILED'

# ---- TC22: compute_bounding_sphere 基本 ----
import numpy as np
np.random.seed(42)
pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
center_cbs, radius_cbs = compute_bounding_sphere(pts)
assert radius_cbs > 0, '[TC22] 包围球半径应为正 FAILED'
assert center_cbs.shape == (3,), '[TC22] 球心形状应为 (3,) FAILED'

# ---- TC23: walker_build 输出类型 ----
import numpy as np
np.random.seed(42)
prob = np.array([0.1, 0.2, 0.3, 0.4])
y, a = walker_build(prob)
assert len(y) == 4 and len(a) == 4, '[TC23] Walker表长度应为 4 FAILED'
assert np.all(y >= 0) and np.all(y <= 1.0 + 1e-12), '[TC23] 阈值应在 [0,1] 范围内 FAILED'

# ---- TC24: walker_sampler 确定性种子 ----
import numpy as np
np.random.seed(42)
prob = np.array([1.0, 0.0])
y, a = walker_build(prob)
np.random.seed(42)
idx = walker_sampler(y, a)
assert 0 <= idx <= 1, '[TC24] Walker采样索引应在范围内 FAILED'

# ---- TC25: disk01_positive_sample 正半平面与圆内 ----
import numpy as np
np.random.seed(42)
pts_disk = disk01_positive_sample(50)
assert pts_disk.shape == (50, 2), '[TC25] 圆盘采样形状应为 (50,2) FAILED'
assert np.all(pts_disk >= 0), '[TC25] 正半圆盘所有坐标应 >= 0 FAILED'
assert np.all(np.linalg.norm(pts_disk, axis=1) <= 1.0 + 1e-12), '[TC25] 采样点应在单位圆内 FAILED'

# ---- TC26: coin_biased 全正面 ----
import numpy as np
np.random.seed(42)
coins = coin_biased(10, 1.0)
assert np.all(coins == 1.0), '[TC26] heads_prob=1 应全为 +1 FAILED'

# ---- TC27: coin_biased 全反面 ----
import numpy as np
np.random.seed(42)
coins2 = coin_biased(10, 0.0)
assert np.all(coins2 == -1.0), '[TC27] heads_prob=0 应全为 -1 FAILED'

# ---- TC28: alnorm P(X<0)=0.5 ----
import numpy as np
p = alnorm(0.0, upper=False)
assert abs(p - 0.5) < 1e-10, '[TC28] 标准正态 P(X<0) 应为 0.5 FAILED'

# ---- TC29: alnorm 对称性 P(X<z) = P(X>-z) upper ----
import numpy as np
p1 = alnorm(1.0, upper=False)
p2 = alnorm(-1.0, upper=True)
assert abs(p1 - p2) < 1e-10, '[TC29] alnorm 对称性 P(X<1)=P(X>1) upper FAILED'

# ---- TC30: mc_estimate_integral 常函数 ----
import numpy as np
np.random.seed(42)
def const_sampler():
    return 1.0
mean, std_err = mc_estimate_integral(const_sampler, 100)
assert abs(mean - 1.0) < 1e-6, '[TC30] 常函数均值应为 1 FAILED'

# ---- TC31: relative_l2_error 零误差 ----
import numpy as np
np.random.seed(42)
a = np.array([1.0, 2.0, 3.0])
err = relative_l2_error(a, a)
assert abs(err) < 1e-14, '[TC31] 完全相同输入 relative_l2_error 应为 0 FAILED'

# ---- TC32: relative_inf_error 有误差 ----
import numpy as np
np.random.seed(42)
exact = np.array([1.0, 2.0, 3.0])
approx = np.array([1.1, 2.0, 3.0])
err = relative_inf_error(approx, exact)
assert err > 0, '[TC32] 有误差时 relative_inf_error 应 > 0 FAILED'

# ---- TC33: convergence_order 线性收敛 ----
import numpy as np
errors = np.array([0.1, 0.05, 0.025])
params = np.array([1.0, 2.0, 4.0])
p = convergence_order(errors, params)
assert len(p) == 2, '[TC33] 收敛阶数数组长度应为 K-1=2 FAILED'

# ---- TC34: fmm_error_budget 分量非负 ----
import numpy as np
budget = fmm_error_budget(200, 5)
assert budget['truncation_error'] >= 0, '[TC34] 截断误差应为非负 FAILED'
assert budget['translation_error'] >= 0, '[TC34] 转换误差应为非负 FAILED'
assert budget['total_error_estimate'] > 0, '[TC34] 总误差估计应为正 FAILED'

# ---- TC35: estimate_truncation_order 返回 rate ----
import numpy as np
np.random.seed(42)
errors_fmm = np.array([0.1, 0.01, 0.001])
orders = np.array([1, 2, 3])
result = estimate_truncation_order(errors_fmm, 1e-10, orders)
assert 'rate' in result, '[TC35] estimate_truncation_order 应返回 rate FAILED'
assert result['rate'] > 0, '[TC35] 截断误差衰减率应为正 FAILED'

# ---- TC36: kolmogorov_smirnov_statistic 范围 ----
import numpy as np
np.random.seed(42)
samples = np.random.randn(1000)
ks = kolmogorov_smirnov_statistic(samples, mu=0.0, sigma=1.0)
assert 0.0 <= ks <= 1.0, '[TC36] KS统计量应在 [0,1] 范围内 FAILED'

# ---- TC37: markov_chain_steady_state 均匀矩阵 ----
import numpy as np
np.random.seed(42)
T = np.ones((3, 3)) / 3.0
pi = markov_chain_steady_state(T)
assert pi.shape == (3,), '[TC37] 稳态分布形状应为 (3,) FAILED'
assert abs(np.sum(pi) - 1.0) < 1e-10, '[TC37] 稳态分布之和应为 1 FAILED'

# ---- TC38: second_eigenvalue_rate 基本 ----
import numpy as np
np.random.seed(42)
T = np.array([[0.5, 0.5], [0.5, 0.5]])
rate = second_eigenvalue_rate(T)
assert rate >= 0.0, '[TC38] 第二特征值绝对值应为非负 FAILED'

# ---- TC39: triangle_area 等边三角形 ----
import numpy as np
np.random.seed(42)
area = triangle_area(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                     np.array([0.5, np.sqrt(3) / 2, 0.0]))
assert abs(area - np.sqrt(3) / 4) < 1e-10, '[TC39] 等边三角形面积应为 sqrt(3)/4 FAILED'

# ---- TC40: refine_triangle_midpoint 产生4子三角形 ----
import numpy as np
np.random.seed(42)
pts_ref, tris = refine_triangle_midpoint(np.array([0.0, 0.0]),
                                          np.array([1.0, 0.0]),
                                          np.array([0.0, 1.0]))
assert len(pts_ref) == 6, '[TC40] 细化后应有 6 个点 FAILED'
assert len(tris) == 4, '[TC40] 细化后应有 4 个三角形 FAILED'

# ---- TC41: project_3d_to_2d 默认投影 xy 平面 ----
import numpy as np
np.random.seed(42)
pts3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
proj = project_3d_to_2d(pts3d)
assert proj.shape == (2, 2), '[TC41] 3D投影到2D形状应为 (2,2) FAILED'
assert abs(proj[0, 0] - 1.0) < 1e-12, '[TC41] 第一点 x 坐标应为 1.0 FAILED'

# ---- TC42: AdaptiveTriMesh element_area ----
import numpy as np
np.random.seed(42)
pts_mesh = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
elements = [(0, 1, 2), (1, 3, 2)]
mesh = AdaptiveTriMesh(pts_mesh, elements)
a = mesh.element_area(0)
assert abs(a - 0.5) < 1e-12, '[TC42] 三角形(0,1,2)面积应为 0.5 FAILED'

# ---- TC43: AdaptiveTriMesh element_diameter ----
import numpy as np
np.random.seed(42)
d = mesh.element_diameter(0)
assert d > 0, '[TC43] 三角形直径应为正 FAILED'

# ---- TC44: generate_test_particles uniform 输出形状 ----
import numpy as np
np.random.seed(42)
pts, chg = generate_test_particles(100, "uniform")
assert pts.shape == (100, 3), '[TC44] uniform 粒子坐标形状应为 (100,3) FAILED'
assert chg.shape == (100,), '[TC44] uniform 电荷形状应为 (100,) FAILED'

# ---- TC45: generate_test_particles gaussian ----
import numpy as np
np.random.seed(42)
pts_g, chg_g = generate_test_particles(50, "gaussian")
assert pts_g.shape == (50, 3), '[TC45] gaussian 粒子坐标形状应为 (50,3) FAILED'

# ---- TC46: generate_test_particles dipole 总电荷为0 ----
import numpy as np
np.random.seed(42)
pts_d, chg_d = generate_test_particles(50, "dipole")
assert pts_d.shape == (50, 3), '[TC46] dipole 粒子坐标形状应为 (50,3) FAILED'
assert abs(np.sum(chg_d)) < 1e-10, '[TC46] 偶极子总电荷应为 0 FAILED'

# ---- TC47: m2m_translate 零位移恒等 ----
import numpy as np
np.random.seed(42)
child_r = [np.array([1.0]), np.array([0.0, 0.0])]
child_i = [np.array([0.0]), np.array([0.0, 0.0])]
parent_r, parent_i = m2m_translate(child_r, child_i,
                                    np.array([0.0, 0.0, 0.0]),
                                    np.array([0.0, 0.0, 0.0]), 1)
assert abs(parent_r[0][0] - 1.0) < 1e-12, '[TC47] M2M零位移应保持矩不变 FAILED'

# ---- TC48: l2l_translate 零位移恒等 ----
import numpy as np
np.random.seed(42)
p_r = [np.array([1.0]), np.array([0.5, 0.0])]
p_i = [np.array([0.0]), np.array([0.0, 0.0])]
c_r, c_i = l2l_translate(p_r, p_i,
                           np.array([0.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 0.0]), 1)
assert abs(c_r[0][0] - 1.0) < 1e-12, '[TC48] L2L零位移应保持系数不变 FAILED'

# ---- TC49: compute_translation_matrix 距离 ----
import numpy as np
np.random.seed(42)
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
matrices = compute_translation_matrix(nodes, 2)
assert (0, 1) in matrices, '[TC49] 转换矩阵字典应包含键 (0,1) FAILED'
assert abs(matrices[(0, 1)]['distance'] - 1.0) < 1e-12, '[TC49] 距离应为 1.0 FAILED'

# ---- TC50: FMM积分测试 run_fmm_experiment 小规模 ----
import numpy as np
np.random.seed(42)
result = run_fmm_experiment(n_particles=50, expansion_order=3, max_depth=3)
assert 'fmm_potential' in result, '[TC50] FMM实验应返回 fmm_potential FAILED'
assert result['fmm_potential'] is not None, '[TC50] FMM势能不应为 None FAILED'
assert np.all(np.isfinite(result['fmm_potential'])), '[TC50] FMM势能应全为有限值 FAILED'
assert result['fmm_potential'].shape == (50,), '[TC50] FMM势能形状应为 (50,) FAILED'

# ---- TC51: nonuniform_particle_sample disk_positive domain ----
import numpy as np
np.random.seed(42)
def density(v):
    return 1.0
samples_nu = nonuniform_particle_sample(density, 30, domain="disk_positive")
assert samples_nu.shape == (30, 3), '[TC51] 非均匀采样disk形状应为 (30,3) FAILED'

print('\n全部 51 个测试通过!\n')
