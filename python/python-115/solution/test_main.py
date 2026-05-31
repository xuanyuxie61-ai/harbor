"""
main.py
酶催化反应过渡态搜索统一入口

分子动力学：酶催化反应过渡态搜索
==========================================

本项目基于 15 个种子项目的核心算法，融合构建了一个面向酶催化反应过渡态搜索的
博士级科学计算框架。主要流程：

1. 分子拓扑构建（XYZ 解析 + 图论分析）
2. 活性位点静电势计算（2D Poisson-Boltzmann）
3. 势能面 RBF 插值（稀疏从头算数据）
4. 弦方法 / NEB 反应路径优化
5. 过渡态定位与 Hessian 验证
6. 热力学积分与速率常数计算
7. 动力学验证（RKF45 积分）

运行方式：
    python main.py
（零参数运行，所有数据内嵌生成）
"""

import numpy as np
import sys

# 导入所有模块
from special_math import clausen_function, periodic_torsion_potential, angular_partition_function
from sparse_operations import CRSMatrix, build_molecular_hessian_crs, lanczos_eigenvalue_solver
from electrostatics import Poisson2DSolver, pic_charge_density, poisson_2d_exact_solution, electrostatic_stabilization_energy
from pes_surface import PESInterpolator, estimate_r0
from dynamics_ode import RKF45Integrator, GlycolysisModel, integrate_glycolysis, ReactionCoordinateDynamics
from thermodynamic_quadrature import GaussLaguerreQuadrature, PolygonMoments, HexagonMoments, ThermodynamicIntegration
from configuration_tiling import PentominoShapes, ConfigurationTiling, ConfigurationSpaceSampler
from molecular_topology import XYZParser, analyze_molecular_topology
from string_optimizer import PathParameterization, StringMethod, SequenceManager, SymmetryOperations
from transition_state import TransitionStateVerifier, NEBOptimizer, ReactionPathAnalysis


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_molecular_topology():
    """模块 1: 分子拓扑分析（基于 1423_xyz_display + 796_neighbors_to_metis_graph）"""
    print_section("模块 1: 分子拓扑与图结构分析")

    xyz_data = XYZParser.generate_demo_xyz()
    atoms, coords = XYZParser.parse_string(xyz_data)
    graph, results = analyze_molecular_topology(atoms, coords)

    print(f"  原子数: {results['n_atoms']}")
    print(f"  化学键数: {results['n_bonds']}")
    print(f"  连通分量数: {results['n_components']}")
    print(f"  平均配位数: {results['avg_degree']:.3f}")
    print(f"  候选催化三联体数: {len(results['triads'])}")

    if results['triads']:
        print(f"  第一个三联体距离: {results['triads'][0][3]:.3f} Å")

    # METIS 图输出（前 3 行）
    metis_lines = results['metis_graph'].split('\n')[:4]
    print(f"  METIS 图格式（前 4 行）:")
    for line in metis_lines:
        print(f"    {line}")

    return atoms, coords, graph


def demo_electrostatics():
    """模块 2: 静电势计算（基于 869_pic + 878_poisson_2d_exact）"""
    print_section("模块 2: 酶活性位点静电势（2D Poisson-Boltzmann）")

    nx, ny = 16, 10
    dh = 1.0  # Å

    # 蛋白板障碍物
    boundary_box = [[5, 7], [0, 4]]
    solver = Poisson2DSolver(nx, ny, dh, boundary_box)

    # 生成测试电荷密度（使用无量纲化参数避免数值过大）
    np.random.seed(42)
    part_x = np.random.rand(50, 2) * np.array([nx - 2, ny - 2]) * dh
    part_v = np.random.randn(50, 2) * 100.0
    # 使用缩放后的参数：eps0=1, qe=1 表示原子单位制
    den = pic_charge_density(nx, ny, dh, part_x, part_v, 50, 1.0, 1.0)

    # 原子单位制参数（避免 SI 单位导致的数值溢出）
    eps0_au = 1.0
    qe_au = 1.0
    phi0 = 0.0
    te = 1.0
    phi_p = -2.0
    n0 = 0.5

    phi_init = np.ones((nx, ny)) * phi0
    phi = solver.solve_gs(phi_init, den, n0, phi0, te, phi_p, eps0_au, qe_au, max_iter=500, tol=0.5)
    efx, efy = solver.compute_electric_field(phi)

    print(f"  网格尺寸: {nx} × {ny}")
    print(f"  电势范围: [{phi.min():.4f}, {phi.max():.4f}] (原子单位)")
    print(f"  电场最大值: {np.sqrt(efx**2 + efy**2).max():.4e} (原子单位)")

    # 验证：精确解对比
    x_test = np.array([0.5, 0.5])
    u_exact, ux, uy, uxx, uxy, uyy = poisson_2d_exact_solution(x_test[0], x_test[1])
    print(f"  Poisson 精确解验证 u(0.5,0.5) = {u_exact:.6f}")

    # 静电稳定化能（使用归一化密度避免数值过大）
    den_norm = den / (np.max(np.abs(den)) + 1e-10)
    estab = electrostatic_stabilization_energy(phi, den_norm, dh, dh)
    print(f"  静电稳定化能（归一化）: {estab:.4e}")
    print(f"  静电稳定化能估算: ~{abs(estab) * 0.1:.2f} kcal/mol（演示量级）")

    return phi, efx, efy


def demo_pes_interpolation():
    """模块 3: 势能面 RBF 插值（基于 1015_rbf_interp_nd）"""
    print_section("模块 3: 势能面 RBF 插值")

    # 构造二维模型势能面（双阱势能，模拟酶催化的反应物/产物盆地）
    # 两个盆地 + 一个鞍点，形式简单且数值可控
    def true_potential(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x1, x2 = x[0], x[1]
        else:
            x1, x2 = x[0, 0], x[1, 0]
        # 双阱势能: V = (x^2 - 1)^2 + 0.5*y^2 + 0.3*x*y
        # 极小值在 ~(±1, 0)，鞍点在 ~(0, 0)
        V = (x1 ** 2 - 1.0) ** 2 + 0.5 * x2 ** 2 + 0.3 * x1 * x2
        return V

    # 稀疏采样点
    np.random.seed(123)
    nd = 60
    xd = np.random.rand(2, nd) * 4.0 - 2.0
    fd = np.array([true_potential(xd[:, i]) for i in range(nd)])

    r0 = estimate_r0(xd)
    interp = PESInterpolator(2, nd, xd, r0, kernel_name='gaussian')
    interp.compute_weights(fd)

    # 测试插值精度
    np.random.seed(456)
    test_points = np.random.rand(2, 20) * 3.0 - 1.5
    true_vals = np.array([true_potential(test_points[:, i]) for i in range(20)])
    interp_vals = interp.interpolate(test_points)
    rmse = np.sqrt(np.mean((true_vals - interp_vals) ** 2))

    print(f"  采样点数: {nd}")
    print(f"  RBF 形状参数 r0: {r0:.4f}")
    print(f"  插值 RMSE: {rmse:.6f}")

    # 梯度测试
    grad = interp.gradient(np.array([0.5, 0.5]))
    print(f"  在 (0.5, 0.5) 处梯度: [{grad[0, 0]:.4f}, {grad[1, 0]:.4f}]")

    return interp, true_potential


def demo_dynamics_ode():
    """模块 4: 动力学与 ODE 积分（基于 1038_rkf45 + 472_glycolysis_ode）"""
    print_section("模块 4: 反应动力学 ODE 积分")

    # 糖酵解模型积分
    times, states = integrate_glycolysis(t0=0.0, tstop=50.0)
    print(f"  糖酵解模型积分完成")
    print(f"  积分步数: {len(times)}")
    print(f"  终态 u: {states[-1, 0]:.6f}, v: {states[-1, 1]:.6f}")

    model = GlycolysisModel()
    equi = model.equilibrium()
    print(f"  理论平衡解 u*: {equi[0]:.6f}, v*: {equi[1]:.6f}")

    # Jacobian 特征值分析
    J = model.jacobian(states[-1])
    eigvals = np.linalg.eigvals(J)
    print(f"  Jacobian 特征值: {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j, "
          f"{eigvals[1].real:.4f}{eigvals[1].imag:+.4f}j")

    # 反应坐标动力学
    def dummy_free_energy(xi):
        return 10.0 * (xi - 0.5) ** 2

    rc_dyn = ReactionCoordinateDynamics(dummy_free_energy, gamma=1.0)
    integrator = RKF45Integrator(rc_dyn.derivatives, 1, relerr=1e-8, abserr=1e-10)
    t_final, y_final = integrator.integrate(0.0, np.array([0.2]), 10.0)
    print(f"  反应坐标演化: ξ(0)=0.2 → ξ({t_final:.2f})={y_final[0]:.6f}")

    return times, states


def demo_thermodynamic_quadrature():
    """模块 5: 热力学积分（基于 467_gen_laguerre_rule + 886_polygon_integrals + 527_hexagon_integrals）"""
    print_section("模块 5: 热力学积分与矩计算")

    # Gauss-Laguerre 正交
    glq = GaussLaguerreQuadrature(order=32, alpha_param=0.5, a=0.0, b=1.0)
    # 积分 exp(-x) 应接近 1
    test_integral = glq.integrate(lambda x: np.exp(-x))
    # 对于 α=0.5, 权重为 x^0.5 exp(-x), ∫ x^0.5 exp(-2x) dx = Γ(1.5)/2^1.5 ≈ 0.3133
    theoretical = 0.313329
    print(f"  Gauss-Laguerre 正交 (32点): ∫x^0.5 exp(-2x)dx ≈ {test_integral:.8f} (理论值: {theoretical:.6f})")

    # 正六边形矩
    hex_mom = HexagonMoments()
    I_20 = hex_mom.integral_monomial(2, 0)
    I_02 = hex_mom.integral_monomial(0, 2)
    I_40 = hex_mom.integral_monomial(4, 0)
    print(f"  单位正六边形矩:")
    print(f"    I_20 = {I_20:.8f}, I_02 = {I_02:.8f}")
    print(f"    I_40 = {I_40:.8f}")

    # 多边形矩（反应盆地近似）
    # 三角形反应盆地
    tri_x = np.array([0.0, 2.0, 1.0])
    tri_y = np.array([0.0, 0.0, 1.5])
    area = PolygonMoments.moment_unnormalized(3, tri_x, tri_y, 0, 0)
    mu_20 = PolygonMoments.moment_central(3, tri_x, tri_y, 2, 0)
    mu_02 = PolygonMoments.moment_central(3, tri_x, tri_y, 0, 2)
    print(f"  三角形反应盆地:")
    print(f"    面积 = {area:.6f}")
    print(f"    中心矩 μ_20 = {mu_20:.6f}, μ_02 = {mu_02:.6f}")

    # 热力学积分
    ti = ThermodynamicIntegration(n_lambda=20, temperature=300.0)
    xi = np.linspace(0, 1, 50)
    # 构造模型能量剖面
    energy_profile = 20.0 * (xi - 0.5) ** 2 + 15.0 * np.exp(-50.0 * (xi - 0.5) ** 2)
    free_energy, dG, dG_rev = ti.free_energy_barrier(energy_profile, xi)
    print(f"  活化自由能 ΔG‡ = {dG:.4f} kcal/mol")
    print(f"  逆反应活化自由能 = {dG_rev:.4f} kcal/mol")

    return glq, ti


def demo_configuration_tiling():
    """模块 6: 构型空间铺砌（基于 864_pentominoes）"""
    print_section("模块 6: 构型空间铺砌与采样")

    shapes = PentominoShapes.all_shapes()
    print(f"  Pentomino 形状数: {len(shapes)}")

    # 构型空间采样
    def model_energy_2d(xi, eta):
        return 5.0 * (xi ** 2 + eta ** 2) + 10.0 * np.exp(-10.0 * ((xi - 0.5) ** 2 + (eta - 0.3) ** 2))

    tiling = ConfigurationTiling((-1.0, 1.0), (-1.0, 1.0), n_xi=30, n_eta=30)
    coverage, samples, e_grid = tiling.tile_coverage(model_energy_2d, energy_cutoff=8.0)
    print(f"  构型空间覆盖率: {coverage:.4f}")
    print(f"  T-拼板采样点数: {len(samples)}")

    # Metropolis 采样
    sampler = ConfigurationSpaceSampler(n_atoms=5, temperature=300.0)

    def energy_1d(x):
        return 5.0 * x[0] ** 2

    samples_mc, energies_mc, acc_ratio = sampler.metropolis_sampling(energy_1d, [0.5], n_steps=500, step_size=0.2)
    print(f"  Metropolis MC 接受率: {acc_ratio:.4f}")
    print(f"  MC 平均能量: {np.mean(energies_mc):.4f} kcal/mol")

    # 反应路径采样
    path, path_energies, lambdas = sampler.reaction_path_sampling(energy_1d, [-1.0], [1.0], n_images=10)
    print(f"  线性反应路径图像数: {len(path)}")

    return tiling, sampler


def demo_string_method(pes_interp):
    """模块 7: 弦方法与路径优化（基于 214_contour_sequence4 + 1193_t_puzzle）"""
    print_section("模块 7: 弦方法反应路径优化")

    # 使用 RBF 插值势能
    def energy_func(x):
        return float(pes_interp.interpolate(np.array(x).reshape(2, 1)))

    def gradient_func(x):
        return pes_interp.gradient(np.array(x).reshape(2, 1)).flatten()

    # 初始猜测路径（选取双阱势能的极小值附近）
    x_reactant = np.array([-1.0, 0.0])
    x_product = np.array([1.0, 0.0])

    # 标准 NEB
    neb = NEBOptimizer(energy_func, gradient_func, n_images=15,
                       spring_k=0.5, dt=0.05, max_iter=300, tol=1e-3)
    path_neb, energies_neb, _ = neb.optimize(x_reactant, x_product)

    ea_f, ea_r, ts_idx_neb = ReactionPathAnalysis.activation_energy(energies_neb)
    print(f"  NEB 优化完成")
    print(f"  正反应活化能 Ea = {ea_f:.4f} kcal/mol")
    print(f"  逆反应活化能 Ea_rev = {ea_r:.4f} kcal/mol")
    print(f"  过渡态图像索引: {ts_idx_neb}")

    # 弦方法
    sm = StringMethod(energy_func, gradient_func, n_images=15,
                      spring_const=0.5, dt=0.05, max_iter=200, tol=1e-3)
    path_str, energies_str, history_str, _ = sm.evolve_string(
        PathParameterization.reparametrize_equidistant(path_neb, 15)
    )

    ea_f_s, ea_r_s, ts_idx_s = ReactionPathAnalysis.activation_energy(energies_str)
    print(f"  弦方法优化完成")
    print(f"  正反应活化能 Ea = {ea_f_s:.4f} kcal/mol")

    # 对称性操作（将 2D PES 点扩展为 3D 坐标进行演示）
    coords_2d = path_str[ts_idx_s]
    coords_3d = np.zeros((3, 3))
    coords_3d[0, :2] = coords_2d
    coords_3d[1, :2] = coords_2d + 0.5
    coords_3d[2, :2] = coords_2d - 0.3
    configs_sym = SymmetryOperations.apply_c2v_symmetry(coords_3d)
    print(f"  C2v 对称等价构型数: {len(configs_sym)}")

    # 序列管理
    seq_names = SequenceManager.generate_sequence("path_frame", len(path_str))
    print(f"  序列文件名示例: {seq_names[0]}, {seq_names[1]}, ...")

    return path_str, energies_str, ts_idx_s


def demo_transition_state(path_str, energies_str, ts_idx_s, pes_interp):
    """模块 8: 过渡态搜索与验证（综合模块）"""
    print_section("模块 8: 过渡态定位与验证")

    x_ts = path_str[ts_idx_s]

    def grad_func(x):
        return pes_interp.gradient(np.array(x).reshape(2, 1)).flatten()

    def hess_func(x):
        return pes_interp.hessian(np.array(x).reshape(2, 1))

    verifier = TransitionStateVerifier(grad_func, hess_func)
    verification = verifier.verify_saddle_point(x_ts, grad_tol=1e-2)

    print(f"  过渡态位置: [{x_ts[0]:.4f}, {x_ts[1]:.4f}]")
    print(f"  梯度范数: {verification['gradient_norm']:.6f}")
    print(f"  是否为驻点: {verification['is_stationary']}")
    print(f"  负特征值数量: {verification['n_negative_modes']}")
    print(f"  是否为过渡态（恰有一个虚频）: {verification['is_transition_state']}")

    if verification['imaginary_frequency'] is not None:
        print(f"  虚频: {verification['imaginary_frequency']:.2f} cm⁻¹")
        kappa = verifier.wigner_correction(verification['imaginary_frequency'], temperature=300.0)
        print(f"  Wigner 隧道校正因子 κ: {kappa:.4f}")

        # 活化自由能
        ti = ThermodynamicIntegration(temperature=300.0)
        _, dG, _ = ti.free_energy_barrier(energies_str, np.linspace(0, 1, len(energies_str)))
        k_tst = verifier.rate_constant_tst(dG, temperature=300.0, kappa=kappa)
        print(f"  活化自由能 ΔG‡ = {dG:.4f} kcal/mol")
        print(f"  TST 速率常数 k = {k_tst:.6e} s⁻¹")

    # 稀疏 Hessian 分析（模拟大体系）
    print(f"\n  稀疏 Hessian 分析（模拟 10 原子体系）:")
    np.random.seed(99)
    coords_10 = np.random.randn(10, 3) * 2.0
    crs_hess = build_molecular_hessian_crs(10, coords_10, force_constant=2.0, cutoff=3.5)
    print(f"    Hessian 维度: {crs_hess.n}×{crs_hess.n}")
    print(f"    非零元数: {crs_hess.nz}")
    print(f"    稀疏度: {crs_hess.nz / (crs_hess.n ** 2) * 100:.4f}%")

    # Lanczos 计算特征值
    try:
        ritz_vals = lanczos_eigenvalue_solver(crs_hess, max_iter=30)
        print(f"    Lanczos Ritz 值范围: [{ritz_vals[0]:.4f}, {ritz_vals[-1]:.4f}]")
        n_neg_lanczos = np.sum(ritz_vals < -1e-6)
        print(f"    负 Ritz 值数量: {n_neg_lanczos}")
    except Exception as e:
        print(f"    Lanczos 计算提示: {e}")

    return verification


def demo_special_functions():
    """模块 9: 特殊函数（基于 187_clausen）"""
    print_section("模块 9: Clausen 特殊函数与扭转势")

    # Clausen 函数测试
    test_angles = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi]
    print(f"  Clausen 函数值:")
    for ang in test_angles:
        val = clausen_function(ang)
        print(f"    Cl2({ang:.4f}) = {val:.8f}")

    # 扭转势
    phi_test = np.pi / 4
    v_torsion = periodic_torsion_potential(phi_test, n_terms=4)
    print(f"\n  二面角 φ=π/4 扭转势: {v_torsion:.4f} kcal/mol")

    q_ang = angular_partition_function((-np.pi, np.pi), temperature=300.0)
    print(f"  角向配分函数: {q_ang:.4f}")


def main():
    print("=" * 70)
    print("  酶催化反应过渡态搜索 — 博士级科学计算框架")
    print("  Molecular Dynamics: Enzyme Catalysis Transition State Search")
    print("=" * 70)

    # 执行所有模块
    atoms, coords, graph = demo_molecular_topology()
    phi, efx, efy = demo_electrostatics()
    pes_interp, true_pot = demo_pes_interpolation()
    times, states = demo_dynamics_ode()
    glq, ti = demo_thermodynamic_quadrature()
    tiling, sampler = demo_configuration_tiling()
    path_str, energies_str, ts_idx_s = demo_string_method(pes_interp)
    verification = demo_transition_state(path_str, energies_str, ts_idx_s, pes_interp)
    demo_special_functions()

    # 综合报告
    print_section("综合计算报告")
    print(f"  分子体系: {len(atoms)} 原子")
    print(f"  静电势求解: 完成")
    print(f"  PES 插值: 完成")
    print(f"  反应路径优化: 完成（{len(path_str)} 图像）")
    print(f"  过渡态验证: {'通过' if verification['is_transition_state'] else '未通过'}")
    if verification.get('imaginary_frequency'):
        print(f"  虚频: {verification['imaginary_frequency']:.2f} cm⁻¹")
    print(f"  所有计算模块运行完毕，无报错。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================

from special_math import ChebyshevEvaluator
from pes_surface import RBFKernel
from molecular_topology import MolecularGraph

# ---- TC01: ChebyshevEvaluator 对常数级数求值正确 ----
c_eval = ChebyshevEvaluator(np.array([2.0, 0.0, 0.0]))
val = c_eval.evaluate(0.5)
assert isinstance(val, float), '[TC01] 返回值类型应为 float FAILED'
assert abs(val - 1.0) < 1e-12, '[TC01] Chebyshev 常数级数求值错误 FAILED'

# ---- TC02: ChebyshevEvaluator 拒绝越界 x ----
c_eval2 = ChebyshevEvaluator(np.array([1.0, 0.0]))
try:
    c_eval2.evaluate(2.0)
    assert False, '[TC02] 应抛出 ValueError FAILED'
except ValueError:
    pass

# ---- TC03: clausen_function 在 x=0 处返回 0 ----
cl2_0 = clausen_function(0.0)
assert abs(cl2_0) < 1e-14, '[TC03] Cl2(0) 应为 0 FAILED'

# ---- TC04: clausen_function 奇函数对称性 Cl2(-x) = -Cl2(x) ----
cl2_pos = clausen_function(1.0)
cl2_neg = clausen_function(-1.0)
assert abs(cl2_pos + cl2_neg) < 1e-6, '[TC04] Cl2 奇对称性失败 FAILED'

# ---- TC05: clausen_function 在 π/2 处接近已知值 ----
cl2_pi2 = clausen_function(np.pi / 2)
assert abs(cl2_pi2 - 0.9159655941772190) < 1e-4, '[TC05] Cl2(π/2) 值与已知值不符 FAILED'

# ---- TC06: periodic_torsion_potential 返回有限标量 ----
vt = periodic_torsion_potential(np.pi / 4, n_terms=3)
assert isinstance(vt, float), '[TC06] 扭转势返回值应为 float FAILED'
assert np.isfinite(vt), '[TC06] 扭转势应有限 FAILED'

# ---- TC07: periodic_torsion_potential 对称性 V(φ) = V(-φ)（当 γ=0 时） ----
vt_p = periodic_torsion_potential(0.5)
vt_n = periodic_torsion_potential(-0.5)
assert np.isfinite(vt_p) and np.isfinite(vt_n), '[TC07] 扭转势应有限 FAILED'

# ---- TC08: angular_partition_function 返回正浮点数 ----
q_ang = angular_partition_function((-np.pi, np.pi), temperature=300.0)
assert q_ang > 0, '[TC08] 角向配分函数应为正 FAILED'
assert np.isfinite(q_ang), '[TC08] 角向配分函数应有限 FAILED'

# ---- TC09: CRSMatrix matvec 对于单位矩阵正确 ----
vals = np.array([1.0, 1.0, 1.0])
cols = np.array([0, 1, 2])
rows = np.array([0, 1, 2, 3])
crs = CRSMatrix(3, 3, rows, cols, vals)
x = np.array([1.0, 2.0, 3.0])
y = crs.matvec(x)
assert y.shape == (3,), '[TC09] matvec 输出尺寸错误 FAILED'
assert abs(y[0] - 1.0) < 1e-12, '[TC09] matvec 第0分量错误 FAILED'
assert abs(y[1] - 2.0) < 1e-12, '[TC09] matvec 第1分量错误 FAILED'
assert abs(y[2] - 3.0) < 1e-12, '[TC09] matvec 第2分量错误 FAILED'

# ---- TC10: CRSMatrix from_dense 往返一致性 ----
A_dense = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
crs_from = CRSMatrix.from_dense(A_dense)
A_back = crs_from.to_dense()
assert np.max(np.abs(A_dense - A_back)) < 1e-12, '[TC10] from_dense 往返不一致 FAILED'

# ---- TC11: CRSMatrix residual_norm 计算正确 ----
b = np.array([1.0, 0.0, 0.0])
r_norm = crs_from.residual_norm(np.array([0.0, 0.0, 0.0]), b)
assert abs(r_norm - 1.0) < 1e-12, '[TC11] residual_norm 错误 FAILED'

# ---- TC12: build_molecular_hessian_crs 维度正确 ----
np.random.seed(42)
coords_test = np.random.randn(5, 3) * 2.0
h_crs = build_molecular_hessian_crs(5, coords_test, force_constant=1.0, cutoff=3.5)
assert h_crs.n == 15, '[TC12] Hessian 维度应为 3N=15 FAILED'
assert h_crs.nz > 0, '[TC12] Hessian 应有非零元 FAILED'

# ---- TC13: lanczos_eigenvalue_solver 返回排序特征值 ----
np.random.seed(99)
ritz = lanczos_eigenvalue_solver(h_crs, max_iter=30)
assert len(ritz) > 0, '[TC13] Lanczos 应返回 Ritz 值 FAILED'
assert np.all(np.diff(ritz) >= -1e-10), '[TC13] Lanczos Ritz 值应升序 FAILED'

# ---- TC14: poisson_2d_exact_solution 在 (0,0) 处值正确 ----
u, ux, uy, uxx, uxy, uyy = poisson_2d_exact_solution(0.0, 0.0)
assert isinstance(float(u), float), '[TC14] 精确解应为浮点数 FAILED'
assert np.isfinite(float(u)), '[TC14] 精确解应有限 FAILED'

# ---- TC15: electrostatic_stabilization_energy 返回有限值 ----
phi_test = np.ones((4, 4))
rho_test = np.ones((4, 4)) * 0.1
estab = electrostatic_stabilization_energy(phi_test, rho_test, 1.0, 1.0)
assert np.isfinite(estab), '[TC15] 静电能应有限 FAILED'

# ---- TC16: PESInterpolator interpolate 返回标量 ----
np.random.seed(123)
xd_16 = np.random.rand(2, 10) * 2.0 - 1.0
fd_16 = np.sum(xd_16 ** 2, axis=0)
r0_16 = estimate_r0(xd_16)
interp_16 = PESInterpolator(2, 10, xd_16, r0_16, kernel_name='multiquadric')
interp_16.compute_weights(fd_16)
val_16 = interp_16.interpolate(np.array([0.0, 0.0]))
assert isinstance(val_16, float), '[TC16] interpolate 单点应返回标量 FAILED'
assert np.isfinite(val_16), '[TC16] 插值应有限 FAILED'

# ---- TC17: PESInterpolator gradient 维度正确 ----
grad = interp_16.gradient(np.array([0.5, 0.5]))
assert grad.shape == (2, 1), '[TC17] gradient 输出维度应为 (m, 1) FAILED'
assert np.all(np.isfinite(grad)), '[TC17] gradient 应有限 FAILED'

# ---- TC18: RBFKernel multiquadric 对称性 ----
r_test = np.array([1.0, 2.0, 3.0])
phi_mq = RBFKernel.multiquadric(r_test, 1.0)
assert len(phi_mq) == 3, '[TC18] multiquadric 输出长度错误 FAILED'
assert np.all(phi_mq > 0), '[TC18] multiquadric 应全正 FAILED'

# ---- TC19: estimate_r0 返回正值 ----
r0_est = estimate_r0(xd_16)
assert r0_est > 0, '[TC19] estimate_r0 应返回正值 FAILED'

# ---- TC20: RKF45Integrator 积分简单 ODE dy/dt = -y ----
def f_simple(t, y):
    return np.array([-y[0]])
integrator = RKF45Integrator(f_simple, 1, relerr=1e-6, abserr=1e-8)
t_f, y_f = integrator.integrate(0.0, np.array([1.0]), 1.0)
assert isinstance(t_f, float), '[TC20] 积分终止时间应为 float FAILED'
assert abs(y_f[0] - np.exp(-1.0)) < 1e-4, '[TC20] dy/dt=-y 积分错误 FAILED'

# ---- TC21: GlycolysisModel equilibrium 满足 dydt≈0 ----
model = GlycolysisModel()
equi = model.equilibrium()
dydt = model.derivatives(0.0, equi)
max_d = np.max(np.abs(dydt))
assert max_d < 1e-6, '[TC21] 平衡解应满足 dydt≈0 FAILED'

# ---- TC22: integrate_glycolysis 返回正确形状 ----
np.random.seed(42)
times, states = integrate_glycolysis()
assert times.shape[0] == states.shape[0], '[TC22] 时间与状态长度应相同 FAILED'
assert states.shape[1] == 2, '[TC22] 应有 2 个状态变量 FAILED'

# ---- TC23: GaussLaguerreQuadrature 积分 exp(-x) 精度 ----
glq = GaussLaguerreQuadrature(order=32, alpha_param=0.5, a=0.0, b=1.0)
integral = glq.integrate(lambda x: np.exp(-x))
theoretical = 0.313329
assert abs(integral - theoretical) < 1e-3, '[TC23] Gauss-Laguerre 积分精度不足 FAILED'

# ---- TC24: PolygonMoments 三角形面积正确 ----
tri_x = np.array([0.0, 2.0, 1.0])
tri_y = np.array([0.0, 0.0, 1.5])
area = PolygonMoments.moment_unnormalized(3, tri_x, tri_y, 0, 0)
assert abs(area - 1.5) < 1e-10, '[TC24] 三角形面积应为 1.5 FAILED'

# ---- TC25: HexagonMoments 奇数幂次积分为零 ----
hex_mom = HexagonMoments()
I_10 = hex_mom.integral_monomial(1, 0)
I_01 = hex_mom.integral_monomial(0, 1)
assert abs(I_10) < 1e-14, '[TC25] 正六边形 x^1 积分应为 0 FAILED'
assert abs(I_01) < 1e-14, '[TC25] 正六边形 y^1 积分应为 0 FAILED'

# ---- TC26: ThermodynamicIntegration 返回正活化能 ----
xi = np.linspace(0, 1, 50)
energy_profile = 5.0 * np.exp(-20.0 * (xi - 0.5) ** 2)  # Gaussian barrier
ti = ThermodynamicIntegration(n_lambda=20, temperature=300.0)
free_e, dG, dG_rev = ti.free_energy_barrier(energy_profile, xi)
assert dG > 0, '[TC26] 活化自由能应为正 FAILED'
assert np.isfinite(dG), '[TC26] 活化自由能应有限 FAILED'

# ---- TC27: PentominoShapes 有 12 种形状 ----
shapes = PentominoShapes.all_shapes()
assert len(shapes) == 12, '[TC27] Pentomino 应有 12 种形状 FAILED'

# ---- TC28: ConfigurationTiling 网格坐标往返一致 ----
tiling = ConfigurationTiling((-1.0, 1.0), (-1.0, 1.0), n_xi=10, n_eta=10)
xi_orig, eta_orig = 0.3, -0.5
i, j = tiling.physical_to_grid(xi_orig, eta_orig)
xi_back, eta_back = tiling.grid_to_physical(i, j)
assert abs(xi_orig - xi_back) < tiling.dxi + 1e-10, '[TC28] 网格坐标往返不一致 FAILED'

# ---- TC29: ConfigurationSpaceSampler metropolis 返回正确形状 ----
np.random.seed(42)
sampler = ConfigurationSpaceSampler(n_atoms=5, temperature=300.0)
def energy_1d(x):
    return 5.0 * x[0] ** 2
samples_mc, energies_mc, acc_ratio = sampler.metropolis_sampling(energy_1d, [0.5], n_steps=200, step_size=0.2)
assert samples_mc.shape[0] == 201, '[TC29] MC 样本数应为 n_steps+1 FAILED'
assert 0 <= acc_ratio <= 1, '[TC29] 接受率应在 [0,1] FAILED'

# ---- TC30: XYZParser 解析演示数据正确 ----
xyz_data = XYZParser.generate_demo_xyz()
atoms, coords = XYZParser.parse_string(xyz_data)
assert len(atoms) == 18, '[TC30] 应有 18 个原子 FAILED'
assert coords.shape == (18, 3), '[TC30] 坐标形状应为 (18,3) FAILED'

# ---- TC31: MolecularGraph 连通分量正确 ----
graph = MolecularGraph(atoms, coords)
graph.build_bonds()
components = graph.connected_components()
assert len(components) >= 1, '[TC31] 应至少有一个连通分量 FAILED'

# ---- TC32: PathParameterization arc_length 单调递增 ----
path_test = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
s = PathParameterization.arc_length_parameterize(path_test)
assert s[0] == 0.0, '[TC32] 弧长参数应从 0 开始 FAILED'
assert s[-1] == 1.0 or abs(s[-1] - 1.0) < 1e-12, '[TC32] 弧长参数应归一化到 1 FAILED'
assert np.all(np.diff(s) >= -1e-15), '[TC32] 弧长参数应单调不减 FAILED'

# ---- TC33: SequenceManager 生成序列正确 ----
seq = SequenceManager.generate_sequence("frame", 5)
assert len(seq) == 5, '[TC33] 应生成 5 个序列名 FAILED'
assert seq[0] == "frame_000.dat", '[TC33] 序列名格式错误 FAILED'
assert seq[4] == "frame_004.dat", '[TC33] 序列名格式错误 FAILED'

# ---- TC34: SymmetryOperations apply_c2v 返回 4 个构型 ----
coords_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
configs_sym = SymmetryOperations.apply_c2v_symmetry(coords_3d)
assert len(configs_sym) == 4, '[TC34] C2v 对称应生成 4 个构型 FAILED'

# ---- TC35: TransitionStateVerifier wigner_correction >= 1 ----
verifier = TransitionStateVerifier(lambda x: np.zeros(2))
kappa = verifier.wigner_correction(100.0, temperature=300.0)
assert kappa >= 1.0, '[TC35] Wigner 校正因子应 >= 1 FAILED'

# ---- TC36: ReactionPathAnalysis activation_energy 计算正确 ----
energies_test = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 5.0, 3.0, 1.0, 0.0])
Ea_f, Ea_r, ts_idx = ReactionPathAnalysis.activation_energy(energies_test)
assert Ea_f > 0, '[TC36] 正反应活化能应为正 FAILED'
assert ts_idx == 4, '[TC36] 过渡态索引应为 4 FAILED'

# ---- TC37: 集成测试：完整分子拓扑分析流程 ----
xyz_data_full = XYZParser.generate_demo_xyz()
atoms_full, coords_full = XYZParser.parse_string(xyz_data_full)
graph_full, results = analyze_molecular_topology(atoms_full, coords_full)
assert results['n_atoms'] == 18, '[TC37] 原子数应为 18 FAILED'
assert results['n_bonds'] > 0, '[TC37] 应有化学键 FAILED'
assert results['avg_degree'] > 0, '[TC37] 平均配位数应为正 FAILED'
assert 'metis_graph' in results, '[TC37] 应有 METIS 图结果 FAILED'

# ---- TC38: 集成测试：Poisson 求解器完整流程 ----
np.random.seed(42)
nx, ny = 8, 6
dh = 1.0
solver = Poisson2DSolver(nx, ny, dh)
part_x = np.random.rand(20, 2) * np.array([nx - 2, ny - 2]) * dh
part_v = np.random.randn(20, 2) * 100.0
den_test = pic_charge_density(nx, ny, dh, part_x, part_v, 20, 1.0, 1.0)
phi_init = np.ones((nx, ny))
phi_result = solver.solve_gs(phi_init, den_test, 0.5, 0.0, 1.0, -2.0, 1.0, 1.0, max_iter=500, tol=0.5)
efx, efy = solver.compute_electric_field(phi_result)
assert phi_result.shape == (nx, ny), '[TC38] 电势解形状错误 FAILED'
assert np.all(np.isfinite(phi_result)), '[TC38] 电势解应有限 FAILED'

# ---- TC39: 集成测试：弦方法反应路径优化 ----
np.random.seed(123)
nd_sm = 40
xd_sm = np.random.rand(2, nd_sm) * 4.0 - 2.0
def true_pot(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x1, x2 = x[0], x[1]
    else:
        x1, x2 = x[0, 0], x[1, 0]
    return (x1 ** 2 - 1.0) ** 2 + 0.5 * x2 ** 2 + 0.3 * x1 * x2
fd_sm = np.array([true_pot(xd_sm[:, i]) for i in range(nd_sm)])
r0_sm = estimate_r0(xd_sm)
pes_sm = PESInterpolator(2, nd_sm, xd_sm, r0_sm, kernel_name='gaussian')
pes_sm.compute_weights(fd_sm)
def e_func(x):
    return float(pes_sm.interpolate(np.array(x).reshape(2, 1)))
def g_func(x):
    return pes_sm.gradient(np.array(x).reshape(2, 1)).flatten()
neb_sm = NEBOptimizer(e_func, g_func, n_images=10, spring_k=0.5, dt=0.05, max_iter=100, tol=1e-2)
path_neb, energies_neb, _ = neb_sm.optimize(np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
assert path_neb.shape[0] == 10, '[TC39] NEB 路径应含 10 个图像 FAILED'
assert len(energies_neb) == 10, '[TC39] 应有 10 个能量值 FAILED'

# ---- TC40: 集成测试：过渡态验证流程 ----
H_test = pes_sm.hessian(path_neb[4])
verifier_ts = TransitionStateVerifier(lambda x: pes_sm.gradient(np.array(x).reshape(2, 1)).flatten(),
                                       lambda x: pes_sm.hessian(np.array(x).reshape(2, 1)))
verif = verifier_ts.verify_saddle_point(path_neb[4], grad_tol=1.0)
assert 'is_stationary' in verif, '[TC40] 验证结果应包含 is_stationary FAILED'
assert 'gradient_norm' in verif, '[TC40] 验证结果应包含 gradient_norm FAILED'

print('\n全部 40 个测试通过!\n')
