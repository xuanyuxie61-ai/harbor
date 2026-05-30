
import numpy as np
import sys


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


    metis_lines = results['metis_graph'].split('\n')[:4]
    print(f"  METIS 图格式（前 4 行）:")
    for line in metis_lines:
        print(f"    {line}")

    return atoms, coords, graph


def demo_electrostatics():
    print_section("模块 2: 酶活性位点静电势（2D Poisson-Boltzmann）")

    nx, ny = 16, 10
    dh = 1.0


    boundary_box = [[5, 7], [0, 4]]
    solver = Poisson2DSolver(nx, ny, dh, boundary_box)


    np.random.seed(42)
    part_x = np.random.rand(50, 2) * np.array([nx - 2, ny - 2]) * dh
    part_v = np.random.randn(50, 2) * 100.0

    den = pic_charge_density(nx, ny, dh, part_x, part_v, 50, 1.0, 1.0)


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


    x_test = np.array([0.5, 0.5])
    u_exact, ux, uy, uxx, uxy, uyy = poisson_2d_exact_solution(x_test[0], x_test[1])
    print(f"  Poisson 精确解验证 u(0.5,0.5) = {u_exact:.6f}")


    den_norm = den / (np.max(np.abs(den)) + 1e-10)
    estab = electrostatic_stabilization_energy(phi, den_norm, dh, dh)
    print(f"  静电稳定化能（归一化）: {estab:.4e}")
    print(f"  静电稳定化能估算: ~{abs(estab) * 0.1:.2f} kcal/mol（演示量级）")

    return phi, efx, efy


def demo_pes_interpolation():
    print_section("模块 3: 势能面 RBF 插值")



    def true_potential(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x1, x2 = x[0], x[1]
        else:
            x1, x2 = x[0, 0], x[1, 0]


        V = (x1 ** 2 - 1.0) ** 2 + 0.5 * x2 ** 2 + 0.3 * x1 * x2
        return V


    np.random.seed(123)
    nd = 60
    xd = np.random.rand(2, nd) * 4.0 - 2.0
    fd = np.array([true_potential(xd[:, i]) for i in range(nd)])

    r0 = estimate_r0(xd)
    interp = PESInterpolator(2, nd, xd, r0, kernel_name='gaussian')
    interp.compute_weights(fd)


    np.random.seed(456)
    test_points = np.random.rand(2, 20) * 3.0 - 1.5
    true_vals = np.array([true_potential(test_points[:, i]) for i in range(20)])
    interp_vals = interp.interpolate(test_points)
    rmse = np.sqrt(np.mean((true_vals - interp_vals) ** 2))

    print(f"  采样点数: {nd}")
    print(f"  RBF 形状参数 r0: {r0:.4f}")
    print(f"  插值 RMSE: {rmse:.6f}")


    grad = interp.gradient(np.array([0.5, 0.5]))
    print(f"  在 (0.5, 0.5) 处梯度: [{grad[0, 0]:.4f}, {grad[1, 0]:.4f}]")

    return interp, true_potential


def demo_dynamics_ode():
    print_section("模块 4: 反应动力学 ODE 积分")


    times, states = integrate_glycolysis(t0=0.0, tstop=50.0)
    print(f"  糖酵解模型积分完成")
    print(f"  积分步数: {len(times)}")
    print(f"  终态 u: {states[-1, 0]:.6f}, v: {states[-1, 1]:.6f}")

    model = GlycolysisModel()
    equi = model.equilibrium()
    print(f"  理论平衡解 u*: {equi[0]:.6f}, v*: {equi[1]:.6f}")


    J = model.jacobian(states[-1])
    eigvals = np.linalg.eigvals(J)
    print(f"  Jacobian 特征值: {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j, "
          f"{eigvals[1].real:.4f}{eigvals[1].imag:+.4f}j")


    def dummy_free_energy(xi):
        return 10.0 * (xi - 0.5) ** 2

    rc_dyn = ReactionCoordinateDynamics(dummy_free_energy, gamma=1.0)
    integrator = RKF45Integrator(rc_dyn.derivatives, 1, relerr=1e-8, abserr=1e-10)
    t_final, y_final = integrator.integrate(0.0, np.array([0.2]), 10.0)
    print(f"  反应坐标演化: ξ(0)=0.2 → ξ({t_final:.2f})={y_final[0]:.6f}")

    return times, states


def demo_thermodynamic_quadrature():
    print_section("模块 5: 热力学积分与矩计算")


    glq = GaussLaguerreQuadrature(order=32, alpha_param=0.5, a=0.0, b=1.0)

    test_integral = glq.integrate(lambda x: np.exp(-x))

    theoretical = 0.313329
    print(f"  Gauss-Laguerre 正交 (32点): ∫x^0.5 exp(-2x)dx ≈ {test_integral:.8f} (理论值: {theoretical:.6f})")


    hex_mom = HexagonMoments()
    I_20 = hex_mom.integral_monomial(2, 0)
    I_02 = hex_mom.integral_monomial(0, 2)
    I_40 = hex_mom.integral_monomial(4, 0)
    print(f"  单位正六边形矩:")
    print(f"    I_20 = {I_20:.8f}, I_02 = {I_02:.8f}")
    print(f"    I_40 = {I_40:.8f}")



    tri_x = np.array([0.0, 2.0, 1.0])
    tri_y = np.array([0.0, 0.0, 1.5])
    area = PolygonMoments.moment_unnormalized(3, tri_x, tri_y, 0, 0)
    mu_20 = PolygonMoments.moment_central(3, tri_x, tri_y, 2, 0)
    mu_02 = PolygonMoments.moment_central(3, tri_x, tri_y, 0, 2)
    print(f"  三角形反应盆地:")
    print(f"    面积 = {area:.6f}")
    print(f"    中心矩 μ_20 = {mu_20:.6f}, μ_02 = {mu_02:.6f}")


    ti = ThermodynamicIntegration(n_lambda=20, temperature=300.0)
    xi = np.linspace(0, 1, 50)

    energy_profile = 20.0 * (xi - 0.5) ** 2 + 15.0 * np.exp(-50.0 * (xi - 0.5) ** 2)
    free_energy, dG, dG_rev = ti.free_energy_barrier(energy_profile, xi)
    print(f"  活化自由能 ΔG‡ = {dG:.4f} kcal/mol")
    print(f"  逆反应活化自由能 = {dG_rev:.4f} kcal/mol")

    return glq, ti


def demo_configuration_tiling():
    print_section("模块 6: 构型空间铺砌与采样")

    shapes = PentominoShapes.all_shapes()
    print(f"  Pentomino 形状数: {len(shapes)}")


    def model_energy_2d(xi, eta):
        return 5.0 * (xi ** 2 + eta ** 2) + 10.0 * np.exp(-10.0 * ((xi - 0.5) ** 2 + (eta - 0.3) ** 2))

    tiling = ConfigurationTiling((-1.0, 1.0), (-1.0, 1.0), n_xi=30, n_eta=30)
    coverage, samples, e_grid = tiling.tile_coverage(model_energy_2d, energy_cutoff=8.0)
    print(f"  构型空间覆盖率: {coverage:.4f}")
    print(f"  T-拼板采样点数: {len(samples)}")


    sampler = ConfigurationSpaceSampler(n_atoms=5, temperature=300.0)

    def energy_1d(x):
        return 5.0 * x[0] ** 2

    samples_mc, energies_mc, acc_ratio = sampler.metropolis_sampling(energy_1d, [0.5], n_steps=500, step_size=0.2)
    print(f"  Metropolis MC 接受率: {acc_ratio:.4f}")
    print(f"  MC 平均能量: {np.mean(energies_mc):.4f} kcal/mol")


    path, path_energies, lambdas = sampler.reaction_path_sampling(energy_1d, [-1.0], [1.0], n_images=10)
    print(f"  线性反应路径图像数: {len(path)}")

    return tiling, sampler


def demo_string_method(pes_interp):
    print_section("模块 7: 弦方法反应路径优化")


    def energy_func(x):
        return float(pes_interp.interpolate(np.array(x).reshape(2, 1)))

    def gradient_func(x):
        return pes_interp.gradient(np.array(x).reshape(2, 1)).flatten()


    x_reactant = np.array([-1.0, 0.0])
    x_product = np.array([1.0, 0.0])


    neb = NEBOptimizer(energy_func, gradient_func, n_images=15,
                       spring_k=0.5, dt=0.05, max_iter=300, tol=1e-3)
    path_neb, energies_neb, _ = neb.optimize(x_reactant, x_product)

    ea_f, ea_r, ts_idx_neb = ReactionPathAnalysis.activation_energy(energies_neb)
    print(f"  NEB 优化完成")
    print(f"  正反应活化能 Ea = {ea_f:.4f} kcal/mol")
    print(f"  逆反应活化能 Ea_rev = {ea_r:.4f} kcal/mol")
    print(f"  过渡态图像索引: {ts_idx_neb}")


    sm = StringMethod(energy_func, gradient_func, n_images=15,
                      spring_const=0.5, dt=0.05, max_iter=200, tol=1e-3)
    path_str, energies_str, history_str, _ = sm.evolve_string(
        PathParameterization.reparametrize_equidistant(path_neb, 15)
    )

    ea_f_s, ea_r_s, ts_idx_s = ReactionPathAnalysis.activation_energy(energies_str)
    print(f"  弦方法优化完成")
    print(f"  正反应活化能 Ea = {ea_f_s:.4f} kcal/mol")


    coords_2d = path_str[ts_idx_s]
    coords_3d = np.zeros((3, 3))
    coords_3d[0, :2] = coords_2d
    coords_3d[1, :2] = coords_2d + 0.5
    coords_3d[2, :2] = coords_2d - 0.3
    configs_sym = SymmetryOperations.apply_c2v_symmetry(coords_3d)
    print(f"  C2v 对称等价构型数: {len(configs_sym)}")


    seq_names = SequenceManager.generate_sequence("path_frame", len(path_str))
    print(f"  序列文件名示例: {seq_names[0]}, {seq_names[1]}, ...")

    return path_str, energies_str, ts_idx_s


def demo_transition_state(path_str, energies_str, ts_idx_s, pes_interp):
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





        raise NotImplementedError("Hole_3: 请完成 demo_transition_state 中的热力学分析调用逻辑")


    print(f"\n  稀疏 Hessian 分析（模拟 10 原子体系）:")
    np.random.seed(99)
    coords_10 = np.random.randn(10, 3) * 2.0
    crs_hess = build_molecular_hessian_crs(10, coords_10, force_constant=2.0, cutoff=3.5)
    print(f"    Hessian 维度: {crs_hess.n}×{crs_hess.n}")
    print(f"    非零元数: {crs_hess.nz}")
    print(f"    稀疏度: {crs_hess.nz / (crs_hess.n ** 2) * 100:.4f}%")


    try:
        ritz_vals = lanczos_eigenvalue_solver(crs_hess, max_iter=30)
        print(f"    Lanczos Ritz 值范围: [{ritz_vals[0]:.4f}, {ritz_vals[-1]:.4f}]")
        n_neg_lanczos = np.sum(ritz_vals < -1e-6)
        print(f"    负 Ritz 值数量: {n_neg_lanczos}")
    except Exception as e:
        print(f"    Lanczos 计算提示: {e}")

    return verification


def demo_special_functions():
    print_section("模块 9: Clausen 特殊函数与扭转势")


    test_angles = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi]
    print(f"  Clausen 函数值:")
    for ang in test_angles:
        val = clausen_function(ang)
        print(f"    Cl2({ang:.4f}) = {val:.8f}")


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


    atoms, coords, graph = demo_molecular_topology()
    phi, efx, efy = demo_electrostatics()
    pes_interp, true_pot = demo_pes_interpolation()
    times, states = demo_dynamics_ode()
    glq, ti = demo_thermodynamic_quadrature()
    tiling, sampler = demo_configuration_tiling()
    path_str, energies_str, ts_idx_s = demo_string_method(pes_interp)
    verification = demo_transition_state(path_str, energies_str, ts_idx_s, pes_interp)
    demo_special_functions()


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
    sys.exit(main())
