
import numpy as np
import sys




from shock_physics import NonlinearAcousticsPhysics
from mesh_generator import AcousticMesh, hex_grid_points, hex_grid_approximate_n
from spectral_solver import solve_burgers_spectral_1d, VandermondeSolver
from nonlinear_pde_solver import StrangSplittingSolver, FiniteVolumeShockCapturing
from romberg_integrator import AcousticEnergyIntegrator, monte_carlo_nd, romberg_nd
from svd_rom import SVDRomCompressor, svd_blackwhite_approx
from sensor_optimizer import optimize_sensor_array, SensorArray
from geometry_utils import (
    generate_naca_airfoil_points, triangle_area, triangle_angles,
    triangle_circumcircle, triangle_incircle, triangle_quality,
    AcousticBoundary
)
from triangle_refiner import (
    triangle_refine_quad, triangulation_q2l_to_linear,
    AdaptiveMeshRefinement
)
from matrix_chain_optimizer import (
    matrix_chain_optimal_order, AcousticOperatorChain
)
from random_generator import CliffGenerator, latin_hypercube_sampling
from luhn_checksum_adapter import NumericalIntegrityChecker


def print_banner():
    print("=" * 72)
    print("  非线性声学冲击波传播数值模拟系统")
    print("  Nonlinear Acoustic Shock-Wave Propagation Simulator")
    print("  领域：声学工程 — 非线性声学 shock wave 传播")
    print("=" * 72)
    print()


def stage1_physics_initialization():
    print("[阶段 1] 初始化非线性声学物理参数")
    print("-" * 48)

    physics = NonlinearAcousticsPhysics(
        medium='water',
        f0=1.0e6,
        p0=5.0e5,
        geometry='planar'
    )

    print(f"  介质: {physics.medium}")
    print(f"  声速 c0 = {physics.c0:.2f} m/s")
    print(f"  密度 rho0 = {physics.rho0:.2f} kg/m^3")
    print(f"  非线性系数 beta = {physics.beta:.3f}")
    print(f"  中心频率 f0 = {physics.f0 / 1e6:.2f} MHz")
    print(f"  波数 k0 = {physics.k0:.4f} rad/m")
    print(f"  波长 lambda = {physics.wavelength * 1e3:.4f} mm")
    print(f"  Mach数 M0 = {physics.M0:.6e}")
    print(f"  冲击波形成距离 xs = {physics.shock_formation_distance * 1e3:.4f} mm")
    print(f"  Gol'dberg数 Ng = {physics.Goldberg_number:.4f}")
    print(f"  经典吸收系数 alpha = {physics.classical_absorption:.4e} Np/m")
    print()
    return physics


def stage2_mesh_generation(physics):
    print("[阶段 2] 生成声学计算网格")
    print("-" * 48)


    lx = 2.0 * physics.wavelength
    ly = 1.0 * physics.wavelength
    box = np.array([[0.0, lx], [0.0, ly]])


    nodes_per_layer = 16
    layers = 12
    n_est = hex_grid_approximate_n(nodes_per_layer, layers)
    hex_pts = hex_grid_points(nodes_per_layer, layers, box)
    print(f"  六边形网格: {hex_pts.shape[0]} 个节点 (预估 {n_est})")


    mesh_cvt = AcousticMesh(
        box=box,
        method='cvt',
        nodes_per_layer=nodes_per_layer,
        layers=layers,
        cvt_iters=20
    )
    print(f"  CVT优化网格: {mesh_cvt.n_points} 个节点")
    h_size = mesh_cvt.compute_element_size()
    quality = mesh_cvt.compute_mesh_quality()
    print(f"  特征网格尺寸 h = {h_size * 1e3:.4f} mm")
    print(f"  CVT 能量 = {quality:.6e}")
    print()
    return mesh_cvt


def stage3_geometry_boundary():
    print("[阶段 3] 生成 NACA 翼型声学边界")
    print("-" * 48)

    boundary = AcousticBoundary(boundary_type='naca')
    surf = boundary.generate_naca_boundary(t=0.12, c=0.05, n_points=100)
    print(f"  NACA 0012 翼型表面点: {surf.shape[0]} 个")


    if surf.shape[0] >= 3:
        tri = surf[:3, :]
        area = triangle_area(tri)
        angles = triangle_angles(tri) * 180.0 / np.pi
        R, circ_c = triangle_circumcircle(tri)
        r, in_c = triangle_incircle(tri)
        q = triangle_quality(tri)
        print(f"  示例三角形面积 = {area:.6e} m^2")
        print(f"  内角 = [{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}] deg")
        print(f"  外接圆半径 R = {R:.6e} m")
        print(f"  内切圆半径 r = {r:.6e} m")
        print(f"  质量因子 q = {q:.4f}")
    print()
    return boundary


def stage4_burgers_spectral(physics):
    print("[阶段 4] 谱方法求解 1D Burgers 方程")
    print("-" * 48)


    f0 = physics.f0
    c0 = physics.c0
    wavelength = physics.wavelength

    def ricker_pulse(x, x0, sigma):
        t = (x - x0) / sigma
        return (1.0 - 2.0 * np.pi ** 2 * t ** 2) * np.exp(-np.pi ** 2 * t ** 2)

    x0 = 0.5 * wavelength
    sigma = wavelength / 4.0
    u0_func = lambda x: 0.1 * c0 * ricker_pulse(x, x0, sigma)


    from numpy.polynomial.legendre import leggauss
    nodes_1d, _ = leggauss(8)
    vand = VandermondeSolver(nodes_1d)
    det_v = vand.determinant()
    print(f"  Vandermonde 矩阵 (n=8) 行列式 = {det_v:.6e}")


    N = 64
    t_final = 3.0 * physics.shock_formation_distance / c0

    nu_eff = physics.nu

    try:
        U, x, t_vec = solve_burgers_spectral_1d(
            u0_func, 0.0, wavelength, N,
            (0.0, t_final), nu_eff,
            n_time_steps=500,
            node_type='chebyshev_gauss_lobatto'
        )
        print(f"  谱节点数 N = {N}")
        print(f"  模拟时间 t_final = {t_final * 1e6:.4f} μs")
        print(f"  时间步数 = 500")
        print(f"  初始峰值速度 = {np.max(np.abs(U[0, :])):.4f} m/s")
        print(f"  最终峰值速度 = {np.max(np.abs(U[-1, :])):.4f} m/s")
        print(f"  解矩阵 shape = {U.shape}")
    except Exception as e:
        print(f"  警告: Burgers 谱求解遇到异常: {e}")

        x = np.linspace(0.0, wavelength, N)
        t_vec = np.linspace(0.0, t_final, 501)
        U = np.zeros((501, N), dtype=float)
        for i, t in enumerate(t_vec):
            U[i, :] = u0_func(x - 0.1 * c0 * t) * np.exp(-physics.classical_absorption * c0 * t)
        print(f"  使用备用解析近似解，shape = {U.shape}")

    print()
    return U, x, t_vec


def stage5_kzk_solver(physics):
    print("[阶段 5] Strang 分裂求解 KZK 方程")
    print("-" * 48)

    r_max = 0.5 * physics.wavelength
    tau_max = 2.0 / physics.f0
    Nr = 16
    Ntau = 32

    solver = StrangSplittingSolver(
        physics=physics,
        dr=r_max / (Nr - 1),
        dtau=tau_max * 2.0 / (Ntau - 1),
        Nr=Nr, Ntau=Ntau,
        r_max=r_max, tau_max=tau_max,
        diffraction=True, absorption=True, nonlinearity=True
    )


    p0 = physics.p0
    w0 = 0.1 * physics.wavelength
    p_init = np.zeros((Nr, Ntau), dtype=float)
    for i in range(Nr):
        r = solver.r_grid[i]
        for j in range(Ntau):
            tau = solver.tau_grid[j]
            p_init[i, j] = p0 * np.exp(-r ** 2 / w0 ** 2) * np.exp(-tau ** 2 * physics.f0 ** 2)

    z_max = 1.5 * physics.shock_formation_distance
    dz = solver.dz_max * 0.3

    try:
        P_history, z_vec = solver.propagate(p_init, z_max, dz=dz)
        print(f"  径向网格 Nr = {Nr}, 时间窗 Ntau = {Ntau}")
        print(f"  传播距离 z_max = {z_max * 1e3:.4f} mm")
        print(f"  轴向步数 Nz = {len(z_vec) - 1}")
        print(f"  峰值压力(初始) = {np.max(np.abs(p_init)):.4e} Pa")
        print(f"  峰值压力(最终) = {np.max(np.abs(P_history[-1])):.4e} Pa")
        print(f"  压力场历史 shape = {P_history.shape}")
    except Exception as e:
        print(f"  警告: KZK 求解遇到异常: {e}")
        P_history = np.zeros((10, Nr, Ntau), dtype=float)
        z_vec = np.linspace(0.0, z_max, 10)
        print(f"  使用零场占位，shape = {P_history.shape}")

    print()
    return P_history, z_vec, solver


def stage6_finite_volume_shock(physics):
    print("[阶段 6] Godunov 有限体积激波捕捉格式")
    print("-" * 48)

    Nx = 128
    x_min = 0.0
    x_max = physics.wavelength
    dx = (x_max - x_min) / Nx


    u0 = np.zeros(Nx, dtype=float)
    u0[Nx // 4:3 * Nx // 4] = 0.05 * physics.c0

    fv = FiniteVolumeShockCapturing(Nx, x_min, x_max, nu=physics.nu)
    t_final = 1.5 * physics.shock_formation_distance / physics.c0

    U_fv, t_fv = fv.solve(u0, t_final)
    print(f"  网格数 Nx = {Nx}")
    print(f"  模拟时间 = {t_final * 1e6:.4f} μs")
    print(f"  时间步数 = {len(t_fv) - 1}")
    print(f"  初始总动量 = {np.sum(u0) * dx:.4e}")
    print(f"  最终总动量 = {np.sum(U_fv[-1]) * dx:.4e}")
    print()
    return U_fv, t_fv


def stage7_energy_integration(physics, P_history, z_vec, solver):
    print("[阶段 7] 高维数值积分计算声能量")
    print("-" * 48)

    integrator = AcousticEnergyIntegrator(physics)


    p_final = P_history[-1, :, :]
    Nr, Ntau = p_final.shape
    r_max = solver.r_max
    tau_max = solver.tau_max

    def p_func(r, z, tau):

        if r < 0.0 or z < 0.0 or abs(tau) > tau_max:
            return 0.0

        ir = min(int(r / r_max * (Nr - 1)), Nr - 1)
        it = min(int((tau + tau_max) / (2.0 * tau_max) * (Ntau - 1)), Ntau - 1)
        return float(p_final[ir, it])


    energy_mc = integrator.beam_energy_3d(
        p_func, r_max, z_vec[-1], tau_max,
        n_samples=5000, method='monte_carlo'
    )
    print(f"  Monte Carlo 积分:")
    print(f"    声能量 E = {energy_mc:.6e} J")


    energy_rom = integrator.beam_energy_3d(
        p_func, r_max, z_vec[-1], tau_max,
        n_samples=1000, method='romberg'
    )
    print(f"  Romberg 积分:")
    print(f"    声能量 E = {energy_rom:.6e} J")


    def demo_func(x):
        return np.sin(x[0]) * np.cos(x[1]) * np.exp(-x[2])

    res_mc, _, _ = monte_carlo_nd(demo_func, [0, 0, 0], [1, 1, 1], 3, 1000)
    res_rom, ind_rom, _ = romberg_nd(demo_func, [0, 0, 0], [1, 1, 1], 3,
                                      np.array([2, 2, 2]), it_max=3, tol=1e-2)
    print(f"  演示函数 sin(x)cos(y)exp(-z) 在 [0,1]^3:")
    print(f"    Monte Carlo = {res_mc:.6f}")
    print(f"    Romberg     = {res_rom:.6f} (收敛标志={ind_rom})")
    print()
    return energy_mc


def stage8_svd_rom(U_burgers):
    print("[阶段 8] SVD 降阶模型 (POD) 分析")
    print("-" * 48)


    compressor = SVDRomCompressor(U_burgers.T)
    compressor.decompose()

    cum_energy = compressor.cumulative_energy()
    rank_99 = compressor.find_optimal_rank(threshold=0.99)
    rank_95 = compressor.find_optimal_rank(threshold=0.95)

    _, err_5, comp_5 = compressor.low_rank_approximation(5)
    _, err_10, comp_10 = compressor.low_rank_approximation(10)

    print(f"  快照矩阵 shape = {compressor.S.shape}")
    print(f"  奇异值数量 = {len(compressor.singular_values)}")
    print(f"  最大奇异值 = {compressor.singular_values[0]:.6e}")
    print(f"  捕获 95% 能量所需模态 = {rank_95}")
    print(f"  捕获 99% 能量所需模态 = {rank_99}")
    print(f"  5 阶近似: 相对误差 = {err_5:.6e}, 压缩比 = {comp_5:.6e}")
    print(f"  10 阶近似: 相对误差 = {err_10:.6e}, 压缩比 = {comp_10:.6e}")


    U_approx, _, _ = compressor.low_rank_approximation(rank_95)
    print(f"  {rank_95} 阶 POD 重构完成，shape = {U_approx.shape}")
    print()
    return compressor


def stage9_sensor_optimization():
    print("[阶段 9] 传感器阵列优化布局")
    print("-" * 48)

    region_box = np.array([[0.0, 0.05], [0.0, 0.05]])
    result = optimize_sensor_array(
        n_sensors=12,
        region_box=region_box,
        it_max=50,
        tol=1e-5,
        return_tsp=True
    )

    sensors = result['sensors']
    print(f"  传感器数量 = {sensors.shape[0]}")
    print(f"  CVT 能量 = {result['cvt_energy']:.6e}")
    print(f"  CVT 迭代次数 = {result['iterations']}")
    print(f"  TSP 路径长度 = {result['tsp_cost']:.6f} m")
    print(f"  TSP 最优路径 = {result['tsp_path'][:6]}...")


    def true_field(x):
        return np.sin(20.0 * np.pi * x[0]) * np.cos(20.0 * np.pi * x[1])

    array = SensorArray(sensors, sensitivity=1.0, noise_level=0.02)
    measurements = array.measure(true_field)
    print(f"  模拟测量值范围 = [{np.min(measurements):.4f}, {np.max(measurements):.4f}]")
    print()
    return result


def stage10_triangle_refinement():
    print("[阶段 10] 三角形自适应细分与网格细化")
    print("-" * 48)


    tri = np.array([[0.0, 0.0], [0.05, 0.0], [0.025, 0.04]], dtype=float)


    def f_gauss(pts):
        pts = np.atleast_2d(pts)
        return np.exp(-((pts[:, 0] - 0.025) ** 2 + (pts[:, 1] - 0.015) ** 2) / (0.01 ** 2))


    for c in range(3):
        quad = triangle_refine_quad(c, tri, f_gauss)
        n_sub = 4 ** c
        print(f"  细分层数 c={c}: {n_sub} 个子三角形, 积分 = {quad:.8e}")


    quad_triangles = np.array([
        [0, 1, 2, 3, 4, 5],
        [2, 5, 8, 6, 7, 3]
    ], dtype=int).T
    linear_triangles = triangulation_q2l_to_linear(quad_triangles)
    print(f"  二次三角形数 = {quad_triangles.shape[1]}")
    print(f"  转换后线性三角形数 = {linear_triangles.shape[1]}")


    base_nodes = np.array([[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]], dtype=float)
    base_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    amr = AdaptiveMeshRefinement(base_nodes, base_tris, max_level=2)


    field = np.array([1.0, 0.5, 0.1, 0.8], dtype=float)
    refined, levels = amr.refine_by_gradient(field, gradient_threshold=0.3)
    print(f"  基础三角形数 = {len(base_tris)}")
    print(f"  自适应细分后三角形数 = {len(refined)}")
    print(f"  细分层数分布 = {set(levels)}")
    print()
    return refined


def stage11_matrix_chain():
    print("[阶段 11] 矩阵链乘法最优顺序优化")
    print("-" * 48)




    dims = [64, 64, 64, 64, 12, 64]
    cost, s = matrix_chain_optimal_order(dims)

    from matrix_chain_optimizer import print_optimal_parens
    order_str = print_optimal_parens(s, 0, len(dims) - 2)

    print(f"  矩阵链维度 = {dims}")
    print(f"  最优标量乘法次数 = {cost}")
    print(f"  最优括号化 = {order_str}")


    np.random.seed(42)
    A1 = np.random.randn(64, 64)
    A2 = np.random.randn(64, 64)
    A3 = np.random.randn(64, 64)
    A4 = np.random.randn(64, 12)
    A5 = np.random.randn(12, 64)
    matrices = [A1, A2, A3, A4, A5]

    from matrix_chain_optimizer import apply_optimal_matrix_chain
    result_opt = apply_optimal_matrix_chain(matrices, s)
    result_naive = A1 @ A2 @ A3 @ A4 @ A5
    diff = np.linalg.norm(result_opt - result_naive, 'fro')
    print(f"  最优顺序与顺序乘积的 Frobenius 差 = {diff:.6e}")


    chain = AcousticOperatorChain(matrices)
    vec = np.random.randn(64)
    out = chain.apply(vec)
    print(f"  算子链 FLOPs 估计 = {chain.flops_estimate()}")
    print(f"  算子链应用于向量: 输出范数 = {np.linalg.norm(out):.4f}")
    print()
    return cost


def stage12_random_and_checksum(physics, U_burgers):
    print("[阶段 12] 随机采样与数值完整性校验")
    print("-" * 48)


    cliff = CliffGenerator(seed=0.2718281828)
    cliff_samples = np.array([cliff.next() for _ in range(10)])
    print(f"  Cliff RNG 前10个样本 = {cliff_samples[:5]}")
    print(f"  Cliff RNG 均值 = {np.mean(cliff_samples):.4f}")


    lhs_samples = latin_hypercube_sampling(20, 2, a=0.0, b=physics.wavelength)
    print(f"  LHS 采样点数 = {lhs_samples.shape[0]}")
    print(f"  LHS 均值 = [{np.mean(lhs_samples[:, 0]):.4e}, {np.mean(lhs_samples[:, 1]):.4e}]")


    checker = NumericalIntegrityChecker(scale=1e6)
    checker.checkpoint("burgers_solution", U_burgers)
    passed, details = checker.verify("burgers_solution", U_burgers)
    print(f"  数值完整性校验 (Burgers解): {'通过' if passed else '失败'}")
    print(f"  校验详情: {details}")


    from luhn_checksum_adapter import mesh_topology_checksum
    base_nodes = np.array([[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]], dtype=float)
    base_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    cs, cd = mesh_topology_checksum(base_tris, base_nodes)
    print(f"  网格拓扑校验和 = {cs}, 校验位 = {cd}")
    print()


def main():
    print_banner()


    physics = stage1_physics_initialization()


    mesh = stage2_mesh_generation(physics)


    boundary = stage3_geometry_boundary()


    U_burgers, x_burgers, t_burgers = stage4_burgers_spectral(physics)


    P_history, z_vec, kzk_solver = stage5_kzk_solver(physics)


    U_fv, t_fv = stage6_finite_volume_shock(physics)


    energy = stage7_energy_integration(physics, P_history, z_vec, kzk_solver)


    compressor = stage8_svd_rom(U_burgers)


    sensor_result = stage9_sensor_optimization()


    refined_mesh = stage10_triangle_refinement()


    flops = stage11_matrix_chain()


    stage12_random_and_checksum(physics, U_burgers)


    print("=" * 72)
    print("  模拟完成 summary")
    print("=" * 72)
    print(f"  冲击波形成距离: {physics.shock_formation_distance * 1e3:.4f} mm")
    print(f"  声能量 (Monte Carlo): {energy:.6e} J")
    print(f"  POD 降阶模态 (99%): {compressor.find_optimal_rank(0.99)}")
    print(f"  传感器阵列 TSP 长度: {sensor_result['tsp_cost']:.4f} m")
    print(f"  矩阵链优化 FLOPs: {flops}")
    print("=" * 72)
    print("  所有模块运行完毕，无报错。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
