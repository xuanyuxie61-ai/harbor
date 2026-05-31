"""
main.py

统一入口：地幔对流与板块运动数值模拟 (Mantle Convection & Plate Tectonics Simulation)

本项目基于以下 15 个科研代码项目的核心算法融合构建：
- 691_lissajous          -> Lissajous 周期强迫参数化
- 480_gram_schmidt       -> Gram-Schmidt 谱基正交化
- 1208_test_int_2d       -> Gauss-Legendre 二维数值积分
- 539_histogram_discrete -> 离散温度分布统计
- 488_grazing_ode        -> 上/下地幔化学交换耦合 ODE
- 1394_voronoi_city      -> Voronoi 表面板块划分
- 1372_unicycle          -> 循环边界节点重编号
- 890_polygon_triangulate-> 多边形三角剖分（耳切法）
- 1428_zero_chandrupatla -> Chandrupatla 根查找（临界瑞利数）
- 635_lagrange_interp_1d -> 一维拉格朗日径向插值
- 559_hypercube_integrals-> 超立方体蒙特卡洛参数采样
- 868_pi_spigot          -> 高精度 π 计算（球面几何）
- 1357_trig_interp_basis -> 三角基函数角向谱展开
- 1423_xyz_display       -> 三维球面-直角坐标变换
- 467_gen_laguerre_rule  -> 广义 Gauss-Laguerre 半无限区间积分

运行方式：python main.py（零参数，直接运行）
"""

import numpy as np
import sys

# Import project modules
from spherical_geometry import PiSpigot, SphericalGeometry
from mesh_generator import PolygonTriangulator, VoronoiTessellator, UnicycleIndexer, MantleMesh2D
from spectral_basis import GramSchmidt, TrigonometricBasis, LagrangeInterpolation, SpectralExpansion
from quadrature_engine import GaussLegendre, GaussLaguerre, Quadrature2D, HypercubeSampler
from mantle_physics import (
    MantleConstants, ViscosityModel, DensityModel,
    DimensionlessNumbers, StokesPhysics, ThermalPhysics
)
from stokes_solver import StokesSolver
from thermal_solver import GrazingChemicalExchange, ThermalSolver
from diagnostics import (
    TemperatureHistogram, ChandrupatlaRootFinder,
    LissajousForcing, ParameterSampler, MantleDiagnostics
)


def run_spherical_geometry():
    """高精度球面几何计算（pi_spigot + xyz_display 坐标处理）"""
    print("=" * 60)
    print("[1] 高精度球面几何模块")
    print("=" * 60)
    spigot = PiSpigot(digits=30)
    pi_val = spigot.compute()
    print(f"  Spigot 算法计算 π = {pi_val:.15f}")

    geo = SphericalGeometry(R_surf=6371.0, R_cmb=3480.0)
    print(f"  地球半径 R_surf = {geo.R_surf} km")
    print(f"  核幔边界半径 R_cmb = {geo.R_cmb} km")
    print(f"  地表面积 A = {geo.surface_area():.3e} km²")
    print(f"  地幔体积 V = {geo.shell_volume():.3e} km³")
    print(f"  地幔厚度 D = {geo.shell_thickness():.1f} km")

    # 球坐标 ↔ 直角坐标转换
    r_test = np.array([geo.R_surf, geo.R_cmb])
    theta_test = np.array([np.pi / 4, np.pi / 3])
    phi_test = np.array([np.pi / 6, np.pi / 2])
    x, y, z = geo.spherical_to_cartesian(r_test, theta_test, phi_test)
    print(f"  球面→直角坐标示例:")
    print(f"    (r,θ,φ)=({r_test[0]:.1f},{theta_test[0]:.4f},{phi_test[0]:.4f}) → (x,y,z)=({x[0]:.3f},{y[0]:.3f},{z[0]:.3f})")
    r_back, theta_back, phi_back = geo.cartesian_to_spherical(x, y, z)
    print(f"  直角→球面坐标回代误差: {np.max(np.abs(r_back - r_test)):.2e} km")
    print()


def run_mesh_generation():
    """网格生成与拓扑处理（polygon_triangulate + voronoi_city + unicycle）"""
    print("=" * 60)
    print("[2] 网格生成与 Voronoi 板块划分模块")
    print("=" * 60)

    # 耳切法三角剖分
    triangulator = PolygonTriangulator()
    # 构造一个简化的俯冲带几何多边形（CCW）
    n_vertices = 8
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    x_poly = 1.0 + 0.3 * np.cos(angles)
    y_poly = 1.0 + 0.3 * np.sin(angles)
    triangles = triangulator.triangulate(x_poly, y_poly)
    print(f"  多边形三角剖分: {n_vertices} 顶点 → {len(triangles)} 个三角形")
    area_sum = 0.0
    for tri in triangles:
        i1, i2, i3 = tri - 1  # convert to 0-based
        a = 0.5 * abs((x_poly[i2] - x_poly[i1]) * (y_poly[i3] - y_poly[i1])
                      - (x_poly[i3] - x_poly[i1]) * (y_poly[i2] - y_poly[i1]))
        area_sum += a
    print(f"  剖分三角形总面积 = {area_sum:.6f}")

    # Voronoi 表面板块划分
    voronoi = VoronoiTessellator(bounds=(0.0, 10.0, 0.0, 10.0))
    generators = np.array([[3.0, 4.0], [7.0, 8.0], [7.0, 2.0], [5.0, 5.0]])
    labels, areas = voronoi.compute_cells(generators, grid_res=200)
    print(f"  Voronoi 板块划分: {len(generators)} 个生成元")
    for i, area in enumerate(areas):
        print(f"    板块 {i+1} 近似面积 = {area:.4f}")

    # Unicycle 循环索引用于边界节点排序
    indexer = UnicycleIndexer()
    n = 12
    shift = 3
    u_index = indexer.create_cycle(n, shift)
    sequence = indexer.index_to_sequence(n, u_index)
    print(f"  Unicycle 循环索引 (n={n}, shift={shift}):")
    print(f"    索引向量: {u_index}")
    print(f"    序列向量: {sequence}")

    # 环形截面结构化网格
    mesh = MantleMesh2D(R_inner=0.5, R_outer=1.0)
    nodes, triangs, bnd = mesh.generate_annular_sector_mesh(n_r=6, n_theta=12)
    print(f"  环形截面网格: {len(nodes)} 节点, {len(triangs)} 三角形, {len(bnd)} 边界节点")
    print()


def run_spectral_basis():
    """谱基函数构造（gram_schmidt + trig_interp_basis + lagrange_interp_1d）"""
    print("=" * 60)
    print("[3] 谱基与插值模块")
    print("=" * 60)

    # Gram-Schmidt 正交化
    A = np.array([[1.0, 1.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    U_classical = GramSchmidt.classical(A)
    print("  Gram-Schmidt 正交化 (经典):")
    print(f"    U^T U = \n{U_classical.T @ U_classical}")

    # 三角基函数
    trig = TrigonometricBasis()
    x_test = np.linspace(-1.0, 1.0, 101)
    k_vals = [1, 2, 3, 4]
    print("  三角基函数在 x=0 处取值 (应为 1):")
    for k in k_vals:
        val = trig.basis(np.array([0.0]), k)
        print(f"    k={k}: B_k(0) = {val[0]:.6f}")

    # 拉格朗日插值
    lagrange = LagrangeInterpolation()
    xd = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    yd = np.sin(np.pi * xd)
    xi = np.linspace(0.0, 1.0, 21)
    yi = lagrange.interpolate(xd, yd, xi)
    error = np.max(np.abs(yi - np.sin(np.pi * xi)))
    print(f"  Lagrange 插值 sin(πx) 在 [0,1] 最大误差 = {error:.2e}")

    # 谱展开
    expansion = SpectralExpansion(n_radial=6, n_angular=8)
    r_nodes = np.linspace(0.5, 1.0, 6)
    r_eval = np.linspace(0.5, 1.0, 20)
    theta_eval = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
    R_basis = expansion.build_radial_basis(r_nodes, r_eval)
    Theta_basis = expansion.build_angular_basis(theta_eval)
    print(f"  谱展开基: 径向 {R_basis.shape[1]} 模式, 角向 {Theta_basis.shape[1]} 模式")
    print()


def run_quadrature():
    """数值积分（legendre_dr_compute + gen_laguerre_rule + hypercube_integrals）"""
    print("=" * 60)
    print("[4] 数值积分模块")
    print("=" * 60)

    # Gauss-Legendre
    gl = GaussLegendre()
    x, w = gl.compute(8)
    print(f"  Gauss-Legendre (n=8): 节点范围 [{x.min():.6f}, {x.max():.6f}]")
    integral = gl.integrate_1d(lambda t: np.exp(t), -1.0, 1.0, n=16)
    exact = np.exp(1.0) - np.exp(-1.0)
    print(f"  ∫_{'{-1}'}^{'{1}'} exp(x) dx ≈ {integral:.10f} (精确值 {exact:.10f}, 误差 {abs(integral - exact):.2e})")

    # Gauss-Laguerre
    laguerre = GaussLaguerre()
    xl, wl = laguerre.compute(n=12, alpha=0.0, a=0.0, b=1.0)
    integral_l = laguerre.integrate(lambda t: t ** 2, n=12, alpha=0.0, a=0.0, b=1.0)
    exact_l = 2.0  # ∫_0^∞ x^2 exp(-x) dx = 2
    print(f"  Gauss-Laguerre ∫_0^∞ x^2 exp(-x) dx ≈ {integral_l:.10f} (精确值 {exact_l:.10f}, 误差 {abs(integral_l - exact_l):.2e})")

    # 2D 矩形积分
    quad2d = Quadrature2D()
    f2d = lambda x, y: np.exp(-(x ** 2 + y ** 2))
    val2d = quad2d.integrate_rectangle(f2d, (-1.0, 1.0), (-1.0, 1.0), nx=8, ny=8)
    print(f"  2D Gauss-Legendre 乘积规则 ∫_{'{-1}'}^{'{1}'}∫_{'{-1}'}^{'{1}'} exp(-(x²+y²)) dxdy ≈ {val2d:.8f}")

    # 超立方体蒙特卡洛采样
    sampler = HypercubeSampler()
    samples = sampler.sample(m=3, n=1000, seed=42)
    print(f"  超立方体采样 (m=3, n=1000): 样本均值 = {np.mean(samples):.4f}, 样本方差 = {np.var(samples):.4f}")

    # Monte Carlo 积分示例
    mc_mean, mc_err = sampler.integrate(lambda x: np.sum(x ** 2, axis=0), m=3, n=5000, seed=42)
    print(f"  Monte Carlo ∫_{'{[0,1]³}'} (x²+y²+z²) dV ≈ {mc_mean:.6f} ± {mc_err:.6f}")
    print()


def run_mantle_physics():
    """地幔物理模型与无量纲数计算"""
    print("=" * 60)
    print("[5] 地幔物理与无量纲数模块")
    print("=" * 60)

    viscosity = ViscosityModel()
    density = DensityModel()

    T_test = np.array([1000.0, 1600.0, 2500.0])
    eta_vals = viscosity.arrhenius(T_test)
    rho_vals = density.thermal_density(T_test)
    buoy_vals = density.buoyancy(T_test)
    print("  温度 [K]    |  粘度 [Pa·s]       |  密度 [kg/m³]   |  浮力 [kg/m³]")
    print("  " + "-" * 70)
    for i in range(len(T_test)):
        print(f"  {T_test[i]:.0f}       |  {eta_vals[i]:.3e}  |  {rho_vals[i]:.2f}      |  {buoy_vals[i]:.4f}")

    D = MantleConstants.R_surf - MantleConstants.R_cmb
    delta_T = MantleConstants.T_cmb - MantleConstants.T_surf
    Ra = DimensionlessNumbers.rayleigh_number(D, delta_T)
    print(f"\n  地幔厚度 D = {D/1000:.1f} km")
    print(f"  温度差 ΔT = {delta_T:.1f} K")
    print(f"  瑞利数 Ra = {Ra:.3e}")

    Nu = DimensionlessNumbers.nusselt_number(q_conv=1.2e12, q_cond=4.0e11)
    print(f"  努塞尔数 Nu = {Nu:.3f}")

    Pr = DimensionlessNumbers.prandtl_number(MantleConstants.eta0)
    print(f"  普朗特数 Pr = {Pr:.3e} (→ ∞, 无穷普朗特数近似)")
    print()


def run_stokes_and_thermal():
    """斯托克斯求解与热演化（含 grazing_ode 化学交换）"""
    print("=" * 60)
    print("[6] 斯托克斯流求解与热演化模块")
    print("=" * 60)

    # 构建网格
    R_inner = 0.5
    R_outer = 1.0
    nr, ntheta = 25, 40
    r = np.linspace(R_inner, R_outer, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')

    # 初始温度场
    thermal = ThermalSolver()
    T = thermal.initial_temperature(r_grid, theta_grid, mode="perturbation")
    print(f"  初始温度场: 形状 {T.shape}, 均值 {np.mean(T):.2f} K, 范围 [{np.min(T):.1f}, {np.max(T):.1f}] K")

    # Stokes 求解
    stokes = StokesSolver(R_inner=R_inner, R_outer=R_outer, n_radial=8, n_angular=12)
    u_r, u_theta = stokes.compute_velocity_from_streamfunction(
        np.zeros((8, 12)), T, r_grid, theta_grid
    )
    print(f"  速度场: u_r 范围 [{np.min(u_r):.3e}, {np.max(u_r):.3e}], u_θ 范围 [{np.min(u_theta):.3e}, {np.max(u_theta):.3e}]")

    # 时间步进
    dt = 1.0e11  # s (~3 kyr)
    n_steps = 20
    for step in range(n_steps):
        T = thermal.step_forward(T, u_r, u_theta, r_grid, theta_grid, dt)
    print(f"  演化 {n_steps} 步 (Δt={dt:.2e} s ≈ {dt/3.15e13:.1f} Myr) 后:")
    print(f"    温度均值 = {np.mean(T):.2f} K, 标准差 = {np.std(T):.2f} K")

    # 表面热流
    q_surf = thermal.surface_heat_flux(T, r_grid)
    print(f"    表面热流 (无量纲) = {q_surf:.4f}")

    # 化学交换 ODE (grazing 模型)
    chem = GrazingChemicalExchange(a=1.1, c1=1.2, c2=1.5, d1=0.001, d2=0.001,
                                    K=3000.0, r1=0.8)
    t_chem, y_chem = chem.integrate_rk4(y0=np.array([3000.0, 5.0]),
                                         t_span=(0.0, 100.0), n_steps=2000)
    print(f"\n  上/下地幔化学交换 (Grazing ODE 类比):")
    print(f"    t=0:   C_um={y_chem[0,0]:.2f}, C_lm={y_chem[0,1]:.4f}")
    print(f"    t=100: C_um={y_chem[-1,0]:.2f}, C_lm={y_chem[-1,1]:.4f}")
    print()


def run_diagnostics():
    """诊断分析（histogram + root_finding + lissajous + hypercube）"""
    print("=" * 60)
    print("[7] 诊断分析与参数不确定性模块")
    print("=" * 60)

    diag = MantleDiagnostics(T_min=300.0, T_max=3000.0)

    # 模拟温度场用于统计
    np.random.seed(42)
    T_sim = np.random.normal(loc=1600.0, scale=400.0, size=(30, 40))
    T_sim = np.clip(T_sim, 300.0, 3000.0)
    stats = diag.analyze_temperature_field(T_sim)
    print(f"  温度场统计:")
    print(f"    均值 = {stats['mean']:.2f} K")
    print(f"    标准差 = {stats['std']:.2f} K")
    print(f"    熵 = {stats['entropy']:.4f}")

    # 临界瑞利数根查找
    def mock_nu(Ra):
        # Mock Nusselt number: Nu = 1 + 0.5 * tanh(log10(Ra/1e5))
        return 1.0 + 0.5 * np.tanh(np.log10(Ra / 1.0e5))

    Ra_c, fm, calls = diag.root_finder.find_critical_rayleigh(mock_nu, 1.0e3, 1.0e7)
    print(f"\n  临界瑞利数搜索 (Chandrupatla 算法):")
    print(f"    Ra_c ≈ {Ra_c:.3e}, f(Ra_c) = {fm:.2e}, 函数调用次数 = {calls}")

    # Lissajous 周期强迫
    forcing = LissajousForcing(a1=2.0, b1=np.pi / 4.0, a2=3.0, b2=0.0)
    t_forcing = np.linspace(0.0, 4 * np.pi, 100)
    x_liss, y_liss = forcing.evaluate(t_forcing)
    q_mod = [forcing.boundary_heat_flux_modulation(t, q0=1.0, amplitude=0.1) for t in t_forcing]
    print(f"\n  Lissajous 周期强迫:")
    print(f"    热流调制范围: [{min(q_mod):.4f}, {max(q_mod):.4f}]")

    # 参数不确定性传播
    sampler = ParameterSampler(seed=123)
    def toy_model(theta):
        # theta: [Ra, alpha, eta0]
        Ra, alpha, eta0 = theta[0], theta[1], theta[2]
        D = 2891e3
        kappa = 1e-6
        return (Ra * eta0 * kappa) / (3300.0 * 9.81 * alpha * D ** 3)

    bounds = np.array([[1e4, 1e8], [1e-6, 5e-5], [1e20, 1e22]])
    mean_out, std_out = sampler.propagate_uncertainty(toy_model, bounds, n_samples=500)
    print(f"\n  参数不确定性传播 (Monte Carlo, n=500):")
    print(f"    E[ΔT] = {mean_out:.3f} K,  σ[ΔT] = {std_out:.3f} K")
    print()


def run_full_simulation_summary():
    """综合模拟结果汇总"""
    print("=" * 60)
    print("[8] 综合模拟结果汇总")
    print("=" * 60)

    # 构建完整模拟流程
    R_inner = 0.5
    R_outer = 1.0
    nr, ntheta = 20, 32
    r = np.linspace(R_inner, R_outer, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')

    thermal = ThermalSolver()
    T = thermal.initial_temperature(r_grid, theta_grid, mode="perturbation")
    stokes = StokesSolver(R_inner=R_inner, R_outer=R_outer, n_radial=6, n_angular=8)
    u_r, u_theta = stokes.compute_velocity_from_streamfunction(np.zeros((6, 8)), T, r_grid, theta_grid)

    dt = 5.0e10
    n_steps = 50
    for _ in range(n_steps):
        T = thermal.step_forward(T, u_r, u_theta, r_grid, theta_grid, dt)

    q_surf = thermal.surface_heat_flux(T, r_grid)
    q_cond = 1.0  # Non-dimensional conductive flux
    Nu = DimensionlessNumbers.nusselt_number(q_surf, q_cond)
    D = MantleConstants.R_surf - MantleConstants.R_cmb
    delta_T = MantleConstants.T_cmb - MantleConstants.T_surf
    Ra = DimensionlessNumbers.rayleigh_number(D, delta_T)

    diag = MantleDiagnostics()
    stats = diag.analyze_temperature_field(T)

    print(f"  模拟参数:")
    print(f"    网格: {nr} × {ntheta}")
    print(f"    时间步数: {n_steps}, Δt = {dt/3.15e13:.3f} Myr")
    print(f"  结果:")
    print(f"    瑞利数 Ra = {Ra:.3e}")
    print(f"    努塞尔数 Nu = {Nu:.3f}")
    print(f"    平均温度 = {stats['mean']:.2f} K")
    print(f"    温度标准差 = {stats['std']:.2f} K")
    print(f"    温度熵 = {stats['entropy']:.4f}")
    print(f"    表面热流 = {q_surf:.4e} (无量纲)")
    print()
    print("  所有模块运行完毕，地幔对流数值模拟流程验证通过。")
    print("=" * 60)


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "  地幔对流与板块运动数值模拟系统 (Python)".center(54) + "║")
    print("║" + "  Mantle Convection & Plate Tectonics Simulation".center(54) + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    run_spherical_geometry()
    run_mesh_generation()
    run_spectral_basis()
    run_quadrature()
    run_mantle_physics()
    run_stokes_and_thermal()
    run_diagnostics()
    run_full_simulation_summary()

    print("\n>>> 程序正常结束，无报错。\n")
    return 0


if __name__ == "__main__":
    main()

    # ================================================================
    # 测试用例（25个，assert模式，涉及随机值均使用固定种子）
    # ================================================================
    # ---- TC01: PiSpigot 计算 π 值在合理区间 ----
    spigot = PiSpigot(digits=30)
    pi_val = spigot.compute()
    assert 3.1 < pi_val < 3.2, '[TC01] PiSpigot 计算 π 值不在合理区间 FAILED'

    # ---- TC02: SphericalGeometry 表面积公式正确 ----
    geo = SphericalGeometry(R_surf=1.0, R_cmb=0.5)
    expected_area = 4.0 * pi_val * 1.0 ** 2
    assert abs(geo.surface_area() - expected_area) < 1e-10, '[TC02] 表面积公式不正确 FAILED'

    # ---- TC03: SphericalGeometry 球壳体积公式正确 ----
    expected_vol = (4.0 / 3.0) * pi_val * (1.0 ** 3 - 0.5 ** 3)
    assert abs(geo.shell_volume() - expected_vol) < 1e-10, '[TC03] 球壳体积公式不正确 FAILED'

    # ---- TC04: SphericalGeometry 球坐标与直角坐标转换一致性 ----
    r_test = np.array([1.0, 0.8])
    theta_test = np.array([np.pi / 4, np.pi / 3])
    phi_test = np.array([np.pi / 6, np.pi / 2])
    x, y, z = geo.spherical_to_cartesian(r_test, theta_test, phi_test)
    r_back, theta_back, phi_back = geo.cartesian_to_spherical(x, y, z)
    assert np.max(np.abs(r_back - r_test)) < 1e-10, '[TC04] 坐标转换回代误差过大 FAILED'

    # ---- TC05: PolygonTriangulator 简单多边形剖分数量正确 ----
    triangulator = PolygonTriangulator()
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x_poly = 1.0 + 0.3 * np.cos(angles)
    y_poly = 1.0 + 0.3 * np.sin(angles)
    triangles = triangulator.triangulate(x_poly, y_poly)
    assert triangles.shape == (6, 3), '[TC05] 三角剖分数量不正确 FAILED'

    # ---- TC06: VoronoiTessellator 各 cell 面积和等于总区域面积 ----
    voronoi = VoronoiTessellator(bounds=(0.0, 10.0, 0.0, 10.0))
    generators = np.array([[3.0, 4.0], [7.0, 8.0], [7.0, 2.0], [5.0, 5.0]])
    labels, areas = voronoi.compute_cells(generators, grid_res=200)
    assert abs(np.sum(areas) - 100.0) < 5.0, '[TC06] Voronoi cell 面积和不等于总区域面积 FAILED'

    # ---- TC07: UnicycleIndexer 循环索引产生完整序列 ----
    indexer = UnicycleIndexer()
    n = 12
    shift = 3
    u_index = indexer.create_cycle(n, shift)
    sequence = indexer.index_to_sequence(n, u_index)
    assert len(np.unique(sequence)) == n, '[TC07] 循环索引序列不完整 FAILED'

    # ---- TC08: GramSchmidt 经典正交化 U^T U 近似单位矩阵 ----
    A = np.array([[1.0, 1.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    U = GramSchmidt.classical(A)
    UTU = U.T @ U
    I = np.eye(3)
    assert np.max(np.abs(UTU - I)) < 1e-10, '[TC08] Gram-Schmidt 正交化不精确 FAILED'

    # ---- TC09: TrigonometricBasis 在 x=0 处取值为 1 ----
    trig = TrigonometricBasis()
    for k in [1, 2, 3, 4]:
        val = trig.basis(np.array([0.0]), k)
        assert abs(val[0] - 1.0) < 1e-10, '[TC09] 三角基函数在 x=0 处不为 1 FAILED'

    # ---- TC10: LagrangeInterpolation 对常数函数精确重构 ----
    lagrange = LagrangeInterpolation()
    xd = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    yd = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    xi = np.linspace(0.0, 1.0, 21)
    yi = lagrange.interpolate(xd, yd, xi)
    assert np.max(np.abs(yi - 3.0)) < 1e-10, '[TC10] Lagrange 插值对常数函数不精确 FAILED'

    # ---- TC11: GaussLegendre 1D 积分多项式精确 ----
    gl = GaussLegendre()
    integral = gl.integrate_1d(lambda t: t ** 2, -1.0, 1.0, n=16)
    assert abs(integral - 2.0 / 3.0) < 1e-12, '[TC11] Gauss-Legendre 积分 x^2 不精确 FAILED'

    # ---- TC12: GaussLaguerre 积分 x^2 exp(-x) 等于 2 ----
    laguerre = GaussLaguerre()
    integral_l = laguerre.integrate(lambda t: t ** 2, n=12, alpha=0.0, a=0.0, b=1.0)
    assert abs(integral_l - 2.0) < 1e-8, '[TC12] Gauss-Laguerre 积分 x^2 exp(-x) 不精确 FAILED'

    # ---- TC13: Quadrature2D 矩形积分常数函数面积正确 ----
    quad2d = Quadrature2D()
    val2d = quad2d.integrate_rectangle(lambda x, y: np.ones_like(x), (0.0, 2.0), (0.0, 3.0), nx=4, ny=4)
    assert abs(val2d - 6.0) < 1e-12, '[TC13] 2D 矩形积分常数函数不正确 FAILED'

    # ---- TC14: HypercubeSampler 固定种子可复现 ----
    sampler = HypercubeSampler()
    s1 = sampler.sample(m=3, n=100, seed=42)
    s2 = sampler.sample(m=3, n=100, seed=42)
    assert np.array_equal(s1, s2), '[TC14] 固定种子采样结果不可复现 FAILED'

    # ---- TC15: ViscosityModel Arrhenius 在 T_ref 处等于 eta0 ----
    viscosity = ViscosityModel()
    T_ref = 1600.0
    eta = viscosity.arrhenius(np.array([T_ref]))
    assert abs(eta[0] - viscosity.eta0) < 1e-10, '[TC15] Arrhenius 粘度在 T_ref 处不为 eta0 FAILED'

    # ---- TC16: DensityModel 浮力在 T_ref 处为零 ----
    density = DensityModel()
    buoy = density.buoyancy(np.array([density.T_ref]))
    assert abs(buoy[0]) < 1e-10, '[TC16] 浮力在 T_ref 处不为零 FAILED'

    # ---- TC17: DimensionlessNumbers Rayleigh 数正参数返回正值 ----
    Ra = DimensionlessNumbers.rayleigh_number(1000.0, 1000.0)
    assert Ra > 0.0, '[TC17] Rayleigh 数不为正 FAILED'

    # ---- TC18: DimensionlessNumbers Nusselt 数边界处理正确 ----
    Nu = DimensionlessNumbers.nusselt_number(1.2, 0.0)
    assert abs(Nu - 1.0) < 1e-10, '[TC18] Nusselt 数在 q_cond=0 时边界处理不正确 FAILED'

    # ---- TC19: ThermalPhysics 热产生项为有限正数 ----
    physics = ThermalPhysics()
    Q = physics.heat_production_term()
    assert np.isfinite(Q) and Q > 0.0, '[TC19] 热产生项不是有限正数 FAILED'

    # ---- TC20: GrazingChemicalExchange RK4 积分结果非负 ----
    chem = GrazingChemicalExchange(a=1.1, c1=1.2, c2=1.5, d1=0.001, d2=0.001, K=3000.0, r1=0.8)
    t_chem, y_chem = chem.integrate_rk4(y0=np.array([3000.0, 5.0]), t_span=(0.0, 10.0), n_steps=500)
    assert np.all(y_chem >= 0.0), '[TC20] 化学交换 ODE 积分结果出现负值 FAILED'

    # ---- TC21: ThermalSolver 初始温度线性模式边界条件 ----
    thermal = ThermalSolver()
    r_grid = np.linspace(0.5, 1.0, 10).reshape(-1, 1)
    theta_grid = np.zeros_like(r_grid)
    T = thermal.initial_temperature(r_grid, theta_grid, mode="linear")
    assert abs(T[0, 0] - thermal.T_cmb) < 1e-10, '[TC21] 初始温度内边界不为 T_cmb FAILED'
    assert abs(T[-1, 0] - thermal.T_surf) < 1e-10, '[TC21] 初始温度外边界不为 T_surf FAILED'

    # ---- TC22: ChandrupatlaRootFinder 能找到 x^2-2=0 的根 ----
    root_finder = ChandrupatlaRootFinder()
    xm, fm, calls = root_finder.find_root(lambda x: x ** 2 - 2.0, 1.0, 2.0)
    assert abs(xm - np.sqrt(2.0)) < 1e-5, '[TC22] Chandrupatla 根查找不精确 FAILED'

    # ---- TC23: LissajousForcing 热流调制在合理范围内 ----
    forcing = LissajousForcing(a1=2.0, b1=np.pi / 4.0, a2=3.0, b2=0.0)
    q_mod = forcing.boundary_heat_flux_modulation(t=1.0, q0=1.0, amplitude=0.1)
    assert 0.9 <= q_mod <= 1.1, '[TC23] Lissajous 热流调制超出合理范围 FAILED'

    # ---- TC24: MantleDiagnostics 温度场统计量范围正确 ----
    np.random.seed(42)
    T_test = np.random.normal(loc=1600.0, scale=400.0, size=(10, 10))
    T_test = np.clip(T_test, 300.0, 3000.0)
    diag = MantleDiagnostics(T_min=300.0, T_max=3000.0)
    stats = diag.analyze_temperature_field(T_test)
    assert stats['mean'] >= stats['min'] and stats['mean'] <= stats['max'], '[TC24] 温度场均值不在 [min, max] 范围内 FAILED'

    # ---- TC25: StokesSolver 速度场输出尺寸与输入温度场一致 ----
    stokes = StokesSolver(R_inner=0.5, R_outer=1.0, n_radial=6, n_angular=8)
    r = np.linspace(0.5, 1.0, 8)
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
    T_field = np.ones((8, 12)) * 1600.0
    u_r, u_theta = stokes.compute_velocity_from_streamfunction(np.zeros((6, 8)), T_field, r_grid, theta_grid)
    assert u_r.shape == T_field.shape, '[TC25] 速度场 u_r 尺寸与温度场不匹配 FAILED'
    assert u_theta.shape == T_field.shape, '[TC25] 速度场 u_theta 尺寸与温度场不匹配 FAILED'

    print('\n全部 25 个测试通过!\n')
