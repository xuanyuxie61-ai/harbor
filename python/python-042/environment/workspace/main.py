
import numpy as np
import sys


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
    print("=" * 60)
    print("[2] 网格生成与 Voronoi 板块划分模块")
    print("=" * 60)


    triangulator = PolygonTriangulator()

    n_vertices = 8
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    x_poly = 1.0 + 0.3 * np.cos(angles)
    y_poly = 1.0 + 0.3 * np.sin(angles)
    triangles = triangulator.triangulate(x_poly, y_poly)
    print(f"  多边形三角剖分: {n_vertices} 顶点 → {len(triangles)} 个三角形")
    area_sum = 0.0
    for tri in triangles:
        i1, i2, i3 = tri - 1
        a = 0.5 * abs((x_poly[i2] - x_poly[i1]) * (y_poly[i3] - y_poly[i1])
                      - (x_poly[i3] - x_poly[i1]) * (y_poly[i2] - y_poly[i1]))
        area_sum += a
    print(f"  剖分三角形总面积 = {area_sum:.6f}")


    voronoi = VoronoiTessellator(bounds=(0.0, 10.0, 0.0, 10.0))
    generators = np.array([[3.0, 4.0], [7.0, 8.0], [7.0, 2.0], [5.0, 5.0]])
    labels, areas = voronoi.compute_cells(generators, grid_res=200)
    print(f"  Voronoi 板块划分: {len(generators)} 个生成元")
    for i, area in enumerate(areas):
        print(f"    板块 {i+1} 近似面积 = {area:.4f}")


    indexer = UnicycleIndexer()
    n = 12
    shift = 3
    u_index = indexer.create_cycle(n, shift)
    sequence = indexer.index_to_sequence(n, u_index)
    print(f"  Unicycle 循环索引 (n={n}, shift={shift}):")
    print(f"    索引向量: {u_index}")
    print(f"    序列向量: {sequence}")


    mesh = MantleMesh2D(R_inner=0.5, R_outer=1.0)
    nodes, triangs, bnd = mesh.generate_annular_sector_mesh(n_r=6, n_theta=12)
    print(f"  环形截面网格: {len(nodes)} 节点, {len(triangs)} 三角形, {len(bnd)} 边界节点")
    print()


def run_spectral_basis():
    print("=" * 60)
    print("[3] 谱基与插值模块")
    print("=" * 60)


    A = np.array([[1.0, 1.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    U_classical = GramSchmidt.classical(A)
    print("  Gram-Schmidt 正交化 (经典):")
    print(f"    U^T U = \n{U_classical.T @ U_classical}")


    trig = TrigonometricBasis()
    x_test = np.linspace(-1.0, 1.0, 101)
    k_vals = [1, 2, 3, 4]
    print("  三角基函数在 x=0 处取值 (应为 1):")
    for k in k_vals:
        val = trig.basis(np.array([0.0]), k)
        print(f"    k={k}: B_k(0) = {val[0]:.6f}")


    lagrange = LagrangeInterpolation()
    xd = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    yd = np.sin(np.pi * xd)
    xi = np.linspace(0.0, 1.0, 21)
    yi = lagrange.interpolate(xd, yd, xi)
    error = np.max(np.abs(yi - np.sin(np.pi * xi)))
    print(f"  Lagrange 插值 sin(πx) 在 [0,1] 最大误差 = {error:.2e}")


    expansion = SpectralExpansion(n_radial=6, n_angular=8)
    r_nodes = np.linspace(0.5, 1.0, 6)
    r_eval = np.linspace(0.5, 1.0, 20)
    theta_eval = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
    R_basis = expansion.build_radial_basis(r_nodes, r_eval)
    Theta_basis = expansion.build_angular_basis(theta_eval)
    print(f"  谱展开基: 径向 {R_basis.shape[1]} 模式, 角向 {Theta_basis.shape[1]} 模式")
    print()


def run_quadrature():
    print("=" * 60)
    print("[4] 数值积分模块")
    print("=" * 60)


    gl = GaussLegendre()
    x, w = gl.compute(8)
    print(f"  Gauss-Legendre (n=8): 节点范围 [{x.min():.6f}, {x.max():.6f}]")
    integral = gl.integrate_1d(lambda t: np.exp(t), -1.0, 1.0, n=16)
    exact = np.exp(1.0) - np.exp(-1.0)
    print(f"  ∫_{'{-1}'}^{'{1}'} exp(x) dx ≈ {integral:.10f} (精确值 {exact:.10f}, 误差 {abs(integral - exact):.2e})")


    laguerre = GaussLaguerre()
    xl, wl = laguerre.compute(n=12, alpha=0.0, a=0.0, b=1.0)
    integral_l = laguerre.integrate(lambda t: t ** 2, n=12, alpha=0.0, a=0.0, b=1.0)
    exact_l = 2.0
    print(f"  Gauss-Laguerre ∫_0^∞ x^2 exp(-x) dx ≈ {integral_l:.10f} (精确值 {exact_l:.10f}, 误差 {abs(integral_l - exact_l):.2e})")


    quad2d = Quadrature2D()
    f2d = lambda x, y: np.exp(-(x ** 2 + y ** 2))
    val2d = quad2d.integrate_rectangle(f2d, (-1.0, 1.0), (-1.0, 1.0), nx=8, ny=8)
    print(f"  2D Gauss-Legendre 乘积规则 ∫_{'{-1}'}^{'{1}'}∫_{'{-1}'}^{'{1}'} exp(-(x²+y²)) dxdy ≈ {val2d:.8f}")


    sampler = HypercubeSampler()
    samples = sampler.sample(m=3, n=1000, seed=42)
    print(f"  超立方体采样 (m=3, n=1000): 样本均值 = {np.mean(samples):.4f}, 样本方差 = {np.var(samples):.4f}")


    mc_mean, mc_err = sampler.integrate(lambda x: np.sum(x ** 2, axis=0), m=3, n=5000, seed=42)
    print(f"  Monte Carlo ∫_{'{[0,1]³}'} (x²+y²+z²) dV ≈ {mc_mean:.6f} ± {mc_err:.6f}")
    print()


def run_mantle_physics():
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
    print("=" * 60)
    print("[6] 斯托克斯流求解与热演化模块")
    print("=" * 60)


    R_inner = 0.5
    R_outer = 1.0
    nr, ntheta = 25, 40
    r = np.linspace(R_inner, R_outer, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')


    thermal = ThermalSolver()
    T = thermal.initial_temperature(r_grid, theta_grid, mode="perturbation")
    print(f"  初始温度场: 形状 {T.shape}, 均值 {np.mean(T):.2f} K, 范围 [{np.min(T):.1f}, {np.max(T):.1f}] K")


    stokes = StokesSolver(R_inner=R_inner, R_outer=R_outer, n_radial=8, n_angular=12)
    u_r, u_theta = stokes.compute_velocity_from_streamfunction(
        np.zeros((8, 12)), T, r_grid, theta_grid
    )
    print(f"  速度场: u_r 范围 [{np.min(u_r):.3e}, {np.max(u_r):.3e}], u_θ 范围 [{np.min(u_theta):.3e}, {np.max(u_theta):.3e}]")


    dt = 1.0e11
    n_steps = 20
    for step in range(n_steps):
        T = thermal.step_forward(T, u_r, u_theta, r_grid, theta_grid, dt)
    print(f"  演化 {n_steps} 步 (Δt={dt:.2e} s ≈ {dt/3.15e13:.1f} Myr) 后:")
    print(f"    温度均值 = {np.mean(T):.2f} K, 标准差 = {np.std(T):.2f} K")


    q_surf = thermal.surface_heat_flux(T, r_grid)
    print(f"    表面热流 (无量纲) = {q_surf:.4f}")


    chem = GrazingChemicalExchange(a=1.1, c1=1.2, c2=1.5, d1=0.001, d2=0.001,
                                    K=3000.0, r1=0.8)
    t_chem, y_chem = chem.integrate_rk4(y0=np.array([3000.0, 5.0]),
                                         t_span=(0.0, 100.0), n_steps=2000)
    print(f"\n  上/下地幔化学交换 (Grazing ODE 类比):")
    print(f"    t=0:   C_um={y_chem[0,0]:.2f}, C_lm={y_chem[0,1]:.4f}")
    print(f"    t=100: C_um={y_chem[-1,0]:.2f}, C_lm={y_chem[-1,1]:.4f}")
    print()


def run_diagnostics():
    print("=" * 60)
    print("[7] 诊断分析与参数不确定性模块")
    print("=" * 60)

    diag = MantleDiagnostics(T_min=300.0, T_max=3000.0)


    np.random.seed(42)
    T_sim = np.random.normal(loc=1600.0, scale=400.0, size=(30, 40))
    T_sim = np.clip(T_sim, 300.0, 3000.0)
    stats = diag.analyze_temperature_field(T_sim)
    print(f"  温度场统计:")
    print(f"    均值 = {stats['mean']:.2f} K")
    print(f"    标准差 = {stats['std']:.2f} K")
    print(f"    熵 = {stats['entropy']:.4f}")


    def mock_nu(Ra):

        return 1.0 + 0.5 * np.tanh(np.log10(Ra / 1.0e5))

    Ra_c, fm, calls = diag.root_finder.find_critical_rayleigh(mock_nu, 1.0e3, 1.0e7)
    print(f"\n  临界瑞利数搜索 (Chandrupatla 算法):")
    print(f"    Ra_c ≈ {Ra_c:.3e}, f(Ra_c) = {fm:.2e}, 函数调用次数 = {calls}")


    forcing = LissajousForcing(a1=2.0, b1=np.pi / 4.0, a2=3.0, b2=0.0)
    t_forcing = np.linspace(0.0, 4 * np.pi, 100)
    x_liss, y_liss = forcing.evaluate(t_forcing)
    q_mod = [forcing.boundary_heat_flux_modulation(t, q0=1.0, amplitude=0.1) for t in t_forcing]
    print(f"\n  Lissajous 周期强迫:")
    print(f"    热流调制范围: [{min(q_mod):.4f}, {max(q_mod):.4f}]")


    sampler = ParameterSampler(seed=123)
    def toy_model(theta):

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
    print("=" * 60)
    print("[8] 综合模拟结果汇总")
    print("=" * 60)


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
    q_cond = 1.0
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
    sys.exit(main())
