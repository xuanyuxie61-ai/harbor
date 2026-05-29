"""
main.py
r 过程核合成多尺度数值模拟平台 — 统一入口

本项目围绕核天体物理中的 r 过程（快中子捕获过程）核合成展开，
集成以下计算模块：

1. 核素空间采样 (nuclide_sampling.py)
2. 核反应率计算 (reaction_rates.py)
3. 核反应网络求解 (nuclear_network.py)
4. 中子输运 (neutron_transport.py)
5. 谱展开 (spectral_expansion.py)
6. 循环矩阵求解 (circulant_solver.py)
7. 非线性求根 (nonlinear_root.py)
8. 共形映射 (conformal_mapping.py)
9. 球面几何 (spherical_geometry.py)
10. 核素编码 (nuclide_encoding.py)
11. 数值积分 (quadrature_rules.py)
12. FEM 拟合 (fem_approximation.py)
13. Voronoi 划分 (voronoi_partition.py)

运行方式：python main.py（零参数）
"""

import numpy as np
import time

from nuclide_sampling import sample_nuclide_mass_chain, build_r_process_nuclide_set
from reaction_rates import build_reaction_rate_table
from nuclear_network import solve_network_bdf2, compute_abundance_peaks
from neutron_transport import neutron_diffusion_solution, neutron_capture_rate_profile
from spectral_expansion import spectral_expand_reaction_rate, spectral_evaluate_reaction_rate
from circulant_solver import circulant_solve, build_circulant_dif2
from nonlinear_root import solve_neutron_chemical_potential
from conformal_mapping import map_accretion_streamline, temperature_field_conformal
from spherical_geometry import icosahedron_vertices, spherical_delaunay_triangulation
from nuclide_encoding import atbash_mirror_map, build_nuclide_grid_path, gaussian_prime_spiral_trajectory
from quadrature_rules import integrate_tetrahedron, wedge_exactness_monomial_integral
from fem_approximation import fem1d_approximate, fem1d_evaluate
from voronoi_partition import partition_nuclear_chart, interpolate_nuclear_data


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_nuclide_sampling():
    """步骤 1：核素空间采样与路径生成"""
    print_section("Step 1: Nuclide Space Sampling & Path Generation")
    a_values = sample_nuclide_mass_chain(a_min=80, a_max=240, n_nuclides=40,
                                          density_profile='r_process_path')
    nuclides = build_r_process_nuclide_set(a_values, beta_stability_offset=10)
    print(f"  Sampled {len(nuclides)} r-process nuclides")

    # 高斯素数螺旋路径
    spiral = gaussian_prime_spiral_trajectory(0 + 0j, 1, max_steps=200)
    print(f"  Gaussian prime spiral length: {len(spiral)}")

    # 镜像核映射
    mirrored = atbash_mirror_map(nuclides[:5])
    print(f"  Mirror nuclei example: {nuclides[0]} -> {mirrored[0]}")

    # Voronoi 分区
    generators, labels = partition_nuclear_chart(nuclides, n_partitions=4)
    print(f"  Nuclear chart partitioned into {len(np.unique(labels))} regions")

    return nuclides


def run_reaction_rates(nuclides):
    """步骤 2：温度依赖反应率计算与谱展开"""
    print_section("Step 2: Temperature-Dependent Reaction Rates & Spectral Expansion")
    T9_range = np.linspace(0.5, 3.0, 50)

    # 简化的核数据表
    S_n_table = {}
    T_half_table = {}
    for z, n, a in nuclides:
        S_n_table[(z, a)] = 8.0 - 0.02 * (a - 100)  # 简化中子分离能
        T_half_table[(z, a)] = max(0.01, 10.0 * np.exp(-0.01 * (a - 80)))  # 简化半衰期

    rates = build_reaction_rate_table(nuclides, T9_range, S_n_table, T_half_table)
    print(f"  Reaction rate table built for {len(nuclides)} nuclides")

    # 谱展开示例
    key = (nuclides[0][0], nuclides[0][2])  # (Z,A)
    cap_rates = rates['capture'][key]
    coeffs, t_min, t_max = spectral_expand_reaction_rate(T9_range, cap_rates, degree=8)
    print(f"  Spectral expansion coefficients (degree 8): max |c_k| = {np.max(np.abs(coeffs)):.3e}")

    # 重构验证
    tau_test = np.linspace(0, 1, 20)
    recon = spectral_evaluate_reaction_rate(tau_test, coeffs, t_min, t_max)
    T_test = tau_test * (t_max - t_min) + t_min
    exact_interp = np.interp(T_test, T9_range, cap_rates)
    err = np.max(np.abs(recon - exact_interp) / (np.abs(exact_interp) + 1e-30))
    print(f"  Spectral reconstruction max relative error: {err:.3e}")

    return rates, T9_range, S_n_table, T_half_table


def run_neutron_transport():
    """步骤 3：中子输运与化学势计算"""
    print_section("Step 3: Neutron Transport & Chemical Potential")
    r = np.linspace(1e3, 1e6, 500)
    R_star = 1e6  # cm
    D = 1e5
    sigma_a = 1e-3
    S0 = 1e20
    phi = neutron_diffusion_solution(r, R_star, D, sigma_a, S0)
    print(f"  Neutron flux at center: {phi[0]:.3e} cm^-2 s^-1")
    print(f"  Neutron flux at surface: {phi[-1]:.3e} cm^-2 s^-1")

    # 俘获率径向分布
    n_n = 1e30  # cm^-3
    sigma_cap = 1e-24  # cm^2
    capture_profile = neutron_capture_rate_profile(r, phi, n_n, sigma_cap)
    print(f"  Max capture rate: {np.max(capture_profile):.3e} cm^-3 s^-1")

    # 化学势
    mu_n, info = solve_neutron_chemical_potential(n_n, 1e9)
    print(f"  Neutron chemical potential: {mu_n:.3e} erg (eta={info.get('eta', 'N/A')})")

    return phi, capture_profile


def run_nuclear_network(nuclides, rates):
    """步骤 4：核反应网络演化"""
    print_section("Step 4: Nuclear Reaction Network Evolution")
    n_nuc = len(nuclides)
    Y0 = np.ones(n_nuc) / n_nuc
    rho = 1e8  # g/cm^3
    n_n = 1e30  # cm^-3
    t_end = 5.0  # s

    t_hist, Y_hist = solve_network_bdf2(nuclides, rates, rho, n_n, Y0,
                                         t_end=t_end, n_steps=200)
    print(f"  Network solved: {len(t_hist)} time steps")
    print(f"  Final total abundance: {np.sum(Y_hist[-1]):.6f}")

    # r 过程峰分析
    A_centers, abundances = compute_abundance_peaks(Y_hist[-1], nuclides)
    peak_idx = np.argmax(abundances)
    print(f"  Dominant abundance peak at A ≈ {A_centers[peak_idx]:.0f}")

    return t_hist, Y_hist, A_centers, abundances


def run_geometric_tools():
    """步骤 5：几何映射与球面网格"""
    print_section("Step 5: Geometric Mapping & Spherical Mesh")
    # 共形映射
    theta = np.linspace(0, 2 * np.pi, 100)
    w_r, w_i = map_accretion_streamline(1.2, theta, offset=0.15)
    print(f"  Accretion streamline mapped to {len(w_r)} points")

    # 球面三角化
    verts = icosahedron_vertices()
    faces = spherical_delaunay_triangulation(verts)
    print(f"  Icosahedron triangulation: {len(verts)} vertices, {len(faces)} faces")

    # 温度场
    rho_grid = np.linspace(1.0, 10.0, 50)
    T_field = temperature_field_conformal(rho_grid, 0.0, 1e9, 1e8)
    print(f"  Temperature field range: [{np.min(T_field):.3e}, {np.max(T_field):.3e}] K")


def run_numerical_integrals():
    """步骤 6：数值积分验证"""
    print_section("Step 6: Numerical Quadrature Verification")
    # 四面体积分：测试积分 x*y*z
    f_test = lambda x, y, z: x * y * z
    val_tet = integrate_tetrahedron(f_test, n_per_dim=8)
    exact_tet = 1.0 / 720.0
    print(f"  Tetrahedron integral: {val_tet:.6e}, exact: {exact_tet:.6e}, err: {abs(val_tet - exact_tet):.3e}")

    # 楔形精确积分
    exact_wedge = wedge_exactness_monomial_integral(2, 1, 0)
    print(f"  Wedge exact integral (x^2*y): {exact_wedge:.6e}")


def run_fem_and_circulant():
    """步骤 7：FEM 拟合与循环矩阵求解"""
    print_section("Step 7: FEM Approximation & Circulant Solver")
    # FEM 拟合反应率-温度曲线
    T_data = np.random.rand(80)
    R_data = np.exp(-2.0 / T_data) + 0.05 * np.random.randn(80)
    mesh = np.linspace(0.1, 3.0, 25)
    coeffs_fem = fem1d_approximate(mesh, T_data, R_data,
                                    weight_approx=1.0, weight_deriv=0.1,
                                    weight_boundary=1e4,
                                    boundary_values=(0.0, 0.0))
    T_test = np.linspace(0.1, 3.0, 100)
    R_fit = fem1d_evaluate(T_test, mesh, coeffs_fem)
    print(f"  FEM fit evaluated at {len(T_test)} points, range: [{np.min(R_fit):.3e}, {np.max(R_fit):.3e}]")

    # 循环矩阵求解（使用非奇异矩阵测试）
    n = 32
    a_circ = np.array([3.0, -1.0] + [0.0] * (n - 3) + [-1.0])
    b = np.random.rand(n)
    x_sol = circulant_solve(a_circ, b)
    # 验证
    from circulant_solver import circulant_matvec
    residual = np.linalg.norm(circulant_matvec(a_circ, x_sol) - b)
    print(f"  Circulant solver residual: {residual:.3e}")


def run_interpolation(nuclides):
    """步骤 8：核数据插值"""
    print_section("Step 8: Nuclear Data Interpolation")
    coords = np.array([(z, n) for z, n, a in nuclides], dtype=float)
    # 模拟一些核数据（分离能）
    data = np.array([8.0 - 0.02 * (a - 100) for z, n, a in nuclides])
    # 查询点
    query = np.array([[30, 50], [40, 65], [50, 82]], dtype=float)
    interp_vals = interpolate_nuclear_data(query, coords, data)
    print(f"  Interpolated S_n at query points: {interp_vals}")


def main():
    print("\n" + "#" * 70)
    print("#  r-Process Nucleosynthesis Multi-Scale Simulation Platform")
    print("#  Nuclear Astrophysics: Neutron Star Merger Environment")
    print("#" * 70)
    start_time = time.time()

    # 步骤 1
    nuclides = run_nuclide_sampling()

    # 步骤 2
    rates, T9_range, S_n_table, T_half_table = run_reaction_rates(nuclides)

    # 步骤 3
    phi, capture_profile = run_neutron_transport()

    # 步骤 4
    t_hist, Y_hist, A_centers, abundances = run_nuclear_network(nuclides, rates)

    # 步骤 5
    run_geometric_tools()

    # 步骤 6
    run_numerical_integrals()

    # 步骤 7
    run_fem_and_circulant()

    # 步骤 8
    run_interpolation(nuclides)

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  Simulation completed successfully in {elapsed:.2f} seconds")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# 补充导入测试所需的函数（main.py 中未全部导入）
from circulant_solver import circulant_matvec, circulant_eigenvalues, circulant_determinant
from conformal_mapping import joukowsky_transform, joukowsky_inverse
from fem_approximation import hat_function, data_bracket
from neutron_transport import spherical_bessel_j0, spherical_bessel_y0, modified_bessel_half
from nonlinear_root import bisect_root
from nuclide_encoding import is_gaussian_prime
from spectral_expansion import shifted_legendre_polynomial_value
from spherical_geometry import sphere_stereograph, sphere_stereograph_inverse
from voronoi_partition import voronoi_nearest_generator
from reaction_rates import beta_decay_rate, alpha_decay_rate, fission_rate
from nuclear_network import solve_network_implicit_euler

# ---- TC01: circulant_solve 对非奇异循环矩阵残差小于1e-10 ----
import numpy as np
np.random.seed(42)
n = 32
a = np.array([3.0, -1.0] + [0.0] * (n - 3) + [-1.0])
b = np.random.rand(n)
x = circulant_solve(a, b)
residual = np.linalg.norm(circulant_matvec(a, x) - b)
assert residual < 1e-10, '[TC01] circulant_solve residual too large'

# ---- TC02: circulant_matvec 与 FFT 直接计算结果一致 ----
np.random.seed(42)
n = 16
a = np.random.rand(n)
x = np.random.rand(n)
y1 = circulant_matvec(a, x)
y2 = np.fft.ifft(np.fft.fft(a) * np.fft.fft(x))
assert np.linalg.norm(y1 - np.real(y2)) < 1e-10, '[TC02] circulant_matvec inconsistent with FFT'

# ---- TC03: build_circulant_dif2 生成的矩阵特征值为实数 ----
a = build_circulant_dif2(64)
lam = circulant_eigenvalues(a)
assert np.allclose(lam.imag, 0, atol=1e-10), '[TC03] circulant_dif2 eigenvalues not real'

# ---- TC04: circulant_determinant 单位循环矩阵行列式为1 ----
a = np.zeros(8)
a[0] = 1.0
det_val = circulant_determinant(a)
assert np.isclose(det_val, 1.0, atol=1e-10), '[TC04] identity circulant determinant != 1'

# ---- TC05: joukowsky_transform 与逆变换对圆外点互逆 ----
z = np.array([2.0 + 0.0j, 3.0 + 1.0j, 1.5 - 0.5j])
w = joukowsky_transform(z)
z_rec = joukowsky_inverse(w, branch='+')
assert np.max(np.abs(z_rec - z)) < 1e-10, '[TC05] Joukowsky inverse inaccurate for |z|>1'

# ---- TC06: map_accretion_streamline 输出长度与输入一致 ----
theta = np.linspace(0, 2 * np.pi, 50)
w_r, w_i = map_accretion_streamline(1.2, theta, offset=0.15)
assert len(w_r) == len(theta) and len(w_i) == len(theta), '[TC06] streamline output length mismatch'

# ---- TC07: temperature_field_conformal 边界值精确 ----
rho = np.array([1.0, 10.0])
T = temperature_field_conformal(rho, 0.0, 1e9, 1e8)
assert np.isclose(T[0], 1e9, atol=1e-6), '[TC07] T inner boundary incorrect'
assert np.isclose(T[-1], 1e8, atol=1e-6), '[TC07] T outer boundary incorrect'

# ---- TC08: hat_function 在中心点取值为1 ----
val = hat_function(0.5, 0.0, 0.5, 1.0)
assert np.isclose(val, 1.0), '[TC08] hat_function peak value != 1'

# ---- TC09: fem1d_approximate 边界条件近似满足 ----
np.random.seed(42)
mesh = np.linspace(0, 1, 11)
x_data = np.random.rand(50)
y_data = np.sin(np.pi * x_data)
coeffs = fem1d_approximate(mesh, x_data, y_data, weight_boundary=1e6, boundary_values=(0.0, 0.0))
y_left = fem1d_evaluate(0.0, mesh, coeffs)
y_right = fem1d_evaluate(1.0, mesh, coeffs)
assert np.isclose(y_left, 0.0, atol=1e-5), '[TC09] FEM left boundary not satisfied'
assert np.isclose(y_right, 0.0, atol=1e-5), '[TC09] FEM right boundary not satisfied'

# ---- TC10: data_bracket 返回正确区间索引 ----
mesh = np.array([0.0, 0.2, 0.5, 1.0])
x_data = np.array([0.1, 0.3, 0.6, 0.9])
idx = data_bracket(mesh, x_data)
expected = np.array([0, 1, 2, 2])
assert np.array_equal(idx, expected), '[TC10] data_bracket indices incorrect'

# ---- TC11: spherical_bessel_j0(0) = 1 ----
val = spherical_bessel_j0(np.array([0.0]))
assert np.isclose(val[0], 1.0), '[TC11] spherical_bessel_j0(0) != 1'

# ---- TC12: modified_bessel_half I_{1/2} 解析公式验证 ----
x = np.array([1.0, 2.0, 5.0])
val = modified_bessel_half(x, kind='I')
exact = np.sqrt(2.0 / (np.pi * x)) * np.sinh(x)
assert np.allclose(val, exact, atol=1e-10), '[TC12] modified_bessel_half I inaccurate'

# ---- TC13: neutron_diffusion_solution 边界为零且中心为正 ----
r = np.linspace(1e3, 1e6, 100)
phi = neutron_diffusion_solution(r, 1e6, 1e5, 1e-3, 1e20)
assert phi[0] > 0, '[TC13] neutron flux at center not positive'
assert np.isclose(phi[-1], 0.0, atol=1e-6), '[TC13] neutron flux boundary not zero'

# ---- TC14: neutron_capture_rate_profile 标量 n_n 可广播 ----
r = np.array([1.0, 2.0, 3.0])
phi = np.array([1e10, 5e9, 2e9])
rate = neutron_capture_rate_profile(r, phi, 1e30, 1e-24)
expected = 1e30 * 1e-24 * phi
assert np.allclose(rate, expected), '[TC14] capture rate profile broadcast failed'

# ---- TC15: bisect_root 对 sin(x)=0.5 根精确 ----
f = lambda x: np.sin(x) - 0.5
root, info = bisect_root(f, 0.0, np.pi / 2.0)
assert np.isclose(root, np.arcsin(0.5), atol=1e-6), '[TC15] bisect_root inaccurate'

# ---- TC16: solve_neutron_chemical_potential 返回有限化学势 ----
mu, info = solve_neutron_chemical_potential(1e30, 1e9)
assert np.isfinite(mu), '[TC16] chemical potential not finite'

# ---- TC17: atbash_mirror_map 两次映射回到原列表 ----
nuclides = [(26, 30, 56), (82, 126, 208)]
m1 = atbash_mirror_map(nuclides)
m2 = atbash_mirror_map(m1)
assert m2 == nuclides, '[TC17] atbash_mirror_map not involutive'

# ---- TC18: is_gaussian_prime (0,3) 为真 ----
assert is_gaussian_prime(0, 3), '[TC18] (0,3) should be Gaussian prime'

# ---- TC19: build_nuclide_grid_path 所有点都在指定范围内 ----
path = build_nuclide_grid_path(20, 40, 20, 50)
assert len(path) > 0, '[TC19] grid path empty'
assert all(20 <= z <= 40 and 20 <= n <= 50 for z, n in path), '[TC19] grid path out of bounds'

# ---- TC20: integrate_tetrahedron 对 f=1 精确等于 1/6 ----
f = lambda x, y, z: 1.0
val = integrate_tetrahedron(f, n_per_dim=4)
assert np.isclose(val, 1.0 / 6.0, atol=1e-12), '[TC20] tetrahedron integral of 1 incorrect'

# ---- TC21: wedge_exactness_monomial_integral (1,1,0) = 1/12 ----
exact = wedge_exactness_monomial_integral(1, 1, 0)
assert np.isclose(exact, 1.0 / 12.0, atol=1e-15), '[TC21] wedge exact integral (1,1,0) incorrect'

# ---- TC22: integrate_tetrahedron 对 xyz 精确等于 1/720 ----
f = lambda x, y, z: x * y * z
val = integrate_tetrahedron(f, n_per_dim=8)
assert np.isclose(val, 1.0 / 720.0, atol=1e-12), '[TC22] tetrahedron integral of xyz incorrect'

# ---- TC23: beta_decay_rate(1.0) = ln(2) ----
val = beta_decay_rate(1.0)
assert np.isclose(val, np.log(2.0)), '[TC23] beta_decay_rate(1.0) != ln(2)'

# ---- TC24: alpha_decay_rate Q_alpha<=0.1 返回 1e-30 ----
val = alpha_decay_rate(92, 238, 0.05)
assert val == 1e-30, '[TC24] alpha_decay_rate small Q boundary incorrect'

# ---- TC25: fission_rate 对轻核素返回 0 ----
val = fission_rate(26, 56, 1e30, 1.0)
assert val == 0.0, '[TC25] fission_rate for light nuclei should be 0'

# ---- TC26: shifted_legendre_polynomial_value P0=1 ----
x_arr = np.array([0.1, 0.5, 0.9])
v = shifted_legendre_polynomial_value(len(x_arr), 0, x_arr)
assert np.allclose(v[:, 0], 1.0), '[TC26] P0 not equal to 1'

# ---- TC27: 谱展开对常数函数精确重构 ----
T = np.linspace(1e9, 10e9, 50)
R = np.ones_like(T) * 3.14
coeffs, t_min, t_max = spectral_expand_reaction_rate(T, R, degree=4)
tau_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
R_recon = spectral_evaluate_reaction_rate(tau_test, coeffs, t_min, t_max)
assert np.allclose(R_recon, 3.14), '[TC27] spectral expansion of constant inaccurate'

# ---- TC28: icosahedron_vertices 12个顶点且模长为1 ----
verts = icosahedron_vertices()
norms = np.linalg.norm(verts, axis=1)
assert len(verts) == 12, '[TC28] icosahedron vertex count != 12'
assert np.allclose(norms, 1.0), '[TC28] icosahedron vertices not on unit sphere'

# ---- TC29: spherical_delaunay_triangulation 对二十面体返回20个面 ----
faces = spherical_delaunay_triangulation(verts)
assert len(faces) == 20, '[TC29] icosahedron face count != 20'

# ---- TC30: sphere_stereograph 北极点逆映射精确 ----
p = np.array([[0, 0, 1]], dtype=float)
q = sphere_stereograph(p)
p_rec = sphere_stereograph_inverse(q)
assert np.max(np.abs(p_rec - p)) < 1e-10, '[TC30] north pole stereographic projection inaccurate'

# ---- TC31: voronoi_nearest_generator 查询点与生成点重合时返回该索引 ----
gens = np.array([[0.0, 0.0], [1.0, 1.0]])
queries = np.array([[0.0, 0.0], [1.0, 1.0]])
idx, dists = voronoi_nearest_generator(queries, gens)
assert np.array_equal(idx, np.array([0, 1])), '[TC31] nearest generator index mismatch'
assert np.allclose(dists, 0.0), '[TC31] nearest generator distance not zero'

# ---- TC32: interpolate_nuclear_data 返回值在已知数据范围 ----
known_nz = np.array([[50, 80], [82, 126], [92, 146]], dtype=float)
known_data = np.array([1.0, 2.0, 3.0])
query_nz = np.array([[60, 90], [85, 130]], dtype=float)
interp = interpolate_nuclear_data(query_nz, known_nz, known_data)
assert np.all((interp >= known_data.min()) & (interp <= known_data.max())), '[TC32] interpolated data out of range'

# ---- TC33: build_r_process_nuclide_set Z+N=A ----
a_vals = np.array([56, 100, 200])
nuclides = build_r_process_nuclide_set(a_vals, beta_stability_offset=5)
assert all(z + n == a for z, n, a in nuclides), '[TC33] Z + N != A'

# ---- TC34: sample_nuclide_mass_chain 返回值在范围内且唯一 ----
np.random.seed(42)
a_vals = sample_nuclide_mass_chain(80, 240, 30)
assert np.all((a_vals >= 80) & (a_vals <= 240)), '[TC34] sampled A out of range'
assert len(a_vals) == len(np.unique(a_vals)), '[TC34] sampled A not unique'

# ---- TC35: solve_network_implicit_euler 丰度守恒 ----
nuclides_net = [(26, 30, 56), (26, 31, 57), (27, 30, 57), (27, 31, 58), (28, 30, 58)]
T9_range = np.array([1.0, 1.5, 2.0])
S_n_table = {(26, 56): 8.0, (26, 57): 7.5, (27, 57): 8.2, (27, 58): 7.8, (28, 58): 8.5}
T_half_table = {(26, 56): 1e10, (26, 57): 1.5, (27, 57): 272.0, (27, 58): 70.8, (28, 58): 1e10}
rates = build_reaction_rate_table(nuclides_net, T9_range, S_n_table, T_half_table)
Y0 = np.ones(len(nuclides_net)) / len(nuclides_net)
t_hist, Y_hist = solve_network_implicit_euler(nuclides_net, rates, rho=1e8, n_n=1e30, Y0=Y0, t_end=1.0, n_steps=50)
assert np.isclose(np.sum(Y_hist[-1]), 1.0, atol=1e-10), '[TC35] abundance conservation violated'

print('\n全部 35 个测试通过!\n')
