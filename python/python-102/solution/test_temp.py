import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maxwell_fem import MaxwellFEM2D
from metasurface_grid import MetasurfaceCVT
from phase_quadrature import PhaseQuadrature, quadrilateral_witherden_rule
from multipole_moments import MultipoleExtractor
from uncertainty_quantify import UncertaintyQuantify
from process_sampling import ProcessSampler
from topology_optimize import TopologyOptimizer
from wavefront_trace import WavefrontTracer
from phase_surface import PhaseSurface
from convergence_utils import ConvergenceAnalysis

print("Running TC01...")
# ---- TC01: MaxwellFEM2D 网格生成节点与单元数量正确 ----
fem = MaxwellFEM2D(wavelength=1.55e-6)
nodes, elements = fem.build_rectangular_mesh(5, 5, (-1e-6, 1e-6), (-1e-6, 1e-6))
assert nodes.shape == ((2 * 5 - 1) * (2 * 5 - 1), 2), '[TC01] 网格节点数量 FAILED'
assert elements.shape == (2 * (5 - 1) * (5 - 1), 6), '[TC01] 网格单元数量 FAILED'

print("Running TC02...")
# ---- TC02: MaxwellFEM2D 介质分布内外区域正确 ----
fem = MaxwellFEM2D(wavelength=1.55e-6)
eps_center = fem.epsilon_profile(0.0, 0.0, (0.0, 0.0), (0.5e-6, 1.0e-6))
eps_out = fem.epsilon_profile(1.0e-6, 1.0e-6, (0.0, 0.0), (0.5e-6, 1.0e-6))
assert abs(eps_center - 3.48 ** 2) < 1e-10, '[TC02] 介质中心 eps FAILED'
assert abs(eps_out - 1.0 ** 2) < 1e-10, '[TC02] 介质外部 eps FAILED'

print("Running TC03...")
# ---- TC03: MaxwellFEM2D PML 内部区域拉伸因子为 1 ----
fem = MaxwellFEM2D(wavelength=1.55e-6, pml_width=0.5e-6)
sx, sy = fem.pml_stretch(0.0, 0.0, (-1.5e-6, 1.5e-6), (-1.5e-6, 1.5e-6))
assert abs(sx - 1.0) < 1e-10 and abs(sy - 1.0) < 1e-10, '[TC03] PML 内部拉伸因子 FAILED'

print("Running TC04...")
# ---- TC04: MaxwellFEM2D T6 形状函数在参考顶点处值正确 ----
fem = MaxwellFEM2D(wavelength=1.55e-6)
N, dNdr, dNds = fem.shape_t6(0.0, 0.0)
assert abs(N[0] - 1.0) < 1e-10 and abs(N[1]) < 1e-10 and abs(N[2]) < 1e-10, '[TC04] T6 形状函数顶点值 FAILED'

print("Running TC05...")
# ---- TC05: MaxwellFEM2D 散射场求解输出尺寸匹配且有限 ----
np.random.seed(42)
fem = MaxwellFEM2D(wavelength=1.55e-6)
E_z, nodes, elements = fem.solve_scattering(nx=5, ny=5,
    domain=(-1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6),
    pillar_center=(0.0, 0.0), pillar_size=(0.3e-6, 0.6e-6))
assert len(E_z) == nodes.shape[0], '[TC05] 散射场节点数不匹配 FAILED'
assert np.isfinite(E_z).all(), '[TC05] 散射场含非有限值 FAILED'

print("Running TC06...")
# ---- TC06: MetasurfaceCVT 能量历史非空且为正 ----
np.random.seed(42)
grid = MetasurfaceCVT(region=(-2e-6, 2e-6, -2e-6, 2e-6))
gens, energy = grid.compute_cvt(n_generators=20, n_samples_per_iter=1000, max_iter=5)
assert len(energy) >= 1, '[TC06] CVT 能量历史为空 FAILED'
assert all(e > 0 for e in energy), '[TC06] CVT 能量非正 FAILED'

print("Running TC07...")
# ---- TC07: MetasurfaceCVT 生成器坐标在区域内 ----
np.random.seed(42)
grid = MetasurfaceCVT(region=(-3e-6, 3e-6, -3e-6, 3e-6))
gens, _ = grid.compute_cvt(n_generators=30, n_samples_per_iter=1500, max_iter=3)
assert gens[:, 0].min() >= -3e-6 and gens[:, 0].max() <= 3e-6, '[TC07] 生成器 x 越界 FAILED'
assert gens[:, 1].min() >= -3e-6 and gens[:, 1].max() <= 3e-6, '[TC07] 生成器 y 越界 FAILED'

print("Running TC08...")
# ---- TC08: MetasurfaceCVT Voronoi 面积和近似区域总面积 ----
np.random.seed(42)
grid = MetasurfaceCVT(region=(-2e-6, 2e-6, -2e-6, 2e-6))
gens = np.array([[-1e-6, -1e-6], [1e-6, 1e-6]])
areas = grid.compute_voronoi_areas(gens, n_samples=50000)
assert abs(np.sum(areas) - grid.Lx * grid.Ly) / (grid.Lx * grid.Ly) < 0.05, '[TC08] Voronoi 面积和 FAILED'

print("Running TC09...")
# ---- TC09: PhaseQuadrature Witherden 规则积分点与权重有效 ----
for p in [1, 3, 5, 7]:
    n, x, y, w = quadrilateral_witherden_rule(p)
    assert len(w) == n, f'[TC09] Witherden p={p} 权重数量不匹配 FAILED'
    assert np.all(w > 0), f'[TC09] Witherden p={p} 权重含非正值 FAILED'
    assert np.all((x >= 0) & (x <= 1)), f'[TC09] Witherden p={p} x 越界 FAILED'
    assert np.all((y >= 0) & (y <= 1)), f'[TC09] Witherden p={p} y 越界 FAILED'

print("Running TC10...")
# ---- TC10: PhaseQuadrature 平均相位延迟为正且传输振幅在 [0,1] ----
pq = PhaseQuadrature(wavelength=1.55e-6)
avg_phase, trans = pq.integrate_phase_delay(0.0, 0.0, 0.3e-6, 0.6e-6, 1.0e-6)
assert avg_phase > 0, '[TC10] 平均相位延迟非正 FAILED'
assert 0.0 <= trans <= 1.0, '[TC10] 传输振幅超出 [0,1] FAILED'

print("Running TC11...")
# ---- TC11: PhaseQuadrature 等效极化率为正 ----
pq = PhaseQuadrature(wavelength=1.55e-6)
alpha = pq.integrate_polarizability(0.0, 0.0, 0.3e-6, 0.6e-6)
assert alpha > 0, '[TC11] 极化率非正 FAILED'

print("Running TC12...")
# ---- TC12: PhaseQuadrature 色散关系有效折射率在合理范围 ----
pq = PhaseQuadrature(wavelength=1.55e-6)
n_effs = pq.compute_dispersion_relation(0.3e-6, 1.0e-6, n_modes=3)
assert len(n_effs) == 3, '[TC12] 色散模式数量 FAILED'
assert all(1.0 < ne < 3.48 for ne in n_effs), '[TC12] 有效折射率范围 FAILED'

print("Running TC13...")
# ---- TC13: MultipoleExtractor Levi-Civita 符号符合反对称性 ----
me = MultipoleExtractor(wavelength=1.55e-6)
assert me._levi_civita(0, 1, 2) == 1, '[TC13] Levi-Civita (0,1,2) FAILED'
assert me._levi_civita(1, 0, 2) == -1, '[TC13] Levi-Civita 反对称 FAILED'
assert me._levi_civita(0, 0, 2) == 0, '[TC13] Levi-Civita 重复指标 FAILED'

print("Running TC14...")
# ---- TC14: MultipoleExtractor 辐射功率各分量非负 ----
me = MultipoleExtractor(wavelength=1.55e-6)
p_dummy = np.array([1e-18, 0, 0])
m_dummy = np.array([0, 1e-21, 0])
powers = me.radiation_powers(p_dummy, m_dummy)
assert powers['P_dipole_electric'] >= 0, '[TC14] 电偶极功率为负 FAILED'
assert powers['P_dipole_magnetic'] >= 0, '[TC14] 磁偶极功率为负 FAILED'
assert powers['P_total'] >= 0, '[TC14] 总辐射功率为负 FAILED'

print("Running TC15...")
# ---- TC15: UncertaintyQuantify Hermite 积分权重和为正 ----
uq = UncertaintyQuantify(dim_num=2, level_max=2)
x_h, w_h = uq.hermite_gauss_rule(5)
assert np.sum(w_h) > 0, '[TC15] Hermite 权重和非正 FAILED'

print("Running TC16...")
# ---- TC16: ProcessSampler 三角形内均匀采样重心接近理论值 ----
np.random.seed(42)
ps = ProcessSampler(seed=42)
tri_pts = ps.uniform_in_triangle(10000, np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
assert 0.30 < tri_pts[:, 0].mean() < 0.37, '[TC16] 三角形采样 x 重心 FAILED'
assert 0.30 < tri_pts[:, 1].mean() < 0.37, '[TC16] 三角形采样 y 重心 FAILED'

print("Running TC17...")
# ---- TC17: ProcessSampler 多变量高斯采样均值接近基值 ----
np.random.seed(42)
ps = ProcessSampler(seed=42)
base = np.array([0.6e-6, 0.3e-6])
sigma = np.array([0.02e-6, 0.01e-6])
samples = ps.sample_process_variations(5000, base, sigma)
assert abs(samples[:, 0].mean() - base[0]) < 1e-7, '[TC17] 采样高度均值偏离 FAILED'
assert abs(samples[:, 1].mean() - base[1]) < 1e-7, '[TC17] 采样宽度均值偏离 FAILED'

print("Running TC18...")
# ---- TC18: TopologyOptimizer 动态规划量化误差非负且有界 ----
np.random.seed(42)
opt = TopologyOptimizer(n_levels=8)
target = np.linspace(0, 2 * np.pi, 50)
quantized, err = opt.quantize_phase_dp(target)
assert err >= 0, '[TC18] 量化误差为负 FAILED'
assert err < 200.0, '[TC18] 量化误差过大 FAILED'

print("Running TC19...")
# ---- TC19: TopologyOptimizer 压缩感知残差小于目标范数的一半 ----
np.random.seed(42)
opt = TopologyOptimizer(n_levels=8)
M, N = 30, 80
A = np.random.randn(M, N) + 1.0j * np.random.randn(M, N)
x_true = np.zeros(N, dtype=np.complex128)
x_true[np.random.choice(N, 5, replace=False)] = np.random.randn(5) + 1.0j * np.random.randn(5)
b = A @ x_true
x_est, res = opt.compressed_inverse_design(A, b, sparsity_factor=0.3)
assert res < np.linalg.norm(b) * 0.5, '[TC19] 压缩感知残差相对过大 FAILED'

print("Running TC20...")
# ---- TC20: WavefrontTracer 光线追踪路径点数至少为 2 ----
nx, ny = 21, 21
x = np.linspace(-2e-6, 2e-6, nx)
y = np.linspace(-2e-6, 2e-6, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
phase = np.zeros_like(X)
tracer = WavefrontTracer(x, y, phase)
px, py, opl = tracer.trace_ray(-1e-6, 0.0, 1e-6, 0.0)
assert len(px) >= 2, '[TC20] 光线追踪路径点不足 FAILED'
assert np.isfinite(opl) and opl >= 0, '[TC20] 光程非有限或非负 FAILED'

print("Running TC21...")
# ---- TC21: WavefrontTracer Dijkstra 最短距离非负 ----
nx, ny = 11, 11
x = np.linspace(-1e-6, 1e-6, nx)
y = np.linspace(-1e-6, 1e-6, ny)
phase = np.zeros((nx, ny))
tracer = WavefrontTracer(x, y, phase)
dist, prev = tracer.dijkstra(0)
assert np.all(dist >= -1e-12), '[TC21] Dijkstra 距离为负 FAILED'

print("Running TC22...")
# ---- TC22: PhaseSurface 悬链面相位全部有限 ----
nx, ny = 21, 21
x = np.linspace(-1e-6, 1e-6, nx)
y = np.linspace(-1e-6, 1e-6, ny)
ps = PhaseSurface(x, y)
phi_cat = ps.catenoid_phase_profile(a_param=1.0e6)
assert np.isfinite(phi_cat).all(), '[TC22] 悬链面相位含非有限值 FAILED'

print("Running TC23...")
# ---- TC23: PhaseSurface 螺旋面相位有限且在合理范围 ----
phi_hel = ps.helicoid_phase_profile(a_param=2.0)
assert np.isfinite(phi_hel).all(), '[TC23] 螺旋面相位含非有限值 FAILED'
assert phi_hel.min() >= -np.pi * 2.01 and phi_hel.max() <= np.pi * 2.01, '[TC23] 螺旋面相位范围 FAILED'

print("Running TC24...")
# ---- TC24: PhaseSurface Laplacian 平滑后相位有限 ----
nx, ny = 11, 11
x = np.linspace(-1e-6, 1e-6, nx)
y = np.linspace(-1e-6, 1e-6, ny)
ps = PhaseSurface(x, y)
phi_disc = np.floor(np.arange(nx * ny).reshape(nx, ny) / 3.0) * (np.pi / 4.0)
phi_smooth = ps.laplacian_smooth(phi_disc, n_iter=20)
assert np.isfinite(phi_smooth).all(), '[TC24] Laplacian 平滑后相位含非有限值 FAILED'

print("Running TC25...")
# ---- TC25: ConvergenceAnalysis L2 范数非负 ----
ca = ConvergenceAnalysis()
val = ca.norm_l2(np.random.randn(10))
assert val >= 0, '[TC25] L2 范数为负 FAILED'

print("Running TC26...")
# ---- TC26: ConvergenceAnalysis 收敛阶估计接近理论二阶 ----
ca = ConvergenceAnalysis()
h = np.array([0.4, 0.2, 0.1, 0.05]) * 1e-6
err = 0.05 * h ** 2.0
p_est, C_est, r2 = ca.estimate_convergence_rate(err, h)
assert abs(p_est - 2.0) < 0.05, '[TC26] 收敛阶估计偏离 FAILED'
assert r2 > 0.999, '[TC26] 拟合 R2 不足 FAILED'

print("Running TC27...")
# ---- TC27: ConvergenceAnalysis GCI 非负 ----
ca = ConvergenceAnalysis()
gci = ca.gci_calculation(0.95, 0.94, 0.92, r=2.0, p=2.0)
assert gci >= 0, '[TC27] GCI 为负 FAILED'

print("Running TC28...")
# ---- TC28: 集成测试 main 函数返回预期键且 E_z 为有限数组 ----
from main import main
result = main()
assert 'E_z' in result, '[TC28] main 返回缺少 E_z FAILED'
assert 'generators' in result, '[TC28] main 返回缺少 generators FAILED'
assert 'quantized_phases' in result, '[TC28] main 返回缺少 quantized_phases FAILED'
assert 'stats' in result, '[TC28] main 返回缺少 stats FAILED'
assert np.isfinite(result['E_z']).all(), '[TC28] main 返回 E_z 含非有限值 FAILED'

print("\n全部 28 个测试通过!\n")
