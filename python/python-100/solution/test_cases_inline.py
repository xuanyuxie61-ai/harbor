# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================

import numpy as np

# ---- TC01: CVTMeshGenerator 生成环形扇区点集形状正确 ----
from mesh_engine import CVTMeshGenerator
np.random.seed(42)
cvt = CVTMeshGenerator(seed=42)
pts, tri = cvt.generate_in_annular_sector(0.05, 0.1, 0.0, np.pi / 3.0, 20, max_iter=5)
assert pts.shape == (20, 2), '[TC01] CVT 点集形状错误 FAILED'
assert tri.shape[1] == 3, '[TC01] CVT 三角形列数错误 FAILED'

# ---- TC02: Mesh2D 构建简单三角形并计算面积 ----
from mesh_engine import Mesh2D
mesh = Mesh2D()
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
elements = np.array([[0, 1, 2]], dtype=int)
mesh.build_from_points_triangles(nodes, elements)
areas = mesh.compute_element_areas()
assert abs(areas[0] - 0.5) < 1.0e-12, '[TC02] 三角形面积计算错误 FAILED'

# ---- TC03: Mesh2D 质量指标对正三角形合理 ----
mesh2 = Mesh2D()
nodes_eq = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]], dtype=float)
mesh2.build_from_points_triangles(nodes_eq, np.array([[0, 1, 2]], dtype=int))
metrics = mesh2.compute_quality_metrics()
assert metrics['aspect_ratio_mean'] < 2.0, '[TC03] 正三角形纵横比异常 FAILED'
assert metrics['min_angle_mean'] > 50.0, '[TC03] 正三角形最小角异常 FAILED'

# ---- TC04: Floyd-Warshall 最短路径返回有限值 ----
mesh3 = Mesh2D()
nodes_line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)
elems_line = np.array([[0, 1, 2], [1, 2, 3]], dtype=int)
mesh3.build_from_points_triangles(nodes_line, elems_line)
sp = mesh3.floyd_warshall_magnetic_path(0)
assert np.isfinite(sp[3]), '[TC04] Floyd-Warshall 最短路径非有限 FAILED'
assert sp[3] > 0.0, '[TC04] Floyd-Warshall 最短路径非正 FAILED'

# ---- TC05: NonlinearMagneticMaterial 零场磁通密度为零 ----
from material_physics import NonlinearMagneticMaterial
mat = NonlinearMagneticMaterial(mu_r_init=5000.0, B_sat=2.0)
B0 = mat.b_field(0.0)
assert abs(B0) < 1.0e-12, '[TC05] 零场B非零 FAILED'

# ---- TC06: NonlinearMagneticMaterial 磁阻率为正 ----
nu = mat.reluctivity(1.0)
assert nu > 0.0, '[TC06] 磁阻率非正 FAILED'
assert np.isfinite(nu), '[TC06] 磁阻率非有限 FAILED'

# ---- TC07: PermanentMagnet 零场退磁曲线等于剩磁 ----
from material_physics import PermanentMagnet
pm = PermanentMagnet(B_r=1.2, mu_rec=1.05)
assert abs(pm.b_field(0.0) - 1.2) < 1.0e-12, '[TC07] 永磁体零场B_r错误 FAILED'

# ---- TC08: LogNormalUncertainty 统计矩计算正确 ----
from material_physics import LogNormalUncertainty
ln = LogNormalUncertainty(mu_ln=0.0, sigma_ln=0.1)
mean_est = ln.mean()
var_est = ln.variance()
assert abs(mean_est - np.exp(0.0 + 0.5 * 0.01)) < 1.0e-10, '[TC08] 对数正态均值错误 FAILED'
assert var_est > 0.0, '[TC08] 对数正态方差非正 FAILED'

# ---- TC09: LogNormalUncertainty from_mean_variance 反解一致 ----
ln2 = LogNormalUncertainty.from_mean_variance(5000.0, 10000.0)
assert abs(ln2.mean() - 5000.0) < 1.0e-6, '[TC09] from_mean_variance 反解均值偏离 FAILED'

# ---- TC10: temperature_dependent_conductivity 20度不变 ----
from material_physics import temperature_dependent_conductivity
sigma_20 = 5.8e7
sigma_t = temperature_dependent_conductivity(sigma_20, 20.0)
assert abs(sigma_t - sigma_20) < 1.0e-12, '[TC10] 20度电导率变化 FAILED'

# ---- TC11: FEM1DRadial 零源求解结果接近零 ----
from fem1d_radial import FEM1DRadial
fem1d = FEM1DRadial(r_min=0.01, r_max=0.05, n_elements=20)
A_sol = fem1d.solve(nu_func=lambda r: 1.0 / (4e-7 * np.pi * 1000.0), source_func=lambda r: 0.0, nquad=3)
assert np.allclose(A_sol, 0.0, atol=1.0e-12), '[TC11] 1D零源求解非零 FAILED'

# ---- TC12: FEM1DRadial B场输出形状与节点一致 ----
A_test = np.linspace(0.0, 1.0, fem1d.n_nodes)
B_r = fem1d.compute_radial_b_field(A_test)
assert B_r.shape == (fem1d.n_nodes,), '[TC12] B场输出形状错误 FAILED'

# ---- TC13: FEM1DRadial 磁场储能非负 ----
W = fem1d.compute_energy(A_test, nu_func=lambda r: 1.0 / (4e-7 * np.pi * 1000.0), nquad=3)
assert W >= 0.0, '[TC13] 1D磁场储能为负 FAILED'

# ---- TC14: FEM2DAxi 小网格线性组装返回 CSR 矩阵 ----
from fem2d_axi import FEM2DAxi
mesh_small = Mesh2D()
mesh_small.build_from_points_triangles(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
    np.array([[0, 1, 2]], dtype=int)
)
fem2d = FEM2DAxi(mesh_small)
K, F = fem2d.assemble_linear(nu_func=lambda x, y: 1.0, source_func=lambda x, y: 1.0)
assert K.shape == (3, 3), '[TC14] 2D刚度矩阵形状错误 FAILED'
assert F.shape == (3,), '[TC14] 2D载荷向量形状错误 FAILED'

# ---- TC15: GaussLegendreQuadrature 5点精确积分4次多项式 ----
from quadrature_engine import GaussLegendreQuadrature
glq = GaussLegendreQuadrature(n_points=5)
integral_val = glq.integrate_1d(lambda x: x ** 4, a=-1.0, b=1.0)
assert abs(integral_val - 2.0 / 5.0) < 1.0e-14, '[TC15] Gauss-Legendre 积分精度 FAILED'

# ---- TC16: TriangleGaussianQuadrature 参考三角形积分常数函数 ----
from quadrature_engine import TriangleGaussianQuadrature
tgq = TriangleGaussianQuadrature(order=7)
ref_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
integral_tri = tgq.integrate_triangle(lambda pts: np.ones(pts.shape[0]), ref_tri)
assert abs(integral_tri - 0.5) < 1.0e-12, '[TC16] 三角形Gauss求积常数函数精度 FAILED'

# ---- TC17: TriangleMonteCarlo 常数函数积分等于面积 ----
from quadrature_engine import TriangleMonteCarlo
tmc = TriangleMonteCarlo(seed=42)
mc_res = tmc.integrate(lambda pts: np.ones(pts.shape[0]), ref_tri, n_samples=20000)
assert abs(mc_res['estimate'] - 0.5) < 0.01, '[TC17] 蒙特卡洛常数积分偏离面积 FAILED'

# ---- TC18: MomentMethodQuadrature 矩方法积分x^2 ----
from quadrature_engine import MomentMethodQuadrature
moments = [1.0, 0.5, 1.0 / 3.0, 0.25, 1.0 / 5.0, 1.0 / 6.0, 1.0 / 7.0]
mmq = MomentMethodQuadrature(moments)
integral_mm = mmq.integrate(lambda x: x ** 2)
assert abs(integral_mm - 1.0 / 3.0) < 1.0e-10, '[TC18] 矩方法积分x^2错误 FAILED'

# ---- TC19: ConditionEstimator hager_estimate 单位矩阵条件数为1 ----
from numerical_analysis import ConditionEstimator
I = np.eye(5)
cond_est = ConditionEstimator.hager_estimate(I)
assert abs(cond_est - 1.0) < 1.0e-8, '[TC19] Hager条件数单位矩阵估计错误 FAILED'

# ---- TC20: FEMErrorEstimator peclet_number 零速度为零 ----
from numerical_analysis import FEMErrorEstimator
pe = FEMErrorEstimator.peclet_number(velocity=0.0, h_max=0.01, diffusivity=1.0e-3)
assert abs(pe) < 1.0e-14, '[TC20] Peclet数零速度非零 FAILED'

# ---- TC21: FEMErrorEstimator 单位矩阵正定 ----
spd = FEMErrorEstimator.check_stiffness_positive_definite(np.eye(4))
assert spd['is_spd'] == True, '[TC21] 单位矩阵正定性判断错误 FAILED'
assert abs(spd['cond_2'] - 1.0) < 1.0e-10, '[TC21] 单位矩阵谱条件数错误 FAILED'

# ---- TC22: RotorDynamics cogging_torque 周期性验证 ----
from rotor_multiphysics import RotorDynamics
rotor = RotorDynamics(J=0.01, B_d=0.001, tau_load=5.0, n_slots=6)
tau1 = rotor.cogging_torque(0.0)
tau2 = rotor.cogging_torque(2.0 * np.pi / 6.0)
assert abs(tau1 - tau2) < 1.0e-12, '[TC22] 齿槽转矩周期性错误 FAILED'

# ---- TC23: NonlinearPeriodEstimator urabe_period 线性极限为2pi ----
from rotor_multiphysics import NonlinearPeriodEstimator
T0 = NonlinearPeriodEstimator.urabe_period(0.0)
assert abs(T0 - 2.0 * np.pi) < 1.0e-12, '[TC23] Urabe周期线性极限错误 FAILED'

# ---- TC24: NonlinearPeriodEstimator estimate_motor_fault_period 返回有限正值 ----
T_fault = NonlinearPeriodEstimator.estimate_motor_fault_period(
    eccentricity=0.05e-3, nominal_airgap=1.0e-3, omega_0=100.0
)
assert T_fault > 0.0, '[TC24] 故障周期非正 FAILED'
assert np.isfinite(T_fault), '[TC24] 故障周期非有限 FAILED'

# ---- TC25: FEM3DProjection 拉伸后3D节点数为2倍2D节点数 ----
from fem3d_projection import FEM3DProjection
nodes_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
elems_2d = np.array([[0, 1, 2]], dtype=int)
fem3d = FEM3DProjection(nodes_2d, elems_2d, axial_length=0.15)
assert fem3d.n_node_3d == 6, '[TC25] 3D投影节点数错误 FAILED'
assert fem3d.n_elem_3d == 3, '[TC25] 3D投影单元数错误 FAILED'

# ---- TC26: build_motor_material_library 返回包含5种材料的字典 ----
from material_physics import build_motor_material_library
lib = build_motor_material_library()
assert len(lib) == 5, '[TC26] 材料库条目数错误 FAILED'
assert 'stator_core' in lib, '[TC26] 材料库缺少定子铁芯 FAILED'

# ---- TC27: FEM1DRadial Thomas算法与numpy结果一致 ----
adiag = np.array([2.0, 3.0, 4.0], dtype=float)
aleft = np.array([0.0, 1.0, 1.0], dtype=float)
arite = np.array([1.0, 1.0, 0.0], dtype=float)
fvec = np.array([1.0, 2.0, 3.0], dtype=float)
u_thomas = FEM1DRadial.solve_tridiagonal(adiag, aleft, arite, fvec)
A_full = np.diag(adiag) + np.diag(aleft[1:], k=-1) + np.diag(arite[:-1], k=1)
u_np = np.linalg.solve(A_full, fvec)
assert np.allclose(u_thomas, u_np, atol=1.0e-12), '[TC27] Thomas算法与numpy不一致 FAILED'

# ---- TC28: Mesh2D tag_elements_by_region 标签正确 ----
mesh_tag = Mesh2D()
mesh_tag.build_from_points_triangles(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=float),
    np.array([[0, 1, 2]], dtype=int)
)
mesh_tag.tag_elements_by_region({1: lambda x, y: True})
assert mesh_tag.elem_tags[0] == 1, '[TC28] 单元标签未正确设置 FAILED'

# ---- TC29: GaussLegendreQuadrature 积分上下限检查 ----
try:
    glq.integrate_1d(lambda x: x, a=1.0, b=1.0)
    assert False, '[TC29] 积分上下限相等未抛异常 FAILED'
except ValueError:
    pass

# ---- TC30: FEMErrorEstimator max_element_size 计算正确 ----
hmax = FEMErrorEstimator.max_element_size(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
    np.array([[0, 1, 2]], dtype=int)
)
assert abs(hmax - np.sqrt(2.0)) < 1.0e-12, '[TC30] 最大单元尺寸计算错误 FAILED'
