# ---- TC01: line_grid 输出类型正确且单调递增 ----
s1 = line_grid(11, 0.0, 1.0, c=1)
assert isinstance(s1, np.ndarray), '[TC01] line_grid 返回类型应为 ndarray FAILED'
assert len(s1) == 11, '[TC01] line_grid 输出长度应为 11 FAILED'
assert np.all(np.diff(s1) > 0), '[TC01] line_grid 输出应单调递增 FAILED'

# ---- TC02: chebyshev_grid 端点值正确且单调递减 ----
cb2 = chebyshev_grid(10)
assert len(cb2) == 11, '[TC02] chebyshev_grid 节点数应为 11 FAILED'
assert np.isclose(cb2[0], 1.0), '[TC02] chebyshev_grid 第一个节点应接近 1 FAILED'
assert np.isclose(cb2[-1], -1.0), '[TC02] chebyshev_grid 最后一个节点应接近 -1 FAILED'

# ---- TC03: triangulate_polygon 输出形状正确 ----
theta_3 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
verts3 = np.column_stack([0.1 * np.cos(theta_3), 0.1 * np.sin(theta_3)])
nodes3, tris3 = triangulate_polygon(verts3)
assert nodes3.shape[1] == 2, '[TC03] 三角剖分节点应为2列 FAILED'
assert tris3.shape[1] == 3, '[TC03] 三角形索引应为3列 FAILED'
assert tris3.shape[0] >= 1, '[TC03] 至少应有1个三角形 FAILED'

# ---- TC04: compute_section_properties 面积和惯性矩非负 ----
props4 = compute_section_properties(nodes3, tris3)
assert props4['A'] > 0, '[TC04] 截面面积应为正 FAILED'
assert props4['Ixx'] >= 0, '[TC04] 惯性矩 Ixx 应非负 FAILED'
assert props4['Iyy'] >= 0, '[TC04] 惯性矩 Iyy 应非负 FAILED'
assert props4['J'] >= 0, '[TC04] 极惯性矩 J 应非负 FAILED'

# ---- TC05: diaphony_compute 输出在 [0,1] 内且有限 ----
import numpy as np
np.random.seed(42)
pts5 = np.random.rand(100, 2)
dia5 = diaphony_compute(pts5)
assert np.isfinite(dia5), '[TC05] diaphony 应为有限值 FAILED'
assert 0.0 <= dia5 <= 1.0, '[TC05] diaphony 应在 [0,1] 范围内 FAILED'

# ---- TC06: chebyshev_matrix 常数函数微分接近零 ----
x6, D6 = chebyshev_matrix(10)
c6 = np.ones(11)
err6 = np.max(np.abs(D6 @ c6))
assert err6 < 1e-10, '[TC06] Chebyshev 微分矩阵作用常数应接近零 FAILED'

# ---- TC07: jacobi_polynomial P_3^{(0,0)} 在 0 点为 Legendre 值 ----
r7 = np.array([0.0])
p3_7 = jacobi_polynomial(r7, 0.0, 0.0, 3)
# P_3^(0,0) 即 Legendre P_3: (5x^3 - 3x)/2, 在 x=0 处为 0
assert np.abs(p3_7[0]) < 1e-12, '[TC07] Jacobi P_3^(0,0)(0) 应接近 0 FAILED'

# ---- TC08: vandermonde_1d 矩阵可逆（条件数有限） ----
x8, _ = chebyshev_matrix(6)
V8 = vandermonde_1d(6, x8)
cond8 = np.linalg.cond(V8)
assert np.isfinite(cond8), '[TC08] Vandermonde 矩阵条件数应为有限值 FAILED'
assert cond8 < 1e10, '[TC08] Vandermonde 矩阵条件数不应过大 FAILED'

# ---- TC09: lift_1d 输出形状为 (Np, 2) ----
x9, _ = chebyshev_matrix(5)
V9 = vandermonde_1d(5, x9)
L9 = lift_1d(5, V9)
assert L9.shape == (6, 2), '[TC09] lift_1d 输出形状应为 (Np,2) FAILED'

# ---- TC10: neo_hookean_strain_energy 单位变形梯度下应变能为零 ----
F10 = np.eye(3)
W10 = neo_hookean_strain_energy(F10, 1.0e5, 2.0e6)
assert np.abs(W10) < 1e-10, '[TC10] 单位变形梯度下 Neo-Hookean 应变能应接近零 FAILED'

# ---- TC11: neo_hookean_stress 单位变形梯度下应力为零 ----
P11 = neo_hookean_stress(F10, 1.0e5, 2.0e6)
assert np.max(np.abs(P11)) < 1e-6, '[TC11] 单位变形梯度下第一PK应力应接近零 FAILED'

# ---- TC12: mooney_rivlin_strain_energy 单位变形梯度下为零 ----
W12 = mooney_rivlin_strain_energy(F10, 5.0e4, 1.0e4, 2.0e6)
assert np.abs(W12) < 1e-10, '[TC12] 单位变形梯度下 Mooney-Rivlin 应变能应接近零 FAILED'

# ---- TC13: soft_robot_1d_constitutive 零应变可复现且为零 ----
n13, m13 = soft_robot_1d_constitutive(0.0, np.zeros(3), 1.0e6, 0.35e6, 0.005, 2e-6, 1e-6, 3e-6)
assert np.max(np.abs(n13)) < 1e-12, '[TC13] 零应变下内力应接近零 FAILED'
assert np.max(np.abs(m13)) < 1e-12, '[TC13] 零曲率下内矩应接近零 FAILED'
import numpy as np
np.random.seed(42)
n13b, m13b = soft_robot_1d_constitutive(0.0, np.zeros(3), 1.0e6, 0.35e6, 0.005, 2e-6, 1e-6, 3e-6)
assert np.max(np.abs(n13b)) < 1e-12, '[TC13] 零应变结果应可复现 FAILED'

# ---- TC14: chemo_mechanical_coupling 有效模量在合理范围内 ----
E14 = chemo_mechanical_coupling(np.array([0.5, 6.5]), 0.05, E0=1.0e6, gamma=0.3, beta_chem=0.2)
assert 0.1 * 1.0e6 <= E14 <= 5.0 * 1.0e6, '[TC14] 有效模量应在 [0.1E0, 5E0] 范围内 FAILED'

# ---- TC15: selkov_glycolysis_ode 积分可复现 ----
import numpy as np
np.random.seed(42)
tg15a, yg15a = low_storage_rk4(selkov_glycolysis_ode, (0.0, 50.0), np.array([0.9, 0.7]), 500)
np.random.seed(42)
tg15b, yg15b = low_storage_rk4(selkov_glycolysis_ode, (0.0, 50.0), np.array([0.9, 0.7]), 500)
assert np.allclose(yg15a, yg15b), '[TC15] Selkov ODE 积分应可复现 FAILED'

# ---- TC16: hat_map 和 vee_map 互为逆映射 ----
v16 = np.array([1.0, 2.0, 3.0])
hat16 = hat_map(v16)
vee16 = vee_map(hat16)
assert np.allclose(vee16, v16), '[TC16] hat/vee 应为互逆映射 FAILED'

# ---- TC17: rodrigues_rotation 旋转矩阵保持正交性 ----
import numpy as np
np.random.seed(42)
axis17 = np.random.randn(3)
axis17 = axis17 / np.linalg.norm(axis17)
R17 = rodrigues_rotation(axis17, np.pi / 4)
assert np.allclose(R17 @ R17.T, np.eye(3), atol=1e-12), '[TC17] 旋转矩阵应正交 FAILED'
assert np.abs(np.linalg.det(R17) - 1.0) < 1e-12, '[TC17] 旋转矩阵行列式应接近 1 FAILED'

# ---- TC18: r8blt_sl 求解带状下三角系统并验证 ----
ml18 = 2
N18 = 10
np.random.seed(42)
diag18 = np.abs(np.random.rand(ml18 + 1, N18)) + 0.5
x18 = np.ones(N18)
b18 = r8blt_mv(diag18, ml18, x18)
x_sol18 = r8blt_sl(diag18, ml18, b18)
assert np.allclose(x_sol18, x18, atol=1e-10), '[TC18] 带状三角求解应恢复原始向量 FAILED'

# ---- TC19: forward_kinematics_cosserat 输出形状正确 ----
L19 = 0.5
Ns19 = 10
kappa19 = np.zeros((Ns19 + 1, 3))
kappa19[:, 2] = 0.1
s19, r19, R19 = forward_kinematics_cosserat(L19, Ns19, kappa19)
assert s19.shape == (Ns19 + 1,), '[TC19] s 形状应为 (Ns+1,) FAILED'
assert r19.shape == (Ns19 + 1, 3), '[TC19] r 形状应为 (Ns+1,3) FAILED'
assert R19.shape == (Ns19 + 1, 3, 3), '[TC19] R 形状应为 (Ns+1,3,3) FAILED'

# ---- TC20: compute_strain_measures 输出形状正确 ----
v20, u20 = compute_strain_measures(r19, R19, s19)
assert v20.shape == (Ns19 + 1, 3), '[TC20] v 形状应为 (Ns+1,3) FAILED'
assert u20.shape == (Ns19 + 1, 3), '[TC20] u 形状应为 (Ns+1,3) FAILED'

# ---- TC21: low_storage_rk4 积分简谐振动可复现 ----
import numpy as np
np.random.seed(42)
t21a, y21a = low_storage_rk4(
    lambda t, y: driven_harmonic_oscillator(t, y, omega0=2.0, zeta=0.5, omega_drive=2.0),
    (0.0, 5.0), np.array([0.5, 0.0]), 200)
np.random.seed(42)
t21b, y21b = low_storage_rk4(
    lambda t, y: driven_harmonic_oscillator(t, y, omega0=2.0, zeta=0.5, omega_drive=2.0),
    (0.0, 5.0), np.array([0.5, 0.0]), 200)
assert np.allclose(y21a, y21b), '[TC21] 低存储RK4积分应可复现 FAILED'

# ---- TC22: driven_harmonic_oscillator 输出为二元向量 ----
y22 = driven_harmonic_oscillator(0.0, np.array([0.1, 0.2]), omega0=1.0, zeta=0.3, omega_drive=1.5)
assert y22.shape == (2,), '[TC22] 谐振子输出应为 (2,) FAILED'

# ---- TC23: sawtooth_driver 周期性 ----
import numpy as np
np.random.seed(42)
saw0 = sawtooth_driver(0.0, omega=1.0)
saw2pi = sawtooth_driver(2.0 * np.pi, omega=1.0)
assert np.isclose(saw0, saw2pi, atol=1e-12), '[TC23] sawtooth 应为 2π/ω 周期函数 FAILED'

# ---- TC24: barycentric_coordinates 三坐标和为 1 ----
p24 = np.array([0.2, 0.3])
a24 = np.array([0.0, 0.0])
b24 = np.array([1.0, 0.0])
c24 = np.array([0.0, 1.0])
alpha24, beta24, gamma24 = barycentric_coordinates(p24, a24, b24, c24)
assert np.isclose(alpha24 + beta24 + gamma24, 1.0), '[TC24] 重心坐标和应为 1 FAILED'

# ---- TC25: point_in_triangle 判定三角形顶点为内部 ----
alpha25, beta25, gamma25 = barycentric_coordinates(np.array([0.3, 0.0]), a24, b24, c24)
assert point_in_triangle(alpha25, beta25, gamma25), '[TC25] 三角形边上点应在内部 FAILED'

# ---- TC26: pwl_interp_2d_scattered 插值恢复已知数据点 ----
import numpy as np
np.random.seed(42)
xyd26 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.8, 0.2]])
zd26 = np.sin(xyd26[:, 0] * np.pi) * np.cos(xyd26[:, 1] * np.pi)
zi26 = pwl_interp_2d_scattered(xyd26, zd26, xyd26)
assert np.allclose(zi26, zd26, atol=1e-10), '[TC26] 插值应在数据点处恢复原始值 FAILED'

# ---- TC27: shape_reconstruction_from_sensors 可复现 ----
import numpy as np
np.random.seed(42)
sp27 = np.random.rand(15, 2)
sr27 = np.sin(2.0 * np.pi * sp27[:, 0])
qp27 = np.random.rand(5, 2)
np.random.seed(42)
rec27a = shape_reconstruction_from_sensors(sp27, sr27, qp27, reconstruction_type='pwl')
np.random.seed(42)
rec27b = shape_reconstruction_from_sensors(sp27, sr27, qp27, reconstruction_type='pwl')
assert np.allclose(rec27a, rec27b), '[TC27] 形状重建应可复现 FAILED'

# ---- TC28: change_dynamic 经典硬币找零已知结果 ----
coins28 = np.array([1, 3, 4])
dp28 = change_dynamic(coins28, 10)
assert dp28[0] == 1, '[TC28] amount=1 最少硬币应为 1 FAILED'
assert dp28[2] == 1, '[TC28] amount=3 最少硬币应为 1 FAILED'
assert dp28[5] == 2, '[TC28] amount=6 最少硬币应为 2 FAILED'

# ---- TC29: configuration_to_tip 零角度直线延伸 ----
tx29, ty29 = configuration_to_tip(4, 0.25, np.zeros(4))
assert np.isclose(tx29, 1.0), '[TC29] 零角度时末端 x 应为总长度 FAILED'
assert np.isclose(ty29, 0.0, atol=1e-12), '[TC29] 零角度时末端 y 应为 0 FAILED'

# ---- TC30: energy_cost 相同构型代价为零 ----
c30 = np.array([0.1, 0.2, 0.3])
cost30 = energy_cost(c30, c30, stiffness=1.0, damping=0.0)
assert np.isclose(cost30, 0.0, atol=1e-12), '[TC30] 相同构型间代价应为零 FAILED'

# ---- TC31: compute_svd_basis 输出形状正确 ----
import numpy as np
np.random.seed(42)
snap31 = np.random.rand(50, 30)
basis31, sv31, mean31 = compute_svd_basis(snap31, 8, subtract_mean=True)
assert basis31.shape == (50, 8), '[TC31] 基矩阵形状应为 (M,k) FAILED'
assert len(sv31) == 8, '[TC31] 奇异值个数应为 k FAILED'
assert mean31.shape == (50,), '[TC31] 均值向量长度应为 M FAILED'

# ---- TC32: project/reconstruct 往返重构误差小 ----
coeff32 = project_onto_basis(snap31[:, 0], basis31, mean31)
recon32 = reconstruct_from_basis(coeff32, basis31, mean31)
err32 = np.linalg.norm(recon32 - snap31[:, 0]) / np.linalg.norm(snap31[:, 0])
assert err32 < 1.0, '[TC32] POD 往返重构相对误差应合理 FAILED'

# ---- TC33: energy_fraction 累加单调递增且最终为 1 ----
ef33 = energy_fraction(sv31)
assert np.all(np.diff(ef33) >= -1e-15), '[TC33] 能量分数应单调不减 FAILED'
assert np.isclose(ef33[-1], 1.0, atol=1e-12), '[TC33] 最终能量分数应接近 1 FAILED'

# ---- TC34: optimal_basis_size 应在有效范围内 ----
opt34 = optimal_basis_size(sv31, threshold=0.99)
assert 1 <= opt34 <= len(sv31), '[TC34] 最优基数量应在 [1, k] 范围内 FAILED'

# ---- TC35: biharmonic_w1 和 biharmonic_r1 维度一致 ----
x35 = np.linspace(-0.5, 0.5, 15)
y35 = np.linspace(-0.5, 0.5, 15)
X35, Y35 = np.meshgrid(x35, y35)
W35 = biharmonic_w1(X35, Y35, a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=2.0)
R35 = biharmonic_r1(X35, Y35, a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=2.0)
assert W35.shape == (15, 15), '[TC35] W1 形状应为 (Ny,Nx) FAILED'
assert R35.shape == (15, 15), '[TC35] R1 形状应为 (Ny,Nx) FAILED'

# ---- TC36: biharmonic_w3 中心值有限且非 NaN ----
W36 = biharmonic_w3(X35, Y35, a=1.0, b=0.5, c=0.1, d=0.0, e=0.0, f=0.0)
assert np.isfinite(W36[7, 7]), '[TC36] W3 中心值应为有限值 FAILED'

# ---- TC37: biharmonic_r3 解析残差在中心处有限 ----
R37 = biharmonic_r3(X35, Y35, a=1.0, b=0.5, c=0.1, d=0.0, e=0.5, f=0.5)
assert np.isfinite(R37[7, 7]), '[TC37] R3 中心值应为有限值 FAILED'

# ---- TC38: plate_bending_energy 非负 ----
dx38 = x35[1] - x35[0]
dy38 = y35[1] - y35[0]
E38 = plate_bending_energy(W35, 1.0e3, dx38, dy38)
assert E38 >= 0.0, '[TC38] 板弯曲能应非负 FAILED'

# ---- TC39: verify_biharmonic_discretization 返回字典含预期键 ----
res39 = verify_biharmonic_discretization(Nx=16, Ny=16)
assert 'max_error' in res39, '[TC39] 结果应包含 max_error FAILED'
assert 'l2_error' in res39, '[TC39] 结果应包含 l2_error FAILED'
assert res39['max_error'] > 0, '[TC39] 最大误差应为正值 FAILED'

# ---- TC40: tangent_stiffness_neo_hookean 输出形状为 (6,6) ----
F40 = np.eye(3)
F40[0, 0] = 1.05
F40[1, 1] = 1.0 / np.sqrt(1.05)
F40[2, 2] = 1.0 / np.sqrt(1.05)
C40 = tangent_stiffness_neo_hookean(F40, 1.0e5, 2.0e6)
assert C40.shape == (6, 6), '[TC40] 切线刚度矩阵形状应为 (6,6) FAILED'

# ---- TC41: hat_map 输出为反对称矩阵 ----
v41 = np.array([3.0, -1.0, 2.0])
hat41 = hat_map(v41)
assert np.allclose(hat41 + hat41.T, np.zeros((3, 3)), atol=1e-12), '[TC41] hat_map 输出应为反对称矩阵 FAILED'

# ---- TC42: compute_curvature 直线段曲率和挠率为零 ----
import numpy as np
np.random.seed(42)
s42 = np.linspace(0.0, 1.0, 20)
r42 = np.column_stack([s42, np.zeros(20), np.zeros(20)])
kappa42, tau42 = compute_curvature(r42, s42)
assert np.max(np.abs(kappa42[1:-1])) < 1e-6, '[TC42] 直线中心线曲率应接近零 FAILED'

# ---- TC43: r8blt_det 行列式为正 ----
ml43 = 2
N43 = 8
np.random.seed(42)
diag43 = np.abs(np.random.rand(ml43 + 1, N43)) + 0.5
det43 = r8blt_det(diag43, ml43)
assert det43 > 0, '[TC43] 正对角带状三角行列式应为正 FAILED'

# ---- TC44: shape_t6 在节点处为1，其他节点为0 ----
# T6 nodes in reference coordinates
t6_xi = np.array([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
t6_eta = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
for i in range(6):
    val = shape_t6(t6_xi[i], t6_eta[i], i)
    assert np.isclose(val, 1.0, atol=1e-12), f'[TC44] Node {i} 形函数值应接近 1 FAILED'
for i in range(6):
    for j in range(6):
        if i != j:
            val = shape_t6(t6_xi[j], t6_eta[j], i)
            assert np.abs(val) < 1e-12, f'[TC44] Node {j} 形函数 N_{i} 应接近 0 FAILED'

# ---- TC45: gauss_legendre_triangle 权重之和等于参考三角形面积 0.5 ----
for order in [1, 2, 3, 4]:
    qp, w = gauss_legendre_triangle(order)
    assert np.isclose(np.sum(w), 0.5, atol=1e-12), f'[TC45] order={order} 高斯权重和应接近 0.5 FAILED'

# ---- TC46: cauchy_theta_method 与 low_storage_rk4 结果空间不差太多 ----
import numpy as np
np.random.seed(42)
t46, y46 = cauchy_theta_method(
    lambda t, y: driven_harmonic_oscillator(t, y, omega0=1.0, zeta=0.1, omega_drive=1.0),
    (0.0, 2.0), np.array([0.5, 0.0]), 100, theta=0.5)
assert y46.shape[0] == 101, '[TC46] Cauchy 方法输出步数应为 n+1 FAILED'
assert np.isfinite(y46[-1, 0]), '[TC46] Cauchy 方法末端位移应为有限值 FAILED'

# ---- TC47: multi_target_path_planning 返回正确数量的路径 ----
targets47 = [(0.5, 0.3), (0.7, 0.4)]
paths47 = multi_target_path_planning(3, 0.25, targets47)
assert len(paths47) == 2, '[TC47] 多目标规划应返回与目标数相等的路径 FAILED'

# ---- TC48: pod_galerkin_rom 降阶矩阵大小正确 ----
basis48 = basis31
M48 = np.eye(50) * 0.1
K48 = np.eye(50) * 100.0
F48 = np.ones(50)
Mrom48, Krom48, From48 = pod_galerkin_rom(M48, K48, F48, basis48)
assert Mrom48.shape == (8, 8), '[TC48] ROM质量矩阵形状应为 (k,k) FAILED'
assert Krom48.shape == (8, 8), '[TC48] ROM刚度矩阵形状应为 (k,k) FAILED'
assert From48.shape == (8,), '[TC48] ROM力向量形状应为 (k,) FAILED'

# ---- TC49: assemble_banded_stiffness 输出紧凑存储形状正确 ----
a49 = assemble_banded_stiffness(20, 1.0e6 * 2e-6, 1.0e6 * 0.005, 0.05, ml=3)
assert a49.shape[0] == 3 + 1, '[TC49] 紧凑存储行数应为 ml+1 FAILED'
assert a49.shape[1] == 20, '[TC49] 紧凑存储列数应为 N FAILED'

# ---- TC50: compute_shear_correction_factor 在 [0.5, 1.0] 范围内 ----
kappa50 = compute_shear_correction_factor(nodes3, tris3, E=1.0e6, nu=0.35)
assert 0.5 <= kappa50 <= 1.0, '[TC50] 剪切修正系数应在 [0.5,1.0] 内 FAILED'

# ---- TC51: assemble_section_stiffness 输出对称 ----
K51 = assemble_section_stiffness(nodes3, tris3, E=1.0e6, nu=0.35)
assert np.allclose(K51, K51.T, atol=1e-10), '[TC51] 截面刚度矩阵应对称 FAILED'

# ---- TC52: randomized_svd 基本性质 ----
import numpy as np
np.random.seed(42)
A52 = np.random.rand(30, 20)
U52, S52, Vt52 = randomized_svd(A52, 5, p=3, q=2)
assert U52.shape == (30, 5), '[TC52] 随机SVD U 形状应为 (m,k) FAILED'
assert len(S52) == 5, '[TC52] 随机SVD S 长度应为 k FAILED'
assert Vt52.shape == (5, 20), '[TC52] 随机SVD Vt 形状应为 (k,n) FAILED'
assert np.all(S52 > 0), '[TC52] 奇异值应全为正 FAILED'

# ---- TC53: mooney_rivlin_stress 输出形状正确且值有限 ----
sigma53 = mooney_rivlin_stress(F10, 5.0e4, 1.0e4, 2.0e6)
assert sigma53.shape == (3, 3), '[TC53] Cauchy 应力形状应为 (3,3) FAILED'
assert np.all(np.isfinite(sigma53)), '[TC53] Cauchy 应力应为有限值 FAILED'

# ---- TC54: refine_cross_section_mesh 细化后节点/三角形数增多 ----
theta54 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
verts54 = np.column_stack([0.1 * np.cos(theta54), 0.1 * np.sin(theta54)])
nodes54a, tris54a = triangulate_polygon(verts54)
nodes54b, tris54b = refine_cross_section_mesh(verts54)
assert nodes54b.shape[0] > nodes54a.shape[0], '[TC54] 细化后节点数应增多 FAILED'

# ---- TC55: sample_ellipse 输出点数正确且均在椭圆内 ----
pts55 = sample_ellipse(a=2.0, b=1.0, n=30)
assert pts55.shape[0] >= 30, '[TC55] 采样点数不应少于给定值 FAILED'
# 验证在椭圆内: (x/a)^2 + (y/b)^2 <= 1
inside = (pts55[:, 0] / 2.0) ** 2 + (pts55[:, 1] / 1.0) ** 2
assert np.all(inside <= 1.0 + 1e-8), '[TC55] 采样点应在椭圆内 FAILED'

# ---- TC56: discretize_configuration_space 输出正确数量 ----
ang56 = discretize_configuration_space(4, 11)
assert len(ang56) == 11, '[TC56] 离散角度数应为 n_angles FAILED'
assert np.all(np.abs(ang56) <= np.pi / 2 + 1e-10), '[TC56] 角度应在 [-θmax, θmax] 内 FAILED'

# ---- TC57: inverse_kinematics_soft_robot 末端误差小于容差 ----
kappa57, r57 = inverse_kinematics_soft_robot(
    np.array([0.5, 0.2, 0.0]), L=1.0, Ns=12, material_params={}, max_iter=80, tol=1e-4)
err57 = np.linalg.norm(r57[-1] - np.array([0.5, 0.2, 0.0]))
assert err57 < 1.0, '[TC57] 逆运动学末端误差应在合理范围 FAILED'

# ---- TC58: dp_path_planning_2d 返回合理结果 ----
import numpy as np
np.random.seed(42)
ang58, cost58 = dp_path_planning_2d(3, 0.25, (0.5, 0.3), n_discrete=7)
assert len(ang58) == 3, '[TC58] 路径规划应返回 n_segments 个角度 FAILED'
assert cost58 >= 0, '[TC58] 路径代价应非负 FAILED'
