# ---- TC01: triangle_area计算直角三角形面积正确 ----
from mesh_quality import triangle_area
area = triangle_area(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0]))
assert abs(area - 1.0) < 1e-10, '[TC01] triangle_area计算直角三角形面积正确 FAILED'

# ---- TC02: 等边三角形Q度量等于1 ----
from mesh_quality import q_measure
nodes_eq = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
tri_eq = np.array([[0, 1, 2]])
q = q_measure(tri_eq, nodes_eq)
assert abs(q[0] - 1.0) < 0.01, '[TC02] 等边三角形Q度量等于1 FAILED'

# ---- TC03: 生成5x5网格应有32个三角形和25个节点 ----
from acoustic_fem_mesh import generate_acoustic_domain
nodes, triangles = generate_acoustic_domain(nx=5, ny=5)
assert triangles.shape[0] == 32, '[TC03] 生成5x5网格应有32个三角形和25个节点 FAILED'
assert nodes.shape[0] == 25, '[TC03] 生成5x5网格应有32个三角形和25个节点 FAILED'

# ---- TC04: 邻接结构满足对称性i在j的邻居中当且仅当j在i的邻居中 ----
from acoustic_fem_mesh import build_adjacency_structure
adj = build_adjacency_structure(nodes, triangles)
for i, neighbors in adj.items():
    for j in neighbors:
        assert i in adj[j], '[TC04] 邻接结构满足对称性i在j的邻居中当且仅当j在i的邻居中 FAILED'

# ---- TC05: 分段线性插值对线性函数精确重构 ----
from helmholtz_solver import pwl_interp_1d
xd = np.array([0.0, 1.0, 2.0, 3.0])
yd = np.array([0.0, 1.0, 2.0, 3.0])
xi = np.array([0.5, 1.5, 2.5])
yi = pwl_interp_1d(xd, yd, xi)
assert np.allclose(yi, xi), '[TC05] 分段线性插值对线性函数精确重构 FAILED'

# ---- TC06: 1D Helmholtz解数组长度等于n+2 ----
x_h, p_h = solve_helmholtz_1d(50, 10.0, np.zeros(50), xlim=(0.0, 1.0))
assert len(x_h) == 52, '[TC06] 1D Helmholtz解数组长度等于n+2 FAILED'
assert len(p_h) == 52, '[TC06] 1D Helmholtz解数组长度等于n+2 FAILED'

# ---- TC07: 刚性ODE精确解在t=0处等于lambda平方除以1加lambda平方 ----
from transducer_dynamics import stiff_ode_exact
y0 = stiff_ode_exact(0.0, lam=-5.0)
expected = 25.0 / 26.0
assert abs(y0 - expected) < 1e-10, '[TC07] 刚性ODE精确解在t=0处等于lambda平方除以1加lambda平方 FAILED'

# ---- TC08: 刚性ODE求解器400步误差小于50步误差且收敛阶大于1.5 ----
from transducer_dynamics import verify_stiff_solver_fix
result = verify_stiff_solver_fix(lam=-5.0, n_steps_list=[50, 100, 200, 400])
assert result['l2_errors'][-1] < result['l2_errors'][0], '[TC08] 刚性ODE求解器400步误差小于50步误差且收敛阶大于1.5 FAILED'
assert np.mean(result['convergence_orders']) > 1.5, '[TC08] 刚性ODE求解器400步误差小于50步误差且收敛阶大于1.5 FAILED'

# ---- TC09: Jacobi椭圆函数m=0退化为sin和cos ----
from nonlinear_acoustics import sncndn
sn, cn, dn = sncndn(0.5, 0.0)
assert abs(sn - np.sin(0.5)) < 1e-10, '[TC09] Jacobi椭圆函数m=0退化为sin和cos FAILED'
assert abs(cn - np.cos(0.5)) < 1e-10, '[TC09] Jacobi椭圆函数m=0退化为sin和cos FAILED'
assert abs(dn - 1.0) < 1e-10, '[TC09] Jacobi椭圆函数m=0退化为sin和cos FAILED'

# ---- TC10: 冲击波形成在t=0时等于初始线性条件 ----
from nonlinear_acoustics import shock_wave_formation
x_shock = np.linspace(0, 1, 100)
u0 = shock_wave_formation(x_shock, 0.0, u0=1.0)
expected = x_shock - 0.5
assert np.allclose(u0, expected), '[TC10] 冲击波形成在t=0时等于初始线性条件 FAILED'

# ---- TC11: Taylor-Green涡t=0时v速度场等于负的cosx乘siny ----
from flow_acoustic_coupling import taylor_green_vortex
X, Y = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, np.pi, 20))
u, v, p = taylor_green_vortex(X, Y, t=0.0)
assert np.allclose(v, -np.cos(X) * np.sin(Y)), '[TC11] Taylor-Green涡t=0时v速度场等于负的cosx乘siny FAILED'

# ---- TC12: Mach数场非负且最大值等于速度模除以声速 ----
from flow_acoustic_coupling import mach_number_field
u_test = np.array([[1.0, 2.0], [0.0, 1.5]])
v_test = np.array([[0.0, 1.0], [0.5, 0.0]])
ma = mach_number_field(u_test, v_test, c0=1000.0)
assert np.all(ma >= 0), '[TC12] Mach数场非负且最大值等于速度模除以声速 FAILED'
expected_max = np.sqrt(5.0) / 1000.0
assert abs(np.max(ma) - expected_max) < 1e-10, '[TC12] Mach数场非负且最大值等于速度模除以声速 FAILED'

# ---- TC13: Stokes-Einstein扩散系数为有限正数 ----
from microbubble_diffusion import diffusion_coefficient
D = diffusion_coefficient(1e-6)
assert D > 0, '[TC13] Stokes-Einstein扩散系数为有限正数 FAILED'
assert np.isfinite(D), '[TC13] Stokes-Einstein扩散系数为有限正数 FAILED'

# ---- TC14: 声辐射力与微泡半径立方成正比 ----
from microbubble_diffusion import acoustic_radiation_force
F1 = acoustic_radiation_force(frequency=5e6, pressure_amplitude=1e5, bubble_radius=1e-6)
F2 = acoustic_radiation_force(frequency=5e6, pressure_amplitude=1e5, bubble_radius=2e-6)
assert F1 > 0, '[TC14] 声辐射力与微泡半径立方成正比 FAILED'
assert abs(F2 / F1 - 8.0) < 0.01, '[TC14] 声辐射力与微泡半径立方成正比 FAILED'

# ---- TC15: Haar单步分解前后能量守恒 ----
from wavelet_denoising import haar_step_1d
signal = np.array([3.0, 1.0, 2.0, 4.0])
approx, detail = haar_step_1d(signal)
E_in = np.sum(signal**2)
E_out = np.sum(approx**2) + np.sum(detail**2)
assert abs(E_in - E_out) < 1e-10, '[TC15] Haar单步分解前后能量守恒 FAILED'

# ---- TC16: Haar小波逆变换完美重构原始信号 ----
from wavelet_denoising import haar_1d, haar_1d_inverse
np.random.seed(42)
signal = np.random.randn(64)
coeffs, details = haar_1d(signal, n_levels=3)
reconstructed = haar_1d_inverse(coeffs, 3)
assert np.allclose(signal, reconstructed), '[TC16] Haar小波逆变换完美重构原始信号 FAILED'

# ---- TC17: PCA均值向量等于按轴0求平均 ----
from pca_feature_extraction import compute_mean_face
np.random.seed(42)
images = np.random.randn(10, 20)
mean_face = compute_mean_face(images)
assert mean_face.shape == (20,), '[TC17] PCA均值向量等于按轴0求平均 FAILED'
assert np.allclose(mean_face, np.mean(images, axis=0)), '[TC17] PCA均值向量等于按轴0求平均 FAILED'

# ---- TC18: PCA重建误差PSNR为有限值 ----
from pca_feature_extraction import compute_reconstruction_error
orig = np.array([1.0, 2.0, 3.0, 4.0])
recon = np.array([1.1, 1.9, 3.1, 3.9])
err = compute_reconstruction_error(orig, recon)
assert np.isfinite(err['psnr']), '[TC18] PCA重建误差PSNR为有限值 FAILED'

# ---- TC19: Hanning窗函数值均在0到1闭区间内 ----
from ultrasound_beamforming import hanning_window
w = hanning_window(16)
assert np.all(w >= 0), '[TC19] Hanning窗函数值均在0到1闭区间内 FAILED'
assert np.all(w <= 1), '[TC19] Hanning窗函数值均在0到1闭区间内 FAILED'

# ---- TC20: 发射聚焦延迟关于阵列中心对称 ----
from ultrasound_beamforming import transmit_focus_delay
delays = transmit_focus_delay(16, 0.3e-3, focus_depth=0.04)
n = len(delays)
for i in range(n // 2):
    assert abs(delays[i] - delays[n - 1 - i]) < 1e-14, '[TC20] 发射聚焦延迟关于阵列中心对称 FAILED'

# ---- TC21: Levenshtein距离相同序列结果为0 ----
from inverse_tomography import levenshtein_distance
dist = levenshtein_distance([1, 2, 3], [1, 2, 3])
assert dist == 0, '[TC21] Levenshtein距离相同序列结果为0 FAILED'

# ---- TC22: 投影矩阵形状为2*n_rays行和n_pixels_x乘n_pixels_y列 ----
from inverse_tomography import build_projection_matrix
A, angles = build_projection_matrix(n_rays=5, n_pixels_x=4, n_pixels_y=4)
assert A.shape == (10, 16), '[TC22] 投影矩阵形状为2*n_rays行和n_pixels_x乘n_pixels_y列 FAILED'

# ---- TC23: SVD反演输出长度等于矩阵列数 ----
from inverse_tomography import solve_tomography_svd
np.random.seed(42)
A_test = np.random.randn(10, 16)
tt = np.random.randn(10)
m = solve_tomography_svd(A_test, tt)
assert len(m) == 16, '[TC23] SVD反演输出长度等于矩阵列数 FAILED'

# ---- TC24: 单位矩阵可识别性比率为1 ----
from inverse_tomography import analyze_system_identifiability
info = analyze_system_identifiability(np.eye(5))
assert abs(info['identifiability_ratio'] - 1.0) < 1e-10, '[TC24] 单位矩阵可识别性比率为1 FAILED'
assert info['rank'] == 5, '[TC24] 单位矩阵可识别性比率为1 FAILED'

# ---- TC25: 主程序main返回包含10个键的结果字典 ----
results = main()
assert isinstance(results, dict), '[TC25] 主程序main返回包含10个键的结果字典 FAILED'
assert len(results) == 10, '[TC25] 主程序main返回包含10个键的结果字典 FAILED'
