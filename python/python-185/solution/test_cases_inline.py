# ---- TC01: generate_phantom_image 输出形状正确 ----
img = generate_phantom_image(64)
assert img.shape == (64, 64), '[TC01] generate_phantom_image 形状应为 (64,64) FAILED'

# ---- TC02: generate_phantom_image 值范围在 [0, 1] ----
img = generate_phantom_image(32)
assert np.min(img) >= 0.0, '[TC02] generate_phantom_image 最小值应为非负 FAILED'
assert np.max(img) <= 1.0, '[TC02] generate_phantom_image 最大值应 <= 1 FAILED'

# ---- TC03: generate_phantom_image 确定性（无随机成分） ----
img1 = generate_phantom_image(32)
img2 = generate_phantom_image(32)
assert np.array_equal(img1, img2), '[TC03] generate_phantom_image 应为确定性函数 FAILED'

# ---- TC04: add_gaussian_noise 输出形状与输入一致 ----
img = generate_phantom_image(32)
import numpy as np
np.random.seed(42)
noisy = add_gaussian_noise(img, 30.0)
assert noisy.shape == img.shape, '[TC04] add_gaussian_noise 形状应与输入一致 FAILED'

# ---- TC05: add_gaussian_noise 输出不含 NaN/Inf ----
img = generate_phantom_image(16)
import numpy as np
np.random.seed(42)
noisy = add_gaussian_noise(img, 20.0)
assert np.all(np.isfinite(noisy)), '[TC05] add_gaussian_noise 含非有限值 FAILED'

# ---- TC06: correlation_gaussian rho=0 时返回 1 ----
c = correlation_gaussian(np.array([0.0]), 0.1)
assert abs(c[0] - 1.0) < 1e-12, '[TC06] correlation_gaussian(rho=0) 应等于 1 FAILED'

# ---- TC07: correlation_gaussian rho=rho0 时返回 exp(-1) ----
c = correlation_gaussian(np.array([0.1]), 0.1)
assert abs(c[0] - np.exp(-1.0)) < 1e-12, '[TC07] correlation_gaussian(rho=rho0) 应等于 exp(-1) FAILED'

# ---- TC08: correlation_gaussian 偶函数对称性 ----
import numpy as np
np.random.seed(42)
rho = np.abs(np.random.randn(5)) * 2
c1 = correlation_gaussian(rho, 0.5)
c2 = correlation_gaussian(-rho, 0.5)
assert np.allclose(c1, c2), '[TC08] correlation_gaussian 应对称 FAILED'

# ---- TC09: soft_thresholding 小绝对值归零 ----
from cs_detector import soft_thresholding
x = np.array([0.01, -0.02, 0.0])
y = soft_thresholding(x, 0.05)
assert np.allclose(y, 0.0), '[TC09] 小值未归零 FAILED'

# ---- TC10: soft_thresholding x>lambda 正确缩减 ----
from cs_detector import soft_thresholding
x = np.array([0.5])
y = soft_thresholding(x, 0.3)
assert abs(y[0] - 0.2) < 1e-12, '[TC10] soft_thresholding 应返回 x-lambda FAILED'

# ---- TC11: soft_thresholding x<-lambda 正确缩减 ----
from cs_detector import soft_thresholding
x = np.array([-0.7])
y = soft_thresholding(x, 0.4)
assert abs(y[0] - (-0.3)) < 1e-12, '[TC11] 负值 soft_thresholding 错误 FAILED'

# ---- TC12: is_prime 已知素数 ----
from sampling_pattern import is_prime
assert is_prime(2), '[TC12] 2 应为素数 FAILED'
assert is_prime(3), '[TC12] 3 应为素数 FAILED'
assert is_prime(17), '[TC12] 17 应为素数 FAILED'
assert is_prime(97), '[TC12] 97 应为素数 FAILED'

# ---- TC13: is_prime 已知非素数 ----
from sampling_pattern import is_prime
assert not is_prime(1), '[TC13] 1 不是素数 FAILED'
assert not is_prime(4), '[TC13] 4 不是素数 FAILED'
assert not is_prime(100), '[TC13] 100 不是素数 FAILED'
assert not is_prime(0), '[TC13] 0 不是素数 FAILED'

# ---- TC14: generate_primes 前 6 个素数 ----
from sampling_pattern import generate_primes
p = generate_primes(6)
assert p == [2, 3, 5, 7, 11, 13], '[TC14] 前 6 个素数序列错误 FAILED'

# ---- TC15: build_2d_chebyshev_basis 输出形状 ----
Psi = build_2d_chebyshev_basis((16, 16), 4)
assert Psi.shape == (256, 16), '[TC15] chebyshev 基矩阵形状应为 (256, 16) FAILED'

# ---- TC16: build_2d_chebyshev_basis 列归一化 ----
Psi = build_2d_chebyshev_basis((8, 8), 3)
norms = np.linalg.norm(Psi, axis=0)
assert np.allclose(norms, 1.0, atol=1e-10), '[TC16] chebyshev 基列未归一化 FAILED'

# ---- TC17: chebyshev 系数往返重建有限且合理 ----
img = generate_phantom_image(16)
coeffs = image_to_chebyshev_coefficients(img, 8)
recon = chebyshev_coefficients_to_image(coeffs, (16, 16), 8)
assert np.all(np.isfinite(recon)), '[TC17] chebyshev 往返结果应不含 NaN/Inf FAILED'
assert recon.shape == img.shape, '[TC17] chebyshev 往返结果形状应一致 FAILED'

# ---- TC18: build_sensing_matrix_gaussian 输出形状 ----
import numpy as np
np.random.seed(42)
Phi = build_sensing_matrix_gaussian(50, 100)
assert Phi.shape == (50, 100), '[TC18] 感知矩阵形状应为 (50, 100) FAILED'

# ---- TC19: build_sensing_matrix_gaussian 列归一化 ----
import numpy as np
np.random.seed(42)
Phi = build_sensing_matrix_gaussian(30, 60, normalize=True)
col_norms = np.linalg.norm(Phi, axis=0)
assert np.allclose(col_norms, 1.0, atol=1e-10), '[TC19] 感知矩阵列未归一化 FAILED'

# ---- TC20: pyramid_unit_volume 返回 4/3 ----
vol = pyramid_unit_volume()
assert abs(vol - 4.0/3.0) < 1e-12, '[TC20] 金字塔体积应为 4/3 FAILED'

# ---- TC21: integrate_triangle_unit_monomial 解析验证 ----
from error_estimator import integrate_triangle_unit_monomial
v = integrate_triangle_unit_monomial(0, 0)
assert abs(v - 0.5) < 1e-12, '[TC21] 三角形单位积分 (0,0) 应为 0.5 FAILED'
v = integrate_triangle_unit_monomial(1, 0)
assert abs(v - 1.0/6.0) < 1e-12, '[TC21] 三角形单位积分 (1,0) 应为 1/6 FAILED'
v = integrate_triangle_unit_monomial(0, 1)
assert abs(v - 1.0/6.0) < 1e-12, '[TC21] 三角形单位积分 (0,1) 应为 1/6 FAILED'

# ---- TC22: integrate_over_triangle 常数函数积分为 0.5 ----
from error_estimator import integrate_over_triangle, twb_rule_data
data = twb_rule_data(4)
f_ones = np.ones(len(data['w']))
val = integrate_over_triangle(f_ones, 4)
assert abs(val - 0.5) < 1e-10, '[TC22] 三角形常数积分应为 0.5 FAILED'

# ---- TC23: t4_shape_functions 单位分解 ----
from mesh_refinement import t4_shape_functions
import numpy as np
np.random.seed(42)
test_ok = False
for _ in range(20):
    xi = np.random.uniform(0, 0.5)
    eta = np.random.uniform(0, 0.5)
    if xi + eta <= 1.0:
        N = t4_shape_functions(xi, eta)
        if abs(np.sum(N) - 1.0) < 1e-12:
            test_ok = True
            break
assert test_ok, '[TC23] T4 形函数单位分解失败 FAILED'

# ---- TC24: t4_shape_functions bubble 在形心最大 ----
from mesh_refinement import t4_shape_functions
N_cent = t4_shape_functions(1.0/3.0, 1.0/3.0)
assert N_cent[3] > 0.99, '[TC24] T4 bubble 函数在形心应接近 1 FAILED'

# ---- TC25: triangulation_t3_to_t4 输出尺寸正确 ----
nodes, triangles = generate_uniform_triangulation(10.0, 10.0, 3, 3)
nodes_t4, triangles_t4 = triangulation_t3_to_t4(nodes, triangles)
n_tri = len(triangles)
assert len(nodes_t4) == len(nodes) + n_tri, '[TC25] T4 节点数应=原节点数+三角形数 FAILED'
assert triangles_t4.shape == (n_tri, 4), '[TC25] T4 三角形形状应为 (n_tri, 4) FAILED'

# ---- TC26: discrete_laplacian_2d 常数图像得零（内部） ----
from dynamic_reconstruction import discrete_laplacian_2d
I_const = np.ones((8, 8))
lap = discrete_laplacian_2d(I_const)
assert np.max(np.abs(lap[1:-1, 1:-1])) < 1e-12, '[TC26] 常数图像内部拉普拉斯应为零 FAILED'

# ---- TC27: r83_cg 精确求解线性系统 ----
n = 20
main_diag = 2.0 * np.ones(n)
off_diag = -1.0 * np.ones(n - 1)
a_r83 = np.zeros((3, n))
a_r83[1, :] = main_diag
a_r83[0, 1:] = off_diag
a_r83[2, :-1] = off_diag
x_true = np.sin(np.linspace(0, np.pi, n))
b = np.zeros(n)
b[0] = main_diag[0] * x_true[0] + off_diag[0] * x_true[1]
b[-1] = off_diag[-1] * x_true[-2] + main_diag[-1] * x_true[-1]
for i in range(1, n - 1):
    b[i] = off_diag[i - 1] * x_true[i - 1] + main_diag[i] * x_true[i] + off_diag[i] * x_true[i + 1]
x_cg = r83_cg(n, a_r83, b, tol=1e-12)
err = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)
assert err < 1e-10, '[TC27] r83_cg 相对误差应 < 1e-10 FAILED'

# ---- TC28: generate_uniform_triangulation 节点与三角形数 ----
nodes, triangles = generate_uniform_triangulation(8.0, 6.0, 5, 4)
assert len(nodes) == 20, '[TC28] 5x4 网格应有 20 节点 FAILED'
assert len(triangles) == 24, '[TC28] 5x4 网格应有 24 三角形 FAILED'

# ---- TC29: evaluate_triangulation_quality 指标在 [0,1] ----
nodes, triangles = generate_uniform_triangulation(10.0, 10.0, 4, 4)
q = evaluate_triangulation_quality(nodes, triangles)
assert 0.0 <= q['alpha_min'] <= 1.0, '[TC29] alpha_min 应在 [0,1] FAILED'
assert 0.0 <= q['q_min'] <= 1.0, '[TC29] q_min 应在 [0,1] FAILED'
assert 0.0 <= q['alpha_ave'] <= 1.0, '[TC29] alpha_ave 应在 [0,1] FAILED'

# ---- TC30: compute_reconstruction_quality 完美重建 ----
img = generate_phantom_image(16)
metrics = compute_reconstruction_quality(img, img)
assert metrics['psnr'] > 100, '[TC30] 完美重建 PSNR 应 > 100 FAILED'
assert abs(metrics['l2_error']) < 1e-12, '[TC30] 完美重建 L2 误差应为 0 FAILED'
assert abs(metrics['mse']) < 1e-14, '[TC30] 完美重建 MSE 应为 0 FAILED'

# ---- TC31: compute_reconstruction_quality SSIM 完美重建 ----
img = generate_phantom_image(16)
metrics = compute_reconstruction_quality(img, img)
assert abs(metrics['ssim'] - 1.0) < 1e-6, '[TC31] 完美重建 SSIM 应接近 1 FAILED'

# ---- TC32: regula_falsi 求根正确 ----
from support_optimizer import regula_falsi
def f(x):
    return x ** 2 - 4.0
root, iters = regula_falsi(f, 0.0, 5.0)
assert abs(root - 2.0) < 1e-6, '[TC32] 假位法 x^2-4=0 根应为 2 FAILED'
assert iters > 0, '[TC32] 假位法迭代次数应 > 0 FAILED'

# ---- TC33: sample_paths_cholesky 固定种子可复现 ----
import numpy as np
np.random.seed(42)
X1 = sample_paths_cholesky(20, 3, 0.2)
np.random.seed(42)
X2 = sample_paths_cholesky(20, 3, 0.2)
assert np.allclose(X1, X2), '[TC33] Cholesky 采样固定种子应可复现 FAILED'

# ---- TC34: sample_paths_cholesky 输出形状 ----
import numpy as np
np.random.seed(42)
X = sample_paths_cholesky(30, 5, 0.15)
assert X.shape == (30, 5), '[TC34] Cholesky 采样形状应为 (30, 5) FAILED'

# ---- TC35: apply_spatial_prior 输出形状 ----
img = generate_phantom_image(16)
smoothed = apply_spatial_prior(img.ravel(), (16, 16), rho0=2.0, sigma=1.0)
assert len(smoothed) == 256, '[TC35] 空间先验输出长度应为 256 FAILED'

# ---- TC36: build_2d_spatial_covariance 可调用 ----
from spatial_prior import build_2d_spatial_covariance
K = build_2d_spatial_covariance((8, 8), rho0=2.0, sigma=1.0)
assert callable(K) or isinstance(K, np.ndarray), '[TC36] 协方差应为可调用或 ndarray FAILED'

# ---- TC37: prime_sampling_indices 输出长度 ----
indices = prime_sampling_indices(100, 25, prime_index=5)
assert len(indices) >= 1, '[TC37] 素数采样应至少返回 1 个索引 FAILED'
assert np.max(indices) < 100, '[TC37] 素数采样索引应 < 信号长度 FAILED'

# ---- TC38: construct_tridiagonal_from_dense 输出形状 ----
A_dense = np.diag(np.ones(10) * 2) + np.diag(np.ones(9) * (-1), 1) + np.diag(np.ones(9) * (-1), -1)
A_r83 = construct_tridiagonal_from_dense(A_dense)
assert A_r83.shape == (3, 10), '[TC38] R83 格式形状应为 (3, 10) FAILED'

# ---- TC39: solve_dynamic_diffusion 能量单调递减 ----
import numpy as np
np.random.seed(42)
img = generate_phantom_image(16)
t_arr, I_series = solve_dynamic_diffusion(img, (0.0, 0.2), 10, D=0.5, alpha=0.1)
energies = [np.sum(I ** 2) for I in I_series]
monotonic = all(energies[i] >= energies[i + 1] - 1e-6 for i in range(len(energies) - 1))
assert monotonic, '[TC39] 扩散能量应单调递减 FAILED'

# ---- TC40: solve_dynamic_diffusion 输出时间数组长度 ----
import numpy as np
np.random.seed(42)
img = generate_phantom_image(16)
t_arr, I_series = solve_dynamic_diffusion(img, (0.0, 0.3), 15, D=0.3, alpha=0.05)
assert len(t_arr) == 16, '[TC40] 时间数组长度应为 n_steps+1=16 FAILED'
assert I_series.shape[0] == 16, '[TC40] 图像序列第一维应为 16 FAILED'

# ---- TC41: adaptive_refinement_by_gradient 低阈值不细化 ----
img = generate_phantom_image(32)
nodes, triangles = generate_uniform_triangulation(32.0, 32.0, 4, 4)
nodes_r, triangles_r = adaptive_refinement_by_gradient(img, nodes, triangles, quality_threshold=0.0)
assert len(triangles_r) >= len(triangles), '[TC41] 低阈值不应减少三角形数 FAILED'

# ---- TC42: FISTA 可复现 ----
import numpy as np
np.random.seed(42)
A = build_sensing_matrix_gaussian(30, 50)
x_true_sparse = np.zeros(50)
x_true_sparse[3] = 1.0
x_true_sparse[15] = -0.8
y_clean = A @ x_true_sparse
c1 = fista_reconstruction(A, y_clean, 0.001, max_iter=200, tol=1e-6)
np.random.seed(42)
A2 = build_sensing_matrix_gaussian(30, 50)
c2 = fista_reconstruction(A2, y_clean, 0.001, max_iter=200, tol=1e-6)
assert np.allclose(c1, c2, atol=1e-10), '[TC42] FISTA 固定种子应可复现 FAILED'

# ---- TC43: orthogonal_matching_pursuit 输出支持集大小 ----
from cs_detector import orthogonal_matching_pursuit
import numpy as np
np.random.seed(42)
A = build_sensing_matrix_gaussian(60, 100)
x_true = np.zeros(100)
x_true[[5, 20, 45]] = [1.0, -0.5, 0.8]
y = A @ x_true
x_omp, support = orthogonal_matching_pursuit(A, y, sparsity=5)
assert len(support) <= 5, '[TC43] OMP 支持集大小不应超过目标稀疏度 FAILED'

# ---- TC44: twb_rule_n 有效强度返回正节点数 ----
from error_estimator import twb_rule_n
n1 = twb_rule_n(1)
n4 = twb_rule_n(4)
assert n1 > 0, '[TC44] 强度 1 应有正节点数 FAILED'
assert n4 > 0, '[TC44] 强度 4 应有正节点数 FAILED'

# ---- TC45: 集成：完整 CS 重建流程输出非空 ----
import numpy as np
np.random.seed(42)
img = generate_phantom_image(32)
Psi = build_2d_chebyshev_basis((32, 32), 6)
N = 32 * 32
m = int(N * 0.3)
Phi = build_sensing_matrix_gaussian(m, N, normalize=True)
A = Phi @ Psi
y_clean = Phi @ img.ravel()
noise = np.random.randn(m) * 1e-4
y = y_clean + noise
c_fista = fista_reconstruction(A, y, 0.0005, max_iter=300, tol=1e-6)
recon = (Psi @ c_fista).reshape((32, 32))
metrics = compute_reconstruction_quality(img, recon)
assert metrics['psnr'] > 0, '[TC45] CS 重建 PSNR 应 > 0 FAILED'
assert np.all(np.isfinite(recon)), '[TC45] CS 重建图像应不含 NaN/Inf FAILED'
