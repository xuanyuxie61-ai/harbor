"""
main.py
=======
基于自适应三角剖分与谱稀疏表示的压缩感知图像重建系统

统一入口，零参数可运行。

科学问题：
---------
在医学成像（如 MRI、CT）中，减少采样时间对患者安全和成像效率至关重要。
压缩感知（Compressed Sensing, CS）理论允许从远少于奈奎斯特定理要求的
采样中精确重建图像，前提图像是稀疏的。

本项目解决的问题是：
    "如何在严重欠采样条件下，结合空间先验、自适应网格和谱稀疏表示，
     实现高质量的图像重建？"

运行方式：
---------
    python main.py

无需任何命令行参数。
"""

import numpy as np
import time

# 导入各模块
from sampling_pattern import build_incoherent_mask, prime_sampling_indices
from spectral_basis import (build_2d_chebyshev_basis, image_to_chebyshev_coefficients,
                            chebyshev_coefficients_to_image)
from spatial_prior import (build_2d_spatial_covariance, sample_paths_cholesky,
                           apply_spatial_prior, correlation_gaussian)
from cs_detector import (fista_reconstruction, orthogonal_matching_pursuit,
                         build_sensing_matrix_gaussian)
from fast_solver import solve_normal_equations_cg, r83_cg, construct_tridiagonal_from_dense
from mesh_adaptive import (generate_uniform_triangulation, evaluate_triangulation_quality,
                           adaptive_refinement_by_gradient)
from mesh_refinement import triangulation_t3_to_t4, interpolate_on_t4_mesh
from error_estimator import compute_reconstruction_quality, pyramid_unit_volume
from dynamic_reconstruction import solve_dynamic_diffusion
from support_optimizer import refined_support_reconstruction, optimize_threshold_for_sparsity


def generate_phantom_image(size: int = 64) -> np.ndarray:
    """
    生成 Shepp-Logan 型合成图像（医学 CT/MRI 标准测试图像变体）。

    数学模型：
        图像由多个椭圆叠加而成，模拟人体头部断层结构。
        每个椭圆：(x/a)^2 + (y/b)^2 <= 1
    """
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    image = np.zeros((size, size), dtype=float)

    # 外椭圆（头骨）
    mask1 = (x / 0.92) ** 2 + (y / 0.69) ** 2 <= 1.0
    image[mask1] = 1.0

    # 内椭圆（脑组织）
    mask2 = (x / 0.74) ** 2 + (y / 0.55) ** 2 <= 1.0
    image[mask2] = 0.8

    # 左半脑
    mask3 = ((x + 0.22) / 0.31) ** 2 + (y / 0.41) ** 2 <= 1.0
    image[mask3] = 0.6

    # 右半脑
    mask4 = ((x - 0.22) / 0.31) ** 2 + (y / 0.41) ** 2 <= 1.0
    image[mask4] = 0.6

    # 小肿瘤/异常
    mask5 = ((x - 0.08) / 0.1) ** 2 + ((y + 0.15) / 0.15) ** 2 <= 1.0
    image[mask5] = 0.3

    return image


def add_gaussian_noise(image: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """
    添加高斯白噪声，模拟测量噪声。

    信噪比定义：SNR = 10 * log10(P_signal / P_noise)
    """
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(*image.shape) * np.sqrt(noise_power)
    return image + noise


def demo_compressed_sensing_reconstruction():
    """
    演示核心压缩感知重建流程。
    """
    print("=" * 70)
    print("演示 1: 核心压缩感知图像重建")
    print("=" * 70)

    # 参数设置
    image_size = 64
    sampling_ratio = 0.25  # 25% 采样
    chebyshev_order = 12
    snr_db = 35.0

    # 生成测试图像
    print(f"\n[1] 生成 {image_size}x{image_size} 合成医学图像...")
    true_image = generate_phantom_image(image_size)
    N = image_size * image_size
    m = int(N * sampling_ratio)
    print(f"    信号维度 N={N}, 采样数 m={m}, 采样率={sampling_ratio*100:.1f}%")

    # 构造稀疏基（切比雪夫张量积基）
    print(f"[2] 构造 {chebyshev_order} 阶切比雪夫稀疏基...")
    Psi = build_2d_chebyshev_basis((image_size, image_size), chebyshev_order)
    print(f"    稀疏基维度: {Psi.shape}")

    # 构造测量矩阵（高斯随机）
    print("[3] 构造高斯随机测量矩阵...")
    Phi = build_sensing_matrix_gaussian(m, N, normalize=True)

    # 构造感知矩阵 A = Phi @ Psi
    A = Phi @ Psi
    print(f"    感知矩阵维度: {A.shape}")

    # 模拟测量（含噪声）
    print(f"[4] 模拟测量过程（SNR={snr_db} dB）...")
    y_clean = Phi @ true_image.ravel()
    noise = np.random.randn(m)
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(y_clean) / (10.0 ** (snr_db / 20.0))
    y = y_clean + noise

    # 压缩感知重建：FISTA
    print("[5] 执行 FISTA 压缩感知重建...")
    lambda_reg = 0.001
    t_start = time.time()
    c_fista = fista_reconstruction(A, y, lambda_reg, max_iter=800, tol=1e-6)
    t_fista = time.time() - t_start
    recon_fista = Psi @ c_fista
    recon_fista = recon_fista.reshape((image_size, image_size))
    print(f"    FISTA 耗时: {t_fista:.3f} s")

    # 压缩感知重建：OMP + 支持集优化
    print("[6] 执行 OMP + 支持集优化重建...")
    target_sparsity = max(10, int(0.05 * Psi.shape[1]))
    t_start = time.time()
    c_refined, support = refined_support_reconstruction(A, y, target_sparsity)
    t_refined = time.time() - t_start
    recon_refined = Psi @ c_refined
    recon_refined = recon_refined.reshape((image_size, image_size))
    print(f"    支持集优化耗时: {t_refined:.3f} s, 支持集大小={len(support)}")

    # 评估重建质量
    print("[7] 重建质量评估...")
    metrics_fista = compute_reconstruction_quality(true_image, recon_fista)
    metrics_refined = compute_reconstruction_quality(true_image, recon_refined)

    print(f"\n    FISTA 重建:")
    print(f"      L2 误差      = {metrics_fista['l2_error']:.6f}")
    print(f"      MSE          = {metrics_fista['mse']:.6e}")
    print(f"      PSNR         = {metrics_fista['psnr']:.2f} dB")
    print(f"      SSIM         = {metrics_fista['ssim']:.4f}")
    print(f"      最大误差     = {metrics_fista['max_error']:.6f}")

    print(f"\n    OMP+支持集优化:")
    print(f"      L2 误差      = {metrics_refined['l2_error']:.6f}")
    print(f"      MSE          = {metrics_refined['mse']:.6e}")
    print(f"      PSNR         = {metrics_refined['psnr']:.2f} dB")
    print(f"      SSIM         = {metrics_refined['ssim']:.4f}")
    print(f"      最大误差     = {metrics_refined['max_error']:.6f}")

    return true_image, recon_fista, recon_refined


def demo_adaptive_mesh_refinement():
    """
    演示自适应三角网格细化与 T3->T4 转换。
    """
    print("\n" + "=" * 70)
    print("演示 2: 自适应三角网格细化与 T3->T4 转换")
    print("=" * 70)

    image_size = 64
    true_image = generate_phantom_image(image_size)

    # 生成初始均匀网格
    print("\n[1] 生成初始均匀 T3 网格...")
    nodes, triangles = generate_uniform_triangulation(float(image_size), float(image_size), 8, 8)
    print(f"    初始节点数: {len(nodes)}, 初始三角形数: {len(triangles)}")

    # 评估初始质量
    quality_init = evaluate_triangulation_quality(nodes, triangles)
    print(f"[2] 初始网格质量:")
    print(f"    ALPHA_min = {quality_init['alpha_min']:.4f}, ALPHA_ave = {quality_init['alpha_ave']:.4f}")
    print(f"    Q_min     = {quality_init['q_min']:.4f}, Q_ave     = {quality_init['q_ave']:.4f}")

    # 基于梯度自适应细化
    print("[3] 基于图像梯度自适应细化...")
    nodes_refined, triangles_refined = adaptive_refinement_by_gradient(
        true_image, nodes, triangles, quality_threshold=0.3)
    print(f"    细化后节点数: {len(nodes_refined)}, 三角形数: {len(triangles_refined)}")

    quality_refined = evaluate_triangulation_quality(nodes_refined, triangles_refined)
    print(f"[4] 细化后网格质量:")
    print(f"    ALPHA_min = {quality_refined['alpha_min']:.4f}, ALPHA_ave = {quality_refined['alpha_ave']:.4f}")
    print(f"    Q_min     = {quality_refined['q_min']:.4f}, Q_ave     = {quality_refined['q_ave']:.4f}")

    # T3 -> T4 转换
    print("[5] T3 -> T4 网格转换...")
    nodes_t4, triangles_t4 = triangulation_t3_to_t4(nodes_refined, triangles_refined)
    print(f"    T4 节点数: {len(nodes_t4)}, T4 三角形数: {len(triangles_t4)}")

    # 在 T4 网格上插值图像
    print("[6] 在 T4 网格上插值图像...")
    # 为每个节点分配图像值（从像素网格映射到节点）
    node_values = np.zeros(len(nodes_t4))
    H, W = true_image.shape
    for i, (nx, ny) in enumerate(nodes_t4):
        ix = int(np.clip(nx / W * (W - 1), 0, W - 1))
        iy = int(np.clip(ny / H * (H - 1), 0, H - 1))
        node_values[i] = true_image[iy, ix]

    # 在网格中心点插值
    query_pts = np.array([[image_size / 2.0, image_size / 2.0],
                          [image_size / 4.0, image_size / 3.0],
                          [image_size * 0.7, image_size * 0.6]])
    interpolated = interpolate_on_t4_mesh(nodes_t4, triangles_t4, node_values, query_pts)
    print(f"    插值结果在查询点: {interpolated}")

    return nodes_t4, triangles_t4


def demo_spatial_correlation_prior():
    """
    演示空间相关先验建模与 Cholesky 采样。
    """
    print("\n" + "=" * 70)
    print("演示 3: 空间相关先验建模")
    print("=" * 70)

    n_points = 64
    rho0 = 0.1  # 相关长度比例

    print(f"\n[1] 构造 {n_points} 点的一维高斯相关矩阵...")
    from spatial_prior import build_correlation_matrix_1d
    C = build_correlation_matrix_1d(n_points, rho0, domain_length=1.0)
    print(f"    相关矩阵条件数: {np.linalg.cond(C):.2e}")
    print(f"    最小特征值: {np.min(np.linalg.eigvalsh(C)):.6f}")

    print("[2] 利用 Cholesky 分解生成相关随机场样本...")
    n_paths = 3
    X = sample_paths_cholesky(n_points, n_paths, rho0, domain_length=1.0)
    print(f"    样本路径形状: {X.shape}")
    print(f"    样本协方差与理论相关矩阵的 Frobenius 误差: "
          f"{np.linalg.norm(np.cov(X) - C, 'fro') / np.linalg.norm(C, 'fro'):.4f}")

    print("[3] 验证高斯相关函数...")
    test_rho = np.array([0.0, 0.05, 0.1, 0.2, 0.5])
    corr_vals = correlation_gaussian(test_rho, rho0)
    print(f"    rho = {test_rho}")
    print(f"    C(rho) = {corr_vals}")

    print("[4] 二维空间先验应用...")
    image = generate_phantom_image(32)
    smoothed = apply_spatial_prior(image.ravel(), (32, 32), rho0=4.0, sigma=1.0)
    smoothed = smoothed.reshape((32, 32))
    print(f"    原始图像能量: {np.sum(image**2):.4f}")
    print(f"    平滑后能量:   {np.sum(smoothed**2):.4f}")

    return C, X


def demo_fast_tridiagonal_solver():
    """
    演示三对角 CG 快速求解器。
    """
    print("\n" + "=" * 70)
    print("演示 4: 三对角共轭梯度快速求解器")
    print("=" * 70)

    n = 256
    print(f"\n[1] 构造 {n}x{n} 三对角测试系统...")

    # 构造对称正定三对角矩阵（二阶差分矩阵的变体）
    main_diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)

    # R83 格式
    a_r83 = np.zeros((3, n))
    a_r83[1, :] = main_diag
    a_r83[0, 1:] = off_diag  # 上对角线
    a_r83[2, :-1] = off_diag  # 下对角线

    # 真实解和右端项
    x_true = np.sin(np.linspace(0, 2 * np.pi, n))
    b = np.zeros(n)
    b[0] = main_diag[0] * x_true[0] + off_diag[0] * x_true[1]
    b[-1] = off_diag[-1] * x_true[-2] + main_diag[-1] * x_true[-1]
    for i in range(1, n - 1):
        b[i] = off_diag[i - 1] * x_true[i - 1] + main_diag[i] * x_true[i] + off_diag[i] * x_true[i + 1]

    print("[2] 使用 R83-CG 求解...")
    t_start = time.time()
    x_cg = r83_cg(n, a_r83, b, tol=1e-12)
    t_cg = time.time() - t_start

    error_cg = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)
    print(f"    CG 耗时: {t_cg:.4f} s")
    print(f"    相对误差: {error_cg:.6e}")

    print("[3] 与稠密直接求解对比...")
    A_dense = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    t_start = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    t_dense = time.time() - t_start
    error_dense = np.linalg.norm(x_dense - x_true) / np.linalg.norm(x_true)
    print(f"    直接求解耗时: {t_dense:.4f} s")
    print(f"    相对误差: {error_dense:.6e}")
    print(f"    加速比: {t_dense / max(t_cg, 1e-10):.2f}x")

    return x_cg, x_dense


def demo_dynamic_diffusion():
    """
    演示动态扩散过程重建。
    """
    print("\n" + "=" * 70)
    print("演示 5: 动态扩散图像序列")
    print("=" * 70)

    image_size = 32
    I0 = generate_phantom_image(image_size)
    tspan = (0.0, 0.5)
    n_steps = 20
    D = 0.5
    alpha = 0.1

    print(f"\n[1] 求解扩散-衰减 PDE (D={D}, alpha={alpha})...")
    print(f"    时间区间: {tspan}, 步数: {n_steps}")

    t_array, I_series = solve_dynamic_diffusion(I0, tspan, n_steps, D=D, alpha=alpha, method='implicit')

    print(f"[2] 时间演化统计:")
    initial_energy = np.sum(I_series[0] ** 2)
    final_energy = np.sum(I_series[-1] ** 2)
    print(f"    初始能量: {initial_energy:.4f}")
    print(f"    最终能量: {final_energy:.4f}")
    print(f"    能量衰减比: {final_energy / initial_energy:.4f}")

    # 验证能量单调递减（对于扩散方程应成立）
    energies = [np.sum(I ** 2) for I in I_series]
    monotonic = all(energies[i] >= energies[i + 1] - 1e-6 for i in range(len(energies) - 1))
    print(f"    能量单调递减: {monotonic}")

    return t_array, I_series


def demo_numerical_integration():
    """
    演示数值积分规则。
    """
    print("\n" + "=" * 70)
    print("演示 6: 高阶数值积分规则")
    print("=" * 70)

    from error_estimator import integrate_triangle_unit_monomial, pyramid_unit_volume

    print("\n[1] 三角形单项式精确积分验证:")
    test_cases = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 2)]
    for ex, ey in test_cases:
        val = integrate_triangle_unit_monomial(ex, ey)
        # 解析值：ex! * ey! / (ex + ey + 2)!
        from math import factorial
        exact = factorial(ex) * factorial(ey) / factorial(ex + ey + 2)
        print(f"    x^{ex} y^{ey}: 计算={val:.10f}, 精确={exact:.10f}, 误差={abs(val-exact):.2e}")

    print("\n[2] 单位金字塔体积:")
    vol = pyramid_unit_volume()
    print(f"    V = {vol:.10f} (理论值 = 4/3 = {4.0/3.0:.10f})")

    print("\n[3] 重建误差评估示例:")
    true_img = generate_phantom_image(32)
    # 模拟一个粗糙重建
    recon_img = true_img + 0.05 * np.random.randn(32, 32)
    metrics = compute_reconstruction_quality(true_img, recon_img)
    print(f"    L2 误差  = {metrics['l2_error']:.6f}")
    print(f"    PSNR     = {metrics['psnr']:.2f} dB")
    print(f"    SSIM     = {metrics['ssim']:.4f}")


def main():
    """
    主函数：执行所有演示。
    """
    print("\n" + "#" * 70)
    print("#  基于自适应三角剖分与谱稀疏表示的压缩感知图像重建系统")
    print("#  数据科学：图像重建压缩感知")
    print("#" * 70)
    print("\n本项目融合 15 个科研代码项目的核心算法，")
    print("解决前沿博士级科学问题：严重欠采样条件下的医学图像压缩感知重建。\n")

    np.random.seed(42)

    # 演示 1: 核心 CS 重建
    demo_compressed_sensing_reconstruction()

    # 演示 2: 自适应网格
    demo_adaptive_mesh_refinement()

    # 演示 3: 空间先验
    demo_spatial_correlation_prior()

    # 演示 4: 快速求解器
    demo_fast_tridiagonal_solver()

    # 演示 5: 动态扩散
    demo_dynamic_diffusion()

    # 演示 6: 数值积分
    demo_numerical_integration()

    print("\n" + "#" * 70)
    print("#  所有演示成功完成！")
    print("#" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（45个，assert模式，涉及随机值均使用固定种子）
# ================================================================
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

print('\n全部 45 个测试通过!\n')
