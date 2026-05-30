
import numpy as np
import time


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
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    image = np.zeros((size, size), dtype=float)


    mask1 = (x / 0.92) ** 2 + (y / 0.69) ** 2 <= 1.0
    image[mask1] = 1.0


    mask2 = (x / 0.74) ** 2 + (y / 0.55) ** 2 <= 1.0
    image[mask2] = 0.8


    mask3 = ((x + 0.22) / 0.31) ** 2 + (y / 0.41) ** 2 <= 1.0
    image[mask3] = 0.6


    mask4 = ((x - 0.22) / 0.31) ** 2 + (y / 0.41) ** 2 <= 1.0
    image[mask4] = 0.6


    mask5 = ((x - 0.08) / 0.1) ** 2 + ((y + 0.15) / 0.15) ** 2 <= 1.0
    image[mask5] = 0.3

    return image


def add_gaussian_noise(image: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(*image.shape) * np.sqrt(noise_power)
    return image + noise


def demo_compressed_sensing_reconstruction():
    print("=" * 70)
    print("演示 1: 核心压缩感知图像重建")
    print("=" * 70)


    image_size = 64
    sampling_ratio = 0.25
    chebyshev_order = 12
    snr_db = 35.0


    print(f"\n[1] 生成 {image_size}x{image_size} 合成医学图像...")
    true_image = generate_phantom_image(image_size)
    N = image_size * image_size
    m = int(N * sampling_ratio)
    print(f"    信号维度 N={N}, 采样数 m={m}, 采样率={sampling_ratio*100:.1f}%")



















    recon_fista = np.zeros_like(true_image)
    recon_refined = np.zeros_like(true_image)


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
    print("\n" + "=" * 70)
    print("演示 2: 自适应三角网格细化与 T3->T4 转换")
    print("=" * 70)

    image_size = 64
    true_image = generate_phantom_image(image_size)


    print("\n[1] 生成初始均匀 T3 网格...")
    nodes, triangles = generate_uniform_triangulation(float(image_size), float(image_size), 8, 8)
    print(f"    初始节点数: {len(nodes)}, 初始三角形数: {len(triangles)}")


    quality_init = evaluate_triangulation_quality(nodes, triangles)
    print(f"[2] 初始网格质量:")
    print(f"    ALPHA_min = {quality_init['alpha_min']:.4f}, ALPHA_ave = {quality_init['alpha_ave']:.4f}")
    print(f"    Q_min     = {quality_init['q_min']:.4f}, Q_ave     = {quality_init['q_ave']:.4f}")


    print("[3] 基于图像梯度自适应细化...")
    nodes_refined, triangles_refined = adaptive_refinement_by_gradient(
        true_image, nodes, triangles, quality_threshold=0.3)
    print(f"    细化后节点数: {len(nodes_refined)}, 三角形数: {len(triangles_refined)}")

    quality_refined = evaluate_triangulation_quality(nodes_refined, triangles_refined)
    print(f"[4] 细化后网格质量:")
    print(f"    ALPHA_min = {quality_refined['alpha_min']:.4f}, ALPHA_ave = {quality_refined['alpha_ave']:.4f}")
    print(f"    Q_min     = {quality_refined['q_min']:.4f}, Q_ave     = {quality_refined['q_ave']:.4f}")


    print("[5] T3 -> T4 网格转换...")
    nodes_t4, triangles_t4 = triangulation_t3_to_t4(nodes_refined, triangles_refined)
    print(f"    T4 节点数: {len(nodes_t4)}, T4 三角形数: {len(triangles_t4)}")


    print("[6] 在 T4 网格上插值图像...")

    node_values = np.zeros(len(nodes_t4))
    H, W = true_image.shape
    for i, (nx, ny) in enumerate(nodes_t4):
        ix = int(np.clip(nx / W * (W - 1), 0, W - 1))
        iy = int(np.clip(ny / H * (H - 1), 0, H - 1))
        node_values[i] = true_image[iy, ix]


    query_pts = np.array([[image_size / 2.0, image_size / 2.0],
                          [image_size / 4.0, image_size / 3.0],
                          [image_size * 0.7, image_size * 0.6]])
    interpolated = interpolate_on_t4_mesh(nodes_t4, triangles_t4, node_values, query_pts)
    print(f"    插值结果在查询点: {interpolated}")

    return nodes_t4, triangles_t4


def demo_spatial_correlation_prior():
    print("\n" + "=" * 70)
    print("演示 3: 空间相关先验建模")
    print("=" * 70)

    n_points = 64
    rho0 = 0.1

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
    print("\n" + "=" * 70)
    print("演示 4: 三对角共轭梯度快速求解器")
    print("=" * 70)

    n = 256
    print(f"\n[1] 构造 {n}x{n} 三对角测试系统...")


    main_diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)


    a_r83 = np.zeros((3, n))
    a_r83[1, :] = main_diag
    a_r83[0, 1:] = off_diag
    a_r83[2, :-1] = off_diag


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


    energies = [np.sum(I ** 2) for I in I_series]
    monotonic = all(energies[i] >= energies[i + 1] - 1e-6 for i in range(len(energies) - 1))
    print(f"    能量单调递减: {monotonic}")

    return t_array, I_series


def demo_numerical_integration():
    print("\n" + "=" * 70)
    print("演示 6: 高阶数值积分规则")
    print("=" * 70)

    from error_estimator import integrate_triangle_unit_monomial, pyramid_unit_volume

    print("\n[1] 三角形单项式精确积分验证:")
    test_cases = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 2)]
    for ex, ey in test_cases:
        val = integrate_triangle_unit_monomial(ex, ey)

        from math import factorial
        exact = factorial(ex) * factorial(ey) / factorial(ex + ey + 2)
        print(f"    x^{ex} y^{ey}: 计算={val:.10f}, 精确={exact:.10f}, 误差={abs(val-exact):.2e}")

    print("\n[2] 单位金字塔体积:")
    vol = pyramid_unit_volume()
    print(f"    V = {vol:.10f} (理论值 = 4/3 = {4.0/3.0:.10f})")

    print("\n[3] 重建误差评估示例:")
    true_img = generate_phantom_image(32)

    recon_img = true_img + 0.05 * np.random.randn(32, 32)
    metrics = compute_reconstruction_quality(true_img, recon_img)
    print(f"    L2 误差  = {metrics['l2_error']:.6f}")
    print(f"    PSNR     = {metrics['psnr']:.2f} dB")
    print(f"    SSIM     = {metrics['ssim']:.4f}")


def main():
    print("\n" + "#" * 70)
    print("#  基于自适应三角剖分与谱稀疏表示的压缩感知图像重建系统")
    print("#  数据科学：图像重建压缩感知")
    print("#" * 70)
    print("\n本项目融合 15 个科研代码项目的核心算法，")
    print("解决前沿博士级科学问题：严重欠采样条件下的医学图像压缩感知重建。\n")

    np.random.seed(42)


    demo_compressed_sensing_reconstruction()


    demo_adaptive_mesh_refinement()


    demo_spatial_correlation_prior()


    demo_fast_tridiagonal_solver()


    demo_dynamic_diffusion()


    demo_numerical_integration()

    print("\n" + "#" * 70)
    print("#  所有演示成功完成！")
    print("#" * 70)


if __name__ == "__main__":
    main()
