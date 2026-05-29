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

    # TODO [Hole_3]: 实现压缩感知重建管道
    # 任务说明：
    #   本段代码需要协调以下跨文件组件完成图像重建：
    #   1. spectral_basis.build_2d_chebyshev_basis() -> 构造稀疏基 Psi (H*W, order^2)
    #   2. cs_detector.build_sensing_matrix_gaussian() -> 构造测量矩阵 Phi (m, N)
    #   3. 计算感知矩阵 A = Phi @ Psi
    #   4. 模拟含噪声测量 y（注意 SNR 定义与噪声功率计算）
    #   5. cs_detector.fista_reconstruction(A, y, lambda_reg) -> FISTA 稀疏系数 c_fista
    #   6. support_optimizer.refined_support_reconstruction() -> OMP+支持集优化 c_refined
    #   7. 通过 Psi @ c 还原图像并 reshape 为 (image_size, image_size)
    #
    # 跨文件协同要点：
    #   - Psi 的列数必须与 A 的列数一致，否则矩阵乘法报错
    #   - Psi 的列归一化状态会影响 c 的量级，从而影响重建图像幅度
    #   - FISTA 的 lambda_reg 需与 Psi 的归一化方式匹配
    # ========================================
    # 请在下方实现完整的重建管道
    # ========================================
    recon_fista = np.zeros_like(true_image)
    recon_refined = np.zeros_like(true_image)

    # 评估重建质量（基于占位结果）
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
