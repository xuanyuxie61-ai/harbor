"""
超声B-scan图像PCA特征提取与降维模块

基于种子项目 326_eigenfaces 的核心算法，
为超声层析成像提供高维回波数据的降维与特征提取。

数学背景:
给定一组超声B-scan图像 {x₁, x₂, ..., x_N}，每张图像为 d 维向量。
PCA寻找正交投影矩阵 W = [w₁, w₂, ..., w_K]，使得投影后数据的方差最大。

优化问题:
    max  wᵀ·S·w
    s.t. wᵀ·w = 1
其中 S = (1/N) Σ (xᵢ - μ)(xᵢ - μ)ᵀ 为样本协方差矩阵。

解析解: w 为 S 的最大特征值对应的特征向量。

Turk-Pentland技巧:
当 d ≫ N 时，直接计算 d×d 协方差矩阵不可行。
利用 A = [x₁-μ, ..., x_N-μ] (d×N)，计算 N×N 矩阵 AᵀA 的特征分解:
    AᵀA·vᵢ = λᵢ·vᵢ
则 A·vᵢ 为协方差矩阵的特征向量方向。

降维后的表示:
    yᵢ = Wᵀ·(xᵢ - μ) ∈ ℝᴷ
重建:
    x̃ᵢ = μ + W·yᵢ
"""

import numpy as np
from typing import Tuple, List


def compute_mean_face(images: np.ndarray) -> np.ndarray:
    """计算平均图像（均值向量）。
    
    参数:
        images: (N, d) 图像数据矩阵，每行一张图像
    
    返回:
        mean_face: (d,) 平均图像
    """
    return np.mean(images, axis=0)


def compute_pca_vectors(images: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算PCA主成分向量（Turk-Pentland方法）。
    
    参数:
        images: (N, d) 图像数据矩阵
        n_components: 保留的主成分数，None时保留全部
    
    返回:
        eigenvectors: (K, d) 主成分向量（特征脸）
        eigenvalues: (K,) 对应的特征值
        mean_face: (d,) 平均图像
    """
    N, d = images.shape
    
    if n_components is None:
        n_components = min(N - 1, d)
    
    n_components = min(n_components, N - 1, d)
    
    mean_face = compute_mean_face(images)
    
    # 中心化处理
    A = images - mean_face  # (N, d)
    
    # Turk-Pentland技巧：计算 A·Aᵀ (N×N) 而非 AᵀA (d×d)
    L = A @ A.T  # (N, N)
    
    # 特征分解
    eigenvalues_full, eigenvectors_full = np.linalg.eigh(L)
    
    # 按特征值降序排序
    idx = np.argsort(eigenvalues_full)[::-1]
    eigenvalues_full = eigenvalues_full[idx]
    eigenvectors_full = eigenvectors_full[:, idx]
    
    # 选取前n_components个
    eigenvalues = eigenvalues_full[:n_components]
    
    # 将 AᵀA 的特征向量转换为 A·Aᵀ 的特征向量
    # v_i = A^T · u_i，然后归一化
    eigenvectors = np.zeros((n_components, d))
    for i in range(n_components):
        if eigenvalues[i] > 1e-14:
            v = A.T @ eigenvectors_full[:, i]
            v = v / np.linalg.norm(v)
            eigenvectors[i] = v
    
    return eigenvectors, eigenvalues, mean_face


def project_image(image: np.ndarray, eigenvectors: np.ndarray,
                  mean_face: np.ndarray) -> np.ndarray:
    """将单张图像投影到PCA子空间。
    
    投影公式:
        y = Wᵀ · (x - μ)
    
    参数:
        image: (d,) 图像向量
        eigenvectors: (K, d) 主成分向量
        mean_face: (d,) 平均图像
    
    返回:
        coefficients: (K,) PCA系数
    """
    centered = image - mean_face
    coefficients = eigenvectors @ centered
    return coefficients


def reconstruct_image(coefficients: np.ndarray, eigenvectors: np.ndarray,
                      mean_face: np.ndarray) -> np.ndarray:
    """从PCA系数重建图像。
    
    重建公式:
        x̃ = μ + Σ yᵢ·wᵢ = μ + W·y
    
    参数:
        coefficients: (K,) PCA系数
        eigenvectors: (K, d) 主成分向量
        mean_face: (d,) 平均图像
    
    返回:
        reconstructed: (d,) 重建图像
    """
    return mean_face + coefficients @ eigenvectors


def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """计算重建误差指标。
    
    返回:
        包含多种误差度量的字典
    """
    diff = original - reconstructed
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    
    # 峰值信噪比
    max_val = np.max(np.abs(original))
    if max_val > 1e-14:
        psnr = 20.0 * np.log10(max_val / rmse)
    else:
        psnr = 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'psnr': float(psnr),
        'max_error': float(np.max(np.abs(diff))),
        'mean_error': float(np.mean(np.abs(diff)))
    }


def generate_synthetic_bscans(n_images: int = 50, n_samples: int = 256,
                              n_lines: int = 64) -> np.ndarray:
    """生成合成超声B-scan图像用于PCA分析。
    
    模拟包含不同深度和强度的反射界面，
    用于验证PCA降维对超声图像特征提取的有效性。
    
    参数:
        n_images: B-scan图像数量
        n_samples: 每线A-scan采样点数
        n_lines: 扫描线数
    
    返回:
        images: (n_images, n_samples*n_lines) 展平的图像矩阵
    """
    image_size = n_samples * n_lines
    images = np.zeros((n_images, image_size))
    
    for img_idx in range(n_images):
        bscan = np.zeros((n_lines, n_samples))
        
        # 随机生成组织界面
        n_interfaces = np.random.randint(2, 6)
        for _ in range(n_interfaces):
            depth = np.random.randint(20, n_samples - 20)
            amplitude = np.random.uniform(0.3, 1.0)
            thickness = np.random.randint(2, 8)
            
            for line in range(n_lines):
                # 深度方向的界面加横向变化
                depth_variation = int(3 * np.sin(2 * np.pi * line / n_lines + img_idx))
                actual_depth = depth + depth_variation
                actual_depth = max(0, min(n_samples - 1, actual_depth))
                
                for t in range(thickness):
                    if actual_depth + t < n_samples:
                        bscan[line, actual_depth + t] += amplitude * np.exp(-t**2 / 4.0)
        
        # 添加随机噪声
        noise_level = np.random.uniform(0.05, 0.15)
        bscan += noise_level * np.random.randn(n_lines, n_samples)
        
        images[img_idx] = bscan.flatten()
    
    return images


def pca_bscan_analysis(n_images: int = 50, n_components: int = 10) -> dict:
    """对合成超声B-scan进行完整的PCA分析。
    
    返回:
        分析结果字典
    """
    images = generate_synthetic_bscans(n_images)
    
    eigenvectors, eigenvalues, mean_face = compute_pca_vectors(images, n_components)
    
    # 计算方差保留率
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues)
    variance_ratio = eigenvalues / (total_variance + 1e-14)
    cumulative_ratio = cumulative_variance / (total_variance + 1e-14)
    
    # 对第一张图像进行重建测试
    test_image = images[0]
    coeffs = project_image(test_image, eigenvectors, mean_face)
    reconstructed = reconstruct_image(coeffs, eigenvectors, mean_face)
    error_info = compute_reconstruction_error(test_image, reconstructed)
    
    return {
        'n_images': n_images,
        'n_components': n_components,
        'eigenvalues': eigenvalues.tolist(),
        'variance_ratio': variance_ratio.tolist(),
        'cumulative_variance_ratio': cumulative_ratio.tolist(),
        'reconstruction_error': error_info
    }
