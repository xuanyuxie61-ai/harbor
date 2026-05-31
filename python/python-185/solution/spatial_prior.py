"""
spatial_prior.py
================
基于高斯相关函数与协方差结构的空间先验建模模块

科学背景：
---------
在医学图像重建中，相邻像素之间存在强烈的空间相关性。这种相关性可以通过
高斯随机场（Gaussian Random Field, GRF）建模：

    X(\mathbf{s}) \sim \mathcal{GP}(\mu(\mathbf{s}), K(\mathbf{s}, \mathbf{s}'))

其中协方差核函数（covariance kernel）常用高斯形式（平方指数核）：

    K(\mathbf{s}, \mathbf{s}') = \sigma^2 \exp\left(-\frac{\|\mathbf{s} - \mathbf{s}'\|^2}{2\rho_0^2}\right)

\rho_0 为相关长度（correlation length），\sigma^2 为方差。

核心算法：
---------
1. 相关矩阵构造：由一维相关函数向量构造托普利茨（Toeplitz）相关矩阵
2. Cholesky 分解：C = L L^T，用于生成相关随机样本
3. 协方差矩阵：K = \text{diag}(\sigma) \cdot C \cdot \text{diag}(\sigma)

来自项目 220_correlation 的核心思想。
"""

import numpy as np
from typing import Tuple


def correlation_gaussian(rho: np.ndarray, rho0: float) -> np.ndarray:
    """
    高斯相关函数（平方指数核）。

    数学公式：
        C(\rho) = \exp\left(-\left(\frac{\rho}{\rho_0}\right)^2\right)

    物理意义：
        描述空间中两点间随距离指数衰减的相关性，\rho_0 控制衰减速率。
        在量子力学中，这对应于谐振子基态的波函数关联。

    参数:
        rho: 距离向量或数组
        rho0: 相关长度（必须为正）
    返回:
        相关函数值，形状与 rho 相同
    """
    rho = np.asarray(rho, dtype=float)
    if rho0 <= 0:
        raise ValueError("相关长度 rho0 必须为正")

    rhohat = rho / rho0
    return np.exp(-rhohat ** 2)


def correlation_to_covariance(C: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    由相关矩阵和标准差构造协方差矩阵。

    数学公式：
        K_{ij} = \sigma_i C_{ij} \sigma_j
        即 K = \text{diag}(\sigma) \cdot C \cdot \text{diag}(\sigma)

    验证条件：
        - C 必须对称：\|C - C^T\|_F < \text{tol}
        - 对角线必须为 1：\frac{1}{n}\sum_i |C_{ii} - 1| < \text{tol}
        - 非对角元素必须在 [-1, 1] 内

    参数:
        C: 相关矩阵，形状为 (n, n)
        sigma: 标准差向量，形状为 (n,)
    返回:
        协方差矩阵 K，形状为 (n, n)
    """
    C = np.asarray(C, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = C.shape[0]

    if C.shape != (n, n):
        raise ValueError("C 必须是方阵")
    if len(sigma) != n:
        raise ValueError("sigma 长度必须与 C 的维度一致")

    tol = np.sqrt(np.finfo(float).eps)

    # 对称性检查
    sym_err = np.linalg.norm(C - C.T, 'fro')
    if sym_err > tol:
        raise ValueError(f"相关矩阵不对称，误差={sym_err:.3e}")

    # 对角线检查
    diag_err = np.mean(np.abs(np.diag(C) - 1.0))
    if diag_err > tol:
        raise ValueError(f"相关矩阵对角线不为 1，平均误差={diag_err:.3e}")

    # 范围检查
    c_min = np.min(C)
    c_max = np.max(C)
    if c_min < -1.0 - tol or c_max > 1.0 + tol:
        raise ValueError(f"相关矩阵元素超出 [-1, 1] 范围：min={c_min:.3e}, max={c_max:.3e}")

    # 构造协方差矩阵
    D = np.diag(sigma)
    K = D @ C @ D
    return K


def build_correlation_matrix_1d(n: int, rho0: float, domain_length: float = 1.0) -> np.ndarray:
    """
    构造一维等距点上的高斯相关矩阵。

    参数:
        n: 点数
        rho0: 相关长度（相对于 domain_length 的比例）
        domain_length: 定义域长度
    返回:
        相关矩阵 C，形状为 (n, n)
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    x = np.linspace(0.0, domain_length, n)
    # 距离矩阵
    dx = np.abs(x[:, None] - x[None, :])
    C = correlation_gaussian(dx, rho0 * domain_length)
    # 数值稳定性修正
    C = 0.5 * (C + C.T)
    # 确保正定性（特征值截断）
    eigvals = np.linalg.eigvalsh(C)
    if np.min(eigvals) < 1e-12:
        C += (1e-12 - np.min(eigvals)) * np.eye(n)
    return C


def sample_paths_cholesky(n: int, n_paths: int, rho0: float,
                          domain_length: float = 1.0) -> np.ndarray:
    """
    利用 Cholesky 分解生成具有指定相关结构的高斯随机场样本路径。

    数学原理：
        设 C = L L^T 为相关矩阵的 Cholesky 分解，
        生成独立标准正态变量 Z ~ \mathcal{N}(0, I)，
        则 X = L Z 满足 Cov(X) = C。

    参数:
        n: 每路径的采样点数
        n_paths: 路径数量（样本数）
        rho0: 相关长度比例
        domain_length: 定义域长度
    返回:
        样本路径矩阵 X，形状为 (n, n_paths)
    """
    C = build_correlation_matrix_1d(n, rho0, domain_length)

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError as e:
        # 若 Cholesky 失败，对矩阵进行微小正则化
        C_reg = C + 1e-10 * np.eye(n)
        L = np.linalg.cholesky(C_reg)

    Z = np.random.randn(n, n_paths)
    X = L @ Z
    return X


def build_2d_spatial_covariance(image_shape: Tuple[int, int],
                                rho0: float, sigma: float = 1.0) -> np.ndarray:
    """
    构造二维图像的空间协方差矩阵（可分离核）。

    对于二维图像，采用可分离的高斯核：
        K((x_1,y_1), (x_2,y_2)) = \sigma^2 \exp\left(-\frac{(x_1-x_2)^2+(y_1-y_2)^2}{2\rho_0^2}\right)

    利用 Kronecker 积性质，二维协方差矩阵可表示为：
        K_{2D} = K_y \otimes K_x
    其中 K_x, K_y 为行方向和列方向的一维协方差矩阵。

    参数:
        image_shape: 图像尺寸 (H, W)
        rho0: 空间相关长度（像素单位）
        sigma: 幅度标准差
    返回:
        二维协方差矩阵，形状为 (H*W, H*W)
        注意：对于大图像，返回的是函数句柄或低秩近似
    """
    H, W = image_shape
    # 对于大图像，不构造完整矩阵，而是返回矩阵-向量乘法函数
    if H * W > 5000:
        # 使用可分离核的快速乘法
        Kx = build_correlation_matrix_1d(W, rho0 / W if W > 1 else 1.0, 1.0)
        Ky = build_correlation_matrix_1d(H, rho0 / H if H > 1 else 1.0, 1.0)

        def mv(v: np.ndarray) -> np.ndarray:
            """协方差矩阵与向量相乘的快速实现。"""
            V = v.reshape((H, W))
            # K_{2D} v = \sigma^2 (K_y \otimes K_x) \text{vec}(V)
            #          = \sigma^2 \text{vec}(K_x V K_y^T)
            result = sigma ** 2 * (Kx @ V @ Ky.T)
            return result.ravel()

        return mv
    else:
        Kx = build_correlation_matrix_1d(W, rho0 / W if W > 1 else 1.0, 1.0)
        Ky = build_correlation_matrix_1d(H, rho0 / H if H > 1 else 1.0, 1.0)
        K2d = sigma ** 2 * np.kron(Ky, Kx)
        return K2d


def apply_spatial_prior(x: np.ndarray, image_shape: Tuple[int, int],
                        rho0: float, sigma: float = 1.0) -> np.ndarray:
    """
    对图像向量施加空间先验（高斯平滑）。

    参数:
        x: 图像向量，形状为 (H*W,) 或展平前的图像
        image_shape: 图像尺寸 (H, W)
        rho0: 相关长度
        sigma: 幅度
    返回:
        施加先验后的图像向量
    """
    x = np.asarray(x, dtype=float).ravel()
    H, W = image_shape
    if len(x) != H * W:
        raise ValueError(f"向量长度 {len(x)} 与图像尺寸 {H*W} 不匹配")

    cov_mv = build_2d_spatial_covariance(image_shape, rho0, sigma)
    if callable(cov_mv):
        return cov_mv(x)
    else:
        return cov_mv @ x
