
import numpy as np
from typing import Tuple


def correlation_gaussian(rho: np.ndarray, rho0: float) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    if rho0 <= 0:
        raise ValueError("相关长度 rho0 必须为正")

    rhohat = rho / rho0
    return np.exp(-rhohat ** 2)


def correlation_to_covariance(C: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = C.shape[0]

    if C.shape != (n, n):
        raise ValueError("C 必须是方阵")
    if len(sigma) != n:
        raise ValueError("sigma 长度必须与 C 的维度一致")

    tol = np.sqrt(np.finfo(float).eps)


    sym_err = np.linalg.norm(C - C.T, 'fro')
    if sym_err > tol:
        raise ValueError(f"相关矩阵不对称，误差={sym_err:.3e}")


    diag_err = np.mean(np.abs(np.diag(C) - 1.0))
    if diag_err > tol:
        raise ValueError(f"相关矩阵对角线不为 1，平均误差={diag_err:.3e}")


    c_min = np.min(C)
    c_max = np.max(C)
    if c_min < -1.0 - tol or c_max > 1.0 + tol:
        raise ValueError(f"相关矩阵元素超出 [-1, 1] 范围：min={c_min:.3e}, max={c_max:.3e}")


    D = np.diag(sigma)
    K = D @ C @ D
    return K


def build_correlation_matrix_1d(n: int, rho0: float, domain_length: float = 1.0) -> np.ndarray:
    if n <= 0:
        raise ValueError("n 必须为正整数")
    x = np.linspace(0.0, domain_length, n)

    dx = np.abs(x[:, None] - x[None, :])
    C = correlation_gaussian(dx, rho0 * domain_length)

    C = 0.5 * (C + C.T)

    eigvals = np.linalg.eigvalsh(C)
    if np.min(eigvals) < 1e-12:
        C += (1e-12 - np.min(eigvals)) * np.eye(n)
    return C


def sample_paths_cholesky(n: int, n_paths: int, rho0: float,
                          domain_length: float = 1.0) -> np.ndarray:
    C = build_correlation_matrix_1d(n, rho0, domain_length)

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError as e:

        C_reg = C + 1e-10 * np.eye(n)
        L = np.linalg.cholesky(C_reg)

    Z = np.random.randn(n, n_paths)
    X = L @ Z
    return X


def build_2d_spatial_covariance(image_shape: Tuple[int, int],
                                rho0: float, sigma: float = 1.0) -> np.ndarray:
    H, W = image_shape

    if H * W > 5000:

        Kx = build_correlation_matrix_1d(W, rho0 / W if W > 1 else 1.0, 1.0)
        Ky = build_correlation_matrix_1d(H, rho0 / H if H > 1 else 1.0, 1.0)

        def mv(v: np.ndarray) -> np.ndarray:
            V = v.reshape((H, W))


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
    x = np.asarray(x, dtype=float).ravel()
    H, W = image_shape
    if len(x) != H * W:
        raise ValueError(f"向量长度 {len(x)} 与图像尺寸 {H*W} 不匹配")

    cov_mv = build_2d_spatial_covariance(image_shape, rho0, sigma)
    if callable(cov_mv):
        return cov_mv(x)
    else:
        return cov_mv @ x
