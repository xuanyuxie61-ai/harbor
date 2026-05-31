"""
utils.py
通用数值工具与辅助函数

包含：
- 数值稳定性工具
- 数组操作
- 特殊函数
- 单位转换
"""

import numpy as np


def safe_normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    安全归一化矢量，处理零矢量情况
    
    Parameters
    ----------
    v : np.ndarray
    axis : int
    
    Returns
    -------
    v_norm : np.ndarray
    """
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    return v / norm


def finite_difference_gradient(func: callable, x: np.ndarray,
                                delta: float = 1e-6) -> np.ndarray:
    """
    数值梯度（中心差分）
    
    Parameters
    ----------
    func : callable
        标量函数
    x : np.ndarray
    delta : float
    
    Returns
    -------
    grad : np.ndarray
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += delta
        x_minus[i] -= delta
        grad[i] = (func(x_plus) - func(x_minus)) / (2.0 * delta)
    
    return grad


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues旋转公式
    
    R = I + sin(theta)*K + (1-cos(theta))*K^2
    
    其中K是轴的叉积矩阵。
    
    Parameters
    ----------
    axis : np.ndarray, shape (3,)
    angle : float
    
    Returns
    -------
    R : np.ndarray, shape (3, 3)
    """
    axis = safe_normalize(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


def levi_civita(i: int, j: int, k: int) -> int:
    """
    Levi-Civita符号 epsilon_ijk
    """
    if i == j or j == k or i == k:
        return 0
    perm = [i, j, k]
    inversions = 0
    for a in range(3):
        for b in range(a + 1, 3):
            if perm[a] > perm[b]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    高斯函数
    
    g(x) = (1/sqrt(2*pi)*sigma) * exp(-(x-mu)^2/(2*sigma^2))
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def fermi_dirac(energy: np.ndarray, mu: float, kbt: float) -> np.ndarray:
    """
    Fermi-Dirac分布
    
    f(E) = 1 / (exp((E-mu)/k_B*T) + 1)
    """
    if kbt < 1e-15:
        return np.where(energy <= mu, 1.0, 0.0)
    
    x = (energy - mu) / kbt
    # 限制指数防止溢出
    x = np.clip(x, -700, 700)
    return 1.0 / (np.exp(x) + 1.0)


def check_symmetric(a: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    检查矩阵是否对称
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_positive_definite(a: np.ndarray) -> bool:
    """
    检查矩阵是否正定（通过Cholesky分解）
    """
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False
