"""
complex_analysis.py
复频域稳定性分析模块
融合种子项目：
  - 131_c8lib（复数矩阵运算）

核心科学内容：
接触系统的动力稳定性可通过复特征值分析判定。
对于含摩擦的离散系统：
M \ddot{u} + C \dot{u} + K u = 0
引入状态空间：A z = \lambda z
若存在 Re(\lambda) > 0，则系统不稳定（摩擦诱发振动/颤振）。
"""
import numpy as np
from typing import Tuple, List
from utils import c8_norm_l2, c8mat_norm_fro


def build_complex_mass_damping_stiffness(
    M: np.ndarray, C: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    构造复模态分析的复数矩阵组。
    在频域中，动力刚度矩阵为：
    D(\omega) = K - \omega^2 M + i \omega C
    """
    return M, C, K


def complex_modal_analysis(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    状态空间复特征值分析。

    状态向量 z = [u; \dot{u}]，状态方程：
    \dot{z} = A z,  A = \begin{bmatrix} 0 & I \\ -M^{-1}K & -M^{-1}C \end{bmatrix}

    求解特征值 \lambda 满足 det(A - \lambda I) = 0。
    若 max(Re(\lambda)) > 0，系统不稳定。
    """
    n = K.shape[0]
    # 使用质量矩阵的近似（一致质量矩阵或集中质量矩阵）
    Minv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def complex_damping_matrix_from_friction(
    K: np.ndarray, contact_nodes: np.ndarray,
    friction_coeff: float, normal_pressure: np.ndarray
) -> np.ndarray:
    r"""
    基于 Coulomb 摩擦构造等效复阻尼矩阵。

    简化模型：在接触节点引入速度依赖的摩擦阻尼：
    C_{eff} = \eta K_{contact}
    其中 \eta = \mu \cdot \bar{p}_n / v_{ref}
    """
    n = K.shape[0]
    C = np.zeros((n, n))
    if len(normal_pressure) == 0:
        return C
    p_avg = np.mean(normal_pressure)
    v_ref = 1.0  # 参考速度
    eta = friction_coeff * p_avg / v_ref
    for node in contact_nodes:
        gdof_x = 2 * node
        gdof_y = 2 * node + 1
        C[gdof_x, gdof_x] += eta * K[gdof_x, gdof_x]
        C[gdof_y, gdof_y] += eta * K[gdof_y, gdof_y] * 0.1
    return C


def stability_criterion(eigenvalues: np.ndarray) -> dict:
    r"""
    基于复特征值的稳定性判据。

    1. 最大实部：\alpha_{max} = max_i Re(\lambda_i)
    2. 阻尼比：\xi_i = -Re(\lambda_i) / |\lambda_i|
    3. 颤振频率：f_i = Im(\lambda_i) / (2\pi)
    """
    alpha_max = float(np.max(np.real(eigenvalues)))
    unstable_count = int(np.sum(np.real(eigenvalues) > 0))
    # 只考虑有物理意义的模态（排除无穷大频率）
    valid = np.abs(eigenvalues) < 1e10
    if np.sum(valid) > 0:
        ev_valid = eigenvalues[valid]
        damping_ratios = -np.real(ev_valid) / (np.abs(ev_valid) + 1e-20)
        flutter_freqs = np.imag(ev_valid) / (2.0 * np.pi)
    else:
        damping_ratios = np.array([0.0])
        flutter_freqs = np.array([0.0])
    return {
        "alpha_max": alpha_max,
        "unstable_count": unstable_count,
        "min_damping_ratio": float(np.min(damping_ratios)),
        "max_flutter_freq_hz": float(np.max(np.abs(flutter_freqs))),
        "critical_modes": unstable_count
    }


def complex_matrix_power_iteration(A_complex: np.ndarray, max_iter: int = 50) -> complex:
    r"""
    复矩阵幂迭代估计主导特征值（融合 c8lib 的复数运算）。

    迭代：z_{k+1} = A z_k / \|A z_k\|_2
    估计：\lambda \approx z_k^H A z_k
    """
    n = A_complex.shape[0]
    z = np.random.randn(n) + 1j * np.random.randn(n)
    z = z / (c8_norm_l2(z) + 1e-20)
    lam = 0.0 + 0.0j
    for _ in range(max_iter):
        w = A_complex @ z
        norm_w = c8_norm_l2(w)
        if norm_w < 1e-20:
            break
        z = w / norm_w
        lam_new = np.vdot(z, A_complex @ z)
        if abs(lam_new - lam) < 1e-12:
            lam = lam_new
            break
        lam = lam_new
    return complex(lam)


def frequency_response_function(K: np.ndarray, M: np.ndarray, C: np.ndarray,
                                 omega_range: np.ndarray,
                                 load_dof: int) -> np.ndarray:
    r"""
    计算接触系统的频响函数（FRF）。

    H(\omega) = (K - \omega^2 M + i \omega C)^{-1}
    在 load_dof 处的位移响应幅值：|u(\omega)|
    """
    n = K.shape[0]
    response = np.zeros(len(omega_range))
    f = np.zeros(n)
    f[load_dof] = 1.0
    for idx, w in enumerate(omega_range):
        D = K - w ** 2 * M + 1j * w * C
        try:
            u = np.linalg.solve(D, f)
            response[idx] = np.abs(u[load_dof])
        except np.linalg.LinAlgError:
            response[idx] = 0.0
    return response
