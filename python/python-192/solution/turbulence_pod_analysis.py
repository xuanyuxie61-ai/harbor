"""
================================================================================
湍流本征正交分解(POD)分析模块 (turbulence_pod_analysis.py)
================================================================================
融合项目:
  - 326_eigenfaces (pc_vectors): 主成分分析/特征向量提取

在湍流研究中，POD（也称Karhunen-Loève分解）用于提取流动中最具能量的相干结构。
本模块提供：
  1. 快照POD方法（Sirovich 1987）
  2. 能量谱分析与模态截断
  3. 降阶模型(ROM)重构

数学基础:
    对速度场快照矩阵 A = [u₁, u₂, ..., u_N]（每列为一个快照），
    寻找最优正交基 {φ_k} 使得投影能量最大：

        max_{φ_k}  Σ_k (A^T φ_k, φ_k) / (φ_k, φ_k)

    等价于求解协方差矩阵 C = A^T A 的特征值问题：

        C v_k = λ_k v_k,    φ_k = A v_k / √λ_k

    模态能量占比: E_k = λ_k / Σ_j λ_j
================================================================================
"""

import numpy as np
from utils_numerical import safe_divide


def snapshot_pod(A: np.ndarray, num_modes: int = None, energy_threshold: float = 0.99) -> dict:
    """
    快照POD方法（Sirovich方法）

    对于大规模数据（空间点数 M >> 快照数 N），直接计算 M×M 协方差矩阵
    不可行。Sirovich技巧转而求解 N×N 矩阵 L = A^T A 的特征值问题：

        L v_k = λ_k v_k

    然后通过 φ_k = A v_k / ||A v_k|| 恢复空间模态。

    算法复杂度从 O(M³) 降至 O(N³ + M N²)。

    参数:
        A: 快照矩阵 (M x N)，每列为一个流场快照（已去均值）
        num_modes: 保留模态数（若为None则按能量阈值自动确定）
        energy_threshold: 累积能量阈值（如0.99表示保留99%能量）

    返回:
        dict 包含模态、特征值、能量占比、降阶基
    """
    M, N = A.shape

    if M == 0 or N == 0:
        return {
            'modes': np.zeros((M, 1)),
            'eigenvalues': np.zeros(1),
            'energy_fraction': np.zeros(1),
            'cum_energy': np.zeros(1),
            'num_modes': 0,
            ' Psi': np.zeros(M)
        }

    # 计算均值并去均值
    Psi = np.mean(A, axis=1)
    A_centered = A - Psi[:, None]

    # 协方差矩阵（快照法）
    L = A_centered.T @ A_centered

    # 正则化防止病态
    L += 1e-12 * np.eye(N) * np.trace(L) / N

    # 特征值分解
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # 失败时 fallback 到 SVD
        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
        eigenvalues = S ** 2
        eigenvectors = Vt.T

    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 归一化特征值（方差）
    eigenvalues = np.maximum(eigenvalues, 0.0) / max(N - 1, 1)

    # 过滤极小特征值
    tol = 1e-10 * np.max(eigenvalues) if np.max(eigenvalues) > 0 else 1e-14
    good_mask = eigenvalues > tol
    num_good = int(np.sum(good_mask))

    eigenvalues = eigenvalues[good_mask]
    eigenvectors = eigenvectors[:, good_mask]

    # 恢复空间模态: φ_k = A v_k
    modes = A_centered @ eigenvectors

    # 归一化模态
    for k in range(modes.shape[1]):
        norm = np.linalg.norm(modes[:, k])
        if norm > 1e-14:
            modes[:, k] /= norm

    # 能量分析
    total_energy = np.sum(eigenvalues)
    energy_fraction = safe_divide(eigenvalues, total_energy)
    cum_energy = np.cumsum(energy_fraction)

    # 确定保留模态数
    if num_modes is None:
        num_modes = int(np.searchsorted(cum_energy, energy_threshold) + 1)
    num_modes = min(num_modes, num_good)

    modes = modes[:, :num_modes]
    eigenvalues = eigenvalues[:num_modes]
    energy_fraction = energy_fraction[:num_modes]
    cum_energy = cum_energy[:num_modes]

    return {
        'modes': modes,
        'eigenvalues': eigenvalues,
        'energy_fraction': energy_fraction,
        'cum_energy': cum_energy,
        'num_modes': num_modes,
        'Psi': Psi,
        'A_centered': A_centered
    }


def reconstruct_from_pod(pod_result: dict, coefficients: np.ndarray = None) -> np.ndarray:
    """
    利用POD模态重构流场

        u_reconstructed = Ψ + Σ_k a_k φ_k

    其中 a_k 为模态系数。若未提供，使用原始投影系数：
        a_k = (u - Ψ, φ_k)
    """
    Psi = pod_result['Psi']
    modes = pod_result['modes']
    A_centered = pod_result['A_centered']

    if coefficients is None:
        coefficients = modes.T @ A_centered

    reconstruction = Psi[:, None] + modes @ coefficients
    return reconstruction


def compute_turbulent_kinetic_energy(u_snapshots: np.ndarray, v_snapshots: np.ndarray) -> dict:
    """
    计算湍动能及其POD分解

    湍动能定义:
        k = (1/2) (u'² + v'²)

    其中 u' = u - ⟨u⟩ 为脉动速度，⟨·⟩ 表示时间/系综平均。

    Reynolds应力张量:
        τ_ij^R = -ρ ⟨u_i' u_j'⟩

    参数:
        u_snapshots: u速度快照 (M x N)
        v_snapshots: v速度快照 (M x N)

    返回:
        dict 包含TKE、Reynolds应力、POD模态
    """
    M, N = u_snapshots.shape

    # 时间平均
    u_mean = np.mean(u_snapshots, axis=1)
    v_mean = np.mean(v_snapshots, axis=1)

    # 脉动
    up = u_snapshots - u_mean[:, None]
    vp = v_snapshots - v_mean[:, None]

    # 湍动能
    tke = 0.5 * (np.mean(up ** 2, axis=1) + np.mean(vp ** 2, axis=1))

    # Reynolds应力
    R_uv = np.mean(up * vp, axis=1)
    R_uu = np.mean(up ** 2, axis=1)
    R_vv = np.mean(vp ** 2, axis=1)

    # 组合快照用于联合POD
    A = np.vstack([up, vp])
    pod = snapshot_pod(A, num_modes=min(20, N, 2 * M))

    return {
        'tke': tke,
        'R_uu': R_uu,
        'R_vv': R_vv,
        'R_uv': R_uv,
        'u_mean': u_mean,
        'v_mean': v_mean,
        'pod': pod
    }


def compute_pod_galerkin_coefficients(pod_modes: np.ndarray, snapshots: np.ndarray) -> np.ndarray:
    """
    计算Galerkin投影系数

    对于第n个快照，系数为:
        a_k^n = (snapshot_n - Ψ, φ_k)

    在Galerkin-ROM中，这些系数的时间演化由低维ODE控制：

        da_k/dt = C_k + Σ_i L_{ki} a_i + Σ_{i,j} Q_{kij} a_i a_j
    """
    Psi = np.mean(snapshots, axis=1)
    A_centered = snapshots - Psi[:, None]
    coefficients = pod_modes.T @ A_centered
    return coefficients


def compute_modal_dynamics(pod_result: dict, dt: float) -> dict:
    """
    分析模态系数的时间动力学

    计算各模态的：
      - 时间自相关系数
      - 主导频率（通过FFT）
      - 模态间互相关性
    """
    A_centered = pod_result['A_centered']
    modes = pod_result['modes']
    coeffs = modes.T @ A_centered
    num_modes = coeffs.shape[0]

    # 自相关系数
    autocorr = []
    for k in range(num_modes):
        c = coeffs[k, :]
        c_norm = c - np.mean(c)
        if np.std(c_norm) < 1e-14:
            autocorr.append(np.zeros(len(c)))
            continue
        corr = np.correlate(c_norm, c_norm, mode='full')
        corr = corr[len(corr) // 2:]
        corr /= corr[0] if corr[0] > 0 else 1.0
        autocorr.append(corr)

    # 主导频率
    frequencies = []
    for k in range(num_modes):
        c = coeffs[k, :]
        fft_vals = np.abs(np.fft.rfft(c))
        freqs = np.fft.rfftfreq(len(c), d=dt)
        if len(freqs) > 1:
            peak_idx = np.argmax(fft_vals[1:]) + 1
            frequencies.append(float(freqs[peak_idx]))
        else:
            frequencies.append(0.0)

    return {
        'coefficients': coeffs,
        'autocorrelation': autocorr,
        'dominant_frequencies': frequencies
    }
