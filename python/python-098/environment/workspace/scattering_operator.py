# -*- coding: utf-8 -*-
"""
scattering_operator.py
基于 svd_gray 的 SVD 压缩思想，构建超表面散射算符的低秩近似，
实现全息传输矩阵的高效表征与降维。

核心科学问题：
  超表面可视为一个线性散射算符 S，将入射模式 |ψ_in⟩ 映射为出射模式 |ψ_out⟩：
      |ψ_out⟩ = S |ψ_in⟩
  对于 N×N 像素化的超表面，S 为 N×N 复矩阵。通过 SVD 低秩近似：
      S ≈ Σ_{k=1}^{R} σ_k u_k v_k^†
  可将计算复杂度从 O(N²) 降至 O(N·R)。

关键公式：
  1. 散射算符元素（Fraunhofer 近似）:
       S_{mn} = A_m exp(i φ_m) · sinc( (x_m - x_n)/Δ ) · sinc( (y_m - y_n)/Δ )
  2. SVD 分解:
       S = U Σ V^†
  3. 秩-R 近似:
       S_R = U_R Σ_R V_R^†
  4. 相对重构误差:
       ε(R) = ||S - S_R||_F / ||S||_F = √( Σ_{k>R} σ_k² / Σ_k σ_k² )
  5. 压缩比:
       C(R) = (N·R + R + N·R) / (N·N) = R(2N+1)/N²
"""

import numpy as np


def build_scattering_operator(n_pixels, aperture_size, wavelength,
                              phase_profile, amplitude_profile=None):
    """
    基于角谱/Fraunhofer 近似构建超表面散射算符 S。

    参数:
        n_pixels:      每边像素数（总像素 N = n_pixels^2）
        aperture_size: 孔径边长 L (m)
        wavelength:    波长 λ (m)
        phase_profile: 2-D array shape (n_pixels, n_pixels)，局部相位 φ(x,y)
        amplitude_profile: 2-D array，局部透射幅度 A(x,y)，默认全 1
    返回:
        S: N×N 复矩阵
        x_coords: 1-D 坐标
    """
    N = n_pixels * n_pixels
    dx = aperture_size / n_pixels
    x = np.linspace(-aperture_size / 2.0 + dx / 2.0,
                    aperture_size / 2.0 - dx / 2.0, n_pixels)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    phase_profile = np.asarray(phase_profile, dtype=float)
    if phase_profile.shape != (n_pixels, n_pixels):
        raise ValueError("phase_profile shape 必须等于 (n_pixels, n_pixels)")

    if amplitude_profile is None:
        amplitude_profile = np.ones((n_pixels, n_pixels), dtype=float)
    else:
        amplitude_profile = np.asarray(amplitude_profile, dtype=float)
        amplitude_profile = np.clip(amplitude_profile, 0.0, 1.0)

    # HOLE 2: 实现散射算符构建
    # 1. 计算局部透射系数 t_m = A_m exp(i φ_m)
    # 2. 构建 sinc 型空间带宽限制传播核 K
    # 3. 散射算符 S = K · diag(t)
    # 空间截止频率 k_c = π / max(dx, λ/2)
    raise NotImplementedError("Hole 2: build_scattering_operator 需要实现")


def svd_compress_scattering(S, rank):
    """
    对散射算符进行 SVD 低秩近似（参考 svd_gray_approximate）。

    参数:
        S:    N×N 复矩阵
        rank: 保留的奇异值数量 R
    返回:
        S_approx: 低秩近似矩阵
        U, s, Vh: SVD 分解结果
        metrics:  dict 包含压缩比、误差等
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    if N == 0:
        return S, None, None, None, {}
    U, s, Vh = np.linalg.svd(S, full_matrices=False)
    R = min(rank, len(s))
    # 低秩重构
    S_approx = U[:, :R] @ np.diag(s[:R]) @ Vh[:R, :]

    frob_orig = np.linalg.norm(S, 'fro')
    frob_diff = np.linalg.norm(S - S_approx, 'fro')
    rel_error = frob_diff / max(frob_orig, 1e-15)
    compression_ratio = (N * R + R + N * R) / max(N * N, 1)
    partial_sum = np.sum(s[:R]) / max(np.sum(s), 1e-15)

    metrics = {
        'rank': R,
        'compression_ratio': float(compression_ratio),
        'relative_error': float(rel_error),
        'partial_sum_ratio': float(partial_sum),
        'singular_values': s[:min(R + 5, len(s))].tolist()
    }
    return S_approx, U, s, Vh, metrics


def evaluate_compression_error(S, rank_list):
    """
    评估不同截断秩下的重构误差，用于选取最优秩。
    """
    S = np.asarray(S, dtype=complex)
    U, s, Vh = np.linalg.svd(S, full_matrices=False)
    frob_orig = np.linalg.norm(S, 'fro')
    results = []
    for r in rank_list:
        R = min(r, len(s))
        S_r = U[:, :R] @ np.diag(s[:R]) @ Vh[:R, :]
        err = np.linalg.norm(S - S_r, 'fro') / max(frob_orig, 1e-15)
        results.append({'rank': R, 'relative_error': float(err)})
    return results


def apply_scattering_operator(S, field_in):
    """
    将散射算符作用于入射场：E_out = S @ E_in。
    包含维度与数值边界检查。
    """
    S = np.asarray(S, dtype=complex)
    field_in = np.asarray(field_in, dtype=complex)
    if S.ndim != 2 or field_in.ndim != 1:
        raise ValueError("S 必须为 2-D 矩阵，field_in 必须为 1-D 向量")
    if S.shape[1] != field_in.shape[0]:
        raise ValueError(f"S 列数 {S.shape[1]} 与 field_in 长度 {field_in.shape[0]} 不匹配")
    field_out = S @ field_in
    # 数值保护：限制极端幅值
    max_amp = np.max(np.abs(field_out))
    if max_amp > 1e6:
        field_out = field_out / max_amp * 1e3
    return field_out
