# -*- coding: utf-8 -*-

import numpy as np


def build_scattering_operator(n_pixels, aperture_size, wavelength,
                              phase_profile, amplitude_profile=None):
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






    raise NotImplementedError("Hole 2: build_scattering_operator 需要实现")


def svd_compress_scattering(S, rank):
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    if N == 0:
        return S, None, None, None, {}
    U, s, Vh = np.linalg.svd(S, full_matrices=False)
    R = min(rank, len(s))

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
    S = np.asarray(S, dtype=complex)
    field_in = np.asarray(field_in, dtype=complex)
    if S.ndim != 2 or field_in.ndim != 1:
        raise ValueError("S 必须为 2-D 矩阵，field_in 必须为 1-D 向量")
    if S.shape[1] != field_in.shape[0]:
        raise ValueError(f"S 列数 {S.shape[1]} 与 field_in 长度 {field_in.shape[0]} 不匹配")
    field_out = S @ field_in

    max_amp = np.max(np.abs(field_out))
    if max_amp > 1e6:
        field_out = field_out / max_amp * 1e3
    return field_out
