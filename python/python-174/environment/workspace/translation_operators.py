
import numpy as np
from math import factorial, sqrt
from spherical_geometry import legendre_associated_normalized


def _fact_ratio(n, m):
    if n < 0 or m < 0:
        return 0.0
    if n >= m:
        val = 1.0
        for k in range(m + 1, n + 1):
            val *= k
        return val
    else:
        val = 1.0
        for k in range(n + 1, m + 1):
            val /= k
        return val


def m2m_translate(child_moments_real, child_moments_imag, child_center, parent_center, order):
    d = child_center - parent_center
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:

        return [m.copy() for m in child_moments_real], [m.copy() for m in child_moments_imag]

    theta_d = np.arccos(np.clip(d[2] / d_norm, -1.0, 1.0))
    phi_d = np.arctan2(d[1], d[0])
    if phi_d < 0:
        phi_d += 2.0 * np.pi

    parent_real = []
    parent_imag = []
    for l in range(order + 1):
        parent_real.append(np.zeros(l + 1))
        parent_imag.append(np.zeros(l + 1))


    for l in range(order + 1):
        for m in range(l + 1):
            val_real = 0.0
            val_imag = 0.0
            for lp in range(l + 1):
                mp_max = min(lp, m)
                for mp in range(mp_max + 1):
                    diff = l - lp
                    m_diff = abs(m - mp)
                    if diff < m_diff:
                        continue

                    coeff = (_fact_ratio(l + m, lp + mp) *
                             _fact_ratio(l - m, lp - mp) /
                             factorial(max(1, diff)))
                    d_power = d_norm ** diff
                    plm = legendre_associated_normalized(diff, m_diff, np.cos(theta_d))
                    y_real = plm[diff] * np.cos((m - mp) * phi_d)
                    y_imag = plm[diff] * np.sin((m - mp) * phi_d)

                    m_r = child_moments_real[lp][mp]
                    m_i = child_moments_imag[lp][mp] if mp > 0 else 0.0
                    val_real += coeff * d_power * (m_r * y_real - m_i * y_imag)
                    val_imag += coeff * d_power * (m_r * y_imag + m_i * y_real)
            parent_real[l][m] = val_real
            parent_imag[l][m] = val_imag

    return parent_real, parent_imag


def m2l_translate(multipole_moments_real, multipole_moments_imag, source_center, target_center, order):
    d = target_center - source_center
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        raise ValueError("源中心和目标中心重合, M2L不适用")

    theta_d = np.arccos(np.clip(d[2] / d_norm, -1.0, 1.0))
    phi_d = np.arctan2(d[1], d[0])
    if phi_d < 0:
        phi_d += 2.0 * np.pi

    local_real = []
    local_imag = []
    for l in range(order + 1):
        local_real.append(np.zeros(l + 1))
        local_imag.append(np.zeros(l + 1))

    for l in range(order + 1):
        for m in range(l + 1):
            val_real = 0.0
            val_imag = 0.0
            for lp in range(order + 1):
                for mp in range(lp + 1):

                    L_total = l + lp
                    m_diff = abs(m - mp)
                    if L_total < m_diff:
                        continue
                    if L_total == 0:
                        kernel = 1.0 / d_norm
                    else:
                        kernel = 1.0 / (d_norm ** (L_total + 1))
                    coeff = ((-1) ** lp) * sqrt(
                        (2 * l + 1) * (2 * lp + 1) / (4.0 * np.pi * (2 * L_total + 1))
                    )

                    comb = (_fact_ratio(l + m, lp + mp) *
                            _fact_ratio(l - m, lp - mp))
                    coeff *= comb

                    plm = legendre_associated_normalized(L_total, m_diff, np.cos(theta_d))
                    y_real = plm[L_total] * np.cos((m - mp) * phi_d)
                    y_imag = plm[L_total] * np.sin((m - mp) * phi_d)

                    m_r = multipole_moments_real[lp][mp]
                    m_i = multipole_moments_imag[lp][mp] if mp > 0 else 0.0
                    val_real += coeff * kernel * (m_r * y_real - m_i * y_imag)
                    val_imag += coeff * kernel * (m_r * y_imag + m_i * y_real)
            local_real[l][m] = val_real
            local_imag[l][m] = val_imag

    return local_real, local_imag


def l2l_translate(parent_coeffs_real, parent_coeffs_imag, parent_center, child_center, order):
    d = child_center - parent_center
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        return [c.copy() for c in parent_coeffs_real], [c.copy() for c in parent_coeffs_imag]

    theta_d = np.arccos(np.clip(d[2] / d_norm, -1.0, 1.0))
    phi_d = np.arctan2(d[1], d[0])
    if phi_d < 0:
        phi_d += 2.0 * np.pi

    child_real = []
    child_imag = []
    for l in range(order + 1):
        child_real.append(np.zeros(l + 1))
        child_imag.append(np.zeros(l + 1))

    for lp in range(order + 1):
        for mp in range(lp + 1):
            val_real = 0.0
            val_imag = 0.0
            for l in range(lp, order + 1):
                m_max = min(l, mp + (l - lp))
                for m in range(mp, m_max + 1):
                    diff = l - lp
                    m_diff = abs(m - mp)
                    if diff < m_diff:
                        continue
                    coeff = (_fact_ratio(l + m, lp + mp) *
                             _fact_ratio(l - m, lp - mp) /
                             factorial(max(1, diff)))
                    d_power = d_norm ** diff
                    plm = legendre_associated_normalized(diff, m_diff, np.cos(theta_d))
                    y_real = plm[diff] * np.cos((m - mp) * phi_d)
                    y_imag = plm[diff] * np.sin((m - mp) * phi_d)

                    l_r = parent_coeffs_real[l][m]
                    l_i = parent_coeffs_imag[l][m] if m > 0 else 0.0
                    val_real += coeff * d_power * (l_r * y_real - l_i * y_imag)
                    val_imag += coeff * d_power * (l_r * y_imag + l_i * y_real)
            child_real[lp][mp] = val_real
            child_imag[lp][mp] = val_imag

    return child_real, child_imag


def compute_translation_matrix(nodes_coords, expansion_order):
    M = nodes_coords.shape[0]
    matrices = {}
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            d = nodes_coords[j] - nodes_coords[i]
            d_norm = np.linalg.norm(d)
            matrices[(i, j)] = {
                "distance": d_norm,
                "direction": d / (d_norm + 1e-15)
            }
    return matrices
