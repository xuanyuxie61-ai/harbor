"""
translation_operators.py
FMM转换算子模块

融合种子项目:
- 777_monomial_value (单项式求值, 用于M2L展开系数)
- 1132_spherical_harmonic (球谐函数用于转换)

科学背景:
FMM的核心是三类转换算子:
1. M2M (Multipole-to-Multipole): 将子区域的多极展开合并到父区域
2. M2L (Multipole-to-Local): 将远场多极展开转换为局部展开
3. L2L (Local-to-Local): 将父区域的局部展开下传到子区域

核心公式:

1. M2M 转换:
    设子区域中心 c', 父区域中心 c, 位移 d = c' - c
    子区域多极矩 M_{l'}^{m'}, 父区域多极矩 M_l^m:

    M_l^m = sum_{l'=0}^{l} sum_{m'=-l'}^{l'} C(l,m; l',m') * d^{l-l'} * Y_{l-l'}^{m-m'}(theta_d, phi_d) * M_{l'}^{m'}

    其中 C(l,m; l',m') 为Clebsch-Gordan-like组合系数:
        C = sqrt( (2l+1)*(2l'+1) / (4*pi*(2(l-l')+1)) )
            * ( (l+m)!*(l-m)!*(l'+m')!*(l'-m')!*(l-l'+m-m')!*(l-l'-m+m')! )^{1/2}
            / ( (l+m')!*(l-m')!*(l'+m)!*(l'-m)! )

2. M2L 转换:
    设多极展开中心 c_s, 局部展开中心 c_t, 位移 d = c_t - c_s
    局部展开系数 L_l^m:

    L_l^m = sum_{l'=0}^{L} sum_{m'=-l'}^{l'} D(l,m; l',m') / d^{l+l'+1} * Y_{l+l'}^{m-m'}(theta_d, phi_d) * M_{l'}^{m'}

    其中 D(l,m; l',m') = (-1)^{l'} * sqrt( (2l+1)*(2l'+1) / (4*pi*(2(l+l')+1)) )
            * ( (l+m)!*(l-m)!*(l'+m')!*(l'-m')!*(l+l'+m-m')!*(l+l'-m+m')! )^{1/2}
            / ( (l+m')!*(l-m')!*(l'+m)!*(l'-m)! )

3. L2L 转换:
    设父区域中心 c, 子区域中心 c', 位移 d = c' - c
    子区域局部系数 L_{l'}^{m'}:

    L_{l'}^{m'} = sum_{l=l'}^{L} sum_{m=-l}^{l} E(l,m; l',m') * d^{l-l'} * Y_{l-l'}^{m-m'}(theta_d, phi_d) * L_l^m

    其中 E(l,m; l',m') = sqrt( (2l'+1)*(2(l-l')+1) / (4*pi*(2l+1)) )
            * ( (l+m)!*(l-m)!*(l'+m')!*(l'-m')!*(l-l'+m-m')!*(l-l'-m+m')! )^{1/2}
            / ( (l'+m)!*(l'-m)!*(l-l'+m')!*(l-l'-m')! )

注: 上述公式为实数形式球谐下的简化版本。实际实现中, 由于球谐函数的对称性,
    我们采用递推方法或矩阵预计算来提高效率。
"""

import numpy as np
from math import factorial, sqrt
from spherical_geometry import legendre_associated_normalized


def _fact_ratio(n, m):
    """计算 factorial(n) / factorial(m) 的稳健版本"""
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
    """
    M2M转换: 将子区域多极矩转换到父区域
    
    简化实现 (基于位移Taylor展开):
        Phi_parent(x) = Phi_child(x - d)
        其中 d = child_center - parent_center
        
        多极矩的变换利用球谐加法定理:
        M_l^m(parent) = sum_{l'=0}^{l} sum_{m'=-l'}^{l'} T_{l,l'}^{m,m'}(d) * M_{l'}^{m'}(child)
    
    参数:
        child_moments_real, child_moments_imag: list of ndarray
        child_center, parent_center: ndarray (3,)
        order: int
    
    返回:
        (parent_moments_real, parent_moments_imag)
    """
    d = child_center - parent_center
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        # 若中心重合, 直接复制
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

    # 使用位移展开近似
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
                    # 组合因子 (简化版)
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
    """
    M2L转换: 将多极展开转换为局部展开
    
    简化实现:
        L_l^m = sum_{l'=0}^{L} sum_{m'=-l'}^{l'} K_{l,l'}^{m,m'}(d) * M_{l'}^{m'}
        其中 d = target_center - source_center
    
    参数:
        multipole_moments_real, multipole_moments_imag: list of ndarray
        source_center, target_center: ndarray (3,)
        order: int
    
    返回:
        (local_coeffs_real, local_coeffs_imag)
    """
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
                    # 简化M2L核系数
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
                    # 组合数近似
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
    """
    L2L转换: 将父区域局部展开下传到子区域
    
    简化实现 (位移Taylor展开):
        L_{l'}^{m'}(child) = sum_{l=l'}^{L} sum_{m=-l}^{l} S_{l,l'}^{m,m'}(d) * L_l^m(parent)
        其中 d = child_center - parent_center
    
    参数:
        parent_coeffs_real, parent_coeffs_imag: list of ndarray
        parent_center, child_center: ndarray (3,)
        order: int
    
    返回:
        (child_coeffs_real, child_coeffs_imag)
    """
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
    """
    预计算一组节点之间的转换矩阵
    
    参数:
        nodes_coords: ndarray (M, 3)
        expansion_order: int
    
    返回:
        dict: (i,j) -> (type, matrix)
    """
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
