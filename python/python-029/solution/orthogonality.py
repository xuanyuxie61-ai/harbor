"""
orthogonality.py
=================
波函数正交性与 L² 内积计算模块

基于种子项目 313_dot_l2 的 L² 内积计算思想，
本模块为核反应波函数提供正交性校验、重叠积分
以及通道耦合矩阵元计算。

核心公式
--------
L² 内积定义:
    ⟨f|g⟩ = ∫_a^b f*(r) g(r) dr

径向波函数正交归一化条件:
    ∫_0^∞ u_{l,j}*(r) u_{l',j'}(r) dr = δ_{ll'} δ_{jj'}

光学模型中的通道耦合矩阵元:
    V_{αβ} = ∫_0^∞ u_α(r) V_{coupl}(r) u_β(r) dr

其中 V_{coupl} 为耦合势 (如变形光学势的多极展开项)。

Gram-Schmidt 正交化:
    对非正交基 {v_i}，构造正交基 {e_i}:
        e_i = v_i - Σ_{j<i} ⟨e_j|v_i⟩ e_j
        e_i = e_i / ||e_i||
"""

import numpy as np


def l2_inner_product(f, g, r, method='trapezoid'):
    """
    计算两个函数在径向网格上的 L² 内积。

    Parameters
    ----------
    f, g : ndarray
        函数值数组 (允许复数)。
    r : ndarray
        径向坐标网格。
    method : str
        积分方法: 'trapezoid', 'simpson'。

    Returns
    -------
    inner : complex
        内积 ⟨f|g⟩ = ∫ f*(r) g(r) dr。
    """
    f = np.asarray(f)
    g = np.asarray(g)
    r = np.asarray(r)

    if len(f) != len(g) or len(f) != len(r):
        raise ValueError("f, g, r 长度必须一致")
    if len(r) < 2:
        raise ValueError("至少需要 2 个网格点")

    integrand = np.conj(f) * g

    if method == 'trapezoid':
        return np.trapezoid(integrand, r)
    elif method == 'simpson':
        from scipy.integrate import simpson
        return simpson(integrand, x=r)
    else:
        raise ValueError(f"未知积分方法: {method}")


def l2_norm(f, r, method='trapezoid'):
    """计算 L² 范数 ||f|| = sqrt(⟨f|f⟩)。"""
    return np.sqrt(np.abs(l2_inner_product(f, f, r, method)))


def normalize_wavefunction(u, r, method='trapezoid'):
    """
    将波函数归一化为单位 L² 范数。

    u_norm(r) = u(r) / ||u||
    """
    norm = l2_norm(u, r, method)
    if norm < 1e-30:
        return u.copy()
    return u / norm


def check_orthogonality(wavefunctions, r, threshold=1e-10):
    """
    检查一组波函数的正交性。

    Parameters
    ----------
    wavefunctions : list of ndarray
        波函数列表。
    r : ndarray
        径向网格。
    threshold : float
        正交性阈值。

    Returns
    -------
    ortho_matrix : ndarray
        重叠矩阵 ⟨u_i|u_j⟩。
    is_orthogonal : bool
        是否近似正交。
    """
    n = len(wavefunctions)
    overlap = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            overlap[i, j] = l2_inner_product(wavefunctions[i], wavefunctions[j], r)

    # 检查对角元接近 1，非对角元接近 0
    diag_deviation = np.max(np.abs(np.diag(overlap) - 1.0))
    offdiag_max = np.max(np.abs(overlap - np.eye(n)))
    is_orthogonal = (diag_deviation < threshold) and (offdiag_max < threshold)

    return overlap, is_orthogonal


def gram_schmidt_orthogonalization(wavefunctions, r, method='trapezoid'):
    """
    对非正交波函数基进行 Gram-Schmidt 正交化。

    Parameters
    ----------
    wavefunctions : list of ndarray
        输入波函数列表。
    r : ndarray
        径向网格。

    Returns
    -------
    ortho_functions : list of ndarray
        正交归一化后的波函数。
    """
    ortho = []
    for u in wavefunctions:
        v = u.copy()
        for e in ortho:
            proj = l2_inner_product(e, v, r, method)
            v = v - proj * e
        norm = l2_norm(v, r, method)
        if norm > 1e-30:
            v = v / norm
        ortho.append(v)
    return ortho


def coupling_matrix_element(u_alpha, u_beta, V_coupl, r, method='trapezoid'):
    """
    计算通道耦合矩阵元。

    V_{αβ} = ∫_0^∞ u_α*(r) V_{coupl}(r) u_β(r) dr

    Parameters
    ----------
    u_alpha, u_beta : ndarray
        通道 α 和 β 的径向波函数。
    V_coupl : ndarray
        耦合势 (实数或复数)。
    r : ndarray
        径向网格。

    Returns
    -------
    V_ab : complex
        耦合矩阵元。
    """
    integrand = np.conj(u_alpha) * V_coupl * u_beta
    if method == 'trapezoid':
        return np.trapezoid(integrand, r)
    else:
        from scipy.integrate import simpson
        return simpson(integrand, x=r)


def deformation_coupling_potential(r, beta_l, R0, a, l_order):
    """
    构造变形核的光学势多极耦合项。

    对于 l 阶表面振动:
        V_{coupl}^{(l)}(r) = -β_l R0 ∂V_WS/∂r

    其中 β_l 为变形参数，V_WS 为 Woods-Saxon 势。

    Parameters
    ----------
    r : ndarray
        径向坐标。
    beta_l : float
        l 阶变形参数。
    R0 : float
        核半径。
    a : float
        弥散参数。
    l_order : int
        多极阶数 (通常为 2, 3, 4)。

    Returns
    -------
    V_coupl : ndarray
        耦合势。
    """
    from optical_potential import woods_saxon_derivative
    # 导数 WS 形状
    dV_dr = woods_saxon_derivative(r, 1.0, R0, a)
    # 变形耦合势
    V_coupl = -beta_l * R0 * dV_dr
    return V_coupl


def coupled_channels_overlap(wavefunctions_dict, r, method='trapezoid'):
    """
    计算耦合道系统中所有通道之间的重叠矩阵。

    Parameters
    ----------
    wavefunctions_dict : dict
        键为通道标签，值为波函数数组。
    r : ndarray
        径向网格。

    Returns
    -------
    overlap_matrix : ndarray
        重叠矩阵。
    channel_labels : list
        通道标签列表。
    """
    labels = list(wavefunctions_dict.keys())
    n = len(labels)
    overlap = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            overlap[i, j] = l2_inner_product(
                wavefunctions_dict[labels[i]],
                wavefunctions_dict[labels[j]],
                r, method
            )

    return overlap, labels


if __name__ == "__main__":
    # 自检
    r = np.linspace(0.01, 10.0, 200)
    f = np.sin(r) * np.exp(-r)
    g = np.cos(r) * np.exp(-r)

    inner = l2_inner_product(f, g, r)
    norm_f = l2_norm(f, r)
    print(f"⟨f|g⟩ = {inner:.6f}")
    print(f"||f|| = {norm_f:.6f}")

    # 正交化测试
    u1 = np.sin(r) * np.exp(-r / 2)
    u2 = r * np.sin(r) * np.exp(-r / 2)
    ortho = gram_schmidt_orthogonalization([u1, u2], r)
    overlap, is_ortho = check_orthogonality(ortho, r)
    print(f"正交化后重叠矩阵:\n{np.round(overlap, 6)}")
    print(f"是否正交: {is_ortho}")

    # 耦合矩阵元
    V_c = deformation_coupling_potential(r, beta_l=0.2, R0=5.0, a=0.65, l_order=2)
    V_ab = coupling_matrix_element(ortho[0], ortho[1], V_c, r)
    print(f"耦合矩阵元 V_01 = {V_ab:.6f}")
