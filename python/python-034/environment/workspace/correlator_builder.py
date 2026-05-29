"""
correlator_builder.py
=====================
强子关联函数（两点函数）的构造与插值。

原项目映射：
  - 632_lagrange：Lagrange 插值用于分离变量之间的场算符重构
  - 615_kdv_exact：KdV 孤子解作为重子波包试探函数

物理背景
--------
在格点 QCD 中，强子质量通过关联函数的指数衰减提取：

    C_H(t) = Σ_x ⟨ O_H(x, t) O_H†(0, 0) ⟩

其中 O_H 为具有目标强子量子数的插值算符。
对于大 t，关联函数呈指数衰减：

    C_H(t) ≈ Z_H exp( -m_H t )

介子算符示例（π 介子）：
    O_π(x) = ū(x) γ_5 d(x)

重子算符示例（核子）：
    O_N(x) = ε_{abc} [ u_a^T(x) C γ_5 d_b(x) ] u_c(x)

孤子波包试探函数
----------------
借鉴 KdV 方程的孤子解，构造重子关联函数的试探波包：

    ψ_solitonic(x, t; v) = - (v/2) sech²( (√v / 2)(x - v t - a) )

此波包在格点上离散化后，可作为重子插值算符的空间形状因子，
改善对基态的重叠。

Lagrange 插值
-------------
为在非格点位置构造关联函数，使用 Lagrange 基多项式：

    L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

通过插值可将粗格点上的关联函数映射到细格点或任意分离处，
用于有限体积质量修正。
"""

import numpy as np
from lattice_gauge import Lattice
from fermion_solver import solve_propagator_cg, point_source


def lagrange_basis_value(npol: int, ipol: int, xpol: np.ndarray,
                         xval: float) -> float:
    """
    计算第 ipol 个 Lagrange 基多项式在 xval 处的值。

    L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

    Parameters
    ----------
    npol : int
        节点数。
    ipol : int
        多项式索引（0-based）。
    xpol : np.ndarray
        插值节点。
    xval : float
        求值点。

    Returns
    -------
    pval : float
        基多项式值。
    """
    if not (0 <= ipol < npol):
        raise ValueError("ipol out of range")
    pval = 1.0
    for j in range(npol):
        if j != ipol:
            denom = xpol[ipol] - xpol[j]
            if abs(denom) < 1e-14:
                raise ValueError("Duplicate nodes in Lagrange interpolation")
            pval *= (xval - xpol[j]) / denom
    return pval


def lagrange_interpolate(xpol: np.ndarray, ypol: np.ndarray,
                         xval: float) -> float:
    """
    Lagrange 插值：通过节点 (xpol, ypol) 估计 xval 处的函数值。

    P(x) = Σ_i y_i L_i(x)
    """
    npol = len(xpol)
    result = 0.0
    for i in range(npol):
        result += ypol[i] * lagrange_basis_value(npol, i, xpol, xval)
    return result


def sech2_soliton(x: np.ndarray, t: float, v: float = 1.0,
                  a: float = 0.0) -> np.ndarray:
    """
    KdV 型孤子波包（sech² 形式）。

    ψ(x, t) = - (v/2) sech²( (√v / 2)(x - v t - a) )

    Parameters
    ----------
    x : np.ndarray
        空间坐标。
    t : float
        时间。
    v : float
        孤子速度。
    a : float
        初始位移。

    Returns
    -------
    psi : np.ndarray
        波包振幅。
    """
    arg = 0.5 * np.sqrt(abs(v)) * (x - v * t - a)
    # 数值稳定性：限制 arg 范围
    arg = np.clip(arg, -50.0, 50.0)
    return -0.5 * abs(v) / np.cosh(arg) ** 2


def meson_correlator_pion(lat: Lattice, propagators: list,
                          source_positions: list) -> np.ndarray:
    """
    构造 π 介子两点关联函数（赝标量通道）。

    算符：O_π(x) = ψ̄(x) γ_5 ψ(x)
    关联函数：
        C_π(t) = Σ_x ⟨ Tr[ γ_5 S(x, t; 0, 0) γ_5 S(0, 0; x, t) ] ⟩

    对于简化的 SU(2) 模型，使用自旋迹近似。

    Parameters
    ----------
    lat : Lattice
        格点几何。
    propagators : list
        各源点的传播子列表。
    source_positions : list
        源位置列表。

    Returns
    -------
    corr : np.ndarray
        长度为 nt 的关联函数。
    """
    # HOLE 3: 构造 π 介子赝标量关联函数
    raise NotImplementedError("Hole 3: implement pion meson correlator construction")


def baryon_correlator_nucleon(lat: Lattice, propagators: list,
                              soliton_enhance: bool = True) -> np.ndarray:
    """
    构造核子两点关联函数（自旋-1/2 重子通道）。

    在简化模型中，核子算符使用三个夸克场的反对称化组合。
    引入孤子波包形状因子改善基态重叠：

        O_N(x) = ε_{abc} q_a(x) q_b(x) q_c(x) × Φ_solitonic(x)

    Parameters
    ----------
    lat : Lattice
        格点几何。
    propagators : list
        传播子列表。
    soliton_enhance : bool
        是否使用孤子波包增强。

    Returns
    -------
    corr : np.ndarray
        关联函数。
    """
    nt = lat.dims[3]
    corr = np.zeros(nt, dtype=complex)

    # 孤子波包在空间网格上的采样
    if soliton_enhance:
        nx = lat.dims[0]
        x_coords = np.arange(nx)
        soliton_weights = sech2_soliton(x_coords, t=0.0, v=0.8)
        # 归一化
        soliton_weights = np.abs(soliton_weights)
        sw_sum = np.sum(soliton_weights)
        if sw_sum > 1e-10:
            soliton_weights /= sw_sum
    else:
        soliton_weights = np.ones(lat.dims[0]) / lat.dims[0]

    for prop in propagators:
        for t in range(nt):
            slice_sum = 0.0
            for idx in range(lat.vol):
                x = lat.index_to_site(idx)
                if x[3] != t:
                    continue
                psi = prop[(x[0], x[1], x[2], x[3])]
                # 简化：三重乘积近似（实际应做完整的色-自旋缩并）
                weight = soliton_weights[x[0]]
                # 取模方作为关联度量
                slice_sum += weight * np.vdot(psi, psi)
            corr[t] += slice_sum

    corr /= len(propagators)
    return corr.real


def correlator_effective_mass(corr: np.ndarray, dt: int = 1) -> np.ndarray:
    """
    计算有效质量。

    m_eff(t) = (1/dt) * log( C(t) / C(t+dt) )

    对于大 t，m_eff(t) → m_H（强子质量）。
    """
    nt = len(corr)
    m_eff = np.zeros(nt - dt)
    for t in range(nt - dt):
        if abs(corr[t + dt]) > 1e-15 and corr[t] / corr[t + dt] > 0:
            m_eff[t] = np.log(corr[t] / corr[t + dt]) / dt
        else:
            m_eff[t] = np.nan
    return m_eff


def correlator_interpolated_mass(corr: np.ndarray, tpol: np.ndarray,
                                 tval: float) -> float:
    """
    利用 Lagrange 插值估计关联函数在任意 t 的值，再计算有效质量。
    """
    c_interp = lagrange_interpolate(tpol, corr[tpol.astype(int)], tval)
    if c_interp > 1e-15:
        c_next = lagrange_interpolate(tpol, corr[tpol.astype(int)], tval + 1.0)
        if c_next > 1e-15 and c_interp / c_next > 0:
            return np.log(c_interp / c_next)
    return np.nan
