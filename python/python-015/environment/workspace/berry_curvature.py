"""
berry_curvature.py
Berry曲率、Berry联络与Berry相位计算

凝聚态物理核心公式：

Berry联络：
    A_n(k) = i <u_n(k)| \nabla_k |u_n(k)>

Berry曲率（张量形式）：
    Omega_{n,ab}(k) = \partial_{k_a} A_{n,b} - \partial_{k_b} A_{n,a}
                    = -2 * Im sum_{m!=n} <u_n|\partial_{k_a}H|m><m|\partial_{k_b}H|n>
                                         / (E_n - E_m)^2

对于Weyl节点（线性模型），Berry曲率有解析形式：
    Omega_{\pm}(k) = \pm (1/2) * k / |k|^3

Berry相位（沿闭合路径C）：
    gamma_n = oint_C A_n(k) · dk = i oint_C <u_n|\nabla_k|u_n> · dk

Chern数（对二维截面S积分）：
    C_n = (1/2*pi) \int_S Omega_{n,xy}(k) d^2k

Weyl荷（对包围Weyl节点的闭合曲面S积分）：
    Q = (1/2*pi) \oint_S Omega(k) · dS = \pm 1
"""

import numpy as np
from typing import Tuple
from weyl_hamiltonian import WeylHamiltonian, velocity_operator


def berry_connection_numeric(ham: WeylHamiltonian, k: np.ndarray,
                              band_index: int = 0,
                              delta: float = 1e-6) -> np.ndarray:
    """
    数值计算Berry联络 A_n(k) = i <u_n| \nabla_k |u_n>
    
    使用中心差分计算本征矢的k空间梯度：
        |\partial_{k_a} u_n> ≈ [|u_n(k + delta*e_a)> - |u_n(k - delta*e_a)>] / (2*delta)
    
    并进行规范固定（消除整体相位不确定性）。
    
    Parameters
    ----------
    ham : WeylHamiltonian
    k : np.ndarray, shape (3,)
    band_index : int
        能带指标（0为价带，1为导带）
    delta : float
        差分步长
    
    Returns
    -------
    A : np.ndarray, shape (3,)
        Berry联络的三个分量
    """
    k = np.asarray(k, dtype=float)
    if k.shape != (3,):
        raise ValueError("k必须是三维矢量")
    
    A = np.zeros(3, dtype=complex)
    
    # 获取参考点的本征矢
    _, u0 = ham.eigenproblem(k)
    u_ref = u0[:, band_index].copy()
    
    # 规范固定：使u_ref的第一个分量为实数且正
    phase = np.exp(-1.0j * np.angle(u_ref[0])) if abs(u_ref[0]) > 1e-14 else 1.0
    u_ref = u_ref * phase
    
    for a in range(3):
        kp = k.copy()
        km = k.copy()
        kp[a] += delta
        km[a] -= delta
        
        _, up = ham.eigenproblem(kp)
        _, um = ham.eigenproblem(km)
        
        up_vec = up[:, band_index]
        um_vec = um[:, band_index]
        
        # 规范固定：与参考点保持相同规范
        if abs(up_vec[0]) > 1e-14:
            up_vec *= np.exp(-1.0j * np.angle(up_vec[0]))
        if abs(um_vec[0]) > 1e-14:
            um_vec *= np.exp(-1.0j * np.angle(um_vec[0]))
        
        # 中心差分
        du = (up_vec - um_vec) / (2.0 * delta)
        A[a] = 1.0j * np.vdot(u_ref, du)
    
    # Berry联络应为实数（对于非简并能带）
    if np.max(np.abs(A.imag)) > 1e-8:
        raise RuntimeWarning(f"Berry联络虚部过大: {np.max(np.abs(A.imag))}")
    
    return A.real


def berry_curvature_numeric(ham: WeylHamiltonian, k: np.ndarray,
                             band_index: int = 0,
                             delta: float = 1e-5) -> np.ndarray:
    """
    数值计算Berry曲率 Omega_n(k)
    
    方法一：通过Berry联络的旋度
        Omega_{ab} = \partial_a A_b - \partial_b A_a
    
    方法二：通过速度算符（更稳定）
        Omega_{n,ab} = -2 * Im sum_{m!=n} <n|v_a|m><m|v_b|n> / (E_n - E_m)^2
    
    这里采用方法二，因为它对规范选择不敏感。
    
    Parameters
    ----------
    ham : WeylHamiltonian
    k : np.ndarray, shape (3,)
    band_index : int
    delta : float
    
    Returns
    -------
    Omega : np.ndarray, shape (3, 3)
        反对称张量，Omega[i,j] = Omega_{ij}
    """
    k = np.asarray(k, dtype=float)
    energies, eigenvectors = ham.eigenproblem(k)
    
    if band_index not in (0, 1):
        raise ValueError("band_index必须是0或1")
    
    n_bands = 2
    Omega = np.zeros((3, 3))
    
    # 计算速度算符矩阵元
    v_ops = velocity_operator(ham, k, delta)
    
    # TODO Hole_2: 基于速度算符矩阵元计算Berry曲率张量
    # 公式: Omega_{n,ab} = -2 * Im sum_{m!=n} <n|v_a|m><m|v_b|n> / (E_n - E_m)^2
    # 其中:
    #   - n_bands = 2 (二能带系统)
    #   - v_ops[a] 是方向a的速度算符矩阵 (2x2)
    #   - eigenvectors[:, band_index] 是本征态 |n>
    #   - 需要遍历 m != band_index 的能带
    #   - 结果 Omega 是 3x3 反对称张量
    raise NotImplementedError("Hole_2: 数值Berry曲率张量计算待实现")


def berry_curvature_analytic_linear(k: np.ndarray, chirality: int = 1) -> np.ndarray:
    """
    线性Weyl模型的Berry曲率解析表达式
    
    对于 H = hbar*v_F * k·sigma，有：
        Omega_{\pm}(k) = \pm (1/2) * k / |k|^3
    
    即反对称张量形式：
        Omega_{ij} = \pm (1/2) * epsilon_{ijk} * k_k / |k|^3
    
    Parameters
    ----------
    k : np.ndarray, shape (3,)
    chirality : int
        +1 或 -1，对应导带/价带
    
    Returns
    -------
    Omega : np.ndarray, shape (3, 3)
    """
    k = np.asarray(k, dtype=float)
    k_norm = np.linalg.norm(k)
    
    if k_norm < 1e-14:
        # Weyl节点处Berry曲率发散，返回正则化值
        return np.zeros((3, 3))
    
    Omega = np.zeros((3, 3))
    sign = chirality
    
    for i in range(3):
        for j in range(3):
            # epsilon_{ijk} * k_k
            k_idx = 3 - i - j  # 对于(0,1)->2, (0,2)->1, (1,2)->0
            if i == j or k_idx < 0 or k_idx > 2:
                continue
            # Levi-Civita符号
            eps = 1 if ((i, j, k_idx) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]) else -1
            Omega[i, j] = sign * 0.5 * eps * k[k_idx] / (k_norm ** 3)
    
    return Omega


def berry_phase_1d(ham: WeylHamiltonian, path: np.ndarray,
                    band_index: int = 0) -> float:
    """
    计算沿一维路径的Berry相位
    
    离散化公式（Wilson loop方法）：
        gamma = -Im ln prod_{i=1}^{N-1} <u_i | u_{i+1}>
    
    其中 |u_i> = |u_n(k_i)> 是第i个k点的本征矢。
    
    Parameters
    ----------
    ham : WeylHamiltonian
    path : np.ndarray, shape (N, 3)
        k空间路径点
    band_index : int
    
    Returns
    -------
    phase : float
        Berry相位（以弧度为单位，模2pi）
    """
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path必须是(N,3)数组")
    
    n_points = path.shape[0]
    if n_points < 2:
        return 0.0
    
    # 收集本征矢
    eigenvectors = []
    for i in range(n_points):
        _, v = ham.eigenproblem(path[i])
        vec = v[:, band_index].copy()
        # 规范固定
        if abs(vec[0]) > 1e-14:
            vec *= np.exp(-1.0j * np.angle(vec[0]))
        eigenvectors.append(vec)
    
    # Wilson loop乘积
    prod = 1.0 + 0.0j
    for i in range(n_points - 1):
        overlap = np.vdot(eigenvectors[i], eigenvectors[i + 1])
        prod *= overlap
    
    # 若路径闭合，添加首尾连接
    if np.linalg.norm(path[0] - path[-1]) < 1e-10:
        overlap = np.vdot(eigenvectors[-1], eigenvectors[0])
        prod *= overlap
    
    gamma = -np.angle(prod)
    return gamma


def chern_number_2d_slice(ham: WeylHamiltonian,
                           kx_range: Tuple[float, float],
                           ky_range: Tuple[float, float],
                           kz_fixed: float,
                           grid_size: int = 40,
                           band_index: int = 0) -> float:
    """
    计算固定kz平面上的Chern数
    
    C_n = (1/2*pi) * \int_{BZ} Omega_{xy}(k) d^2k
    
    数值上采用离散化求和：
        C = (1/2*pi) * sum_{ij} Omega_{xy}(k_{ij}) * dkx * dky
    
    Parameters
    ----------
    ham : WeylHamiltonian
    kx_range, ky_range : tuple of float
    kz_fixed : float
    grid_size : int
    band_index : int
    
    Returns
    -------
    chern : float
        Chern数（理论上应为整数）
    """
    kx = np.linspace(kx_range[0], kx_range[1], grid_size)
    ky = np.linspace(ky_range[0], ky_range[1], grid_size)
    dkx = (kx_range[1] - kx_range[0]) / (grid_size - 1) if grid_size > 1 else 0.0
    dky = (ky_range[1] - ky_range[0]) / (grid_size - 1) if grid_size > 1 else 0.0
    
    total = 0.0
    for i in range(grid_size):
        for j in range(grid_size):
            k = np.array([kx[i], ky[j], kz_fixed])
            Omega = berry_curvature_numeric(ham, k, band_index)
            total += Omega[0, 1]  # Omega_xy
    
    chern = total * dkx * dky / (2.0 * np.pi)
    return chern


def weyl_charge_surface_integral(ham: WeylHamiltonian,
                                  center: np.ndarray,
                                  radius: float,
                                  n_theta: int = 20,
                                  n_phi: int = 20,
                                  band_index: int = 0) -> float:
    """
    通过Berry曲率在闭合曲面上的面积分计算Weyl荷
    
    Q = (1/2*pi) \oint_S Omega · dS
    
    参数化球面：
        k = center + radius * (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    
    Parameters
    ----------
    ham : WeylHamiltonian
    center : np.ndarray, shape (3,)
        Weyl节点位置（积分曲面中心）
    radius : float
        积分球面半径
    n_theta, n_phi : int
        球面离散化网格数
    band_index : int
    
    Returns
    -------
    charge : float
        Weyl荷（理论上应为±1）
    """
    center = np.asarray(center, dtype=float)
    
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    d_theta = np.pi / (n_theta - 1) if n_theta > 1 else 0.0
    d_phi = 2.0 * np.pi / (n_phi - 1) if n_phi > 1 else 0.0
    
    total = 0.0
    for i in range(n_theta):
        st = np.sin(theta[i])
        ct = np.cos(theta[i])
        for j in range(n_phi):
            sp = np.sin(phi[j])
            cp = np.cos(phi[j])
            
            # 球面上的点
            r_vec = radius * np.array([st * cp, st * sp, ct])
            k = center + r_vec
            
            # Berry曲率
            Omega = berry_curvature_numeric(ham, k, band_index)
            
            # 面元矢量 dS = r^2 * sin(theta) * (sin(theta)*cos(phi), ...)
            # 即 dS = r_vec * r * sin(theta) * d_theta * d_phi
            dS = r_vec * radius * st * d_theta * d_phi
            
            # Omega · dS
            # Omega是反对称张量，矢量形式：Omega_vec = (Omega_yz, Omega_zx, Omega_xy)
            omega_vec = np.array([Omega[1, 2], Omega[2, 0], Omega[0, 1]])
            total += np.dot(omega_vec, dS)
    
    charge = total / (2.0 * np.pi)
    return charge
