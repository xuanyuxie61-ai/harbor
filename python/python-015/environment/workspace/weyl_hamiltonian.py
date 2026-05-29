"""
weyl_hamiltonian.py
Weyl半金属哈密顿量建模与本征问题求解

凝聚态物理：拓扑半金属中的Weyl节点

核心物理模型：
在Weyl节点附近，低能有效哈密顿量可写为Weil哈密顿量：
    H(k) = hbar * v_F * (k_x * sigma_x + k_y * sigma_y + k_z * sigma_z) + M(k) * I

其中 sigma_i 为Pauli矩阵，v_F为Fermi速度，M(k)为质量项（可破缺对称性）。

本征值：E_{\pm}(k) = M(k) \pm hbar * v_F * |k|

对于双Weyl节点模型（如TaAs类材料），在紧束缚近似下：
    H(k) = d_0(k) * I + sum_{i=1}^3 d_i(k) * sigma_i

其中：
    d_0(k) = m_0 + m_1*(cos(k_x) + cos(k_y)) + m_2*cos(k_z)
    d_x(k) = A * sin(k_x)
    d_y(k) = A * sin(k_y)
    d_z(k) = 2*B_1*(2 - cos(k_x) - cos(k_y))*sin(k_z) + 2*B_2*sin(2*k_z)

Berry联络：A_n(k) = i <u_n(k)| \nabla_k |u_n(k)>
Berry曲率：Omega_n(k) = \nabla_k x A_n(k)
"""

import numpy as np
from typing import Tuple, List


class WeylHamiltonian:
    """
    Weyl半金属哈密顿量类
    
    支持两种模型：
    1. 线性Weyl模型：H = hbar*v_F * k·sigma（单Weyl节点附近）
    2. 紧束缚双Weyl模型（如TaAs原型）
    """
    
    def __init__(self, model_type: str = "linear", hbar: float = 1.0, v_f: float = 1.0):
        """
        Parameters
        ----------
        model_type : str
            "linear" 或 "tight_binding"
        hbar : float
            约化普朗克常数（自然单位制下可设为1）
        v_f : float
            Fermi速度（eV·Å量级，自然单位制下可设为1）
        """
        if model_type not in ("linear", "tight_binding"):
            raise ValueError(f"不支持的模型类型: {model_type}")
        self.model_type = model_type
        self.hbar = hbar
        self.v_f = v_f
        
        # Pauli矩阵
        self.sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self.sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        self.sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        self.I2 = np.eye(2, dtype=complex)
        
        # 紧束缚模型参数（模拟TaAs类材料的典型值）
        if model_type == "tight_binding":
            self.m0 = 0.5
            self.m1 = -0.3
            self.m2 = -0.2
            self.A = 0.4
            self.B1 = 0.3
            self.B2 = 0.1
    
    def build_hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        构建给定k点的2x2哈密顿量矩阵
        
        Parameters
        ----------
        k : np.ndarray, shape (3,) or (N, 3)
            动量空间坐标（单位：1/Å或倒格矢单位）
        
        Returns
        -------
        H : np.ndarray
            shape (2,2) 若k为(3,)，shape (N,2,2) 若k为(N,3)
        """
        k = np.atleast_2d(k)
        n_points = k.shape[0]
        
        if self.model_type == "linear":
            # H = hbar * v_F * (k_x sigma_x + k_y sigma_y + k_z sigma_z)
            H = np.zeros((n_points, 2, 2), dtype=complex)
            for i in range(n_points):
                H[i] = self.hbar * self.v_f * (
                    k[i, 0] * self.sigma_x +
                    k[i, 1] * self.sigma_y +
                    k[i, 2] * self.sigma_z
                )
        else:
            # 紧束缚模型
            # TODO Hole_1: 实现紧束缚Weyl模型的哈密顿量构建
            # H = d0*I + dx*sigma_x + dy*sigma_y + dz*sigma_z
            # 其中:
            #   d0 = m0 + m1*(cos(kx) + cos(ky)) + m2*cos(kz)
            #   dx = A * sin(kx)
            #   dy = A * sin(ky)
            #   dz = 2*B1*(2 - cos(kx) - cos(ky))*sin(kz) + 2*B2*sin(2*kz)
            raise NotImplementedError("Hole_1: 紧束缚模型哈密顿量构建待实现")
        
        if n_points == 1:
            return H[0]
        return H
    
    def eigenproblem(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解哈密顿量的本征值和本征矢
        
        Returns
        -------
        energies : np.ndarray, shape (N, 2)
            两个能带的能量本征值，按升序排列
        eigenvectors : np.ndarray, shape (N, 2, 2)
            对应的本征矢（列矢量）
        """
        H = self.build_hamiltonian(k)
        
        if H.ndim == 2:
            # 单点
            energies, eigenvectors = np.linalg.eigh(H)
            return energies, eigenvectors
        
        n_points = H.shape[0]
        energies = np.zeros((n_points, 2))
        eigenvectors = np.zeros((n_points, 2, 2), dtype=complex)
        
        for i in range(n_points):
            e, v = np.linalg.eigh(H[i])
            energies[i] = e
            eigenvectors[i] = v
        
        return energies, eigenvectors
    
    def d_vectors(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取紧束缚模型中的d矢量分量 (d0, d1, d2, d3)
        
        对于H = d0*I + d·sigma，有：
            d0 = Tr(H)/2
            d_i = Tr(H * sigma_i)/2
        
        Returns
        -------
        d0 : np.ndarray, shape (N,)
        d_vec : np.ndarray, shape (N, 3)
        """
        k = np.atleast_2d(k)
        n_points = k.shape[0]
        
        if self.model_type == "linear":
            d0 = np.zeros(n_points)
            d_vec = self.hbar * self.v_f * k
        else:
            H = self.build_hamiltonian(k)
            d0 = np.zeros(n_points)
            d_vec = np.zeros((n_points, 3))
            for i in range(n_points):
                d0[i] = 0.5 * np.trace(H[i]).real
                d_vec[i, 0] = 0.5 * np.trace(H[i] @ self.sigma_x).real
                d_vec[i, 1] = 0.5 * np.trace(H[i] @ self.sigma_y).real
                d_vec[i, 2] = 0.5 * np.trace(H[i] @ self.sigma_z).real
        
        return d0, d_vec
    
    def monomial_expansion_value(self, k: np.ndarray, exponents: np.ndarray,
                                  coefficients: np.ndarray) -> np.ndarray:
        """
        计算哈密顿量参数的多项式展开值
        
        基于种子项目331_ellipse_monte_carlo中的monomial_value思想，
        用于计算能带的高阶修正项：
            E(k) ≈ sum_{alpha} c_alpha * k_x^{alpha_x} * k_y^{alpha_y} * k_z^{alpha_z}
        
        Parameters
        ----------
        k : np.ndarray, shape (N, 3)
        exponents : np.ndarray, shape (M, 3)
            每项的多项式指数
        coefficients : np.ndarray, shape (M,)
            每项的系数
        
        Returns
        -------
        values : np.ndarray, shape (N,)
        """
        k = np.atleast_2d(k)
        n_points = k.shape[0]
        n_terms = exponents.shape[0]
        
        if coefficients.shape[0] != n_terms:
            raise ValueError("系数数量与指数项数量不匹配")
        
        values = np.zeros(n_points)
        for j in range(n_points):
            val = 0.0
            for i in range(n_terms):
                # product_{d=1}^3 k_d^{e_d}
                monomial = 1.0
                for dim in range(3):
                    e = int(exponents[i, dim])
                    if e > 0:
                        monomial *= k[j, dim] ** e
                val += coefficients[i] * monomial
            values[j] = val
        
        return values
    
    def weyl_node_position_linear(self) -> np.ndarray:
        """
        线性模型中Weyl节点位于k=0
        """
        return np.zeros(3)
    
    def find_weyl_nodes_tight_binding(self, grid_size: int = 64,
                                       bz_bounds: np.ndarray = None) -> np.ndarray:
        """
        在紧束缚模型中数值搜索Weyl节点位置
        
        Weyl节点满足：E_+(k) = E_-(k) 即 d_vec(k) = 0
        
        使用三维网格搜索，然后在零点附近的精细搜索。
        
        Parameters
        ----------
        grid_size : int
            每个维度的粗网格点数
        bz_bounds : np.ndarray
            布里渊区边界，shape (3, 2)，默认[-pi, pi]^3
        
        Returns
        -------
        nodes : np.ndarray, shape (N, 3)
            Weyl节点的位置列表
        """
        if bz_bounds is None:
            bz_bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
        
        # 粗网格搜索
        kx = np.linspace(bz_bounds[0, 0], bz_bounds[0, 1], grid_size)
        ky = np.linspace(bz_bounds[1, 0], bz_bounds[1, 1], grid_size)
        kz = np.linspace(bz_bounds[2, 0], bz_bounds[2, 1], grid_size)
        
        d0, d_vec = self.d_vectors(
            np.array([[x, y, z] for x in kx for y in ky for z in kz])
        )
        d_norm = np.linalg.norm(d_vec, axis=1)
        
        # 寻找d_norm的极小值点
        threshold = 0.1 * np.max(d_norm)
        candidate_idx = np.where(d_norm < threshold)[0]
        
        # 通过局部聚类找到不同的Weyl节点
        candidates = np.array([[x, y, z] for x in kx for y in ky for z in kz])[candidate_idx]
        
        if len(candidates) == 0:
            return np.zeros((0, 3))
        
        # 简单的聚类：合并距离小于网格间距的候选点
        nodes = []
        used = set()
        dk = np.array([
            (bz_bounds[0, 1] - bz_bounds[0, 0]) / grid_size,
            (bz_bounds[1, 1] - bz_bounds[1, 0]) / grid_size,
            (bz_bounds[2, 1] - bz_bounds[2, 0]) / grid_size
        ])
        merge_dist = 2.0 * np.linalg.norm(dk)
        
        for i, c in enumerate(candidates):
            if i in used:
                continue
            cluster = [c]
            used.add(i)
            for j in range(i + 1, len(candidates)):
                if j not in used and np.linalg.norm(c - candidates[j]) < merge_dist:
                    cluster.append(candidates[j])
                    used.add(j)
            # 取聚类中心作为节点位置
            nodes.append(np.mean(cluster, axis=0))
        
        return np.array(nodes)


def band_gap(energies: np.ndarray) -> np.ndarray:
    """
    计算能带间隙：Delta(k) = E_+(k) - E_-(k)
    
    Parameters
    ----------
    energies : np.ndarray, shape (N, 2)
    
    Returns
    -------
    gap : np.ndarray, shape (N,)
    """
    if energies.ndim != 2 or energies.shape[1] != 2:
        raise ValueError("energies必须是(N,2)数组")
    return energies[:, 1] - energies[:, 0]


def velocity_operator(ham: WeylHamiltonian, k: np.ndarray,
                       delta: float = 1e-6) -> np.ndarray:
    """
    数值计算速度算符 v = (1/hbar) * dH/dk
    
    使用中心差分：
        v_i = (1/hbar) * [H(k + delta*e_i) - H(k - delta*e_i)] / (2*delta)
    
    Parameters
    ----------
    ham : WeylHamiltonian
    k : np.ndarray, shape (3,)
    delta : float
        差分步长
    
    Returns
    -------
    v : np.ndarray, shape (3, 2, 2)
        三个方向的速度算符矩阵
    """
    v = np.zeros((3, 2, 2), dtype=complex)
    for i in range(3):
        kp = k.copy()
        km = k.copy()
        kp[i] += delta
        km[i] -= delta
        v[i] = (ham.build_hamiltonian(kp) - ham.build_hamiltonian(km)) / (2.0 * delta * ham.hbar)
    return v
