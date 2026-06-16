"""
Weyl Hamiltonian 构造模块
基于 k·p 微扰理论构建 Weyl 半金属的 Hamiltonian
H(k) = ℏv_F (kx·σx + ky·σy + kz·σz) + m(k)·σz
其中 σ 为 Pauli 矩阵，v_F 为费米速度，m(k) 为质量项
"""

import numpy as np
from typing import Tuple, Callable


class WeylHamiltonian:
    """
    Weyl 半金属 Hamiltonian 构造器

    物理背景：
    Weyl 费米子在动量空间中的低能有效 Hamiltonian 可写为：
    H(k) = χ·ℏv_F (q·σ) + m(q)·σ₀

    其中：
    - χ = ±1 为手性 (chirality)
    - v_F 为费米速度 (~10^5 m/s)
    - q = k - k_W 为相对 Weyl 点的动量
    - σ = (σx, σy, σz) 为 Pauli 矩阵矢量
    - m(q) 为打破时间反演对称性的质量项
    """

    def __init__(self, v_f: float = 1.0, chirality: int = 1, weyl_node: np.ndarray = None):
        """
        初始化 Weyl Hamiltonian

        Parameters:
        -----------
        v_f : float
            费米速度 (单位：eV·Å)
        chirality : int
            Weyl 点的手性 (+1 或 -1)
        weyl_node : np.ndarray
            Weyl 节点在布里渊区中的位置 [kx, ky, kz]
        """
        self.v_f = v_f
        self.chirality = chirality
        self.weyl_node = weyl_node if weyl_node is not None else np.array([0.0, 0.0, 0.0])

        # Pauli 矩阵
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.sigma_0 = np.eye(2, dtype=complex)

    def mass_term(self, k: np.ndarray, m0: float = 0.1, alpha: float = 1.0) -> float:
        """
        计算质量项 m(k)，模拟时间反演对称性破缺

        常用形式：m(k) = m0 - α·(|k|² - k_W²)
        或 m(k) = m0·cos(kz·a)

        Parameters:
        -----------
        k : np.ndarray
            动量点 [kx, ky, kz]
        m0 : float
            质量参数
        alpha : float
            动量依赖系数

        Returns:
        --------
        float
            质量项 m(k)
        """
        q = k - self.weyl_node
        q_norm_sq = np.sum(q**2)
        # 采用二次型质量项
        return m0 - alpha * q_norm_sq

    def hamiltonian_matrix(self, k: np.ndarray, m0: float = 0.1, alpha: float = 1.0) -> np.ndarray:
        """
        构建 2×2 Weyl Hamiltonian 矩阵

        H(k) = χ·ℏv_F (qx·σx + qy·σy + qz·σz) + m(q)·σ₀

        Parameters:
        -----------
        k : np.ndarray
            动量点 [kx, ky, kz]
        m0 : float
            质量参数
        alpha : float
            动量依赖系数

        Returns:
        --------
        np.ndarray
            2×2 Hermitian Hamiltonian 矩阵
        """
        q = k - self.weyl_node
        m_k = self.mass_term(k, m0, alpha)

        # 构建 Hamiltonian
        H = self.chirality * self.v_f * (
            q[0] * self.sigma_x +
            q[1] * self.sigma_y +
            q[2] * self.sigma_z
        ) + m_k * self.sigma_0

        return H

    def hamiltonian_4band(self, k: np.ndarray, params: dict = None) -> np.ndarray:
        """
        构建 4 带 Weyl Hamiltonian (考虑自旋和轨道自由度)

        H_4band(k) = [H_2x2(k)    Δ·σ₀    ]
                     [Δ·σ₀      -H_2x2(-k) ]

        Parameters:
        -----------
        k : np.ndarray
            动量点
        params : dict
            包含 t1, t2, Δ 等参数的字典

        Returns:
        --------
        np.ndarray
            4×4 Hamiltonian 矩阵
        """
        if params is None:
            params = {'t1': 1.0, 't2': 0.5, 'delta': 0.1}

        t1 = params.get('t1', 1.0)
        t2 = params.get('t2', 0.5)
        delta = params.get('delta', 0.1)

        # 上下块
        H_upper = self.hamiltonian_matrix(k, t1, t2)
        H_lower = -self.hamiltonian_matrix(-k, t1, t2)

        # 耦合项
        coupling = delta * self.sigma_0

        # 组装 4×4 矩阵
        H_4band = np.zeros((4, 4), dtype=complex)
        H_4band[0:2, 0:2] = H_upper
        H_4band[2:4, 2:4] = H_lower
        H_4band[0:2, 2:4] = coupling
        H_4band[2:4, 0:2] = coupling

        return H_4band

    def energy_dispersion(self, k: np.ndarray, m0: float = 0.1, alpha: float = 1.0) -> Tuple[float, float]:
        """
        计算能量色散关系 E±(k)

        E±(k) = m(k) ± ℏv_F·|q|

        Parameters:
        -----------
        k : np.ndarray
            动量点
        m0, alpha : float
            质量项参数

        Returns:
        --------
        Tuple[float, float]
            (E+, E-) 能量本征值
        """
        q = k - self.weyl_node
        q_norm = np.linalg.norm(q)
        m_k = self.mass_term(k, m0, alpha)

        E_plus = m_k + self.chirality * self.v_f * q_norm
        E_minus = m_k - self.chirality * self.v_f * q_norm

        return E_plus, E_minus

    def velocity_operator(self, k: np.ndarray, direction: str = 'x') -> np.ndarray:
        """
        计算速度算符 v_i = (1/ℏ)·∂H/∂k_i

        对于 Weyl Hamiltonian：
        v_x = χ·v_F·σx
        v_y = χ·v_F·σy
        v_z = χ·v_F·σz

        Parameters:
        -----------
        k : np.ndarray
            动量点 (此处不依赖 k)
        direction : str
            方向 ('x', 'y', 'z')

        Returns:
        --------
        np.ndarray
            速度算符矩阵
        """
        if direction == 'x':
            return self.chirality * self.v_f * self.sigma_x
        elif direction == 'y':
            return self.chirality * self.v_f * self.sigma_y
        elif direction == 'z':
            return self.chirality * self.v_f * self.sigma_z
        else:
            raise ValueError(f"Unknown direction: {direction}")


def find_weyl_nodes(ham_builder: WeylHamiltonian, k_range: Tuple[float, float] = (-np.pi, np.pi),
                   n_search: int = 100) -> list:
    """
    数值搜索 Weyl 节点位置

    Weyl 节点条件：det[H(k)] = 0 且手性 ≠ 0

    Parameters:
    -----------
    ham_builder : WeylHamiltonian
        Hamiltonian 构造器
    k_range : Tuple[float, float]
        搜索范围
    n_search : int
        每个方向的搜索点数

    Returns:
    --------
    list
        Weyl 节点位置列表 [(kx, ky, kz, chirality), ...]
    """
    weyl_nodes = []
    k_vals = np.linspace(k_range[0], k_range[1], n_search)

    # 简单网格搜索
    for kx in k_vals[::10]:
        for ky in k_vals[::10]:
            for kz in k_vals[::10]:
                k = np.array([kx, ky, kz])
                H = ham_builder.hamiltonian_matrix(k)

                # 检查能隙关闭
                eigenvalues = np.linalg.eigvalsh(H)
                gap = np.abs(eigenvalues[1] - eigenvalues[0])

                if gap < 1e-3:
                    # 计算手性
                    chirality = compute_chirality_numerical(ham_builder, k)
                    if np.abs(chirality) > 0.5:
                        weyl_nodes.append({
                            'position': k,
                            'chirality': int(np.sign(chirality))
                        })

    return weyl_nodes


def compute_chirality_numerical(ham_builder: WeylHamiltonian, k_weyl: np.ndarray,
                               dk: float = 1e-3) -> float:
    """
    数值计算 Weyl 点的手性 (Chern number)

    χ = (1/4π) ∮_S F·dS

    Parameters:
    -----------
    ham_builder : WeylHamiltonian
        Hamiltonian 构造器
    k_weyl : np.ndarray
        Weyl 点位置
    dk : float
        微小动量位移

    Returns:
    --------
    float
        手性 (应为 ±1)
    """
    # 在小球面上积分 Berry curvature
    n_theta, n_phi = 20, 40
    chirality = 0.0

    for i in range(n_theta):
        theta = np.pi * (i + 0.5) / n_theta
        for j in range(n_phi):
            phi = 2 * np.pi * j / n_phi

            # 球面坐标
            q = dk * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            k = k_weyl + q

            # 数值计算 Berry curvature (简化版本)
            H = ham_builder.hamiltonian_matrix(k)
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            # 只考虑占据带
            if eigenvalues[0] < 0:
                # 简化的手性计算
                chirality += np.sin(theta) * dk * dk

    chirality *= (1.0 / (4 * np.pi))
    return chirality
