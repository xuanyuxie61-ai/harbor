"""
svd_analysis.py
事件涨落的主成分分析与降维

基于种子项目:
- 1186_svd_faces: SVD分解、主成分提取、数据投影

物理应用:
1. 事件-by-事件能量密度涨落的SVD/PCA分析
2. 流谐波涨落的模式分解
3. 噪声过滤与信号重建
4. 不同碰撞中心度类别的判别分析

数学模型:
给定N个事件的横向能量密度分布矩阵 A (M×N，M为空间格点数):
A = U Σ V^T

其中U的列为主成分空间模式，V的行为主成分时间/事件模式。
"""

import numpy as np
from typing import Tuple, List, Optional


class EventSVDAnalyzer:
    """
    基于SVD的重离子碰撞事件涨落分析器。
    """

    def __init__(self, n_components: int = 10):
        """
        初始化SVD分析器。

        Parameters
        ----------
        n_components : int
            保留的主成分数量
        """
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.mean_profile = None

    def fit(self, event_data: np.ndarray) -> 'EventSVDAnalyzer':
        """
        对事件数据进行SVD分解。

        Parameters
        ----------
        event_data : np.ndarray
            事件数据矩阵 (n_pixels, n_events)

        Returns
        -------
        self
        """
        data = np.asarray(event_data, dtype=float)
        if data.ndim != 2:
            raise ValueError("event_data必须是2维矩阵")

        # 减去平均事件轮廓
        self.mean_profile = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean_profile

        # SVD分解
        self.U, self.S, self.Vt = np.linalg.svd(centered, full_matrices=False)

        # 限制主成分数
        n_comp = min(self.n_components, len(self.S))
        self.U = self.U[:, :n_comp]
        self.S = self.S[:n_comp]
        self.Vt = self.Vt[:n_comp, :]

        return self

    def explained_variance_ratio(self) -> np.ndarray:
        """
        计算各主成分解释的方差比例。

        r_k = σ_k² / Σ_j σ_j²

        Returns
        -------
        np.ndarray
            方差比例数组
        """
        if self.S is None:
            return np.array([])
        total = np.sum(self.S ** 2)
        if total < 1e-15:
            return np.zeros_like(self.S)
        return (self.S ** 2) / total

    def cumulative_variance(self) -> np.ndarray:
        """
        计算累积解释方差。

        Returns
        -------
        np.ndarray
            累积方差比例
        """
        ratios = self.explained_variance_ratio()
        return np.cumsum(ratios)

    def reconstruct(self, n_modes: Optional[int] = None) -> np.ndarray:
        """
        使用前n_modes个主成分重建数据。

        A_recon = mean + U[:, :n] · diag(S[:n]) · Vt[:n, :]

        Parameters
        ----------
        n_modes : int, optional
            使用的主成分数，默认全部

        Returns
        -------
        np.ndarray
            重建的数据矩阵
        """
        if self.U is None or self.S is None or self.Vt is None:
            raise ValueError("请先调用fit()")
        n = n_modes if n_modes is not None else len(self.S)
        n = min(n, len(self.S))

        recon = self.U[:, :n] @ np.diag(self.S[:n]) @ self.Vt[:n, :]
        recon = recon + self.mean_profile
        return recon

    def project_event(self, event: np.ndarray) -> np.ndarray:
        """
        将单个事件投影到主成分空间。

        coefficients = U^T · (event - mean)

        Parameters
        ----------
        event : np.ndarray
            单个事件数据 (n_pixels,)

        Returns
        -------
        np.ndarray
            主成分系数
        """
        if self.U is None or self.mean_profile is None:
            raise ValueError("请先调用fit()")
        event_centered = event - self.mean_profile.flatten()
        coeffs = self.U.T @ event_centered
        return coeffs

    def event_distance(self, event1: np.ndarray,
                       event2: np.ndarray) -> float:
        """
        计算两个事件在主成分空间的欧氏距离。

        d = ||U^T (e1 - e2)||

        Parameters
        ----------
        event1, event2 : np.ndarray
            事件数据

        Returns
        -------
        float
            距离
        """
        c1 = self.project_event(event1)
        c2 = self.project_event(event2)
        return float(np.linalg.norm(c1 - c2))

    def fluctuation_modes(self) -> np.ndarray:
        """
        获取涨落空间模式 (U矩阵的列)。

        Returns
        -------
        np.ndarray
            空间模式 (n_pixels, n_components)
        """
        if self.U is None:
            raise ValueError("请先调用fit()")
        return self.U

    def event_weights(self) -> np.ndarray:
        """
        获取事件在主成分上的权重 (Vt的行)。

        Returns
        -------
        np.ndarray
            事件权重 (n_components, n_events)
        """
        if self.Vt is None:
            raise ValueError("请先调用fit()")
        return self.Vt


class FlowHarmonicDecomposition:
    """
    流谐波的SVD辅助分解。
    """

    @staticmethod
    def flow_vector(qn_x: float, qn_y: float) -> Tuple[float, float]:
        """
        计算流矢量 Q_n = (Q_n^x, Q_n^y)。

        |Q_n| = √(Q_n^x² + Q_n^y²)
        Ψ_n = atan2(Q_n^y, Q_n^x) / n

        Parameters
        ----------
        qn_x, qn_y : float
            流矢量分量

        Returns
        -------
        Tuple[float, float]
            (|Q_n|, Ψ_n)
        """
        magnitude = np.sqrt(qn_x ** 2 + qn_y ** 2)
        psi_n = np.arctan2(qn_y, qn_x)
        return magnitude, psi_n

    @staticmethod
    def eccentricity_from_flow(v2: float, 
                                 response_coeff: float = 0.18) -> float:
        """
        从椭圆流反推初始偏心距 (线性响应近似)。

        ε₂ ≈ v₂ / κ

        Parameters
        ----------
        v2 : float
            椭圆流系数
        response_coeff : float
            响应系数 κ

        Returns
        -------
        float
            偏心距估计
        """
        if response_coeff < 1e-15:
            return 0.0
        return v2 / response_coeff

    @staticmethod
    def cumulant_v2(particles_phi: np.ndarray) -> float:
        """
        使用二阶累积量计算 v₂{2}。

        v₂{2}² = ⟨cos[2(φ₁ - φ₂)]⟩

        Parameters
        ----------
        particles_phi : np.ndarray
            粒子方位角数组 [rad]

        Returns
        -------
        float
            v₂{2}
        """
        n = len(particles_phi)
        if n < 2:
            return 0.0
        cos_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                cos_sum += np.cos(2.0 * (particles_phi[i] - particles_phi[j]))
        denom = n * (n - 1) / 2.0
        if denom < 1e-15:
            return 0.0
        v2_sq = cos_sum / denom
        return float(np.sqrt(max(v2_sq, 0.0)))

    @staticmethod
    def cumulant_v4(particles_phi: np.ndarray) -> float:
        """
        使用四阶累积量计算 v₄{4}。

        v₄{4}⁴ = 2⟨cos[4(φ₁ - φ₂)]⟩² - ⟨cos[4(φ₁ - φ₂ + φ₃ - φ₄)]⟩

        Parameters
        ----------
        particles_phi : np.ndarray
            粒子方位角数组

        Returns
        -------
        float
            v₄{4}
        """
        n = len(particles_phi)
        if n < 4:
            return 0.0
        # 二粒子关联
        c2 = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c2 += np.cos(4.0 * (particles_phi[i] - particles_phi[j]))
        c2 /= (n * (n - 1) / 2.0)

        # 四粒子关联 (简化计算)
        c4 = 0.0
        count = 0
        for i in range(min(n, 20)):
            for j in range(i + 1, min(n, 20)):
                for k in range(j + 1, min(n, 20)):
                    for l in range(k + 1, min(n, 20)):
                        c4 += np.cos(4.0 * (particles_phi[i] - particles_phi[j] +
                                            particles_phi[k] - particles_phi[l]))
                        count += 1
        if count > 0:
            c4 /= count
        else:
            c4 = 0.0

        v4_4 = 2.0 * c2 ** 2 - c4
        return float(np.sign(v4_4) * (abs(v4_4) ** 0.25))

    def event_plane_resolution(self, n_subevents: int = 3) -> float:
        """
        事件平面分辨率估计 (简化模型)。

        R_n ≈ √(π/2) · χ_n · exp(-χ_n²/2) · [I₀(χ_n²/2) + I₁(χ_n²/2)]

        Returns
        -------
        float
            分辨率 (简化返回)
        """
        # 简化: 假设χ ≈ 1
        chi = 1.0
        from scipy.special import ive, iv
        # 使用修正贝塞尔函数
        try:
            r = np.sqrt(np.pi / 2.0) * chi * np.exp(-chi ** 2 / 2.0) * (
                iv(0, chi ** 2 / 2.0) + iv(1, chi ** 2 / 2.0)
            )
        except Exception:
            r = 0.7  # 默认值
        return float(r)
