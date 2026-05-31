"""
svd_reaction_coords.py
反应坐标降维与主成分分析模块

整合原项目:
  - 1191_svd_snowfall: 奇异值分解与主成分分析

科学背景:
  分子动力学轨迹数据维度极高 (3N 个坐标)。
  通过奇异值分解 (SVD) 可以提取主要的集体运动模式 (反应坐标):
  
  给定轨迹矩阵 X ∈ R^{M × 3N} (M 个时间步, N 个原子):
    X = U Σ V^T
  
  其中:
    U ∈ R^{M×M}: 左奇异向量 (时间模式)
    Σ ∈ R^{M×3N}: 奇异值对角矩阵
    V ∈ R^{3N×3N}: 右奇异向量 (空间模式/反应坐标)
  
  主成分 (Principal Components):
    PC_k(t) = X(t) · v_k
  
  方差贡献:
    Var_k = σ_k² / Σ_j σ_j²
  
  反应坐标通常对应最大奇异值的方向，
  描述反应路径上的主要构型变化
"""

import numpy as np
from typing import Tuple, List, Optional


class ReactionCoordinateAnalyzer:
    """
    反应坐标分析器
    
    使用 SVD/PCA 从 MD 轨迹中提取反应坐标
    """

    def __init__(self):
        self.U = None
        self.S = None
        self.Vt = None
        self.mean = None
        self.n_frames = None
        self.n_dof = None

    def fit(self, trajectory: np.ndarray):
        """
        对轨迹数据进行 SVD
        
        参数:
          trajectory: (n_frames, n_dof) 或 (n_frames, n_atoms, 3)
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim == 3:
            traj = traj.reshape(traj.shape[0], -1)
        self.n_frames, self.n_dof = traj.shape
        self.mean = np.mean(traj, axis=0)
        X_centered = traj - self.mean
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)

    def variance_explained(self, n_components: Optional[int] = None) -> np.ndarray:
        """
        计算各主成分的方差贡献率
        
        公式:
          Var_k = σ_k² / Σ_j σ_j²
        """
        if self.S is None:
            raise RuntimeError("必须先调用 fit()")
        var = self.S ** 2
        var_ratio = var / np.sum(var)
        if n_components is not None:
            return var_ratio[:n_components]
        return var_ratio

    def principal_components(self, trajectory: np.ndarray,
                             n_components: int = 3) -> np.ndarray:
        """
        计算前 n_components 个主成分投影
        
        PC_k(t) = (X(t) - mean) · v_k
        """
        if self.Vt is None:
            raise RuntimeError("必须先调用 fit()")
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim == 3:
            traj = traj.reshape(traj.shape[0], -1)
        X_centered = traj - self.mean
        return X_centered @ self.Vt[:n_components].T

    def reaction_coordinate(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取第一主成分作为反应坐标
        
        反应坐标 q(t) 定义为最大方差方向上的投影:
          q(t) = PC_1(t)
        """
        pcs = self.principal_components(trajectory, n_components=1)
        return pcs[:, 0]

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        从主成分系数重建构型
        
        X_recon = mean + Σ_k coeff_k * v_k
        """
        if self.Vt is None:
            raise RuntimeError("必须先调用 fit()")
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        return self.mean + coeffs @ self.Vt[:coeffs.shape[1]]

    def free_energy_profile(self, reaction_coord: np.ndarray,
                            temperature_k: float = 500.0,
                            n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算反应坐标上的自由能面
        
        公式:
          F(q) = -k_B T * ln P(q)
        
        其中 P(q) 为反应坐标的一维分布直方图
        """
        from utils import BOLTZMANN_KB
        kb_t = BOLTZMANN_KB * temperature_k

        hist, bin_edges = np.histogram(reaction_coord, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 避免 log(0)
        hist = np.maximum(hist, 1e-300)
        free_energy = -kb_t * np.log(hist)
        free_energy = free_energy - np.min(free_energy)
        return bin_centers, free_energy

    def commutator_analysis(self, trajectory: np.ndarray,
                            state_a_mask: np.ndarray,
                            state_b_mask: np.ndarray) -> np.ndarray:
        """
        计算可交换性函数 (Committor Analysis)
        
        可交换性 p_B(q) 表示处于反应坐标 q 的构型
        最终到达状态 B (而非状态 A) 的概率
        
        公式 (近似):
          p_B(q) = N_B(q) / (N_A(q) + N_B(q))
        
        其中 N_A, N_B 为反应坐标 bin 中属于各自状态的构型数
        """
        rc = self.reaction_coordinate(trajectory)
        n_bins = 50
        hist_a, bins = np.histogram(rc[state_a_mask], bins=n_bins, range=(rc.min(), rc.max()))
        hist_b, _ = np.histogram(rc[state_b_mask], bins=n_bins, range=(rc.min(), rc.max()))

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        p_b = np.zeros(n_bins)
        for i in range(n_bins):
            total = hist_a[i] + hist_b[i]
            if total > 0:
                p_b[i] = hist_b[i] / total
        return bin_centers, p_b

    def collectivity_index(self, n_components: int = 3) -> float:
        """
        计算主成分的集体性指数
        
        公式 (Riccardi et al.):
          κ = (1/N) * exp(-Σ_i p_i ln p_i)
        
        其中 p_i = v_{1,i}² 为第一个右奇异向量的分量权重
        
        κ → 1 表示高度集体运动
        κ → 0 表示局部运动
        """
        if self.Vt is None:
            raise RuntimeError("必须先调用 fit()")
        v1 = self.Vt[0] ** 2
        v1 = v1 / np.sum(v1)
        v1 = np.maximum(v1, 1e-300)
        entropy = -np.sum(v1 * np.log(v1))
        kappa = np.exp(entropy) / len(v1)
        return float(kappa)


def generate_test_trajectory(n_atoms: int = 10, n_frames: int = 200) -> np.ndarray:
    """
    生成测试 MD 轨迹数据
    
    模拟从反应物到产物的构型变化:
      - 前 1/3: 反应物态 (小幅度振动)
      - 中 1/3: 过渡态附近 (大幅度涨落)
      - 后 1/3: 产物态 (小幅度振动)
    """
    rng = np.random.default_rng(42)
    traj = np.zeros((n_frames, n_atoms, 3))

    # 平衡位置
    r_eq = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        angle = 2.0 * np.pi * i / n_atoms
        r_eq[i] = [np.cos(angle), np.sin(angle), 0.0]

    for t in range(n_frames):
        frac = t / n_frames
        # 反应坐标: 从反应物 (0) 到产物 (1)
        if frac < 0.33:
            state = 0.0
            amp = 0.05
        elif frac < 0.67:
            state = (frac - 0.33) / 0.34
            amp = 0.15
        else:
            state = 1.0
            amp = 0.05

        # 产物态几何略有不同
        r_prod = r_eq.copy()
        r_prod[:, 0] *= 1.2
        r_prod[:, 1] *= 0.8

        r_t = (1 - state) * r_eq + state * r_prod
        noise = rng.normal(0.0, amp, size=(n_atoms, 3))
        traj[t] = r_t + noise

    return traj
