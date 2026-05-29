"""
svd_analysis.py — 地磁场主导空间模态提取与降维分析模块

原项目映射: 1192_svd_sphere — SVD分解与球面采样分析

改造思路:
  将MATLAB的svd_sphere()改写为Python，用于从地核发电机模拟的时间序列中
  提取主导空间模态。通过SVD(奇异值分解)或PCA，将高维时空场数据
  B(r,θ,t) 分解为空间模态与时间系数的乘积，揭示发电机的主导动力学结构。

科学背景:
  地磁场的时空演化可表示为:
    B(r,θ,t) = Σ_k σ_k u_k(r,θ) v_k(t)
  其中 u_k 为空间本征模态(empirical orthogonal functions, EOFs)，
  v_k 为对应的时间主成分(principal components)。

  第1模态通常对应于轴向偶极子场，第2模态对应于四极子或非偶极子成分。
  当第2模态的能量占比 σ₂²/Σσ² 超过阈值时，系统可能发生极性反转。

  在球面上，SVD分析也可用于研究地磁场在不同球壳层上的各向异性:
    A = U Σ V^T, 其中 A_{ij} = B(r_i, θ_j)
    奇异值 σ_k 反映了第k个空间尺度的能量占比。
"""

import numpy as np
from typing import Tuple, List


class SVDDynamoAnalysis:
    """
    基于SVD的地磁场模态分析器。
    """

    def __init__(self, nr: int, ntheta: int):
        """
        初始化分析器。

        参数:
            nr: 径向网格数
            ntheta: 极角网格数
        """
        self.nr = nr
        self.ntheta = ntheta

    def decompose_field(
        self,
        field_2d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对二维场矩阵执行SVD分解。

        参数:
            field_2d: (nr, ntheta) 轴对称场快照
        返回:
            U: (nr, k) 左奇异向量 (径向模态)
            S: (k,) 奇异值
            Vt: (k, ntheta) 右奇异向量转置 (角度模态)
        """
        if field_2d.shape != (self.nr, self.ntheta):
            raise ValueError(f"field_2d shape {field_2d.shape} does not match ({self.nr}, {self.ntheta})")

        U, S, Vt = np.linalg.svd(field_2d, full_matrices=False)
        return U, S, Vt

    def analyze_time_series(
        self,
        time_series: List[np.ndarray],
    ) -> dict:
        """
        对时间序列的场快照进行联合SVD分析 (snapshot POD方法)。

        构造快照矩阵 X = [B_1, B_2, ..., B_N]，其中每列是一个展平后的场快照。
        对 X 执行SVD: X = U Σ V^T

        参数:
            time_series: 场快照列表，每个元素为 (nr, ntheta) 数组
        返回:
            分析结果字典，包含奇异值、模态、能量占比等
        """
        n_snapshots = len(time_series)
        if n_snapshots == 0:
            return {"error": "empty time series"}

        n_points = self.nr * self.ntheta
        X = np.zeros((n_points, n_snapshots))
        for i, snapshot in enumerate(time_series):
            X[:, i] = snapshot.reshape(-1)

        # 中心化处理 (移除时间平均)
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_centered = X - X_mean

        # 经济型SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 能量分析
        total_energy = np.sum(S ** 2)
        energy_ratio = (S ** 2) / total_energy if total_energy > 1e-30 else np.zeros_like(S)
        cumulative_energy = np.cumsum(energy_ratio)

        # 识别主导模态
        rank_dominant = int(np.searchsorted(cumulative_energy, 0.95)) + 1

        return {
            "singular_values": S,
            "energy_ratio": energy_ratio,
            "cumulative_energy": cumulative_energy,
            "rank_95": rank_dominant,
            "spatial_modes": U,  # (n_points, n_snapshots)
            "temporal_coeffs": Vt,  # (n_snapshots, n_snapshots)
            "mean_field": X_mean.reshape(self.nr, self.ntheta),
        }

    def dipole_quadrupole_analysis(
        self,
        br_field: np.ndarray,
        r_grid: np.ndarray,
        theta_grid: np.ndarray,
    ) -> dict:
        """
        分析径向磁场 B_r 的偶极子与四极子成分。

        在球面上，将 B_r(r_o, θ) 展开为Legendre级数:
          B_r(θ) = Σ_l b_l P_l(cosθ)
        其中 b_l = (2l+1)/2 ∫_{-1}^1 B_r(θ) P_l(x) dx

        偶极子强度: D = |b_1| / sqrt(Σ b_l²)
        四极子强度: Q = |b_2| / sqrt(Σ b_l²)

        参数:
            br_field: (nr, ntheta) 径向磁场分量
            r_grid: 径向坐标
            theta_grid: 极角坐标
        返回:
            多极矩分析字典
        """
        # 取CMB处的径向场 (最外层)
        br_cmb = br_field[-1, :]
        x = np.cos(theta_grid)

        # 计算 Legendre 展开系数 (数值积分)
        max_l = 8
        coeffs = []
        for l in range(max_l + 1):
            # P_l(x) 在 Gauss-Legendre 点上
            pl = np.polynomial.legendre.legvander(x, max_l)[:, l]
            # 梯形法则积分 (θ均匀网格)
            integrand = br_cmb * pl * np.sin(theta_grid)
            coeff = (2.0 * l + 1.0) / 2.0 * np.trapz(integrand, x)
            coeffs.append(coeff)

        coeffs = np.array(coeffs)
        norm = np.linalg.norm(coeffs)
        if norm < 1e-30:
            norm = 1.0

        dipole_ratio = abs(coeffs[1]) / norm if len(coeffs) > 1 else 0.0
        quadrupole_ratio = abs(coeffs[2]) / norm if len(coeffs) > 2 else 0.0
        octupole_ratio = abs(coeffs[3]) / norm if len(coeffs) > 3 else 0.0

        return {
            "legendre_coeffs": coeffs,
            "dipole_ratio": dipole_ratio,
            "quadrupole_ratio": quadrupole_ratio,
            "octupole_ratio": octupole_ratio,
            "dipole_tilt": np.arctan2(coeffs[1].imag if np.iscomplexobj(coeffs) else 0.0, coeffs[1].real if np.iscomplexobj(coeffs) else coeffs[1]),
        }

    def field_anisotropy_tensor(
        self,
        field_r: np.ndarray,
        field_theta: np.ndarray,
        field_phi: np.ndarray,
    ) -> np.ndarray:
        """
        计算磁场的不变性(anisotropy)张量。

        定义二阶矩张量:
          M_{ij} = ⟨B_i B_j⟩ / ⟨B²⟩
        其中 ⟨·⟩ 表示在计算域上的体积平均。

        该张量的特征值分析揭示磁场的各向异性:
          λ₁ ≥ λ₂ ≥ λ₃,  λ₁ + λ₂ + λ₃ = 1
          若 λ₁ ≈ 1, λ₂ ≈ λ₃ ≈ 0: 强各向异性 (主导单向场)
          若 λ₁ ≈ λ₂ ≈ λ₃ ≈ 1/3: 近似各向同性
        """
        b2 = field_r ** 2 + field_theta ** 2 + field_phi ** 2
        b2_avg = np.mean(b2)
        if b2_avg < 1e-30:
            return np.eye(3) / 3.0

        M = np.zeros((3, 3))
        components = [field_r, field_theta, field_phi]
        for i in range(3):
            for j in range(3):
                M[i, j] = np.mean(components[i] * components[j]) / b2_avg

        # 对称化
        M = 0.5 * (M + M.T)
        eigvals = np.linalg.eigvalsh(M)
        return eigvals[::-1]  # 降序

    def snapshot_pod_reconstruction(
        self,
        time_series: List[np.ndarray],
        rank: int,
    ) -> Tuple[List[np.ndarray], float]:
        """
        使用低秩POD重构时间序列。

        参数:
            time_series: 原始场快照列表
            rank: 保留的模态数
        返回:
            reconstructed: 重构后的场快照列表
            relative_error: 相对重构误差
        """
        result = self.analyze_time_series(time_series)
        U = result["spatial_modes"]
        S = result["singular_values"]
        Vt = result["temporal_coeffs"]
        X_mean = result["mean_field"].reshape(-1, 1)

        n_snapshots = len(time_series)
        rank = min(rank, len(S))

        # 低秩近似: X ≈ X_mean + U_r Σ_r V_r^T
        Ur = U[:, :rank]
        Sr = np.diag(S[:rank])
        Vr = Vt[:rank, :]

        X_recon = X_mean + Ur @ Sr @ Vr

        reconstructed = []
        original = np.zeros((self.nr * self.ntheta, n_snapshots))
        for i, snap in enumerate(time_series):
            original[:, i] = snap.reshape(-1)
            reconstructed.append(X_recon[:, i].reshape(self.nr, self.ntheta))

        frob_error = np.linalg.norm(original - X_recon, "fro")
        frob_original = np.linalg.norm(original, "fro")
        rel_error = frob_error / frob_original if frob_original > 1e-30 else 0.0

        return reconstructed, rel_error
