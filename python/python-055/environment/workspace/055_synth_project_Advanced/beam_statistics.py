"""
beam_statistics.py
基于种子项目 567_hypersphere_positive_distance 的超球面正象限采样统计思想，
构建多波束声纳波束开角覆盖的统计特性分析模块。

科学背景：多波束声纳系统通过多个不同开角的波束覆盖海底扇形区域。
各波束在三维指向空间中的方向向量构成单位球面上的点集。
利用超球面（这里是 S² 球面）上随机点距离分布的统计特性，
可定量评估波束覆盖的均匀性与盲区风险。

核心公式：
    设两个波束方向单位向量为 u, v ∈ S²_{+}（上半球面），
    其弦距离 d = ||u - v||₂ = 2 sin(Δθ/2)，
    其中 Δθ 为两波束夹角。
    在均匀分布假设下，d 的期望与方差可通过蒙特卡洛估计：
        μ_d = E[d] ,  σ²_d = Var(d)
"""

import numpy as np


class BeamStatisticsAnalyzer:
    """
    波束角度覆盖统计分析仪。

    将波束方向向量视为单位正超球面（positive hypersphere）上的采样点，
    利用成对距离分布评估覆盖均匀性。
    """

    def __init__(self, dim: int = 3):
        """
        参数:
            dim: 空间维度，声纳波束方向为 3 维（dim=3）
        """
        if dim < 2:
            raise ValueError("维度必须 >= 2")
        self.dim = dim

    @staticmethod
    def sample_positive_hypersphere(dim: int) -> np.ndarray:
        """
        在单位正超球面上生成均匀随机采样点。

        算法：
            1. 生成 dim 维独立标准正态变量 x ~ N(0, I)
            2. 归一化：x ← x / ||x||
            3. 取绝对值映射到正象限：x ← |x|
            4. 重新归一化（可选，这里由于对称性保持单位长度）

        数学依据：高斯分布的径向对称性保证归一化后在球面上均匀。

        参数:
            dim: 空间维度
        返回:
            单位方向向量，形状 (dim,)
        """
        x = np.abs(np.random.randn(dim))
        norm = np.linalg.norm(x)
        if norm < 1e-15:
            # 退化保护：若全部接近 0，返回第一个坐标轴方向
            x = np.zeros(dim)
            x[0] = 1.0
            return x
        return x / norm

    def compute_distance_stats(self, n_samples: int = 5000) -> tuple:
        """
        估计正超球面上随机点对距离的统计量。

        参数:
            n_samples: 蒙特卡洛采样对数
        返回:
            (mu, variance) 距离均值与方差
        """
        if n_samples < 2:
            raise ValueError("n_samples 必须 >= 2")

        distances = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            p = self.sample_positive_hypersphere(self.dim)
            q = self.sample_positive_hypersphere(self.dim)
            distances[i] = np.linalg.norm(p - q)

        mu = float(np.mean(distances))
        if n_samples > 1:
            var = float(np.sum((distances - mu) ** 2) / (n_samples - 1))
        else:
            var = 0.0
        return mu, var

    def analyze_beam_coverage(self, beam_directions: np.ndarray) -> dict:
        """
        分析给定波束方向集合的覆盖统计特性。

        参数:
            beam_directions: 形状 (n_beams, dim) 的数组，每行为单位方向向量
        返回:
            统计字典，包含最小夹角、平均夹角、覆盖均匀度指数等
        """
        beam_directions = np.asarray(beam_directions, dtype=np.float64)
        if beam_directions.ndim != 2 or beam_directions.shape[1] != self.dim:
            raise ValueError(f"beam_directions 形状应为 (n_beams, {self.dim})")

        n_beams = beam_directions.shape[0]
        if n_beams < 2:
            return {
                'n_beams': n_beams,
                'min_angle_deg': 0.0,
                'mean_angle_deg': 0.0,
                'coverage_uniformity': 0.0,
            }

        # 归一化确保单位长度
        norms = np.linalg.norm(beam_directions, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        beam_directions = beam_directions / norms

        # 计算所有点对距离
        distances = []
        angles = []
        for i in range(n_beams):
            for j in range(i + 1, n_beams):
                d = np.linalg.norm(beam_directions[i] - beam_directions[j])
                distances.append(d)
                # 由 d = 2 sin(θ/2) 反解 θ
                sin_half = np.clip(d / 2.0, 0.0, 1.0)
                theta = 2.0 * np.arcsin(sin_half)
                angles.append(theta)

        distances = np.array(distances)
        angles = np.array(angles)

        # 均匀度指数：基于距离变异系数的倒数（越大越均匀）
        mean_dist = np.mean(distances)
        std_dist = np.std(distances, ddof=1)
        uniformity = mean_dist / (std_dist + 1e-12)

        return {
            'n_beams': n_beams,
            'min_angle_deg': float(np.degrees(np.min(angles))),
            'max_angle_deg': float(np.degrees(np.max(angles))),
            'mean_angle_deg': float(np.degrees(np.mean(angles))),
            'std_angle_deg': float(np.degrees(np.std(angles, ddof=1))),
            'mean_chord_distance': float(mean_dist),
            'coverage_uniformity': float(uniformity),
        }

    def generate_optimal_fan_beams(
        self,
        n_beams: int,
        max_opening_angle_deg: float = 60.0,
        azimuth_deg: float = 0.0
    ) -> np.ndarray:
        """
        生成扇形多波束方向向量，覆盖垂直到最大开角范围。

        参数:
            n_beams: 波束数量
            max_opening_angle_deg: 最大开角（从垂直向下算起）
            azimuth_deg: 方位角（度）
        返回:
            方向向量数组，形状 (n_beams, 3)
        """
        if n_beams < 1:
            raise ValueError("波束数必须 >= 1")

        max_theta = np.radians(max_opening_angle_deg)
        azimuth = np.radians(azimuth_deg)

        # 在 [0, max_theta] 内等角度分布
        thetas = np.linspace(0.0, max_theta, n_beams)

        directions = np.zeros((n_beams, 3), dtype=np.float64)
        for i, theta in enumerate(thetas):
            # 球坐标：theta 从 z 轴（垂直向下）偏离
            # 注意：声纳通常 z 轴向下为正
            directions[i, 0] = np.sin(theta) * np.cos(azimuth)
            directions[i, 1] = np.sin(theta) * np.sin(azimuth)
            directions[i, 2] = np.cos(theta)

        return directions
