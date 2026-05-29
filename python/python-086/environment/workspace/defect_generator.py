# -*- coding: utf-8 -*-
"""
defect_generator.py
壳体几何缺陷生成与蒙特卡洛采样

融合种子项目:
  - 069_ball_monte_carlo: 球体内蒙特卡洛积分与采样
  - 314_double_c_data: 双"C"型复杂数据生成

科学背景:
  薄壁壳体对初始几何缺陷极为敏感。Koiter 理论表明，
  微小缺陷 (δ/t ≈ 0.01) 即可使实际屈曲载荷降低至经典值的 20%~50%。

  缺陷模态通常表示为 Fourier 级数:
    w̄(x,θ) = Σ_m Σ_n A_{mn} sin(mπx/L) cos(nθ + φ_{mn})

  其中幅值 A_{mn} 服从随机分布，常用方法:
    1. 单模态缺陷: 仅含一个 (m,n) 模态
    2. 多模态缺陷: 多个模态叠加
    3. 实测缺陷数据库投影

  本模块采用蒙特卡洛方法在 Fourier 系数空间 (m,n,A,φ) 中采样，
  生成符合统计特征的几何缺陷场。
"""

import numpy as np


class DefectGenerator:
    """
    圆柱壳几何缺陷生成器
    """

    def __init__(self, geometry, seed: int = 42):
        self.geom = geometry
        self.rng = np.random.default_rng(seed)

    def single_mode_defect(self, m: int, n: int, amplitude: float,
                           phase: float = 0.0) -> callable:
        """
        单模态缺陷场

        w̄(x,θ) = A sin(mπx/L) cos(nθ + φ)

        Parameters
        ----------
        m : int
            轴向半波数
        n : int
            环向波数
        amplitude : float
            缺陷幅值 A
        phase : float
            相位角 φ

        Returns
        -------
        defect_func : callable
            输入 (theta, x) 输出缺陷幅值 w_bar
        """
        R, L = self.geom.R, self.geom.L

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            return amplitude * np.sin(m * np.pi * x / L) * np.cos(n * theta + phase)
        return defect_func

    def double_c_defect(self, n1: int, n2: int, amplitude: float) -> callable:
        """
        双"C"型缺陷场 (基于 314_double_c_data 的双结构思想)

        叠加两个反向旋转的螺旋模态，形成类似双C的嵌套结构:
          w̄ = A [ sin(m₁πx/L)cos(n₁θ) - sin(m₂πx/L)cos(n₂θ) ]
        这种缺陷模式在实验中常见于制造误差导致的局部凹陷。

        Parameters
        ----------
        n1, n2 : int
            两个模态的环向波数
        amplitude : float
            总幅值

        Returns
        -------
        defect_func : callable
        """
        R, L = self.geom.R, self.geom.L
        m1 = max(1, n1 // 2)
        m2 = max(1, n2 // 2)

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            w1 = np.sin(m1 * np.pi * x / L) * np.cos(n1 * theta)
            w2 = np.sin(m2 * np.pi * L / L) * np.cos(n2 * theta)
            return amplitude * (w1 - 0.7 * w2)
        return defect_func

    def monte_carlo_multi_mode(self, n_modes: int = 10,
                               amplitude_ratio: float = 0.01,
                               distribution: str = "gaussian") -> callable:
        """
        蒙特卡洛多模态缺陷场 (基于 069_ball_monte_carlo 的随机采样思想)

        在 Fourier 空间中随机采样模态幅值:
          w̄(x,θ) = Σ_{k=1}^{N_modes} A_k sin(m_k πx/L) cos(n_k θ + φ_k)

        幅值谱满足幂律衰减:
          E[A_k²] ∝ 1 / (m_k² + n_k²)

        Parameters
        ----------
        n_modes : int
            叠加模态数
        amplitude_ratio : float
            最大缺陷幅值与厚度之比 (δ/t)
        distribution : str
            "gaussian" 或 "uniform"

        Returns
        -------
        defect_func : callable
        """
        t = self.geom.t
        L = self.geom.L
        max_amp = amplitude_ratio * t

        modes = []
        for k in range(n_modes):
            m = self.rng.integers(1, 6)
            n = self.rng.integers(1, 12)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            # 幂律衰减
            decay = 1.0 / np.sqrt(m ** 2 + n ** 2 + 1.0)
            if distribution == "gaussian":
                a = self.rng.normal(0.0, max_amp * decay)
            else:
                a = self.rng.uniform(-max_amp * decay, max_amp * decay)
            modes.append((m, n, a, phase))

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            w = np.zeros_like(theta, dtype=float)
            for m, n, a, ph in modes:
                w += a * np.sin(m * np.pi * x / L) * np.cos(n * theta + ph)
            return w
        return defect_func

    def ball_volume_defect(self, n_samples: int = 1000,
                           amplitude_ratio: float = 0.01) -> callable:
        """
        基于单位球内蒙特卡洛采样的缺陷场

        将每个采样点 p = (p_x, p_y, p_z) 映射为缺陷模态参数:
          m = 1 + floor(5 * |p_x|)
          n = 1 + floor(10 * |p_y|)
          A ∝ p_z * δ_max
        球坐标映射保证采样均匀覆盖模态空间。

        Parameters
        ----------
        n_samples : int
            蒙特卡洛采样点数
        amplitude_ratio : float
            幅值比 δ/t

        Returns
        -------
        defect_func : callable
        """
        t = self.geom.t
        max_amp = amplitude_ratio * t
        L = self.geom.L

        # 单位球内均匀采样 (基于 ball_monte_carlo 的 rejection 方法)
        samples = []
        while len(samples) < n_samples:
            xyz = self.rng.normal(size=3)
            r = np.linalg.norm(xyz)
            if r > 1e-10 and r <= 1.0:
                samples.append(xyz / r)

        modes = []
        n_modes = min(n_samples, 20)
        for s in samples[:n_modes]:
            m = 1 + int(5.0 * abs(s[0]))
            n = 1 + int(10.0 * abs(s[1]))
            a = s[2] * max_amp / (m + n)
            ph = self.rng.uniform(0.0, 2.0 * np.pi)
            modes.append((m, n, a, ph))

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            w = np.zeros_like(theta, dtype=float)
            for m, n, a, ph in modes:
                w += a * np.sin(m * np.pi * x / L) * np.cos(n * theta + ph)
            return w
        return defect_func

    def apply_defect_to_mesh(self, mesh, defect_func: callable) -> np.ndarray:
        """
        将缺陷场施加到网格节点上，返回修正后的节点坐标

        Parameters
        ----------
        mesh : ShellTriMesh
        defect_func : callable
            defect_func(theta, x) -> w_bar

        Returns
        -------
        new_nodes : (N, 3) ndarray
            含缺陷的节点坐标
        """
        nodes = mesh.nodes.copy()
        R = mesh.geom.R
        theta = np.arctan2(nodes[:, 1], nodes[:, 0])
        x = nodes[:, 2]
        w_bar = defect_func(theta, x)
        # 沿法向偏移
        normal = mesh.geom.surface_normal(theta)
        new_nodes = nodes + normal * w_bar[:, np.newaxis]
        return new_nodes

    def defect_statistics(self, mesh, defect_func: callable) -> dict:
        """
        计算缺陷场的统计特征

        Returns
        -------
        stats : dict
        """
        theta = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
        x = mesh.nodes[:, 2]
        w = defect_func(theta, x)
        t = self.geom.t
        stats = {
            'max_defect': float(np.max(np.abs(w))),
            'rms_defect': float(np.sqrt(np.mean(w ** 2))),
            'defect_to_thickness': float(np.max(np.abs(w)) / t),
            'mean_defect': float(np.mean(w))
        }
        return stats
