"""
激波几何重构与后激波统计 (from 1337_triangulation_histogram).

Burkardt 原程序：给定三角剖分和数据点，统计每个三角形内部
包含的点数，比较 Ai/Atotal 与 Ni/Ntotal 以判断均匀性。

在超新星中我们将 1D 激波推广为**准 2D 激波曲面** (在 (θ, φ) 方向
采用低阶球谐展开)，并使用"三角剖分直方图"思想诊断后激波
物质分布的各向异性。

激波曲面用球谐 Y_lm 展开到 l_max = 2 (5 个系数)：
  R_shock(θ, φ) = Σ_{l=0}^{2} Σ_{m=-l}^{l} a_{lm} Y_{lm}(θ, φ)

构造等角三角剖分 (HEALPix 简化版)：将单位球面分割为 N_tri 个
近似等面积三角形。对每个三角形，用重心坐标判断激波脉动采样
点是否落入，统计各三角形落入频率 f_i。

均匀性指标：
  χ^2 = Σ (f_i - f_expected)^2 / f_expected
  f_expected = N_sample / N_tri
χ^2 / (N_tri - 1) ≈ 1 表示激波分布各向同性。

后激波压强跳跃比 (Rankine-Hugoniot):
  P_2/P_1 = (2 γ M^2 - (γ-1)) / (γ+1)
  ρ_2/ρ_1 = (γ+1) M^2 / ((γ-1) M^2 + 2)
  其中 M = v_shock / c_s1 为激波 Mach 数.
"""
from __future__ import annotations
import math
import numpy as np


def spherical_harmonic_Y(l: int, m: int, theta: float, phi: float) -> complex:
    """球谐函数 Y_l^m(θ, φ) (未归一化, 实部).

    对 l ≤ 2 手动实现：
      Y_0^0 = 1
      Y_1^0 = cos θ,   Y_1^±1 = sin θ e^{±iφ}
      Y_2^0 = (3 cos^2 θ - 1)/2,
      Y_2^±1 = sin θ cos θ e^{±iφ},
      Y_2^±2 = sin^2 θ e^{±2iφ}
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    if l == 0 and m == 0:
        return complex(1.0)
    if l == 1:
        if m == 0:
            return complex(ct)
        if m == 1:
            return complex(st * math.cos(phi))
        if m == -1:
            return complex(st * math.sin(phi))
    if l == 2:
        if m == 0:
            return complex(0.5 * (3.0 * ct * ct - 1.0))
        if m == 1:
            return complex(st * ct * math.cos(phi))
        if m == -1:
            return complex(st * ct * math.sin(phi))
        if m == 2:
            return complex(st * st * math.cos(2.0 * phi))
        if m == -2:
            return complex(st * st * math.sin(2.0 * phi))
    return complex(0.0)


class ShockSurface:
    """准 2D 激波曲面 (l_max = 2 球谐表示)."""

    def __init__(self, R0: float = 1.5e7, delta: float = 0.1):
        self.R0 = float(R0)
        self.delta = float(delta)
        # 球谐系数 (实部)
        self.a_lm = {}
        self.a_lm[(0, 0)] = 1.0  # 归一化使平均半径 = R0
        for l in (1, 2):
            for m in range(-l, l + 1):
                self.a_lm[(l, m)] = 0.0

    def perturb(self, rng: np.random.Generator, amplitude: float = 0.05):
        """对非径向模式施加随机扰动 (模拟 SASI 激发).

        SASI 在 l=1 和 l=2 模式上最强。
        """
        for l in (1, 2):
            for m in range(-l, l + 1):
                self.a_lm[(l, m)] = amplitude * rng.standard_normal()

    def radius_at(self, theta: float, phi: float) -> float:
        """计算 (θ, φ) 方向的激波半径."""
        val = 0.0
        for (l, m), a in self.a_lm.items():
            val += a * spherical_harmonic_Y(l, m, theta, phi).real
        return self.R0 * val

    def sample_directions(self, n_theta: int = 8, n_phi: int = 16):
        """生成方向采样点 (theta, phi) 及其激波半径."""
        thetas = (np.arange(n_theta) + 0.5) * math.pi / n_theta
        phis = (np.arange(n_phi) + 0.5) * 2.0 * math.pi / n_phi
        theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
        r_grid = np.zeros_like(theta_grid)
        for i in range(n_theta):
            for j in range(n_phi):
                r_grid[i, j] = self.radius_at(thetas[i], phis[j])
        return theta_grid, phi_grid, r_grid


class IcosphereMesh:
    """简化等面积球面三角剖分 (基于经纬度网格).

    将 (θ, φ) 域 [0, π] × [0, 2π] 分割为 N_θ × N_φ 个四边形,
    每个再切为 2 个三角形, 共 2 N_θ N_φ 个三角形.
    每个三角形的面积 ≈ 4π / (2 N_θ N_φ).
    """

    def __init__(self, n_theta: int = 6, n_phi: int = 12):
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.n_tri = 2 * n_theta * n_phi
        self.expected_area = 4.0 * math.pi / self.n_tri
        # 预计算三角形顶点
        self.triangles = []
        dtheta = math.pi / n_theta
        dphi = 2.0 * math.pi / n_phi
        for i in range(n_theta):
            t0 = i * dtheta
            t1 = (i + 1) * dtheta
            for j in range(n_phi):
                p0 = j * dphi
                p1 = (j + 1) * dphi
                # 四边形 (t0,p0), (t1,p0), (t1,p1), (t0,p1)
                # 切为 2 个三角形
                self.triangles.append(((t0, p0), (t1, p0), (t1, p1)))
                self.triangles.append(((t0, p0), (t1, p1), (t0, p1)))

    def point_in_triangle(self, p, tri) -> bool:
        """判断点 p=(θ, φ) 是否在三角形 tri 内 (重心坐标).

        在 (θ, φ) 参数平面判断 (小尺度近似).
        """
        a, b, c = tri
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def histogram(self, sample_points) -> np.ndarray:
        """统计落入每个三角形的样本点数 (from 1337_triangulation_histogram)."""
        counts = np.zeros(self.n_tri, dtype=np.int64)
        for pt in sample_points:
            for k, tri in enumerate(self.triangles):
                if self.point_in_triangle(pt, tri):
                    counts[k] += 1
                    break
        return counts

    def uniformity_chi2(self, counts: np.ndarray) -> float:
        """χ^2 均匀性检验."""
        N = float(counts.sum())
        expected = N / self.n_tri if N > 0 else 1.0
        chi2 = float(np.sum((counts - expected) ** 2 / max(expected, 1.0e-30)))
        return chi2


def rankine_hugoniot(M: float, gamma: float = 5.0 / 3.0) -> dict:
    """Rankine-Hugoniot 激波跳跃关系.

  给定 Mach 数 M, 返回压强比、密度比、温度比、下游 Mach 数.
  """
    if M <= 1.0:
        return {'pressure_ratio': 1.0, 'density_ratio': 1.0,
                'temperature_ratio': 1.0, 'downstream_M': 1.0}
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    p_ratio = (2.0 * gamma * M ** 2 - gm1) / gp1
    rho_ratio = gp1 * M ** 2 / (gm1 * M ** 2 + 2.0)
    T_ratio = p_ratio / rho_ratio
    # 下游 Mach 数
    M2_sq = (gm1 * M ** 2 + 2.0) / (2.0 * gamma * M ** 2 - gm1)
    M2 = math.sqrt(max(M2_sq, 0.0))
    return {'pressure_ratio': p_ratio, 'density_ratio': rho_ratio,
            'temperature_ratio': T_ratio, 'downstream_M': M2}
