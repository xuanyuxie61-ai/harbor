# -*- coding: utf-8 -*-
"""
ejecta_mesh.py
==============
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

球坐标 (r, theta, phi) 网格生成、自适应加密与二次→线性降阶.
用于离散 kilonova 抛射物的辐射转移计算域.

球坐标网格
----------
节点::

    r_i     = r_min * (r_max / r_min)^{i/(N_r-1)}    (对数网格, i = 0..N_r-1)
    theta_j = j * pi / (N_theta - 1)                 (均匀, j = 0..N_theta-1)
    phi_k   = k * 2 pi / (N_phi - 1)                 (均匀, k = 0..N_phi-1)

体积元::

    dV_{ijk} = r_i^2 sin(theta_j) * dr_i * dtheta_j * dphi_k
             = r_i^2 sin(theta_j) * (r_{i+1} - r_{i-1})/2
               * (theta_{j+1} - theta_{j-1})/2 * (phi_{k+1} - phi_{k-1})/2

二次→线性降阶 (映射 1346_triangulation_q2l)
-------------------------------------------
球壳上的"二次三角"由 6 个节点定义 (顶点 + 边中点). 降阶为 4 个线性三角形,
用于构造光线追踪的三角形面片::

    T_q2l: (n1,n2,n3,n4,n5,n6) -> [(n1,n4,n6), (n2,n5,n4),
                                    (n3,n6,n5), (n4,n5,n6)]

映射种子项目
-----------
- 406 (fem2d_mesh_display) → 二维网格读取、节点/单元结构与 base-one 校正
- 1346 (triangulation_q2l) → 二次三角 → 线性三角的 1-to-4 细分
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 1D 网格
# ---------------------------------------------------------------------------
def logarithmic_radial_grid(r_min: float, r_max: float, N: int) -> List[float]:
    """生成对数间距的径向网格.

    r_i = r_min * (r_max/r_min)^{i/(N-1)}
    """
    if r_min <= 0 or r_max <= r_min or N < 2:
        raise ValueError("Require 0 < r_min < r_max, N >= 2")
    log_ratio = math.log(r_max / r_min)
    return [r_min * math.exp(log_ratio * i / (N - 1)) for i in range(N)]


def uniform_angular_grid(a: float, b: float, N: int) -> List[float]:
    """[a, b] 上的均匀角度网格."""
    if N < 2:
        raise ValueError("N >= 2")
    return [a + (b - a) * i / (N - 1) for i in range(N)]


# ---------------------------------------------------------------------------
# 球坐标网格组装
# ---------------------------------------------------------------------------
class SphericalMesh:
    """存储球坐标 (r, theta, phi) 的网格几何信息.

    约定:
        r     : N_r     个节点 (对数)
        theta : N_theta 个节点 (0 = 北极, pi = 南极)
        phi   : N_phi   个节点 (0..2pi, 周期性)
    """

    def __init__(self, N_r: int, N_theta: int, N_phi: int,
                 r_min: float = 1.0e7, r_max: float = 1.0e13):
        self.N_r = N_r
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.r = logarithmic_radial_grid(r_min, r_max, N_r)
        self.theta = uniform_angular_grid(0.0, math.pi, N_theta)
        self.phi = uniform_angular_grid(0.0, 2.0 * math.pi, N_phi)
        self._compute_geometry()

    def _compute_geometry(self) -> None:
        """计算 dr, dtheta, dphi 与体积元.

        体积元精确计算::

            dV_{ijk} = (r_{i+1/2}^3 - r_{i-1/2}^3)/3
                       * |cos(theta_{j-1/2}) - cos(theta_{j+1/2})|
                       * (phi_{k+1/2} - phi_{k-1/2})

        这比 r^2 sin(theta) dr dtheta dphi 的一阶近似准确得多.
        """
        # 半格点
        r_half = [self.r[0]]
        for i in range(self.N_r - 1):
            r_half.append(0.5 * (self.r[i] + self.r[i + 1]))
        r_half.append(self.r[-1])
        theta_half = [self.theta[0]]
        for j in range(self.N_theta - 1):
            theta_half.append(0.5 * (self.theta[j] + self.theta[j + 1]))
        theta_half.append(self.theta[-1])
        phi_half = [self.phi[0]]
        for k in range(self.N_phi - 1):
            phi_half.append(0.5 * (self.phi[k] + self.phi[k + 1]))
        phi_half.append(self.phi[-1])
        # dr, dtheta, dphi (名义值, 用于微分算子)
        self.dr = [0.0] * self.N_r
        for i in range(self.N_r):
            if i == 0:
                self.dr[i] = self.r[1] - self.r[0]
            elif i == self.N_r - 1:
                self.dr[i] = self.r[-1] - self.r[-2]
            else:
                self.dr[i] = 0.5 * (self.r[i + 1] - self.r[i - 1])
        self.dtheta = [0.0] * self.N_theta
        for j in range(self.N_theta):
            if j == 0:
                self.dtheta[j] = self.theta[1] - self.theta[0]
            elif j == self.N_theta - 1:
                self.dtheta[j] = self.theta[-1] - self.theta[-2]
            else:
                self.dtheta[j] = 0.5 * (self.theta[j + 1] - self.theta[j - 1])
        self.dphi = [0.0] * self.N_phi
        for k in range(self.N_phi):
            if k == 0:
                self.dphi[k] = self.phi[1] - self.phi[0]
            elif k == self.N_phi - 1:
                self.dphi[k] = self.phi[-1] - self.phi[-2]
            else:
                self.dphi[k] = 0.5 * (self.phi[k + 1] - self.phi[k - 1])
        # 体积元 (3D) - 精确积分
        self.dV = [[[0.0] * self.N_phi
                    for _ in range(self.N_theta)]
                   for _ in range(self.N_r)]
        for i in range(self.N_r):
            r_vol = (r_half[i + 1] ** 3 - r_half[i] ** 3) / 3.0
            for j in range(self.N_theta):
                theta_vol = abs(math.cos(theta_half[j]) - math.cos(theta_half[j + 1]))
                for k in range(self.N_phi):
                    phi_vol = phi_half[k + 1] - phi_half[k]
                    self.dV[i][j][k] = r_vol * theta_vol * phi_vol

    def total_volume(self) -> float:
        """网格总容积 [cm^3]. 应近似于 (4/3) pi (r_max^3 - r_min^3)."""
        return sum(self.dV[i][j][k]
                   for i in range(self.N_r)
                   for j in range(self.N_theta)
                   for k in range(self.N_phi))

    def expected_volume(self) -> float:
        """解析球壳体积 (4/3) pi (r_max^3 - r_min^3)."""
        return (4.0 / 3.0) * math.pi * (self.r[-1] ** 3 - self.r[0] ** 3)


# ---------------------------------------------------------------------------
# 二次 → 线性降阶 (映射 1346)
# ---------------------------------------------------------------------------
def quadratic_to_linear_triangles(tri6_nodes: List[Tuple[int, ...]]
                                  ) -> List[Tuple[int, int, int]]:
    """将 6 节点二次三角列表转化为 4 × 3 节点线性三角列表.

    对于每个二次三角 (n1, n2, n3, n4, n5, n6)::

        (n1, n4, n6), (n2, n5, n4), (n3, n6, n5), (n4, n5, n6)

    Parameters
    ----------
    tri6_nodes : List of 6-tuples  二次三角节点索引

    Returns
    -------
    List of 3-tuples  线性三角
    """
    tri3 = []
    for (n1, n2, n3, n4, n5, n6) in tri6_nodes:
        tri3.append((n1, n4, n6))
        tri3.append((n2, n5, n4))
        tri3.append((n3, n6, n5))
        tri3.append((n4, n5, n6))
    return tri3


# ---------------------------------------------------------------------------
# 球面三角形面片生成 (用于光线追踪)
# ---------------------------------------------------------------------------
def sphere_surface_triangles(N_theta: int, N_phi: int, offset: int = 0
                             ) -> List[Tuple[int, int, int]]:
    """在 (theta, phi) 球面上生成三角形面片, 形成 2D 网格.

    节点编号:  (j, k) -> offset + j * N_phi + k
    每个四边形 (j,k)-(j,k+1)-(j+1,k+1)-(j+1,k) 切分为两个三角形.
    """
    tris = []
    for j in range(N_theta - 1):
        for k in range(N_phi - 1):
            a = offset + j * N_phi + k
            b = offset + j * N_phi + (k + 1)
            c = offset + (j + 1) * N_phi + (k + 1)
            d = offset + (j + 1) * N_phi + k
            tris.append((a, b, c))
            tris.append((a, c, d))
    return tris


# ---------------------------------------------------------------------------
# 自适应加密: 基于梯度
# ---------------------------------------------------------------------------
def gradient_based_refinement_indicator(
    u: List[float], threshold: float
) -> List[int]:
    """标记需要加密的单元索引 (基于一阶梯度).

    |u_{i+1} - u_{i-1}| / (2 h) > threshold  -> 标记

    Parameters
    ----------
    u         : List[float]  场量 (如密度)
    threshold : float       梯度阈值

    Returns
    -------
    List[int]  需要加密的单元索引
    """
    N = len(u)
    flagged = []
    for i in range(1, N - 1):
        grad = abs(u[i + 1] - u[i - 1]) / 2.0
        if grad > threshold:
            flagged.append(i)
    return flagged


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证体积积分精度."""
    mesh = SphericalMesh(N_r=40, N_theta=20, N_phi=40,
                         r_min=1.0e7, r_max=1.0e13)
    V_num = mesh.total_volume()
    V_ana = mesh.expected_volume()
    rel_err = abs(V_num - V_ana) / V_ana
    if rel_err > 0.05:
        raise AssertionError(f"Volume error {rel_err:.3e} > 5%")
    # Q2L 降阶
    tri6 = [(1, 2, 3, 4, 5, 6)]
    tri3 = quadratic_to_linear_triangles(tri6)
    assert len(tri3) == 4, "Q2L should produce 4 triangles per quadratic"
    return True


if __name__ == "__main__":
    _self_check()
    print("ejecta_mesh self-check passed.")
    mesh = SphericalMesh(N_r=30, N_theta=15, N_phi=30)
    print(f"  mesh dimensions     : {mesh.N_r} x {mesh.N_theta} x {mesh.N_phi}")
    print(f"  numerical volume    : {mesh.total_volume():.4e} cm^3")
    print(f"  analytical volume   : {mesh.expected_volume():.4e} cm^3")
    print(f"  relative error      : {abs(mesh.total_volume() / mesh.expected_volume() - 1):.3e}")
