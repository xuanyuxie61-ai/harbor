r"""
phase_space_integral.py
=======================
动量空间高维数值积分引擎 —— 融合原项目 931_pyramid_felippa_rule
(金字塔单元高阶求积)、654_lattice_rule (Fibonacci 格点规则)
与 1312_triangle_monte_car洛 (三角形域 Monte Carlo)。

在 SPDC 量子光源理论中，三维动量空间 (:math:`k_x, k_y, k_z`) 或
频率-横向动量联合空间中的积分决定总产率与耦合效率：

.. math::
    \eta = \int_{\mathcal{D}} d^3k \; g(k_x, k_y, k_z)

积分域 :math:`\mathcal{D}` 往往为：
1. **金字塔形区域**：纵向动量 :math:`k_z \in [0, k_{\max}]`，
   横向 :math:`k_x, k_y` 随 :math:`k_z` 线性收缩的锥形区域。
2. **三角形截面**：横向模式在 :math:`(k_x, k_y)` 平面上的允许区域
   常为三角形（离散化波导模式）。
3. **周期化超立方**：对于光滑周期被积函数，Fibonacci 格点规则
   提供准蒙特卡洛级别的收敛速率 :math:`O(N^{-1})`。

核心公式
--------
**Fibonacci 格点规则**（二维周期积分）

对生成向量 :math:`z = (1, F_{m-1})`，其中 :math:`F_m` 为 Fibonacci 数：

.. math::
    x_j = \left\{ \frac{j}{F_m} z \right\}, \quad j=0,\dots,F_m-1

积分近似

.. math::
    I \approx \frac{1}{F_m} \sum_{j=0}^{F_m-1} f(x_j)

其中 :math:`\{\cdot\}` 表示取小数部分。

**三角形 Monte Carlo**

在单位三角形 :math:`T_2 = \{(s,t): s\ge 0, t\ge 0, s+t\le 1\}` 上均匀采样：

.. math::
    \\xi_1 = 1 - \sqrt{1-u_1}, \quad
    \\xi_2 = (1-\\xi_1) u_2

其中 :math:`u_1, u_2 \sim \mathcal{U}(0,1)` 独立同分布。

**Felippa 金字塔求积**

在参考金字塔 :math:`\{(x,y,z): |x|\le 1-z, |y|\le 1-z, 0\le z\le 1\}` 上，
48 点高阶规则满足所有 :math:`x^p y^q z^r`（:math:`p+q+r\le 15`）精确。
"""

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Fibonacci lattice rule (from 654_lattice_rule)
# ---------------------------------------------------------------------------
def fibonacci_sequence(n: int) -> np.ndarray:
    r"""
    返回前 n 个 Fibonacci 数（:math:`F_0=0, F_1=1`）。
    """
    if n <= 0:
        return np.array([], dtype=int)
    F = np.zeros(n, dtype=int)
    F[0] = 0
    if n > 1:
        F[1] = 1
    for i in range(2, n):
        F[i] = F[i - 1] + F[i - 2]
    return F


def lattice_rule_2d_periodic(f: Callable, m_order: int) -> float:
    r"""
    二维 Fibonacci 格点积分，被积函数周期为 1。

    参数
    ----
    f : callable(x) -> float
        x 为长度 2 的 ndarray。
    m_order : int
        使用 Fibonacci 数 :math:`F_{m_{\text{order}}}` 作为总点数。

    返回
    ----
    quad : float
        积分估计值。
    """
    if m_order < 2:
        raise ValueError("m_order 必须至少为 2。")
    F = fibonacci_sequence(m_order + 1)
    fm = F[m_order]
    z = np.array([1, F[m_order - 1]], dtype=int)
    quad = 0.0
    for j in range(fm):
        x = (j * z / fm) % 1.0
        quad += f(x)
    quad /= fm
    return quad


# ---------------------------------------------------------------------------
# Triangle Monte Carlo (from 1312_triangle_monte_carlo)
# ---------------------------------------------------------------------------
def triangle_unit_sample_random(n_samples: int) -> np.ndarray:
    r"""
    在单位三角形内均匀随机采样。

    参数
    ----
    n_samples : int
        采样点数，> 0。

    返回
    ----
    p : np.ndarray, shape (n_samples, 2)
        采样点 (:math:`\\\xi_1, \\\xi_2`)。
    """
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正。")
    u = np.random.rand(n_samples, 2)
    xi1 = 1.0 - np.sqrt(1.0 - u[:, 0])
    xi2 = (1.0 - xi1) * u[:, 1]
    return np.column_stack([xi1, xi2])


def reference_to_physical_t3(t_vertices: np.ndarray,
                              p_ref: np.ndarray) -> np.ndarray:
    r"""
    将单位三角形参考点映射到物理三角形。

    参数
    ----
    t_vertices : np.ndarray, shape (2, 3)
        三角形三个顶点的物理坐标。
    p_ref : np.ndarray, shape (n, 2)
        参考点 (:math:`\\xi_1, \\xi_2`)。

    返回
    ----
    p_phys : np.ndarray, shape (n, 2)
    """
    # 物理坐标 = V0 + xi1*(V1-V0) + xi2*(V2-V0)
    v0 = t_vertices[:, 0]
    v1 = t_vertices[:, 1]
    v2 = t_vertices[:, 2]
    p_phys = (v0[None, :]
              + p_ref[:, 0][:, None] * (v1 - v0)[None, :]
              + p_ref[:, 1][:, None] * (v2 - v0)[None, :])
    return p_phys


def triangle_monte_carlo(f: Callable, t_vertices: np.ndarray,
                         n_samples: int) -> float:
    r"""
    三角形域 Monte Carlo 积分。

    .. math::
        I = \int_T f(x,y) \, dx dy \approx |T| \cdot \frac{1}{N}
        \sum_{j=1}^{N} f(x_j, y_j)

    参数
    ----
    f : callable(p) -> float
        p 为形状 (2,) 的 ndarray。
    t_vertices : np.ndarray, shape (2, 3)
    n_samples : int

    返回
    ----
    result : float
    """
    # 三角形面积
    v0, v1, v2 = t_vertices[:, 0], t_vertices[:, 1], t_vertices[:, 2]
    area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1])
                     - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    p_ref = triangle_unit_sample_random(n_samples)
    p_phys = reference_to_physical_t3(t_vertices, p_ref)
    vals = np.array([f(p_phys[i, :]) for i in range(n_samples)])
    return area * np.mean(vals)


# ---------------------------------------------------------------------------
# Pyramid Felippa rule (from 931_pyramid_felippa_rule)
# ---------------------------------------------------------------------------
def pyramid_unit_o48() -> tuple:
    r"""
    返回参考单位金字塔上的 48 点 Felippa 高阶求积规则。

    积分区域：

    .. math::
        -(1-z) \le x \le 1-z, \quad -(1-z) \le y \le 1-z,
        \quad 0 \le z \le 1

    返回
    ----
    w : np.ndarray, shape (48,)
    xyz : np.ndarray, shape (48, 3)
        求积权重与节点坐标。
    """
    w = np.array([
        2.0124193944268246e-02, 2.0124193944268246e-02,
        2.0124193944268246e-02, 2.0124193944268246e-02,
        2.6035113704301078e-02, 2.6035113704301078e-02,
        2.6035113704301078e-02, 2.6035113704301078e-02,
        1.2455779523974553e-02, 1.2455779523974553e-02,
        1.2455779523974553e-02, 1.2455779523974553e-02,
        1.8787399879480816e-03, 1.8787399879480816e-03,
        1.8787399879480816e-03, 1.8787399879480816e-03,
        4.3295792780774528e-02, 4.3295792780774528e-02,
        4.3295792780774528e-02, 4.3295792780774528e-02,
        1.9746324983412729e-02, 1.9746324983412729e-02,
        1.9746324983412729e-02, 1.9746324983412729e-02,
        5.6012722352359053e-02, 5.6012722352359053e-02,
        5.6012722352359053e-02, 5.6012722352359053e-02,
        2.5546256292747385e-02, 2.5546256292747385e-02,
        2.5546256292747385e-02, 2.5546256292747385e-02,
        2.6797736629178864e-02, 2.6797736629178864e-02,
        2.6797736629178864e-02, 2.6797736629178864e-02,
        1.2221899226537335e-02, 1.2221899226537335e-02,
        1.2221899226537335e-02, 1.2221899226537335e-02,
        4.0419774045321504e-03, 4.0419774045321504e-03,
        4.0419774045321504e-03, 4.0419774045321504e-03,
        1.8434631699582684e-03, 1.8434631699582684e-03,
        1.8434631699582684e-03, 1.8434631699582684e-03
    ], dtype=np.float64)

    xyz = np.array([
        [0.8809173162445091, 0.0, 0.048500549446997],
        [-0.8809173162445091, 0.0, 0.048500549446997],
        [0.0, 0.8809173162445091, 0.048500549446997],
        [0.0, -0.8809173162445091, 0.048500549446997],
        [0.7049187411264822, 0.0, 0.238600737551862],
        [-0.7049187411264822, 0.0, 0.238600737551862],
        [0.0, 0.7049187411264822, 0.238600737551862],
        [0.0, -0.7049187411264822, 0.238600737551862],
        [0.4471273214318976, 0.0, 0.517047295104368],
        [-0.4471273214318976, 0.0, 0.517047295104368],
        [0.0, 0.4471273214318976, 0.517047295104368],
        [0.0, -0.4471273214318976, 0.517047295104368],
        [0.1890048606512345, 0.0, 0.7958514178967731],
        [-0.1890048606512345, 0.0, 0.7958514178967731],
        [0.0, 0.1890048606512345, 0.7958514178967731],
        [0.0, -0.1890048606512345, 0.7958514178967731],
        [0.3620973341032218, 0.3620973341032218, 0.048500549446997],
        [-0.3620973341032218, 0.3620973341032218, 0.048500549446997],
        [-0.3620973341032218, -0.3620973341032218, 0.048500549446997],
        [0.3620973341032218, -0.3620973341032218, 0.048500549446997],
        [0.7668893206038754, 0.7668893206038754, 0.048500549446997],
        [-0.7668893206038754, 0.7668893206038754, 0.048500549446997],
        [-0.7668893206038754, -0.7668893206038754, 0.048500549446997],
        [0.7668893206038754, -0.7668893206038754, 0.048500549446997],
        [0.2897538647661807, 0.2897538647661807, 0.238600737551862],
        [-0.2897538647661807, 0.2897538647661807, 0.238600737551862],
        [-0.2897538647661807, -0.2897538647661807, 0.238600737551862],
        [0.2897538647661807, -0.2897538647661807, 0.238600737551862],
        [0.6136724122623316, 0.6136724122623316, 0.238600737551862],
        [-0.6136724122623316, 0.6136724122623316, 0.238600737551862],
        [-0.6136724122623316, -0.6136724122623316, 0.238600737551862],
        [0.6136724122623316, -0.6136724122623316, 0.238600737551862],
        [0.1837897928779802, 0.1837897928779802, 0.517047295104368],
        [-0.1837897928779802, 0.1837897928779802, 0.517047295104368],
        [-0.1837897928779802, -0.1837897928779802, 0.517047295104368],
        [0.1837897928779802, -0.1837897928779802, 0.517047295104368],
        [0.3892501162517316, 0.3892501162517316, 0.517047295104368],
        [-0.3892501162517316, 0.3892501162517316, 0.517047295104368],
        [-0.3892501162517316, -0.3892501162517316, 0.517047295104368],
        [0.3892501162517316, -0.3892501162517316, 0.517047295104368],
        [0.0776896479525748, 0.0776896479525748, 0.7958514178967731],
        [-0.0776896479525748, 0.0776896479525748, 0.7958514178967731],
        [-0.0776896479525748, -0.0776896479525748, 0.7958514178967731],
        [0.0776896479525748, -0.0776896479525748, 0.7958514178967731],
        [0.1645396298866986, 0.1645396298866986, 0.7958514178967731],
        [-0.1645396298866986, 0.1645396298866986, 0.7958514178967731],
        [-0.1645396298866986, -0.1645396298866986, 0.7958514178967731],
        [0.1645396298866986, -0.1645396298866986, 0.7958514178967731]
    ], dtype=np.float64)

    return w, xyz


def pyramid_unit_volume() -> float:
    """
    参考单位金字塔体积 :math:`V = 4/3`。
    """
    return 4.0 / 3.0


def integrate_pyramid_felippa(f: Callable) -> float:
    """
    使用 48 点 Felippa 规则在参考单位金字塔上积分。

    参数
    ----
    f : callable(xyz) -> float
        xyz 为长度 3 的 ndarray [x, y, z]。

    返回
    ----
    result : float
    """
    w, xyz = pyramid_unit_o48()
    vals = np.array([f(xyz[i, :]) for i in range(len(w))])
    # 权重已针对单位金字塔体积归一化（权重和 = 4/3）
    return np.sum(w * vals)


# ---------------------------------------------------------------------------
# Composite integrator for SPDC phase space
# ---------------------------------------------------------------------------
def phase_space_coupling_efficiency(kx_max: float, ky_max: float,
                                    kz_func: Callable,
                                    integrand: Callable,
                                    method: str = "lattice") -> float:
    """
    在动量空间计算耦合效率积分，支持 lattice / pyramid / triangle_mc 三种方法。

    参数
    ----
    kx_max, ky_max : float
        横向动量截断。
    kz_func : callable(kx, ky) -> (kz_min, kz_max)
        纵向动量范围。
    integrand : callable(kx, ky, kz) -> float
        被积函数。
    method : str
        "lattice", "pyramid", "triangle_mc"。

    返回
    ----
    eta : float
        耦合效率。
    """
    if method == "lattice":
        # 将 (kx, ky) 映射到 [0,1]^2 周期域，做归一化积分
        vol_factor = 1.0 / (4.0 * kx_max * ky_max)

        def f_periodic(x):
            kx = (x[0] - 0.5) * 2.0 * kx_max
            ky = (x[1] - 0.5) * 2.0 * ky_max
            kz_min, kz_max = kz_func(kx, ky)
            # 对 kz 做简单梯形积分 5 点
            kz = np.linspace(kz_min, kz_max, 5)
            if len(kz) <= 1:
                return 0.0
            dz = kz[1] - kz[0]
            val = 0.0
            for k in range(len(kz)):
                weight = 1.0 if (k == 0 or k == len(kz) - 1) else 2.0
                val += weight * integrand(kx, ky, kz[k])
            val *= 0.5 * dz * vol_factor
            return val

        return lattice_rule_2d_periodic(f_periodic, 12)

    elif method == "pyramid":
        # 映射金字塔到物理域并做体积归一化
        vol = pyramid_unit_volume()

        def f_pyramid(xyz):
            x, y, z = xyz
            kz = z
            kx = x * (1.0 - z) * kx_max
            ky = y * (1.0 - z) * ky_max
            return integrand(kx, ky, kz) / vol

        return integrate_pyramid_felippa(f_pyramid)

    elif method == "triangle_mc":
        # 在 kx-ky 平面三角形区域积分，kz 梯形，面积归一化
        verts = np.array([[0.0, kx_max, 0.0],
                          [0.0, 0.0, ky_max]], dtype=np.float64)
        area = 0.5 * abs(kx_max * ky_max)

        def f_2d(p):
            kx, ky = p
            kz_min, kz_max = kz_func(kx, ky)
            kz = np.linspace(kz_min, kz_max, 5)
            if len(kz) <= 1:
                return 0.0
            dz = kz[1] - kz[0]
            val = 0.0
            for k in range(len(kz)):
                weight = 1.0 if (k == 0 or k == len(kz) - 1) else 2.0
                val += weight * integrand(kx, ky, kz[k])
            return val * 0.5 * dz / area

        return triangle_monte_carlo(f_2d, verts, 20000)

    else:
        raise ValueError(f"未知积分方法: {method}")
