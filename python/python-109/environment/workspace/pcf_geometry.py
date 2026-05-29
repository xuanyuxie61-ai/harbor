"""
pcf_geometry.py
光子晶体光纤（PCF）几何结构与横截面网格生成

融合原项目:
  - 184_circle_segment: 圆段几何解析计算
  - 1305_triangle_grid: 三角形区域规则网格生成

科学背景:
  光子晶体光纤的横截面由周期性排列的空气孔构成，通常呈六边形晶格。
  空气孔的圆形截面与石英基底的圆段区域决定了光纤的有效模场面积、
  数值孔径及非线性系数。本模块提供圆段解析几何工具与三角形网格
  生成器，用于精确描述PCF的横截面结构。
"""

import numpy as np


def circle_segment_area_from_angle(r: float, theta: float) -> float:
    """
    计算圆扇形（circle segment）的面积。

    公式:
        A = (r^2 / 2) * (theta - sin(theta))

    其中 r 为圆半径，theta 为扇形中心角（弧度）。
    该公式来源于对圆扇形区域的几何积分:
        A = integral_{0}^{theta} integral_{0}^{r} rho drho dphi
          - (1/2) * r^2 * sin(theta)

    Parameters
    ----------
    r : float
        圆的半径，必须 r > 0。
    theta : float
        扇形中心角（弧度），0 <= theta <= 2*pi。

    Returns
    -------
    float
        圆扇形面积。

    Raises
    ------
    ValueError
        若 r <= 0 或 theta 超出 [0, 2*pi]。
    """
    if r <= 0.0:
        raise ValueError("circle_segment_area_from_angle: r must be > 0")
    if theta < 0.0 or theta > 2.0 * np.pi:
        raise ValueError("circle_segment_area_from_angle: theta must be in [0, 2*pi]")
    area = r * r * (theta - np.sin(theta)) / 2.0
    return float(area)


def circle_segment_height_from_angle(r: float, theta: float) -> float:
    """
    由扇形中心角计算圆段高度（矢高）。

    公式:
        h = r * (1 - cos(theta/2))

    几何推导：设弦AB对应的中心角为theta，C为圆心，Q为弦AB的中点，
    则 CQ = r*cos(theta/2)，矢高 h = r - CQ = r*(1 - cos(theta/2))。

    Parameters
    ----------
    r : float
        圆半径，r > 0。
    theta : float
        中心角（弧度）。

    Returns
    -------
    float
        圆段高度 h。
    """
    if r <= 0.0:
        raise ValueError("circle_segment_height_from_angle: r must be > 0")
    if theta < 0.0 or theta > 2.0 * np.pi:
        raise ValueError("circle_segment_height_from_angle: theta must be in [0, 2*pi]")
    h = r * (1.0 - np.cos(theta / 2.0))
    return float(h)


def circle_segment_centroid_from_angle(r: float, theta: float) -> tuple:
    """
    计算圆段区域的质心位置（相对于圆心，沿角平分线方向）。

    公式（解析推导）:
        对于中心角为 theta 的圆扇形区域，质心到圆心的距离为:
        d = (4 * r * sin^3(theta/2)) / (3 * (theta - sin(theta)))

    完整推导基于极坐标下的面积分:
        x_c = (1/A) * integral_{-theta/2}^{theta/2} integral_{0}^{r} (rho*cos(phi)) * rho drho dphi
        y_c = 0   （由对称性）

    Parameters
    ----------
    r : float
        圆半径。
    theta : float
        中心角（弧度）。

    Returns
    -------
    tuple
        (d, 0.0) 其中 d 为质心沿角平分线到圆心的距离。
    """
    if r <= 0.0:
        raise ValueError("circle_segment_centroid_from_angle: r must be > 0")
    if theta <= 0.0 or theta > 2.0 * np.pi:
        raise ValueError("circle_segment_centroid_from_angle: theta must be in (0, 2*pi]")
    num = 4.0 * r * (np.sin(theta / 2.0) ** 3)
    den = 3.0 * (theta - np.sin(theta))
    if abs(den) < 1e-15:
        return (0.0, 0.0)
    d = num / den
    return (float(d), 0.0)


def hexagonal_lattice_points(pitch: float, n_rings: int) -> np.ndarray:
    """
    生成六边形晶格上的格点坐标（二维）。

    六边形晶格由基矢:
        a1 = (pitch, 0)
        a2 = (pitch/2, pitch*sqrt(3)/2)
    张成。对于光子晶体光纤，空气孔中心位于这些格点上。

    Parameters
    ----------
    pitch : float
        晶格常数（孔间距），pitch > 0。
    n_rings : int
        环数（不包括中心孔），n_rings >= 1。

    Returns
    -------
    np.ndarray
        形状为 (N, 2) 的数组，N 为孔数。
    """
    if pitch <= 0.0:
        raise ValueError("hexagonal_lattice_points: pitch must be > 0")
    if n_rings < 1:
        raise ValueError("hexagonal_lattice_points: n_rings must be >= 1")
    points = []
    a1 = np.array([pitch, 0.0])
    a2 = np.array([pitch * 0.5, pitch * np.sqrt(3.0) / 2.0])
    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if abs(i) + abs(j) + abs(-i - j) <= 2 * n_rings:
                p = i * a1 + j * a2
                points.append(p)
    pts = np.array(points)
    # 去重
    pts = np.unique(np.round(pts, 12), axis=0)
    return pts


def pcf_air_holes_geometry(pitch: float, n_rings: int, hole_radius: float) -> dict:
    """
    计算光子晶体光纤空气孔阵列的几何参数。

    返回填充率 f、纤芯等效半径 r_core、圆段总面积等。

    公式:
        填充率 f = (N_holes * pi * r_hole^2) / A_unit_cell
        其中 A_unit_cell = (sqrt(3)/2) * pitch^2 为单位晶胞面积。

    Parameters
    ----------
    pitch : float
        晶格常数。
    n_rings : int
        空气孔环数。
    hole_radius : float
        空气孔半径。

    Returns
    -------
    dict
        包含几何参数的字典。
    """
    if hole_radius >= pitch / 2.0:
        raise ValueError("pcf_air_holes_geometry: hole_radius must be < pitch/2")
    points = hexagonal_lattice_points(pitch, n_rings)
    n_holes = points.shape[0]
    unit_cell_area = np.sqrt(3.0) / 2.0 * pitch * pitch
    hole_area = np.pi * hole_radius * hole_radius
    filling_fraction = n_holes * hole_area / (unit_cell_area * n_holes) if n_holes > 0 else 0.0
    # 等效纤芯半径（第一环内接圆）
    r_core = pitch - hole_radius
    # 石英基底占单位晶胞的圆段面积
    silica_fraction = 1.0 - filling_fraction
    return {
        "n_holes": int(n_holes),
        "pitch": float(pitch),
        "hole_radius": float(hole_radius),
        "filling_fraction": float(filling_fraction),
        "silica_fraction": float(silica_fraction),
        "r_core": float(r_core),
        "hole_centers": points,
    }


def triangle_grid(n: int, t: np.ndarray) -> np.ndarray:
    """
    在三角形区域 T 上生成规则网格点（重心坐标插值）。

    给定三角形顶点 T = [t0, t1, t2]（2x3数组），将每条边分成 n 等分，
    生成 ((n+1)*(n+2))/2 个网格点。

    网格点通过重心坐标生成:
        P = (i/n) * t0 + (j/n) * t1 + (k/n) * t2
    其中 i + j + k = n，i, j, k >= 0。

    Parameters
    ----------
    n : int
        每条边的分割数，n >= 1。
    t : np.ndarray
        形状为 (2, 3) 的顶点坐标数组。

    Returns
    -------
    np.ndarray
        形状为 (2, ng) 的网格点坐标，ng = ((n+1)*(n+2))/2。
    """
    if n < 1:
        raise ValueError("triangle_grid: n must be >= 1")
    if t.shape != (2, 3):
        raise ValueError("triangle_grid: t must have shape (2, 3)")
    ng = ((n + 1) * (n + 2)) // 2
    tg = np.zeros((2, ng))
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            tg[:, p] = (i * t[:, 0] + j * t[:, 1] + k * t[:, 2]) / n
            p += 1
    return tg


def pcf_transverse_triangle_grid(pitch: float, n_rings: int, subdivisions: int = 8) -> np.ndarray:
    """
    在光子晶体光纤横截面的一个代表性三角形单元上生成网格。

    PCF的六边形晶格可分解为等边三角形单元。选取一个基本三角形:
        t0 = (0, 0)
        t1 = (pitch, 0)
        t2 = (pitch/2, pitch*sqrt(3)/2)
    在其上生成网格点，用于后续横向模式计算。

    Parameters
    ----------
    pitch : float
        晶格常数。
    n_rings : int
        环数（仅用于边界检查）。
    subdivisions : int, optional
        三角形细分次数，默认 8。

    Returns
    -------
    np.ndarray
        网格点坐标，形状 (2, ng)。
    """
    t = np.array([
        [0.0, pitch, pitch * 0.5],
        [0.0, 0.0, pitch * np.sqrt(3.0) / 2.0]
    ])
    return triangle_grid(subdivisions, t)


def effective_mode_area(pitch: float, hole_radius: float, n_rings: int) -> float:
    """
    估算光子晶体光纤的有效模场面积 A_eff。

    经验公式（基于圆段几何修正）:
        A_eff = pi * r_core^2 * (1 - c1 * f + c2 * f^2)
    其中 f 为空气孔填充率，r_core 为等效纤芯半径，
    c1, c2 为拟合系数（源于Birks等人的实验数据）。

    严格定义基于光强分布的面积分:
        A_eff = (integral |E|^2 dA)^2 / (integral |E|^4 dA)
    此处采用几何近似以快速估算。

    Parameters
    ----------
    pitch : float
        晶格常数。
    hole_radius : float
        空气孔半径。
    n_rings : int
        环数。

    Returns
    -------
    float
        有效模场面积（um^2）。
    """
    geo = pcf_air_holes_geometry(pitch, n_rings, hole_radius)
    f = geo["filling_fraction"]
    r_core = geo["r_core"]
    c1 = 2.5
    c2 = 1.8
    a_eff = np.pi * r_core * r_core * (1.0 - c1 * f + c2 * f * f)
    return float(max(a_eff, 1e-15))


def nonlinear_coefficient(pitch: float, hole_radius: float, n_rings: int,
                          n2: float = 2.6e-20, wavelength: float = 1.55e-6) -> float:
    """
    计算光纤的非线性系数 gamma (W^{-1} m^{-1})。

    公式:
        gamma = (2 * pi / lambda) * (n2 / A_eff)

    其中:
        lambda 为真空波长，
        n2 为石英的非线性折射率（~2.6e-20 m^2/W），
        A_eff 为有效模场面积。

    Parameters
    ----------
    pitch : float
        晶格常数（单位 m）。
    hole_radius : float
        空气孔半径（单位 m）。
    n_rings : int
        环数。
    n2 : float
        非线性折射率（m^2/W）。
    wavelength : float
        工作波长（m）。

    Returns
    -------
    float
        非线性系数 gamma。
    """
    if wavelength <= 0.0:
        raise ValueError("nonlinear_coefficient: wavelength must be > 0")
    a_eff = effective_mode_area(pitch, hole_radius, n_rings)
    # a_eff 已经是 m^2
    gamma = (2.0 * np.pi / wavelength) * (n2 / a_eff)
    return float(gamma)
