"""
 monte_carlo_utils.py
 
 融合种子项目:
   - 501_hand_area: 多边形面积计算、点在多边形内判断
   - 713_maple_area: 网格法、蒙特卡洛、准蒙特卡洛面积估计
   - 952_quadrilateral: 四边形面积、点在四边形内、凸性判断
 
 科学应用:
   在全波形反演中用于复杂地质构造截面的面积估计、射线-单元相交测试、
   以及有限元网格单元的质量检查。
"""

import numpy as np


def polygon_area_2d(vertices):
    """
    计算二维多边形的有向面积。
    
    公式: A = 0.5 * sum_{i=1}^{N} x_i * (y_{i+1} - y_{i-1})
    其中下标循环（N+1 -> 1, 0 -> N）。
    
    Parameters
    ----------
    vertices : ndarray, shape (N, 2)
        多边形顶点，按顺时针或逆时针排列。
    
    Returns
    -------
    area : float
        有向面积（逆时针为正，顺时针为负）。
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must be of shape (N, 2)")
    n = vertices.shape[0]
    if n < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.sum(x * (np.roll(y, -1) - np.roll(y, 1)))
    return area


def polygon_contains_point_2d(vertices, point):
    """
    判断二维点是否在多边形内部（射线交叉法）。
    
    基于 Jordan 曲线定理，计算从点出发的水平射线与多边形边界的交叉次数。
    若为奇数次则在内部，偶数次则在外部。
    
    Parameters
    ----------
    vertices : ndarray, shape (N, 2)
        多边形顶点。
    point : ndarray, shape (2,)
        待测试点。
    
    Returns
    -------
    inside : bool
        True 如果在内部。
    """
    vertices = np.asarray(vertices, dtype=float)
    point = np.asarray(point, dtype=float)
    n = vertices.shape[0]
    inside = False
    x, y = point[0], point[1]
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        # 检查边是否跨越水平射线 y
        if ((y1 > y) != (y2 > y)):
            # 计算交点 x 坐标
            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if xinters > x:
                inside = not inside
    return inside


def quadrilateral_area(quad):
    """
    计算四边形面积（分解为两个三角形）。
    
    Parameters
    ----------
    quad : ndarray, shape (4, 2)
        四边形四个顶点。
    
    Returns
    -------
    area : float
        四边形面积（对非凸四边形也适用）。
    """
    quad = np.asarray(quad, dtype=float)
    if quad.shape != (4, 2):
        raise ValueError("quad must be of shape (4, 2)")
    # 三角形1: 顶点 0, 1, 2
    t1 = 0.5 * abs(
        quad[0, 0] * (quad[1, 1] - quad[2, 1]) +
        quad[1, 0] * (quad[2, 1] - quad[0, 1]) +
        quad[2, 0] * (quad[0, 1] - quad[1, 1])
    )
    # 三角形2: 顶点 0, 2, 3
    t2 = 0.5 * abs(
        quad[0, 0] * (quad[2, 1] - quad[3, 1]) +
        quad[2, 0] * (quad[3, 1] - quad[0, 1]) +
        quad[3, 0] * (quad[0, 1] - quad[2, 1])
    )
    return t1 + t2


def quadrilateral_is_convex(quad):
    """
    判断四边形是否为凸四边形。
    
    通过计算四个内角，检查是否都小于 pi 且总和为 2*pi。
    
    Parameters
    ----------
    quad : ndarray, shape (4, 2)
        四边形顶点。
    
    Returns
    -------
    is_convex : bool
        True 如果为凸四边形。
    """
    quad = np.asarray(quad, dtype=float)
    angles = quadrilateral_angles(quad)
    angle_sum = np.sum(angles)
    return (
        np.all(angles > 0.0) and
        np.all(angles < np.pi) and
        abs(angle_sum - 2.0 * np.pi) < 1.0
    )


def quadrilateral_angles(quad):
    """
    计算四边形的四个内角（弧度）。
    
    Parameters
    ----------
    quad : ndarray, shape (4, 2)
        四边形顶点。
    
    Returns
    -------
    angles : ndarray, shape (4,)
        四个内角。
    """
    quad = np.asarray(quad, dtype=float)
    angles = np.zeros(4)
    for i in range(4):
        p1 = quad[(i - 1) % 4]
        p2 = quad[i]
        p3 = quad[(i + 1) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        # 边界处理：零向量
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-14 or norm2 < 1e-14:
            angles[i] = 0.0
            continue
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    return angles


def quadrilateral_contains_point(quad, point):
    """
    判断点是否在凸四边形内部（角度法）。
    
    Parameters
    ----------
    quad : ndarray, shape (4, 2)
        凸四边形顶点。
    point : ndarray, shape (2,)
        待测试点。
    
    Returns
    -------
    inside : bool
        True 如果在内部。
    """
    quad = np.asarray(quad, dtype=float)
    point = np.asarray(point, dtype=float)
    # 如果四边形非凸，退化为多边形测试
    if not quadrilateral_is_convex(quad):
        return polygon_contains_point_2d(quad, point)
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        p3 = quad[(i + 2) % 4]
        # 计算四边形在顶点 p2 处的内角
        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-14 or norm2 < 1e-14:
            return False
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_quad = np.arccos(cos_angle)
        # 计算点与两条边形成的夹角
        w1 = p1 - p2
        w2 = point - p2
        norm_w1 = np.linalg.norm(w1)
        norm_w2 = np.linalg.norm(w2)
        if norm_w1 < 1e-14 or norm_w2 < 1e-14:
            return False
        cos_p = np.dot(w1, w2) / (norm_w1 * norm_w2)
        cos_p = np.clip(cos_p, -1.0, 1.0)
        angle_p = np.arccos(cos_p)
        if angle_quad < angle_p:
            return False
    return True


def area_estimate_mc(boundary, width, height, sample_num, rng=None):
    """
    使用蒙特卡洛方法估计区域面积比例。
    
    公式: A_est = (N_inside / N_total) * A_bbox
    
    Parameters
    ----------
    boundary : ndarray, shape (M, 2)
        区域边界顶点。
    width : float
        包围盒宽度。
    height : float
        包围盒高度。
    sample_num : int
        采样点数。
    rng : numpy.random.Generator, optional
        随机数生成器。
    
    Returns
    -------
    estimate : float
        面积比例估计值 [0, 1]。
    """
    if rng is None:
        rng = np.random.default_rng()
    if sample_num <= 0:
        return 0.0
    x = width * rng.random(sample_num)
    y = height * rng.random(sample_num)
    inside_count = 0
    for i in range(sample_num):
        if polygon_contains_point_2d(boundary, np.array([x[i], y[i]])):
            inside_count += 1
    return inside_count / sample_num


def area_estimate_grid(boundary, width, height, n_grid):
    """
    使用均匀网格法估计区域面积比例。
    
    Parameters
    ----------
    boundary : ndarray, shape (M, 2)
        区域边界顶点。
    width : float
        包围盒宽度。
    height : float
        包围盒高度。
    n_grid : int
        每维网格点数。
    
    Returns
    -------
    estimate : float
        面积比例估计值 [0, 1]。
    """
    if n_grid <= 0:
        return 0.0
    dx = width / (n_grid + 1)
    dy = height / (n_grid + 1)
    xlo = 0.5 * dx
    xhi = width - 0.5 * dx
    ylo = 0.5 * dy
    yhi = height - 0.5 * dy
    if n_grid == 1:
        gx = np.array([0.5 * (xlo + xhi)])
        gy = np.array([0.5 * (ylo + yhi)])
    else:
        gx = np.linspace(xlo, xhi, n_grid)
        gy = np.linspace(ylo, yhi, n_grid)
    XG, YG = np.meshgrid(gx, gy)
    inside = np.zeros_like(XG, dtype=bool)
    for i in range(n_grid):
        for j in range(n_grid):
            inside[j, i] = polygon_contains_point_2d(
                boundary, np.array([XG[j, i], YG[j, i]])
            )
    return np.sum(inside) / (n_grid * n_grid)


def hammersley_sequence(i1, i2, m, n_base):
    """
    生成 Hammersley 准随机序列。
    
    Hammersley 序列定义:
      r_1(k) = k / N
      r_j(k) = sum_{l} (d_l * p_j^{-l-1})
    其中 d_l 是 k 在素数 p_j 进制下的各位数字。
    
    Parameters
    ----------
    i1, i2 : int
        序列起始和结束索引。
    m : int
        空间维度。
    n_base : int
        第一维的基数。
    
    Returns
    -------
    r : ndarray, shape (m, abs(i2-i1)+1)
        Hammersley 序列点。
    """
    primes = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
        353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
        419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
        467, 479, 487, 491, 499, 503, 509, 521, 523, 541
    ], dtype=int)
    if n_base <= 0:
        n_base = 1
    step = 1 if i1 <= i2 else -1
    l = abs(i2 - i1) + 1
    r = np.zeros((m, l))
    k_idx = 0
    for i in range(i1, i2 + step, step):
        r[0, k_idx] = (i % (n_base + 1)) / n_base
        t = np.full(m - 1, i, dtype=int)
        prime_inv = 1.0 / primes[:m - 1].astype(float)
        while np.any(t != 0):
            for j in range(m - 1):
                d = int(t[j] % primes[j])
                r[j + 1, k_idx] += d * prime_inv[j]
                prime_inv[j] /= primes[j]
                t[j] = t[j] // primes[j]
        k_idx += 1
    return r


def area_estimate_qmc(boundary, width, height, sample_num):
    """
    使用准蒙特卡洛 (Hammersley 序列) 估计区域面积比例。
    
    Parameters
    ----------
    boundary : ndarray, shape (M, 2)
        区域边界顶点。
    width : float
        包围盒宽度。
    height : float
        包围盒高度。
    sample_num : int
        采样点数。
    
    Returns
    -------
    estimate : float
        面积比例估计值 [0, 1]。
    """
    if sample_num <= 0:
        return 0.0
    seq = hammersley_sequence(0, sample_num - 1, 2, sample_num - 1)
    x = width * seq[0, :]
    y = height * seq[1, :]
    inside_count = 0
    for i in range(sample_num):
        if polygon_contains_point_2d(boundary, np.array([x[i], y[i]])):
            inside_count += 1
    return inside_count / sample_num
