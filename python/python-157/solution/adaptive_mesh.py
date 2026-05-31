"""
adaptive_mesh.py
爆轰波前自适应网格生成模块
融合来源：373_fem_basis_t3_display（有限元 T3 基函数与三角形面积计算）
           261_cvt_square_uniform（CVT 迭代点重分布）

用于在爆轰波前区域生成高分辨率三角形网格，
在反应区密集布点，在已燃区稀疏布点。
"""
import numpy as np
from combustion_utils import check_positive, check_nonnegative, cholesky_factor, solve_lower_triangular


def triangle_area(t):
    r"""
    计算三角形面积:
        area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
    融合来源：373_fem_basis_t3_display 中的 area 计算。
    """
    t = np.asarray(t, dtype=float)
    if t.shape != (2, 3):
        raise ValueError("t must be shape (2,3), got " + str(t.shape))
    area = abs(t[0, 0] * (t[1, 1] - t[1, 2]) +
               t[0, 1] * (t[1, 2] - t[1, 0]) +
               t[0, 2] * (t[1, 0] - t[1, 1]))
    if area < 1.0e-14:
        return 0.0
    return area * 0.5


def basis_t3(t, i, p):
    r"""
    T3（线性三角形）基函数在点 p 处的值与导数。
    融合来源：373_fem_basis_t3_display 的 basis_11_t3。

    输入:
        t: (2,3) 三角形顶点坐标
        i: 节点索引 (0, 1, 2)
        p: (2,) 评估点坐标
    返回:
        phi, dphi_dx, dphi_dy
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    area = triangle_area(t)
    if area <= 0.0:
        raise ValueError("Triangle has zero or negative area")
    if i not in (0, 1, 2):
        raise ValueError("Node index i must be 0, 1, or 2")

    ip1 = (i + 1) % 3
    ip2 = (i + 2) % 3

    phi = ((t[0, ip2] - t[0, ip1]) * (p[1] - t[1, ip1]) -
           (t[1, ip2] - t[1, ip1]) * (p[0] - t[0, ip1])) / (2.0 * area)

    dphi_dx = -(t[1, ip2] - t[1, ip1]) / (2.0 * area)
    dphi_dy = (t[0, ip2] - t[0, ip1]) / (2.0 * area)
    return phi, dphi_dx, dphi_dy


def sample_square_uniform(n):
    r"""
    在单位正方形 [0,1]^2 内均匀随机采样 n 个点。
    融合来源：261_cvt_square_uniform 的 square_uniform。
    """
    return np.random.rand(n, 2)


def cvt_iteration(points, n_samples=1000, n_iter=20):
    r"""
    对正方形域内的点进行 CVT（Centroidal Voronoi Tessellation）迭代优化。
    融合来源：261_cvt_square_uniform。

    算法:
        1. 在域内生成大量样本点
        2. 对每个样本点，找到最近的生成点（generator）
        3. 将每个生成点更新为其 Voronoi 单元的质心
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    check_positive(n, "n_points")
    check_positive(n_samples, "n_samples")
    check_positive(n_iter, "n_iter")

    for it in range(n_iter):
        samples = sample_square_uniform(n_samples)
        # 对每个样本找到最近点
        counts = np.zeros(n)
        centroids = np.zeros((n, 2))
        for s in samples:
            dists = np.sum((points - s) ** 2, axis=1)
            j = np.argmin(dists)
            counts[j] += 1
            centroids[j] += s
        # 更新生成点位置
        for j in range(n):
            if counts[j] > 0:
                points[j] = centroids[j] / counts[j]
            else:
                # 若某点无样本，随机重置
                points[j] = np.random.rand(2)
    return points


def adaptive_density_function(x, y, wave_x=0.5, wave_width=0.05,
                              max_density=10.0, min_density=1.0):
    r"""
    自适应密度函数：在波前附近密度高，远离波前密度低。

        rho_mesh(x,y) = min_density + (max_density - min_density) * exp(-((x-wave_x)/wave_width)^2)
    """
    dx = x - wave_x
    gauss = np.exp(-(dx / wave_width) ** 2)
    return min_density + (max_density - min_density) * gauss


class AdaptiveDetonationMesh:
    r"""
    爆轰波前自适应三角形网格生成器。

    在波前附近（反应区）生成密集网格，利用 CVT 迭代将节点
    按自适应密度函数重新分布，然后使用 Delaunay 三角化。
    """

    def __init__(self, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                 n_base=400, wave_x=0.5, wave_width=0.05):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_base = n_base
        self.wave_x = wave_x
        self.wave_width = wave_width

    def generate(self, cvt_samples=2000, cvt_iter=15):
        r"""
        生成自适应网格节点与三角形单元。
        返回:
            nodes: (n_nodes, 2) 节点坐标
            elements: (n_elem, 3) 三角形单元节点索引
        """
        # 初始随机分布
        points = sample_square_uniform(self.n_base)

        # 基于密度函数进行拒绝采样预分布
        accepted = []
        max_trials = self.n_base * 20
        trial = 0
        while len(accepted) < self.n_base and trial < max_trials:
            trial += 1
            p = np.random.rand(2)
            x = self.x_min + p[0] * (self.x_max - self.x_min)
            y = self.y_min + p[1] * (self.y_max - self.y_min)
            dens = adaptive_density_function(
                x, y, self.wave_x, self.wave_width,
                max_density=10.0, min_density=1.0
            )
            if np.random.rand() < dens / 10.0:
                accepted.append([x, y])

        if len(accepted) < self.n_base // 2:
            # 若拒绝采样不足，回退到均匀分布
            accepted = sample_square_uniform(self.n_base).tolist()

        points = np.array(accepted[:self.n_base])

        # CVT 迭代优化节点分布
        points = cvt_iteration(points, n_samples=cvt_samples, n_iter=cvt_iter)

        # 映射回物理域
        points[:, 0] = self.x_min + points[:, 0] * (self.x_max - self.x_min)
        points[:, 1] = self.y_min + points[:, 1] * (self.y_max - self.y_min)

        # 简单三角化：对节点按 x 排序后生成三角形
        # 这里采用简化方法：将正方形划分为小方块后对角线分割
        # 实际应用中应使用 Delaunay，但为减少外部依赖，使用简化网格生成
        nodes = points
        elements = self._simple_triangulation(nodes)
        return nodes, elements

    def _simple_triangulation(self, nodes):
        r"""
        基于节点近邻的简化三角化方法。
        将空间划分为网格单元，在每个单元内生成三角形。
        这是一种简化实现，保持代码自包含性。
        """
        n = nodes.shape[0]
        # 计算平均邻近距离
        if n < 3:
            return np.zeros((0, 3), dtype=int)

        # 使用 k-d 树思想：对每个节点找两个最近邻构成三角形
        elements = []
        used = set()
        for i in range(n):
            dists = np.sum((nodes - nodes[i]) ** 2, axis=1)
            dists[i] = np.inf
            # 找最近的两个不同节点
            j = np.argmin(dists)
            dists[j] = np.inf
            k = np.argmin(dists)
            tri = tuple(sorted((i, j, k)))
            if tri not in used:
                # 检查三角形面积是否非零
                t = nodes[[i, j, k]].T
                area = triangle_area(t)
                if area > 1.0e-12:
                    used.add(tri)
                    elements.append([i, j, k])
            if len(elements) >= 2 * n:
                break

        if len(elements) == 0:
            return np.zeros((0, 3), dtype=int)
        return np.array(elements, dtype=int)

    def element_quality(self, nodes, elements):
        r"""
        计算网格质量指标：最小角与最大角之比。
        返回平均质量（1.0 为等边三角形，0.0 为退化）。
        """
        qualities = []
        for elem in elements:
            t = nodes[elem].T
            area = triangle_area(t)
            if area < 1.0e-14:
                qualities.append(0.0)
                continue
            # 计算三边长
            a = np.linalg.norm(nodes[elem[1]] - nodes[elem[0]])
            b = np.linalg.norm(nodes[elem[2]] - nodes[elem[1]])
            c = np.linalg.norm(nodes[elem[0]] - nodes[elem[2]])
            # 使用内接圆半径 / 外接圆半径 作为质量度量
            # r_in = 2*Area / (a+b+c)
            # R_circ = a*b*c / (4*Area)
            # quality = 2 * r_in / R_circ = 8*Area^2 / ((a+b+c)*a*b*c)
            denom = (a + b + c) * a * b * c
            if denom < 1.0e-14:
                qualities.append(0.0)
            else:
                q = 8.0 * area * area / denom
                qualities.append(min(q, 1.0))
        return np.mean(qualities) if qualities else 0.0
