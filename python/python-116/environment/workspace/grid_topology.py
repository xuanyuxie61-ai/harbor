"""
grid_topology.py
膜表面网格拓扑与边界追踪模块

本模块为脂质双分子层模拟生成各类计算网格（矩形、极坐标、三角），
并追踪膜边界的等边多边形近似（受种子项目 107_boundary_word_equilateral
与 905_pram 的边界词启发）。

参考种子项目: 492_gridlines (极坐标/矩形/三角网格)
                107_boundary_word_equilateral (等边多边形边界词)
                905_pram (PRAM 多连方边界词)

物理背景:
    脂质双分子层在热涨落下可形成局部曲率，需要用三角网格离散
    中 surface（mid-surface）。对于平面双层，采用矩形格点；
    对于囊泡或管状结构，采用极坐标/柱坐标网格。
    边界追踪用于定义模拟区域的拓扑边界，例如定义水通道区域
    或膜的缺陷边缘。
"""

import numpy as np


class GridGenerator:
    """
    多类型计算网格生成器。
    """

    @staticmethod
    def rectangular_grid(nx, ny, xlim=(-5.0, 5.0), ylim=(-5.0, 5.0)):
        """
        生成矩形网格。

        Returns
        -------
        X, Y : ndarray, shape (nx, ny)
            网格坐标。
        dx, dy : float
            网格间距。
        """
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny 必须至少为 2。")
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = (xlim[1] - xlim[0]) / (nx - 1) if nx > 1 else 1.0
        dy = (ylim[1] - ylim[0]) / (ny - 1) if ny > 1 else 1.0
        return X, Y, dx, dy

    @staticmethod
    def polar_grid(r_min, r_max, nr, n_ang, center=(0.0, 0.0)):
        """
        生成极坐标网格（用于囊泡或管状双层模拟）。

        Parameters
        ----------
        r_min, r_max : float
            径向范围（单位: nm）。
        nr : int
            径向层数。
        n_ang : int
            角向分割数。
        center : tuple
            极坐标中心。

        Returns
        -------
        R, Theta : ndarray, shape (nr, n_ang)
            极坐标网格。
        X, Y : ndarray
            对应的笛卡尔坐标。
        """
        if r_min < 0 or r_max <= r_min:
            raise ValueError("径向范围无效。")
        if nr < 1 or n_ang < 3:
            raise ValueError("nr ≥ 1, n_ang ≥ 3。")

        r = np.linspace(r_min, r_max, nr)
        theta = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        X = center[0] + R * np.cos(Theta)
        Y = center[1] + R * np.sin(Theta)
        return R, Theta, X, Y

    @staticmethod
    def triangular_grid(nx, ny, xlim=(-5.0, 5.0), ylim=(-5.0, 5.0)):
        """
        生成三角网格（通过矩形网格的对角线分割）。

        Returns
        -------
        nodes : ndarray, shape (N, 2)
            节点坐标列表。
        triangles : ndarray, shape (M, 3)
            每个三角形由三个节点索引组成。
        """
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny 必须至少为 2。")
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        nodes = []
        node_id = {}
        idx = 0
        for i in range(nx):
            for j in range(ny):
                nodes.append([X[i, j], Y[i, j]])
                node_id[(i, j)] = idx
                idx += 1
        nodes = np.array(nodes)

        triangles = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                n00 = node_id[(i, j)]
                n10 = node_id[(i + 1, j)]
                n01 = node_id[(i, j + 1)]
                n11 = node_id[(i + 1, j + 1)]
                # 对角线分割为两个三角形
                triangles.append([n00, n10, n11])
                triangles.append([n00, n11, n01])

        return nodes, np.array(triangles, dtype=int)


class BoundaryTracer:
    """
    膜边界追踪器。

    受种子项目 107_boundary_word_equilateral 和 905_pram 启发，
    使用六方向步进（等边三角晶格）或四方向步进（方晶格）追踪边界。

    六方向编码（等边三角晶格）:
        0: ( 0, +1)   北
        1: (+1,  0)   东
        2: (+1, -1)   东南
        3: ( 0, -1)   南
        4: (-1,  0)   西
        5: (-1, +1)   西北

    边界词: 一系列方向编码，描述从起点出发沿边界行走的路径。
    """

    HEX_STEPS = np.array([
        [0, 1], [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1]
    ], dtype=int)

    SQR_STEPS = np.array([
        [0, 1], [1, 0], [0, -1], [-1, 0]
    ], dtype=int)

    def __init__(self, grid_type='hex'):
        if grid_type not in ('hex', 'square'):
            raise ValueError("grid_type 必须是 'hex' 或 'square'。")
        self.grid_type = grid_type
        self.steps = self.HEX_STEPS if grid_type == 'hex' else self.SQR_STEPS

    def trace_boundary(self, mask, start_ij):
        """
        从起点出发，沿 mask 的边界追踪生成边界词。

        Parameters
        ----------
        mask : ndarray, shape (nx, ny), bool
            True 表示膜区域内部。
        start_ij : tuple (i, j)
            起始格点。

        Returns
        -------
        boundary_word : list of int
            方向编码序列。
        path : ndarray, shape (L, 2)
            边界路径坐标。
        """
        mask = np.asarray(mask, dtype=bool)
        nx, ny = mask.shape
        i, j = start_ij
        if not (0 <= i < nx and 0 <= j < ny):
            raise ValueError("起点超出网格范围。")
        if not mask[i, j]:
            raise ValueError("起点必须位于膜区域内。")

        boundary_word = []
        path = [(i, j)]
        visited = set()
        visited.add((i, j))

        # 简单的边界追踪：总是尝试顺时针沿边界走
        current_dir = 0
        max_steps = 4 * nx * ny
        for _ in range(max_steps):
            found = False
            for d in range(len(self.steps)):
                nd = (current_dir + d) % len(self.steps)
                di, dj = self.steps[nd]
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    if (ni, nj) not in visited or len(path) > 2:
                        boundary_word.append(int(nd))
                        i, j = ni, nj
                        path.append((i, j))
                        visited.add((i, j))
                        current_dir = (nd - 1) % len(self.steps)
                        found = True
                        break
            if not found or (len(path) > 2 and path[-1] == path[0]):
                break

        return boundary_word, np.array(path, dtype=int)

    def word_to_polygon(self, word, start=(0, 0)):
        """
        将边界词转换为多边形顶点坐标。
        """
        pts = [np.array(start, dtype=float)]
        p = np.array(start, dtype=float)
        for d in word:
            if d < 0 or d >= len(self.steps):
                continue
            p = p + self.steps[d].astype(float)
            pts.append(p.copy())
        return np.array(pts)

    def pram_like_boundary(self, scale=1.0):
        """
        生成 PRAM 风格的边界词（种子项目 905_pram 的扩展）。
        描述一个近矩形的多连方区域，带有圆角修正。

        物理意义: 在双层膜中，一个局部畴（domain）的边界形状。
        """
        # PRAM 近似边界词: 8步东 + 8步北 + 14步西 + ...
        # 简化为一个六边形近似
        word = (
            [1] * (8 * scale) +
            [0] * (8 * scale) +
            [4] * (14 * scale) +
            [3] * (8 * scale) +
            [2] * (4 * scale) +
            [5] * (4 * scale)
        )
        return [int(w) for w in word]

    def equilateral_triangle_boundary(self, side_length=5):
        """
        等边三角形边界词（种子项目 107_boundary_word_equilateral 的扩展）。
        用于描述膜上的三角形缺陷或纳米畴。
        """
        if side_length <= 0:
            raise ValueError("边长必须为正。")
        # 在六方格上构造等边三角形
        word = []
        for _ in range(side_length):
            word.extend([1, 0])  # 东-北
        for _ in range(side_length):
            word.extend([4, 3])  # 西-南
        for _ in range(side_length):
            word.extend([2, 2])  # 东南-东南
        return word

    def compute_perimeter_and_area(self, word):
        """
        由边界词估算周长与面积。

        对于六方格，Pick 定理推广:
            A = (1/2) Σ (x_i y_{i+1} - x_{i+1} y_i)
        """
        poly = self.word_to_polygon(word)
        if len(poly) < 3:
            return 0.0, 0.0
        # 周长
        diffs = np.diff(poly, axis=0, append=poly[:1])
        perimeter = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
        # 面积（鞋带公式）
        x = poly[:, 0]
        y = poly[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return float(perimeter), float(area)


def membrane_surface_metric(nodes, triangles):
    """
    计算三角网格上的第一基本形式度量（度量张量）。

    对于参数曲面 r(u,v)，第一基本形式:
        ds² = E du² + 2F dudv + G dv²
    其中 E = r_u·r_u, F = r_u·r_v, G = r_v·r_v。

    本函数对每个三角形计算局部 (E, F, G) 并平均。
    """
    if len(triangles) == 0:
        return 1.0, 0.0, 1.0

    E_total = F_total = G_total = 0.0
    for tri in triangles:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        # 局部基向量
        ru = p1 - p0
        rv = p2 - p0
        E_total += np.dot(ru, ru)
        F_total += np.dot(ru, rv)
        G_total += np.dot(rv, rv)

    n_tri = len(triangles)
    return E_total / n_tri, F_total / n_tri, G_total / n_tri
