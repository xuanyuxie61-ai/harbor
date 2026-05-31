"""
多铁性材料有限元网格生成模块
融合来源: 754_mesh_display (网格读取与索引处理) + 408_fem2d_poisson_rectangle (T6 二次元网格生成)

功能:
- 生成二维矩形区域上的六节点二次元 (T6) 三角形网格
- 节点编号、元素连通性、边界标记
- 带宽计算与网格质量评估
- 支持 Hilbert 曲线重排序接口
"""

import numpy as np
from typing import Tuple, Optional


class MultiferroicMesh:
    """
    多铁性模拟用的二维有限元网格。
    """

    def __init__(self, nx: int, ny: int,
                 xl: float = 0.0, xr: float = 1.0,
                 yb: float = 0.0, yt: float = 1.0):
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny 至少为 2")
        self.nx = nx
        self.ny = ny
        self.xl = xl
        self.xr = xr
        self.yb = yb
        self.yt = yt
        self.element_order = 6
        self.element_num = 2 * (nx - 1) * (ny - 1)
        self.node_num = (2 * nx - 1) * (2 * ny - 1)
        self.node_xy = self._generate_nodes()
        self.element_node = self._generate_elements()
        self.boundary_flags = self._mark_boundary_nodes()
        self.half_bandwidth = self._compute_half_bandwidth()

    def _generate_nodes(self) -> np.ndarray:
        """
        生成 T6 网格节点坐标。
        节点排列顺序:
        角点 -> 边中点 -> 面心点 的层次结构。
        """
        node_xy = np.zeros((self.node_num, 2), dtype=float)
        node = 0
        for j in range(2 * self.ny - 1):
            y = self.yb + j * (self.yt - self.yb) / (2 * self.ny - 2)
            for i in range(2 * self.nx - 1):
                x = self.xl + i * (self.xr - self.xl) / (2 * self.nx - 2)
                node_xy[node, 0] = x
                node_xy[node, 1] = y
                node += 1
        return node_xy

    def _generate_elements(self) -> np.ndarray:
        """
        生成 T6 三角形元素连通表。
        每个矩形单元划分为 2 个三角形，每个三角形 6 个节点。
        源自 fem2d_poisson_rectangle 中 grid_t6 的算法。
        """
        element_node = np.zeros((self.element_order, self.element_num), dtype=int)
        element = 0
        for j in range(1, self.ny):
            for i in range(1, self.nx):
                # 局部节点编号映射
                sw = (j - 1) * 2 * (2 * self.nx - 1) + 2 * i - 1
                w = sw + 1
                nw = sw + 2
                s = sw + 2 * self.nx - 1
                c = s + 1
                n = s + 2
                se = s + 2 * self.nx - 1
                e = se + 1
                ne = se + 2

                # 三角形 1 (0-based 索引)
                element_node[0, element] = sw - 1
                element_node[1, element] = se - 1
                element_node[2, element] = nw - 1
                element_node[3, element] = s - 1
                element_node[4, element] = c - 1
                element_node[5, element] = w - 1
                element += 1

                # 三角形 2 (0-based 索引)
                element_node[0, element] = ne - 1
                element_node[1, element] = nw - 1
                element_node[2, element] = se - 1
                element_node[3, element] = n - 1
                element_node[4, element] = c - 1
                element_node[5, element] = e - 1
                element += 1

        return element_node

    def _mark_boundary_nodes(self) -> np.ndarray:
        """
        标记边界节点: 1 表示边界，0 表示内部。
        源自 mesh_display 中 mesh_base_one 的边界检测思想。
        """
        flags = np.zeros(self.node_num, dtype=int)
        for node in range(self.node_num):
            x, y = self.node_xy[node]
            if (abs(x - self.xl) < 1e-12 or abs(x - self.xr) < 1e-12 or
                abs(y - self.yb) < 1e-12 or abs(y - self.yt) < 1e-12):
                flags[node] = 1
        return flags

    def _compute_half_bandwidth(self) -> int:
        """
        计算刚度矩阵的半带宽。
        源自 fem2d_poisson_rectangle 中 bandwidth 函数。
        """
        nhba = 0
        for e in range(self.element_num):
            for iln in range(self.element_order):
                i = self.element_node[iln, e]
                if i >= 0:
                    for jln in range(self.element_order):
                        j = self.element_node[jln, e]
                        nhba = max(nhba, abs(j - i))
        return nhba

    def element_area(self, element: int) -> float:
        """计算指定三角形元素的面积（基于三个角点）。"""
        n1 = self.element_node[0, element]
        n2 = self.element_node[1, element]
        n3 = self.element_node[2, element]
        x1, y1 = self.node_xy[n1]
        x2, y2 = self.node_xy[n2]
        x3, y3 = self.node_xy[n3]
        area = 0.5 * abs(y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2))
        return area

    def get_element_centroid(self, element: int) -> Tuple[float, float]:
        """返回元素质心坐标。"""
        xc = 0.0
        yc = 0.0
        for i in range(3):
            n = self.element_node[i, element]
            xc += self.node_xy[n, 0]
            yc += self.node_xy[n, 1]
        return xc / 3.0, yc / 3.0

    def apply_reordering(self, order: np.ndarray):
        """应用节点重排序。"""
        if len(order) != self.node_num:
            raise ValueError("重排序数组长度与节点数不符")
        old_to_new = np.empty(self.node_num, dtype=int)
        old_to_new[order] = np.arange(self.node_num, dtype=int)
        self.node_xy = self.node_xy[order]
        self.element_node = old_to_new[self.element_node]
        self.boundary_flags = self.boundary_flags[order]


def qbf_t6(x: float, y: float, element: int, inode: int,
           mesh: MultiferroicMesh) -> Tuple[float, float, float]:
    """
    评估 T6 二次基函数及其导数。
    源自 fem2d_poisson_rectangle 中 qbf 的算法，用于参考坐标到物理坐标的映射。

    返回:
        b:    基函数值
        dbdx: ∂b/∂x
        dbdy: ∂b/∂y
    """
    if not (0 <= element < mesh.element_num):
        raise IndexError("元素索引越界")
    if not (0 <= inode < 6):
        raise IndexError("局部节点索引越界")

    # 提取元素节点坐标
    xn = np.zeros(6, dtype=float)
    yn = np.zeros(6, dtype=float)
    for i in range(6):
        n = mesh.element_node[i, element]
        xn[i] = mesh.node_xy[n, 0]
        yn[i] = mesh.node_xy[n, 1]

    # 计算参考坐标 (r, s)
    det = (xn[1] - xn[0]) * (yn[2] - yn[0]) - (xn[2] - xn[0]) * (yn[1] - yn[0])
    if abs(det) < 1e-30:
        raise ValueError("退化元素，面积近似为零")

    r = ((yn[2] - yn[0]) * (x - xn[0]) + (xn[0] - xn[2]) * (y - yn[0])) / det
    s = ((yn[0] - yn[1]) * (x - xn[0]) + (xn[1] - xn[0]) * (y - yn[0])) / det

    drdx = (yn[2] - yn[0]) / det
    drdy = (xn[0] - xn[2]) / det
    dsdx = (yn[0] - yn[1]) / det
    dsdy = (xn[1] - xn[0]) / det

    # T6 二次基函数 (参考单元)
    if inode == 0:
        b = 2.0 * (1.0 - r - s) * (0.5 - r - s)
        dbdr = -3.0 + 4.0 * r + 4.0 * s
        dbds = -3.0 + 4.0 * r + 4.0 * s
    elif inode == 1:
        b = 2.0 * r * (r - 0.5)
        dbdr = -1.0 + 4.0 * r
        dbds = 0.0
    elif inode == 2:
        b = 2.0 * s * (s - 0.5)
        dbdr = 0.0
        dbds = -1.0 + 4.0 * s
    elif inode == 3:
        b = 4.0 * r * (1.0 - r - s)
        dbdr = 4.0 - 8.0 * r - 4.0 * s
        dbds = -4.0 * r
    elif inode == 4:
        b = 4.0 * r * s
        dbdr = 4.0 * s
        dbds = 4.0 * r
    elif inode == 5:
        b = 4.0 * s * (1.0 - r - s)
        dbdr = -4.0 * s
        dbds = 4.0 - 4.0 * r - 8.0 * s
    else:
        raise ValueError("无效局部节点索引")

    dbdx = dbdr * drdx + dbds * dsdx
    dbdy = dbdr * drdy + dbds * dsdy
    return b, dbdx, dbdy


def generate_quadrature_points(mesh: MultiferroicMesh, nq: int = 3):
    """
    为每个元素生成高斯求积点坐标与权重（基于三角形参考单元）。
    使用 3 点或 7 点高斯规则。
    """
    if nq == 3:
        # 3 点高斯规则，精度 2
        qr = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        qs = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        qw = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif nq == 7:
        # 7 点高斯规则，精度 5
        qr = np.array([1.0/3.0, 0.059715871789770, 0.797426985353087, 0.141430305474747,
                       0.059715871789770, 0.797426985353087, 0.141430305474747])
        qs = np.array([1.0/3.0, 0.059715871789770, 0.059715871789770, 0.797426985353087,
                       0.797426985353087, 0.141430305474747, 0.141430305474747])
        qw = np.array([0.1125, 0.066197076394253, 0.066197076394253, 0.066197076394253,
                       0.062969590272413, 0.062969590272413, 0.062969590272413])
    else:
        raise ValueError("仅支持 nq=3 或 nq=7")

    xq = np.zeros((nq, mesh.element_num))
    yq = np.zeros((nq, mesh.element_num))
    wq = np.zeros(nq)

    for q in range(nq):
        wq[q] = qw[q]
        for e in range(mesh.element_num):
            # 参考坐标到物理坐标（线性映射）
            n0 = mesh.element_node[0, e]
            n1 = mesh.element_node[1, e]
            n2 = mesh.element_node[2, e]
            x0, y0 = mesh.node_xy[n0]
            x1, y1 = mesh.node_xy[n1]
            x2, y2 = mesh.node_xy[n2]
            xq[q, e] = x0 + (x1 - x0) * qr[q] + (x2 - x0) * qs[q]
            yq[q, e] = y0 + (y1 - y0) * qr[q] + (y2 - y0) * qs[q]

    return wq, xq, yq
