"""
cochlea_geometry.py
===================
人工耳蜗几何建模模块

基于种子项目:
  - 1210_test_interp: 插值方法用于患者个性化耳蜗轮廓重建
  - 150_cg_lab_triangles: 点到线有符号距离用于电极-蜗轴距离计算
  - 490_grf_io: 图结构 I/O 思想用于神经纤维连接拓扑

科学背景:
  耳蜗呈蜗牛壳状螺旋结构，常用参数化方程描述其中心线（modiolar axis）。
  本模块提供:
    1) 基于插值的个性化耳蜗轮廓重建
    2) 电极阵列到蜗轴的有符号距离计算
    3) 螺旋神经节神经元(SGN)的图拓扑连接
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


class CochleaGeometry:
    """
    人工耳蜗三维几何模型。

    耳蜗中心线参数方程（对数螺旋近似）:
        r(θ) = r_0 * exp(-b * θ)
        x(θ) = r(θ) * cos(θ)
        y(θ) = r(θ) * sin(θ)

    其中 r_0 为基底起始半径，b 为螺旋紧缩系数。
    """

    def __init__(self, r0=3.5, b=0.15, theta_max=4.5 * np.pi,
                 scala_height=1.2, scala_width=2.0):
        """
        Parameters
        ----------
        r0 : float
            基底起始半径 (mm)
        b : float
            螺旋紧缩系数，典型值 0.15
        theta_max : float
            最大螺旋角 (rad)，约 2.25 圈
        scala_height : float
            鼓阶/前庭阶高度 (mm)
        scala_width : float
            鼓阶/前庭阶宽度 (mm)
        """
        self.r0 = float(r0)
        self.b = float(b)
        self.theta_max = float(theta_max)
        self.scala_height = float(scala_height)
        self.scala_width = float(scala_width)
        self._centerline = None
        self._normals = None
        self._build_centerline()

    def _build_centerline(self, n_points=400):
        """构建蜗轴中心线及其法向。"""
        theta = np.linspace(0.0, self.theta_max, n_points)
        r = self.r0 * np.exp(-self.b * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # 中心线切向量
        dr_dtheta = -self.b * r
        dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
        dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)
        norm = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
        norm = np.where(norm < 1e-14, 1.0, norm)
        tangent = np.column_stack((dx_dtheta / norm, dy_dtheta / norm))

        # 法向量 (指向外侧)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

        self._centerline = {
            'theta': theta,
            'r': r,
            'points': np.column_stack((x, y)),
            'tangent': tangent,
            'normal': normal,
        }

    def centerline_at(self, theta):
        """给定 θ，返回中心线坐标。"""
        r = self.r0 * np.exp(-self.b * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x, y])

    def signed_distance_to_modiolar_axis(self, points):
        """
        计算点到蜗轴中心线的有符号距离。

        符号规则: 正值表示点在法向量指向的一侧（外侧），
                  负值表示点在另一侧（内侧）。

        Parameters
        ----------
        points : ndarray, shape (N, 2)
            查询点坐标 (mm)

        Returns
        -------
        distances : ndarray, shape (N,)
            有符号距离 (mm)
        closest_idx : ndarray, shape (N,)
            最近中心线索引
        """
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be of shape (N, 2)")

        cl = self._centerline['points']
        # 广播计算距离
        diffs = cl[np.newaxis, :, :] - points[:, np.newaxis, :]  # (N, M, 2)
        dists_sq = np.sum(diffs**2, axis=2)
        closest_idx = np.argmin(dists_sq, axis=1)

        # 计算到最近线段的有符号距离
        n_points = points.shape[0]
        signed_dists = np.empty(n_points)
        for i in range(n_points):
            idx = closest_idx[i]
            p = points[i]
            p1 = cl[idx]
            if idx + 1 < len(cl):
                p2 = cl[idx + 1]
            else:
                p2 = cl[idx]

            # 线段方向向量
            dv = p2 - p1
            dv_norm = np.linalg.norm(dv)
            if dv_norm < 1e-14:
                dist_vec = p - p1
                signed_dists[i] = np.linalg.norm(dist_vec)
                continue

            # 单位法向量
            nv = np.array([-dv[1], dv[0]]) / dv_norm
            # 有符号距离 = 法向量 · (点 - 线段端点)
            signed_dists[i] = np.dot(nv, p - p1)

        return signed_dists, closest_idx

    def interpolate_patient_geometry(self, known_thetas, known_radii):
        """
        基于已知数据点插值重建患者个性化耳蜗轮廓。

        基于种子 1210_test_interp 的插值思想，使用三次样条。

        Parameters
        ----------
        known_thetas : ndarray
            已知测量点的 θ 值
        known_radii : ndarray
            对应测量点的 r 值

        Returns
        -------
        cs : CubicSpline
            插值后的 r(θ) 样条函数
        """
        known_thetas = np.asarray(known_thetas, dtype=float)
        known_radii = np.asarray(known_radii, dtype=float)
        if len(known_thetas) < 4:
            raise ValueError("至少需要 4 个数据点进行三次样条插值")
        if not np.all(np.diff(known_thetas) > 0):
            raise ValueError("known_thetas 必须严格递增")
        if np.any(known_radii <= 0):
            raise ValueError("半径必须为正")

        cs = CubicSpline(known_thetas, known_radii)
        return cs

    def build_sgn_graph(self, n_neurons=200):
        """
        构建螺旋神经节神经元(SGN)的图拓扑。

        基于种子 490_grf_io 的图结构思想。
        SGN 沿蜗轴呈带状分布，每个神经元与邻近神经元有侧向连接。

        Returns
        -------
        nodes : ndarray, shape (n_neurons, 2)
            神经元胞体位置
        edges : list of tuple
            神经元间连接边 (i, j)
        edge_weights : ndarray
            连接权重（距离倒数）
        """
        theta = np.linspace(0.0, self.theta_max, n_neurons)
        r = self.r0 * np.exp(-self.b * theta)
        # 神经元位于骨螺旋板(bony shelf)上，距离蜗轴约 0.5 mm 外侧
        offset = 0.5
        nodes = np.column_stack((
            (r + offset) * np.cos(theta),
            (r + offset) * np.sin(theta)
        ))

        edges = []
        edge_weights = []
        for i in range(n_neurons):
            # 每个神经元连接到其前后邻居（纵向）
            for j in [i - 1, i + 1]:
                if 0 <= j < n_neurons:
                    edges.append((i, j))
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    w = 1.0 / max(dist, 1e-6)
                    edge_weights.append(w)

        return nodes, edges, np.array(edge_weights)

    def get_scala_cross_section(self, theta):
        """
        获取给定 θ 处的鼓阶横截面轮廓（近似矩形）。

        Returns
        -------
        polygon : ndarray, shape (4, 2)
            横截面四边形顶点
        """
        center = self.centerline_at(theta)
        # 法向量（外侧）
        dr = -self.b * self.r0 * np.exp(-self.b * theta)
        r_val = self.r0 * np.exp(-self.b * theta)
        dx = dr * np.cos(theta) - r_val * np.sin(theta)
        dy = dr * np.sin(theta) + r_val * np.cos(theta)
        tangent = np.array([dx, dy])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-14)
        normal = np.array([-tangent[1], tangent[0]])

        # 矩形: 沿法向 ±width/2, 垂直方向 ±height/2
        perp = np.array([-normal[1], normal[0]])
        polygon = np.array([
            center + normal * self.scala_width / 2 + perp * self.scala_height / 2,
            center + normal * self.scala_width / 2 - perp * self.scala_height / 2,
            center - normal * self.scala_width / 2 - perp * self.scala_height / 2,
            center - normal * self.scala_width / 2 + perp * self.scala_height / 2,
        ])
        return polygon
