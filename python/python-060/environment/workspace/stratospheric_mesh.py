# -*- coding: utf-8 -*-
"""
stratospheric_mesh.py
平流层三维网格生成与优化模块。

核心思想：在平流层区域（经度 λ∈[0,2π], 纬度 φ∈[-π/2,π/2], 高度 z∈[15,50] km）
上构造非结构化网格，采用 Delaunay 三角剖分（球面投影层）和 Centroidal Voronoi
Tessellation (CVT) 迭代优化。

融合来源：
  - 469_geompack: Delaunay 三角剖分
  - 264_cvtp: 周期性 CVT 迭代优化
  - 1006_random_data: 区域内随机采样
  - 891_polygonal_surface_display: 多边形面片拓扑
  - 1255_tetrahedron_slice_movie: 三维几何切片/相交
  - 545_house: 多边形边界定义
"""

import numpy as np
from utils import uniform_in_triangle, polygon_area_2d, polygon_centroid_2d


def delaunay_triangulation_2d(points):
    r"""
    二维 Delaunay 三角剖分（基于 Bowyer-Watson 增量算法）。

    对平流层应用：在每条等高度面上，将经纬度坐标投影到切平面后进行
    局部三角剖分，再插值回球面。

    Parameters
    ----------
    points : ndarray, shape (n, 2)
        二维点集 (x, y)。

    Returns
    -------
    triangles : ndarray, shape (m, 3)
        每个三角形的三个顶点索引。
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 3:
        return np.empty((0, 3), dtype=int)

    # 构造超三角形
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    dx = maxx - minx
    dy = maxy - miny
    dmax = max(dx, dy)
    midx = 0.5 * (minx + maxx)
    midy = 0.5 * (miny + maxy)
    pts_all = np.vstack([
        pts,
        [[midx - 20 * dmax, midy - dmax],
         [midx, midy + 20 * dmax],
         [midx + 20 * dmax, midy - dmax]]
    ])

    triangles = [[n, n + 1, n + 2]]

    def circumcircle(p1, p2, p3):
        d = 2.0 * (p1[0] * (p2[1] - p3[1]) +
                   p2[0] * (p3[1] - p1[1]) +
                   p3[0] * (p1[1] - p2[1]))
        if abs(d) < 1e-14:
            return None, None, float('inf')
        ux = ((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1]) +
              (p2[0] ** 2 + p2[1] ** 2) * (p3[1] - p1[1]) +
              (p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / d
        uy = ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0]) +
              (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p3[0]) +
              (p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / d
        r2 = (ux - p1[0]) ** 2 + (uy - p1[1]) ** 2
        return ux, uy, r2

    def point_in_circumcircle(pt, tri):
        p = [pts_all[tri[i]] for i in range(3)]
        cx, cy, r2 = circumcircle(*p)
        if cx is None:
            return False
        d2 = (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2
        return d2 <= r2 + 1e-12

    for i in range(n):
        pt = pts_all[i]
        bad = []
        for t in triangles:
            if point_in_circumcircle(pt, t):
                bad.append(t)

        polygon = []
        for t in bad:
            edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
            for e in edges:
                shared = False
                for ot in bad:
                    if ot is t:
                        continue
                    oedges = [(ot[0], ot[1]), (ot[1], ot[2]), (ot[2], ot[0])]
                    if (e in oedges) or ((e[1], e[0]) in oedges):
                        shared = True
                        break
                if not shared:
                    polygon.append(e)

        triangles = [t for t in triangles if t not in bad]
        for e in polygon:
            triangles.append([e[0], e[1], i])

    super_idx = {n, n + 1, n + 2}
    triangles = [
        t for t in triangles
        if not any(v in super_idx for v in t)
    ]
    return np.array(triangles, dtype=int)


class StratosphericMesh:
    r"""
    平流层三维非结构化网格。

    定义域：
      - 经度 λ ∈ [0, 2π]
      - 纬度 φ ∈ [-π/3, π/3]（ tropics / mid-latitudes ）
      - 高度 z ∈ [15, 50] km

    网格构造策略：
      1. 在每条等高度面上生成经纬度采样点；
      2. 使用 Delaunay 三角剖分建立面内拓扑；
      3. 层间通过棱柱单元连接；
      4. 可选 CVT 迭代优化水平分布。
    """

    def __init__(self, n_lon=24, n_lat=12, n_alt=8,
                 lon_range=(0.0, 2 * np.pi),
                 lat_range=(-np.pi / 3.0, np.pi / 3.0),
                 alt_range=(15.0, 50.0)):
        self.n_lon = n_lon
        self.n_lat = n_lat
        self.n_alt = n_alt
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.alt_range = alt_range
        self._generate_nodes()
        self._build_layers()
        self._build_vertical_topology()

    def _generate_nodes(self):
        r"""
        生成节点坐标 (lon, lat, alt)。
        水平方向使用准均匀采样（含极区加密）。
        """
        lons = np.linspace(self.lon_range[0], self.lon_range[1], self.n_lon, endpoint=False)
        # 纬度使用 arcsin 分布以近似等面积
        s = np.linspace(-1.0, 1.0, self.n_lat)
        lats = np.arcsin(s * np.sin(self.lat_range[1]))
        alts = np.linspace(self.alt_range[0], self.alt_range[1], self.n_alt)

        self.nodes = []
        self.node_alt = []
        self.node_lonlat = []
        for z in alts:
            for la in lats:
                for lo in lons:
                    self.nodes.append([lo, la, z])
                    self.node_alt.append(z)
                    self.node_lonlat.append([lo, la])
        self.nodes = np.array(self.nodes)
        self.node_alt = np.array(self.node_alt)
        self.node_lonlat = np.array(self.node_lonlat)
        self.n_nodes = self.nodes.shape[0]

    def _build_layers(self):
        r"""
        对每个高度层进行 Delaunay 三角剖分。
        使用平面投影（等距圆柱投影）：x = λ, y = sin φ。
        """
        self.layer_triangles = []
        nodes_per_layer = self.n_lon * self.n_lat
        for k in range(self.n_alt):
            idx0 = k * nodes_per_layer
            pts = self.node_lonlat[idx0:idx0 + nodes_per_layer].copy()
            # 投影到平面
            proj = np.column_stack([pts[:, 0], np.sin(pts[:, 1])])
            # 周期处理：复制最左侧点到右侧
            dup = []
            dup_map = {}
            for i in range(self.n_lat):
                lo = self.lon_range[1]
                la = np.arcsin(proj[i * self.n_lon, 1])
                dup.append([lo + 0.05, np.sin(la)])
                dup_map[len(proj) + i] = i * self.n_lon  # 映射回原始
            if dup:
                proj_ext = np.vstack([proj, np.array(dup)])
            else:
                proj_ext = proj
            tri = delaunay_triangulation_2d(proj_ext)
            # 映射回原始索引
            mapped = []
            for t in tri:
                mt = []
                for v in t:
                    if v in dup_map:
                        mt.append(dup_map[v] + idx0)
                    elif v < nodes_per_layer:
                        mt.append(v + idx0)
                if len(mt) == 3:
                    mapped.append(mt)
            self.layer_triangles.append(np.array(mapped, dtype=int))

    def _build_vertical_topology(self):
        r"""
        构建层间棱柱拓扑：每个三角形柱体包含上下两层各一个三角形。
        同时计算每个单元的质心和体积权重。
        """
        self.prisms = []
        self.cell_centroids = []
        self.cell_volumes = []
        nodes_per_layer = self.n_lon * self.n_lat
        R_earth = 6371.0  # km
        for k in range(self.n_alt - 1):
            tri = self.layer_triangles[k]
            z_low = self.alt_range[0] + k * (self.alt_range[1] - self.alt_range[0]) / (self.n_alt - 1)
            z_high = self.alt_range[0] + (k + 1) * (self.alt_range[1] - self.alt_range[0]) / (self.n_alt - 1)
            dz = z_high - z_low
            for t in tri:
                # 下层三个节点
                n1, n2, n3 = t
                # 上层对应节点
                n1u = n1 + nodes_per_layer
                n2u = n2 + nodes_per_layer
                n3u = n3 + nodes_per_layer
                self.prisms.append([n1, n2, n3, n1u, n2u, n3u])

                # 计算质心（球面坐标平均）
                pts = self.nodes[[n1, n2, n3, n1u, n2u, n3u]]
                cent = pts.mean(axis=0)
                self.cell_centroids.append(cent)

                # 近似体积：三角形面积 × dz（球面近似）
                lon = self.nodes[[n1, n2, n3], 0]
                lat = self.nodes[[n1, n2, n3], 1]
                x = R_earth * np.cos(lat) * np.cos(lon)
                y = R_earth * np.cos(lat) * np.sin(lon)
                # 平面投影面积
                area = 0.5 * abs(
                    x[0] * (y[1] - y[2]) +
                    x[1] * (y[2] - y[0]) +
                    x[2] * (y[0] - y[1])
                )
                vol = area * dz
                self.cell_volumes.append(max(vol, 1e-10))

        self.prisms = np.array(self.prisms, dtype=int)
        self.cell_centroids = np.array(self.cell_centroids)
        self.cell_volumes = np.array(self.cell_volumes)
        self.n_cells = self.prisms.shape[0]

    def cvt_optimize_horizontal(self, n_iter=5, n_samples=5000):
        r"""
        对每层水平网格执行周期性 CVT 迭代优化。

        CVT 能量泛函：

            \mathcal{F}(\mathbf{g}_1,\dots,\mathbf{g}_N)
            = \sum_{i=1}^{N} \int_{V_i} \rho(\mathbf{x})
                \|\mathbf{x} - \mathbf{g}_i\|^2 \, d\mathbf{x}

        其中 V_i 为 Voronoi 单元，g_i 为生成点。

        Parameters
        ----------
        n_iter : int
            CVT 迭代次数。
        n_samples : int
            每次迭代的蒙特卡洛采样数。
        """
        nodes_per_layer = self.n_lon * self.n_lat
        R_earth = 6371.0
        for k in range(self.n_alt):
            idx0 = k * nodes_per_layer
            # 提取当前层生成点（投影坐标）
            gen = self.node_lonlat[idx0:idx0 + nodes_per_layer].copy()
            gen_proj = np.column_stack([gen[:, 0], np.sin(gen[:, 1])])
            a = np.array([self.lon_range[0], gen_proj[:, 1].min()])
            b = np.array([self.lon_range[1], gen_proj[:, 1].max()])

            for _iter in range(n_iter):
                gen_new = np.zeros_like(gen_proj)
                counts = np.zeros(nodes_per_layer)
                for _s in range(n_samples):
                    x = np.random.rand(2) * (b - a) + a
                    # 找最近生成点（周期性经度）
                    dists = np.zeros(nodes_per_layer)
                    for i in range(nodes_per_layer):
                        dx = abs(x[0] - gen_proj[i, 0])
                        dx = min(dx, b[0] - a[0] - dx)
                        dy = x[1] - gen_proj[i, 1]
                        dists[i] = dx ** 2 + dy ** 2
                    nearest = int(np.argmin(dists))
                    gen_new[nearest] += x
                    counts[nearest] += 1
                for i in range(nodes_per_layer):
                    if counts[i] > 0:
                        gen_proj[i] = gen_new[i] / counts[i]
                # 边界钳制
                gen_proj[:, 0] = np.clip(gen_proj[:, 0], a[0], b[0])
                gen_proj[:, 1] = np.clip(gen_proj[:, 1], a[1], b[1])

            # 写回节点坐标
            for i in range(nodes_per_layer):
                self.nodes[idx0 + i, 0] = gen_proj[i, 0]
                self.nodes[idx0 + i, 1] = np.arcsin(np.clip(gen_proj[i, 1], -0.999, 0.999))
                self.node_lonlat[idx0 + i] = [self.nodes[idx0 + i, 0],
                                               self.nodes[idx0 + i, 1]]

        # 重新构建拓扑
        self._build_layers()
        self._build_vertical_topology()

    def get_cell_nodes(self, cell_idx):
        r"""返回单元包含的节点索引。"""
        return self.prisms[cell_idx]

    def get_node_coordinates(self, node_idx):
        r"""返回节点坐标 (lon, lat, alt)。"""
        return self.nodes[node_idx]
