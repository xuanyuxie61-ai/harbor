"""
fractal_media.py
================================================================================
分形多孔介质生成与非结构化六边形网格采样模块

基于种子项目：
  - 526_hexagon_chaos：六边形混沌博弈（IFS）、重心坐标与均匀采样

科学背景：
  天然含水层的孔隙结构常具有分形特征：裂隙网络、土壤团聚体和溶蚀孔洞
  在不同尺度上呈现自相似性。分形几何为描述这种多尺度异质性提供了数学框架。

  分形维数 D_f 与孔隙度 φ 的关系（Menger sponge 类型模型）：
      φ(L) ∝ L^{3 - D_f}
  其中 L 为观测尺度，D_f ∈ [2, 3] 为分形维数。

  六边形网格在地下水模型中的优势：
    1. 各向同性：六边形单元消除矩形网格的方向依赖性
    2. 无网格取向效应：弥散张量的旋转不变性更好保持
    3. 适用于 MODFLOW-USG 等非结构化地下水模型

  本模块实现：
    - 迭代函数系统（IFS）生成分形孔隙结构
    - 正六边形域上的均匀随机采样（用于粒子初始位置）
    - 六边形网格的节点生成与邻接关系
================================================================================
"""

import numpy as np
from typing import List, Tuple


def hexagon_vertices(center: Tuple[float, float] = (0.0, 0.0),
                      radius: float = 1.0) -> np.ndarray:
    """
    计算正六边形的 6 个顶点坐标。

    顶点位置（逆时针）：
        v_k = (cx + R cos(kπ/3), cy + R sin(kπ/3)),  k = 0, ..., 5
    """
    if radius <= 0:
        raise ValueError("半径必须为正")
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    verts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return verts


def sample_uniform_hexagon(n_samples: int, center: Tuple[float, float] = (0.0, 0.0),
                           radius: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    在正六边形内部进行均匀随机采样（拒绝-free 方法）。

    算法：将六边形分解为 3 个菱形，每个菱形由两个等边三角形组成。
    1. 等概率选择其中一个菱形
    2. 在菱形内使用重心坐标生成均匀点：
         p = v0 + u1 (v1 - v0) + u2 (v2 - v0)
       其中 u1, u2 ~ U(0,1)，若 u1 + u2 > 1 则反射到另一侧
    """
    if n_samples < 1:
        raise ValueError("样本数必须 ≥ 1")
    rng = np.random.default_rng(seed)
    verts = hexagon_vertices(center, radius)
    # 六边形中心点
    c = np.array(center)

    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        # 选择 6 个三角形扇区之一
        k = rng.integers(0, 6)
        v0 = c
        v1 = verts[k]
        v2 = verts[(k + 1) % 6]

        u1 = rng.random()
        u2 = rng.random()
        if u1 + u2 > 1.0:
            u1 = 1.0 - u1
            u2 = 1.0 - u2
        samples[i] = v0 + u1 * (v1 - v0) + u2 * (v2 - v0)
    return samples


def hexagon_grid(nx: int, ny: int, radius: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    """
    生成蜂窝状六边形网格节点与单元连通性。

    节点排列采用交错行布局：
        偶数行 y = j * (√3 R)
        奇数行 y = j * (√3 R)，x 偏移 1.5 R

    每个六边形单元由中心节点和其 6 个邻接节点定义。

    返回
    -------
    coords : np.ndarray, shape (n_nodes, 2)
    elements : List[List[int]]
        每个单元的节点编号列表
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny 必须 ≥ 1")
    dy = radius * np.sqrt(3)
    nodes = []
    node_id = {}
    for j in range(ny):
        y = j * dy
        x_offset = 0.0 if j % 2 == 0 else 1.5 * radius
        for i in range(nx):
            x = x_offset + i * 3.0 * radius
            nid = len(nodes)
            node_id[(i, j)] = nid
            nodes.append([x, y])

    coords = np.array(nodes)

    # 构建单元：每个内部节点作为六边形中心，连接6邻域
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            # 简化的四边形/三角形混合单元
            n0 = node_id.get((i, j))
            n1 = node_id.get((i + 1, j))
            n2 = node_id.get((i, j + 1))
            n3 = node_id.get((i + 1, j + 1))
            if None not in (n0, n1, n2, n3):
                # 每个“菱形”拆为两个三角形
                elements.append([n0, n1, n3])
                elements.append([n0, n3, n2])

    return coords, elements


class FractalPorousMedia:
    """
    使用迭代函数系统（IFS）生成分形多孔介质渗透率场。
    """

    def __init__(self, n_iterations: int = 4, n_points: int = 10000,
                 seed: int = 42):
        self.n_iterations = n_iterations
        self.n_points = n_points
        self.rng = np.random.default_rng(seed)

    def generate_sierpinski_carpet_permeability(self, grid_res: int = 64) -> np.ndarray:
        """
        生成 Sierpinski carpet 类型的分形渗透率场。

        Sierpinski carpet 构造：
          1. 初始正方形 [0,1]²
          2. 将正方形均分为 3×3 的 9 个子正方形
          3. 移除中心子正方形
          4. 对剩余 8 个子正方形递归重复步骤 2-3

        渗透率映射：
          - 被移除区域（孔隙/裂隙）：K = K_max（高渗透通道）
          - 保留区域（基质）：K = K_min（低渗透基质）

        分形维数：
          D_f = log(8) / log(3) ≈ 1.893
        """
        if grid_res < 3:
            raise ValueError("网格分辨率必须 ≥ 3")
        K = np.ones((grid_res, grid_res), dtype=float)
        K_min = 1e-4
        K_max = 10.0

        def remove_center(arr, level):
            if level == 0:
                return
            m, n = arr.shape
            if m < 3 or n < 3:
                return
            # 中心块坐标
            cm, cn = m // 3, n // 3
            arr[cm:2 * cm, cn:2 * cn] = K_max  # 高渗透通道
            # 递归处理 8 个外围子块
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    sub = arr[i * cm:(i + 1) * cm, j * cn:(j + 1) * cn]
                    remove_center(sub, level - 1)

        remove_center(K, self.n_iterations)
        K[K == 1.0] = K_min  # 基质
        return K

    def generate_ifs_attractor(self, n_points: int = 5000) -> np.ndarray:
        """
        使用混沌博弈（chaos game）生成分形吸引子点云。

        算法（Barnsley fern 变体用于裂隙网络）：
          1. 随机选取初始点 p0 在凸包内
          2. 迭代 n_points 次：
               随机选择仿射变换 T_k
               p_{i+1} = T_k(p_i)
          3. 返回点云
        """
        # 定义 4 个仿射变换（模拟裂隙网络的自相似结构）
        transforms = [
            {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.16, "e": 0.0, "f": 0.0, "prob": 0.01},
            {"a": 0.85, "b": 0.04, "c": -0.04, "d": 0.85, "e": 0.0, "f": 1.6, "prob": 0.85},
            {"a": 0.2, "b": -0.26, "c": 0.23, "d": 0.22, "e": 0.0, "f": 1.6, "prob": 0.07},
            {"a": -0.15, "b": 0.28, "c": 0.26, "d": 0.24, "e": 0.0, "f": 0.44, "prob": 0.07},
        ]
        probs = np.cumsum([t["prob"] for t in transforms])

        points = np.zeros((n_points, 2))
        p = np.array([0.0, 0.0])
        for i in range(n_points):
            r = self.rng.random()
            idx = np.searchsorted(probs, r)
            t = transforms[idx]
            x = t["a"] * p[0] + t["b"] * p[1] + t["e"]
            y = t["c"] * p[0] + t["d"] * p[1] + t["f"]
            p = np.array([x, y])
            points[i] = p
        return points

    def fractal_dimension_boxcount(self, points: np.ndarray, n_boxes: int = 20) -> float:
        """
        使用盒计数法估计点云的分形维数。

        定义：
            N(ε) ~ ε^{-D_f}
        其中 N(ε) 是边长为 ε 的盒子中至少包含一个点的盒子数。
        对数线性回归：
            D_f = -d(log N) / d(log ε)
        """
        if len(points) == 0:
            raise ValueError("点云不能为空")
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        L = max(xmax - xmin, ymax - ymin)
        if L <= 0:
            return 0.0

        counts = []
        epsilons = []
        for k in range(1, n_boxes + 1):
            eps = L / k
            # 将点映射到盒子索引
            ix = np.floor((points[:, 0] - xmin) / eps).astype(int)
            iy = np.floor((points[:, 1] - ymin) / eps).astype(int)
            boxes = set(zip(ix, iy))
            counts.append(len(boxes))
            epsilons.append(eps)

        counts = np.array(counts, dtype=float)
        epsilons = np.array(epsilons, dtype=float)
        valid = counts > 0
        if valid.sum() < 2:
            return 0.0
        logN = np.log(counts[valid])
        logE = np.log(1.0 / epsilons[valid])
        # 线性回归斜率 = D_f
        A = np.vstack([logE, np.ones(len(logE))]).T
        D_f, _ = np.linalg.lstsq(A, logN, rcond=None)[0]
        return float(max(0.0, D_f))


if __name__ == "__main__":
    samples = sample_uniform_hexagon(1000, radius=2.0)
    assert samples.shape == (1000, 2)
    # 检查是否都在六边形内（到中心距离 < 半径，但六边形是内切圆）
    dists = np.linalg.norm(samples, axis=1)
    assert np.all(dists <= 2.0 + 1e-6)

    fpm = FractalPorousMedia(n_iterations=3)
    K_field = fpm.generate_sierpinski_carpet_permeability(grid_res=81)
    assert K_field.shape == (81, 81)

    pts = fpm.generate_ifs_attractor(2000)
    D_est = fpm.fractal_dimension_boxcount(pts)
    assert D_est > 0.5
    print("fractal_media: 自测试通过")
