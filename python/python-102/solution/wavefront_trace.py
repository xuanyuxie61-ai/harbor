"""
wavefront_trace.py
==================
基于 Dijkstra 最短路径算法的超构表面波前追踪与等光程路径计算。

本模块源自项目 287_dijkstra 的核心算法，将其应用于超构表面上的
光线/波前传播路径追踪。

科学背景：
超构表面通过调控局域相位 Φ(x,y) 来控制光波传播方向。
根据广义斯涅尔定律（Generalized Snell's Law）：
    n_t sin(θ_t) - n_i sin(θ_i) = (λ₀ / 2π) dΦ/dx

波前的传播可以建模为在相位梯度场上的最短路径问题：
定义等效度量（metric）：
    ds² = n_eff²(x,y) (dx² + dy²)
其中 n_eff(x,y) = 1 + (λ₀ / 2π) ∇Φ(x,y) 为等效折射率分布。

光程（Optical Path Length, OPL）：
    OPL = ∫ n_eff(x,y) ds

费马原理指出光线沿光程取极值的路径传播。
在离散化网格上，这等价于在加权图上寻找最短路径，
可使用 Dijkstra 算法高效求解。

数学模型：
    图 G = (V, E)
    顶点 V：超构表面网格节点
    边权重 w_{ij} = (n_i + n_j)/2 * ||r_i - r_j||
    Dijkstra 目标：min Σ w_{ij} 从源点到目标点
"""

import numpy as np


class WavefrontTracer:
    """
    超构表面波前追踪器，基于 Dijkstra 最短路径算法。
    """

    def __init__(self, x_grid, y_grid, phase_map):
        """
        Parameters
        ----------
        x_grid : ndarray, shape (nx,)
        y_grid : ndarray, shape (ny,)
        phase_map : ndarray, shape (nx, ny)
            超构表面局域相位分布 Φ(x,y) [rad]
        """
        self.x_grid = np.array(x_grid, dtype=np.float64)
        self.y_grid = np.array(y_grid, dtype=np.float64)
        self.phase_map = np.array(phase_map, dtype=np.float64)
        self.nx = len(x_grid)
        self.ny = len(y_grid)
        self.dx = x_grid[1] - x_grid[0] if self.nx > 1 else 1.0
        self.dy = y_grid[1] - y_grid[0] if self.ny > 1 else 1.0
        self.k0 = 2.0 * np.pi / 1.55e-6  # 默认波长

        # 计算相位梯度 → 等效折射率扰动
        dPhidx, dPhidy = np.gradient(phase_map, self.dx, self.dy)
        # 广义斯涅尔定律的等效折射率
        # n_eff ≈ 1 + (1/k0) |∇Φ|  （一阶近似）
        self.n_eff_map = 1.0 + np.sqrt(dPhidx ** 2 + dPhidy ** 2) / self.k0
        # 边界处理：确保 n_eff ≥ 1
        self.n_eff_map = np.clip(self.n_eff_map, 1.0, 5.0)

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------
    def build_graph(self):
        """
        从网格构建加权图。每个网格点连接上下左右 4 个邻居（可扩展为 8 邻域）。

        顶点编号：v = i * ny + j
        边权重：w = n_eff_avg * distance

        Returns
        -------
        adjacency : dict
            adjacency[v] = [(u, weight), ...]
        """
        adjacency = {}
        nx, ny = self.nx, self.ny
        for i in range(nx):
            for j in range(ny):
                v = i * ny + j
                adjacency[v] = []
                n_v = self.n_eff_map[i, j]
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1, j, self.dx))
                if i < nx - 1:
                    neighbors.append((i + 1, j, self.dx))
                if j > 0:
                    neighbors.append((i, j - 1, self.dy))
                if j < ny - 1:
                    neighbors.append((i, j + 1, self.dy))
                # 对角线邻居（8-连通）
                if i > 0 and j > 0:
                    neighbors.append((i - 1, j - 1, np.sqrt(self.dx ** 2 + self.dy ** 2)))
                if i > 0 and j < ny - 1:
                    neighbors.append((i - 1, j + 1, np.sqrt(self.dx ** 2 + self.dy ** 2)))
                if i < nx - 1 and j > 0:
                    neighbors.append((i + 1, j - 1, np.sqrt(self.dx ** 2 + self.dy ** 2)))
                if i < nx - 1 and j < ny - 1:
                    neighbors.append((i + 1, j + 1, np.sqrt(self.dx ** 2 + self.dy ** 2)))

                for ii, jj, dist in neighbors:
                    n_u = self.n_eff_map[ii, jj]
                    weight = 0.5 * (n_v + n_u) * dist
                    u = ii * ny + jj
                    adjacency[v].append((u, weight))
        return adjacency

    # ------------------------------------------------------------------
    # Dijkstra 算法（源自 287_dijkstra）
    # ------------------------------------------------------------------
    def dijkstra(self, source_idx):
        """
        Dijkstra 最短路径算法。

        Parameters
        ----------
        source_idx : int
            源点顶点编号

        Returns
        -------
        dist : ndarray, shape (nv,)
            从源点到各顶点的最短距离（光程）
        prev : ndarray, shape (nv,)
            最短路径上的前驱节点
        """
        adjacency = self.build_graph()
        nv = self.nx * self.ny
        INF = 1e20
        dist = np.full(nv, INF, dtype=np.float64)
        prev = np.full(nv, -1, dtype=np.int32)
        visited = np.zeros(nv, dtype=bool)

        dist[source_idx] = 0.0

        for _ in range(nv):
            # 找未访问中距离最小的节点
            min_dist = INF
            u = -1
            for v in range(nv):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    u = v
            if u == -1:
                break
            visited[u] = True

            # 松弛邻边
            for v, weight in adjacency.get(u, []):
                if not visited[v] and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u

        return dist, prev

    def trace_ray(self, x_src, y_src, x_tgt, y_tgt):
        """
        追踪从源点到目标点的光线/波前路径。

        Returns
        -------
        path_x : ndarray
        path_y : ndarray
        opl : float
            总光程
        """
        # 找到最近的网格点
        i_src = np.argmin(np.abs(self.x_grid - x_src))
        j_src = np.argmin(np.abs(self.y_grid - y_src))
        i_tgt = np.argmin(np.abs(self.x_grid - x_tgt))
        j_tgt = np.argmin(np.abs(self.y_grid - y_tgt))

        source = i_src * self.ny + j_src
        target = i_tgt * self.ny + j_tgt

        dist, prev = self.dijkstra(source)
        opl = dist[target]

        # 回溯路径
        if prev[target] == -1 and target != source:
            # 不可达
            return np.array([x_src, x_tgt]), np.array([y_src, y_tgt]), np.inf

        path = []
        v = target
        while v != -1:
            i = v // self.ny
            j = v % self.ny
            path.append((self.x_grid[i], self.y_grid[j]))
            if v == source:
                break
            v = prev[v]
        path = np.array(path[::-1])
        return path[:, 0], path[:, 1], opl

    # ------------------------------------------------------------------
    # 波前等时面计算
    # ------------------------------------------------------------------
    def compute_wavefront_contour(self, x_src, y_src, time_delay,
                                   n_contours=10):
        """
        计算从源点出发、经过不同时间延迟后的波前等时面（等光程面）。

        在离散网格上，这对应于 Dijkstra 距离场的等值线。
        """
        i_src = np.argmin(np.abs(self.x_grid - x_src))
        j_src = np.argmin(np.abs(self.y_grid - y_src))
        source = i_src * self.ny + j_src

        dist, _ = self.dijkstra(source)
        dist_grid = dist.reshape(self.nx, self.ny)

        # 光程到时间：t = OPL / c
        c0 = 2.99792458e8
        time_grid = dist_grid / c0

        # 提取等时面（使用阈值近似）
        contours = []
        max_time = np.max(time_grid[time_grid < 1e19])
        if max_time <= 0:
            return contours
        delays = np.linspace(0, min(time_delay, max_time), n_contours + 1)[1:]

        for td in delays:
            mask = np.abs(time_grid - td) < 0.05 * td
            if np.any(mask):
                xs = []
                ys = []
                for i in range(self.nx):
                    for j in range(self.ny):
                        if mask[i, j]:
                            xs.append(self.x_grid[i])
                            ys.append(self.y_grid[j])
                if len(xs) > 0:
                    contours.append((td, np.array(xs), np.array(ys)))
        return contours

    def phase_gradient_direction(self, x, y):
        """
        计算给定位置的局部光线传播方向（与相位梯度正交）。
        """
        i = np.argmin(np.abs(self.x_grid - x))
        j = np.argmin(np.abs(self.y_grid - y))
        dPhidx, dPhidy = np.gradient(self.phase_map, self.dx, self.dy)
        gx = dPhidx[i, j]
        gy = dPhidy[i, j]
        norm = np.sqrt(gx ** 2 + gy ** 2)
        if norm < 1e-15:
            return np.array([1.0, 0.0])
        # 光线方向 ∝ ∇Φ
        return np.array([gx, gy]) / norm

    def evaluate_beam_steering(self, aperture_x_range, aperture_y_range):
        """
        评估超构表面的光束偏转效率。
        通过追踪从孔径中心发出的光线，统计其偏转角度分布。
        """
        x_min, x_max = aperture_x_range
        y_min, y_max = aperture_y_range
        x_src = 0.5 * (x_min + x_max)
        y_src = 0.5 * (y_min + y_max)

        # 在孔径边缘选取多个目标点
        n_angles = 36
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radius = 0.5 * min(x_max - x_min, y_max - y_min)

        steering_angles = []
        opls = []
        for ang in angles:
            x_tgt = x_src + radius * np.cos(ang)
            y_tgt = y_src + radius * np.sin(ang)
            px, py, opl = self.trace_ray(x_src, y_src, x_tgt, y_tgt)
            if opl < np.inf and len(px) >= 2:
                dx_end = px[-1] - px[-2]
                dy_end = py[-1] - py[-2]
                steer_ang = np.arctan2(dy_end, dx_end)
                steering_angles.append(steer_ang)
                opls.append(opl)

        steering_angles = np.array(steering_angles)
        opls = np.array(opls)
        return steering_angles, opls


def demo():
    """演示：在聚焦相位超构表面上追踪光线。"""
    nx, ny = 81, 81
    x = np.linspace(-5e-6, 5e-6, nx)
    y = np.linspace(-5e-6, 5e-6, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 聚焦透镜相位
    k0 = 2.0 * np.pi / 1.55e-6
    f = 20.0e-6
    phase = -k0 * (np.sqrt(X ** 2 + Y ** 2 + f ** 2) - f)
    phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi

    tracer = WavefrontTracer(x, y, phase)

    # 追踪一条光线
    px, py, opl = tracer.trace_ray(0.0, 0.0, 3.0e-6, 0.0)
    print(f"[wavefront_trace] 光线追踪路径点数: {len(px)}")
    print(f"[wavefront_trace] 总光程: {opl:.6e} m")

    # 光束偏转评估
    steer_angles, opls_all = tracer.evaluate_beam_steering((-4e-6, 4e-6), (-4e-6, 4e-6))
    print(f"[wavefront_trace] 评估光线数: {len(steer_angles)}")
    print(f"[wavefront_trace] 平均偏转角度: {np.degrees(np.mean(steer_angles)):.2f}°")
    print(f"[wavefront_trace] 光程标准差: {opls_all.std():.3e} m")
    return px, py, opl


if __name__ == "__main__":
    demo()
