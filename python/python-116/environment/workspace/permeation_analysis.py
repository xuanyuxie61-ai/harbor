"""
permeation_analysis.py
膜渗透路径分析模块

本模块利用 Dijkstra 最短路径算法分析水分子（或小分子）
穿过脂质双分子层的最低能垒路径。

参考种子项目: 287_dijkstra (Dijkstra 最短路径算法)

物理背景:
    小分子跨膜渗透遵循溶解-扩散机制:
        P = K_D * D / d
    其中 K_D 为分配系数，D 为扩散系数，d 为膜厚度。

    在自由能景观中，渗透对应于寻找从膜外水相 (z=-∞) 到
    膜内水相 (z=+∞) 的最低自由能路径（Minimum Free Energy Path, MFEP）。
    将膜横截面离散化为图节点，节点间边权为自由能差，
    则 MFEP 等价于最短路径问题。

    图构造:
      - 节点: (i, j, k) 对应空间位置 (x_i, y_j, z_k)
      - 边权: w_{ab} = exp(β [F(b) - F(a)]_+)  （Metropolis 型跃迁率）
      或简化为 w_{ab} = |ΔF| + λ |r_b - r_a|

    Dijkstra 算法保证找到从源点到所有其他节点的最短路径。
"""

import numpy as np
import heapq


class DijkstraPermeation:
    """
    Dijkstra 算法在膜渗透分析中的应用（受种子项目 287_dijkstra 启发）。
    """

    def __init__(self, nx, ny, nz, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
                 zlim=(-3.0, 3.0)):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    def node_index(self, i, j, k):
        """三维格点展平为一维节点索引。"""
        return (i * self.ny + j) * self.nz + k

    def inverse_index(self, idx):
        """一维节点索引反解三维坐标。"""
        k = idx % self.nz
        j = (idx // self.nz) % self.ny
        i = idx // (self.ny * self.nz)
        return i, j, k

    def build_graph_from_free_energy(self, free_energy_field, beta=1.0):
        """
        由自由能场构建带权图。

        Parameters
        ----------
        free_energy_field : ndarray, shape (nx, ny, nz)
            每个格点的自由能值 F(i,j,k)。
        beta : float
            1/(k_B T)。

        Returns
        -------
        graph : dict
            graph[node] = [(neighbor, weight), ...]
        """
        fe = np.asarray(free_energy_field)
        if fe.shape != (self.nx, self.ny, self.nz):
            raise ValueError("free_energy_field 形状不匹配。")

        n_nodes = self.nx * self.ny * self.nz
        graph = {n: [] for n in range(n_nodes)}

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    node = self.node_index(i, j, k)
                    for di, dj, dk in [
                        (1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1)
                    ]:
                        ii = i + di
                        jj = j + dj
                        kk = k + dk
                        if not (0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz):
                            continue
                        neighbor = self.node_index(ii, jj, kk)
                        dF = fe[ii, jj, kk] - fe[i, j, k]
                        # 边权: 正向能垒 + 距离惩罚
                        weight = np.exp(beta * max(dF, 0.0)) + 0.1
                        weight = float(weight)
                        graph[node].append((neighbor, weight))
        return graph

    def dijkstra(self, graph, source):
        """
        Dijkstra 最短路径算法。

        Parameters
        ----------
        graph : dict
            邻接表表示的图。
        source : int
            源节点。

        Returns
        -------
        dist : ndarray
            从源点到各节点的最短距离。
        prev : ndarray
            最短路径上的前驱节点。
        """
        n_nodes = len(graph)
        dist = np.full(n_nodes, np.inf)
        prev = np.full(n_nodes, -1, dtype=int)
        dist[source] = 0.0
        visited = np.zeros(n_nodes, dtype=bool)

        # 优先队列: (距离, 节点)
        pq = [(0.0, source)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True

            for v, w in graph[u]:
                if visited[v]:
                    continue
                alt = d_u + w
                if alt < dist[v] - 1e-12:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))

        return dist, prev

    def find_mfep(self, free_energy_field, source_z_layer=0, target_z_layer=None, beta=1.0):
        """
        寻找从膜一侧到另一侧的最低自由能路径（MFEP）。

        Parameters
        ----------
        free_energy_field : ndarray
        source_z_layer : int
            起始 z 层索引（通常为 0，膜外）。
        target_z_layer : int or None
            目标 z 层索引（默认为 nz-1）。
        beta : float

        Returns
        -------
        path : list of int
            路径节点序列。
        path_cost : float
            路径总代价。
        """
        if target_z_layer is None:
            target_z_layer = self.nz - 1
        if not (0 <= source_z_layer < self.nz and 0 <= target_z_layer < self.nz):
            raise ValueError("z 层索引超出范围。")

        graph = self.build_graph_from_free_energy(free_energy_field, beta)

        # 源点: source_z_layer 上所有节点的虚拟源
        # 合并为超级源点
        super_source = len(graph)
        graph[super_source] = []
        for i in range(self.nx):
            for j in range(self.ny):
                node = self.node_index(i, j, source_z_layer)
                graph[super_source].append((node, 0.0))

        dist, prev = self.dijkstra(graph, super_source)

        # 找到 target_z_layer 上距离最小的节点
        best_target = None
        best_dist = np.inf
        for i in range(self.nx):
            for j in range(self.ny):
                node = self.node_index(i, j, target_z_layer)
                if dist[node] < best_dist:
                    best_dist = dist[node]
                    best_target = node

        if best_target is None or best_dist == np.inf:
            return [], np.inf

        # 回溯路径
        path = []
        node = best_target
        while node != -1 and node != super_source:
            path.append(node)
            node = prev[node]
        path.reverse()

        return path, float(best_dist)

    def permeability_coefficient(self, path, free_energy_field, beta, D0=1e-6):
        """
        由 MFEP 估算渗透系数 P。

        过渡态理论近似:
            P = D0 * exp(-β ΔF‡) / d
        其中 ΔF‡ 为路径上的最大自由能垒。
        """
        if len(path) == 0:
            return 0.0
        fe_max = np.max([free_energy_field[self.inverse_index(n)] for n in path])
        fe_min = np.min([free_energy_field[self.inverse_index(n)] for n in path])
        barrier = max(fe_max - fe_min, 0.0)
        d = abs(self.zlim[1] - self.zlim[0])
        P = D0 * np.exp(-beta * barrier) / d
        return float(P)


class FreeEnergyFieldGenerator:
    """
    生成跨膜自由能场的模型函数。
    """

    @staticmethod
    def asymmetric_double_well(z, z0=1.0, V0=15.0, sigma=0.5, asym=2.0):
        """
        非对称双势阱模型:
            F(z) = V0 * [exp(-(z-z0)²/(2σ²)) + asym*exp(-(z+z0)²/(2σ²))]
        描述膜的不对称渗透能垒（例如外层与内层头基化学差异）。
        """
        return V0 * (np.exp(-(z - z0) ** 2 / (2.0 * sigma ** 2)) +
                     asym * np.exp(-(z + z0) ** 2 / (2.0 * sigma ** 2)))

    @staticmethod
    def harmonic_plus_barrier(z, k=5.0, z0=1.5, V_barrier=10.0):
        """
        谐振子加中央势垒:
            F(z) = (k/2) z² + V_barrier / cosh²(z/z0)
        描述膜中心的疏水势垒叠加于水相的谐振恢复力。
        """
        return 0.5 * k * z ** 2 + V_barrier / np.cosh(z / z0) ** 2

    @staticmethod
    def generate_3d_field(nx, ny, nz, xlim, ylim, zlim, model='double_well', **kwargs):
        """
        生成三维自由能场（x,y 方向均匀，z 方向变化）。
        """
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        z = np.linspace(zlim[0], zlim[1], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        if model == 'double_well':
            F_z = FreeEnergyFieldGenerator.asymmetric_double_well(Z, **kwargs)
        elif model == 'harmonic_barrier':
            F_z = FreeEnergyFieldGenerator.harmonic_plus_barrier(Z, **kwargs)
        else:
            F_z = np.zeros_like(Z)

        # 添加 x-y 平面的小扰动模拟局部缺陷
        defect = 0.5 * np.sin(2 * np.pi * X / (xlim[1] - xlim[0])) * \
                 np.cos(2 * np.pi * Y / (ylim[1] - ylim[0]))
        return F_z + defect
