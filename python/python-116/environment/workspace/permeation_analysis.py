
import numpy as np
import heapq


class DijkstraPermeation:

    def __init__(self, nx, ny, nz, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
                 zlim=(-3.0, 3.0)):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    def node_index(self, i, j, k):
        return (i * self.ny + j) * self.nz + k

    def inverse_index(self, idx):
        k = idx % self.nz
        j = (idx // self.nz) % self.ny
        i = idx // (self.ny * self.nz)
        return i, j, k

    def build_graph_from_free_energy(self, free_energy_field, beta=1.0):
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

                        weight = np.exp(beta * max(dF, 0.0)) + 0.1
                        weight = float(weight)
                        graph[node].append((neighbor, weight))
        return graph

    def dijkstra(self, graph, source):
        n_nodes = len(graph)
        dist = np.full(n_nodes, np.inf)
        prev = np.full(n_nodes, -1, dtype=int)
        dist[source] = 0.0
        visited = np.zeros(n_nodes, dtype=bool)


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
        if target_z_layer is None:
            target_z_layer = self.nz - 1
        if not (0 <= source_z_layer < self.nz and 0 <= target_z_layer < self.nz):
            raise ValueError("z 层索引超出范围。")

        graph = self.build_graph_from_free_energy(free_energy_field, beta)



        super_source = len(graph)
        graph[super_source] = []
        for i in range(self.nx):
            for j in range(self.ny):
                node = self.node_index(i, j, source_z_layer)
                graph[super_source].append((node, 0.0))

        dist, prev = self.dijkstra(graph, super_source)


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


        path = []
        node = best_target
        while node != -1 and node != super_source:
            path.append(node)
            node = prev[node]
        path.reverse()

        return path, float(best_dist)

    def permeability_coefficient(self, path, free_energy_field, beta, D0=1e-6):
        if len(path) == 0:
            return 0.0
        fe_max = np.max([free_energy_field[self.inverse_index(n)] for n in path])
        fe_min = np.min([free_energy_field[self.inverse_index(n)] for n in path])
        barrier = max(fe_max - fe_min, 0.0)
        d = abs(self.zlim[1] - self.zlim[0])
        P = D0 * np.exp(-beta * barrier) / d
        return float(P)


class FreeEnergyFieldGenerator:

    @staticmethod
    def asymmetric_double_well(z, z0=1.0, V0=15.0, sigma=0.5, asym=2.0):
        return V0 * (np.exp(-(z - z0) ** 2 / (2.0 * sigma ** 2)) +
                     asym * np.exp(-(z + z0) ** 2 / (2.0 * sigma ** 2)))

    @staticmethod
    def harmonic_plus_barrier(z, k=5.0, z0=1.5, V_barrier=10.0):
        return 0.5 * k * z ** 2 + V_barrier / np.cosh(z / z0) ** 2

    @staticmethod
    def generate_3d_field(nx, ny, nz, xlim, ylim, zlim, model='double_well', **kwargs):
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


        defect = 0.5 * np.sin(2 * np.pi * X / (xlim[1] - xlim[0])) * \
                 np.cos(2 * np.pi * Y / (ylim[1] - ylim[0]))
        return F_z + defect
