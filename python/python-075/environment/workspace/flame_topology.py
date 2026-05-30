
import numpy as np


class FlameFrontGraph:

    def __init__(self, c_field, threshold=0.5):
        self.c_field = c_field
        self.threshold = threshold
        self.burned = (c_field > threshold).astype(int)
        self.nx, self.ny = c_field.shape
        self._build_graph()

    def _build_graph(self):
        self.adj = {}
        self.node_to_idx = {}
        self.idx_to_node = []
        idx = 0
        for i in range(self.nx):
            for j in range(self.ny):
                if self.burned[i, j]:
                    self.node_to_idx[(i, j)] = idx
                    self.idx_to_node.append((i, j))
                    idx += 1

        self.n_nodes = idx
        for idx, (i, j) in enumerate(self.idx_to_node):
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.nx and 0 <= nj < self.ny and self.burned[ni, nj]:
                    neighbors.append(self.node_to_idx[(ni, nj)])
            self.adj[idx] = neighbors

    def find_connected_components(self):
        if self.n_nodes == 0:
            return []

        visited = np.zeros(self.n_nodes, dtype=bool)
        components = []

        for start in range(self.n_nodes):
            if visited[start]:
                continue


            queue = [start]
            visited[start] = True
            component = []

            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in self.adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            components.append(component)

        return components

    def component_sizes(self):
        comps = self.find_connected_components()
        return [len(c) for c in comps]

    def flame_front_length(self):
        perimeter = 0
        for i in range(self.nx):
            for j in range(self.ny):
                if self.burned[i, j]:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if ni < 0 or ni >= self.nx or nj < 0 or nj >= self.ny:
                            perimeter += 1
                        elif not self.burned[ni, nj]:
                            perimeter += 1
        return perimeter

    def distance_from_front(self):
        if self.n_nodes == 0:
            return np.full((self.nx, self.ny), np.inf)


        front_nodes = set()
        for idx, (i, j) in enumerate(self.idx_to_node):
            is_front = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= self.nx or nj < 0 or nj >= self.ny:
                    is_front = True
                    break
                if not self.burned[ni, nj]:
                    is_front = True
                    break
            if is_front:
                front_nodes.add(idx)


        distance = np.full(self.n_nodes, np.inf)
        queue = list(front_nodes)
        for node in queue:
            distance[node] = 0.0

        d = 0
        while queue:
            next_queue = []
            d += 1
            for node in queue:
                for neighbor in self.adj[node]:
                    if distance[neighbor] == np.inf:
                        distance[neighbor] = d
                        next_queue.append(neighbor)
            queue = next_queue


        dist_grid = np.full((self.nx, self.ny), np.inf)
        for idx, (i, j) in enumerate(self.idx_to_node):
            dist_grid[i, j] = distance[idx]
        return dist_grid

    def fractal_dimension_estimate(self):
        max_size = max(self.nx, self.ny)
        sizes = []
        counts = []
        size = max_size
        while size >= 2:
            n_boxes_x = int(np.ceil(self.nx / size))
            n_boxes_y = int(np.ceil(self.ny / size))
            count = 0
            for bx in range(n_boxes_x):
                for by in range(n_boxes_y):
                    i0 = bx * size
                    i1 = min((bx + 1) * size, self.nx)
                    j0 = by * size
                    j1 = min((by + 1) * size, self.ny)

                    has_burned = np.any(self.burned[i0:i1, j0:j1])
                    has_unburned = np.any(1 - self.burned[i0:i1, j0:j1])
                    if has_burned and has_unburned:
                        count += 1
            if count > 0:
                sizes.append(size)
                counts.append(count)
            size //= 2

        if len(counts) < 2:
            return 1.0

        log_counts = np.log(np.array(counts, dtype=float))
        log_sizes = np.log(np.array(sizes, dtype=float))

        A = np.vstack([-log_sizes, np.ones(len(log_sizes))]).T
        D_f, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        return max(1.0, min(2.0, D_f))


def flame_surface_area_evolution_step(c_field, threshold=0.5):
    active = (np.abs(c_field - 0.5) < 0.2).astype(int)
    nx, ny = c_field.shape


    neighbor_count = np.zeros((nx, ny), dtype=int)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbor_count += np.roll(np.roll(active, di, axis=0), dj, axis=1)

    new_c = c_field.copy()

    mask_active = active == 1
    mask_survive = mask_active & ((neighbor_count == 2) | (neighbor_count == 3))
    mask_die = mask_active & ~mask_survive


    mask_inactive = active == 0
    mask_ignite = mask_inactive & (neighbor_count == 3)


    new_c[mask_die & (c_field > 0.5)] += 0.1
    new_c[mask_die & (c_field <= 0.5)] -= 0.1
    new_c[mask_ignite] = 0.5
    new_c[mask_survive] = np.clip(new_c[mask_survive] + 0.02, 0.3, 0.7)

    return np.clip(new_c, 0.0, 1.0)


def track_flame_front_evolution(c_field_list, threshold=0.5):
    metrics = {
        'n_components': [],
        'front_length': [],
        'fractal_dim': [],
        'max_component_size': [],
    }
    for c in c_field_list:
        graph = FlameFrontGraph(c, threshold)
        sizes = graph.component_sizes()
        metrics['n_components'].append(len(sizes))
        metrics['front_length'].append(graph.flame_front_length())
        metrics['fractal_dim'].append(graph.fractal_dimension_estimate())
        metrics['max_component_size'].append(max(sizes) if sizes else 0)
    return metrics
