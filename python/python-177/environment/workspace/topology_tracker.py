# -*- coding: utf-8 -*-

import numpy as np


class TopologyTracker:

    def __init__(self, levelset):
        self.ls = levelset
        self.history = {
            'num_components': [],
            'volumes': [],
            'euler_chars': []
        }

    def _build_interface_graph(self, band_width=None):
        phi = self.ls.phi
        nx, ny = self.ls.nx, self.ls.ny
        if band_width is None:
            band_width = 2.0 * max(self.ls.dx, self.ls.dy)

        nodes = []
        node_index = {}
        idx = 0
        for i in range(nx):
            for j in range(ny):
                if np.abs(phi[i, j]) < band_width:
                    nodes.append((i, j))
                    node_index[(i, j)] = idx
                    idx += 1

        num_nodes = len(nodes)
        adj = {k: [] for k in range(num_nodes)}


        for k, (i, j) in enumerate(nodes):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) in node_index:
                    adj[k].append(node_index[(ni, nj)])

        return adj, nodes

    def bfs_distance(self, adj, start_node, num_nodes):
        distance = np.full(num_nodes, np.inf)
        distance[start_node] = 0
        queue = [start_node]
        head = 0

        while head < len(queue):
            current = queue[head]
            head += 1
            d = distance[current]
            for neighbor in adj[current]:
                if distance[neighbor] == np.inf:
                    distance[neighbor] = d + 1
                    queue.append(neighbor)

        return distance

    def find_connected_components(self, band_width=None):
        adj, nodes = self._build_interface_graph(band_width)
        num_nodes = len(nodes)
        if num_nodes == 0:
            return [], []

        visited = np.zeros(num_nodes, dtype=bool)
        components = []

        for start in range(num_nodes):
            if not visited[start]:
                dists = self.bfs_distance(adj, start, num_nodes)
                component = [i for i in range(num_nodes) if dists[i] < np.inf]
                for i in component:
                    visited[i] = True
                components.append(component)

        return components, nodes

    def compute_euler_characteristic_approx(self, band_width=None):
        phi = self.ls.phi
        nx, ny = self.ls.nx, self.ls.ny
        if band_width is None:
            band_width = 2.0 * max(self.ls.dx, self.ls.dy)


        face_count = 0
        edge_count = 0
        vertex_count = 0

        face_mask = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if np.abs(phi[i, j]) < band_width:
                    face_mask[i, j] = True
                    face_count += 1


        for i in range(nx - 1):
            for j in range(ny):
                if face_mask[i, j] and face_mask[i + 1, j]:
                    edge_count += 1
        for i in range(nx):
            for j in range(ny - 1):
                if face_mask[i, j] and face_mask[i, j + 1]:
                    edge_count += 1


        for i in range(nx):
            for j in range(ny):
                if face_mask[i, j]:
                    vertex_count += 1

        chi = vertex_count - edge_count + face_count
        return chi

    def detect_topological_event(self, prev_components, curr_components):
        if curr_components > prev_components:
            return "SPLIT"
        elif curr_components < prev_components:
            return "MERGE"
        else:
            return "NO_CHANGE"

    def update_history(self):
        components, nodes = self.find_connected_components()
        num_comp = len(components)
        self.history['num_components'].append(num_comp)


        volumes = []
        dx, dy = self.ls.dx, self.ls.dy
        phi = self.ls.phi
        for comp in components:
            vol = 0.0
            for idx in comp:
                i, j = nodes[idx]
                if phi[i, j] < 0:
                    vol += dx * dy
            volumes.append(vol)
        self.history['volumes'].append(volumes)

        chi = self.compute_euler_characteristic_approx()
        self.history['euler_chars'].append(chi)

    def get_summary(self):
        n_steps = len(self.history['num_components'])
        if n_steps == 0:
            return "No topology history recorded."

        summary = []
        summary.append(f"Topology evolution over {n_steps} time steps:")
        summary.append(f"  Initial components: {self.history['num_components'][0]}")
        summary.append(f"  Final components: {self.history['num_components'][-1]}")
        summary.append(f"  Max components: {max(self.history['num_components'])}")
        summary.append(f"  Min components: {min(self.history['num_components'])}")

        events = []
        for i in range(1, n_steps):
            event = self.detect_topological_event(
                self.history['num_components'][i - 1],
                self.history['num_components'][i]
            )
            if event != "NO_CHANGE":
                events.append(f"Step {i}: {event}")

        if events:
            summary.append("  Detected events:")
            for ev in events:
                summary.append(f"    {ev}")
        else:
            summary.append("  No topological events detected.")

        return "\n".join(summary)
