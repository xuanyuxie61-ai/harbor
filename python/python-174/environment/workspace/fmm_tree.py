
import numpy as np


class OctreeNode:

    def __init__(self, center, half_size, depth=0, max_depth=8, max_particles=10, order=4):
        self.center = np.asarray(center, dtype=float)
        self.half_size = float(half_size)
        self.depth = depth
        self.max_depth = max_depth
        self.max_particles = max_particles
        self.order = order
        self.particle_indices = []
        self.children = None
        self.radius = np.sqrt(3.0) * half_size
        self.parent = None


        self.multipole_moments_real = []
        self.multipole_moments_imag = []
        self.local_coeffs_real = []
        self.local_coeffs_imag = []
        for l in range(order + 1):
            self.multipole_moments_real.append(np.zeros(l + 1))
            self.multipole_moments_imag.append(np.zeros(l + 1))
            self.local_coeffs_real.append(np.zeros(l + 1))
            self.local_coeffs_imag.append(np.zeros(l + 1))

    def is_leaf(self):
        return self.children is None

    def contains(self, point):
        point = np.asarray(point)
        return np.all(np.abs(point - self.center) <= self.half_size + 1e-12)

    def insert(self, point_idx, point, all_points=None):
        if self.is_leaf():
            self.particle_indices.append(point_idx)
            if (len(self.particle_indices) > self.max_particles
                    and self.depth < self.max_depth):
                if all_points is None:
                    all_points = point
                self._split(all_points)
        else:

            child_idx = self._child_index(point)
            self.children[child_idx].insert(point_idx, point, all_points)

    def _child_index(self, point):
        dx = point[0] >= self.center[0]
        dy = point[1] >= self.center[1]
        dz = point[2] >= self.center[2]
        return (int(dx) << 2) | (int(dy) << 1) | int(dz)

    def _split(self, points):
        if self.children is not None:
            return
        h = self.half_size * 0.5
        offsets = [
            [-h, -h, -h], [-h, -h,  h], [-h,  h, -h], [-h,  h,  h],
            [ h, -h, -h], [ h, -h,  h], [ h,  h, -h], [ h,  h,  h]
        ]
        self.children = []
        for off in offsets:
            child = OctreeNode(
                self.center + np.array(off),
                h,
                self.depth + 1,
                self.max_depth,
                self.max_particles,
                self.order
            )
            child.parent = self
            self.children.append(child)


        temp_indices = self.particle_indices.copy()
        self.particle_indices = []
        for idx in temp_indices:
            child_idx = self._child_index(points[idx])
            self.children[child_idx].particle_indices.append(idx)

    def refine_cvt(self, points, charges, max_iter=5):
        if len(self.particle_indices) == 0:
            return
        local_points = points[self.particle_indices]
        local_charges = np.abs(charges[self.particle_indices])
        total_c = np.sum(local_charges)
        if total_c < 1e-15:
            return
        for _ in range(max_iter):
            centroid = np.sum(local_charges[:, None] * local_points, axis=0) / total_c
            self.center = 0.5 * self.center + 0.5 * centroid

    def collect_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.collect_leaves())
        return leaves

    def get_all_nodes(self):
        nodes = [self]
        if self.children is not None:
            for child in self.children:
                nodes.extend(child.get_all_nodes())
        return nodes

    def bounding_quadrilateral_area(self):
        hs = self.half_size
        quad = np.array([
            [self.center[0] - hs, self.center[1] - hs],
            [self.center[0] + hs, self.center[1] - hs],
            [self.center[0] + hs, self.center[1] + hs],
            [self.center[0] - hs, self.center[1] + hs]
        ])

        area = (2 * hs) * (2 * hs)
        return area, quad

    def well_separated_from(self, other, separation_param=2.0):
        dist = np.linalg.norm(self.center - other.center)
        return dist > separation_param * max(self.radius, other.radius)

    def is_adjacent(self, other):

        dist_per_dim = np.abs(self.center - other.center)
        sum_half = self.half_size + other.half_size
        return np.all(dist_per_dim <= sum_half + 1e-12)

    def get_neighbors(self, all_nodes):
        neighbors = []
        for node in all_nodes:
            if node is not self and self.is_adjacent(node):
                neighbors.append(node)
        return neighbors

    def get_interaction_list(self, all_nodes):
        if self.parent is None:
            return []
        interaction = []
        parent_neighbors = self.parent.get_neighbors(all_nodes)
        for pn in parent_neighbors:
            if pn.is_leaf():
                if pn is not self and not self.is_adjacent(pn) and pn not in interaction:
                    interaction.append(pn)
            else:
                for child in pn.children:
                    if child is not self and not self.is_adjacent(child) and child not in interaction:
                        interaction.append(child)
        return interaction


class FMMOctree:

    def __init__(self, points, charges, max_depth=6, max_particles=20, order=4, separation_param=2.0):
        self.points = np.asarray(points, dtype=float)
        self.charges = np.asarray(charges, dtype=float)
        self.N = self.points.shape[0]
        self.max_depth = max_depth
        self.max_particles = max_particles
        self.order = order
        self.separation_param = separation_param


        min_coord = np.min(self.points, axis=0)
        max_coord = np.max(self.points, axis=0)
        center = 0.5 * (min_coord + max_coord)
        size = np.max(max_coord - min_coord)
        if size < 1e-10:
            size = 1.0
        half_size = size * 0.5 * 1.01

        self.root = OctreeNode(center, half_size, depth=0, max_depth=max_depth,
                               max_particles=max_particles, order=order)


        for i in range(self.N):
            self.root.insert(i, self.points[i], self.points)


        self._redistribute_particles(self.root)


        leaves = self.root.collect_leaves()
        for leaf in leaves:
            leaf.refine_cvt(self.points, self.charges)

    def _redistribute_particles(self, node):
        if node.is_leaf():
            return

        particles_to_redistribute = []

        for child in node.children:
            particles_to_redistribute.extend(child.particle_indices)
            child.particle_indices = []




        pass

    def _build_downward(self, node):
        if node.is_leaf():
            return

        if len(node.particle_indices) > 0:
            temp = node.particle_indices.copy()
            node.particle_indices = []
            for idx in temp:
                child_idx = node._child_index(self.points[idx])
                node.children[child_idx].particle_indices.append(idx)
        for child in node.children:
            self._build_downward(child)

    def rebuild(self):

        min_coord = np.min(self.points, axis=0)
        max_coord = np.max(self.points, axis=0)
        center = 0.5 * (min_coord + max_coord)
        size = np.max(max_coord - min_coord)
        if size < 1e-10:
            size = 1.0
        half_size = size * 0.5 * 1.01

        self.root = OctreeNode(center, half_size, depth=0, max_depth=self.max_depth,
                               max_particles=self.max_particles, order=self.order)
        for i in range(self.N):
            self.root.insert(i, self.points[i])
        self._build_downward(self.root)

    def get_all_nodes(self):
        return self.root.get_all_nodes()

    def get_leaves(self):
        return self.root.collect_leaves()

    def compute_moments_upward(self):







        raise NotImplementedError("Hole_2: 请实现compute_moments_upward")
