
import numpy as np


class SphericalTriMesh:

    def __init__(self, refinement_level: int = 3, radius: float = 6371.0):
        if refinement_level < 0 or refinement_level > 6:
            raise ValueError("refinement_level 必须在 [0, 6] 范围内")
        if radius <= 0:
            raise ValueError("地球半径必须为正")

        self.radius = float(radius)
        self.refinement_level = refinement_level
        self._build_icosahedron()
        for _ in range(refinement_level):
            self._refine()
        self._project_to_sphere()
        self._compute_element_areas()
        self._compute_node_weights()

    def _build_icosahedron(self):
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ], dtype=np.float64)

        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.int32)

        self.vertices = vertices
        self.faces = faces
        self.n_nodes = len(vertices)
        self.n_elements = len(faces)

    def _refine(self):
        new_vertices = list(self.vertices)
        edge_midpoint = {}
        new_faces = []

        def get_midpoint(vi, vj):
            key = tuple(sorted((vi, vj)))
            if key not in edge_midpoint:
                mid = (new_vertices[vi] + new_vertices[vj]) / 2.0
                edge_midpoint[key] = len(new_vertices)
                new_vertices.append(mid)
            return edge_midpoint[key]

        for face in self.faces:
            v0, v1, v2 = face
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            new_faces.append([v0, a, c])
            new_faces.append([v1, b, a])
            new_faces.append([v2, c, b])
            new_faces.append([a, b, c])

        self.vertices = np.array(new_vertices, dtype=np.float64)
        self.faces = np.array(new_faces, dtype=np.int32)
        self.n_nodes = len(self.vertices)
        self.n_elements = len(self.faces)

    def _project_to_sphere(self):
        norms = np.linalg.norm(self.vertices, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        self.vertices = self.radius * (self.vertices / norms)

    def _compute_element_areas(self):
        self.element_areas = np.zeros(self.n_elements, dtype=np.float64)
        for i, face in enumerate(self.faces):
            v0 = self.vertices[face[0]] / self.radius
            v1 = self.vertices[face[1]] / self.radius
            v2 = self.vertices[face[2]] / self.radius


            a = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            b = np.arccos(np.clip(np.dot(v2, v0), -1.0, 1.0))
            c = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0))

            s = 0.5 * (a + b + c)

            s = np.clip(s, 1e-15, np.pi - 1e-15)
            tan_val = (
                np.tan(s / 2.0)
                * np.tan(np.clip((s - a) / 2.0, 1e-15, np.pi))
                * np.tan(np.clip((s - b) / 2.0, 1e-15, np.pi))
                * np.tan(np.clip((s - c) / 2.0, 1e-15, np.pi))
            )
            tan_val = max(tan_val, 0.0)
            E = 4.0 * np.arctan(np.sqrt(tan_val))
            self.element_areas[i] = E * self.radius ** 2

    def _compute_node_weights(self):
        self.node_weights = np.zeros(self.n_nodes, dtype=np.float64)
        for i, face in enumerate(self.faces):
            area = self.element_areas[i]
            for vi in face:
                self.node_weights[vi] += area / 3.0

        total_area = 4.0 * np.pi * self.radius ** 2
        weight_sum = np.sum(self.node_weights)
        if weight_sum > 0:
            self.node_weights *= (total_area / weight_sum)

    def get_lat_lon(self, idx: int):
        x, y, z = self.vertices[idx]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r < 1e-15:
            return 0.0, 0.0
        lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon

    def triangle_grid_points(self, face_idx: int, n_sub: int = 4):
        if n_sub < 1:
            n_sub = 1
        face = self.faces[face_idx]
        v0 = self.vertices[face[0]]
        v1 = self.vertices[face[1]]
        v2 = self.vertices[face[2]]

        points = []
        for i in range(n_sub + 1):
            for j in range(n_sub + 1 - i):
                k = n_sub - i - j
                lam0 = i / float(n_sub)
                lam1 = j / float(n_sub)
                lam2 = k / float(n_sub)
                p = lam0 * v0 + lam1 * v1 + lam2 * v2

                norm = np.linalg.norm(p)
                if norm > 1e-15:
                    p = self.radius * (p / norm)
                points.append(p)
        return np.array(points, dtype=np.float64)

    def pyramid_quadrature_on_element(self, face_idx: int, precision: int = 5):
        face = self.faces[face_idx]
        v0 = self.vertices[face[0]]
        v1 = self.vertices[face[1]]
        v2 = self.vertices[face[2]]


        centroid = (v0 + v1 + v2) / 3.0
        centroid = self.radius * (centroid / np.linalg.norm(centroid))

        e1 = v1 - v0
        e1 = e1 / (np.linalg.norm(e1) + 1e-15)
        e2 = v2 - v0
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / (np.linalg.norm(e2) + 1e-15)



        if precision <= 2:

            local_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]])
            local_wts = np.array([1.0])
        elif precision <= 4:

            local_pts = np.array([
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
                [1.0 / 6.0, 1.0 / 6.0],
            ])
            local_wts = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        else:

            a1 = 0.059715871789770
            b1 = 0.797426985353087
            a2 = 0.797426985353087
            b2 = 0.101286507323456
            a3 = 0.101286507323456
            b3 = 0.059715871789770
            local_pts = np.array([
                [1.0 / 3.0, 1.0 / 3.0],
                [a1, b1], [b1, a1], [1.0 - a1 - b1, a1],
                [a2, b2], [b2, a2], [1.0 - a2 - b2, a2],
            ])
            w0 = 0.225000000000000
            w1 = 0.132394152788506
            w2 = 0.125939180544827
            local_wts = np.array([w0, w1, w1, w1, w2, w2, w2])


        quad_points = []
        for pt, wt in zip(local_pts, local_wts):
            lam0 = 1.0 - pt[0] - pt[1]
            lam1 = pt[0]
            lam2 = pt[1]
            p = lam0 * v0 + lam1 * v1 + lam2 * v2

            norm = np.linalg.norm(p)
            if norm > 1e-15:
                p = self.radius * (p / norm)
            quad_points.append(p)

        quad_points = np.array(quad_points, dtype=np.float64)

        area = self.element_areas[face_idx]
        quad_weights = local_wts * area
        return quad_points, quad_weights

    def tetrahedral_volume_integral(self, field_values: np.ndarray):
        if len(field_values) != self.n_nodes:
            raise ValueError("场值维度必须等于节点数")

        total = 0.0
        volume_sum = 0.0
        shell_thickness = 1.0

        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            v3 = v0 + shell_thickness * normal

            mat = np.array([v1 - v0, v2 - v0, v3 - v0])
            vol = abs(np.linalg.det(mat)) / 6.0

            avg_val = np.mean(field_values[face])
            total += avg_val * vol
            volume_sum += vol

        return total, volume_sum

    def get_neighbors(self, node_idx: int):
        neighbors = set()
        for face in self.faces:
            if node_idx in face:
                for vi in face:
                    if vi != node_idx:
                        neighbors.add(vi)
        return list(neighbors)


def compute_global_mean_temperature(mesh: SphericalTriMesh, temperature: np.ndarray):
    if len(temperature) != mesh.n_nodes:
        raise ValueError("温度场维度与网格节点数不匹配")

    weights = mesh.node_weights
    weight_sum = np.sum(weights)
    if weight_sum < 1e-15:
        return np.mean(temperature)

    return np.sum(temperature * weights) / weight_sum
