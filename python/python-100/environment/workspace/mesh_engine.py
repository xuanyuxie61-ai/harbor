
import numpy as np
from scipy.spatial import Delaunay


class CVTMeshGenerator:

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def generate_in_annular_sector(
        self,
        r_inner: float,
        r_outer: float,
        theta_min: float,
        theta_max: float,
        n_points: int,
        n_samples_per_point: int = 1000,
        max_iter: int = 50,
        tol: float = 1.0e-6,
    ) -> tuple:
        if r_inner < 0 or r_outer <= r_inner:
            raise ValueError("半径参数无效")
        if theta_max <= theta_min:
            raise ValueError("角度范围无效")


        points = self._sample_annular_sector(
            r_inner, r_outer, theta_min, theta_max, n_points
        )

        for it in range(max_iter):

            tri = Delaunay(points)
            triangles = tri.simplices



            sample_num = n_samples_per_point * n_points
            samples = self._sample_annular_sector(
                r_inner, r_outer, theta_min, theta_max, sample_num
            )


            dists = np.linalg.norm(
                samples[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2
            )
            nearest = np.argmin(dists, axis=1)

            new_points = np.zeros_like(points)
            counts = np.zeros(n_points)
            for i in range(n_points):
                mask = nearest == i
                count = np.sum(mask)
                if count > 0:
                    new_points[i] = np.mean(samples[mask], axis=0)
                    counts[i] = count
                else:

                    new_points[i] = points[i]
                    counts[i] = 1


            displacement = np.max(np.linalg.norm(new_points - points, axis=1))
            points = new_points
            if displacement < tol:
                break


        tri = Delaunay(points)
        triangles = tri.simplices
        return points, triangles

    def _sample_annular_sector(
        self, r_in: float, r_out: float, th_min: float, th_max: float, n: int
    ) -> np.ndarray:

        r_sq = self.rng.uniform(r_in**2, r_out**2, size=n)
        r = np.sqrt(r_sq)
        theta = self.rng.uniform(th_min, th_max, size=n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))


class Mesh2D:


    TAG_AIR_GAP = 1
    TAG_STATOR_CORE = 2
    TAG_ROTOR_CORE = 3
    TAG_MAGNET = 4
    TAG_WINDING = 5
    TAG_SHAFT = 6
    TAG_OUTER_BOUNDARY = 10
    TAG_INNER_BOUNDARY = 11

    def __init__(self):
        self.nodes = np.zeros((0, 2), dtype=float)
        self.elements = np.zeros((0, 3), dtype=int)
        self.node_tags = np.zeros(0, dtype=int)
        self.elem_tags = np.zeros(0, dtype=int)
        self.boundaries = []

    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    def n_elements(self) -> int:
        return self.elements.shape[0]

    def build_from_points_triangles(self, points: np.ndarray, triangles: np.ndarray):
        self.nodes = np.asarray(points, dtype=float)
        self.elements = np.asarray(triangles, dtype=int)
        n_node = self.nodes.shape[0]
        n_elem = self.elements.shape[0]
        self.node_tags = np.zeros(n_node, dtype=int)
        self.elem_tags = np.zeros(n_elem, dtype=int)
        self._validate()

    def _validate(self):
        n_node = self.n_nodes()
        n_elem = self.n_elements()
        if n_elem > 0:
            min_idx = self.elements.min()
            max_idx = self.elements.max()
            if min_idx < 0 or max_idx >= n_node:
                raise ValueError(
                    f"单元节点索引越界: [{min_idx}, {max_idx}], 节点数={n_node}"
                )

        for e in range(n_elem):
            v = self.elements[e]
            if len(set(v)) != 3:
                raise ValueError(f"退化单元 {e}: 节点 {v}")

    def compute_element_areas(self) -> np.ndarray:
        n = self.n_elements()
        areas = np.zeros(n)
        for e in range(n):
            v = self.elements[e]
            p1, p2, p3 = self.nodes[v[0]], self.nodes[v[1]], self.nodes[v[2]]
            areas[e] = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
        return areas

    def compute_quality_metrics(self) -> dict:
        n = self.n_elements()
        aspect_ratios = np.zeros(n)
        min_angles = np.zeros(n)
        max_angles = np.zeros(n)
        areas = np.zeros(n)

        for e in range(n):
            v = self.elements[e]
            p1, p2, p3 = self.nodes[v[0]], self.nodes[v[1]], self.nodes[v[2]]


            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)


            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            areas[e] = area


            if a > 0 and b > 0 and c > 0:
                ang_a = np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1.0, 1.0))
                ang_b = np.arccos(np.clip((a * a + c * c - b * b) / (2 * a * c), -1.0, 1.0))
                ang_c = np.pi - ang_a - ang_b
                angles_deg = np.degrees([ang_a, ang_b, ang_c])
                min_angles[e] = np.min(angles_deg)
                max_angles[e] = np.max(angles_deg)


                R = a * b * c / (4.0 * area + 1.0e-30)

                s = 0.5 * (a + b + c)
                r_in = area / (s + 1.0e-30)
                aspect_ratios[e] = R / (2.0 * r_in + 1.0e-30)
            else:
                aspect_ratios[e] = np.inf
                min_angles[e] = 0.0
                max_angles[e] = 180.0


        valid = areas > 1.0e-14
        metrics = {
            "aspect_ratio_mean": float(np.mean(aspect_ratios[valid])) if np.any(valid) else np.inf,
            "aspect_ratio_max": float(np.max(aspect_ratios[valid])) if np.any(valid) else np.inf,
            "min_angle_mean": float(np.mean(min_angles[valid])) if np.any(valid) else 0.0,
            "min_angle_min": float(np.min(min_angles[valid])) if np.any(valid) else 0.0,
            "max_angle_max": float(np.max(max_angles[valid])) if np.any(valid) else 180.0,
            "area_min": float(np.min(areas[valid])) if np.any(valid) else 0.0,
            "area_max": float(np.max(areas[valid])) if np.any(valid) else 0.0,
            "area_ratio": (
                float(np.max(areas[valid]) / (np.min(areas[valid]) + 1.0e-30))
                if np.any(valid)
                else np.inf
            ),
            "n_degenerate": int(np.sum(~valid)),
        }
        return metrics

    def tag_elements_by_region(self, region_funcs: dict):
        n = self.n_elements()
        self.elem_tags = np.zeros(n, dtype=int)
        centroids = np.zeros((n, 2))
        for e in range(n):
            v = self.elements[e]
            centroids[e] = np.mean(self.nodes[v], axis=0)

        for tag, func in region_funcs.items():
            mask = np.array([func(c[0], c[1]) for c in centroids])
            self.elem_tags[mask] = tag

    def build_connectivity_matrix(self) -> np.ndarray:
        n = self.n_nodes()
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0.0)

        for e in range(self.n_elements()):
            v = self.elements[e]
            for i in range(3):
                for j in range(i + 1, 3):
                    vi, vj = v[i], v[j]
                    dist = np.linalg.norm(self.nodes[vi] - self.nodes[vj])
                    adj[vi, vj] = min(adj[vi, vj], dist)
                    adj[vj, vi] = min(adj[vj, vi], dist)
        return adj

    def floyd_warshall_magnetic_path(self, source_idx: int) -> np.ndarray:
        adj = self.build_connectivity_matrix()
        n = adj.shape[0]
        D = adj.copy()

        for k in range(n):
            for j in range(n):

                d_ik = D[:, k]
                d_kj = D[k, j]
                d_ij = D[:, j]
                D[:, j] = np.minimum(d_ij, d_ik + d_kj)

        return D[source_idx, :]

    def extract_boundary_edges(self) -> list:
        edge_count = {}
        for e in range(self.n_elements()):
            v = self.elements[e]
            edges = [(min(v[i], v[j]), max(v[i], v[j])) for i, j in [(0, 1), (1, 2), (2, 0)]]
            for ed in edges:
                edge_count[ed] = edge_count.get(ed, 0) + 1

        self.boundaries = []
        for ed, count in edge_count.items():
            if count == 1:
                self.boundaries.append((ed[0], ed[1], self.TAG_OUTER_BOUNDARY))
        return self.boundaries

    def write_msh(self, filename: str):
        self.extract_boundary_edges()
        v_num = self.n_nodes()
        t_num = self.n_elements()
        e_num = len(self.boundaries)

        with open(filename, "w") as f:
            f.write(f"{v_num}  {t_num}  {e_num}\n")
            for i in range(v_num):
                tag = int(self.node_tags[i]) if i < len(self.node_tags) else 0
                f.write(f"{self.nodes[i, 0]:.15g}  {self.nodes[i, 1]:.15g}  {tag}\n")
            for e in range(t_num):
                tag = int(self.elem_tags[e]) if e < len(self.elem_tags) else 0
                v = self.elements[e]
                f.write(f"{v[0]+1}  {v[1]+1}  {v[2]+1}  {tag}\n")
            for ed in self.boundaries:
                f.write(f"{ed[0]+1}  {ed[1]+1}  {ed[2]}\n")

    @classmethod
    def read_msh(cls, filename: str):
        mesh = cls()
        with open(filename, "r") as f:
            parts = f.readline().strip().split()
            if len(parts) < 3:
                raise ValueError("MSH文件头格式错误")
            v_num, t_num, e_num = map(int, parts)

            nodes = []
            node_tags = []
            for _ in range(v_num):
                line = f.readline().strip()
                while line == "":
                    line = f.readline().strip()
                vals = line.split()
                nodes.append([float(vals[0]), float(vals[1])])
                node_tags.append(int(vals[2]) if len(vals) > 2 else 0)

            elements = []
            elem_tags = []
            for _ in range(t_num):
                line = f.readline().strip()
                while line == "":
                    line = f.readline().strip()
                vals = line.split()
                elements.append([int(vals[0]) - 1, int(vals[1]) - 1, int(vals[2]) - 1])
                elem_tags.append(int(vals[3]) if len(vals) > 3 else 0)

            boundaries = []
            for _ in range(e_num):
                line = f.readline().strip()
                while line == "":
                    line = f.readline().strip()
                vals = line.split()
                boundaries.append((int(vals[0]) - 1, int(vals[1]) - 1, int(vals[2])))

        mesh.nodes = np.array(nodes, dtype=float)
        mesh.elements = np.array(elements, dtype=int)
        mesh.node_tags = np.array(node_tags, dtype=int)
        mesh.elem_tags = np.array(elem_tags, dtype=int)
        mesh.boundaries = boundaries
        mesh._validate()
        return mesh


def build_simple_pmsm_mesh(
    R_so: float = 0.1,
    R_si: float = 0.08,
    R_ro: float = 0.075,
    R_ri: float = 0.05,
    n_stator: int = 200,
    n_rotor: int = 150,
    n_airgap: int = 80,
    n_slots: int = 6,
    seed: int = 42,
) -> Mesh2D:
    cvt = CVTMeshGenerator(seed=seed)
    all_points = []
    all_triangles = []
    point_offset = 0


    dtheta = 2.0 * np.pi / n_slots
    for k in range(n_slots):
        th1 = k * dtheta + 0.02
        th2 = (k + 1) * dtheta - 0.02
        pts, tri = cvt.generate_in_annular_sector(
            R_si, R_so, th1, th2, n_stator // n_slots, max_iter=20
        )
        all_points.append(pts)
        all_triangles.append(tri + point_offset)
        point_offset += pts.shape[0]


    pts, tri = cvt.generate_in_annular_sector(
        R_ri, R_ro, 0.0, 2.0 * np.pi, n_rotor, max_iter=20
    )
    all_points.append(pts)
    all_triangles.append(tri + point_offset)
    point_offset += pts.shape[0]


    pts, tri = cvt.generate_in_annular_sector(
        R_ro, R_si, 0.0, 2.0 * np.pi, n_airgap, max_iter=20
    )
    all_points.append(pts)
    all_triangles.append(tri + point_offset)
    point_offset += pts.shape[0]


    nodes = np.vstack(all_points)
    elements = np.vstack(all_triangles)

    mesh = Mesh2D()
    mesh.build_from_points_triangles(nodes, elements)


    def in_stator(x, y):
        r = np.sqrt(x * x + y * y)
        return R_si < r < R_so

    def in_rotor(x, y):
        r = np.sqrt(x * x + y * y)
        return R_ri < r < R_ro

    def in_airgap(x, y):
        r = np.sqrt(x * x + y * y)
        return R_ro < r < R_si

    def in_shaft(x, y):
        r = np.sqrt(x * x + y * y)
        return r < R_ri

    mesh.tag_elements_by_region({
        mesh.TAG_STATOR_CORE: in_stator,
        mesh.TAG_ROTOR_CORE: in_rotor,
        mesh.TAG_AIR_GAP: in_airgap,
        mesh.TAG_SHAFT: in_shaft,
    })

    return mesh
