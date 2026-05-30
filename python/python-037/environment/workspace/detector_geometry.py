
import numpy as np
from typing import List, Tuple
from utils import triangle_area_2d, barycentric_to_cartesian








FEKETE_RULES = {
    1: {
        "degree": 3,
        "points": np.array([
            [1.0/3.0, 1.0/3.0, 1.0/3.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]),
        "weights": np.array([-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0]),
    },
    2: {
        "degree": 6,
        "points": np.array([
            [0.091576213509771, 0.454213840770114, 0.454213840770114],
            [0.454213840770114, 0.091576213509771, 0.454213840770114],
            [0.454213840770114, 0.454213840770114, 0.091576213509771],
            [0.445948490915965, 0.277025758135518, 0.277025758135518],
            [0.277025758135518, 0.445948490915965, 0.277025758135518],
            [0.277025758135518, 0.277025758135518, 0.445948490915965],
            [1.0/3.0, 1.0/3.0, 1.0/3.0],
        ]),
        "weights": np.array([
            0.053347235608838, 0.053347235608838, 0.053347235608838,
            0.077113760890257, 0.077113760890257, 0.077113760890257,
            0.1125,
        ]),
    },
    3: {
        "degree": 7,
        "points": np.array([
            [0.333333333333333, 0.333333333333333, 0.333333333333333],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ]),
        "weights": np.array([
            0.1125,
            0.066197076394253, 0.066197076394253, 0.066197076394253,
            0.062969590272414, 0.062969590272414, 0.062969590272414,
        ]),
    },
}


def get_fekete_rule(rule_id: int) -> Tuple[np.ndarray, np.ndarray, int]:
    if rule_id not in FEKETE_RULES:
        available = list(FEKETE_RULES.keys())
        raise ValueError(f"get_fekete_rule: 不支持的规则编号 {rule_id}，可用: {available}")
    data = FEKETE_RULES[rule_id]
    bary = data["points"]
    weights = data["weights"]
    degree = data["degree"]

    ref_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cart = barycentric_to_cartesian(ref_tri, bary)
    return cart, weights, degree






def reference_to_physical_t3(physical_tri: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    if physical_tri.shape != (3, 2):
        raise ValueError("reference_to_physical_t3: physical_tri 必须为 (3,2)")
    v1 = physical_tri[0]
    J = np.column_stack([
        physical_tri[1] - physical_tri[0],
        physical_tri[2] - physical_tri[0],
    ])
    phys = ref_points @ J.T + v1
    return phys






class Mesh2D:

    def __init__(self):
        self.vertices: np.ndarray = np.zeros((0, 2))
        self.triangles: np.ndarray = np.zeros((0, 3), dtype=int)
        self.edges: np.ndarray = np.zeros((0, 2), dtype=int)
        self.vertex_labels: np.ndarray = np.zeros((0,), dtype=int)
        self.triangle_labels: np.ndarray = np.zeros((0,), dtype=int)

    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    def n_triangles(self) -> int:
        return self.triangles.shape[0]

    def n_edges(self) -> int:
        return self.edges.shape[0]

    def triangle_area(self, tri_idx: int) -> float:
        tri = self.triangles[tri_idx]
        verts = self.vertices[tri]
        return triangle_area_2d(verts)

    def total_area(self) -> float:
        return sum(self.triangle_area(i) for i in range(self.n_triangles()))

    def integrate_scalar(self, scalar_fn, rule_id: int = 2) -> float:
        ref_pts, ref_w, _ = get_fekete_rule(rule_id)
        total = 0.0
        for t_idx in range(self.n_triangles()):
            tri = self.triangles[t_idx]
            phys_tri = self.vertices[tri]
            area = triangle_area_2d(phys_tri)


            phys_pts = reference_to_physical_t3(phys_tri, ref_pts)
            for i in range(len(ref_w)):
                x, y = phys_pts[i]
                total += area * ref_w[i] * scalar_fn(x, y)
        return total






def mesh_base_one(triangles: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return triangles + 1, edges + 1


def mesh_base_zero(triangles: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return triangles - 1, edges - 1


def read_msh_file(filename: str) -> Mesh2D:
    mesh = Mesh2D()
    with open(filename, 'r') as f:
        header = f.readline().strip().split()
        v_num = int(header[0])
        t_num = int(header[1])
        e_num = int(header[2])

        vertices = []
        v_labels = []
        for _ in range(v_num):
            parts = f.readline().strip().split()
            x, y = float(parts[0]), float(parts[1])
            label = int(parts[2]) if len(parts) > 2 else 0
            vertices.append([x, y])
            v_labels.append(label)

        triangles = []
        t_labels = []
        for _ in range(t_num):
            parts = f.readline().strip().split()
            i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
            label = int(parts[3]) if len(parts) > 3 else 0
            triangles.append([i, j, k])
            t_labels.append(label)

        edges = []
        e_labels = []
        for _ in range(e_num):
            parts = f.readline().strip().split()
            i, j = int(parts[0]), int(parts[1])
            label = int(parts[2]) if len(parts) > 2 else 0
            edges.append([i, j])
            e_labels.append(label)

    mesh.vertices = np.array(vertices)
    mesh.vertex_labels = np.array(v_labels, dtype=int)
    mesh.triangles = np.array(triangles, dtype=int)

    if mesh.triangles.min() > 0:
        mesh.triangles -= 1
    mesh.triangle_labels = np.array(t_labels, dtype=int)
    mesh.edges = np.array(edges, dtype=int)
    if mesh.edges.min() > 0:
        mesh.edges -= 1
    return mesh


def write_msh_file(filename: str, mesh: Mesh2D) -> None:
    tri_out, edge_out = mesh_base_one(mesh.triangles.copy(), mesh.edges.copy())
    with open(filename, 'w') as f:
        f.write(f"{mesh.n_vertices()} {mesh.n_triangles()} {mesh.n_edges()}\n")
        for i in range(mesh.n_vertices()):
            x, y = mesh.vertices[i]
            lab = mesh.vertex_labels[i] if i < len(mesh.vertex_labels) else 0
            f.write(f"{x:.16e} {y:.16e} {lab}\n")
        for i in range(mesh.n_triangles()):
            ii, jj, kk = tri_out[i]
            lab = mesh.triangle_labels[i] if i < len(mesh.triangle_labels) else 0
            f.write(f"{ii} {jj} {kk} {lab}\n")
        for i in range(mesh.n_edges()):
            ii, jj = edge_out[i]
            lab = 0
            f.write(f"{ii} {jj} {lab}\n")


def create_sample_detector_mesh() -> Mesh2D:
    mesh = Mesh2D()
    R = 5.0
    n_boundary = 12


    center = np.array([0.0, 0.0])
    verts = [center]
    v_labels = [0]


    angles = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    for a in angles:
        verts.append([R * np.cos(a), R * np.sin(a)])
        v_labels.append(1)

    mesh.vertices = np.array(verts)
    mesh.vertex_labels = np.array(v_labels, dtype=int)


    tri = []
    for i in range(n_boundary):
        i1 = i + 1
        i2 = ((i + 1) % n_boundary) + 1
        tri.append([0, i1, i2])
    mesh.triangles = np.array(tri, dtype=int)
    mesh.triangle_labels = np.zeros(mesh.n_triangles(), dtype=int)


    edges = []
    for i in range(n_boundary):
        i1 = i + 1
        i2 = ((i + 1) % n_boundary) + 1
        edges.append([i1, i2])
    mesh.edges = np.array(edges, dtype=int)

    return mesh






if __name__ == "__main__":

    ref_pts, ref_w, deg = get_fekete_rule(2)

    area_approx = np.sum(ref_w) * 0.5
    assert abs(area_approx - 0.5) < 1e-12, f"Fekete 面积测试失败: {area_approx}"


    mesh = create_sample_detector_mesh()
    assert mesh.n_vertices() == 13
    assert mesh.n_triangles() == 12
    total_area = mesh.total_area()
    expected_area = np.pi * 25.0
    assert abs(total_area - expected_area) / expected_area < 0.15, f"面积偏差过大: {total_area}"


    def f_const(x, y):
        return 1.0
    int_const = mesh.integrate_scalar(f_const, rule_id=2)
    assert abs(int_const - total_area) / total_area < 1e-10, "常数积分测试失败"

    print("detector_geometry.py: 所有自测通过")
