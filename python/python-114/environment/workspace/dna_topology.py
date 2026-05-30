
import numpy as np


class DnaCoarseGrainTopology:

    def __init__(self, num_bases: int = 180, bases_per_bead: int = 3,
                 helix_radius_nm: float = 1.0, rise_per_base_nm: float = 0.34,
                 bases_per_turn: float = 10.5):
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")
        if bases_per_bead <= 0:
            raise ValueError("bases_per_bead must be positive")
        if helix_radius_nm <= 0 or rise_per_base_nm <= 0 or bases_per_turn <= 0:
            raise ValueError("Geometric parameters must be positive")

        self.num_bases = num_bases
        self.bases_per_bead = bases_per_bead
        self.helix_radius = helix_radius_nm
        self.rise_per_base = rise_per_base_nm
        self.bases_per_turn = bases_per_turn
        self.num_beads = int(np.ceil(num_bases / bases_per_bead))

    def generate_helix_coordinates(self) -> np.ndarray:
        n = self.num_beads
        t = np.arange(n, dtype=float)
        angular_freq = 2.0 * np.pi / (self.bases_per_turn / self.bases_per_bead)
        h = self.bases_per_bead * self.rise_per_base

        x = self.helix_radius * np.cos(angular_freq * t)
        y = self.helix_radius * np.sin(angular_freq * t)
        z = h * t

        coords = np.column_stack((x, y, z))
        return coords

    def generate_backbone_tangents(self) -> np.ndarray:
        coords = self.generate_helix_coordinates()
        n = self.num_beads
        tangents = np.zeros_like(coords)


        for i in range(1, n - 1):
            tangents[i] = coords[i + 1] - coords[i - 1]
            norm = np.linalg.norm(tangents[i])
            if norm > 1e-12:
                tangents[i] /= norm


        if n > 1:
            tangents[0] = coords[1] - coords[0]
            tangents[-1] = coords[-1] - coords[-2]
            for idx in (0, -1):
                norm = np.linalg.norm(tangents[idx])
                if norm > 1e-12:
                    tangents[idx] /= norm
        else:
            tangents[0] = np.array([0.0, 0.0, 1.0])

        return tangents

    def compute_persistence_length_correction(self, persistence_length_nm: float = 3.0) -> np.ndarray:
        kB_T = 2.479
        delta_s = self.bases_per_bead * self.rise_per_base
        if delta_s <= 0:
            raise ValueError("bond length must be positive")
        k_bend = kB_T * persistence_length_nm / (delta_s ** 3)
        return np.full(self.num_beads, k_bend)

    def double_resolution_grid(self, grid_1d: np.ndarray) -> np.ndarray:
        if grid_1d.size == 0:
            return grid_1d.copy()
        m = grid_1d.shape[0]
        if grid_1d.ndim == 1:
            out = np.repeat(grid_1d, 2)
        else:

            out = np.zeros((2 * m,) + grid_1d.shape[1:], dtype=grid_1d.dtype)
            for i in range(m):
                out[2 * i] = grid_1d[i]
                out[2 * i + 1] = grid_1d[i]
        return out

    def generate_ssdna_with_bubble(self, bubble_start: int = 60, bubble_length: int = 30) -> dict:
        coords = self.generate_helix_coordinates()
        n = self.num_beads
        is_bubble = np.zeros(n, dtype=bool)


        b_start = max(0, int(bubble_start / self.bases_per_bead))
        b_end = min(n, int((bubble_start + bubble_length) / self.bases_per_bead))
        is_bubble[b_start:b_end] = True


        binding_sites = np.ones(n, dtype=float) * 0.1
        binding_sites[is_bubble] = 1.0

        return {
            'coords': coords,
            'is_bubble': is_bubble,
            'binding_sites': binding_sites,
            'num_beads': n,
            'bubble_indices': np.arange(b_start, b_end)
        }


def generate_sammon_helix(n_points: int, radius: float = 1.0, pitch: float = 3.4) -> np.ndarray:
    z = np.arange(n_points, dtype=float) / np.sqrt(2.0)
    x = radius * np.cos(z)
    y = radius * np.sin(z)
    return np.column_stack((x, y, z * pitch / 3.4))


def compute_worm_like_chain_end_to_end(num_bases: int, persistence_length_nm: float = 3.0,
                                        base_rise_nm: float = 0.34) -> float:
    L = num_bases * base_rise_nm
    if L <= 0:
        return 0.0
    if persistence_length_nm <= 0:
        return L
    ratio = L / persistence_length_nm
    rms = np.sqrt(2.0 * persistence_length_nm * L * (1.0 - (1.0 - np.exp(-ratio)) / ratio))
    return rms


class TetMesh:

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

    def integrate_nodal_values(self, nodal_values: np.ndarray) -> tuple:
        from tet_mesh_core import integrate_over_tet_mesh
        integral, total_volume = integrate_over_tet_mesh(
            self.nodes, self.elements, nodal_values
        )
        return integral, total_volume

    def compute_surface_area(self) -> float:
        face_count = {}
        for e in range(self.n_elements):
            en = self.elements[e, :4]
            faces = [
                tuple(sorted([en[0], en[1], en[2]])),
                tuple(sorted([en[0], en[1], en[3]])),
                tuple(sorted([en[0], en[2], en[3]])),
                tuple(sorted([en[1], en[2], en[3]])),
            ]
            for f in faces:
                face_count[f] = face_count.get(f, 0) + 1

        surface_area = 0.0
        for f, count in face_count.items():
            if count == 1:
                p1, p2, p3 = self.nodes[f[0]], self.nodes[f[1]], self.nodes[f[2]]
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p1 - p3)
                s = 0.5 * (a + b + c)
                area = np.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
                surface_area += area
        return surface_area

    def to_xml_string(self) -> str:
        lines = []
        lines.append('<?xml version="1.0"?>')
        lines.append('<TetMesh>')
        lines.append(f'  <Nodes count="{self.n_nodes}">')
        for i in range(self.n_nodes):
            x, y, z = self.nodes[i]
            lines.append(f'    <Node id="{i}" x="{x:.8e}" y="{y:.8e}" z="{z:.8e}"/>')
        lines.append('  </Nodes>')
        lines.append(f'  <Elements count="{self.n_elements}">')
        for i in range(self.n_elements):
            n1, n2, n3, n4 = self.elements[i]
            lines.append(f'    <Tet id="{i}" n0="{n1}" n1="{n2}" n2="{n3}" n3="{n4}"/>')
        lines.append('  </Elements>')
        lines.append('</TetMesh>')
        return '\n'.join(lines)


def generate_nucleosome_tet_mesh(
    n_rings: int = 4,
    n_theta: int = 8,
    n_z: int = 4,
    major_radius: float = 5.5,
    minor_radius: float = 3.3,
    pitch: float = 2.7,
) -> TetMesh:
    from tet_mesh_core import generate_tet_mesh_box


    nodes_box, elements_box = generate_tet_mesh_box(
        nx=n_rings + 1,
        ny=n_theta + 1,
        nz=n_z + 1,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        zlim=(0.0, 2.0 * np.pi * n_rings),
    )


    nodes = nodes_box.copy()
    for i in range(nodes.shape[0]):
        u = nodes_box[i, 0]
        v = nodes_box[i, 1]
        w = nodes_box[i, 2]


        theta_ring = np.arctan2(v, u) if (abs(u) > 1e-12 or abs(v) > 1e-12) else 0.0
        r_frac = np.sqrt(u * u + v * v)
        r_local = minor_radius * r_frac


        phi = w / (2.0 * np.pi * n_rings) * 2.0 * np.pi * n_rings

        angle = w

        R_eff = major_radius + r_local * np.cos(theta_ring)
        nodes[i, 0] = R_eff * np.cos(angle)
        nodes[i, 1] = R_eff * np.sin(angle)
        nodes[i, 2] = pitch * angle / (2.0 * np.pi) + r_local * np.sin(theta_ring)

    return TetMesh(nodes, elements_box)


def compute_dsb_repair_compartment_volume(
    mesh: TetMesh,
    gamma_density: np.ndarray,
    threshold: float = 0.15,
) -> tuple:
    gamma_density = np.asarray(gamma_density, dtype=np.float64)
    compartment_volume = 0.0
    total_signal = 0.0

    for e in range(mesh.n_elements):
        en = mesh.elements[e, :4]
        p = mesh.nodes[en]
        M = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = abs(np.linalg.det(M)) / 6.0
        avg_density = np.mean(gamma_density[en])
        total_signal += vol * avg_density
        if avg_density > threshold:
            compartment_volume += vol

    return compartment_volume, total_signal
