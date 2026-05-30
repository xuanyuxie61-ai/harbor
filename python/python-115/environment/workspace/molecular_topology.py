
import numpy as np



COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
    'P': 1.07, 'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'Fe': 1.32, 'Zn': 1.22, 'Mg': 1.41, 'Ca': 1.76, 'Na': 1.66,
    'K': 1.96, 'Cu': 1.32, 'Mn': 1.39, 'Co': 1.26, 'Ni': 1.21
}


DEFAULT_RADIUS = 1.5


class XYZParser:

    @staticmethod
    def parse_string(xyz_string):
        lines = xyz_string.strip().split('\n')
        if len(lines) < 3:
            raise ValueError("XYZ 数据格式错误")

        n_atoms = int(lines[0].strip())

        atoms = []
        coords = []

        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            if len(parts) < 4:
                raise ValueError(f"XYZ 第 {i+1} 行格式错误")
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(element)
            coords.append([x, y, z])

        return atoms, np.array(coords, dtype=float)

    @staticmethod
    def generate_demo_xyz():

        xyz_data = """18
demo enzyme active site
C   0.000   0.000   0.000
O   1.200   0.000   0.000
O  -0.800   0.900   0.000
C  -0.500  -1.200   0.000
N   1.500   1.200   0.500
C   2.800   1.300   0.200
O   3.500   0.400  -0.100
C   3.200   2.600   0.300
N  -1.500   0.200   0.800
C  -2.200   1.300   1.200
O  -1.800   2.400   1.500
C  -3.600   1.100   1.200
O   0.200  -2.300   0.200
H  -1.100  -1.200  -0.900
H   0.100   0.800   0.800
H   3.700   3.100  -0.500
H   2.600   3.200   1.000
S  -4.500   2.400   0.800
"""
        return xyz_data


class MolecularGraph:

    def __init__(self, atoms, coordinates):
        self.atoms = atoms
        self.coords = np.asarray(coordinates, dtype=float)
        self.n_atoms = len(atoms)
        self.adjacency = [[] for _ in range(self.n_atoms)]
        self.bonds = []

    def build_bonds(self, scale_factor=1.2):
        for i in range(self.n_atoms):
            ri = COVALENT_RADII.get(self.atoms[i], DEFAULT_RADIUS)
            for j in range(i + 1, self.n_atoms):
                rj = COVALENT_RADII.get(self.atoms[j], DEFAULT_RADIUS)
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                cutoff = scale_factor * (ri + rj)
                if dist < cutoff and dist > 0.3:
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)
                    self.bonds.append((i, j, dist))

    def get_degree(self, i):
        return len(self.adjacency[i])

    def get_neighbors(self, i):
        return self.adjacency[i].copy()

    def metis_format(self):
        n_edges = len(self.bonds)
        lines = [f"{self.n_atoms} {n_edges}"]
        for i in range(self.n_atoms):
            neighbors = sorted(self.adjacency[i])
            if neighbors:
                lines.append(" ".join(str(n + 1) for n in neighbors))
            else:
                lines.append("")
        return "\n".join(lines)

    def connected_components(self):
        visited = [False] * self.n_atoms
        components = []

        for start in range(self.n_atoms):
            if not visited[start]:
                comp = []
                stack = [start]
                visited[start] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for neigh in self.adjacency[node]:
                        if not visited[neigh]:
                            visited[neigh] = True
                            stack.append(neigh)
                components.append(comp)

        return components

    def find_catalytic_triad(self):
        triads = []
        for i in range(self.n_atoms):
            if self.atoms[i] in ['O', 'N']:
                for j in self.adjacency[i]:
                    if self.atoms[j] in ['C', 'N']:
                        for k in self.adjacency[j]:
                            if k != i and self.atoms[k] in ['O', 'N', 'S']:
                                d_ik = np.linalg.norm(self.coords[i] - self.coords[k])
                                if 2.5 < d_ik < 4.0:
                                    triads.append((i, j, k, d_ik))
        return triads

    def subgraph_around_atom(self, center_idx, radius=3):
        visited = {center_idx}
        frontier = {center_idx}
        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                for neigh in self.adjacency[node]:
                    if neigh not in visited:
                        visited.add(neigh)
                        new_frontier.add(neigh)
            frontier = new_frontier
        return sorted(list(visited))


def analyze_molecular_topology(atoms, coords):
    graph = MolecularGraph(atoms, coords)
    graph.build_bonds()

    results = {
        'n_atoms': graph.n_atoms,
        'n_bonds': len(graph.bonds),
        'components': graph.connected_components(),
        'n_components': len(graph.connected_components()),
        'triads': graph.find_catalytic_triad(),
        'metis_graph': graph.metis_format()
    }


    avg_degree = np.mean([graph.get_degree(i) for i in range(graph.n_atoms)])
    results['avg_degree'] = avg_degree

    return graph, results
