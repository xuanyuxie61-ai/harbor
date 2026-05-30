
import numpy as np



LIQUID_DROP_PARAMS = {
    'volume': 15.75,
    'surface': 17.8,
    'coulomb': 0.711,
    'asymmetry': 23.7,
    'pairing': 11.18,
}


class Nuclide:

    def __init__(self, Z, A, mass_excess=None):
        self.Z = int(Z)
        self.A = int(A)
        self.N = self.A - self.Z
        self.mass_excess = mass_excess

    def binding_energy(self):
        a_v = LIQUID_DROP_PARAMS['volume']
        a_s = LIQUID_DROP_PARAMS['surface']
        a_c = LIQUID_DROP_PARAMS['coulomb']
        a_a = LIQUID_DROP_PARAMS['asymmetry']
        a_p = LIQUID_DROP_PARAMS['pairing']


        B_vol = a_v * self.A

        B_sur = -a_s * (self.A ** (2.0 / 3.0))

        B_cou = -a_c * self.Z * (self.Z - 1.0) / (self.A ** (1.0 / 3.0))

        B_asym = -a_a * ((self.A - 2.0 * self.Z) ** 2) / self.A

        delta = 0.0
        if self.Z % 2 == 0 and self.N % 2 == 0:
            delta = a_p / np.sqrt(self.A)
        elif self.Z % 2 == 1 and self.N % 2 == 1:
            delta = -a_p / np.sqrt(self.A)

        return B_vol + B_sur + B_cou + B_asym + delta

    def neutron_separation_energy(self):
        if self.A <= 1:
            return 0.0

        this_BE = self.binding_energy()

        prev = Nuclide(self.Z, self.A - 1)
        prev_BE = prev.binding_energy()
        return this_BE - prev_BE

    def proton_separation_energy(self):
        if self.Z <= 1 or self.A <= 1:
            return 0.0
        this_BE = self.binding_energy()
        prev = Nuclide(self.Z - 1, self.A - 1)
        prev_BE = prev.binding_energy()
        return this_BE - prev_BE

    def __repr__(self):
        return f"Nuclide(Z={self.Z}, A={self.A}, BE={self.binding_energy():.3f} MeV)"


class NuclearDataAggregator:

    def __init__(self):
        self.nuclides = {}

    def add(self, nuclide):
        key = (nuclide.Z, nuclide.A)
        self.nuclides[key] = nuclide

    def aggregate_by_Z(self):
        stats = {}
        for (Z, A), nuc in self.nuclides.items():
            if Z not in stats:
                stats[Z] = {'count': 0, 'A_list': [], 'BE_list': []}
            stats[Z]['count'] += 1
            stats[Z]['A_list'].append(A)
            stats[Z]['BE_list'].append(nuc.binding_energy())


        for Z in stats:
            stats[Z]['A_mean'] = np.mean(stats[Z]['A_list'])
            stats[Z]['BE_mean'] = np.mean(stats[Z]['BE_list'])
            stats[Z]['BE_max'] = np.max(stats[Z]['BE_list'])
            stats[Z]['BE_min'] = np.min(stats[Z]['BE_list'])
        return stats

    def get_mass_table_array(self):
        data = []
        for (Z, A), nuc in self.nuclides.items():
            data.append([Z, A, nuc.N, nuc.binding_energy(),
                         nuc.neutron_separation_energy(),
                         nuc.proton_separation_energy()])
        return np.array(data)


class SphericalShellMesh:

    def __init__(self, R_max=15.0, n_r=100, n_theta=30, n_phi=60):
        self.R_max = R_max
        self.n_r = n_r
        self.n_theta = n_theta
        self.n_phi = n_phi



        t = np.linspace(0.0, 1.0, n_r)
        self.r_nodes = R_max * (t ** 1.5)
        self.r_nodes[0] = 1e-6


        self.theta_nodes = np.linspace(0.0, np.pi, n_theta)
        self.phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_phi)

        self.n_vertices = n_r * n_theta * n_phi
        self.n_elements = (n_r - 1) * (n_theta - 1) * (n_phi - 1)

    def get_vertex_coordinates(self):
        r = self.r_nodes
        theta = self.theta_nodes
        phi = self.phi_nodes
        R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')
        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)
        return X.flatten(), Y.flatten(), Z.flatten()

    def write_mesh_file(self, filename):
        X, Y, Z = self.get_vertex_coordinates()
        with open(filename, 'w') as f:
            f.write(f"# SphericalShellMesh: {self.n_r}x{self.n_theta}x{self.n_phi}\n")
            f.write(f"{self.n_vertices} {self.n_elements} 0\n")
            for i in range(self.n_vertices):
                f.write(f"{i+1} {X[i]:.8e} {Y[i]:.8e} {Z[i]:.8e} 0\n")

    def read_mesh_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        header = lines[1].strip().split()
        n_v = int(header[0])
        coords = np.zeros((n_v, 3))
        for i in range(n_v):
            parts = lines[3 + i].strip().split()
            coords[i, :] = [float(parts[1]), float(parts[2]), float(parts[3])]
        return coords


def generate_nuclear_mass_table(Z_range, A_range_func):
    agg = NuclearDataAggregator()
    for Z in Z_range:
        A_min, A_max = A_range_func(Z)
        for A in range(A_min, A_max + 1):
            if A >= Z:
                nuc = Nuclide(Z, A)
                agg.add(nuc)
    return agg


def compute_q_value_reaction(Z_target, A_target, Z_proj, A_proj, Z_out, A_out):
    target = Nuclide(Z_target, A_target)
    proj = Nuclide(Z_proj, A_proj)
    Z_res = Z_target + Z_proj - Z_out
    A_res = A_target + A_proj - A_out
    if Z_res < 0 or A_res < Z_res:
        return -999.0
    residual = Nuclide(Z_res, A_res)
    outgoing = Nuclide(Z_out, A_out)

    Q = (residual.binding_energy() + outgoing.binding_energy()
         - target.binding_energy() - proj.binding_energy())
    return Q


if __name__ == "__main__":

    nuc = Nuclide(26, 56)
    print(nuc)
    print(f"S_n = {nuc.neutron_separation_energy():.3f} MeV")
    print(f"S_p = {nuc.proton_separation_energy():.3f} MeV")

    agg = generate_nuclear_mass_table(range(20, 30), lambda Z: (Z + 20, Z + 40))
    stats = agg.aggregate_by_Z()
    print(f"Z=26 同位素数目: {stats[26]['count']}")

    mesh = SphericalShellMesh(R_max=10.0, n_r=20, n_theta=10, n_phi=20)
    print(f"网格顶点数: {mesh.n_vertices}")

    Q = compute_q_value_reaction(26, 56, 0, 1, 0, 1)
    print(f"n + 56Fe -> n + 56Fe 的 Q 值 ≈ {Q:.3f} MeV")
