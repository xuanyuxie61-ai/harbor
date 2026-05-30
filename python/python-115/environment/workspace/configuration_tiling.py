
import numpy as np


class PentominoShapes:

    SHAPES = {
        'F': np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),
        'I': np.array([[1, 1, 1, 1, 1]]),
        'L': np.array([[0, 0, 0, 1], [1, 1, 1, 1]]),
        'N': np.array([[1, 1, 0, 0], [0, 1, 1, 1]]),
        'P': np.array([[1, 1], [1, 1], [1, 0]]),
        'T': np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
        'U': np.array([[1, 0, 1], [1, 1, 1]]),
        'V': np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),
        'W': np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
        'X': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        'Y': np.array([[0, 0, 1, 0], [1, 1, 1, 1]]),
        'Z': np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
    }

    @classmethod
    def get_shape(cls, name):
        name = name.upper()
        if name not in cls.SHAPES:
            raise ValueError(f"未知 Pentomino 形状: {name}")
        return cls.SHAPES[name].copy()

    @classmethod
    def all_shapes(cls):
        return {k: v.copy() for k, v in cls.SHAPES.items()}


class ConfigurationTiling:

    def __init__(self, xi_range, eta_range, n_xi=20, n_eta=20):
        self.xi_min, self.xi_max = xi_range
        self.eta_min, self.eta_max = eta_range
        self.n_xi = n_xi
        self.n_eta = n_eta
        self.dxi = (self.xi_max - self.xi_min) / n_xi
        self.deta = (self.eta_max - self.eta_min) / n_eta

    def grid_to_physical(self, i, j):
        xi = self.xi_min + (i + 0.5) * self.dxi
        eta = self.eta_min + (j + 0.5) * self.deta
        return xi, eta

    def physical_to_grid(self, xi, eta):
        i = int((xi - self.xi_min) / self.dxi)
        j = int((eta - self.eta_min) / self.deta)
        i = max(0, min(i, self.n_xi - 1))
        j = max(0, min(j, self.n_eta - 1))
        return i, j

    def tile_coverage(self, energy_func, energy_cutoff):
        accessible = np.zeros((self.n_xi, self.n_eta), dtype=bool)
        energy_grid = np.zeros((self.n_xi, self.n_eta), dtype=float)

        for i in range(self.n_xi):
            for j in range(self.n_eta):
                xi, eta = self.grid_to_physical(i, j)
                e = energy_func(xi, eta)
                energy_grid[i, j] = e
                if e < energy_cutoff:
                    accessible[i, j] = True


        samples = []
        coverage = accessible.copy()


        t_shape = PentominoShapes.get_shape('T')
        sh, sw = t_shape.shape

        for i in range(self.n_xi - sw + 1):
            for j in range(self.n_eta - sh + 1):
                if np.all(coverage[i:i + sw, j:j + sh]):

                    cx = i + sw // 2
                    cy = j + sh // 2
                    xi_s, eta_s = self.grid_to_physical(cx, cy)
                    samples.append((xi_s, eta_s, energy_grid[cx, cy]))
                    coverage[i:i + sw, j:j + sh] = False

        coverage_ratio = np.sum(accessible) / (self.n_xi * self.n_eta)
        return coverage_ratio, samples, energy_grid

    def rotate_tile(self, tile, k):
        return np.rot90(tile, k=k)

    def reflect_tile(self, tile):
        return np.fliplr(tile)


class ConfigurationSpaceSampler:

    def __init__(self, n_atoms, temperature=300.0):
        self.n_atoms = n_atoms
        self.kB = 0.0019872041
        self.T = temperature
        self.beta = 1.0 / (self.kB * temperature)

    def metropolis_sampling(self, energy_func, x0, n_steps=1000, step_size=0.1):
        x = np.asarray(x0, dtype=float).copy()
        e_curr = energy_func(x)
        samples = [x.copy()]
        energies = [e_curr]
        n_accept = 0

        for _ in range(n_steps):
            x_trial = x + np.random.randn(len(x)) * step_size
            e_trial = energy_func(x_trial)
            delta_e = e_trial - e_curr

            if delta_e < 0 or np.random.rand() < np.exp(-self.beta * delta_e):
                x = x_trial
                e_curr = e_trial
                n_accept += 1

            samples.append(x.copy())
            energies.append(e_curr)

        acceptance_ratio = n_accept / n_steps
        return np.array(samples), np.array(energies), acceptance_ratio

    def reaction_path_sampling(self, energy_func, x_reactant, x_product, n_images=20):
        x_R = np.asarray(x_reactant, dtype=float)
        x_P = np.asarray(x_product, dtype=float)
        lambdas = np.linspace(0, 1, n_images)
        path = []
        energies = []

        for lam in lambdas:
            x_img = (1.0 - lam) * x_R + lam * x_P
            path.append(x_img)
            energies.append(energy_func(x_img))

        return np.array(path), np.array(energies), lambdas
