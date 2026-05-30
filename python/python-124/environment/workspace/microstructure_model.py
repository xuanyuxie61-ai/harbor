
import numpy as np
from typing import Dict, Tuple, List





PENTOMINO_SHAPES: Dict[str, np.ndarray] = {
    'F': np.array([[0, 1, 1],
                   [1, 1, 0],
                   [0, 1, 0]], dtype=int),
    'I': np.array([[1, 1, 1, 1, 1]], dtype=int),
    'L': np.array([[0, 0, 0, 1],
                   [1, 1, 1, 1]], dtype=int),
    'N': np.array([[1, 1, 0, 0],
                   [0, 1, 1, 1]], dtype=int),
    'P': np.array([[1, 1],
                   [1, 1],
                   [1, 0]], dtype=int),
    'T': np.array([[1, 1, 1],
                   [0, 1, 0],
                   [0, 1, 0]], dtype=int),
    'U': np.array([[1, 0, 1],
                   [1, 1, 1]], dtype=int),
    'V': np.array([[1, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1]], dtype=int),
    'W': np.array([[1, 0, 0],
                   [1, 1, 0],
                   [0, 1, 1]], dtype=int),
    'X': np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=int),
    'Y': np.array([[0, 0, 1, 0],
                   [1, 1, 1, 1]], dtype=int),
    'Z': np.array([[1, 1, 0],
                   [0, 1, 0],
                   [0, 1, 1]], dtype=int),
}

PENTOMINO_NAMES = list(PENTOMINO_SHAPES.keys())


def get_pentomino_matrix(name: str) -> np.ndarray:
    key = name.upper()
    if key not in PENTOMINO_SHAPES:
        raise ValueError(f"Unknown pentomino name '{name}'. Valid: {PENTOMINO_NAMES}")
    return PENTOMINO_SHAPES[key].copy()


def rotate_matrix_90(mat: np.ndarray, k: int = 1) -> np.ndarray:
    return np.rot90(mat, k=k)


def flip_matrix(mat: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.flip(mat, axis=axis)


class TrabecularMicrostructure:

    def __init__(self, grid_size: int = 15, pattern_seed: int = 42):
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        self.grid_size = grid_size
        self.pattern_seed = pattern_seed
        self.rng = np.random.default_rng(pattern_seed)
        self.rve_grid = self._build_rve()
        self.porosity = self._compute_porosity()
        self.specific_surface = self._compute_specific_surface()
        self.effective_modulus = self._compute_effective_young_modulus()

    def _build_rve(self) -> np.ndarray:
        n = self.grid_size
        grid = np.zeros((n, n), dtype=int)


        names = ['I', 'L', 'N', 'T', 'X', 'V', 'W']
        self.rng.shuffle(names)

        placed = 0
        for name in names:
            p = get_pentomino_matrix(name)

            p = rotate_matrix_90(p, k=self.rng.integers(0, 4))
            if self.rng.random() > 0.5:
                p = flip_matrix(p, axis=self.rng.integers(0, 2))

            ph, pw = p.shape

            max_attempts = 50
            for _ in range(max_attempts):
                i = self.rng.integers(0, n - ph + 1)
                j = self.rng.integers(0, n - pw + 1)

                region = grid[i:i+ph, j:j+pw]
                overlap = np.sum((region == 1) & (p == 1))
                if overlap <= 1:
                    region[p == 1] = 1
                    placed += 1
                    break


        if np.mean(grid) < 0.05:

            c = n // 2
            grid[c-1:c+2, c-1:c+2] = 1

        return grid

    def _compute_porosity(self) -> float:
        total = self.rve_grid.size
        solid = np.sum(self.rve_grid)
        void = total - solid
        phi = void / total
        return float(phi)

    def _compute_specific_surface(self) -> float:
        grid = self.rve_grid
        n = grid.shape[0]
        interface_edges = 0


        for i in range(n):
            for j in range(n - 1):
                if grid[i, j] != grid[i, j + 1]:
                    interface_edges += 1


        for i in range(n - 1):
            for j in range(n):
                if grid[i, j] != grid[i + 1, j]:
                    interface_edges += 1


        sv = interface_edges / (n * n)
        return float(sv)

    def _compute_effective_young_modulus(self) -> float:
        E_bone = 17.0e3
        rho_rel = 1.0 - self.porosity
        if rho_rel <= 0:
            return 0.0
        C = 1.0
        n_exp = 2.0
        E_eff = E_bone * C * (rho_rel ** n_exp)
        return float(E_eff)

    def get_local_density(self, x: float, y: float) -> float:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("Coordinates must be in [0,1]^2")
        n = self.grid_size
        i = min(int(y * n), n - 1)
        j = min(int(x * n), n - 1)
        return float(self.rve_grid[i, j])

    def generate_microstructure_report(self) -> Dict[str, float]:
        return {
            "grid_size": self.grid_size,
            "porosity": self.porosity,
            "relative_density": 1.0 - self.porosity,
            "specific_surface": self.specific_surface,
            "effective_young_modulus_MPa": self.effective_modulus,
            "effective_young_modulus_GPa": self.effective_modulus / 1000.0,
        }


def build_trabecular_field(nx: int, ny: int, cortical_mask: np.ndarray,
                           seed_offset: int = 0) -> np.ndarray:
    density = np.zeros(nx * ny)
    rho_cortical = 1.8
    rho_marrow = 0.001



    micro = TrabecularMicrostructure(grid_size=15, pattern_seed=42 + seed_offset)
    rho_trabecular = (1.0 - micro.porosity) * rho_cortical

    density[cortical_mask] = rho_cortical
    density[~cortical_mask] = max(rho_trabecular, 0.05)

    return density
