
import numpy as np
from scipy.special import jv, jn_zeros


class SemanticEmbeddingBases:

    def __init__(self, radius: float = 1.0, max_mode_m: int = 5, max_mode_n: int = 5):
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        if max_mode_m < 1:
            raise ValueError(f"max_mode_m must be at least 1, got {max_mode_m}")
        if max_mode_n < 0:
            raise ValueError(f"max_mode_n must be non-negative, got {max_mode_n}")

        self.radius = float(radius)
        self.max_mode_m = int(max_mode_m)
        self.max_mode_n = int(max_mode_n)
        self._compute_bessel_zeros()
        self._build_basis_index_map()

    def _compute_bessel_zeros(self):
        self.bessel_zeros = {}
        self.wavenumbers = {}

        for n in range(self.max_mode_n + 1):



            if n == 0:

                zeros = jn_zeros(n, self.max_mode_m)
            else:

                positive_zeros = jn_zeros(n, self.max_mode_m)
                zeros = np.concatenate([[0.0], positive_zeros])

            self.bessel_zeros[n] = zeros
            self.wavenumbers[n] = zeros / self.radius

    def _build_basis_index_map(self):
        self.basis_list = []
        for n in range(self.max_mode_n + 1):
            if n == 0:
                for m in range(1, self.max_mode_m + 1):
                    self.basis_list.append((m, n, 'cos'))
            else:
                for m in range(self.max_mode_m + 1):
                    self.basis_list.append((m, n, 'cos'))
                    self.basis_list.append((m, n, 'sin'))
        self.num_bases = len(self.basis_list)

    def evaluate_basis(self, r: np.ndarray, theta: np.ndarray,
                       m: int, n: int, angular_type: str) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)

        if np.any(r < 0.0) or np.any(r > self.radius):
            raise ValueError(f"r must be in [0, {self.radius}], got range [{r.min()}, {r.max()}]")

        if n == 0 and m == 0:
            raise ValueError("For n=0, m=0 is illegal (no zero at origin for J_0)")



        if n == 0:
            rho = self.bessel_zeros[n][m - 1]
        else:
            rho = self.bessel_zeros[n][m]
        k = rho / self.radius


        radial = jv(n, k * r)


        if angular_type == 'cos':
            angular = np.cos(n * theta)
        elif angular_type == 'sin':
            angular = np.sin(n * theta)
        else:
            raise ValueError(f"angular_type must be 'cos' or 'sin', got {angular_type}")

        return radial * angular

    def evaluate_all_bases(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        num_points = r.size
        Phi = np.zeros((num_points, self.num_bases))

        for idx, (m, n, angular_type) in enumerate(self.basis_list):
            Phi[:, idx] = self.evaluate_basis(r, theta, m, n, angular_type).flatten()

        return Phi

    def project_semantic_vector(self, semantic_field: np.ndarray,
                                r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        semantic_field = np.asarray(semantic_field, dtype=float).flatten()
        r = np.asarray(r, dtype=float).flatten()
        theta = np.asarray(theta, dtype=float).flatten()

        if semantic_field.size != r.size or semantic_field.size != theta.size:
            raise ValueError("semantic_field, r, theta must have the same number of points")

        Phi = self.evaluate_all_bases(r, theta)
        coeffs = np.zeros(self.num_bases)



        dr = np.diff(np.sort(np.unique(r))).mean() if len(np.unique(r)) > 1 else 1.0
        dtheta = np.diff(np.sort(np.unique(theta))).mean() if len(np.unique(theta)) > 1 else 1.0
        weights = r * dr * dtheta

        for j in range(self.num_bases):

            numerator = np.sum(semantic_field * Phi[:, j] * weights)

            denominator = np.sum(Phi[:, j] ** 2 * weights)
            if abs(denominator) > 1e-15:
                coeffs[j] = numerator / denominator
            else:
                coeffs[j] = 0.0

        return coeffs

    def reconstruct_semantic_field(self, coeffs: np.ndarray,
                                   r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.size != self.num_bases:
            raise ValueError(f"coeffs size must be {self.num_bases}, got {coeffs.size}")

        Phi = self.evaluate_all_bases(r, theta)
        return Phi @ coeffs

    def basis_orthogonality_check(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        Phi = self.evaluate_all_bases(r, theta)
        dr = np.diff(np.sort(np.unique(r))).mean() if len(np.unique(r)) > 1 else 1.0
        dtheta = np.diff(np.sort(np.unique(theta))).mean() if len(np.unique(theta)) > 1 else 1.0
        weights = np.asarray(r, dtype=float).flatten() * dr * dtheta

        gram = np.zeros((self.num_bases, self.num_bases))
        for i in range(self.num_bases):
            for j in range(self.num_bases):
                gram[i, j] = np.sum(Phi[:, i] * Phi[:, j] * weights)


        diag = np.sqrt(np.diag(gram))
        diag[diag < 1e-15] = 1.0
        gram_norm = gram / np.outer(diag, diag)
        return gram_norm


def demo():
    print("=" * 60)
    print("Helmholtz语义嵌入正交基系统演示")
    print("=" * 60)

    bases = SemanticEmbeddingBases(radius=1.0, max_mode_m=3, max_mode_n=2)
    print(f"\n正交基总数: {bases.num_bases}")
    print(f"基函数列表: {bases.basis_list}")


    r_grid = np.linspace(0.01, 1.0, 20)
    theta_grid = np.linspace(0, 2 * np.pi, 40)
    R, T = np.meshgrid(r_grid, theta_grid)
    r_flat = R.flatten()
    theta_flat = T.flatten()


    semantic_field = np.exp(-((r_flat - 0.5) ** 2 + (np.sin(theta_flat)) ** 2) / 0.1)


    coeffs = bases.project_semantic_vector(semantic_field, r_flat, theta_flat)
    print(f"\n谱系数 (前10个): {coeffs[:10]}")


    reconstructed = bases.reconstruct_semantic_field(coeffs, r_flat, theta_flat)
    error = np.linalg.norm(semantic_field - reconstructed) / np.linalg.norm(semantic_field)
    print(f"\n重构相对误差: {error:.6e}")


    gram = bases.basis_orthogonality_check(r_flat, theta_flat)
    off_diag_max = np.max(np.abs(gram - np.eye(bases.num_bases)))
    print(f"正交性偏差 (非对角元最大绝对值): {off_diag_max:.6e}")

    print("\n模块运行完成")
    return bases, coeffs, error


if __name__ == "__main__":
    demo()
