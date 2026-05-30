
import numpy as np
from scipy.special import comb


def bernstein_basis(n, u):
    if n < 0:
        raise ValueError("n 必须非负。")
    u = np.atleast_1d(u)
    u = np.clip(u, 0.0, 1.0)

    if n == 0:
        return np.ones_like(u)

    B = np.zeros(u.shape + (n + 1,))

    B[..., 0] = 1.0 - u
    B[..., 1] = u

    for i in range(2, n + 1):
        B[..., i] = u * B[..., i - 1]
        for j in range(i - 1, 0, -1):
            B[..., j] = u * B[..., j - 1] + (1.0 - u) * B[..., j]
        B[..., 0] = (1.0 - u) * B[..., 0]

    return B


def bernstein_to_monomial_matrix(n):
    M = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        for k in range(j + 1):
            M[j, k] = comb(j, k, exact=True) / comb(n, k, exact=True)
    return M


def monomial_to_bernstein_matrix(n):
    M = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        for j in range(k, n + 1):
            M[k, j] = ((-1) ** (j - k)) * comb(n, j, exact=True) * comb(j, k, exact=True)
    return M


class MembraneDensityProfile:

    def __init__(self, z_min=-3.0, z_max=3.0, n_bernstein=10):
        if z_max <= z_min:
            raise ValueError("z_max 必须大于 z_min。")
        if n_bernstein < 1:
            raise ValueError("Bernstein 次数必须至少为 1。")
        self.z_min = z_min
        self.z_max = z_max
        self.n = n_bernstein
        self.coeffs = np.zeros(n_bernstein + 1)
        self.coeffs[0] = 1.0

    def _z_to_u(self, z):
        u = (z - self.z_min) / (self.z_max - self.z_min)
        return np.clip(u, 0.0, 1.0)

    def fit_density(self, z_samples, rho_samples, weights=None):
        z_samples = np.asarray(z_samples)
        rho_samples = np.asarray(rho_samples)
        if len(z_samples) != len(rho_samples):
            raise ValueError("z_samples 与 rho_samples 长度不一致。")

        u = self._z_to_u(z_samples)
        B = bernstein_basis(self.n, u)
        if B.ndim > 2:
            B = B.reshape(-1, B.shape[-1])

        if weights is not None:
            W = np.diag(np.sqrt(np.asarray(weights)))
            Bw = W @ B
            rw = W @ rho_samples
        else:
            Bw = B
            rw = rho_samples


        AtA = Bw.T @ Bw
        Atb = Bw.T @ rw

        reg = 1e-10 * np.eye(self.n + 1)
        self.coeffs = np.linalg.solve(AtA + reg, Atb)
        self.coeffs = np.clip(self.coeffs, 0.0, None)

    def evaluate(self, z):
        u = self._z_to_u(z)
        B = bernstein_basis(self.n, u)
        if B.ndim > 2:
            B = B.reshape(-1, B.shape[-1])
        return B @ self.coeffs

    def headgroup_distance(self, threshold=0.5):
        z_grid = np.linspace(self.z_min, self.z_max, 1000)
        rho = self.evaluate(z_grid)
        rho_max = np.max(rho)
        if rho_max <= 0:
            return 0.0
        mask = rho > threshold * rho_max
        if not np.any(mask):
            return 0.0
        z_active = z_grid[mask]
        return float(z_active[-1] - z_active[0])

    def membrane_thickness_from_gaussian_fit(self):
        z_grid = np.linspace(self.z_min, self.z_max, 1000)
        rho = self.evaluate(z_grid)


        half = len(z_grid) // 2
        idx_left = np.argmax(rho[:half])
        idx_right = half + np.argmax(rho[half:])
        z0_left = z_grid[idx_left]
        z0_right = z_grid[idx_right]
        d_hh = abs(z0_right - z0_left)
        return float(d_hh)

    def area_compressibility_modulus(self, area_samples, tension_samples):
        area_samples = np.asarray(area_samples)
        tension_samples = np.asarray(tension_samples)
        if len(area_samples) < 2:
            return 0.0
        lnA = np.log(area_samples)

        A_mat = np.vstack([lnA, np.ones_like(lnA)]).T
        coeff, _, _, _ = np.linalg.lstsq(A_mat, tension_samples, rcond=None)
        k_a = float(coeff[0])
        return k_a if k_a > 0 else 0.0

    def bending_rigidity_helfrich(self, temperature, thickness):
        if thickness <= 0:
            return 0.0
        d_mono = thickness / 2.0
        k_c = self.area_compressibility_modulus(
            np.array([1.0, 1.01, 1.02]),
            np.array([0.0, 1.0, 2.0])
        ) * d_mono ** 2 / 24.0


        k_a_typical = 200.0
        k_c = k_a_typical * (d_mono ** 2) / 24.0
        return float(k_c)

    def lipid_number_density(self, z, lipid_mass=0.7):
        rho = self.evaluate(z)
        if lipid_mass <= 0:
            raise ValueError("脂质质量必须为正。")
        return rho / lipid_mass
