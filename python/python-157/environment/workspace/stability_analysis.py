import numpy as np
from combustion_utils import check_positive, check_nonnegative


class DetonationStability:

    def __init__(self, xi, znd_sol, gamma=1.4, Q=2.5e6):
        self.xi = np.asarray(xi, dtype=float)
        self.znd = np.asarray(znd_sol, dtype=float)
        self.gamma = gamma
        self.Q = Q
        self.npts = len(xi)

    def _compute_local_jacobian(self, idx):
        rho, u, p, lam = self.znd[idx]
        if rho <= 0.0 or p <= 0.0:
            return np.zeros((4, 4))

        a = np.sqrt(self.gamma * p / rho)
        v_rel = u

        J = np.zeros((4, 4))

        J[0, 0] = -v_rel / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0
        J[0, 1] = -rho / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0


        J[1, 0] = 0.0
        J[1, 1] = -v_rel / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0
        J[1, 2] = -1.0 / (rho * (self.xi[1] - self.xi[0])) if self.npts > 1 else 0.0


        cv = 1.0 / (self.gamma - 1.0) * (8.314 / 0.029)
        T = p / (rho * (8.314 / 0.029))
        k = 1.0e8 * np.exp(-8.314e4 / (8.314 * max(T, 100.0)))
        dlam_dt = -k * max(1.0 - lam, 0.0)

        J[2, 0] = (self.gamma - 1.0) * T * v_rel
        J[2, 1] = -v_rel * v_rel
        J[2, 2] = -v_rel / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0
        J[2, 3] = self.Q * rho * k


        J[3, 3] = -k if lam < 1.0 else 0.0

        return J

    def global_stability_matrix(self):
        J_avg = np.zeros((4, 4))
        dx_total = self.xi[-1] - self.xi[0]
        if dx_total <= 0.0:
            dx_total = 1.0

        for i in range(self.npts - 1):
            dx = self.xi[i + 1] - self.xi[i]
            Ji = self._compute_local_jacobian(i)
            J_avg += Ji * dx
        J_avg /= dx_total
        return J_avg

    def eigenvalue_analysis(self):
        J = self.global_stability_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(J)

        idx = np.argsort(-eigenvalues.real)
        return eigenvalues[idx], eigenvectors[:, idx]

    def instability_modes(self):
        evals, evecs = self.eigenvalue_analysis()
        modes = []
        for ev in evals:
            alpha = ev.real
            omega = ev.imag
            if alpha > 1.0e-6:
                modes.append({
                    'growth_rate': alpha,
                    'frequency': abs(omega) / (2.0 * np.pi),
                    'eigenvalue': ev
                })
        return modes

    def pulsation_frequency_estimate(self):
        evals, _ = self.eigenvalue_analysis()

        idx = np.argmax(np.abs(evals.imag))
        return abs(evals[idx].imag) / (2.0 * np.pi)
