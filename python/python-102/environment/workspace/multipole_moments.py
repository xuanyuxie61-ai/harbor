
import numpy as np
from numpy.linalg import cholesky, eigh


class MultipoleExtractor:

    def __init__(self, wavelength=1.55e-6, n_bg=1.0):
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength
        self.eps0 = 8.854187817e-12
        self.mu0 = 4.0 * np.pi * 1.0e-7
        self.c = 1.0 / np.sqrt(self.eps0 * self.mu0)
        self.n_bg = n_bg
        self.eta0 = np.sqrt(self.mu0 / self.eps0)




    def extract_dipole_moments(self, r_obs, E_scat, H_scat):
        N = r_obs.shape[0]
        k = self.k0 * self.n_bg
        prefactor_E = k ** 2 / (4.0 * np.pi * self.eps0)
        prefactor_H = k ** 2 / (4.0 * np.pi)

        A = np.zeros((6 * N, 6), dtype=np.complex128)
        b = np.zeros(6 * N, dtype=np.complex128)

        for i in range(N):
            r = r_obs[i]
            r_mag = np.linalg.norm(r)
            if r_mag < 1e-18:
                continue
            n_hat = r / r_mag
            phase = np.exp(1.0j * k * r_mag) / r_mag




            for comp in range(3):

                for j in range(3):
                    val = (n_hat[comp] * n_hat[j] - (1.0 if comp == j else 0.0))
                    A[6 * i + comp, j] = prefactor_E * val * phase

                for j in range(3):

                    val = 0.0
                    for a in range(3):
                        for b_idx in range(3):
                            eps = self._levi_civita(comp, a, b_idx)
                            if eps != 0:
                                val += eps * n_hat[a] * (1.0 if j == b_idx else 0.0)
                    A[6 * i + comp, 3 + j] = prefactor_E * val * phase / self.c
                b[6 * i + comp] = E_scat[i, comp]


            for comp in range(3):
                for j in range(3):
                    val = 0.0
                    for a in range(3):
                        for b_idx in range(3):
                            eps = self._levi_civita(comp, a, b_idx)
                            if eps != 0:
                                val += eps * n_hat[a] * (1.0 if j == b_idx else 0.0)
                    A[6 * i + 3 + comp, j] = prefactor_H * val * phase
                for j in range(3):
                    val = (n_hat[comp] * n_hat[j] - (1.0 if comp == j else 0.0))
                    A[6 * i + 3 + comp, 3 + j] = -prefactor_H * val * phase / self.c
                b[6 * i + 3 + comp] = H_scat[i, comp]


        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        p = x[:3]
        m = x[3:6]
        return p, m

    @staticmethod
    def _levi_civita(i, j, k):
        if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            return 1
        elif (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
            return -1
        else:
            return 0




    def multipole_moment_method(self, angular_samples, field_samples, max_order=4):
        N = len(angular_samples)
        l_max = max_order
        n_coeff = (l_max + 1) ** 2



        M = np.zeros((n_coeff, n_coeff), dtype=np.complex128)


        Y_vals = np.zeros((N, n_coeff), dtype=np.complex128)
        for idx in range(N):
            theta, phi = angular_samples[idx]
            l_idx = 0
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    Y_vals[idx, l_idx] = self._spherical_harmonic(l, m, theta, phi)
                    l_idx += 1


        for p in range(n_coeff):
            for q in range(n_coeff):
                M[p, q] = np.sum(np.conj(Y_vals[:, p]) * Y_vals[:, q] * np.abs(field_samples) ** 2)
                M[p, q] *= 4.0 * np.pi / N


        M = 0.5 * (M + M.conj().T)



        try:
            R = cholesky(M)
        except np.linalg.LinAlgError:

            M += 1e-12 * np.eye(n_coeff)
            R = cholesky(M)


        n = n_coeff
        alpha = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            if abs(R[i, i]) > 1e-14:
                alpha[i] = R[i, i + 1] / R[i, i]


        J = np.diag(alpha, 1) + np.diag(alpha, -1)

        eigvals, eigvecs = eigh(J)


        field_moments = np.zeros(n_coeff, dtype=np.complex128)
        for p in range(n_coeff):
            field_moments[p] = np.sum(np.conj(Y_vals[:, p]) * field_samples)
            field_moments[p] *= 4.0 * np.pi / N

        coefficients = field_moments
        return coefficients

    def _spherical_harmonic(self, l, m, theta, phi):
        from scipy.special import sph_harm

        Y = sph_harm(m, l, phi, theta)
        return Y




    def radiation_powers(self, p, m, Q_e=None, Q_m=None):
        omega = self.k0 * self.c
        mu0 = self.mu0

        P_p = mu0 * omega ** 4 / (12.0 * np.pi * self.c) * np.sum(np.abs(p) ** 2)
        P_m = mu0 * omega ** 4 / (12.0 * np.pi * self.c ** 3) * np.sum(np.abs(m) ** 2)

        P_Qe = 0.0
        if Q_e is not None:
            P_Qe = mu0 * omega ** 6 / (240.0 * np.pi * self.c ** 3) * np.sum(np.abs(Q_e) ** 2)

        P_Qm = 0.0
        if Q_m is not None:
            P_Qm = mu0 * omega ** 6 / (240.0 * np.pi * self.c ** 3) * np.sum(np.abs(Q_m) ** 2)

        return {'P_dipole_electric': P_p,
                'P_dipole_magnetic': P_m,
                'P_quadrupole_electric': P_Qe,
                'P_quadrupole_magnetic': P_Qm,
                'P_total': P_p + P_m + P_Qe + P_Qm}


def demo():
    me = MultipoleExtractor(wavelength=1.55e-6)


    k = me.k0
    N = 120
    theta = np.linspace(0.1, np.pi - 0.1, N)
    phi = np.linspace(0, 2 * np.pi, N)
    theta_g, phi_g = np.meshgrid(theta, phi)
    theta_f = theta_g.flatten()
    phi_f = phi_g.flatten()


    p_true = np.array([1.0 + 0.5j, 0.3 - 0.2j, 0.1 + 0.0j]) * 1e-18
    m_true = np.array([0.2 + 0.1j, 0.8 - 0.3j, 0.1 + 0.2j]) * 1e-21

    r_obs = np.stack([
        np.sin(theta_f) * np.cos(phi_f),
        np.sin(theta_f) * np.sin(phi_f),
        np.cos(theta_f)
    ], axis=1) * 1.0e-3

    E_scat = np.zeros((len(theta_f), 3), dtype=np.complex128)
    H_scat = np.zeros((len(theta_f), 3), dtype=np.complex128)

    for i in range(len(theta_f)):
        n = r_obs[i] / np.linalg.norm(r_obs[i])

        cross_p = np.cross(n, p_true)
        E_scat[i] = (k ** 2 / (4 * np.pi * me.eps0)) * (
            np.cross(cross_p, n) + np.cross(n, m_true) / me.c
        ) * np.exp(1.0j * k * 1.0e-3) / 1.0e-3
        H_scat[i] = (k ** 2 / (4 * np.pi)) * (
            np.cross(n, p_true) - np.cross(np.cross(n, m_true), n) / me.c
        ) * np.exp(1.0j * k * 1.0e-3) / 1.0e-3

    p_est, m_est = me.extract_dipole_moments(r_obs, E_scat, H_scat)

    print("[multipole_moments] 电偶极矩估计:")
    print(f"  p = [{p_est[0]:.3e}, {p_est[1]:.3e}, {p_est[2]:.3e}] C·m")
    print("[multipole_moments] 磁偶极矩估计:")
    print(f"  m = [{m_est[0]:.3e}, {m_est[1]:.3e}, {m_est[2]:.3e}] A·m²")

    powers = me.radiation_powers(p_est, m_est)
    print(f"[multipole_moments] 总辐射功率: {powers['P_total']:.4e} W")
    return p_est, m_est, powers


if __name__ == "__main__":
    demo()
