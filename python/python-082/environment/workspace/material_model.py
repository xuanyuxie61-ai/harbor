
import numpy as np

_EPS = 1e-12


class CompositeMaterial:

    def __init__(self, E1=150.0e9, E2=10.0e9, G12=5.0e9, nu12=0.3,
                 X_T=1500.0e6, X_C=1200.0e6,
                 Y_T=50.0e6, Y_C=200.0e6,
                 S_L=80.0e6, S_T=40.0e6,
                 G_ft=12000.0, G_fc=10000.0,
                 G_mt=1000.0, G_mc=2000.0):
        self.E1 = float(E1)
        self.E2 = float(E2)
        self.G12 = float(G12)
        self.nu12 = float(nu12)
        self.nu21 = self.nu12 * self.E2 / (self.E1 + _EPS)
        self.X_T = float(X_T)
        self.X_C = float(X_C)
        self.Y_T = float(Y_T)
        self.Y_C = float(Y_C)
        self.S_L = float(S_L)
        self.S_T = float(S_T)
        self.G_ft = float(G_ft)
        self.G_fc = float(G_fc)
        self.G_mt = float(G_mt)
        self.G_mc = float(G_mc)


        denom = 1.0 - self.nu12 * self.nu21
        if abs(denom) < _EPS:
            denom = _EPS
        self.Q = np.array([
            [self.E1 / denom, self.nu21 * self.E1 / denom, 0.0],
            [self.nu12 * self.E2 / denom, self.E2 / denom, 0.0],
            [0.0, 0.0, self.G12]
        ])

    def degraded_stiffness(self, d_f, d_m, d_s=0.0):







        raise NotImplementedError("Hole 1: degraded_stiffness needs implementation.")

    def hashin_failure_indices(self, sigma):
        sigma = np.asarray(sigma, dtype=float).flatten()
        s11, s22, s12 = sigma[0], sigma[1], sigma[2]


        if s11 >= 0:
            F_ft = (s11 / (self.X_T + _EPS)) ** 2 + (s12 / (self.S_L + _EPS)) ** 2
            F_fc = 0.0
        else:
            F_ft = 0.0
            F_fc = (abs(s11) / (self.X_C + _EPS)) ** 2


        if s22 >= 0:
            F_mt = (s22 / (self.Y_T + _EPS)) ** 2 + (s12 / (self.S_L + _EPS)) ** 2
            F_mc = 0.0
        else:
            F_mt = 0.0
            term1 = (s22 / (2.0 * self.S_T + _EPS)) ** 2
            coef = (self.Y_C / (2.0 * self.S_T + _EPS)) ** 2 - 1.0
            term2 = coef * (s22 / (self.Y_C + _EPS))
            F_mc = term1 + term2 + (s12 / (self.S_L + _EPS)) ** 2

        return {
            'F_ft': F_ft,
            'F_fc': F_fc,
            'F_mt': F_mt,
            'F_mc': F_mc
        }

    def damage_evolution(self, phi_ft, phi_fc, phi_mt, phi_mc, L_c=1.0e-3):
        def _exp_damage(phi, sigma0, Gc):
            if phi <= 1.0:
                return 0.0
            A = 2.0 * L_c * Gc / (sigma0 ** 2 + _EPS)
            d = 1.0 - np.exp(A * (1.0 - phi)) / (phi + _EPS)
            return np.clip(d, 0.0, 1.0 - _EPS)

        d_f_t = _exp_damage(phi_ft, self.X_T, self.G_ft)
        d_f_c = _exp_damage(phi_fc, self.X_C, self.G_fc)
        d_f = max(d_f_t, d_f_c)

        d_m_t = _exp_damage(phi_mt, self.Y_T, self.G_mt)
        d_m_c = _exp_damage(phi_mc, self.Y_C, self.G_mc)
        d_m = max(d_m_t, d_m_c)

        return d_f, d_m

    def thermodynamic_force(self, epsilon, d_f, d_m, d_s=0.0):
        epsilon = np.asarray(epsilon, dtype=float).flatten()[:3]
        C0 = self.Q
        C_d = self.degraded_stiffness(d_f, d_m, d_s)


        dd = 1e-6
        C_df = (self.degraded_stiffness(d_f + dd, d_m, d_s) - C_d) / dd
        C_dm = (self.degraded_stiffness(d_f, d_m + dd, d_s) - C_d) / dd

        Y_f = -0.5 * epsilon @ C_df @ epsilon
        Y_m = -0.5 * epsilon @ C_dm @ epsilon
        return Y_f, Y_m


class LaminateProperties:

    def __init__(self, material, fiber_angles, thicknesses):
        self.material = material
        self.fiber_angles = np.asarray(fiber_angles, dtype=float)
        self.thicknesses = np.asarray(thicknesses, dtype=float)
        if len(self.fiber_angles) != len(self.thicknesses):
            raise ValueError("Angles and thicknesses must have same length.")
        self.n_plys = len(self.fiber_angles)

    def abd_matrix(self):
        A = np.zeros((3, 3), dtype=float)
        B = np.zeros((3, 3), dtype=float)
        D = np.zeros((3, 3), dtype=float)

        z = -0.5 * np.sum(self.thicknesses)
        for k in range(self.n_plys):
            z_prev = z
            z += self.thicknesses[k]
            theta = np.deg2rad(self.fiber_angles[k])
            c, s = np.cos(theta), np.sin(theta)
            T = np.array([
                [c * c, s * s, 2.0 * s * c],
                [s * s, c * c, -2.0 * s * c],
                [-s * c, s * c, c * c - s * s]
            ])
            Q_bar = np.linalg.inv(T) @ self.material.Q @ np.linalg.inv(T).T
            dz = z - z_prev
            A += Q_bar * dz
            B += 0.5 * Q_bar * (z ** 2 - z_prev ** 2)
            D += (1.0 / 3.0) * Q_bar * (z ** 3 - z_prev ** 3)

        return A, B, D

    def engineering_constants(self):
        A, B, D = self.abd_matrix()
        h = np.sum(self.thicknesses)
        det = A[0, 0] * A[1, 1] - A[0, 1] ** 2
        if abs(det) < _EPS:
            det = _EPS
        E_x = det / (A[1, 1] * h + _EPS)
        E_y = det / (A[0, 0] * h + _EPS)
        nu_xy = A[0, 1] / (A[1, 1] + _EPS)
        G_xy = A[2, 2] / h
        return {
            'E_x': E_x, 'E_y': E_y,
            'nu_xy': nu_xy, 'G_xy': G_xy
        }
