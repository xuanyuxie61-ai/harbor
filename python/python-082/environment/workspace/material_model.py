"""
material_model.py
=================
Progressive damage material model for fiber-reinforced composites.

Implements continuum damage mechanics (CDM) based on Hashin failure criteria
with exponential stiffness degradation. This module contains the core
constitutive equations for composite damage evolution.

Scientific formulas:
--------------------
1. Plane-stress reduced stiffness matrix Q (material coordinates):

       Q = [[ E1/(1-nu12*nu21),  nu21*E1/(1-nu12*nu21),  0         ],
            [ nu12*E2/(1-nu12*nu21), E2/(1-nu12*nu21),    0         ],
            [ 0,                    0,                    G12       ]]

2. Hashin failure criteria (2D plane stress):
   Fiber tension   (sigma_11 >= 0):
       F_ft = (sigma_11 / X_T)^2 + (sigma_12 / S_L)^2 >= 1

   Fiber compression (sigma_11 < 0):
       F_fc = (sigma_11 / X_C)^2 >= 1

   Matrix tension  (sigma_22 >= 0):
       F_mt = (sigma_22 / Y_T)^2 + (sigma_12 / S_L)^2 >= 1

   Matrix compression (sigma_22 < 0):
       F_mc = (sigma_22 / (2*S_T))^2
              + [(Y_C/(2*S_T))^2 - 1] * (sigma_22 / Y_C)
              + (sigma_12 / S_L)^2 >= 1

3. Damage evolution (exponential softening):
   For each mode r, the damage variable d_r evolves as:

       d_r = 1 - (e^{A_r * (1 - phi_r)} / phi_r)

   where phi_r = epsilon_eq^{(r)} / epsilon_0^{(r)} is the equivalent
   strain ratio, and A_r is a material parameter controlling the
   fracture energy dissipation:

       A_r = 2 * L_c * G_c^{(r)} / (sigma_0^{(r)})^2

4. Effective stress hypothesis:
   sigma_eff = C_0 : epsilon
   sigma     = M^{-1}(d) : sigma_eff
   where M(d) = diag(1-d_f, 1-d_m, 1-d_s) is the damage effect tensor.

5. Degraded stiffness:
   C_d = M^{-1} * C_0 * M^{-T}

Numerical robustness:
- All denominators are guarded with epsilon = 1e-12.
- Damage variables are clipped to [0, 1].
- Stiffness matrices are symmetrized.
"""

import numpy as np

_EPS = 1e-12


class CompositeMaterial:
    """
    Material properties for a unidirectional composite ply.

    Default values correspond to a typical T300/976 graphite/epoxy
    lamina (SI units).
    """

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
        self.G_ft = float(G_ft)   # Fiber tension fracture energy (J/m^2)
        self.G_fc = float(G_fc)   # Fiber compression fracture energy
        self.G_mt = float(G_mt)   # Matrix tension fracture energy
        self.G_mc = float(G_mc)   # Matrix compression fracture energy

        # Precompute Q matrix
        denom = 1.0 - self.nu12 * self.nu21
        if abs(denom) < _EPS:
            denom = _EPS
        self.Q = np.array([
            [self.E1 / denom, self.nu21 * self.E1 / denom, 0.0],
            [self.nu12 * self.E2 / denom, self.E2 / denom, 0.0],
            [0.0, 0.0, self.G12]
        ])

    def degraded_stiffness(self, d_f, d_m, d_s=0.0):
        """
        Compute degraded stiffness matrix given damage variables.

        Parameters
        ----------
        d_f, d_m, d_s : float
            Fiber, matrix, and shear damage in [0, 1].

        Returns
        -------
        C_d : ndarray (3, 3)
            Degraded stiffness in material coordinates.
        """
        # === HOLE 1 ===
        # TODO: Implement the degraded stiffness matrix based on continuum damage mechanics.
        # The effective stress hypothesis gives: sigma = M^{-1}(d) : sigma_eff
        # where M(d) = diag(1-d_f, 1-d_m, 1-d_s) is the damage effect tensor.
        # The degraded stiffness should be: C_d = M^{-1} * Q * M^{-T}
        # Remember to symmetrize the result for numerical stability.
        # Clip damage variables to [0, 1 - _EPS] to avoid division by zero.
        raise NotImplementedError("Hole 1: degraded_stiffness needs implementation.")

    def hashin_failure_indices(self, sigma):
        """
        Evaluate Hashin failure criteria for a stress state.

        Parameters
        ----------
        sigma : array_like, shape (3,)
            [sigma_11, sigma_22, sigma_12] in material coordinates (Pa).

        Returns
        -------
        indices : dict
            {'F_ft', 'F_fc', 'F_mt', 'F_mc'}
        """
        sigma = np.asarray(sigma, dtype=float).flatten()
        s11, s22, s12 = sigma[0], sigma[1], sigma[2]

        # Fiber tension
        if s11 >= 0:
            F_ft = (s11 / (self.X_T + _EPS)) ** 2 + (s12 / (self.S_L + _EPS)) ** 2
            F_fc = 0.0
        else:
            F_ft = 0.0
            F_fc = (abs(s11) / (self.X_C + _EPS)) ** 2

        # Matrix tension
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
        """
        Exponential damage evolution law based on equivalent strain ratio.

        Parameters
        ----------
        phi_* : float
            Equivalent strain ratio phi = epsilon_eq / epsilon_0 for each mode.
        L_c : float
            Characteristic element length (m).

        Returns
        -------
        d_f, d_m : float
            Updated fiber and matrix damage variables.
        """
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
        """
        Compute thermodynamic forces (energy release rates) Y_r.

        Y_f = -dPsi/dd_f = 0.5 * epsilon^T * (dC/dd_f) * epsilon
        """
        epsilon = np.asarray(epsilon, dtype=float).flatten()[:3]
        C0 = self.Q
        C_d = self.degraded_stiffness(d_f, d_m, d_s)

        # Numerical derivative of stiffness w.r.t damage
        dd = 1e-6
        C_df = (self.degraded_stiffness(d_f + dd, d_m, d_s) - C_d) / dd
        C_dm = (self.degraded_stiffness(d_f, d_m + dd, d_s) - C_d) / dd

        Y_f = -0.5 * epsilon @ C_df @ epsilon
        Y_m = -0.5 * epsilon @ C_dm @ epsilon
        return Y_f, Y_m


class LaminateProperties:
    """
    ABAQUS-style ply-by-ply laminate property container.
    """

    def __init__(self, material, fiber_angles, thicknesses):
        self.material = material
        self.fiber_angles = np.asarray(fiber_angles, dtype=float)
        self.thicknesses = np.asarray(thicknesses, dtype=float)
        if len(self.fiber_angles) != len(self.thicknesses):
            raise ValueError("Angles and thicknesses must have same length.")
        self.n_plys = len(self.fiber_angles)

    def abd_matrix(self):
        """
        Compute the classical laminate ABD matrix.

        A = sum_k Q_bar_k * (z_k - z_{k-1})
        B = 0.5 * sum_k Q_bar_k * (z_k^2 - z_{k-1}^2)
        D = (1/3) * sum_k Q_bar_k * (z_k^3 - z_{k-1}^3)

        Returns
        -------
        A, B, D : ndarray (3, 3)
            Extensional, coupling, and bending stiffness matrices.
        """
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
        """
        Estimate effective engineering constants from the A matrix
        for a symmetric laminate (B=0).

        E_x = (A11*A22 - A12^2) / (A22*h)
        E_y = (A11*A22 - A12^2) / (A11*h)
        nu_xy = A12 / A22
        G_xy = A66 / h
        """
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
