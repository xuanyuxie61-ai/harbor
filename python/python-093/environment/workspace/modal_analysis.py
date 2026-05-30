#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg as la
from utils import chebyshev_to_monomial_matrix, legendre_to_monomial_matrix


class NormalModeAnalyzer:

    def __init__(self, env, z_min=0.0, z_max=None, n_cheb=64):
        self.env = env
        self.z_min = z_min
        self.z_max = z_max if z_max is not None else env.depth_max
        self.n_cheb = n_cheb

        j = np.arange(n_cheb + 1)
        self.xi = np.cos(np.pi * j / n_cheb)

        self.z_nodes = 0.5 * (self.z_max - self.z_min) * self.xi + 0.5 * (self.z_max + self.z_min)

        self.D = self._chebyshev_differentiation_matrix()

        self.D2 = self.D @ self.D

    def _chebyshev_differentiation_matrix(self):
        N = self.n_cheb
        x = self.xi
        c = np.ones(N + 1)
        c[0] = 2.0
        c[-1] = 2.0
        c *= ((-1.0) ** np.arange(N + 1))
        X = np.tile(x, (N + 1, 1))
        dX = X - X.T + np.eye(N + 1)
        D = np.outer(c, 1.0 / c.T) / (dX + np.eye(N + 1))
        D = D - np.diag(np.sum(D, axis=1))
        return D

    def solve_eigenproblem(self, n_modes=None):
        J = 2.0 / (self.z_max - self.z_min)
        D2_phys = J ** 2 * self.D2
        kz = self.env.wavenumber(self.z_nodes)
        k2 = kz ** 2





        raise NotImplementedError("HOLE 3: Normal mode eigenproblem missing")
        kr = np.array([])
        eigvecs = np.zeros((len(self.z_nodes), 0), dtype=np.complex128)
        return kr, eigvecs, self.z_nodes

    def estimate_mode_count_wkb(self):
        H = self.z_max
        c_min = self.env.c0
        f = self.env.frequency
        k = 2.0 * np.pi * f / c_min
        N_max = k * H / np.pi - 0.5
        return int(max(np.floor(N_max), 0)) + 1

    def estimate_mode_count_diophantine(self):
        return self.estimate_mode_count_wkb()

    def modal_phase_velocity(self, kr):
        return self.env.omega / np.real(kr)

    def modal_group_velocity(self, phi, kr, dz=None):
        z = self.z_nodes
        c = self.env.sound_speed(z)
        phi2 = np.abs(phi) ** 2
        if dz is None:
            dz = np.diff(z)
            dz = np.concatenate([dz, [dz[-1]]])
        num = np.sum(phi2 / (c ** 2) * dz)
        den = np.real(kr) * np.sum(phi2 / c * dz)
        if abs(den) < 1e-20:
            return 0.0
        return float(num / den)

    def modal_excitation_coefficients(self, phi, z_s):
        dz = np.diff(self.z_nodes)
        dz = np.concatenate([dz, [dz[-1]]])
        norms = np.sqrt(np.sum(np.abs(phi) ** 2 * dz, axis=0))
        norms = np.maximum(norms, 1e-20)

        phi_at_zs = np.zeros(phi.shape[1], dtype=np.complex128)
        for n in range(phi.shape[1]):
            phi_at_zs[n] = np.interp(z_s, self.z_nodes, phi[:, n])
        return phi_at_zs / norms

    def propagate_modes(self, kr, phi, z_s, r_target):
        A = self.modal_excitation_coefficients(phi, z_s)
        p = np.zeros(len(self.z_nodes), dtype=np.complex128)
        for n in range(len(kr)):
            if np.real(kr[n]) <= 0:
                continue
            phase = np.exp(1j * kr[n] * r_target)
            amp = 1.0 / np.sqrt(max(np.real(kr[n]) * r_target, 1e-6))
            p += A[n] * phi[:, n] * phase * amp
        return p

    def mode_dispersion_curve(self, phi, kr, frequencies):
        original_freq = self.env.frequency
        curves = []
        for n in range(min(5, len(kr))):
            vg_list = []
            for f in frequencies:
                self.env.frequency = f
                self.env.omega = 2.0 * np.pi * f
                self.env.k0 = self.env.omega / self.env.c0
                kr_new, phi_new, _ = self.solve_eigenproblem(n_modes=n + 1)
                if len(kr_new) > n:
                    vg = self.modal_group_velocity(phi_new[:, n], kr_new[n])
                    vg_list.append((f, vg))
            self.env.frequency = original_freq
            self.env.omega = 2.0 * np.pi * original_freq
            self.env.k0 = self.env.omega / self.env.c0
            if vg_list:
                curves.append(np.array(vg_list))
        return curves


class ModalConstraintSolver:

    @staticmethod
    def solve_inequality_integer(a, b):
        a = float(a)
        b = float(b)
        if a <= 0:
            return []
        n_max = int(np.floor((b - 1e-12) / a))
        if n_max < 0:
            return []
        return list(range(n_max + 1))

    @staticmethod
    def backtrack_solutions(coeffs, target, max_n=100):
        coeffs = np.asarray(coeffs, dtype=np.float64)
        solutions = []

        def backtrack(idx, current_sum, current_vec):
            if idx == len(coeffs):
                solutions.append(tuple(current_vec))
                return
            max_val = int((target - current_sum) / max(coeffs[idx], 1e-15))
            max_val = min(max_val, max_n)
            for v in range(max_val + 1):
                current_vec.append(v)
                backtrack(idx + 1, current_sum + v * coeffs[idx], current_vec)
                current_vec.pop()

        backtrack(0, 0.0, [])
        return solutions
