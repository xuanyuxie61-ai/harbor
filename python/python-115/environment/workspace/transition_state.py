
import numpy as np
from sparse_operations import CRSMatrix, lanczos_eigenvalue_solver


class TransitionStateVerifier:

    def __init__(self, gradient_func, hessian_func=None):
        self.gradient_func = gradient_func
        self.hessian_func = hessian_func

    def verify_saddle_point(self, x_ts, grad_tol=1e-3):
        grad = self.gradient_func(x_ts)
        grad_norm = np.linalg.norm(grad)

        result = {
            'gradient_norm': grad_norm,
            'is_stationary': grad_norm < grad_tol,
            'n_negative_modes': None,
            'is_transition_state': False,
            'imaginary_frequency': None
        }

        if self.hessian_func is not None:
            H = self.hessian_func(x_ts)
            eigenvalues = np.linalg.eigvalsh(H)
            n_neg = np.sum(eigenvalues < -1e-6)
            result['n_negative_modes'] = n_neg
            result['is_transition_state'] = (n_neg == 1)

            if n_neg >= 1:


                c_cm_fs = 2.998e-5
                lam_neg = eigenvalues[eigenvalues < -1e-6][0]
                freq_cm = np.sqrt(abs(lam_neg)) / (2.0 * np.pi * c_cm_fs)
                result['imaginary_frequency'] = freq_cm
                result['eigenvalues'] = eigenvalues

        return result

    def wigner_correction(self, imaginary_freq_cm, temperature=300.0):
        h = 6.626e-34
        c = 2.998e10
        kB = 1.381e-23

        x = h * c * abs(imaginary_freq_cm) / (kB * temperature)
        kappa = 1.0 + x ** 2 / 24.0
        return kappa

    def rate_constant_tst(self, delta_G, temperature=300.0, kappa=1.0):



        raise NotImplementedError("Hole_2: 请实现 rate_constant_tst 方法")


class NEBOptimizer:

    def __init__(self, energy_func, gradient_func, n_images=20,
                 spring_k=0.1, dt=0.01, max_iter=1000, tol=1e-4):
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        self.n_images = n_images
        self.spring_k = spring_k
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol

    def _compute_tangent(self, path, energies, i):
        if energies[i + 1] > energies[i - 1]:
            tau = path[i + 1] - path[i]
        else:
            tau = path[i] - path[i - 1]
        norm = np.linalg.norm(tau)
        if norm > 1e-12:
            return tau / norm
        return np.zeros_like(tau)

    def optimize(self, x_reactant, x_product):
        x_R = np.asarray(x_reactant, dtype=float)
        x_P = np.asarray(x_product, dtype=float)
        dim = len(x_R)


        path = np.zeros((self.n_images, dim), dtype=float)
        for i in range(self.n_images):
            lam = i / (self.n_images - 1.0)
            path[i] = (1.0 - lam) * x_R + lam * x_P

        energies_history = []

        for it in range(self.max_iter):
            energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(self.n_images)])
            energies_history.append(energies.copy())

            forces = np.zeros_like(path)
            max_force = 0.0

            for i in range(1, self.n_images - 1):
                tau = self._compute_tangent(path, energies, i)
                grad = gradients[i]


                f_perp = grad - np.dot(grad, tau) * tau


                f_spring = self.spring_k * (
                        np.linalg.norm(path[i + 1] - path[i]) -
                        np.linalg.norm(path[i] - path[i - 1])
                ) * tau

                forces[i] = -f_perp + f_spring
                max_force = max(max_force, np.linalg.norm(forces[i]))


            path[1:self.n_images - 1] += self.dt * forces[1:self.n_images - 1]

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
        return path, final_energies, energies_history

    def climbing_image(self, x_reactant, x_product, n_climb_steps=200):

        path, energies, _ = self.optimize(x_reactant, x_product)


        ts_idx = np.argmax(energies[1:self.n_images - 1]) + 1


        for it in range(n_climb_steps):
            energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(self.n_images)])

            forces = np.zeros_like(path)
            max_force = 0.0

            for i in range(1, self.n_images - 1):
                tau = self._compute_tangent(path, energies, i)
                grad = gradients[i]

                if i == ts_idx:

                    f_parallel = np.dot(grad, tau) * tau
                    forces[i] = -(grad - 2.0 * f_parallel)
                else:
                    f_perp = grad - np.dot(grad, tau) * tau
                    f_spring = self.spring_k * (
                            np.linalg.norm(path[i + 1] - path[i]) -
                            np.linalg.norm(path[i] - path[i - 1])
                    ) * tau
                    forces[i] = -f_perp + f_spring

                max_force = max(max_force, np.linalg.norm(forces[i]))

            path[1:self.n_images - 1] += self.dt * forces[1:self.n_images - 1]

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
        return path, final_energies, ts_idx


class ReactionPathAnalysis:

    @staticmethod
    def find_transition_state(path, energies):

        ts_idx = np.argmax(energies[1:len(energies) - 1]) + 1
        return ts_idx, path[ts_idx], energies[ts_idx]

    @staticmethod
    def activation_energy(energies):
        ts_idx = np.argmax(energies[1:len(energies) - 1]) + 1
        E_r = energies[0]
        E_p = energies[-1]
        E_ts = energies[ts_idx]
        Ea_forward = E_ts - E_r
        Ea_reverse = E_ts - E_p
        return Ea_forward, Ea_reverse, ts_idx

    @staticmethod
    def reaction_coordinate_values(path):
        s = np.zeros(len(path))
        for i in range(1, len(path)):
            s[i] = s[i - 1] + np.linalg.norm(path[i] - path[i - 1])
        return s

    @staticmethod
    def curvature_analysis(path, energies):
        s = ReactionPathAnalysis.reaction_coordinate_values(path)
        if len(s) < 3:
            return np.zeros(len(s))


        ds = np.gradient(s)
        path_flat = path.reshape(len(path), -1)
        d2gamma = np.zeros(len(path))

        for i in range(1, len(path) - 1):
            if ds[i] > 1e-12:
                d2 = (path_flat[i + 1] - 2 * path_flat[i] + path_flat[i - 1]) / (ds[i] ** 2)
                d2gamma[i] = np.linalg.norm(d2)

        return d2gamma
