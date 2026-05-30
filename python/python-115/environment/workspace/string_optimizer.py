
import numpy as np
from scipy.interpolate import CubicSpline


class PathParameterization:

    @staticmethod
    def arc_length_parameterize(path):
        path = np.asarray(path, dtype=float)
        n_images = path.shape[0]
        s = np.zeros(n_images, dtype=float)

        for i in range(1, n_images):
            s[i] = s[i - 1] + np.linalg.norm(path[i] - path[i - 1])

        if s[-1] > 0:
            s /= s[-1]
        return s

    @staticmethod
    def reparametrize_equidistant(path, n_new=None):
        path = np.asarray(path, dtype=float)
        if n_new is None:
            n_new = path.shape[0]

        s_old = PathParameterization.arc_length_parameterize(path)
        s_new = np.linspace(0, 1, n_new)

        dim = path.shape[1]
        path_new = np.zeros((n_new, dim), dtype=float)

        for d in range(dim):
            cs = CubicSpline(s_old, path[:, d])
            path_new[:, d] = cs(s_new)

        return path_new


class StringMethod:

    def __init__(self, energy_func, gradient_func, n_images=20,
                 spring_const=1.0, dt=0.01, max_iter=500, tol=1e-4):
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        self.n_images = n_images
        self.spring_const = spring_const
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol

    def compute_tangents(self, path, energies):
        path = np.asarray(path, dtype=float)
        n = path.shape[0]
        tangents = np.zeros_like(path)

        for i in range(1, n - 1):
            v_max = max(abs(energies[i + 1] - energies[i]),
                        abs(energies[i] - energies[i - 1]))
            if v_max < 1e-12:
                tau = path[i + 1] - path[i - 1]
            else:
                if energies[i + 1] > energies[i - 1]:
                    tau = (energies[i + 1] - energies[i]) * (path[i + 1] - path[i])
                    tau += (energies[i] - energies[i - 1]) * (path[i] - path[i - 1])
                else:
                    tau = (energies[i - 1] - energies[i]) * (path[i] - path[i - 1])
                    tau += (energies[i] - energies[i + 1]) * (path[i + 1] - path[i])

            norm = np.linalg.norm(tau)
            if norm > 1e-12:
                tangents[i] = tau / norm

        return tangents

    def evolve_string(self, path_init):
        path = np.asarray(path_init, dtype=float).copy()
        n_images = path.shape[0]


        history = [path.copy()]
        energy_history = []

        for it in range(self.max_iter):

            energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(n_images)])
            energy_history.append(energies.copy())


            tangents = self.compute_tangents(path, energies)


            force_perp = np.zeros_like(path)
            max_force = 0.0
            for i in range(1, n_images - 1):
                grad = gradients[i]
                tau = tangents[i]
                f_parallel = np.dot(grad, tau) * tau
                f_perp = grad - f_parallel
                force_perp[i] = -f_perp
                max_force = max(max_force, np.linalg.norm(f_perp))


            path[1:n_images - 1] += self.dt * force_perp[1:n_images - 1]


            path = PathParameterization.reparametrize_equidistant(path, n_images)
            history.append(path.copy())

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
        return path, final_energies, history, energy_history

    def climbing_image_neb(self, path_init):
        path = np.asarray(path_init, dtype=float).copy()
        n_images = path.shape[0]

        for it in range(self.max_iter):
            energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(n_images)])
            tangents = self.compute_tangents(path, energies)


            ts_idx = np.argmax(energies[1:n_images - 1]) + 1

            forces = np.zeros_like(path)
            max_force = 0.0

            for i in range(1, n_images - 1):
                grad = gradients[i]
                tau = tangents[i]
                f_parallel = np.dot(grad, tau) * tau
                f_perp = grad - f_parallel

                if i == ts_idx:

                    forces[i] = -(f_perp - f_parallel)
                else:

                    f_spring = self.spring_const * (
                            np.linalg.norm(path[i + 1] - path[i]) -
                            np.linalg.norm(path[i] - path[i - 1])
                    ) * tau
                    forces[i] = -f_perp + f_spring

                max_force = max(max_force, np.linalg.norm(forces[i]))

            path[1:n_images - 1] += self.dt * forces[1:n_images - 1]
            path = PathParameterization.reparametrize_equidistant(path, n_images)

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
        return path, final_energies, ts_idx


class SequenceManager:

    @staticmethod
    def increment_filename(filename):
        import re
        match = re.search(r'(\d+)(?!.*\d)', filename)
        if not match:
            return None
        num_str = match.group(1)
        new_num = str(int(num_str) + 1).zfill(len(num_str))
        return filename[:match.start()] + new_num + filename[match.end():]

    @staticmethod
    def generate_sequence(base_name, n_frames, extension='dat'):
        names = []
        for i in range(n_frames):
            names.append(f"{base_name}_{i:03d}.{extension}")
        return names


class SymmetryOperations:

    @staticmethod
    def rotate_coordinates(coords, axis, angle):
        coords = np.asarray(coords, dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        c = np.cos(angle)
        s = np.sin(angle)
        n = axis

        R = np.array([
            [c + n[0] ** 2 * (1 - c), n[0] * n[1] * (1 - c) - n[2] * s,
             n[0] * n[2] * (1 - c) + n[1] * s],
            [n[1] * n[0] * (1 - c) + n[2] * s, c + n[1] ** 2 * (1 - c),
             n[1] * n[2] * (1 - c) - n[0] * s],
            [n[2] * n[0] * (1 - c) - n[1] * s, n[2] * n[1] * (1 - c) + n[0] * s,
             c + n[2] ** 2 * (1 - c)]
        ])

        return coords @ R.T

    @staticmethod
    def reflect_coordinates(coords, normal):
        coords = np.asarray(coords, dtype=float)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        R = np.eye(3) - 2.0 * np.outer(n, n)
        return coords @ R.T

    @staticmethod
    def apply_c2v_symmetry(coords, z_axis=None):
        if z_axis is None:
            z_axis = np.array([0, 0, 1])

        configs = [coords.copy()]

        configs.append(SymmetryOperations.rotate_coordinates(coords, z_axis, np.pi))

        configs.append(SymmetryOperations.reflect_coordinates(coords, np.array([0, 1, 0])))

        configs.append(SymmetryOperations.reflect_coordinates(coords, np.array([1, 0, 0])))

        return configs
