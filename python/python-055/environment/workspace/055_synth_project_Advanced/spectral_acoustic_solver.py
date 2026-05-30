
import numpy as np
from scipy.integrate import quad


class SpectralAcousticSolver:

    def __init__(self, depth: float = 4000.0, frequency: float = 12000.0):
        self.H = float(depth)
        self.f = float(frequency)
        self.omega = 2.0 * np.pi * self.f

    def _basis(self, z: np.ndarray, i: int) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)

        xi = z / self.H

        return (xi ** i) * (xi - 1.0)

    def _basis_derivative(self, z: np.ndarray, i: int) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        xi = z / self.H

        dphi_dxi = i * (xi ** (i - 1)) * (xi - 1.0) + xi ** i
        return dphi_dxi / self.H

    def _compute_B_entry(self, i: int, j: int) -> float:
        def integrand(z):
            return self._basis_derivative(np.array([z]), i)[0] * \
                   self._basis_derivative(np.array([z]), j)[0]
        val, _ = quad(integrand, 0.0, self.H, limit=100)
        return float(val)

    def _compute_K_entry(self, i: int, j: int, k_func) -> float:
        def integrand(z):
            kz = k_func(np.array([z]))[0]
            return (kz ** 2) * self._basis(np.array([z]), i)[0] * \
                   self._basis(np.array([z]), j)[0]
        val, _ = quad(integrand, 0.0, self.H, limit=100)
        return float(val)

    def _compute_F_entry(self, i: int, source_func) -> float:
        def integrand(z):
            return source_func(np.array([z]))[0] * self._basis(np.array([z]), i)[0]
        val, _ = quad(integrand, 0.0, self.H, limit=100)
        return float(val)

    def solve(
        self,
        n_basis: int,
        sound_speed_func,
        source_func,
        verbose: bool = False
    ) -> dict:
        if n_basis < 1:
            raise ValueError("n_basis 必须 >= 1")

        k_func = lambda z: self.omega / sound_speed_func(z)


        B = np.zeros((n_basis, n_basis), dtype=np.float64)
        K = np.zeros((n_basis, n_basis), dtype=np.float64)
        F = np.zeros(n_basis, dtype=np.float64)

        for i in range(n_basis):
            for j in range(n_basis):
                B[i, j] = self._compute_B_entry(i + 1, j + 1)
                K[i, j] = self._compute_K_entry(i + 1, j + 1, k_func)
            F[i] = self._compute_F_entry(i + 1, source_func)


        A = -B + K

        cond_num = np.linalg.cond(A)
        if verbose:
            print(f"  矩阵条件数: {cond_num:.4e}")


        try:
            coeffs = np.linalg.solve(A, F)
        except np.linalg.LinAlgError:

            coeffs, _, _, _ = np.linalg.lstsq(A, F, rcond=None)


        def solution(z):
            z = np.asarray(z, dtype=np.float64)
            p = np.zeros_like(z)
            for i in range(n_basis):
                p += coeffs[i] * self._basis(z, i + 1)
            return p

        def solution_derivative(z):
            z = np.asarray(z, dtype=np.float64)
            dp = np.zeros_like(z)
            for i in range(n_basis):
                dp += coeffs[i] * self._basis_derivative(z, i + 1)
            return dp

        return {
            'coeffs': coeffs,
            'solution': solution,
            'derivative': solution_derivative,
            'matrix_A': A,
            'rhs_F': F,
            'cond_number': float(cond_num),
            'n_basis': n_basis,
        }

    def compute_errors(
        self,
        coeffs: np.ndarray,
        exact_solution_func,
        exact_derivative_func = None
    ) -> dict:
        def l2_integrand(z):
            z_arr = np.array([z])
            u_num = np.zeros_like(z_arr)
            for i in range(len(coeffs)):
                u_num += coeffs[i] * self._basis(z_arr, i + 1)
            diff = u_num[0] - exact_solution_func(z_arr)[0]
            return diff ** 2

        l2_sq, _ = quad(l2_integrand, 0.0, self.H, limit=100)
        err_l2 = np.sqrt(l2_sq)

        if exact_derivative_func is not None:
            def h1_integrand(z):
                z_arr = np.array([z])
                u_num = np.zeros_like(z_arr)
                du_num = np.zeros_like(z_arr)
                for i in range(len(coeffs)):
                    u_num += coeffs[i] * self._basis(z_arr, i + 1)
                    du_num += coeffs[i] * self._basis_derivative(z_arr, i + 1)
                diff_u = u_num[0] - exact_solution_func(z_arr)[0]
                diff_du = du_num[0] - exact_derivative_func(z_arr)[0]
                return diff_u ** 2 + diff_du ** 2

            h1_sq, _ = quad(h1_integrand, 0.0, self.H, limit=100)
            err_h1 = np.sqrt(h1_sq)
        else:
            err_h1 = err_l2

        return {
            'err_l2': float(err_l2),
            'err_h1': float(err_h1),
        }

    def solve_with_reference(
        self,
        n_basis: int,
        sound_speed_func,
        source_func,
        exact_solution_func = None,
        exact_derivative_func = None
    ) -> dict:
        result = self.solve(n_basis, sound_speed_func, source_func)
        if exact_solution_func is not None:
            errors = self.compute_errors(
                result['coeffs'], exact_solution_func, exact_derivative_func
            )
            result.update(errors)
        return result
