
import numpy as np


class CompanionMatrixEigenvalue:

    @staticmethod
    def hermite_companion_matrix(coeffs):
        coeffs = np.asarray(coeffs, dtype=float)
        n = len(coeffs) - 1

        if n < 1:
            raise ValueError("多项式次数必须至少为 1")
        if abs(coeffs[-1]) < 1e-14:
            raise ValueError("首项系数不能为零")

        A = np.zeros((n, n))


        for i in range(n - 1):
            A[i, i + 1] = 0.5


        for i in range(1, n):
            A[i, i - 1] = i


        for j in range(n):
            A[n - 1, j] -= coeffs[j] / (2.0 * coeffs[n])

        return A

    @staticmethod
    def chebyshev_companion_matrix(coeffs):
        coeffs = np.asarray(coeffs, dtype=float)
        n = len(coeffs) - 1

        if n < 1:
            raise ValueError("多项式次数必须至少为 1")

        A = np.zeros((n, n))

        A[0, 1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = 0.5
            A[i, i + 1] = 0.5
        A[n - 1, :] -= coeffs[:-1] / (2.0 * coeffs[-1])
        if n >= 2:
            A[n - 1, n - 2] += 0.5

        return A

    @staticmethod
    def find_roots(coeffs, basis='power'):
        if basis == 'power':
            return np.roots(coeffs[::-1])
        elif basis == 'hermite':
            A = CompanionMatrixEigenvalue.hermite_companion_matrix(coeffs)
            return np.linalg.eigvals(A)
        elif basis == 'chebyshev':
            A = CompanionMatrixEigenvalue.chebyshev_companion_matrix(coeffs)
            return np.linalg.eigvals(A)
        else:
            raise ValueError(f"不支持的基函数类型: {basis}")


class LogarithmicNorm:

    @staticmethod
    def log_norm_l1(A):
        A = np.asarray(A, dtype=complex)
        n = A.shape[0]

        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=0)
        return np.max(c + d)

    @staticmethod
    def log_norm_l2(A):
        A = np.asarray(A, dtype=complex)
        B = 0.5 * (A + A.conj().T)
        eigenvalues = np.linalg.eigvalsh(B)
        return np.max(eigenvalues)

    @staticmethod
    def log_norm_inf(A):
        A = np.asarray(A, dtype=complex)
        n = A.shape[0]

        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=1)
        return np.max(c + d)

    @staticmethod
    def log_norm(A, p=2):
        if p == 1:
            return LogarithmicNorm.log_norm_l1(A)
        elif p == 2:
            return LogarithmicNorm.log_norm_l2(A)
        elif p == np.inf:
            return LogarithmicNorm.log_norm_inf(A)
        else:
            raise ValueError("p 必须为 1, 2 或 np.inf")


class LinearStabilityAnalysis:

    @staticmethod
    def compute_jacobian_1d_diffusion_reaction(n, dx, D, reaction_derivative):
        J = np.zeros((n, n))


        for i in range(1, n - 1):
            J[i, i - 1] = D / (dx ** 2)
            J[i, i] = -2.0 * D / (dx ** 2)
            J[i, i + 1] = D / (dx ** 2)


        for i in range(n):
            J[i, i] += reaction_derivative[i]

        return J

    @staticmethod
    def stability_criterion_eigenvalues(jacobian):
        eigenvalues = np.linalg.eigvals(jacobian)
        max_real = np.max(np.real(eigenvalues))

        return {
            'stable': max_real < 0,
            'max_real': max_real,
            'eigenvalues': eigenvalues
        }

    @staticmethod
    def cfl_condition_1d_advection_diffusion(v, D, dx, safety_factor=0.5):
        dt_adv = dx / max(abs(v), 1e-14)
        dt_diff = dx ** 2 / (2.0 * max(D, 1e-14))
        return safety_factor * min(dt_adv, dt_diff)

    @staticmethod
    def cfl_condition_2d_navier_stokes(vx_max, vy_max, nu, dx, dy, safety_factor=0.25):
        dt_adv_x = dx / max(abs(vx_max), 1e-14)
        dt_adv_y = dy / max(abs(vy_max), 1e-14)
        dt_diff = 0.5 / (nu * (1.0 / dx ** 2 + 1.0 / dy ** 2))

        return safety_factor * min(dt_adv_x, dt_adv_y, dt_diff)


class BifurcationAnalysis:

    @staticmethod
    def logistic_map(x, r):
        return r * x * (1.0 - x)

    @staticmethod
    def find_attractors(r, x0=0.5, warmup=100, n_iter=500, tol=1e-5):
        x = x0
        for _ in range(warmup):
            x = BifurcationAnalysis.logistic_map(x, r)

        attractors = [x]
        for _ in range(n_iter):
            x = BifurcationAnalysis.logistic_map(x, r)

            exists = False
            for a in attractors:
                if abs(x - a) < tol:
                    exists = True
                    break
            if not exists:
                attractors.append(x)

        return np.array(attractors)

    @staticmethod
    def lyapunov_exponent_logistic(r, x0=0.5, n_iter=10000):
        x = x0
        lyap_sum = 0.0

        for _ in range(n_iter):
            x = BifurcationAnalysis.logistic_map(x, r)
            derivative = abs(r * (1.0 - 2.0 * x))
            if derivative < 1e-14:
                derivative = 1e-14
            lyap_sum += np.log(derivative)

        return lyap_sum / n_iter

    @staticmethod
    def phase_transition_bifurcation_parameter(T_undercooling, gamma, m_L, C0, D_l, k_p):

        delta_T0 = -m_L * C0 * (1.0 - k_p)
        if abs(delta_T0) < 1e-14:
            delta_T0 = 1e-14

        d0 = gamma / delta_T0


        V_tip = D_l * T_undercooling / (gamma)


        sigma_star = 2.0 * D_l * d0 * V_tip / (D_l ** 2)

        return {
            'capillary_length': d0,
            'characteristic_velocity': V_tip,
            'stability_parameter': sigma_star,
            'unstable': sigma_star < 0.025
        }
