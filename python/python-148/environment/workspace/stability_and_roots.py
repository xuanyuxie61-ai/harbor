
import numpy as np
from numpy.linalg import eigvalsh, norm


def logarithmic_norm(A, p=2):
    A = np.asarray(A, dtype=np.complex128 if np.iscomplexobj(A) else float)
    if p == 1:

        diag = np.diag(A).real
        col_sums = np.sum(np.abs(A), axis=0) - np.abs(diag)
        return float(np.max(diag + col_sums))
    elif p == np.inf:

        diag = np.diag(A).real
        row_sums = np.sum(np.abs(A), axis=1) - np.abs(diag)
        return float(np.max(diag + row_sums))
    elif p == 2:

        sym_part = 0.5 * (A + A.conj().T)
        eigenvalues = eigvalsh(sym_part)
        return float(np.max(eigenvalues))
    else:
        raise ValueError("p must be 1, 2, or np.inf")


def exponential_bound_estimate(A, t, p=2):
    mu = logarithmic_norm(A, p)
    return np.exp(mu * t)


def cauchy_polynomial_root_bound(coeffs):
    c = np.asarray(coeffs, dtype=float)
    n = len(c) - 1
    if n < 0:
        return 0.0
    if n == 0:
        return 0.0
    abs_c = np.abs(c)

    def q(x):
        if x <= 0:
            return -abs_c[-1]
        val = abs_c[0] * (x ** n)
        for k in range(1, n + 1):
            val -= abs_c[k] * (x ** (n - k))
        return val


    R = 1.0
    while q(R) < 0:
        R *= 2.0
        if R > 1e12:
            break


    lo, hi = 0.0, R
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if q(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def polynomial_from_matrix(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    coeffs = np.zeros(n + 1, dtype=float)
    coeffs[0] = 1.0
    B = np.eye(n)
    for k in range(1, n + 1):
        B = A @ B
        trace_Bk = np.trace(B)
        coeffs[k] = -trace_Bk / k
        B = B + coeffs[k] * np.eye(n)
    return coeffs


def neural_mass_jacobian(ei_oscillator, E_star, I_star):




    pass


class BCIStabilityAnalyzer:

    def __init__(self, ei_oscillator, feedback_gain=0.5):
        self.ei_osc = ei_oscillator
        self.K = feedback_gain

    def find_equilibrium(self, E_guess=0.3, I_guess=0.1, max_iter=100, tol=1e-10):
        from utils import sigmoid_activation
        x = np.array([E_guess, I_guess], dtype=float)
        for _ in range(max_iter):
            E, I = x
            s_val = 0.0
            x_e = self.ei_osc.a_ee * E - self.ei_osc.a_ei * I + self.ei_osc.P_e + self.ei_osc.k_e * s_val
            x_i = self.ei_osc.a_ie * E - self.ei_osc.a_ii * I + self.ei_osc.P_i + self.ei_osc.k_i * s_val
            f1 = -E + sigmoid_activation(x_e, self.ei_osc.theta_e, self.ei_osc.sigma_e)
            f2 = -I + sigmoid_activation(x_i, self.ei_osc.theta_i, self.ei_osc.sigma_i)
            F = np.array([f1, f2], dtype=float)
            J = neural_mass_jacobian(self.ei_osc, E, I)
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            x = x + dx
            if norm(dx) < tol:
                break
        return x

    def analyze_open_loop_stability(self):
        eq = self.find_equilibrium()
        J = neural_mass_jacobian(self.ei_osc, eq[0], eq[1])
        eigenvalues = np.linalg.eigvals(J)
        max_real = float(np.max(eigenvalues.real))
        mu1 = logarithmic_norm(J, p=1)
        mu2 = logarithmic_norm(J, p=2)
        mu_inf = logarithmic_norm(J, p=np.inf)

        char_poly_coeffs = polynomial_from_matrix(J)
        root_bound = cauchy_polynomial_root_bound(char_poly_coeffs)
        return {
            'equilibrium': eq,
            'jacobian': J,
            'eigenvalues': eigenvalues,
            'max_real_part': max_real,
            'mu_1': mu1,
            'mu_2': mu2,
            'mu_inf': mu_inf,
            'characteristic_polynomial': char_poly_coeffs,
            'cauchy_root_bound': root_bound,
            'is_stable': max_real < 0
        }

    def analyze_closed_loop_stability(self, B=np.array([[1.0], [0.0]]),
                                      C=np.array([[1.0, 0.0]])):
        eq = self.find_equilibrium()
        J = neural_mass_jacobian(self.ei_osc, eq[0], eq[1])
        A_cl = J - B * self.K * C
        eigenvalues = np.linalg.eigvals(A_cl)
        max_real = float(np.max(eigenvalues.real))
        mu2 = logarithmic_norm(A_cl, p=2)
        char_poly_coeffs = polynomial_from_matrix(A_cl)
        root_bound = cauchy_polynomial_root_bound(char_poly_coeffs)
        return {
            'closed_loop_matrix': A_cl,
            'eigenvalues': eigenvalues,
            'max_real_part': max_real,
            'mu_2': mu2,
            'cauchy_root_bound': root_bound,
            'is_stable': max_real < 0
        }

    def compute_lyapunov_exponents(self, n_steps=5000, dt=0.001):
        n_dim = 2
        Q = np.eye(n_dim)
        exponents = np.zeros(n_dim)

        x = self.find_equilibrium()
        for step in range(n_steps):

            J = neural_mass_jacobian(self.ei_osc, x[0], x[1])
            M = J @ Q

            Q, R = np.linalg.qr(M)
            for i in range(n_dim):
                exponents[i] += np.log(max(abs(R[i, i]), 1e-15))


        exponents /= (n_steps * dt)
        return exponents
