
import numpy as np


class OrthogonalPolynomialFit:

    def __init__(self, x_data, f_data, weights, n_terms):
        x_data = np.asarray(x_data, dtype=np.float64)
        f_data = np.asarray(f_data, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        n_terms = int(n_terms)

        if x_data.ndim != 1 or f_data.ndim != 1 or weights.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays.")
        if x_data.size != f_data.size or x_data.size != weights.size:
            raise ValueError("x_data, f_data, and weights must have the same length.")
        if n_terms < 1:
            raise ValueError("n_terms must be at least 1.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
        if np.sum(weights) == 0:
            raise ValueError("Sum of weights must be positive.")

        self.x_data = x_data
        self.n_terms = n_terms
        self._fit(x_data, f_data, weights)

    def _fit(self, x, f, w):
        w = w / np.sum(w)
        x = x.flatten()
        f = f.flatten()
        M = x.size


        B = np.zeros(self.n_terms)
        C = np.zeros(self.n_terms)
        D = np.zeros(self.n_terms)
        S = np.zeros(self.n_terms)


        Pjm1 = np.ones(M)
        S[0] = np.sum(w * Pjm1 ** 2)
        D[0] = np.sum(w * f * Pjm1) / S[0]
        error = f - D[0] * Pjm1

        if self.n_terms == 1:
            self.B = B
            self.C = C
            self.D = D
            self.S = S
            self.residual = error
            return


        B[0] = np.sum(w * x) / np.sum(w)
        Pj = x - B[0]
        S[0] = np.sum(w * Pjm1 ** 2)

        P = Pj * w
        D[1] = np.sum(error * Pj) / np.sum(w * Pj ** 2)
        error = error - D[1] * Pj

        for j in range(2, self.n_terms):
            P = Pj * w
            B[j - 1] = np.sum(x * Pj * P) / np.sum(w * Pj ** 2)
            C[j - 1] = np.sum(w * Pj ** 2) / np.sum(w * Pjm1 ** 2)
            S[j - 1] = np.sum(w * Pj ** 2)

            P_new = (x - B[j - 1]) * Pj - C[j - 1] * Pjm1
            D[j] = np.sum(error * P_new) / np.sum(w * P_new ** 2)
            error = error - D[j] * P_new

            Pjm1 = Pj
            Pj = P_new

        S[self.n_terms - 1] = np.sum(w * Pj ** 2)
        self.B = B
        self.C = C
        self.D = D
        self.S = S
        self.residual = error

    def evaluate(self, xq):
        xq = np.asarray(xq, dtype=np.float64)
        M = xq.size
        yq = np.zeros(M)

        Pjm1 = np.ones(M)
        yq += self.D[0] * Pjm1

        if self.n_terms == 1:
            return yq

        Pj = xq - self.B[0]
        yq += self.D[1] * Pj

        for j in range(2, self.n_terms):
            P_new = (xq - self.B[j - 1]) * Pj - self.C[j - 1] * Pjm1
            yq += self.D[j] * P_new
            Pjm1 = Pj
            Pj = P_new

        return yq

    def basis_polynomial(self, xq, j):
        if j < 0 or j >= self.n_terms:
            raise ValueError("Invalid basis index.")
        xq = np.asarray(xq, dtype=np.float64)
        M = xq.size
        if j == 0:
            return np.ones(M)
        Pjm1 = np.ones(M)
        Pj = xq - self.B[0]
        if j == 1:
            return Pj
        for k in range(2, j + 1):
            P_new = (xq - self.B[k - 1]) * Pj - self.C[k - 1] * Pjm1
            Pjm1 = Pj
            Pj = P_new
        return Pj


def fit_permeability_field_1d(x, k_data, n_terms=6):

    weights = np.ones_like(x)
    return OrthogonalPolynomialFit(x, k_data, weights, n_terms)


def fit_thermal_conductivity_field_1d(x, lambda_data, n_terms=6):
    weights = np.ones_like(x)
    return OrthogonalPolynomialFit(x, lambda_data, weights, n_terms)
