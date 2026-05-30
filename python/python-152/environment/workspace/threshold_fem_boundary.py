import numpy as np
from scipy.linalg import solve_banded


class ThresholdFEM:

    def __init__(self, p_left: float = 0.0, p_right: float = 0.5, n_elements: int = 100):
        self.p_left = p_left
        self.p_right = p_right
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.h = (p_right - p_left) / n_elements

    def _a_coeff(self, p: float) -> float:
        return max(p * (1.0 - p), 1e-12)

    def _b_coeff(self, p: float) -> float:
        return 1.0

    def _f_coeff(self, p: float, code_distance: float, nu: float = 1.0) -> float:
        p_th = 0.1
        alpha = 50.0
        val = ((p / max(p_th, 1e-12)) ** (code_distance / 2.0))
        if p < p_th:
            val *= np.exp(-alpha * (p_th - p) ** 2)
        return val

    def assemble_and_solve(self, code_distance: float) -> tuple:
        n = self.n_nodes
        h = self.h


        adiag = np.zeros(n)
        aleft = np.zeros(n - 1)
        arite = np.zeros(n - 1)
        rhs = np.zeros(n)

        for i in range(1, n - 1):
            x_i = self.p_left + i * h
            x_im1 = x_i - h
            a_mid = 0.5 * (self._a_coeff(x_i) + self._a_coeff(x_im1))
            a_mid_next = 0.5 * (self._a_coeff(x_i + h) + self._a_coeff(x_i))
            b_mid = self._b_coeff(x_i)
            f_mid = self._f_coeff(x_i, code_distance)

            adiag[i] += a_mid / h + a_mid_next / h + b_mid * (2.0 * h / 3.0)
            aleft[i - 1] += -a_mid / h + b_mid * (h / 6.0)
            arite[i] += -a_mid_next / h + b_mid * (h / 6.0)
            rhs[i] += f_mid * h


        adiag[0] = 1.0
        rhs[0] = 0.0
        adiag[-1] = 1.0
        rhs[-1] = 1.0
        aleft[-1] = 0.0
        arite[0] = 0.0


        ab = np.zeros((3, n))
        ab[0, 1:] = arite
        ab[1, :] = adiag
        ab[2, :-1] = aleft
        u = solve_banded((1, 1), ab, rhs)
        nodes = np.linspace(self.p_left, self.p_right, n)
        return nodes, u


class EdgeDetectorThreshold:

    def __init__(self, p_values: np.ndarray, P_L_values: np.ndarray):
        if len(p_values) != len(P_L_values):
            raise ValueError("p_values and P_L_values must have same length.")
        self.p = p_values.copy()
        self.P_L = P_L_values.copy()

    def derivative_edge_detection(self, window: int = 5) -> float:
        n = len(self.p)
        dP = np.zeros(n)
        for i in range(window, n - window):

            x_local = self.p[i - window:i + window + 1]
            y_local = self.P_L[i - window:i + window + 1]
            x_mean = np.mean(x_local)
            y_mean = np.mean(y_local)
            num = np.sum((x_local - x_mean) * (y_local - y_mean))
            den = np.sum((x_local - x_mean) ** 2)
            dP[i] = num / max(den, 1e-12)
        idx = np.argmax(np.abs(dP))
        return self.p[idx]

    def sigmoid_fit_threshold(self) -> float:

        y = np.clip(self.P_L, 1e-6, 1.0 - 1e-6)
        logit_y = np.log(y / (1.0 - y))

        A = np.vstack([np.ones_like(self.p), self.p]).T
        a, b = np.linalg.lstsq(A, logit_y, rcond=None)[0]
        p_th = -a / b
        return p_th

    def second_derivative_zero_crossing(self) -> float:
        n = len(self.p)
        d2P = np.zeros(n)
        h = np.mean(np.diff(self.p))
        for i in range(2, n - 2):
            d2P[i] = (self.P_L[i + 1] - 2 * self.P_L[i] + self.P_L[i - 1]) / (h ** 2)

        for i in range(2, n - 3):
            if d2P[i] * d2P[i + 1] < 0:
                return 0.5 * (self.p[i] + self.p[i + 1])
        return self.p[n // 2]

    def fx_edge_profile(self, steepness: float = 10.0) -> np.ndarray:
        p_th = self.sigmoid_fit_threshold()
        return 1.0 / (1.0 + np.exp(-steepness * (self.p - p_th)))

    def fxy_shepp_logan_like(self, width: float = 0.05) -> np.ndarray:
        p_th = self.sigmoid_fit_threshold()
        profile = np.zeros_like(self.p)
        for offset in [-width, 0.0, width]:
            profile += np.exp(-((self.p - (p_th + offset)) / (width / 3)) ** 2)
        pmax = np.max(profile)
        return profile / max(pmax, 1e-12)


def finite_size_scaling(p_values: np.ndarray, P_L_values: np.ndarray,
                        code_distances: list, nu: float = 1.0) -> dict:
    from scipy.optimize import minimize

    def collapse_quality(params):
        p_th, nu_fit = params
        if p_th <= 0 or p_th >= 0.5 or nu_fit <= 0.1 or nu_fit >= 5.0:
            return 1e6
        total_var = 0.0
        count = 0
        for i, d in enumerate(code_distances):
            x = (p_values - p_th) * (d ** (1.0 / nu_fit))
            y = P_L_values[i, :]

            if i == 0:
                master_x = x
                master_y = y
                continue

            y_interp = np.interp(master_x, x, y, left=0.0, right=1.0)
            total_var += np.mean((y_interp - master_y) ** 2)
            count += 1
        return total_var / max(count, 1)

    result = minimize(collapse_quality, x0=[0.1, 1.0],
                      method='Nelder-Mead',
                      bounds=[(0.01, 0.4), (0.5, 3.0)])
    return {
        "p_th": result.x[0],
        "nu": result.x[1],
        "quality": result.fun
    }
