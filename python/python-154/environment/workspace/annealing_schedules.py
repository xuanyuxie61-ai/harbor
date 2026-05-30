
import numpy as np
from math import log, exp, gamma as math_gamma


def incomplete_beta(x: float, p: float, q: float, max_iter: int = 1000,
                    eps: float = 1e-14) -> float:
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0,1]")
    if p <= 0.0 or q <= 0.0:
        raise ValueError("p, q must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0


    if p < (p + q) * x:
        xx = 1.0 - x
        cx = x
        pp = q
        qq = p
        indx = True
    else:
        xx = x
        cx = 1.0 - x
        pp = p
        qq = q
        indx = False

    beta_log = math_gamma(pp) + math_gamma(qq) - math_gamma(pp + qq)

    term = 1.0
    ai = 1.0
    value = 1.0
    ns = int(np.floor(qq + cx * (pp + qq)))
    rx = xx / cx
    temp = qq - ai
    if ns == 0:
        rx = xx

    for _ in range(max_iter):
        term *= temp * rx / (pp + ai)
        value += term
        temp_abs = abs(term)
        if temp_abs <= eps and temp_abs <= eps * value:
            break
        ai += 1.0
        ns -= 1
        if 0 <= ns:
            temp = qq - ai
            if ns == 0:
                rx = xx
        else:
            temp = pp + qq

    else:

        pass

    value *= exp(pp * log(xx) + (qq - 1.0) * log(cx) - beta_log) / pp
    if indx:
        value = 1.0 - value
    return float(np.clip(value, 0.0, 1.0))


def collatz_polynomial_next(p1: np.ndarray) -> np.ndarray:
    p1 = np.asarray(p1, dtype=int)
    if not np.all(np.isin(p1, [0, 1])):
        raise ValueError("coefficients must be binary (0 or 1)")

    n = p1.size - 1
    while n >= 0 and p1[n] == 0:
        n -= 1
    if n < 0:
        return np.array([0])
    if n == 0:
        return p1[:1].copy()
    if p1[0] == 0:

        p2 = p1[1:n + 1].copy()
    else:

        p2 = np.zeros(n + 2, dtype=int)
        p2[0:n + 1] = p1[0:n + 1]
        p2[1:n + 2] = (p2[1:n + 2] + p1[0:n + 1]) % 2
        p2[0] = (p2[0] + 1) % 2
    return p2


class AnnealingSchedule:

    def __init__(self, T_total: float = 1.0, n_steps: int = 200):
        if T_total <= 0:
            raise ValueError("T_total must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.T_total = float(T_total)
        self.n_steps = int(n_steps)
        self.times = np.linspace(0.0, T_total, n_steps)
        self.s_values = self.times / T_total

    def linear(self) -> tuple:
        A = 1.0 - self.s_values
        B = self.s_values.copy()
        return A, B

    def polynomial_slowdown(self, degree: int = 3, s_star: float = 0.5) -> tuple:
        if not (0.0 <= s_star <= 1.0):
            raise ValueError("s_star must be in [0,1]")
        p_param = float(degree)
        q_param = float(degree)
        B = np.array([incomplete_beta(s, p_param, q_param) for s in self.s_values])
        A = 1.0 - B
        return A, B

    def logistic_schedule(self, kappa: float = 10.0, s0: float = 0.5) -> tuple:
        B = 1.0 / (1.0 + np.exp(-kappa * (self.s_values - s0)))
        A = 1.0 - B
        return A, B

    def collatz_inspired_schedule(self, n_iter: int = 8) -> tuple:

        p = np.array([1, 1, 1], dtype=int)
        coeffs = []
        for _ in range(n_iter):
            p = collatz_polynomial_next(p)
            deg = p.size - 1
            hw = np.count_nonzero(p)
            coeffs.append(hw / max(deg, 1))

        B = self.s_values.copy()
        eps = 0.03
        for k, c in enumerate(coeffs):
            B += eps * c * np.sin(2.0 * np.pi * (k + 1) * self.s_values)

        B = np.clip(B, 0.0, 1.0)

        for i in range(1, len(B)):
            if B[i] < B[i - 1]:
                B[i] = B[i - 1]
        A = 1.0 - B
        return A, B

    def adiabatic_optimal_local(self, gap_estimate: float = 0.1,
                                 s_star: float = 0.4) -> tuple:
        w = gap_estimate / 2.0
        B = 0.5 * (1.0 + np.tanh((self.s_values - s_star) / w))
        A = 1.0 - B
        return A, B

    def generate_full_hamiltonian_schedule(self, schedule_type: str = "polynomial",
                                            **kwargs) -> dict:
        if schedule_type == "linear":
            A, B = self.linear()
        elif schedule_type == "polynomial":
            A, B = self.polynomial_slowdown(**kwargs)
        elif schedule_type == "logistic":
            A, B = self.logistic_schedule(**kwargs)
        elif schedule_type == "collatz":
            A, B = self.collatz_inspired_schedule(**kwargs)
        elif schedule_type == "adiabatic":
            A, B = self.adiabatic_optimal_local(**kwargs)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")


        dt = self.T_total / max(self.n_steps - 1, 1)
        dA_dt = np.gradient(A, dt)
        dB_dt = np.gradient(B, dt)

        return {
            "times": self.times,
            "s": self.s_values,
            "A": A,
            "B": B,
            "dA_dt": dA_dt,
            "dB_dt": dB_dt,
            "type": schedule_type,
        }
