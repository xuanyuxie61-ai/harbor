
import numpy as np
from scipy.integrate import solve_ivp


def chiral_lagrangian_deriv(t, y, mu_chiral: float = 2.0,
                            f0: float = 0.092, B0: float = 2.7e3,
                            mq: float = 0.0035):
    u, v = y

    mpi_sq = 2.0 * B0 * mq / (f0 ** 2)
    omega0_sq = mpi_sq
    gamma = 0.5

    dudt = v
    dvdt = mu_chiral * (1.0 - u ** 2) * v - omega0_sq * u - gamma * v
    return np.array([dudt, dvdt])


def solve_chiral_oscillator(y0: np.ndarray, t_span: tuple,
                            mu_chiral: float = 2.0) -> tuple:
    sol = solve_ivp(
        lambda t, y: chiral_lagrangian_deriv(t, y, mu_chiral),
        t_span, y0, method='RK45', rtol=1e-9, atol=1e-12,
        dense_output=True
    )
    t = np.linspace(t_span[0], t_span[1], 500)
    y = sol.sol(t)
    return t, y


class QuarkMesonReactionNetwork:

    def __init__(self, a: float = 1.0, b: float = 0.5,
                 kc: float = 0.1, kn: float = 0.05,
                 rmax1: float = 1.0, rmax2: float = 0.3,
                 e1: float = 0.2, e2: float = 0.1):
        self.a = a
        self.b = b
        self.kc = kc
        self.kn = kn
        self.rmax1 = rmax1
        self.rmax2 = rmax2
        self.e1 = e1
        self.e2 = e2


        self.S = np.array([
            [-a,  a,  0.0],
            [-b,  b,  0.0],
            [ 1.0, -1.0, -1.0],
            [ 0.0,  0.0,  1.0],
        ])

    def rates(self, conc: np.ndarray) -> np.ndarray:
        q, qbar, pi, rho = conc
        q = max(q, 1e-12)
        qbar = max(qbar, 1e-12)
        pi = max(pi, 1e-12)

        r1 = self.rmax1 * q * qbar / (self.kc + q)
        r2 = self.e1 * pi
        r3 = self.rmax2 * pi ** 2 / (self.kn + pi)

        return np.array([r1, r2, r3])

    def deriv(self, t: float, conc: np.ndarray) -> np.ndarray:
        r = self.rates(conc)
        return self.S @ r

    def solve(self, c0: np.ndarray, t_span: tuple, n_points: int = 200) -> tuple:
        sol = solve_ivp(
            self.deriv, t_span, c0, method='BDF',
            rtol=1e-8, atol=1e-10, dense_output=True
        )
        t = np.linspace(t_span[0], t_span[1], n_points)
        c = sol.sol(t)
        return t, c


def chiral_condensate_from_reaction(t: np.ndarray, c: np.ndarray,
                                    f0: float = 0.092) -> np.ndarray:
    q = c[0, :]
    qbar = c[1, :]
    pi = c[2, :]
    sigma = 0.5 * (q + qbar) - pi / (f0 ** 2)
    return sigma


def pion_decay_constant_from_dynamics(mpi: float, B0: float = 2.7e3,
                                      mq: float = 0.0035) -> float:
    mq_mev = mq * 1e3
    fpi = np.sqrt((2.0 * mq_mev * B0) / (mpi ** 2 + 1e-10))
    return fpi
