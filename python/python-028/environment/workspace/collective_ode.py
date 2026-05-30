
import numpy as np
from math import sqrt, sin, cos, pi


class CollectiveHamiltonian:
    def __init__(self, mass_number=100, beta_eq=0.2, gamma_eq=0.0):
        self.A = mass_address = mass_number
        self.beta_eq = beta_eq
        self.gamma_eq = gamma_eq



        self.B_beta = 0.06 * self.A ** (5.0 / 3.0)
        self.B_gamma = self.B_beta



        self.C_beta = 80.0 + 0.3 * self.A
        self.C_gamma = 60.0 + 0.2 * self.A


        self.kappa_beta = 5.0
        self.kappa_gamma = 3.0



        self.I3 = 0.02 * self.A ** (5.0 / 3.0)


        self.G_pair = 25.0 / self.A
        self.delta_pair = 12.0 / sqrt(self.A)


        self.lambda_beta = 0.5
        self.lambda_gamma = 0.5
        self.lambda_R = 0.3

    def pairing_energy(self, beta, gamma):
        return -self.delta_pair ** 2 / self.G_pair * (
            1.0 - 0.5 * (beta - self.beta_eq) ** 2 - 0.3 * gamma ** 2
        )

    def potential_energy(self, beta, gamma):
        V_harmonic = 0.5 * self.C_beta * (beta - self.beta_eq) ** 2
        V_harmonic += 0.5 * self.C_gamma * gamma ** 2
        V_pair = self.pairing_energy(beta, gamma)

        V_cubic = -2.0 * beta ** 3 * cos(3.0 * gamma)
        return V_harmonic + V_pair + V_cubic


def collective_derivatives(t, state, ham):
    beta, gamma, pi_beta, pi_gamma, R3, phi = state


    beta_safe = max(abs(beta), 1e-6)

    d_beta = pi_beta / ham.B_beta
    d_gamma = pi_gamma / (ham.B_gamma * beta_safe ** 2)


    dV_dbeta = ham.C_beta * (beta - ham.beta_eq)
    dV_dbeta += (2.0 * ham.delta_pair ** 2 / ham.G_pair *
                 0.5 * (beta - ham.beta_eq))
    dV_dbeta += -6.0 * beta_safe ** 2 * cos(3.0 * gamma)

    dV_dgamma = ham.C_gamma * gamma
    dV_dgamma += (2.0 * ham.delta_pair ** 2 / ham.G_pair * 0.3 * gamma)
    dV_dgamma += 6.0 * beta_safe ** 3 * sin(3.0 * gamma)

    d_pi_beta = -dV_dbeta - ham.lambda_beta * pi_beta
    d_pi_gamma = -dV_dgamma - ham.lambda_gamma * pi_gamma


    F_ext = 0.1 * sin(0.5 * t) * np.exp(-0.01 * t)
    d_R3 = -ham.lambda_R * R3 + F_ext
    d_phi = R3 / ham.I3

    return np.array([d_beta, d_gamma, d_pi_beta, d_pi_gamma, d_R3, d_phi])


def rk4_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_collective_motion(ham, t_span, n_steps, y0=None):
    t_min, t_max = t_span
    h = (t_max - t_min) / n_steps
    t_array = np.linspace(t_min, t_max, n_steps + 1)

    if y0 is None:
        y0 = np.array([ham.beta_eq, ham.gamma_eq, 0.0, 0.0, 0.0, 0.0])

    y_array = np.zeros((n_steps + 1, 6))
    y_array[0] = y0
    energy_array = np.zeros(n_steps + 1)


    beta, gamma, pi_beta, pi_gamma, R3, phi = y0
    E0 = (pi_beta ** 2 / (2.0 * ham.B_beta) +
          pi_gamma ** 2 / (2.0 * ham.B_gamma * max(beta, 1e-6) ** 2) +
          ham.potential_energy(beta, gamma) +
          0.5 * R3 ** 2 / ham.I3)
    energy_array[0] = E0

    for n in range(n_steps):
        y_array[n + 1] = rk4_step(collective_derivatives,
                                   t_array[n], y_array[n], h, ham)
        beta, gamma, pi_beta, pi_gamma, R3, phi = y_array[n + 1]
        E = (pi_beta ** 2 / (2.0 * ham.B_beta) +
             pi_gamma ** 2 / (2.0 * ham.B_gamma * max(beta, 1e-6) ** 2) +
             ham.potential_energy(beta, gamma) +
             0.5 * R3 ** 2 / ham.I3)
        energy_array[n + 1] = E

    return t_array, y_array, energy_array


def adiabatic_invariant(y_array, ham, dt):
    beta = y_array[:, 0]
    pi_beta = y_array[:, 2]


    crossings = 0
    invariant = 0.0
    for i in range(1, len(beta)):
        if beta[i] * beta[i - 1] < 0:
            crossings += 1
        invariant += abs(pi_beta[i]) * abs(beta[i] - beta[i - 1])

    if crossings > 0:
        invariant /= (2.0 * pi * crossings)
    return invariant
