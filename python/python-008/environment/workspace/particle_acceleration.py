
import numpy as np



_A21 = 2.71644396264860
_A31 = -6.95653259006152
_A32 = 0.78313689457981
_A41 = 0.0
_A42 = 0.48257353309214
_A43 = 0.26171080165848
_A51 = 0.47012396888046
_A52 = 0.36597075368373
_A53 = 0.08906615686702
_A54 = 0.07483912056879

_Q1 = 2.12709852335625
_Q2 = 2.73245878238737
_Q3 = 11.22760917474960
_Q4 = 13.36199560336697


def bohm_diffusion(gamma, B):
    m_e = 9.10938356e-28
    c = 2.99792458e10
    e = 4.80320427e-10

    r_L = gamma * m_e * c ** 2 / (e * B)
    D = (1.0 / 3.0) * r_L * c

    return D


def acceleration_coefficient(gamma, u1, u2):
    c = 2.99792458e10
    beta_sh = (u1 - u2) / c
    beta_sh = np.clip(beta_sh, 0.0, 1.0)
    return (4.0 / 3.0) * beta_sh * gamma


def fi_dsa(gamma, u1, u2):
    return acceleration_coefficient(gamma, u1, u2)


def gi_dsa(gamma, B):
    D = bohm_diffusion(gamma, B)
    D = max(D, 0.0)
    return np.sqrt(2.0 * D)


def rk4_ti_step(x, t, h, q, fi, gi):
    n1 = np.random.randn()
    w1 = n1 * np.sqrt(_Q1 * q / h)
    k1 = h * fi(x) + h * gi(x) * w1

    t2 = t + _A21 * h
    x2 = x + _A21 * k1
    n2 = np.random.randn()
    w2 = n2 * np.sqrt(_Q2 * q / h)
    k2 = h * fi(x2) + h * gi(x2) * w2

    t3 = t + (_A31 + _A32) * h
    x3 = x + _A31 * k1 + _A32 * k2
    n3 = np.random.randn()
    w3 = n3 * np.sqrt(_Q3 * q / h)
    k3 = h * fi(x3) + h * gi(x3) * w3

    t4 = t + (_A41 + _A42 + _A43) * h
    x4 = x + _A41 * k1 + _A42 * k2
    n4 = np.random.randn()
    w4 = n4 * np.sqrt(_Q4 * q / h)
    k4 = h * fi(x4) + h * gi(x4) * w4

    xstar = x + _A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4
    return xstar


def accelerate_electrons(gamma_0, n_particles, t_max, dt, B, u1, u2,
                         q_noise=1.0, gamma_max=1e8):
    n_steps = max(1, int(t_max / dt))
    gamma = np.full(n_particles, float(gamma_0))

    for _ in range(n_steps):
        for i in range(n_particles):
            gamma[i] = rk4_ti_step(
                gamma[i], 0.0, dt, q_noise,
                lambda x: fi_dsa(x, u1, u2),
                lambda x: gi_dsa(x, B)
            )

        gamma = np.clip(gamma, 1.0, gamma_max)

    return gamma
