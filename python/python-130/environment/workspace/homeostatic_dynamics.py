# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import ellipk, ellipj
from typing import Tuple, Optional
from numerical_integrator import rk1_integrate, rk4_integrate


def spring_deriv(t: float, y: np.ndarray, m: float, b: float, k: float, F_ext: float = 0.0) -> np.ndarray:
    if m <= 0.0:
        raise ValueError("m must be positive.")
    if b < 0.0 or k < 0.0:
        raise ValueError("b and k must be non-negative.")

    w = y[0]
    v = y[1]

    dudt = v
    dvdt = -(k / m) * w - (b / m) * v + F_ext / m

    return np.array([dudt, dvdt])


def classify_damping(m: float, b: float, k: float) -> str:
    if m <= 0.0 or k < 0.0 or b < 0.0:
        raise ValueError("Invalid parameters.")

    disc = b * b - 4.0 * m * k
    if disc < 0.0:
        return "underdamped"
    elif np.isclose(disc, 0.0):
        return "critically_damped"
    else:
        return "overdamped"


def spring_parameters(
    m: float = 1.0,
    b: float = 0.5,
    k: float = 1.0,
    w_target: float = 0.5,
) -> dict:
    omega_n = np.sqrt(k / m)
    zeta = b / (2.0 * np.sqrt(m * k)) if k > 0 else np.inf
    tau = 2.0 * m / b if b > 0 else np.inf
    regime = classify_damping(m, b, k)

    return {
        "omega_n": omega_n,
        "zeta": zeta,
        "tau": tau,
        "regime": regime,
        "w_target": w_target,
    }


def simulate_homeostatic_response(
    w0: float = 0.2,
    v0: float = 0.0,
    m: float = 1.0,
    b: float = 0.5,
    k: float = 1.0,
    F_ext: float = 0.0,
    t_final: float = 50.0,
    n_steps: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    params = spring_parameters(m, b, k)

    def rhs(t, y):
        return spring_deriv(t, y, m, b, k, F_ext)

    t, y = rk4_integrate(rhs, (0.0, t_final), np.array([w0, v0]), n_steps)
    return t, y, params


def pendulum_nonlinear_exact(
    t: np.ndarray,
    g: float = 9.81,
    l: float = 1.0,
    theta0: float = np.pi / 3.0,
    thetadot0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if l <= 0.0:
        raise ValueError("l must be positive.")

    omega = np.sqrt(g / l)
    k0 = np.sin(theta0 / 2.0)
    ep = 4.0 * g / l
    e0 = thetadot0 ** 2 + ep * k0 ** 2

    if e0 < 1e-15:
        return np.zeros_like(t), np.zeros_like(t)

    k = np.sqrt(e0 / ep)



    m = min(k ** 2, 0.999999)


    if abs(k0) > 1e-15:

        sn_val = k0 / k if k > 1e-15 else 0.0
        sn_val = np.clip(sn_val, -1.0, 1.0)



        if m < 0.99:
            u0 = ellipk(m) * np.arcsin(sn_val) / (np.pi / 2)
        else:
            u0 = np.arcsin(sn_val)
    else:
        u0 = 0.0


    u = omega * t + u0
    sn, cn, dn = ellipj(u, m)


    theta = 2.0 * np.sign(cn) * np.arcsin(np.clip(np.abs(k * sn), 0.0, 1.0))
    thetadot = np.sign(thetadot0) * np.sqrt(e0) * cn

    return theta, thetadot


def compute_pendulum_period(
    g: float = 9.81,
    l: float = 1.0,
    theta0: float = np.pi / 3.0,
) -> float:
    if l <= 0.0:
        raise ValueError("l must be positive.")

    omega = np.sqrt(g / l)
    k = np.sin(theta0 / 2.0)
    k2 = k ** 2

    K = ellipk(k2)
    T = 4.0 * K / omega
    return T


def simulate_network_synchronization(
    n_neurons: int = 10,
    g: float = 1.0,
    l: float = 1.0,
    coupling: float = 0.1,
    t_final: float = 20.0,
    n_steps: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_neurons < 1:
        raise ValueError("n_neurons must be >= 1.")
    if l <= 0.0 or g < 0.0:
        raise ValueError("Invalid pendulum parameters.")

    rng = np.random.default_rng(seed)


    theta0 = rng.uniform(-np.pi / 2.0, np.pi / 2.0, n_neurons)
    thetadot0 = rng.uniform(-0.5, 0.5, n_neurons)
    y0 = np.concatenate([theta0, thetadot0])

    def rhs(t, y):
        theta = y[:n_neurons]
        thetadot = y[n_neurons:]


        dtheta = thetadot
        dthetadot = -(g / l) * np.sin(theta)


        for i in range(n_neurons):
            dthetadot[i] += coupling * np.sum(np.sin(theta - theta[i]))

        return np.concatenate([dtheta, dthetadot])

    t, y_history = rk4_integrate(rhs, (0.0, t_final), y0, n_steps)
    theta = y_history[:, :n_neurons]
    thetadot = y_history[:, n_neurons:]

    return t, theta, thetadot


def simulate_homeostatic_plasticity_pipeline(
    n_synapses: int = 5,
    t_final: float = 30.0,
) -> dict:
    rng = np.random.default_rng(130)

    results = []
    for i in range(n_synapses):
        w0 = rng.uniform(0.1, 0.9)
        m = rng.uniform(0.5, 2.0)
        b = rng.uniform(0.2, 1.0)
        k = rng.uniform(0.5, 2.0)
        F = rng.uniform(-0.1, 0.1)

        t, y, params = simulate_homeostatic_response(
            w0=w0, m=m, b=b, k=k, F_ext=F, t_final=t_final, n_steps=3000
        )
        results.append({
            "t": t,
            "w": y[:, 0],
            "v": y[:, 1],
            "params": params,
        })


    t_net, theta_net, thetadot_net = simulate_network_synchronization(
        n_neurons=8, t_final=t_final, n_steps=3000
    )

    return {
        "synapses": results,
        "network_t": t_net,
        "network_theta": theta_net,
        "network_thetadot": thetadot_net,
    }


if __name__ == "__main__":
    t, y, params = simulate_homeostatic_response()
    print(f"Damping regime: {params['regime']}")
    print(f"Natural frequency: {params['omega_n']:.4f}")
    print(f"Final weight: {y[-1, 0]:.6f}")

    theta, thetadot = pendulum_nonlinear_exact(np.linspace(0, 10, 100))
    print(f"Pendulum period: {compute_pendulum_period():.4f}")
