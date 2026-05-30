# -*- coding: utf-8 -*-

import numpy as np


def glycolysis_rhs(t, y, a=0.08, b=0.6):
    u, v = y[0], y[1]
    dudt = -u + a * v + u ** 2 * v
    dvdt = b - a * v - u ** 2 * v
    return np.array([dudt, dvdt], dtype=float)


def pendulum_nonlinear_rhs(t, y, g=9.81, l=1.0):
    theta, omega = y[0], y[1]
    dtheta = omega
    domega = -(g / l) * np.sin(theta)
    return np.array([dtheta, domega], dtype=float)


def pendulum_exact_solution(t, theta0=0.5, omega0=0.0, g=9.81, l=1.0):
    k0 = np.sin(theta0 / 2.0)
    omega_freq = np.sqrt(g / l)
    ep = 4.0 * g / l
    e0 = omega0 ** 2 + ep * k0 ** 2
    k = np.sqrt(e0 / ep)


    if k < 1e-6:
        theta = theta0 * np.cos(omega_freq * t)
        omega = -theta0 * omega_freq * np.sin(omega_freq * t)
        return theta, omega


    chi = 1.0 / (k + 1e-15)
    sn_val = np.tanh(omega_freq * t * k)
    cn_val = 1.0 / np.cosh(omega_freq * t * k)

    theta = 2.0 * np.sign(cn_val) * np.arcsin(np.clip(np.abs(sn_val), 0, 1))
    omega = np.sign(omega0) * np.sqrt(e0) * cn_val

    return theta, omega


def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y_new


def adams_bashforth_3_step(f, t, y, dt, history, *args):
    if len(history) < 3:

        return y + dt * f(t, y, *args)

    fn = history[-1]
    fn1 = history[-2]
    fn2 = history[-3]
    y_new = y + (dt / 12.0) * (23.0 * fn - 16.0 * fn1 + 5.0 * fn2)
    return y_new


def fractional_step_ns_2d(u, v, p, dt, dx, dy, nu, forcing_u, forcing_v):
    nx, ny = u.shape


    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dx)

        result[0, :] = (f[1, :] - f[0, :]) / dx
        result[-1, :] = (f[-1, :] - f[-2, :]) / dx
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dy)
        result[:, 0] = (f[:, 1] - f[:, 0]) / dy
        result[:, -1] = (f[:, -1] - f[:, -2]) / dy
        return result

    def laplacian(f):
        result = np.zeros_like(f)
        result[1:-1, 1:-1] = (
            (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx ** 2
            + (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy ** 2
        )
        return result


    conv_u = u * ddx(u) + v * ddy(u)
    conv_v = u * ddx(v) + v * ddy(v)

    u_star = u + dt * (-conv_u + nu * laplacian(u) + forcing_u)
    v_star = v + dt * (-conv_v + nu * laplacian(v) + forcing_v)


    div_u_star = ddx(u_star) + ddy(v_star)


    p_corr = np.zeros_like(p)
    for _ in range(50):
        p_new = np.zeros_like(p_corr)
        p_new[1:-1, 1:-1] = 0.25 * (
            p_corr[2:, 1:-1] + p_corr[:-2, 1:-1]
            + p_corr[1:-1, 2:] + p_corr[1:-1, :-2]
            - dx * dy * div_u_star[1:-1, 1:-1] / dt
        )
        p_corr = p_new


    dpdx = ddx(p_corr)
    dpdy = ddy(p_corr)

    u_new = u_star - dt * dpdx
    v_new = v_star - dt * dpdy
    p_new = p + p_corr

    return u_new, v_new, p_new


def fractional_step_ns_3d(u, v, w, p, dt, dx, dy, dz, nu,
                          forcing_u, forcing_v, forcing_w):
    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
        result[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
        result[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        result[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
        result[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
        return result

    def ddz(f):
        result = np.zeros_like(f)
        result[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
        result[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
        result[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
        return result

    def laplacian(f):
        result = np.zeros_like(f)
        result[1:-1, 1:-1, 1:-1] = (
            (f[2:, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / dx ** 2
            + (f[1:-1, 2:, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / dy ** 2
            + (f[1:-1, 1:-1, 2:] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / dz ** 2
        )
        return result










    raise NotImplementedError("Hole 2: Fractional step NS 3D time marching not implemented")
