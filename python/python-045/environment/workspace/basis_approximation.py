#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def binomial_coefficient(n, k):
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - k + i) / i
    return result


def bernstein_basis(n, k, t):
    t = np.asarray(t, dtype=np.float64)
    if np.any(t < 0.0) or np.any(t > 1.0):

        t = np.clip(t, 0.0, 1.0)
    if k < 0 or k > n:
        return np.zeros_like(t)

    log_c = np.log(binomial_coefficient(n, k))

    result = np.zeros_like(t)
    mask = (t > 0.0) & (t < 1.0)
    result[~mask & (t == 0.0) & (k == 0)] = 1.0
    result[~mask & (t == 1.0) & (k == n)] = 1.0
    if np.any(mask):
        t_m = t[mask]
        log_val = log_c + k * np.log(t_m) + (n - k) * np.log(1.0 - t_m)
        result[mask] = np.exp(log_val)
    return result if result.shape != () else float(result)


def bernstein_basis_recursive(n, t):
    t = np.asarray(t, dtype=np.float64)
    t = np.clip(t, 0.0, 1.0)

    if n == 0:
        return np.ones((1,) + t.shape)


    be = np.zeros((n + 1,) + t.shape, dtype=np.float64)
    be[0] = 1.0 - t
    be[1] = t

    for j in range(2, n + 1):
        be[j] = t * be[j - 1]
        for k in range(j - 1, 0, -1):
            be[k] = t * be[k - 1] + (1.0 - t) * be[k]
        be[0] = (1.0 - t) * be[0]

    return be


class BernsteinResistivityProfile:

    def __init__(self, coefficients, z_max=10000.0):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.n = len(self.coefficients) - 1
        self.z_max = float(z_max)
        if self.z_max <= 0.0:
            raise ValueError("z_max 必须为正")
        if np.any(self.coefficients <= 0.0):
            raise ValueError("Bernstein 系数必须为正以保证电阻率为正")

    def evaluate(self, z):
        z = np.asarray(z, dtype=np.float64)
        if np.any(z < 0.0) or np.any(z > self.z_max):
            z = np.clip(z, 0.0, self.z_max)
        t = z / self.z_max
        basis = bernstein_basis_recursive(self.n, t)
        rho = np.dot(self.coefficients, basis)
        return rho

    def derivative(self, z):
        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, 0.0, self.z_max)
        t = z / self.z_max
        if self.n == 0:
            return np.zeros_like(z)
        dc = np.diff(self.coefficients)
        basis = bernstein_basis_recursive(self.n - 1, t)
        drhodz = (self.n / self.z_max) * np.dot(dc, basis)
        return drhodz

    def roughness(self):
        if self.n < 2:
            return 0.0
        d2c = np.diff(self.coefficients, 2)
        return np.sum(d2c ** 2)

    def to_layer_model(self, n_layers):
        if n_layers < 2:
            raise ValueError("层数至少为 2")
        z_interfaces = np.linspace(0.0, self.z_max, n_layers + 1)
        thicknesses = np.diff(z_interfaces)
        z_centers = (z_interfaces[:-1] + z_interfaces[1:]) / 2.0
        resistivities = self.evaluate(z_centers)

        resistivities = np.maximum(resistivities, 1e-6)
        return resistivities, thicknesses


class Bernstein2DResistivity:

    def __init__(self, coefficients, y_max=50000.0, z_max=10000.0):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        if self.coefficients.ndim != 2:
            raise ValueError("系数必须是二维矩阵")
        self.ny, self.nz = self.coefficients.shape
        self.ny -= 1
        self.nz -= 1
        self.y_max = float(y_max)
        self.z_max = float(z_max)
        if np.any(self.coefficients <= 0.0):
            raise ValueError("系数必须为正")

    def evaluate(self, y, z):
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        y = np.clip(y, 0.0, self.y_max)
        z = np.clip(z, 0.0, self.z_max)
        ty = y / self.y_max
        tz = z / self.z_max
        By = bernstein_basis_recursive(self.ny, ty)
        Bz = bernstein_basis_recursive(self.nz, tz)

        if y.ndim == 0 and z.ndim == 0:
            rho = np.dot(By.T, np.dot(self.coefficients, Bz))
        else:

            rho = np.einsum('i...,ij,j...->...', By, self.coefficients, Bz)
        return rho

    def roughness_yz(self):
        ry = 0.0
        rz = 0.0
        if self.ny >= 2:
            d2y = np.diff(self.coefficients, 2, axis=0)
            ry = np.sum(d2y ** 2)
        if self.nz >= 2:
            d2z = np.diff(self.coefficients, 2, axis=1)
            rz = np.sum(d2z ** 2)
        return ry + rz


if __name__ == "__main__":

    t = np.linspace(0, 1, 101)
    basis = bernstein_basis_recursive(3, t)
    print("Bernstein 基函数和:", np.sum(basis, axis=0)[:5])

    coeffs = np.array([100.0, 80.0, 50.0, 30.0, 20.0])
    profile = BernsteinResistivityProfile(coeffs, z_max=5000.0)
    z_test = np.array([0.0, 1250.0, 2500.0, 3750.0, 5000.0])
    rho_test = profile.evaluate(z_test)
    drho_test = profile.derivative(z_test)
    print(f"ρ(z) = {rho_test}")
    print(f"dρ/dz = {drho_test}")

    rho_layers, thick = profile.to_layer_model(4)
    print(f"离散化电阻率: {rho_layers}")
    print(f"厚度: {thick}")
