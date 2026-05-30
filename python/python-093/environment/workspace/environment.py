#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class OceanEnvironment:

    def __init__(self,
                 c0=1500.0,
                 z_axis=1000.0,
                 B=1000.0,
                 epsilon=0.0057,
                 rho0=1024.0,
                 kappa_T=4.6e-10,
                 g=9.80665,
                 P0=1.01325e5,
                 seabed_type="clay",
                 seabed_cp=1700.0,
                 seabed_cs=800.0,
                 seabed_rho=1900.0,
                 seabed_loss=0.5,
                 depth_max=4000.0,
                 frequency=100.0):
        self.c0 = float(c0)
        self.z_axis = float(z_axis)
        self.B = float(B)
        self.epsilon = float(epsilon)
        self.rho0 = float(rho0)
        self.kappa_T = float(kappa_T)
        self.g = float(g)
        self.P0 = float(P0)
        self.seabed_type = seabed_type
        self.seabed_cp = float(seabed_cp)
        self.seabed_cs = float(seabed_cs)
        self.seabed_rho = float(seabed_rho)
        self.seabed_loss = float(seabed_loss)
        self.depth_max = float(depth_max)
        self.frequency = float(frequency)
        self.omega = 2.0 * np.pi * self.frequency
        self.k0 = self.omega / self.c0


        self.bathymetry_params = {
            'H0': depth_max,
            'H1': 0.0,
            'beta': 1.0,
            'r0': 0.0,
            'L': 1.0,
            'H2': 500.0,
            'r_c': 20000.0,
            'sigma_r': 5000.0,
        }

    def sound_speed(self, z):
        z = np.asarray(z, dtype=np.float64)
        eta = 2.0 * (z - self.z_axis) / self.B
        c = self.c0 * (1.0 + self.epsilon * (eta + np.exp(-eta) - 1.0))

        c_min = 1400.0
        return np.maximum(c, c_min)

    def absorption_db_per_km(self, f_khz=None):
        if f_khz is None:
            f_khz = self.frequency / 1000.0
        f = float(f_khz)
        alpha = (0.11 * f * f / (1.0 + f * f)
                 + 44.0 * f * f / (4100.0 + f * f)
                 + 2.75e-4 * f * f
                 + 0.0033)
        return alpha

    def absorption_np_per_m(self, f_khz=None):
        alpha_db = self.absorption_db_per_km(f_khz)
        return alpha_db / 8685.889638

    def wavenumber(self, z):
        z = np.asarray(z, dtype=np.float64)
        c = self.sound_speed(z)
        alpha = self.absorption_np_per_m()
        k = self.omega / c + 1j * alpha
        return k

    def refractive_index(self, z):
        z = np.asarray(z, dtype=np.float64)
        return self.c0 / self.sound_speed(z)

    def refractive_index_squared_deviation(self, z):
        n = self.refractive_index(z)
        return n * n - 1.0

    def density(self, z):
        z = np.asarray(z, dtype=np.float64)
        return self.rho0 * (1.0 + self.kappa_T * self.rho0 * self.g * z)

    def impedance(self, z):
        return self.density(z) * self.sound_speed(z)

    def bathymetry(self, r):
        r = np.asarray(r, dtype=np.float64)
        p = self.bathymetry_params
        h = p['H0'] + p['H1'] * np.tanh(p['beta'] * (r - p['r0']) / p['L'])
        h += p['H2'] * np.exp(-((r - p['r_c']) / p['sigma_r']) ** 2)

        h = np.clip(h, 100.0, self.depth_max * 1.5)
        return h

    def seabed_reflection_coefficient(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        cw = self.c0
        cb = self.seabed_cp
        rw = self.rho0
        rb = self.seabed_rho
        sin_theta = np.sin(theta)

        sin_thetab = (cw / cb) * sin_theta

        cos_thetab = np.sqrt(np.maximum(0.0, 1.0 - sin_thetab ** 2))
        cos_theta = np.cos(theta)
        num = rb * cb * cos_theta - rw * cw * cos_thetab
        den = rb * cb * cos_theta + rw * cw * cos_thetab
        R = safe_divide(num, den, fill_value=-1.0)

        loss_factor = np.exp(-2.0 * self.seabed_loss * sin_theta)
        return R * loss_factor


def safe_divide(a, b, fill_value=0.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.full_like(a, fill_value, dtype=np.float64)
    mask = np.abs(b) > np.finfo(np.float64).eps * 100
    result[mask] = a[mask] / b[mask]
    return result
