#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from source_field import sincn_fun
from special_functions import alnorm


class PropagationLoss:

    def __init__(self, U, r_grid, z_grid, seafloor_depth):
        self.U = np.asarray(U, dtype=np.complex128)
        self.r_grid = np.asarray(r_grid, dtype=np.float64)
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.seafloor_depth = np.asarray(seafloor_depth, dtype=np.float64)
        self.nr = len(r_grid)
        self.nz = len(z_grid)

    def coherent_tl(self):
        intensity = np.abs(self.U) ** 2
        intensity = np.maximum(intensity, 1e-30)
        return -10.0 * np.log10(intensity)

    def incoherent_tl_frequency_average(self, U_list):
        avg_intensity = np.zeros_like(self.U, dtype=np.float64)
        for Uf in U_list:
            avg_intensity += np.abs(Uf) ** 2
        avg_intensity /= len(U_list)
        avg_intensity = np.maximum(avg_intensity, 1e-30)
        return -10.0 * np.log10(avg_intensity)

    def depth_averaged_tl(self, r_index=None):
        if r_index is None:
            r_index = slice(None)
        U_slice = self.U[r_index, :]
        z = self.z_grid
        tl_dasl = np.zeros(U_slice.shape[0], dtype=np.float64)
        for i in range(U_slice.shape[0]):
            intensity = np.abs(U_slice[i, :]) ** 2
            H = max(z[-1] - z[0], 1e-6)
            avg_int = np.trapezoid(intensity, z) / H
            avg_int = max(avg_int, 1e-30)
            tl_dasl[i] = -10.0 * np.log10(avg_int)
        return tl_dasl

    def tl_at_receiver(self, z_receiver):

        j = np.argmin(np.abs(self.z_grid - z_receiver))
        intensity = np.abs(self.U[:, j]) ** 2
        intensity = np.maximum(intensity, 1e-30)
        return -10.0 * np.log10(intensity)

    def vla_beamform(self, z_vla, weights, theta_look, k0):
        z_vla = np.asarray(z_vla, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        output = np.zeros(self.nr, dtype=np.complex128)
        for i in range(self.nr):
            for j, zj in enumerate(z_vla):

                u_zj = np.interp(zj, self.z_grid, np.real(self.U[i, :])) \
                       + 1j * np.interp(zj, self.z_grid, np.imag(self.U[i, :]))
                output[i] += weights[j] * u_zj * np.exp(-1j * k0 * zj * np.sin(theta_look))
        return output

    def convergence_zone_analysis(self, tl_field, threshold_db=5.0):
        zones = []
        for i in range(1, self.nr - 1):
            for j in range(1, self.nz - 1):
                center = tl_field[i, j]
                neighbors = [
                    tl_field[i - 1, j], tl_field[i + 1, j],
                    tl_field[i, j - 1], tl_field[i, j + 1]
                ]
                if all(center < n - threshold_db for n in neighbors):
                    zones.append((i, j, self.r_grid[i], self.z_grid[j], center))
        return zones

    def shadow_zone_detection(self, tl_field, tl_threshold=80.0):
        shadow_mask = tl_field > tl_threshold

        shadows = []
        for i in range(self.nr):
            mask = shadow_mask[i, :]
            if np.any(mask):
                j_min = np.argmax(mask)
                j_max = len(mask) - 1 - np.argmax(mask[::-1])
                shadows.append((i, j_min, j_max))
        return shadows


class ReceiverArray:

    def __init__(self, r_positions, z_positions):
        self.r = np.asarray(r_positions, dtype=np.float64)
        self.z = np.asarray(z_positions, dtype=np.float64)
        self.n_receivers = len(r_positions)

    def dolph_chebyshev_weights(self, sidelobe_db=-30):
        n = self.n_receivers
        if n <= 1:
            return np.ones(n)

        sigma = n / (2.0 * np.sqrt(np.log(10.0) * abs(sidelobe_db) / 20.0))
        idx = np.arange(n) - (n - 1) / 2.0
        w = np.exp(-0.5 * (idx / sigma) ** 2)
        return w / np.sum(w)

    def sinc_weights(self, mainlobe_width):
        if self.n_receivers <= 1:
            return np.ones(1)
        dz = np.mean(np.diff(self.z)) if len(self.z) > 1 else 1.0
        idx = np.arange(self.n_receivers) - (self.n_receivers - 1) / 2.0
        w = sincn_fun(idx * dz / mainlobe_width)
        return w / np.sum(w)

    def extract_signals(self, U, r_grid, z_grid):
        signals = np.zeros(self.n_receivers, dtype=np.complex128)
        for k in range(self.n_receivers):

            i = np.argmin(np.abs(r_grid - self.r[k]))
            j = np.argmin(np.abs(z_grid - self.z[k]))
            signals[k] = U[i, j]
        return signals


class MultipathStatistics:

    def __init__(self, tl_obj, c_water=1500.0):
        self.tl_obj = tl_obj
        self.c = c_water

    def delay_spread_estimate(self, scattering_volume_mean_distance):
        return scattering_volume_mean_distance / self.c

    def coherence_bandwidth(self, delay_spread):
        if delay_spread < 1e-12:
            return np.inf
        return 1.0 / (2.0 * np.pi * delay_spread)

    def fading_depth_statistics(self, r_index, z_index, window_r=5):
        i0 = max(0, r_index - window_r)
        i1 = min(self.tl_obj.nr, r_index + window_r + 1)
        tl_local = self.tl_obj.coherent_tl()[i0:i1, z_index]
        return float(np.mean(tl_local)), float(np.std(tl_local))
