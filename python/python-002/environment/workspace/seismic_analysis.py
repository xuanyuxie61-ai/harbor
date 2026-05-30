# -*- coding: utf-8 -*-

import numpy as np
from numerical_utils import cooley_tukey_fft, inverse_fft
from typing import Tuple, Optional


class SeismicAnalysis:

    def __init__(self, n_modes: int = 50):
        self.n_modes = n_modes

    @staticmethod
    def acoustic_cutoff_frequency(radius: np.ndarray, sound_speed: np.ndarray) -> float:
        c_s = np.asarray(sound_speed, dtype=np.float64)
        R = np.max(radius) if len(radius) > 0 else 7e10
        return float(np.mean(c_s)) / R if R > 0 else 1e-6

    @staticmethod
    def large_frequency_separation(radius: np.ndarray, sound_speed: np.ndarray) -> float:
        r = np.asarray(radius, dtype=np.float64)
        cs = np.asarray(sound_speed, dtype=np.float64)
        cs = np.maximum(cs, 1e3)
        integrand = 1.0 / cs

        integral = np.trapz(integrand, r)
        if integral <= 0:
            return 100.0
        dnu = 1.0e6 / (2.0 * integral)
        return float(dnu)

    @staticmethod
    def small_frequency_separation(radius: np.ndarray, density: np.ndarray,
                                   sound_speed: np.ndarray) -> float:
        dnu = SeismicAnalysis.large_frequency_separation(radius, sound_speed)

        return dnu / 6.0

    def compute_p_mode_frequencies(self, n_max: int, l_max: int,
                                   dnu: float, epsilon: float = 1.5) -> np.ndarray:
        freqs = []
        for l in range(l_max + 1):
            for n in range(n_max + 1):
                nu = dnu * (n + l / 2.0 + epsilon)
                freqs.append((n, l, nu))
        return np.array(freqs, dtype=[('n', int), ('l', int), ('nu', float)])

    def compute_g_mode_frequencies(self, n_g: int, l: int,
                                   N_brunt: float, R: float) -> np.ndarray:
        freqs = []
        for n in range(1, n_g + 1):
            nu = (n * np.pi / (l + 1)) * N_brunt / (2.0 * np.pi) * 1e6
            freqs.append((n, l, nu))
        return np.array(freqs, dtype=[('n', int), ('l', int), ('nu', float)])

    def frequency_spectrum_fft(self, time_series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        ts = np.asarray(time_series, dtype=np.float64)
        n = len(ts)
        if n < 4:
            return np.array([]), np.array([])


        ts = ts - np.mean(ts)

        window = np.hanning(n)
        ts_win = ts * window


        spectrum = cooley_tukey_fft(ts_win)
        power = np.abs(spectrum) ** 2


        freqs = np.fft.fftfreq(n, d=dt) * 1e6

        pos_mask = freqs >= 0
        return freqs[pos_mask], power[pos_mask]

    def echelle_diagram_data(self, freqs: np.ndarray, dnu: float) -> Tuple[np.ndarray, np.ndarray]:
        nu = np.asarray(freqs, dtype=np.float64)
        x = np.mod(nu, dnu)
        return x, nu

    def mode_inertia(self, radius: np.ndarray, density: np.ndarray,
                     eigenfunction: np.ndarray) -> float:
        r = np.asarray(radius, dtype=np.float64)
        rho = np.asarray(density, dtype=np.float64)
        xi = np.asarray(eigenfunction, dtype=np.float64)

        dr = np.diff(r)
        dr = np.append(dr, dr[-1])
        dm = 4.0 * np.pi * r ** 2 * rho * dr
        E = np.sum(xi ** 2 * dm)
        M = np.sum(dm)
        return float(E / M) if M > 0 else 0.0
