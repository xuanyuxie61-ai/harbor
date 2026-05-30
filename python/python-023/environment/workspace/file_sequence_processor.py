#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def generate_filename_sequence(base_name, n_files):
    filenames = []
    
    for i in range(1, n_files + 1):

        suffix = f"_{i:03d}"
        

        if '.' in base_name:
            parts = base_name.rsplit('.', 1)
            filename = f"{parts[0]}{suffix}.{parts[1]}"
        else:
            filename = f"{base_name}{suffix}"
        
        filenames.append(filename)
    
    return filenames


def process_field_timeseries(omega_solutions, params, n_frames=20):
    n_modes = len(omega_solutions)
    if n_modes == 0:
        return {'n_frames': 0, 'min_amp': 0.0, 'max_amp': 0.0}
    

    Omega_e = params['Omega_e']
    t_values = np.linspace(0, 10.0 / Omega_e, n_frames)
    

    amplitudes = np.zeros((n_modes, n_frames))
    phases = np.zeros((n_modes, n_frames))
    
    for m in range(n_modes):
        k = omega_solutions[m, 0]
        omega = complex(omega_solutions[m, 1])
        omega_r = omega.real
        gamma = omega.imag
        
        for t_idx, t in enumerate(t_values):

            amp = np.exp(gamma * t) * (1.0 + 0.1 * np.sin(omega_r * t))
            

            amp = np.clip(amp, 0.0, 10.0)
            
            phase = omega_r * t + 0.05 * np.random.randn()
            
            amplitudes[m, t_idx] = amp
            phases[m, t_idx] = phase
    

    all_amps = amplitudes.flatten()
    min_amp = np.min(all_amps)
    max_amp = np.max(all_amps)
    mean_amp = np.mean(all_amps)
    rms_amp = np.sqrt(np.mean(all_amps**2))
    

    dt = t_values[1] - t_values[0]
    

    total_field = np.sum(amplitudes * np.cos(phases), axis=0)
    fft_field = np.fft.rfft(total_field)
    freqs = np.fft.rfftfreq(n_frames, dt)
    psd = np.abs(fft_field)**2 * dt / n_frames
    

    if len(freqs) > 1:
        dominant_freq = freqs[np.argmax(psd[1:]) + 1]
    else:
        dominant_freq = 0.0
    

    filenames = generate_filename_sequence("wave_field.dat", n_frames)
    
    stats = {
        'n_frames': n_frames,
        'n_modes': n_modes,
        'min_amp': min_amp,
        'max_amp': max_amp,
        'mean_amp': mean_amp,
        'rms_amp': rms_amp,
        'dominant_freq': dominant_freq,
        'filenames': filenames
    }
    
    return stats


def extract_grid_from_sequence(data_sequence, nx, ny, orientation=1):
    N = len(data_sequence)
    
    if N != nx * ny:

        if N < nx * ny:
            padded = np.zeros(nx * ny)
            padded[:N] = data_sequence
            data_sequence = padded
        else:
            data_sequence = data_sequence[:nx * ny]
    
    if orientation == 1:
        grid = data_sequence.reshape((nx, ny))
    else:
        grid = data_sequence.reshape((ny, nx)).T
    
    return grid


def compute_temporal_correlation(field_series, max_lag=None):
    n = len(field_series)
    if max_lag is None:
        max_lag = n // 4
    
    max_lag = min(max_lag, n - 1)
    

    field_centered = field_series - np.mean(field_series)
    norm = np.sum(field_centered**2)
    
    if norm < 1e-30:
        return np.ones(max_lag + 1)
    
    correlation = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            correlation[lag] = 1.0
        else:
            corr = np.sum(field_centered[:-lag] * field_centered[lag:])
            correlation[lag] = corr / norm
    
    return correlation
