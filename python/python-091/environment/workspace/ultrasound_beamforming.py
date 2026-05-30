
import numpy as np
from typing import Tuple, Optional


SOUND_SPEED_TISSUE = 1540.0


def array_steering_vector(n_elements: int, element_spacing: float,
                          angle: float, frequency: float,
                          c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    k = 2.0 * np.pi * frequency / c
    n = np.arange(n_elements)
    phases = -k * n * element_spacing * np.sin(angle)
    return np.exp(1j * phases)


def hanning_window(n_elements: int) -> np.ndarray:
    n = np.arange(n_elements)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (n_elements - 1)))


def hamming_window(n_elements: int) -> np.ndarray:
    n = np.arange(n_elements)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (n_elements - 1))


def chebyshev_window(n_elements: int, sidelobe_level: float = -40.0) -> np.ndarray:


    beta = np.cosh(np.arccosh(10**(-sidelobe_level / 20.0)) / (n_elements - 1))
    
    n = np.arange(n_elements)
    x = np.cos(np.pi * n / (n_elements - 1))
    

    window = np.zeros(n_elements)
    for i in range(n_elements):
        if abs(x[i]) <= 1.0 / beta:
            window[i] = np.cos((n_elements - 1) * np.arccos(beta * x[i]))
        else:
            window[i] = np.cosh((n_elements - 1) * np.arccosh(beta * abs(x[i])))
    
    window = window / np.max(window)
    return window


def delay_and_sum_beamforming(channel_data: np.ndarray,
                               sampling_rate: float,
                               element_positions: np.ndarray,
                               focus_depths: np.ndarray,
                               steering_angle: float = 0.0,
                               window_type: str = 'hanning',
                               c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    n_elements, n_samples = channel_data.shape
    n_depths = len(focus_depths)
    dt = 1.0 / sampling_rate


    if window_type == 'hanning':
        weights = hanning_window(n_elements)
    elif window_type == 'hamming':
        weights = hamming_window(n_elements)
    elif window_type == 'chebyshev':
        weights = chebyshev_window(n_elements)
    else:
        weights = np.ones(n_elements)

    beamformed = np.zeros(n_depths, dtype=complex)
    time_axis = np.arange(n_samples) * dt

    for d_idx, z in enumerate(focus_depths):

        focus_x = z * np.sin(steering_angle)
        focus_z = z * np.cos(steering_angle)

        summed_signal = np.zeros(n_samples, dtype=complex)

        for elem_idx in range(n_elements):

            dx = focus_x - element_positions[elem_idx]
            dz = focus_z
            distance = np.sqrt(dx**2 + dz**2)


            delay = 2.0 * distance / c


            delay_samples = delay / dt
            int_delay = int(delay_samples)
            frac_delay = delay_samples - int_delay


            if int_delay >= n_samples - 1:
                continue


            shifted = np.zeros(n_samples, dtype=complex)
            for t_idx in range(n_samples):
                src_idx = t_idx + int_delay
                if src_idx < n_samples - 1:
                    shifted[t_idx] = ((1.0 - frac_delay) * channel_data[elem_idx, src_idx] +
                                      frac_delay * channel_data[elem_idx, src_idx + 1])
                elif src_idx < n_samples:
                    shifted[t_idx] = channel_data[elem_idx, src_idx]

            summed_signal += weights[elem_idx] * shifted


        time_idx = int(2.0 * z / (c * dt))
        time_idx = min(time_idx, n_samples - 1)
        beamformed[d_idx] = summed_signal[time_idx]

    return np.abs(beamformed)


def transmit_focus_delay(n_elements: int, element_spacing: float,
                         focus_depth: float, steering_angle: float = 0.0,
                         c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    x_elements = (np.arange(n_elements) - (n_elements - 1) / 2.0) * element_spacing
    focus_x = focus_depth * np.sin(steering_angle)
    focus_z = focus_depth * np.cos(steering_angle)

    distances = np.sqrt((x_elements - focus_x)**2 + focus_z**2)
    max_distance = np.max(distances)
    delays = (max_distance - distances) / c

    return delays


def beam_pattern(n_elements: int, element_spacing: float,
                 frequency: float, angles: np.ndarray,
                 window_type: str = 'hanning',
                 c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    wavelength = c / frequency
    k = 2.0 * np.pi / wavelength

    if window_type == 'hanning':
        weights = hanning_window(n_elements)
    elif window_type == 'hamming':
        weights = hamming_window(n_elements)
    elif window_type == 'chebyshev':
        weights = chebyshev_window(n_elements)
    else:
        weights = np.ones(n_elements)

    pattern = np.zeros(len(angles))
    n = np.arange(n_elements)

    for i, theta in enumerate(angles):
        phases = k * n * element_spacing * np.sin(theta)
        sv = np.exp(1j * phases)
        pattern[i] = np.abs(np.sum(weights * sv))


    pattern = pattern / np.max(pattern)
    pattern_db = 20.0 * np.log10(pattern + 1e-10)
    pattern_db = np.clip(pattern_db, -80.0, 0.0)

    return pattern_db


def simulate_array_response(n_elements: int = 64,
                            element_spacing: float = 0.3e-3,
                            frequency: float = 5e6,
                            sampling_rate: float = 40e6,
                            n_samples: int = 2048,
                            scatterer_depths: np.ndarray = None,
                            scatterer_amplitudes: np.ndarray = None,
                            c: float = SOUND_SPEED_TISSUE) -> Tuple[np.ndarray, np.ndarray]:
    element_positions = (np.arange(n_elements) - (n_elements - 1) / 2.0) * element_spacing
    dt = 1.0 / sampling_rate
    time_axis = np.arange(n_samples) * dt

    if scatterer_depths is None:
        scatterer_depths = np.array([0.02, 0.035, 0.05])
    if scatterer_amplitudes is None:
        scatterer_amplitudes = np.array([1.0, 0.7, 0.4])

    channel_data = np.zeros((n_elements, n_samples), dtype=complex)


    pulse_width = 2.0 / frequency
    envelope = np.exp(-(time_axis - pulse_width)**2 / (2.0 * (pulse_width / 4.0)**2))
    pulse = envelope * np.sin(2.0 * np.pi * frequency * time_axis)

    for elem_idx in range(n_elements):
        for scat_idx, (depth, amp) in enumerate(zip(scatterer_depths, scatterer_amplitudes)):

            dx = element_positions[elem_idx]
            distance = np.sqrt(dx**2 + depth**2)
            delay = 2.0 * distance / c


            delayed_pulse = np.zeros(n_samples)
            for t_idx, t in enumerate(time_axis):
                t_delayed = t - delay
                if t_delayed >= 0 and t_delayed < time_axis[-1]:

                    src_idx = int(t_delayed / dt)
                    frac = t_delayed / dt - src_idx
                    if src_idx < n_samples - 1:
                        delayed_pulse[t_idx] = ((1.0 - frac) * pulse[src_idx] +
                                                frac * pulse[src_idx + 1])


            attenuation = amp / (distance + 1e-6)
            channel_data[elem_idx] += attenuation * delayed_pulse


    noise_level = 0.05 * np.max(np.abs(channel_data))
    channel_data += noise_level * np.random.randn(n_elements, n_samples)

    return channel_data, element_positions
