"""
sensor_noise.py
===============
Sensor noise models for swarm robot perception.

Incorporates:
  - gray_salt_and_pepper / gray_uniform_noise (from 581_image_noise)

Scientific role:
  Real swarm robots operate with imperfect sensors. This module models
  two canonical noise mechanisms:
    1. Salt-and-pepper noise: catastrophic sensor failures (outliers).
    2. Uniform noise: quantization and thermal noise.
  These are applied to the scalar field measurements that robots collect
  from the environment, affecting their local decision-making and the
  emergent consensus dynamics.
"""

import numpy as np


def salt_and_pepper_noise(measurements: np.ndarray, level: float = 0.05):
    """
    Add salt-and-pepper noise to scalar measurements.

    With probability `level/2` a measurement is set to the minimum
    (black, 0.0) and with probability `level/2` to the maximum (white, 1.0).

    Parameters
    ----------
    measurements : ndarray
        Input measurements in [0, 1].
    level : float
        Total corruption probability in [0, 1].

    Returns
    -------
    noisy : ndarray
        Corrupted measurements.
    """
    if not (0.0 <= level <= 1.0):
        raise ValueError("level must be in [0, 1].")
    measurements = np.asarray(measurements, dtype=float)
    noisy = measurements.copy()
    r = np.random.rand(*measurements.shape)
    noisy[r <= level * 0.5] = 0.0
    noisy[r >= 1.0 - level * 0.5] = 1.0
    return noisy


def uniform_noise(measurements: np.ndarray, level: float = 0.05):
    """
    Add uniform noise to scalar measurements.

    A fraction `level` of the entries are replaced by uniform random
    values in [0, 1], modeling quantization errors.

    Parameters
    ----------
    measurements : ndarray
        Input measurements.
    level : float
        Fraction of corrupted entries in [0, 1].

    Returns
    -------
    noisy : ndarray
        Corrupted measurements.
    """
    if not (0.0 <= level <= 1.0):
        raise ValueError("level must be in [0, 1].")
    measurements = np.asarray(measurements, dtype=float)
    noisy = measurements.copy()
    r = np.random.rand(*measurements.shape)
    mask = r <= level
    noisy[mask] = np.random.rand(int(np.count_nonzero(mask)))
    return noisy


def gaussian_sensor_noise(measurements: np.ndarray, sigma: float = 0.02):
    """
    Add Gaussian white noise to measurements.

    Models thermal noise in LIDAR / ultrasound ranging sensors:
        m_noisy = m + N(0, sigma^2)

    Parameters
    ----------
    measurements : ndarray
        Input measurements.
    sigma : float
        Standard deviation of noise.

    Returns
    -------
    noisy : ndarray
        Noisy measurements, clipped to [0, 1].
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    measurements = np.asarray(measurements, dtype=float)
    noisy = measurements + sigma * np.random.randn(*measurements.shape)
    return np.clip(noisy, 0.0, 1.0)


def apply_sensor_noise(measurements: np.ndarray, config: dict):
    """
    Apply composite sensor noise according to a configuration dict.

    Parameters
    ----------
    measurements : ndarray
        Clean measurements.
    config : dict
        Keys: 'salt_pepper_level', 'uniform_level', 'gaussian_sigma'.

    Returns
    -------
    noisy : ndarray
        Measurements after all noise stages.
    """
    noisy = np.asarray(measurements, dtype=float)
    if config.get("salt_pepper_level", 0.0) > 0:
        noisy = salt_and_pepper_noise(noisy, config["salt_pepper_level"])
    if config.get("uniform_level", 0.0) > 0:
        noisy = uniform_noise(noisy, config["uniform_level"])
    if config.get("gaussian_sigma", 0.0) > 0:
        noisy = gaussian_sensor_noise(noisy, config["gaussian_sigma"])
    return noisy
