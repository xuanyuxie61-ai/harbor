
import numpy as np


def salt_and_pepper_noise(measurements: np.ndarray, level: float = 0.05):
    if not (0.0 <= level <= 1.0):
        raise ValueError("level must be in [0, 1].")
    measurements = np.asarray(measurements, dtype=float)
    noisy = measurements.copy()
    r = np.random.rand(*measurements.shape)
    noisy[r <= level * 0.5] = 0.0
    noisy[r >= 1.0 - level * 0.5] = 1.0
    return noisy


def uniform_noise(measurements: np.ndarray, level: float = 0.05):
    if not (0.0 <= level <= 1.0):
        raise ValueError("level must be in [0, 1].")
    measurements = np.asarray(measurements, dtype=float)
    noisy = measurements.copy()
    r = np.random.rand(*measurements.shape)
    mask = r <= level
    noisy[mask] = np.random.rand(int(np.count_nonzero(mask)))
    return noisy


def gaussian_sensor_noise(measurements: np.ndarray, sigma: float = 0.02):
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    measurements = np.asarray(measurements, dtype=float)
    noisy = measurements + sigma * np.random.randn(*measurements.shape)
    return np.clip(noisy, 0.0, 1.0)


def apply_sensor_noise(measurements: np.ndarray, config: dict):
    noisy = np.asarray(measurements, dtype=float)
    if config.get("salt_pepper_level", 0.0) > 0:
        noisy = salt_and_pepper_noise(noisy, config["salt_pepper_level"])
    if config.get("uniform_level", 0.0) > 0:
        noisy = uniform_noise(noisy, config["uniform_level"])
    if config.get("gaussian_sigma", 0.0) > 0:
        noisy = gaussian_sensor_noise(noisy, config["gaussian_sigma"])
    return noisy
