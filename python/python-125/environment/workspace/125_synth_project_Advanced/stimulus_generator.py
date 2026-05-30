
import numpy as np
from typing import Tuple






def binomial_coefficient(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    
    return result


def comb_lexicographic(n: int, p: int, L: int) -> np.ndarray:
    if L < 1 or L > binomial_coefficient(n, p):
        raise ValueError(f"L={L} out of range [1, C({n},{p})]")
    
    c = np.zeros(p + 1, dtype=np.int64)
    remaining = L
    
    for i in range(1, p + 1):
        c[i] = c[i - 1] + 1
        while True:
            bc = binomial_coefficient(n - c[i], p - i)
            if bc < remaining:
                remaining -= bc
                c[i] += 1
            else:
                break
    
    return c[1:]


def explore_synaptic_combinations(
    n_synapses: int,
    subset_size: int,
    n_combinations: int
) -> np.ndarray:
    total = binomial_coefficient(n_synapses, subset_size)
    n_combinations = min(n_combinations, total)
    

    indices = np.linspace(1, total, n_combinations, dtype=np.int64)
    
    combinations = np.zeros((n_combinations, subset_size), dtype=np.int64)
    for i, L in enumerate(indices):
        combinations[i] = comb_lexicographic(n_synapses, subset_size, int(L))
    
    return combinations






def sinusoidal_grating(
    nx: int,
    ny: int,
    spatial_freq: float,
    orientation: float,
    contrast: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    

    X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
    
    stimulus = contrast * np.sin(2.0 * np.pi * spatial_freq * X_rot + phase)
    

    stimulus = (stimulus + 1.0) / 2.0
    
    return stimulus


def gaussian_blob(
    nx: int,
    ny: int,
    sigma_x: float,
    sigma_y: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    amplitude: float = 1.0
) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    
    stimulus = amplitude * np.exp(
        -((X - center_x) ** 2 / (2.0 * sigma_x ** 2) +
          (Y - center_y) ** 2 / (2.0 * sigma_y ** 2))
    )
    
    return stimulus


def white_noise_stimulus(
    nx: int,
    ny: int,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    return np.random.random((ny, nx))


def drifting_grating(
    nx: int,
    ny: int,
    n_frames: int,
    spatial_freq: float,
    temporal_freq: float,
    orientation: float,
    dt: float = 0.01,
    contrast: float = 1.0
) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    
    X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
    
    stimulus_seq = np.zeros((n_frames, ny, nx), dtype=np.float64)
    for frame in range(n_frames):
        t = frame * dt
        phase = -2.0 * np.pi * temporal_freq * t
        stim = contrast * np.sin(2.0 * np.pi * spatial_freq * X_rot + phase)
        stimulus_seq[frame] = (stim + 1.0) / 2.0
    
    return stimulus_seq






def safe_math_eval(expr_str: str, x: float) -> float:
    safe_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }
    
    try:

        expr = expr_str.lower().replace('^', '**')
        result = eval(expr, {"__builtins__": {}}, {**safe_dict, 'x': x})
        return float(result)
    except Exception:
        return 0.0


def evaluate_tuning_function(
    func_str: str,
    x_values: np.ndarray
) -> np.ndarray:
    return np.array([safe_math_eval(func_str, x) for x in x_values], dtype=np.float64)
