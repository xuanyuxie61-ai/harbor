
import numpy as np
from typing import Tuple






def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _next_prime_ge(n: int) -> int:
    while not _is_prime(n):
        n += 1
    return n


def _binomial_table_mod_p(max_n: int, p: int) -> np.ndarray:
    table = np.zeros((max_n, max_n), dtype=np.int64)
    table[0, 0] = 1 % p
    
    for i in range(1, max_n):
        table[i, 0] = 1 % p
        for j in range(1, i + 1):
            table[i, j] = (table[i - 1, j] + table[i - 1, j - 1]) % p
    
    return table


def faure_sequence_1d(key: int, p: int, max_digits: int = 20) -> float:
    quasi = 0.0
    p_power = 1.0 / p
    
    k = key
    for _ in range(max_digits):
        digit = k % p
        quasi += digit * p_power
        k //= p
        p_power /= p
        if k == 0:
            break
    
    return quasi


def faure_generate(
    dim_num: int,
    n: int,
    skip: int = 0
) -> np.ndarray:
    p = _next_prime_ge(dim_num)
    max_digits = 20
    

    binom_table = _binomial_table_mod_p(max_digits + 1, p)
    
    r = np.zeros((dim_num, n), dtype=np.float64)
    
    for point_idx in range(n):
        key = skip + point_idx
        

        y = np.zeros(max_digits, dtype=np.int64)
        k = key
        pos = 0
        while k > 0 and pos < max_digits:
            y[pos] = k % p
            k //= p
            pos += 1
        

        r[0, point_idx] = faure_sequence_1d(key, p, max_digits)
        

        for dim in range(1, dim_num):
            y_new = np.zeros(max_digits, dtype=np.int64)
            for j in range(max_digits):
                for i in range(j, max_digits):
                    if i < binom_table.shape[0] and j < binom_table.shape[1]:
                        y_new[j] = (y_new[j] + y[i] * binom_table[i, j]) % p
            
            quasi = 0.0
            p_power = 1.0 / p
            for j in range(max_digits):
                quasi += y_new[j] * p_power
                p_power /= p
            
            r[dim, point_idx] = quasi
    
    return r






def generate_inhomogeneous_poisson_spikes(
    rate_func: callable,
    t_start: float,
    t_end: float,
    dt: float = 0.001,
    max_rate: float = None,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    

    if max_rate is None:
        t_samples = np.linspace(t_start, t_end, 1000)
        rate_samples = np.array([rate_func(t) for t in t_samples])
        max_rate = np.max(rate_samples) * 1.2
        max_rate = max(max_rate, 1e-6)
    

    n_steps = int((t_end - t_start) / dt)
    candidate_spikes = []
    
    for i in range(n_steps):
        t = t_start + i * dt

        if np.random.random() < max_rate * dt:
            candidate_spikes.append(t + np.random.random() * dt)
    

    spike_times = []
    for t in candidate_spikes:
        rate_t = rate_func(t)
        if rate_t > max_rate:
            rate_t = max_rate
        if np.random.random() < rate_t / max_rate:
            spike_times.append(t)
    
    return np.array(spike_times, dtype=np.float64)


def simulate_rgc_spike_train(
    bipolar_response: np.ndarray,
    time_array: np.ndarray,
    baseline_rate: float = 5.0,
    gain: float = 20.0,
    refractory_period: float = 0.002,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    
    dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.001
    N = len(bipolar_response)
    

    rate = baseline_rate + gain * np.maximum(0.0, bipolar_response)
    rate = np.maximum(rate, 0.0)
    
    max_rate = np.max(rate) * 1.2
    max_rate = max(max_rate, 1e-6)
    
    spike_times = []
    last_spike = -refractory_period - 1.0
    
    for i in range(N):
        t = time_array[i]
        if t - last_spike < refractory_period:
            continue
        

        p_spike = rate[i] * dt
        p_spike = min(p_spike, 1.0)
        
        if np.random.random() < p_spike:
            spike_times.append(t)
            last_spike = t
    
    return np.array(spike_times, dtype=np.float64)






def sample_neural_parameter_space(
    n_samples: int,
    param_ranges: dict,
    skip: int = 100
) -> dict:
    dim_num = len(param_ranges)
    param_names = list(param_ranges.keys())
    

    faure_points = faure_generate(dim_num, n_samples, skip)
    
    samples = {}
    for i, name in enumerate(param_names):
        pmin, pmax = param_ranges[name]
        samples[name] = pmin + faure_points[i, :] * (pmax - pmin)
    
    return samples
