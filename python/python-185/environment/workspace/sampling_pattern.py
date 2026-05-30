
import numpy as np
from typing import Tuple, List


def is_prime(n: int) -> bool:
    if not isinstance(n, int):
        raise TypeError("is_prime(): 输入必须是整数")
    if n < 0:
        return False
    if n == 0 or n == 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    bound = int(np.sqrt(n)) + 1
    for i in range(3, bound, 2):
        if n % i == 0:
            return False
    return True


def generate_primes(count: int) -> List[int]:
    if count <= 0:
        return []
    primes = []
    candidate = 2
    while len(primes) < count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def prime_sampling_indices(signal_length: int, num_samples: int, prime_index: int = 5) -> np.ndarray:
    if signal_length <= 0:
        raise ValueError("signal_length 必须为正整数")
    if num_samples <= 0 or num_samples > signal_length:
        raise ValueError("num_samples 必须在 [1, signal_length] 范围内")

    primes = generate_primes(prime_index + 1)
    p = primes[prime_index]


    while np.gcd(p, signal_length) != 1 and prime_index < len(primes) - 1:
        prime_index += 1
        p = primes[prime_index]

    indices = np.mod(np.arange(num_samples) * p, signal_length)

    _, unique_idx = np.unique(indices, return_index=True)
    indices = np.sort(indices[np.sort(unique_idx)])


    if len(indices) < num_samples:
        remaining = np.setdiff1d(np.arange(signal_length), indices)
        extra = np.random.choice(remaining, num_samples - len(indices), replace=False)
        indices = np.sort(np.concatenate([indices, extra]))

    return indices.astype(int)


def circle_kspace_sampling(image_shape: Tuple[int, int], num_radial_lines: int,
                           samples_per_line: int, max_radius: float = None) -> np.ndarray:
    if len(image_shape) != 2:
        raise ValueError("image_shape 必须是二元组 (H, W)")
    H, W = image_shape
    if H <= 0 or W <= 0:
        raise ValueError("图像尺寸必须为正")
    if num_radial_lines <= 0 or samples_per_line <= 0:
        raise ValueError("径向线数量和每线采样点数必须为正")

    if max_radius is None:
        max_radius = min(H, W) / 2.0


    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    coords = []
    for j in range(num_radial_lines):
        theta = j * golden_angle

        radii = np.linspace(0.0, max_radius, samples_per_line)

        if samples_per_line > 1:
            radii[0] = 1e-6
        for r in radii:
            kx = r * np.cos(theta)
            ky = r * np.sin(theta)
            coords.append([kx, ky])

    return np.array(coords, dtype=float)


def build_incoherent_mask(image_shape: Tuple[int, int], sampling_ratio: float) -> np.ndarray:
    if not (0.0 < sampling_ratio <= 1.0):
        raise ValueError("sampling_ratio 必须在 (0, 1] 范围内")

    H, W = image_shape
    mask = np.zeros((H, W), dtype=bool)


    low_freq_ratio = 0.15
    cy, cx = H // 2, W // 2
    low_r = int(min(H, W) * low_freq_ratio / 2)
    y_grid, x_grid = np.ogrid[:H, :W]
    center_mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= low_r ** 2
    mask[center_mask] = True


    num_radial = max(8, int(np.sqrt(H * W * sampling_ratio) / 4))
    samples_per_radial = max(4, int(min(H, W) * sampling_ratio * 2))
    coords = circle_kspace_sampling(image_shape, num_radial, samples_per_radial)


    for kx, ky in coords:
        ix = int(round(kx)) + cx
        iy = int(round(ky)) + cy
        if 0 <= ix < W and 0 <= iy < H:
            mask[iy, ix] = True


    total_needed = int(H * W * sampling_ratio)
    current = np.count_nonzero(mask)
    if current < total_needed:
        remaining = np.column_stack(np.where(~mask))
        if len(remaining) > 0:
            step = max(1, len(remaining) // (total_needed - current))

            primes = generate_primes(20)
            p = primes[7 % len(primes)]
            idx = np.mod(np.arange(0, len(remaining), step) * p, len(remaining))
            idx = np.unique(idx)[:total_needed - current]
            mask[remaining[idx, 0], remaining[idx, 1]] = True

    return mask
