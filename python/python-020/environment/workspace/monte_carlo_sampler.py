# -*- coding: utf-8 -*-
import numpy as np
from utils import safe_exp





_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541
], dtype=int)


def radical_inverse(n, base):
    if n < 0:
        raise ValueError("n 必须为非负整数")
    if base < 2:
        raise ValueError("base 必须 ≥ 2")

    result = 0.0
    base_inv = 1.0 / base
    factor = base_inv
    while n > 0:
        digit = n % base
        result += digit * factor
        n //= base
        factor *= base_inv
    return result


def hammersley_sequence(i1, i2, m, n_base=None):
    if i1 < 0 or i2 < 0:
        raise ValueError("索引必须为非负整数")
    if m < 1:
        raise ValueError("维数 m 必须 ≥ 1")
    if m - 1 > len(_PRIMES):
        raise ValueError(f"维数过大，当前仅支持前 {len(_PRIMES)} 个素数")

    N = abs(i2 - i1) + 1
    if n_base is None:
        n_base = N
    if n_base < 1:
        raise ValueError("n_base 必须 ≥ 1")

    r = np.zeros((m, N), dtype=float)

    step = 1 if i1 <= i2 else -1
    col = 0
    for i in range(i1, i2 + step, step):
        r[0, col] = (i % n_base) / n_base if n_base > 0 else 0.0
        for j in range(1, m):
            r[j, col] = radical_inverse(i, _PRIMES[j - 1])
        col += 1

    return r






def map_hammersley_to_disk(hammersley_points, radius):
    if hammersley_points.shape[0] != 2:
        raise ValueError("输入必须是二维点集")
    N = hammersley_points.shape[1]
    u = hammersley_points[0, :]
    v = hammersley_points[1, :]


    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    v = np.clip(v, 0.0, 1.0)

    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    z = r * np.exp(1j * theta)
    return z


def map_hammersley_to_annulus(hammersley_points, r_inner, r_outer):
    if hammersley_points.shape[0] != 2:
        raise ValueError("输入必须是二维点集")
    u = hammersley_points[0, :]
    v = hammersley_points[1, :]
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    v = np.clip(v, 0.0, 1.0)

    r_sq = r_inner ** 2 + (r_outer ** 2 - r_inner ** 2) * u
    r = np.sqrt(r_sq)
    theta = 2.0 * np.pi * v
    z = r * np.exp(1j * theta)
    return z






def variational_monte_carlo_energy(
    wavefunction_log_prob_fn,
    local_energy_fn,
    sampler_fn,
    n_samples=10000,
    thermalization=1000,
    skip_interval=10
):
    np.random.seed(42)
    current = sampler_fn()
    current_log_prob = wavefunction_log_prob_fn(current)

    energies = []
    samples = []
    accepted = 0
    total = 0

    step_size = 0.3
    n_total = thermalization + n_samples * skip_interval

    for step in range(n_total):

        proposal = current + step_size * (np.random.randn(len(current)) + 1j * np.random.randn(len(current)))
        proposal_log_prob = wavefunction_log_prob_fn(proposal)


        log_ratio = proposal_log_prob - current_log_prob
        if log_ratio >= 0 or np.random.rand() < np.exp(min(log_ratio, 0.0)):
            current = proposal
            current_log_prob = proposal_log_prob
            accepted += 1
        total += 1


        if step > 0 and step % 100 == 0:
            acc_rate = accepted / total
            if acc_rate > 0.5:
                step_size *= 1.1
            elif acc_rate < 0.2:
                step_size *= 0.9
            step_size = np.clip(step_size, 0.01, 2.0)


        if step >= thermalization and (step - thermalization) % skip_interval == 0:
            e_local = local_energy_fn(current)
            energies.append(e_local)
            samples.append(current.copy())

    if len(energies) < 10:
        raise RuntimeError("采样数不足，无法可靠估计能量")

    energies = np.array(energies)
    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    energy_err = energy_std / np.sqrt(len(energies))

    return energy_mean, energy_err, np.array(samples)


def qmc_integration(f, hammersley_points, domain_volume):
    N = hammersley_points.shape[1]
    if N == 0:
        return 0.0
    s = 0.0
    for i in range(N):
        s += f(hammersley_points[:, i])
    return domain_volume * s / N





def test_monte_carlo_sampler():
    print("=" * 60)
    print("[monte_carlo_sampler.py] 准蒙特卡洛采样测试")
    print("=" * 60)


    print("\n1. Hammersley序列生成测试:")
    r = hammersley_sequence(0, 99, 2)
    print(f"   生成100个2维Hammersley点，shape={r.shape}")
    print(f"   前5个点: {r[:, :5].T}")


    print("\n2. Radical inverse测试:")
    for n in [0, 1, 2, 3, 4, 5]:
        phi2 = radical_inverse(n, 2)
        phi3 = radical_inverse(n, 3)
        print(f"   n={n}: Φ_2={phi2:.6f}, Φ_3={phi3:.6f}")


    print("\n3. 圆盘映射测试:")
    r_2d = hammersley_sequence(0, 999, 2)
    z = map_hammersley_to_disk(r_2d, radius=5.0)
    r_vals = np.abs(z)
    print(f"   1000个映射点的 |z| 均值: {np.mean(r_vals):.4f}")
    print(f"   |z| 最大值: {np.max(r_vals):.4f}, 最小值: {np.min(r_vals):.6f}")


    print("\n4. 准蒙特卡洛积分测试 (圆盘上 ∫∫ r² dA):")
    R = 2.0
    exact = 0.5 * np.pi * R ** 4
    for N in [100, 500, 1000, 5000]:
        pts = hammersley_sequence(0, N - 1, 2)
        z_pts = map_hammersley_to_disk(pts, R)
        area = np.pi * R ** R

        f_vals = np.abs(z_pts) ** 2
        integral = np.mean(f_vals) * np.pi * R * R
        err = abs(integral - exact)
        print(f"   N={N:5d}: 估计={integral:.6f}, 精确={exact:.6f}, 误差={err:.6e}")


    print("\n5. 变分Monte Carlo测试（一维谐振子基态能量 = 0.5 ħω）:")
    def log_prob_ho(x):
        return -np.sum(np.abs(x) ** 2)
    def local_energy_ho(x):
        return 0.5 - 0.5 * np.sum(np.abs(x) ** 2) + 0.5 * np.sum(x.real ** 2)
    def sampler_ho():
        return np.random.randn(1) + 1j * 0.0

    E_mean, E_err, _ = variational_monte_carlo_energy(
        log_prob_ho, local_energy_ho, sampler_ho,
        n_samples=2000, thermalization=500, skip_interval=5
    )
    print(f"   估计基态能量: {E_mean:.4f} ± {E_err:.4f} (理论值 0.5)")

    print("\n[monte_carlo_sampler.py] 测试完成。\n")


if __name__ == "__main__":
    test_monte_carlo_sampler()
