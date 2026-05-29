# -*- coding: utf-8 -*-
"""
monte_carlo_sampler.py
准蒙特卡洛采样与变分Monte Carlo

核心物理：
  在Laughlin态的变分能量计算中，需要计算多体期望值：
      ⟨O⟩ = ∫ d^{2N}r |Ψ_m(r)|² O(r) / ∫ d^{2N}r |Ψ_m(r)|²

  直接积分的维数灾难使得解析或确定性积分不可行，
  因此采用准蒙特卡洛（Quasi-Monte Carlo, QMC）方法，
  利用低差异序列（low-discrepancy sequences）替代伪随机数，
  获得 O((log N)^d / N) 的收敛速率，优于纯随机采样的 O(1/√N)。

  这里使用 Hammersley 序列，其 d 维点集定义为：
      x_n = (n/N, Φ_{p_1}(n), Φ_{p_2}(n), ..., Φ_{p_{d-1}}(n))
  其中 Φ_p(n) 为 n 在素数基 p 下的 radical inverse 函数：
      Φ_p(n) = Σ_{k=0}^∞ a_k(n) p^{-k-1}
  且 n = Σ_{k=0}^∞ a_k(n) p^k 为 n 的 p 进制展开。

本模块融合原项目：
  - 498_hammersley（Hammersley低差异序列）
"""
import numpy as np
from utils import safe_exp

# ============================================================================
# 1. Hammersley低差异序列（融合原项目 498_hammersley）
# ============================================================================

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
    """
    计算整数 n 在基 base 下的 radical inverse：
        Φ_base(n) = a_0/base + a_1/base² + a_2/base³ + ...
    其中 n = a_0 + a_1·base + a_2·base² + ...

    参数:
        n    : int, 非负整数
        base : int, 素数基

    返回:
        phi  : float, radical inverse 值
    """
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
    """
    生成 Hammersley 低差异序列。

    公式：
        对于第 i 个点 (i = i1, ..., i2)，第 j 个坐标为：
          x_{i,0} = i / N     (若 n_base 指定，则 x_{i,0} = i / n_base)
          x_{i,j} = Φ_{p_j}(i),   j = 1, ..., m-1
    其中 p_j 为第 j 个素数。

    参数:
        i1    : int, 起始索引
        i2    : int, 结束索引
        m     : int, 空间维数
        n_base: int or None, 第一坐标的分母（若None则用 N = i2-i1+1）

    返回:
        r     : ndarray, shape (m, N)，每列为一个点
    """
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


# ============================================================================
# 2. 将Hammersley序列映射到物理空间
# ============================================================================

def map_hammersley_to_disk(hammersley_points, radius):
    """
    将 Hammersley 2D 点映射到圆盘内。

    采用面积保持映射：
        r = R √u
        θ = 2π v
    其中 (u, v) 为 Hammersley 序列的坐标。

    参数:
        hammersley_points : ndarray, shape (2, N)
        radius            : float, 圆盘半径

    返回:
        z                 : ndarray, shape (N,), 复坐标
    """
    if hammersley_points.shape[0] != 2:
        raise ValueError("输入必须是二维点集")
    N = hammersley_points.shape[1]
    u = hammersley_points[0, :]
    v = hammersley_points[1, :]

    # 边界处理：确保 u ∈ (0, 1) 避免 r = 0
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    v = np.clip(v, 0.0, 1.0)

    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    z = r * np.exp(1j * theta)
    return z


def map_hammersley_to_annulus(hammersley_points, r_inner, r_outer):
    """
    将 Hammersley 2D 点映射到环形区域（annulus）。

    采用面积保持映射：
        r = √[r_inner² + (r_outer² - r_inner²) u]
        θ = 2π v
    """
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


# ============================================================================
# 3. 变分Monte Carlo能量估计
# ============================================================================

def variational_monte_carlo_energy(
    wavefunction_log_prob_fn,
    local_energy_fn,
    sampler_fn,
    n_samples=10000,
    thermalization=1000,
    skip_interval=10
):
    """
    使用Metropolis-Hastings变分蒙特卡洛计算局部能量的期望值。

    算法：
        1. 从初始构型 r_0 出发
        2. 提议新构型 r' = r + δ·RandomNormal
        3. 计算接受概率 A = min(1, |Ψ(r')|² / |Ψ(r)|²)
        4. 以概率 A 接受 r'
        5. 在热化后收集样本，计算 E = ⟨E_L⟩ = ⟨Ψ^{-1} H Ψ⟩

    参数:
        wavefunction_log_prob_fn : callable, 返回 ln|Ψ(r)|²
        local_energy_fn          : callable, 返回局部能量 E_L(r)
        sampler_fn               : callable, 返回初始构型
        n_samples                : int, 采样数
        thermalization           : int, 热化步数
        skip_interval            : int, 样本间隔（降低自相关）

    返回:
        energy_mean, energy_err, samples
    """
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
        # 提议新构型
        proposal = current + step_size * (np.random.randn(len(current)) + 1j * np.random.randn(len(current)))
        proposal_log_prob = wavefunction_log_prob_fn(proposal)

        # Metropolis 接受准则
        log_ratio = proposal_log_prob - current_log_prob
        if log_ratio >= 0 or np.random.rand() < np.exp(min(log_ratio, 0.0)):
            current = proposal
            current_log_prob = proposal_log_prob
            accepted += 1
        total += 1

        # 自适应调整步长
        if step > 0 and step % 100 == 0:
            acc_rate = accepted / total
            if acc_rate > 0.5:
                step_size *= 1.1
            elif acc_rate < 0.2:
                step_size *= 0.9
            step_size = np.clip(step_size, 0.01, 2.0)

        # 收集样本
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
    """
    准蒙特卡洛积分：
        I ≈ (V/N) Σ_i f(x_i)
    其中 {x_i} 为 Hammersley 序列。

    参数:
        f                : callable, 被积函数
        hammersley_points: ndarray, shape (d, N)
        domain_volume    : float, 积分区域体积

    返回:
        integral         : float, 积分估计值
    """
    N = hammersley_points.shape[1]
    if N == 0:
        return 0.0
    s = 0.0
    for i in range(N):
        s += f(hammersley_points[:, i])
    return domain_volume * s / N


# ============================================================================
# 4. 测试接口
# ============================================================================
def test_monte_carlo_sampler():
    """测试准蒙特卡洛采样模块。"""
    print("=" * 60)
    print("[monte_carlo_sampler.py] 准蒙特卡洛采样测试")
    print("=" * 60)

    # 测试 Hammersley 序列
    print("\n1. Hammersley序列生成测试:")
    r = hammersley_sequence(0, 99, 2)
    print(f"   生成100个2维Hammersley点，shape={r.shape}")
    print(f"   前5个点: {r[:, :5].T}")

    # 测试radical inverse
    print("\n2. Radical inverse测试:")
    for n in [0, 1, 2, 3, 4, 5]:
        phi2 = radical_inverse(n, 2)
        phi3 = radical_inverse(n, 3)
        print(f"   n={n}: Φ_2={phi2:.6f}, Φ_3={phi3:.6f}")

    # 测试圆盘映射
    print("\n3. 圆盘映射测试:")
    r_2d = hammersley_sequence(0, 999, 2)
    z = map_hammersley_to_disk(r_2d, radius=5.0)
    r_vals = np.abs(z)
    print(f"   1000个映射点的 |z| 均值: {np.mean(r_vals):.4f}")
    print(f"   |z| 最大值: {np.max(r_vals):.4f}, 最小值: {np.min(r_vals):.6f}")

    # 测试准蒙特卡洛积分：计算圆盘上的 ∫∫ (x² + y²) dxdy = πR⁴/2
    print("\n4. 准蒙特卡洛积分测试 (圆盘上 ∫∫ r² dA):")
    R = 2.0
    exact = 0.5 * np.pi * R ** 4
    for N in [100, 500, 1000, 5000]:
        pts = hammersley_sequence(0, N - 1, 2)
        z_pts = map_hammersley_to_disk(pts, R)
        area = np.pi * R ** R
        # 被积函数 f = |z|² = r²
        f_vals = np.abs(z_pts) ** 2
        integral = np.mean(f_vals) * np.pi * R * R
        err = abs(integral - exact)
        print(f"   N={N:5d}: 估计={integral:.6f}, 精确={exact:.6f}, 误差={err:.6e}")

    # 测试VMC框架（简单谐振子基态能量）
    print("\n5. 变分Monte Carlo测试（一维谐振子基态能量 = 0.5 ħω）:")
    def log_prob_ho(x):
        return -np.sum(np.abs(x) ** 2)  # |Ψ|² ~ exp(-x²)
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
