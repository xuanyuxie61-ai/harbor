"""
hammersley_phase_space.py — Hammersley 准随机序列与 tt̄ 相空间采样
=================================================================
本模块实现 Hammersley 低差异序列用于顶夸克对产生
相空间的准蒙特卡罗 (QMC) 采样。

数学基础:
  Hammersley 序列在 [0,1]^d 中生成 N 个点, 其星差异:
    D*_N ≤ C (ln N)^(d-1) / N
  远优于随机蒙特卡罗的 O(N^{-1/2})。

  定义: 第 i 个点 (i=0,...,N-1) 在 d 维:
    第 1 维: r_i^(1) = i/N
    第 j 维 (j≥2): r_i^(j) = φ_{p_j}(i) ( radical inverse in base p_j)
  其中 radical inverse:
    φ_p(i) = Σ_k a_k p^{-(k+1)}
    i = Σ_k a_k p^k (p 进制展开)

  在 tt̄ 相空间中, 我们需要映射 [0,1]^6 → 四动量空间:
    - 2 个变量: 部分子动量分数 x₁, x₂
    - 2 个变量: 散射角 cos θ*, φ* (在 tt̄ 质心系)
    - 2 个变量: 衰变角度 (每个顶夸克)

  重要性采样: 将均匀分布映射到物理分布
    w_i = f(x_i) / g(x_i)
    其中 g 是采样分布, f 是目标分布

  Koksma-Hlawka 不等式:
    |Q_N(f) - I(f)| ≤ V(f) D*_N
    其中 V(f) 是 f 的 Hardy-Krause 有界变差

映射种子项目:
  - 498_hammersley: Hammersley 序列生成器
  - 566_hypersphere_monte_carlo: 球面均匀采样
"""

import numpy as np


def _radical_inverse(base, index):
    """
    计算 radical inverse 函数 φ_p(i)。

    将整数 i 在 base p 下展开: i = Σ a_k p^k
    然后反射小数点: φ_p(i) = Σ a_k p^{-(k+1)}

    例如: φ_2(6) = φ_2(110_2) = 0.011_2 = 1/4 + 1/8 = 0.375

    参数:
        base: 素数基底 p
        index: 非负整数 i

    返回:
        φ_p(i) ∈ [0, 1)
    """
    if index == 0:
        return 0.0

    result = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    n = index

    while n > 0:
        digit = n % base
        result += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base

    return result


def _first_n_primes(n):
    """返回前 n 个素数。"""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    if n <= len(primes):
        return primes[:n]

    # 生成更多素数 (简单筛法)
    candidate = primes[-1] + 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 2
    return primes[:n]


def hammersley_sequence(n_points, n_dim, skip=0):
    """
    生成 Hammersley 准随机点集。

    N 个点在 d 维 [0,1]^d 中:
      H_i = (i/N, φ_{p_1}(i), φ_{p_2}(i), ..., φ_{p_{d-1}}(i))

    其中 p_j 是第 j 个素数。

    星差异界:
      D*_N(H) ≤ C_d (ln N)^{d-1} / N
      对于 d=6: D*_N ~ O((ln N)^5 / N)
      N=10000 时 D* ~ 10^{-3}, 而 MC 为 ~10^{-2}

    参数:
        n_points: 点数 N
        n_dim: 维度 d
        skip: 跳过前 skip 个点 (避免原点)

    返回:
        points: (N, d) 数组, 每行是一个 [0,1]^d 中的点
    """
    if n_points <= 0:
        return np.zeros((0, n_dim))

    primes = _first_n_primes(max(n_dim - 1, 1))
    points = np.zeros((n_points, n_dim))

    for i in range(n_points):
        idx = i + skip

        # 第 1 维: 均匀序列
        points[i, 0] = (idx + 0.5) / n_points

        # 第 j 维 (j≥2): radical inverse in base p_{j-1}
        for j in range(1, n_dim):
            points[i, j] = _radical_inverse(primes[j - 1], idx + 1)

    return points


def hammersley_to_ttbar_phase_space(hammersley_points, m_top, sqrt_s=13000.0):
    """
    将 Hammersley 点映射到 tt̄ 相空间。

    从 [0,1]^6 映射到物理相空间:
      u₁ → x₁ (部分子动量分数, 映射到 [τ_min, 1])
      u₂ → x₂ = τ/x₁ (约束: x₂ ∈ [τ_min, 1])
      u₃ → cos θ* ∈ [-1, 1] (散射极角)
      u₄ → φ* ∈ [0, 2π] (散射方位角)
      u₅ → cos θ_d1 ∈ [-1, 1] (顶夸克衰变角)
      u₆ → φ_d1 ∈ [0, 2π] (衰变方位角)

    相空间体积元:
      dΦ_2 = (β/(32π²)) dcosθ* dφ* (两体相空间)
      dΦ_full = dΦ_2 × dΦ_decay1 × dΦ_decay2

    参数:
        hammersley_points: (N, 6) Hammersley 点
        m_top: 顶夸克质量 (GeV)
        sqrt_s: 质心系能量 (GeV)

    返回:
        phase_points: 物理相空间点字典
        weights: 相空间权重 (包含 Jacobian)
    """
    N = hammersley_points.shape[0]
    tau_min = 4.0 * m_top**2 / sqrt_s**2

    phase_points = {
        'x1': np.zeros(N),
        'x2': np.zeros(N),
        'cos_theta_star': np.zeros(N),
        'phi_star': np.zeros(N),
        'cos_theta_decay': np.zeros(N),
        'phi_decay': np.zeros(N),
    }
    weights = np.zeros(N)

    beta_threshold = np.sqrt(max(1.0 - 4.0 * m_top**2 / (sqrt_s**2), 0.0))

    valid_count = 0
    for i in range(N):
        u = hammersley_points[i]

        # x₁: 对数映射, 集中在阈值区域
        x1 = tau_min * (1.0 / tau_min) ** u[0]

        # x₂: 由 τ = x₁x₂ 约束
        tau = tau_min + (1.0 - tau_min) * u[0] * u[1]
        x2 = tau / x1 if x1 > 0 else 0.0

        if x2 <= 0.0 or x2 >= 1.0 or x1 <= 0.0 or x1 >= 1.0:
            weights[i] = 0.0
            continue

        # cos θ*: 均匀映射
        cos_theta = 2.0 * u[2] - 1.0
        phi = 2.0 * PI * u[3]

        # 衰变角度
        cos_theta_d = 2.0 * u[4] - 1.0
        phi_d = 2.0 * PI * u[5]

        # 质心系能量
        M_tt = np.sqrt(x1 * x2) * sqrt_s
        if M_tt < 2.0 * m_top:
            weights[i] = 0.0
            continue

        beta = np.sqrt(max(1.0 - 4.0 * m_top**2 / M_tt**2, 0.0))

        # 相空间 Jacobian
        # dΦ_2 = β/(32π²) dcosθ dφ
        jacobian_2body = beta / (32.0 * PI**2)

        # 部分子通量: 1/(2M²) × dx₁dx₂
        flux_jacobian = 1.0 / (2.0 * M_tt**2) * tau

        # 总权重
        weights[i] = jacobian_2body * flux_jacobian * 4.0 * PI  # 对方位角积分

        phase_points['x1'][valid_count] = x1
        phase_points['x2'][valid_count] = x2
        phase_points['cos_theta_star'][valid_count] = cos_theta
        phase_points['phi_star'][valid_count] = phi
        phase_points['cos_theta_decay'][valid_count] = cos_theta_d
        phase_points['phi_decay'][valid_count] = phi_d

        valid_count += 1

    # 截断到有效点
    for key in phase_points:
        phase_points[key] = phase_points[key][:valid_count]
    weights = weights[:valid_count]

    return phase_points, weights


def star_discrepancy(points):
    """
    计算点集的星差异 D*_N (品质因数)。

    D*_N = sup_{B} |A(B;N)/N - V(B)|
    其中 sup 取遍所有 anchored box B = [0, x₁]×...×[0, x_d]。

    近似计算 (一维边际):
      D*_N ≈ max_j max_i |i/N - x_(j,i)|
      其中 x_(j,i) 是第 j 维排序后的值。

    参数:
        points: (N, d) 数组

    返回:
        星差异估计值
    """
    N, d = points.shape
    max_discrepancy = 0.0

    for j in range(d):
        sorted_x = np.sort(points[:, j])
        for i in range(N):
            disc1 = abs((i + 1.0) / N - sorted_x[i])
            disc2 = abs(i / N - sorted_x[i])
            max_discrepancy = max(max_discrepancy, disc1, disc2)

    return max_discrepancy


def qmc_integration(func, n_dim, n_points, m_top=172.5):
    """
    使用 Hammersley QMC 进行多维积分。

    ∫_{[0,1]^d} f(u) du ≈ (1/N) Σ_{i=1}^N f(u_i)
    误差界: |Q_N - I| ≤ V(f) × D*_N (Koksma-Hlawka)

    参数:
        func: 被积函数 f: [0,1]^d → R
        n_dim: 维度
        n_points: 采样点数
        m_top: 顶夸克质量

    返回:
        (积分值, 误差估计)
    """
    points = hammersley_sequence(n_points, n_dim)
    values = np.array([func(points[i]) for i in range(n_points)])

    integral = np.mean(values)

    # 误差估计 (基于差异)
    D_star = star_discrepancy(points)
    # 变差的粗略估计 (使用样本方差)
    var_f = np.var(values)
    error_est = np.sqrt(var_f) * D_star * np.sqrt(n_points)

    return integral, error_est


# 本地常量
PI = np.pi
