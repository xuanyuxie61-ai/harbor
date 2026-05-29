"""
monte_carlo_sampler.py
蒙特卡洛采样与验证模块

融合种子项目:
- 303_disk01_positive_monte_carlo (单位正半圆盘采样)
- 1399_walker_sample (Walker别名方法离散概率采样)
- 195_coin_simulation (偏置硬币/随机采样)
- 024_asa005 (正态分布累积密度, 用于误差估计)

科学背景:
在FMM中, 蒙特卡洛方法用于:
1. 按电荷分布非均匀采样粒子位置
2. 验证FMM计算结果与直接求和的统计一致性
3. 通过随机采样估计高维积分 (如多极矩积分)

核心公式:
- Walker别名方法: 对离散概率分布X(1..N)实现O(1)期望时间采样
  预处理: 构造阈值向量Y和别名向量A, 使得 Y(i) + sum_{j: A(j)=i} (1-Y(j)) = N*X(i)
  采样: 均匀选i~U[1,N], 若rand < Y(i) 返回i, 否则返回A(i)

- 圆盘采样 ( rejection / polar ):
    极坐标法: r = sqrt(U), theta = 2*pi*V, 其中 U,V ~ U[0,1]
    对于正半圆盘: x = |x0|/||v||, y = |y0|/||v||, 再缩放 r

- 中心极限定理与误差估计:
    蒙特卡洛估计量误差 ~ sigma / sqrt(N)
    置信区间: [mu - z_{alpha/2}*sigma/sqrt(N), mu + z_{alpha/2}*sigma/sqrt(N)]
    其中 z_{alpha/2} 通过正态累积分布函数的反函数求得
"""

import numpy as np


def walker_build(prob):
    """
    构建Walker别名采样表 (融合1399_walker_sample)
    
    参数:
        prob: ndarray (N,), 概率分布 (非负, 和不必为1)
    
    返回:
        y: ndarray (N,), 阈值向量
        a: ndarray (N,), 别名向量
    """
    prob = np.asarray(prob, dtype=float)
    if prob.size == 0:
        raise ValueError("概率分布不能为空")
    if np.any(prob < 0):
        raise ValueError("概率必须非负")
    n = prob.size
    s = np.sum(prob)
    if s < 1e-15:
        prob = np.ones(n) / n
    else:
        prob = prob / s

    # 缩放概率
    y = prob * n
    a = np.arange(n)

    # 分离大于1和小于1的索引
    small = []
    large = []
    for i in range(n):
        if y[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        l = small.pop()
        g = large.pop()
        a[l] = g
        y[g] = y[g] - (1.0 - y[l])
        if y[g] < 1.0:
            small.append(g)
        else:
            large.append(g)

    # 数值稳定性处理
    for i in large:
        y[i] = 1.0
    for i in small:
        y[i] = 1.0

    return y, a


def walker_sampler(y, a):
    """
    使用Walker别名表采样一个索引
    
    参数:
        y: ndarray (N,), 阈值向量
        a: ndarray (N,), 别名向量
    
    返回:
        int: 采样索引 (0-based)
    """
    n = y.size
    i = np.random.randint(0, n)
    r = np.random.rand()
    if r < y[i]:
        return i
    else:
        return int(a[i])


def disk01_positive_sample(n):
    """
    在单位正半圆盘 (x>=0, y>=0, x^2+y^2<=1) 中均匀采样n个点
    (融合303_disk01_positive_monte_carlo)
    
    方法:
        1. 在单位圆边界上均匀取方向 (正半平面)
        2. 径向采样: r = sqrt(U), U~U[0,1]
    """
    if n <= 0:
        raise ValueError("n必须为正整数")
    p = np.random.normal(size=(n, 2))
    p = np.abs(p)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    p = p / norms
    r = np.sqrt(np.random.rand(n, 1))
    return r * p


def coin_biased(n, heads_prob):
    """
    生成n次偏置硬币抛掷结果 (融合195_coin_simulation)
    
    参数:
        n: int, 抛掷次数
        heads_prob: float, 正面概率 (0 <= heads_prob <= 1)
    
    返回:
        ndarray (n,), +1表示正面, -1表示反面
    """
    if n < 0:
        raise ValueError("n必须非负")
    heads_prob = float(np.clip(heads_prob, 0.0, 1.0))
    v = (np.random.rand(n) < heads_prob).astype(float)
    return 2.0 * v - 1.0


def alnorm(x, upper=True):
    """
    计算标准正态分布累积密度函数 (融合024_asa005)
    
    公式:
        Phi(x) = integral_{-inf}^{x} (1/sqrt(2*pi)) * exp(-t^2/2) dt
    
    使用Hill (1973) Algorithm AS 66的近似公式
    
    参数:
        x: float, 积分端点
        upper: bool, True则积分从x到+inf, False则从-inf到x
    
    返回:
        float, 概率值
    """
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r_const = 0.398942280385
    utzero = 18.66

    up = upper
    z = float(x)

    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and (not up or utzero < z):
        return 0.0 if up else 1.0

    y_val = 0.5 * z * z

    if z <= con:
        value = 0.5 - z * (p - q * y_val
                           / (y_val + a1 + b1
                              / (y_val + a2 + b2
                                 / (y_val + a3))))
    else:
        value = (r_const * np.exp(-y_val)
                 / (z + c1 + d1
                    / (z + c2 + d2
                       / (z + c3 + d3
                          / (z + c4 + d4
                             / (z + c5 + d5
                                / (z + c6)))))))

    if not up:
        value = 1.0 - value

    return value


def mc_estimate_integral(f_sampler, n_samples):
    """
    蒙特卡洛估计积分值
    
    公式:
        I ≈ (1/N) * sum_{i=1}^{N} f(x_i)
        sigma^2 = (1/(N-1)) * sum (f(x_i) - I)^2
        标准误差 = sigma / sqrt(N)
    
    参数:
        f_sampler: callable, 返回一个样本值
        n_samples: int, 样本数
    
    返回:
        (mean, std_err): 估计均值与标准误差
    """
    if n_samples <= 1:
        raise ValueError("n_samples至少为2")
    samples = np.array([f_sampler() for _ in range(n_samples)])
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    std_err = std / np.sqrt(n_samples)
    return mean, std_err


def monte_carlo_fmm_verification(particles, charges, fmm_potential, direct_potential,
                                  n_sample_pairs=None, confidence=0.95):
    """
    使用蒙特卡洛方法验证FMM计算结果
    
    随机采样若干粒子对, 比较FMM势能与直接求和势能的统计一致性
    
    公式:
        相对误差向量: e_i = |phi_fmm(i) - phi_direct(i)| / |phi_direct(i)|
        样本均值: mu_e = (1/M) sum e_i
        95%置信区间: mu_e ± z_{0.025} * sigma_e / sqrt(M)
    
    参数:
        particles: ndarray (N, 3)
        charges: ndarray (N,)
        fmm_potential: ndarray (N,), FMM计算的势能
        direct_potential: ndarray (N,), 直接求和势能
        n_sample_pairs: int, 采样对数 (默认 min(N, 1000))
        confidence: float, 置信水平
    
    返回:
        dict: 包含mean_error, std_error, confidence_interval
    """
    N = particles.shape[0]
    if n_sample_pairs is None:
        n_sample_pairs = min(N, 1000)
    n_sample_pairs = min(n_sample_pairs, N)

    idx = np.random.choice(N, size=n_sample_pairs, replace=False)
    rel_err = np.abs((fmm_potential[idx] - direct_potential[idx])
                     / (np.abs(direct_potential[idx]) + 1e-15))

    mean_err = np.mean(rel_err)
    std_err = np.std(rel_err, ddof=1) / np.sqrt(n_sample_pairs)

    # 置信区间: 用正态近似, z = Phi^{-1}((1+confidence)/2)
    # 这里用查找表近似逆正态CDF
    target = (1.0 + confidence) / 2.0
    # 二分查找求逆CDF
    z_low, z_high = 0.0, 5.0
    for _ in range(50):
        z_mid = (z_low + z_high) / 2.0
        if alnorm(z_mid, upper=False) < target:
            z_low = z_mid
        else:
            z_high = z_mid
    z_val = (z_low + z_high) / 2.0

    ci_low = max(0.0, mean_err - z_val * std_err)
    ci_high = mean_err + z_val * std_err

    return {
        "mean_relative_error": float(mean_err),
        "std_error": float(std_err),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "confidence_level": confidence,
        "z_score": float(z_val),
        "n_samples": n_sample_pairs
    }


def nonuniform_particle_sample(prob_density, n_samples, domain="sphere"):
    """
    根据非均匀概率密度采样粒子位置
    
    使用Walker别名方法对离散化概率进行采样, 再映射到空间坐标
    
    参数:
        prob_density: callable, 概率密度函数 f(x,y,z) >= 0
        n_samples: int, 采样数
        domain: str, "sphere"或"disk_positive"
    """
    if domain == "sphere":
        # 在球面上离散化方向, 用Walker采样
        n_bins_theta = 20
        n_bins_phi = 40
        theta_edges = np.linspace(0, np.pi, n_bins_theta + 1)
        phi_edges = np.linspace(0, 2 * np.pi, n_bins_phi + 1)
        probs = []
        centers = []
        for i in range(n_bins_theta):
            for j in range(n_bins_phi):
                t = (theta_edges[i] + theta_edges[i + 1]) / 2.0
                p = (phi_edges[j] + phi_edges[j + 1]) / 2.0
                st = np.sin(t)
                # 立体角权重
                domega = st * (theta_edges[i + 1] - theta_edges[i]) * (phi_edges[j + 1] - phi_edges[j])
                val = prob_density(np.array([np.sin(t) * np.cos(p),
                                              np.sin(t) * np.sin(p),
                                              np.cos(t)]))
                probs.append(max(0.0, val) * domega)
                centers.append((t, p))
        probs = np.array(probs)
        y, a = walker_build(probs)
        samples = []
        for _ in range(n_samples):
            idx = walker_sampler(y, a)
            t, p = centers[idx]
            # 在bin内均匀扰动
            dt = theta_edges[1] - theta_edges[0]
            dp = phi_edges[1] - phi_edges[0]
            t += (np.random.rand() - 0.5) * dt
            p += (np.random.rand() - 0.5) * dp
            t = np.clip(t, 0, np.pi)
            p = np.clip(p, 0, 2 * np.pi)
            samples.append(np.array([np.sin(t) * np.cos(p),
                                      np.sin(t) * np.sin(p),
                                      np.cos(t)]))
        return np.array(samples)
    elif domain == "disk_positive":
        pts = disk01_positive_sample(n_samples)
        # 扩展为3D, z=0
        return np.column_stack([pts, np.zeros(n_samples)])
    else:
        raise ValueError(f"未知domain: {domain}")
