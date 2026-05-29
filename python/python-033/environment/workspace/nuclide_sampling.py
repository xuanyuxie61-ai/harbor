"""
nuclide_sampling.py
基于种子项目 245_cvt_1d_nonuniform 和 456_gaussian_prime_spiral 的核素采样

在 r 过程核合成中，需要在核素图 (N,Z) 上选择代表性的核素进行网络计算。
CVT（Centroidal Voronoi Tessellation）提供了一种最优采样方法：
    每个生成点为其 Voronoi 单元的质心，最小化量化误差：
        F = Σ_i ∫_{V_i} ρ(x) ||x - g_i||² dx

对于 r 过程，密度函数 ρ 与 (n,γ) 反应路径概率相关，
在远离稳定谷的丰中子区域密度更高。

一维非均匀 CVT 算法（Lloyd 迭代）：
    1. 根据密度函数生成随机样本
    2. 将样本分配到最近的生成点（Voronoi 单元）
    3. 更新生成点为单元内样本的均值
    4. 重复直到收敛
"""

import numpy as np


def cvt_1d_nonuniform(n_generators, density_type='power', n_samples=50000,
                       n_steps=50, init_type='grid'):
    """
    计算一维非均匀 CVT。

    参数:
        n_generators : int, 生成点数量
        density_type : str, 密度函数类型
                       'uniform', 'sqrt', 'power', 'log', 'arctan', 'chebyshev'
        n_samples : int, 每步采样数
        n_steps : int, Lloyd 迭代步数
        init_type : str, 初始化方式 'random', 'grid', 'zeros'

    返回:
        generators : ndarray, 生成点坐标（在 [0,1] 区间）
    """
    # 初始化生成点
    if init_type == 'random':
        generators = np.sort(np.random.rand(n_generators))
    elif init_type == 'grid':
        generators = np.linspace(0.0, 1.0, n_generators)
    else:
        generators = np.zeros(n_generators)

    for step in range(n_steps):
        # 根据密度生成样本
        u = np.random.rand(n_samples)
        if density_type == 'uniform':
            samples = u
        elif density_type == 'sqrt':
            samples = u ** 2
        elif density_type == 'power':
            samples = u ** 0.5
        elif density_type == 'log':
            samples = np.log(1.0 + (np.e - 1.0) * u)
        elif density_type == 'arctan':
            samples = np.tan(np.pi / 4.0 * u)
            samples = np.clip(samples, 0.0, 1.0)
        elif density_type == 'chebyshev':
            samples = np.sin(np.pi / 2.0 * u) ** 2
        else:
            samples = u

        # 分配到最近的生成点
        # 向量化：计算每个样本到所有生成点的距离
        indices = np.argmin(np.abs(samples[:, None] - generators[None, :]), axis=1)

        # 更新生成点为单元质心
        new_generators = np.zeros(n_generators)
        for i in range(n_generators):
            cell_samples = samples[indices == i]
            if len(cell_samples) > 0:
                new_generators[i] = np.mean(cell_samples)
            else:
                # 空单元：保持原位或随机重置
                new_generators[i] = generators[i]

        # 排序并保持边界
        new_generators = np.sort(new_generators)
        new_generators[0] = max(new_generators[0], 0.0)
        new_generators[-1] = min(new_generators[-1], 1.0)

        generators = new_generators

    return generators


def sample_nuclide_mass_chain(a_min, a_max, n_nuclides,
                               density_profile='r_process_path'):
    """
    在质量数 A 轴上采样核素，密度与 r 过程路径相关。

    r 过程路径近似服从：
        P(A) ∝ exp(-(A - A_peak)² / (2σ²)) + 背景
    其中 A_peak ≈ 130 和 195 为两个 r 过程峰。

    参数:
        a_min, a_max : int, 质量数范围
        n_nuclides : int, 采样核素数
        density_profile : str, 密度轮廓

    返回:
        a_values : ndarray, 采样质量数
    """
    n_generators = n_nuclides
    generators = cvt_1d_nonuniform(n_generators, density_type='power',
                                    n_samples=20000, n_steps=30, init_type='grid')
    # 映射到 [a_min, a_max]
    a_values = a_min + generators * (a_max - a_min)
    a_values = np.round(a_values).astype(int)
    a_values = np.clip(a_values, a_min, a_max)
    a_values = np.unique(a_values)
    return a_values


def build_r_process_nuclide_set(a_values, beta_stability_offset=5):
    """
    为每个质量数 A 构建 r 过程核素集合 (Z,N,A)。

    r 过程核素位于稳定谷的丰中子侧，采用近似关系：
        Z ≈ A / (1.98 + 0.0158 A^{2/3})   (近似稳定线)
        N = A - Z
    r 过程偏移：N_r = N_stable + offset

    参数:
        a_values : ndarray, 质量数列表
        beta_stability_offset : int, 丰中子偏移

    返回:
        nuclides : list of tuple, [(Z,N,A), ...]
    """
    nuclides = []
    for A in a_values:
        if A <= 0:
            continue
        # 近似稳定线（Seeger 公式简化）
        Z_stable = int(A / (1.98 + 0.0158 * (A ** (2.0 / 3.0))))
        Z_stable = max(1, min(Z_stable, A - 1))
        # r 过程位于丰中子侧
        N_stable = A - Z_stable
        N_rp = N_stable + beta_stability_offset
        Z_rp = A - N_rp
        if Z_rp < 1:
            Z_rp = 1
            N_rp = A - 1
        nuclides.append((int(Z_rp), int(N_rp), int(A)))
    return nuclides


def test_nuclide_sampling():
    """自包含测试"""
    gens = cvt_1d_nonuniform(20, density_type='power', n_steps=20)
    print(f"[nuclide_sampling] CVT generators range: [{gens[0]:.4f}, {gens[-1]:.4f}]")

    a_vals = sample_nuclide_mass_chain(80, 240, 30)
    print(f"[nuclide_sampling] Sampled mass numbers: {a_vals[:10]} ...")

    nuclides = build_r_process_nuclide_set(a_vals, beta_stability_offset=8)
    print(f"[nuclide_sampling] Built {len(nuclides)} r-process nuclides")
    if nuclides:
        print(f"[nuclide_sampling] Example: Z={nuclides[0][0]}, N={nuclides[0][1]}, A={nuclides[0][2]}")


if __name__ == "__main__":
    test_nuclide_sampling()
