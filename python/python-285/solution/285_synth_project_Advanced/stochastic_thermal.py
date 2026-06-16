"""
随机热涨落模块
===============
对应种子项目: 855_pdflib (概率分布函数库 → 热噪声采样)

物理背景:
    有限温度下, 多铁性材料的序参量受到热涨落影响.
    LGD 方程需加入随机项:
        ∂P/∂t = -L·(δF/δP) + ξ_P(t)

    其中 ξ 为高斯白噪声, 满足涨落-耗散定理:
        <ξ(t)> = 0
        <ξᵢ(r,t)·ξⱼ(r',t')> = 2k_BT·L·δᵢⱼ·δ(r-r')·δ(t-t')

    离散化后:
        ξᵢ(r,t) ~ N(0, σ²)
        σ² = 2·k_B·T·L / (dx·dy·dt)

    本模块还实现:
    - 多元正态分布采样 (关联噪声)
    - Gamma 分布 (用于非平衡热浴模型)
    - Beta 分布 (用于有界随机变量)
    - Cholesky 分解生成关联高斯场

核心公式:
    涨落-耗散定理 (Callen-Welton):
        S_ξ(ω) = 2k_BT · Re[Z(ω)]
    其中 Z(ω) 为广义阻抗.

    空间关联噪声 (非局域热浴):
        <ξ(r)ξ(r')> = σ²·C(|r-r'|)
    C(r) = exp(-r²/(2λ²))  [高斯关联]
"""

import numpy as np


class ThermalNoiseGenerator:
    """
    热噪声生成器, 基于涨落-耗散定理.

    提供多种分布的随机数生成:
    - 高斯噪声 (基本热涨落)
    - 多元正态 (空间关联噪声)
    - Gamma 分布 (非平衡热浴)
    - Beta 分布 (有界涨落)
    - Chi 分布 (能量涨落)
    - 指数分布 (激活事件间隔)
    """

    def __init__(self, nx, ny, dx, dy, dt, temperature, L_P, seed=None):
        """
        参数:
            nx, ny: 网格尺寸
            dx, dy: 网格间距 (m)
            dt: 时间步长 (s)
            temperature: 温度 (K)
            L_P: LGD 动力学系数 (m/(V·s))
            seed: 随机种子
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.T = temperature
        self.L_P = L_P
        self.rng = np.random.RandomState(seed)

        # 涨落-耗散定理: σ² = 2·k_B·T·L / (dx·dy·dt)
        from multiferroic_constants import K_BOLTZMANN
        self.sigma_sq = (2.0 * K_BOLTZMANN * temperature * L_P /
                         (dx * dy * dt))
        self.sigma = np.sqrt(max(self.sigma_sq, 0.0))

    # ============================================================
    # 高斯噪声 (基本热涨落)
    # ============================================================

    def gaussian_noise(self, shape=None):
        """
        生成高斯白噪声场.

        ξ(r) ~ N(0, σ²)
        σ² = 2k_BT·L / (dx·dy·dt)

        物理含义:
            每个网格点独立的高斯随机变量,
            方差由涨落-耗散定理确定.

        参数:
            shape: 输出形状, 默认 (nx, ny, 3)

        返回:
            noise: 噪声场
        """
        if shape is None:
            shape = (self.nx, self.ny, 3)
        return self.rng.normal(0.0, self.sigma, size=shape)

    # ============================================================
    # 多元正态分布 (空间关联噪声)
    # ============================================================

    def correlated_noise(self, correlation_length):
        """
        生成空间关联高斯噪声.

        关联函数: C(r) = exp(-r²/(2λ²))
        通过 Cholesky 分解生成.

        步骤:
            1. 构建关联矩阵 K[i,j] = C(|rᵢ-rⱼ|)
            2. Cholesky 分解: K = L·Lᵀ
            3. 噪声: ξ = L·z, z~N(0,σ²I)

        参数:
            correlation_length: 关联长度 λ (m)

        返回:
            noise: shape (nx*ny,) 的关联噪声向量
        """
        n_total = self.nx * self.ny
        lambda_sq = max(correlation_length ** 2, 1e-30)

        # 构建坐标
        x = np.linspace(0, self.dx * (self.nx - 1), self.nx)
        y = np.linspace(0, self.dy * (self.ny - 1), self.ny)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        # 计算距离矩阵 (仅上三角, 节省内存)
        # 对小规模系统使用完整矩阵
        K = np.zeros((n_total, n_total))
        for i in range(n_total):
            diff = coords - coords[i]
            r_sq = np.sum(diff ** 2, axis=1)
            K[i, :] = np.exp(-r_sq / (2.0 * lambda_sq))

        # 添加正则化
        K += 1e-10 * np.eye(n_total)

        # Cholesky 分解 (对应种子项目 855 中的 r8po_fa)
        try:
            L_chol = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # 若 Cholesky 失败, 使用特征值分解
            eigvals, eigvecs = np.linalg.eigh(K)
            eigvals = np.maximum(eigvals, 0.0)
            L_chol = eigvecs * np.sqrt(eigvals)[np.newaxis, :]

        # 生成噪声
        z = self.rng.normal(0.0, self.sigma, size=n_total)
        noise_flat = L_chol @ z

        return noise_flat.reshape(self.nx, self.ny)

    # ============================================================
    # 其他分布 (对应种子项目 855_pdflib)
    # ============================================================

    def gamma_noise(self, shape_param, scale_param, size=None):
        """
        Gamma 分布噪声.

        p(x) = x^{k-1} · exp(-x/θ) / (θ^k · Γ(k))

        物理应用: 非平衡热浴中的能量涨落.

        参数:
            shape_param: 形状参数 k
            scale_param: 尺度参数 θ
            size: 输出大小

        返回:
            samples: Gamma 分布样本
        """
        if size is None:
            size = (self.nx, self.ny)
        return self.rng.gamma(shape_param, scale_param, size=size)

    def beta_noise(self, a, b, size=None):
        """
        Beta 分布噪声 (有界随机变量).

        p(x) = x^{a-1}·(1-x)^{b-1} / B(a,b),  x ∈ [0,1]

        物理应用: 有界序参量涨落 (如极化分量归一化后).

        参数:
            a, b: Beta 分布参数
            size: 输出大小

        返回:
            samples: Beta 分布样本
        """
        if size is None:
            size = (self.nx, self.ny)
        return self.rng.beta(a, b, size=size)

    def chi_noise(self, dof, size=None):
        """
        Chi 分布噪声.

        p(x) = 2^{1-k/2} · x^{k-1} · exp(-x²/2) / Γ(k/2)

        物理应用: 能量涨落 (E ~ χ² 分布).

        参数:
            dof: 自由度 k
            size: 输出大小

        返回:
            samples: Chi 分布样本
        """
        if size is None:
            size = (self.nx, self.ny)
        return np.sqrt(self.rng.chisquare(dof, size=size))

    def exponential_noise(self, rate, size=None):
        """
        指数分布噪声.

        p(x) = λ·exp(-λx),  x ≥ 0

        物理应用: 激活事件 (畴壁跳跃) 的等待时间.
            τ ~ Exp(ν₀·exp(-ΔE/(k_BT)))

        参数:
            rate: 率参数 λ
            size: 输出大小

        返回:
            samples: 指数分布样本
        """
        if size is None:
            size = (self.nx, self.ny)
        return self.rng.exponential(1.0 / rate, size=size)

    def multinomial_sample(self, n_trials, probabilities):
        """
        多项式分布采样.

        P(X₁=x₁,...,Xₖ=xₖ) = n!/(x₁!...xₖ!) · p₁^x₁...pₖ^xₖ

        物理应用: 多畴态之间的随机跃迁.

        参数:
            n_trials: 试验次数 n
            probabilities: 各类别概率 [p₁,...,pₖ]

        返回:
            counts: 各类别计数 [x₁,...,xₖ]
        """
        return self.rng.multinomial(n_trials, probabilities)

    # ============================================================
    # 多元正态采样 (对应种子项目 855 中的 r8vec_multinormal_sample)
    # ============================================================

    def multivariate_normal_sample(self, mean, cov, size=1):
        """
        多元正态分布采样.

        x ~ N(μ, Σ)
        通过 Cholesky 分解: x = μ + L·z, z~N(0,I)

        物理应用: 多序参量的联合热涨落.

        参数:
            mean: 均值向量, shape (d,)
            cov: 协方差矩阵, shape (d, d)
            size: 采样数量

        返回:
            samples: shape (size, d)
        """
        d = len(mean)
        cov_reg = cov + 1e-12 * np.eye(d)

        try:
            L = np.linalg.cholesky(cov_reg)
            z = self.rng.normal(0.0, 1.0, size=(size, d))
            samples = mean + z @ L.T
        except np.linalg.LinAlgError:
            samples = self.rng.multivariate_normal(mean, cov_reg, size=size)

        return samples

    # ============================================================
    # 噪声强度自适应
    # ============================================================

    def update_temperature(self, new_T):
        """
        更新温度 (用于模拟退火或温度扫描).

        重新计算噪声标准差:
            σ = sqrt(2·k_B·T·L / (dx·dy·dt))
        """
        from multiferroic_constants import K_BOLTZMANN
        self.T = new_T
        self.sigma_sq = (2.0 * K_BOLTZMANN * new_T * self.L_P /
                         (self.dx * self.dy * self.dt))
        self.sigma = np.sqrt(max(self.sigma_sq, 0.0))

    def noise_energy(self, noise_field):
        """
        计算噪声场的能量 (用于监控).

        E_noise = ½ ∫∫ |ξ|² dx dy

        期望值:
            <E_noise> = ½ · n_pts · σ² · dx · dy
        """
        return 0.5 * np.sum(noise_field ** 2) * self.dx * self.dy
