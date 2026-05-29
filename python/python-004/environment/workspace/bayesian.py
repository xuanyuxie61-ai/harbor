"""
bayesian.py
引力波参数估计的贝叶斯推断模块。

融合种子项目:
- 699_log_normal_truncated_ab: 截断对数正态分布 → 黑洞质量先验采样
- 1144_square_felippa_rule: 方形区域高斯求积 → 参数空间后验积分

核心公式:
1. 贝叶斯定理:
   p(θ|d) = p(d|θ) p(θ) / p(d)
   
   其中:
     θ = (m1, m2, a1, a2, D_L, ι, φ, ψ, t_c) 为引力波参数
     d 为观测数据
     p(d|θ) 为似然函数
     p(θ) 为先验分布
     p(d) = ∫ p(d|θ) p(θ) dθ 为证据

2. 高斯似然函数 (频域):
   ln p(d|θ) = -1/2 ⟨d - h(θ) | d - h(θ)⟩ + const
   
   其中内积:
     ⟨a|b⟩ = 4 Re ∫_{f_min}^{f_max} ã*(f) b̃(f) / S_n(f) df

3. 对数正态先验 (黑洞质量):
   p(m) = (1/(m σ √(2π))) * exp(-(ln m - μ)^2 / (2σ^2)) / Z
   
   其中 Z 为截断归一化常数:
   Z = Φ((ln b - μ)/σ) - Φ((ln a - μ)/σ)
   a, b 为质量的下限和上限。

4. 方形区域高斯求积 (Felippa):
   ∫_{[a1,b1]×[a2,b2]} f(x,y) dx dy ≈ Σ_{i,j} w_i w_j f(x_i, y_j)
   
   用于在二维参数子空间（如质量平面）上计算边缘后验。
"""

import numpy as np
from scipy.special import erf


# ---------------------------------------------------------------------------
# 截断对数正态分布 (源自 699_log_normal_truncated_ab)
# ---------------------------------------------------------------------------

def log_normal_pdf(x, mu, sigma):
    """
    对数正态分布概率密度函数。
    
    公式:
        p(x) = 1/(x σ √(2π)) * exp(-(ln x - μ)^2 / (2σ^2))
    """
    if sigma <= 0:
        raise ValueError("sigma 必须为正")
    if x <= 0:
        return 0.0
    z = (np.log(x) - mu) / sigma
    return np.exp(-0.5 * z**2) / (x * sigma * np.sqrt(2.0 * np.pi))


def log_normal_cdf(x, mu, sigma):
    """
    对数正态分布累积分布函数。
    
    公式:
        Φ(x) = 0.5 * [1 + erf((ln x - μ)/(σ√2))]
    """
    if sigma <= 0:
        raise ValueError("sigma 必须为正")
    if x <= 0:
        return 0.0
    z = (np.log(x) - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def log_normal_cdf_inv(p, mu, sigma):
    """
    对数正态分布逆CDF（分位数函数）。
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return np.inf
    z = mu + sigma * np.sqrt(2.0) * erfinv_safe(2.0 * p - 1.0)
    return np.exp(z)


def erfinv_safe(y):
    """安全版本的 erf 逆函数。"""
    y = np.clip(y, -0.999999, 0.999999)
    from scipy.special import erfinv
    return erfinv(y)


def log_normal_truncated_sample(mu, sigma, a, b, size=None):
    """
    从截断对数正态分布 [a, b] 中采样。
    
    公式:
        X = CDF^{-1}( CDF(a) + U*(CDF(b) - CDF(a)) )
        U ~ Uniform(0,1)
    """
    if a >= b:
        raise ValueError("a 必须小于 b")
    if sigma <= 0:
        raise ValueError("sigma 必须为正")
    
    cdf_a = log_normal_cdf(a, mu, sigma)
    cdf_b = log_normal_cdf(b, mu, sigma)
    
    if cdf_b <= cdf_a:
        raise ValueError("CDF(b) 必须大于 CDF(a)")
    
    if size is None:
        u = np.random.rand()
    else:
        u = np.random.rand(size)
    
    cdf_val = cdf_a + u * (cdf_b - cdf_a)
    return log_normal_cdf_inv(cdf_val, mu, sigma)


def log_normal_truncated_mean(mu, sigma, a, b):
    """
    截断对数正态分布的均值。
    
    公式:
        E[X] = exp(μ + σ^2/2) * [Φ(β - σ) - Φ(α - σ)] / [Φ(β) - Φ(α)]
        其中 α = (ln a - μ)/σ, β = (ln b - μ)/σ
    """
    alpha = (np.log(a) - mu) / sigma
    beta = (np.log(b) - mu) / sigma
    
    Z = 0.5 * (erf(beta / np.sqrt(2.0)) - erf(alpha / np.sqrt(2.0)))
    if np.abs(Z) < 1e-15:
        return 0.5 * (a + b)
    
    numerator = 0.5 * (erf((beta - sigma) / np.sqrt(2.0)) - erf((alpha - sigma) / np.sqrt(2.0)))
    return np.exp(mu + 0.5 * sigma**2) * numerator / Z


def log_normal_truncated_variance(mu, sigma, a, b):
    """
    截断对数正态分布的方差。
    """
    mean = log_normal_truncated_mean(mu, sigma, a, b)
    alpha = (np.log(a) - mu) / sigma
    beta = (np.log(b) - mu) / sigma
    
    Z = 0.5 * (erf(beta / np.sqrt(2.0)) - erf(alpha / np.sqrt(2.0)))
    if np.abs(Z) < 1e-15:
        return ((b - a)**2) / 12.0
    
    term1 = np.exp(2.0 * mu + sigma**2)
    term2_num = 0.5 * (erf((beta - 2.0 * sigma) / np.sqrt(2.0)) - erf((alpha - 2.0 * sigma) / np.sqrt(2.0)))
    term2 = term1 * term2_num / Z
    
    return term2 - mean**2


# ---------------------------------------------------------------------------
# 引力波参数先验分布
# ---------------------------------------------------------------------------

class GWPrior:
    """
    引力波双黑洞系统的先验分布。
    
    参数空间:
        m1, m2: 主/次级黑洞质量 [M_sun]
        a1, a2: 无量纲自旋参数 [-1, 1]
        D_L: 光度距离 [Mpc]
        inclination: 轨道倾角 [0, π]
        phi_c: 并合相位 [0, 2π]
        psi: 偏振角 [0, π]
        t_c: 并合时间 [s]
    """
    
    def __init__(self, m_min=5.0, m_max=100.0, D_L_min=100.0, D_L_max=5000.0,
                 mu_mass=3.5, sigma_mass=0.5):
        self.m_min = m_min
        self.m_max = m_max
        self.D_L_min = D_L_min
        self.D_L_max = D_L_max
        self.mu_mass = mu_mass
        self.sigma_mass = sigma_mass
    
    def sample(self, n_samples=1):
        """
        从先验分布中采样参数。
        
        质量先验: 截断对数正态（天体物理质量分布）
        自旋先验: 均匀分布 [-0.99, 0.99]
        距离先验: 均匀分布在体积元上 p(D_L) ∝ D_L^2
        角度先验: 各向同性
        """
        samples = []
        for _ in range(n_samples):
            # 质量: 截断对数正态
            m1 = log_normal_truncated_sample(self.mu_mass, self.sigma_mass, self.m_min, self.m_max)
            m2 = log_normal_truncated_sample(self.mu_mass, self.sigma_mass, self.m_min, min(m1, self.m_max))
            
            # 确保 m1 >= m2
            if m2 > m1:
                m1, m2 = m2, m1
            
            # 自旋: 截断正态近似
            a1 = np.clip(np.random.normal(0.0, 0.3), -0.99, 0.99)
            a2 = np.clip(np.random.normal(0.0, 0.3), -0.99, 0.99)
            
            # 距离: 体积加权
            u = np.random.rand()
            D_L = (self.D_L_min**3 + u * (self.D_L_max**3 - self.D_L_min**3))**(1.0 / 3.0)
            
            # 角度: 各向同性
            cos_inclination = 2.0 * np.random.rand() - 1.0
            inclination = np.arccos(np.clip(cos_inclination, -1.0, 1.0))
            phi_c = 2.0 * np.pi * np.random.rand()
            psi = np.pi * np.random.rand()
            t_c = np.random.normal(0.0, 0.1)  # 近似以 0 为中心
            
            samples.append({
                'm1': m1, 'm2': m2,
                'a1': a1, 'a2': a2,
                'D_L': D_L,
                'inclination': inclination,
                'phi_c': phi_c,
                'psi': psi,
                't_c': t_c
            })
        
        return samples if n_samples > 1 else samples[0]
    
    def log_prior(self, params):
        """计算先验的对数概率密度。"""
        lp = 0.0
        
        # 质量先验
        for m in [params['m1'], params['m2']]:
            if m < self.m_min or m > self.m_max:
                return -np.inf
            lp += np.log(log_normal_pdf(m, self.mu_mass, self.sigma_mass) + 1e-300)
        
        # 自旋先验
        for a in [params['a1'], params['a2']]:
            if np.abs(a) > 1.0:
                return -np.inf
            lp += np.log(0.5)  # 均匀 [-1,1] 的对数密度
        
        # 距离先验 p(D_L) ∝ D_L^2
        D_L = params['D_L']
        if D_L < self.D_L_min or D_L > self.D_L_max:
            return -np.inf
        lp += 2.0 * np.log(D_L) - np.log(self.D_L_max**3 - self.D_L_min**3)
        
        # 角度先验（各向同性为常数）
        return lp


# ---------------------------------------------------------------------------
# 方形区域高斯求积 (源自 1144_square_felippa_rule)
# ---------------------------------------------------------------------------

def line_unit_gauss(order):
    """
    一维标准区间 [-1,1] 上的 Gauss-Legendre 节点和权重。
    支持阶数 1 到 5。
    """
    if order == 1:
        x = np.array([0.0])
        w = np.array([2.0])
    elif order == 2:
        x = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
        w = np.array([1.0, 1.0])
    elif order == 3:
        x = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    elif order == 4:
        x1 = np.sqrt(3.0/7.0 - 2.0/7.0*np.sqrt(6.0/5.0))
        x2 = np.sqrt(3.0/7.0 + 2.0/7.0*np.sqrt(6.0/5.0))
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        x = np.array([-x2, -x1, x1, x2])
        w = np.array([w2, w1, w1, w2])
    elif order == 5:
        x1 = np.sqrt(5.0 - 2.0*np.sqrt(10.0/7.0)) / 3.0
        x2 = np.sqrt(5.0 + 2.0*np.sqrt(10.0/7.0)) / 3.0
        w1 = (322.0 + 13.0*np.sqrt(70.0)) / 900.0
        w2 = (322.0 - 13.0*np.sqrt(70.0)) / 900.0
        x = np.array([-x2, -x1, 0.0, x1, x2])
        w = np.array([w2, w1, 128.0/225.0, w1, w2])
    else:
        raise ValueError(f"不支持的一维求积阶数: {order}")
    
    return x, w


def square_felippa_rule(a, b, order_1d=(5, 5)):
    """
    二维方形区域 [a1,b1]×[a2,b2] 上的高斯求积规则。
    
    公式:
        ∫∫ f(x,y) dx dy ≈ Σ_i Σ_j w_i w_j f(x_i, y_j)
        
    在引力波贝叶斯推断中，用于计算质量平面上的边缘后验:
        p(m1, m2|d) = ∫ p(θ|d) d(其他参数)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    if a.shape != (2,) or b.shape != (2,):
        raise ValueError("a 和 b 必须为二维向量")
    
    x1, w1 = line_unit_gauss(order_1d[0])
    x2, w2 = line_unit_gauss(order_1d[1])
    
    # 变换到实际区间
    x1_scaled = 0.5 * (b[0] - a[0]) * x1 + 0.5 * (b[0] + a[0])
    x2_scaled = 0.5 * (b[1] - a[1]) * x2 + 0.5 * (b[1] + a[1])
    w1_scaled = 0.5 * (b[0] - a[0]) * w1
    w2_scaled = 0.5 * (b[1] - a[1]) * w2
    
    # 构建二维网格
    X1, X2 = np.meshgrid(x1_scaled, x2_scaled, indexing='ij')
    W1, W2 = np.meshgrid(w1_scaled, w2_scaled, indexing='ij')
    
    points = np.column_stack([X1.ravel(), X2.ravel()])
    weights = (W1 * W2).ravel()
    
    return points, weights


def marginal_posterior_mass_plane(log_posterior_func, m1_range, m2_range, order=5):
    """
    使用方形高斯求积计算质量平面上的边缘后验。
    
    参数:
        log_posterior_func: 函数 log_posterior(m1, m2)，固定其他参数
        m1_range: (m1_min, m1_max)
        m2_range: (m2_min, m2_max)
        order: 每维求积阶数
    
    返回:
        网格上的后验值和归一化常数（证据）
    """
    a = np.array([m1_range[0], m2_range[0]])
    b = np.array([m1_range[1], m2_range[1]])
    
    points, weights = square_felippa_rule(a, b, order_1d=(order, order))
    
    log_post = np.zeros(len(points))
    for i, (m1, m2) in enumerate(points):
        try:
            log_post[i] = log_posterior_func(m1, m2)
        except Exception:
            log_post[i] = -np.inf
    
    # 数值稳定性处理：减去最大值
    finite_mask = np.isfinite(log_post)
    if not np.any(finite_mask):
        # 所有值都是 -inf，返回均匀分布
        post_unnorm = np.ones_like(log_post)
        log_post_max = 0.0
    else:
        log_post_max = np.max(log_post[finite_mask])
        post_unnorm = np.exp(log_post - log_post_max)
    post_unnorm[~np.isfinite(log_post)] = 0.0
    
    # 计算证据
    evidence = np.sum(weights * post_unnorm) * np.exp(log_post_max)
    
    return points, post_unnorm, evidence


# ---------------------------------------------------------------------------
# 简化似然函数与后验采样
# ---------------------------------------------------------------------------

def gaussian_likelihood(signal, template, noise_psd=1.0):
    """
    简化的高斯似然函数（时域）。
    
    公式:
        ln L = -1/2 Σ_i (d_i - h_i)^2 / σ^2
    """
    residual = signal - template
    return -0.5 * np.sum(residual**2) / noise_psd


def metropolis_hastings(log_posterior_func, initial_params, n_steps=5000, step_sizes=None):
    """
    Metropolis-Hastings MCMC 采样。
    
    在引力波参数估计中，用于从后验分布 p(θ|d) 中采样。
    
    公式:
        1. 提议: θ' = θ + N(0, Σ)
        2. 接受率: α = min(1, p(θ'|d) / p(θ|d))
        3. 以概率 α 接受 θ'
    """
    params = initial_params.copy()
    
    if step_sizes is None:
        step_sizes = {
            'm1': 2.0, 'm2': 2.0,
            'a1': 0.05, 'a2': 0.05,
            'D_L': 100.0,
            'inclination': 0.1,
            'phi_c': 0.3,
            'psi': 0.3,
            't_c': 0.01
        }
    
    samples = []
    log_post_current = log_posterior_func(params)
    
    n_accepted = 0
    for _ in range(n_steps):
        # 提议新参数
        params_prop = {}
        for key in params:
            if key in step_sizes:
                params_prop[key] = params[key] + step_sizes[key] * np.random.randn()
            else:
                params_prop[key] = params[key]
        
        # 边界处理
        params_prop['m1'] = max(1.0, params_prop['m1'])
        params_prop['m2'] = max(1.0, min(params_prop['m2'], params_prop['m1']))
        params_prop['a1'] = np.clip(params_prop['a1'], -0.99, 0.99)
        params_prop['a2'] = np.clip(params_prop['a2'], -0.99, 0.99)
        params_prop['D_L'] = max(10.0, params_prop['D_L'])
        params_prop['inclination'] = np.clip(params_prop['inclination'], 0.0, np.pi)
        
        log_post_prop = log_posterior_func(params_prop)
        
        # 接受率
        if np.isfinite(log_post_prop):
            delta = log_post_prop - log_post_current
            if delta > 0 or np.random.rand() < np.exp(delta):
                params = params_prop
                log_post_current = log_post_prop
                n_accepted += 1
        
        samples.append(params.copy())
    
    acceptance_rate = n_accepted / n_steps
    return samples, acceptance_rate
