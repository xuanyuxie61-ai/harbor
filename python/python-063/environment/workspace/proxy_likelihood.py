"""
proxy_likelihood.py
古气候代理数据观测似然模块

基于截断对数正态分布 (Truncated Log-Normal) 的代理观测误差模型，
融合种子项目 699_log_normal_truncated_ab（截断对数正态分布的 PDF/CDF/采样）。

代理类型:
    1. tree_ring:   树轮宽度 — 温度敏感，乘性截断对数正态误差
    2. ice_core:    冰芯氧同位素 — 线性温度关系，加性高斯误差
    3. lake_sediment: 湖泊沉积物 — 非线性关系，混合误差
"""

import numpy as np


SQRT2PI = np.sqrt(2.0 * np.pi)


def normal_01_cdf(x):
    """
    标准正态累积分布函数。
    使用分段多项式/有理近似（Adams 算法）。
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    abs_x = np.abs(x)

    # |x| <= 1.28
    mask1 = abs_x <= 1.28
    if np.any(mask1):
        x1 = x[mask1]
        a = np.array([0.5, 0.398942280444, 0.0, -0.03626817857, 0.0,
                      0.01376850678, 0.0, -0.0038468995, 0.0, 0.0009016275])
        t = x1 * x1
        poly = (a[0] + a[1]*x1 + t*(a[2] + a[3]*x1 + t*(a[4] + a[5]*x1
               + t*(a[6] + a[7]*x1 + t*(a[8] + a[9]*x1)))))
        result[mask1] = poly

    # 1.28 < |x| <= 12.7
    mask2 = (abs_x > 1.28) & (abs_x <= 12.7)
    if np.any(mask2):
        x2 = abs_x[mask2]
        b = np.array([0.398942280385, 0.0, -0.03988004123, 0.0, 0.0038095835,
                      0.0, -0.000318030, 0.0, 0.00002058])
        t = 1.0 / (1.0 + 0.2316419 * x2)
        poly = (b[0] + t*(b[1] + t*(b[2] + t*(b[3] + t*(b[4] + t*(b[5]
               + t*(b[6] + t*(b[7] + t*b[8]))))))))
        q = poly * np.exp(-0.5 * x2 * x2)
        pos = x[mask2] >= 0
        res = np.zeros_like(x2)
        res[pos] = 1.0 - q[pos]
        res[~pos] = q[~pos]
        result[mask2] = res

    # |x| > 12.7
    mask3 = abs_x > 12.7
    if np.any(mask3):
        pos = x[mask3] >= 0
        result[mask3] = np.where(pos, 1.0, 0.0)

    return result


def normal_01_cdf_inv(p):
    """
    标准正态分位数函数（逆 CDF），使用 Wichura 算法 AS 241。
    采用 np.polyval 避免深层括号导致的缩进/语法问题。
    """
    p = np.asarray(p, dtype=np.float64)
    result = np.zeros_like(p)
    q = p - 0.5
    mask_center = np.abs(q) <= 0.425

    if np.any(mask_center):
        r = 0.180625 - q[mask_center]**2
        num_c = [2.5090809287301226727e3, 3.3430575583588128105e4,
                 6.7265770927008700853e4, 4.5921953931549871457e4,
                 1.3731693765509461125e4, 1.9715909503065514427e3,
                 1.3314136788658342429e2, 3.3871328727963666080e0]
        den_c = [5.2264952788528545610e3, 8.3895870404983970412e4,
                 2.0996759200495332527e5, 2.2956168828376951064e5,
                 9.9928587571567665716e4, 1.6067454290310800919e4,
                 8.6258806209246785755e2, 1.0]
        num = np.polyval(num_c, r)
        den = np.polyval(den_c, r)
        result[mask_center] = q[mask_center] * num / den

    mask_tail = ~mask_center
    if np.any(mask_tail):
        tail_idx = np.where(mask_tail)[0]
        r = np.where(p[mask_tail] > 0.5, 1.0 - p[mask_tail], p[mask_tail])
        r = -np.log(r)
        for k, idx in enumerate(tail_idx):
            rv = r[k]
            if rv <= 5.0:
                num_c = [7.7454501427834140764e-4, 2.27238449892691845833e-2,
                         2.4178072517741171243e-1, 1.27045825245236883258,
                         3.64784832476320460504, 5.7694972214606914055,
                         4.6303378461565452959, 1.42343711074968357734]
                den_c = [1.05075007164441684324e-9, 5.475938084995344946e-4,
                         1.51986665636164571966e-2, 1.4810397642748007459e-1,
                         6.8972103490685375859e-1, 1.6763848301838038494,
                         2.05319162663775882187, 1.0]
                num = np.polyval(num_c, rv)
                den = np.polyval(den_c, rv)
            else:
                num_c = [6.6579046435011037772e0, 3.05326634961232344035e1,
                         5.04598536917905979166e1, 3.15251012994246677635e1,
                         6.05519622082551528612, 2.37598772771897739775,
                         5.03318966494356080678e-1]
                den_c = [1.36661917444457262247e2, 7.95854036644138e2,
                         1.77763632642703475278e3, 1.48172274331424159751e3,
                         4.77846592983032808346e2, 4.99126312092038080447e1,
                         1.0]
                num = np.polyval(num_c, rv)
                den = np.polyval(den_c, rv)
            sign = 1.0 if p[idx] > 0.5 else -1.0
            result[idx] = sign * (num / den)
    return result


def log_normal_truncated_ab_pdf(x, mu, sigma, a, b):
    """
    截断对数正态分布概率密度函数:
        f(x) = (1/(x*sigma*sqrt(2*pi)*Z)) * exp(-(ln x - mu)^2 / (2*sigma^2))
    其中 Z = Phi((ln b - mu)/sigma) - Phi((ln a - mu)/sigma)。
    当 x 不在 [a, b] 区间时返回 0。
    """
    x = np.asarray(x, dtype=np.float64)
    x_safe = np.where(x <= 0, 1e-300, x)
    a0 = (np.log(a) - mu) / sigma
    b0 = (np.log(b) - mu) / sigma
    Z = normal_01_cdf(b0) - normal_01_cdf(a0)
    Z = max(Z, 1e-300)
    u = (np.log(x_safe) - mu) / sigma
    pdf = np.exp(-0.5 * u * u) / (x_safe * sigma * SQRT2PI * Z)
    pdf = np.where((x >= a) & (x <= b), pdf, 0.0)
    return pdf


def log_normal_truncated_ab_mean(mu, sigma, a, b):
    """截断对数正态分布的解析均值。"""
    a0 = (np.log(a) - mu) / sigma
    b0 = (np.log(b) - mu) / sigma
    denom = normal_01_cdf(b0) - normal_01_cdf(a0)
    denom = max(denom, 1e-300)
    numer = normal_01_cdf(sigma - a0) - normal_01_cdf(sigma - b0)
    return np.exp(mu + 0.5 * sigma**2) * numer / denom


def log_normal_truncated_ab_sample(mu, sigma, a, b, size=None):
    """
    截断对数正态分布采样（逆变换采样）。
    先生成 U ~ Uniform(0,1)，映射到截断区间 [F(a), F(b)]，再应用逆 CDF。
    """
    if size is None:
        size = 1
    U = np.random.uniform(size=size)
    a0 = (np.log(a) - mu) / sigma
    b0 = (np.log(b) - mu) / sigma
    Fa = normal_01_cdf(a0)
    Fb = normal_01_cdf(b0)
    p = Fa + U * (Fb - Fa)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    z = normal_01_cdf_inv(p)
    return np.exp(mu + sigma * z)


class ProxyObservationModel:
    """
    古气候代理观测模型。

    三类代理:
      - tree_ring:    树轮宽度，乘性截断对数正态噪声
      - ice_core:     delta18O，加性高斯噪声
      - lake_sediment: 沉积指数，乘性截断对数正态噪声
    """

    PROXY_TYPES = ['tree_ring', 'ice_core', 'lake_sediment']

    def __init__(self, proxy_type, location, params=None):
        if proxy_type not in self.PROXY_TYPES:
            raise ValueError(f"Unknown proxy type: {proxy_type}")
        self.proxy_type = proxy_type
        self.location = location
        self.params = params if params is not None else self._default_params()

    def _default_params(self):
        if self.proxy_type == 'tree_ring':
            return {
                'sensitivity': 0.15,
                'base_width': 1.0,
                'ln_mu': 0.0,
                'ln_sigma': 0.3,
                'ln_a': 0.1,
                'ln_b': 5.0
            }
        elif self.proxy_type == 'ice_core':
            return {
                'slope': -0.5,
                'intercept': -30.0,
                'noise_std': 0.5
            }
        else:  # lake_sediment
            return {
                'sensitivity': 0.1,
                'nonlinear_exp': 1.5,
                'ln_mu': 0.05,
                'ln_sigma': 0.25,
                'ln_a': 0.2,
                'ln_b': 3.0
            }

    def forward_model(self, T):
        """
        观测算子 H: 局地温度 T [K] -> 代理观测期望值。
        """
        T = np.asarray(T, dtype=np.float64)
        if self.proxy_type == 'tree_ring':
            T_opt = 288.0
            width = self.params['base_width'] * np.exp(
                -self.params['sensitivity'] * ((T - T_opt) / T_opt)**2
            )
            return width
        elif self.proxy_type == 'ice_core':
            return self.params['slope'] * T + self.params['intercept']
        else:
            T_ref = 280.0
            return self.params['sensitivity'] * (T / T_ref)**self.params['nonlinear_exp']

    def sample_observation(self, T):
        """从代理模型中采样含噪声的观测值。"""
        expected = self.forward_model(T)
        if self.proxy_type == 'tree_ring':
            noise = log_normal_truncated_ab_sample(
                self.params['ln_mu'], self.params['ln_sigma'],
                self.params['ln_a'], self.params['ln_b'],
                size=np.asarray(expected).shape
            )
            return expected * noise
        elif self.proxy_type == 'ice_core':
            noise = np.random.normal(0.0, self.params['noise_std'],
                                     size=np.asarray(expected).shape)
            return expected + noise
        else:
            noise = log_normal_truncated_ab_sample(
                self.params['ln_mu'], self.params['ln_sigma'],
                self.params['ln_a'], self.params['ln_b'],
                size=np.asarray(expected).shape
            )
            return expected * noise

    def log_likelihood(self, observation, T):
        """计算 log p(observation | T)。"""
        expected = self.forward_model(T)
        if self.proxy_type == 'tree_ring' or self.proxy_type == 'lake_sediment':
            ratio = observation / np.maximum(expected, 1e-300)
            pdf = log_normal_truncated_ab_pdf(
                ratio, self.params['ln_mu'], self.params['ln_sigma'],
                self.params['ln_a'], self.params['ln_b']
            )
        else:
            diff = observation - expected
            pdf = np.exp(-0.5 * (diff / self.params['noise_std'])**2) / \
                  (self.params['noise_std'] * SQRT2PI)
        pdf = np.maximum(pdf, 1e-300)
        return np.log(pdf)
