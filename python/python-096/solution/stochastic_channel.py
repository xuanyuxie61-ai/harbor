"""
stochastic_channel.py
=====================
随机信道与统计噪声建模模块

核心算法来源：
  - 1008_random_walk_1d_simulation：一维随机游走
  - 033_asa076：标准正态累积分布函数 alnorm、Owen T 函数 tfn/tha
  - 542_histogram_pdf_2d_sample：二维离散 CDF/PDF 采样

在电磁学波束赋形中的角色：
  1. 一维随机游走模拟射频链路中相位漂移的随机过程
  2. alnorm / Owen T 函数用于计算天线方向图旁瓣电平的概率分布
  3. 二维直方图 PDF 采样用于生成空间衰落信道增益的蒙特卡罗样本
"""

import numpy as np
from typing import Tuple, Optional


def alnorm(x: float, upper: bool = False) -> float:
    """
    计算标准正态分布累积密度函数（Algorithm AS 66, Hill 1973）。

    来源：033_asa076

    数学定义：
      \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-t^2/2} dt

      upper=True  时计算 \int_{x}^{+\infty} \phi(t) dt
      upper=False 时计算 \int_{-\infty}^{x} \phi(t) dt

    参数：
        x:     积分端点
        upper: 是否计算上尾概率
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
    r = 0.398942280385
    utzero = 18.66

    up = upper
    z = float(x)

    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and (not up or utzero < z):
        return 0.0 if up else 1.0

    y = 0.5 * z * z

    if z <= con:
        value = 0.5 - z * (p - q * y / (y + a1 + b1 / (y + a2 + b2 / (y + a3))))
    else:
        value = r * np.exp(-y) / (z + c1 + d1 / (z + c2 + d2 / (z + c3 + d3 / (z + c4 + d4 / (z + c5 + d5 / (z + c6))))))

    if not up:
        value = 1.0 - value
    return value


def tfn_owen(x: float, fx: float) -> float:
    """
    计算 Owen 的 T 函数。

    来源：033_asa076 (Algorithm AS 76)

    数学定义：
      T(h, a) = \frac{1}{2\pi} \int_{0}^{a} \frac{e^{-h^2(1+x^2)/2}}{1+x^2} dx

    在双变量正态概率计算中具有核心作用：
      L(h, k; \rho) = \Phi(h)\Phi(k) + \sum_{i=0}^{\infty} \frac{\rho^{i}}{i!} \phi^{(i)}(h) \phi^{(i)}(k)
      亦可由 T 函数直接计算。
    """
    ng = 5
    r = np.array([0.1477621, 0.1346334, 0.1095432, 0.0747257, 0.0333357])
    tp = 0.159155
    tv1 = 1.0e-35
    tv2 = 15.0
    tv3 = 15.0
    tv4 = 1.0e-5
    u = np.array([0.0744372, 0.2166977, 0.3397048, 0.4325317, 0.4869533])

    if abs(x) < tv1:
        return tp * np.arctan(fx)
    if tv2 < abs(x):
        return 0.0
    if abs(fx) < tv1:
        return 0.0

    xs = -0.5 * x * x
    x2 = fx
    fxs = fx * fx

    if tv3 <= np.log(1.0 + fxs) - xs * fxs:
        x1 = 0.5 * fx
        fxs = 0.25 * fxs
        for _ in range(100):
            rt = fxs + 1.0
            x2_new = x1 + (xs * fxs + tv3 - np.log(rt)) / (2.0 * x1 * (1.0 / rt - xs))
            fxs = x2_new * x2_new
            if abs(x2_new - x1) < tv4:
                x2 = x2_new
                break
            x1 = x2_new
        else:
            x2 = x1

    rt = 0.0
    for i in range(ng):
        r1 = 1.0 + fxs * (0.5 + u[i]) ** 2
        r2 = 1.0 + fxs * (0.5 - u[i]) ** 2
        rt += r[i] * (np.exp(xs * r1) / r1 + np.exp(xs * r2) / r2)

    return rt * x2 * tp


def tha_owen(h1: float, h2: float, a1: float, a2: float) -> float:
    """
    计算 T(h1/h2, a1/a2)，处理任意实数输入的边界情况。

    来源：033_asa076
    """
    if h2 == 0.0:
        return 0.0
    h = h1 / h2
    if a2 == 0.0:
        g = alnorm(h, False)
        value = g / 2.0 if h < 0.0 else (1.0 - g) / 2.0
        if a1 < 0.0:
            value = -value
        return value
    a = a1 / a2
    if abs(h) < 0.3 and 7.0 < abs(a):
        lam = abs(a * h)
        ex = np.exp(-lam * lam / 2.0)
        g = alnorm(lam, False)
        c1 = (ex / lam + np.sqrt(2.0 * np.pi) * (g - 0.5)) / (2.0 * np.pi)
        c2 = ((lam * lam + 2.0) * ex / (lam ** 3) + np.sqrt(2.0 * np.pi) * (g - 0.5)) / (12.0 * np.pi)
        ah = abs(h)
        value = 0.25 - c1 * ah + c2 * ah ** 3
        if a < 0.0:
            value = -abs(value)
        else:
            value = abs(value)
        return value

    absa = abs(a)
    if absa <= 1.0:
        return tfn_owen(h, a)
    ah = absa * h
    gh = alnorm(h, False)
    gah = alnorm(ah, False)
    value = 0.5 * (gh + gah) - gh * gah - tfn_owen(ah, 1.0 / absa)
    if a < 0.0:
        value = -value
    return value


def bivariate_normal_cdf(h: float, k: float, rho: float) -> float:
    """
    计算相关系数为 rho 的标准双变量正态累积分布。

    数学公式（Owen, 1956）：
      \Phi_2(h, k; \rho) = \Phi(h)\Phi(k)
        + \sum_{i=0}^{\infty} \frac{\rho^{i+1}}{i!} \phi^{(i)}(h) \phi^{(i)}(k)

    或者使用 T 函数：
      \Phi_2(h, k; \rho) = \Phi(h)\Phi(k) + T(h, a_h) + T(k, a_k) + \delta
      其中 a_h = (k/h - \rho)/\sqrt{1-\rho^2},  a_k = (h/k - \rho)/\sqrt{1-\rho^2}

    这里采用更稳定的 Drezner 近似思想的简化版：
      当 h*k*rho 接近奇异时使用 T 函数分解。
    """
    if rho <= -1.0 + 1e-12:
        return max(0.0, alnorm(h, False) + alnorm(k, False) - 1.0)
    if rho >= 1.0 - 1e-12:
        return min(alnorm(h, False), alnorm(k, False))

    # 简化但鲁棒的实现：使用 Sheppard 公式（适用于 h>0, k>0）
    # 一般情况使用组合方法
    Phi_h = alnorm(h, False)
    Phi_k = alnorm(k, False)
    # 使用 tfn 的一阶修正
    if abs(h) > 1e-8 and abs(k) > 1e-8:
        sqrt_term = np.sqrt(1.0 - rho * rho)
        ah = (k / h - rho) / sqrt_term
        ak = (h / k - rho) / sqrt_term
        delta = 0.0
        if h * k > 0:
            if h < 0:
                delta = -0.5
            elif k < 0:
                delta = -0.5
        term_h = tha_owen(h, 1.0, ah, 1.0) if abs(ah) < 1e6 else 0.0
        term_k = tha_owen(k, 1.0, ak, 1.0) if abs(ak) < 1e6 else 0.0
        result = Phi_h * Phi_k + term_h + term_k + delta
        return max(0.0, min(1.0, result))
    return Phi_h * Phi_k


class RandomWalkPhaseNoise:
    """
    一维随机游走相位噪声模型。

    来源：1008_random_walk_1d_simulation

    物理背景：
      在相控阵中，每个通道的相位受温度漂移、时钟抖动等影响。
      这些影响可建模为离散时间随机游走：

        \phi_{n+1} = \phi_n + \Delta\phi_n,  \Delta\phi_n \sim \{-\delta, +\delta\}

      其中 \delta 为单步相位扰动量（rad）。
      理论上，E[\phi_n^2] = n \delta^2（扩散律）。
    """

    def __init__(self, step_delta: float = 0.01, seed: Optional[int] = None):
        self.step_delta = step_delta
        if seed is not None:
            np.random.seed(seed)

    def simulate(self, step_num: int, walk_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        模拟 walk_num 条轨迹，每条 step_num 步。

        返回：
            time:   (step_num+1,) 时间序列
            x2_ave: (step_num+1,) 均方位移
            x2_max: (step_num+1,) 最大位移平方
        """
        if step_num < 1 or walk_num < 1:
            raise ValueError("step_num 和 walk_num 必须 >= 1")
        x2_ave = np.zeros(step_num + 1, dtype=float)
        x2_max = np.zeros(step_num + 1, dtype=float)

        for _ in range(walk_num):
            x = 0.0
            for step in range(1, step_num + 1):
                if np.random.rand() <= 0.5:
                    x -= self.step_delta
                else:
                    x += self.step_delta
                x2_ave[step] += x * x
                x2_max[step] = max(x2_max[step], x * x)

        x2_ave /= walk_num
        time = np.arange(step_num + 1, dtype=float)
        return time, x2_ave, x2_max

    def theoretical_msd(self, n: int) -> float:
        """理论均方位移 E[\phi_n^2] = n * delta^2。"""
        return n * self.step_delta ** 2


def set_discrete_cdf_2d(pdf_mat: np.ndarray) -> np.ndarray:
    """
    由二维离散 PDF 构造 CDF。

    来源：542_histogram_pdf_2d_sample

    参数：
        pdf_mat: (m1, m2) 概率密度矩阵（无需归一化）
    返回：
        cdf_mat: (m1, m2) 累积分布矩阵
    """
    pdf_mat = np.asarray(pdf_mat, dtype=float)
    total = 0.0
    cdf_mat = np.zeros_like(pdf_mat)
    m1, m2 = pdf_mat.shape
    for j in range(m2):
        for i in range(m1):
            total += pdf_mat[i, j]
            cdf_mat[i, j] = total
    if total > 0:
        cdf_mat /= total
    return cdf_mat


def discrete_cdf_to_xy(m1: int, m2: int, cdf_mat: np.ndarray,
                       xb: np.ndarray, yb: np.ndarray,
                       n: int, u: np.ndarray) -> np.ndarray:
    """
    根据二维离散 CDF 反演采样坐标。

    来源：542_histogram_pdf_2d_sample

    参数：
        m1, m2: 网格维度
        cdf_mat: CDF 矩阵
        xb: (m1+1,) x 边界
        yb: (m2+1,) y 边界
        n: 采样点数
        u: (n,) 均匀随机数 [0,1]
    返回：
        s: (2, n) 采样坐标
    """
    s = np.zeros((2, n), dtype=float)
    low = 0.0
    for j in range(m2):
        for i in range(m1):
            high = cdf_mat[i, j]
            mask = (low <= u) & (u <= high)
            count = np.sum(mask)
            if count > 0:
                r = np.random.rand(2, count)
                s[0, mask] = (1.0 - r[0, :]) * xb[i] + r[0, :] * xb[i + 1]
                s[1, mask] = (1.0 - r[1, :]) * yb[j] + r[1, :] * yb[j + 1]
            low = high
    return s


def sample_spatial_fading(n_samples: int, x_range: Tuple[float, float] = (-1.0, 1.0),
                          y_range: Tuple[float, float] = (-1.0, 1.0),
                          correlation_length: float = 0.3,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    基于 2D 直方图 PDF 采样的空间衰落信道增益生成。

    物理模型：
      使用指数相关模型构造二维 PDF：
        PDF(x,y) \propto exp(-(x^2+y^2)/(2 L_c^2))

      然后通过离散 CDF 反演采样生成具有空间相关性的衰落样本。
    """
    if seed is not None:
        np.random.seed(seed)
    m1, m2 = 20, 20
    xb = np.linspace(x_range[0], x_range[1], m1 + 1)
    yb = np.linspace(y_range[0], y_range[1], m2 + 1)
    xc = 0.5 * (xb[:-1] + xb[1:])
    yc = 0.5 * (yb[:-1] + yb[1:])
    xv, yv = np.meshgrid(xc, yc, indexing='ij')
    pdf_mat = np.exp(-(xv ** 2 + yv ** 2) / (2.0 * correlation_length ** 2))
    cdf_mat = set_discrete_cdf_2d(pdf_mat)
    u = np.random.rand(n_samples)
    samples = discrete_cdf_to_xy(m1, m2, cdf_mat, xb, yb, n_samples, u)
    # 将样本位置映射为对数正态衰落增益
    r = np.sqrt(samples[0, :] ** 2 + samples[1, :] ** 2)
    gains = np.exp(-r / correlation_length)
    return gains


def sidelobe_level_cdf(level_db: float, n_elements: int,
                       array_factor_std: float = 1.0) -> float:
    """
    使用正态分布近似计算旁瓣电平低于阈值的概率。

    数学模型：
      对于大型均匀阵列，旁瓣包络近似服从瑞利分布，其累积分布为：
        P(SLL < L) = 1 - exp(-L^2 / (2\sigma^2))

      当 L 较大时，可用正态近似：
        P(SLL < L) \approx \Phi((L - \mu)/\sigma)

    这里使用 alnorm 进行标准正态 CDF 计算。
    """
    # 对于 N 元阵列，平均旁瓣电平约为 -10 log10(N) dB
    mean_sll = -10.0 * np.log10(max(n_elements, 1))
    z = (level_db - mean_sll) / array_factor_std
    return alnorm(z, False)
