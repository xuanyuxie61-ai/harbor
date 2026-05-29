"""
边界通量蒙特卡洛模块：基于 hexagon_monte_carlo，
在六边形边界采样区域上估算冠层侧边界碳通量。

核心公式：
  六边形单位区域面积：
      A = 3 * sqrt(3) / 2

  蒙特卡洛积分：
      integral_{Hex} f(x,y) dA ≈ (A / N) * sum_{i=1}^N f(x_i, y_i)

  侧边界碳通量（Fick 定律类比）：
      F_boundary = -D * dC/dn * L_perimeter
      其中 dC/dn 用法向差分近似。
"""
import numpy as np


def hexagon01_area():
    """单位六边形面积（顶点在 (1,0), (1/2, sqrt(3)/2), ...）。"""
    return 3.0 * np.sqrt(3.0) / 2.0


def hexagon01_sample(n):
    """
    在单位六边形内均匀采样 n 个点。
    返回: x, y
    """
    # 拒绝采样：在包围盒内采样并筛选
    samples = []
    while len(samples) < n:
        n_batch = max(n - len(samples), 100)
        x = np.random.uniform(-1.0, 1.0, n_batch)
        y = np.random.uniform(-np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0, n_batch)
        # 六边形条件
        mask = (np.abs(x) <= 1.0) & (np.abs(y) <= np.sqrt(3.0) / 2.0) & \
               (np.abs(x) + np.abs(y) / np.sqrt(3.0) <= 1.0)
        samples.extend(list(zip(x[mask], y[mask])))
    samples = np.array(samples[:n])
    return samples[:, 0], samples[:, 1]


def hexagon01_monte_carlo(n, func):
    """
    六边形上蒙特卡洛积分。
    func: 接受 x, y 数组，返回函数值数组
    """
    area = hexagon01_area()
    x, y = hexagon01_sample(n)
    vals = func(x, y)
    return (area / n) * np.sum(vals)


def estimate_lateral_flux(n_samples, diffusivity, concentration_gradient):
    """
    估算冠层侧边界碳通量。
    concentration_gradient: 函数 dC/dn(x,y) (umol/mol / m)
    返回: 通量 (umol/m^2/s)
    """
    def integrand(x, y):
        return -diffusivity * concentration_gradient(x, y)
    flux = hexagon01_monte_carlo(n_samples, integrand)
    return flux
