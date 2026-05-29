"""
nutrient_diffusion.py

肿瘤微环境中营养物质（氧气、葡萄糖）扩散-反应方程求解模块

本模块融合以下种子项目的核心算法：
  - 646_laplace_radial_exact: 径向 Laplace 方程的精确解
  - 1056_sandia_sparse: 稀疏网格（Sparse Grid）Clenshaw-Curtis 数值积分

科学背景：
  肿瘤微环境（TME）中的氧气分布由反应-扩散方程控制：

    D_O2 * nabla^2 C - lambda * C - f(C, rho) = 0

  其中 C 为氧气浓度，D_O2 为扩散系数，lambda 为自然衰减率，
  f(C, rho) 为肿瘤细胞消耗项，采用 Michaelis-Menten 动力学：

    f(C, rho) = V_max * rho * C / (K_m + C)

  在径向对称假设下， Laplace 方程 nabla^2 u = 0 的通解为：
    2D: u(r) = a * log(r) + b
    3D: u(r) = a / r + b

  对于高维参数积分（如多参数药物响应曲面），采用稀疏网格积分：
    Q_l^{(d)} f = sum_{|k|_1 <= l+d-1} (Delta_{k_1} x ... x Delta_{k_d}) f
  其中 Delta_k = Q_k - Q_{k-1} 为逐层差分，Q_k 为 1D Clenshaw-Curtis 积分。
"""

import numpy as np
from typing import Callable, Tuple


def laplace_radial_2d_exact(
    x: np.ndarray, y: np.ndarray, a: float, b: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    2D 径向 Laplace 方程的精确解及其导数。

    方程:  u_xx + u_yy = 0
    解:    r = sqrt(x^2 + y^2)
           u(r) = a * log(r) + b

    导数:
        u_x   = a * x / r^2
        u_y   = a * y / r^2
        u_xx  = a * (-2*x^2/r^2 + 1) / r^2
        u_xy  = -2*a*x*y / r^4
        u_yy  = a * (-2*y^2/r^2 + 1) / r^2

    参数:
        x, y: 空间坐标数组
        a, b: 径向解参数

    返回:
        u, ux, uy, uxx, uxy, uyy
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x ** 2 + y ** 2
    r2 = np.where(r2 < 1e-15, 1e-15, r2)  # 数值鲁棒性
    r = np.sqrt(r2)

    u = a * np.log(r) + b
    ux = a * x / r2
    uy = a * y / r2
    uxx = a * (-2.0 * x ** 2 / r2 + 1.0) / r2
    uxy = -2.0 * a * x * y / (r2 ** 2)
    uyy = a * (-2.0 * y ** 2 / r2 + 1.0) / r2

    return u, ux, uy, uxx, uxy, uyy


def laplace_radial_3d_exact(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, a: float, b: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    3D 径向 Laplace 方程的精确解。

    方程:  u_xx + u_yy + u_zz = 0
    解:    r = sqrt(x^2 + y^2 + z^2)
           u(r) = a / r + b
    """
    r2 = x ** 2 + y ** 2 + z ** 2
    r2 = np.where(r2 < 1e-15, 1e-15, r2)
    r = np.sqrt(r2)

    u = a / r + b
    ux = -a * x / (r2 * r)
    uy = -a * y / (r2 * r)
    uz = -a * z / (r2 * r)
    return u, ux, uy, uz


def oxygen_diffusion_steady_state_radial(
    r: np.ndarray, R_tumor: float, C_boundary: float,
    D: float = 1.0e-5, consumption_rate: float = 0.01
) -> np.ndarray:
    """
    求解径向对称稳态氧气扩散方程的近似解析解。

    控制方程（简化线性消耗）：
        D * (1/r) * d/dr (r * dC/dr) - k * C = 0

    通解（修正 Bessel 函数零阶）：
        C(r) = A * I_0(sqrt(k/D)*r) + B * K_0(sqrt(k/D)*r)

    边界条件：
        C(R_tumor) = C_boundary
        dC/dr(0) = 0   (对称性)

    参数:
        r: 径向坐标数组
        R_tumor: 肿瘤半径
        C_boundary: 边界浓度
        D: 扩散系数 (cm^2/s)
        consumption_rate: 消耗率常数 k

    返回:
        C: 氧气浓度分布
    """
    from scipy.special import i0, k0, i1

    r = np.asarray(r, dtype=float)
    r = np.clip(r, 1e-12, R_tumor)

    alpha = np.sqrt(consumption_rate / D)

    # 在 r=0 处 K_0 发散，故 B=0
    # C(r) = A * I_0(alpha * r)
    # 由 C(R) = C_boundary 得 A = C_boundary / I_0(alpha * R)
    i0_alphaR = i0(alpha * R_tumor)
    if abs(i0_alphaR) < 1e-15:
        i0_alphaR = 1e-15

    A = C_boundary / i0_alphaR
    C = A * i0(alpha * r)

    # 数值保护
    C = np.clip(C, 0.0, C_boundary)
    return C


def cc_abscissa(order: int, i: int) -> float:
    """
    Clenshaw-Curtis 求积公式的第 i 个节点。

    节点公式:
        x_i = cos( (order - i) * pi / (order - 1) ),  i = 1..order
    当 order = 1 时，唯一节点为 0。
    """
    if order < 1:
        raise ValueError("cc_abscissa: order 必须 >= 1")
    if i < 1 or i > order:
        raise ValueError("cc_abscissa: i 超出范围")
    if order == 1:
        return 0.0
    if 2 * (order - i) == order - 1:
        return 0.0
    return np.cos((order - i) * np.pi / (order - 1))


def cc_weights(n: int) -> np.ndarray:
    """
    计算 n 点 Clenshaw-Curtis 求积权重。

    权重公式（基于离散余弦变换）：
        w_i = c_i / (n-1) * [1 - sum_{j=1}^{floor((n-1)/2)} b_j * cos(2*j*theta_i)/(4*j^2-1)]
    其中 theta_i = (i-1)*pi/(n-1), b_j = 1 (j=(n-1)/2) 否则 b_j = 2, c_i = 1 (端点) 否则 c_i = 2。
    """
    if n < 1:
        raise ValueError("cc_weights: n 必须 >= 1")
    w = np.zeros(n)
    if n == 1:
        w[0] = 2.0
        return w

    theta = np.zeros(n)
    for i in range(n):
        theta[i] = i * np.pi / (n - 1)

    for i in range(n):
        w[i] = 1.0
        for j in range(1, (n - 1) // 2 + 1):
            if 2 * j == n - 1:
                b = 1.0
            else:
                b = 2.0
            w[i] -= b * np.cos(2.0 * j * theta[i]) / (4.0 * j * j - 1.0)

    w[0] = w[0] / (n - 1)
    w[1:n - 1] = 2.0 * w[1:n - 1] / (n - 1)
    w[n - 1] = w[n - 1] / (n - 1)
    return w


def clenshaw_curtis_integrate(f: Callable[[np.ndarray], np.ndarray],
                               a: float, b: float, n: int) -> float:
    """
    使用 n 点 Clenshaw-Curtis 公式计算 f 在 [a,b] 上的积分。

    积分变换:  x in [a,b]  <=>  t in [-1,1]
        x = (1-t)/2 * a + (t+1)/2 * b
        dx = (b-a)/2 * dt
    """
    if n < 1:
        raise ValueError("clenshaw_curtis_integrate: n 必须 >= 1")
    if b <= a:
        raise ValueError("clenshaw_curtis_integrate: 需要 b > a")

    t = np.array([cc_abscissa(n, i + 1) for i in range(n)])
    w = cc_weights(n)
    x = 0.5 * ((1.0 - t) * a + (t + 1.0) * b)
    fx = f(x)
    fx = np.asarray(fx).ravel()
    if fx.shape[0] != n:
        raise ValueError("clenshaw_curtis_integrate: f 输出维度不匹配")

    q = np.sum(w * fx) * (b - a) / 2.0
    return float(q)


def sparse_grid_monomial_integral(dim: int, level: int,
                                   exponents: np.ndarray) -> float:
    """
    基于稀疏网格的 d 维单项式积分估计（2D 组合技术实现）。

    稀疏网格组合公式（Smolyak 构造，d=2）：
        Q_L^{(2)} = sum_{l=1}^{L+1} sum_{|i|_1 = l+1} (Delta_{i1} x Delta_{i2})
      其中 Delta_k = Q_k - Q_{k-1}，Q_0 = 0。

    1D Clenshaw-Curtis 规则层级：
        order = 2^{level} + 1  (level >= 1),  order = 1 (level = 0)

    参数:
        dim: 空间维数 d（当前实现支持 d=1,2）
        level: 稀疏网格层级 l
        exponents: 长度为 dim 的指数向量 alpha

    返回:
        integral: 积分估计值
    """
    if dim < 1:
        raise ValueError("sparse_grid_monomial_integral: dim >= 1")
    if level < 0:
        raise ValueError("sparse_grid_monomial_integral: level >= 0")
    exponents = np.asarray(exponents, dtype=int)
    if exponents.shape[0] != dim:
        raise ValueError("sparse_grid_monomial_integral: exponents 长度不匹配")

    def _level_to_order_closed(lv):
        if lv == 0:
            return 1
        return 2 ** lv + 1

    def _cc_rule(lv):
        """返回 1D CC 规则 (pts in [0,1], weights)"""
        o = _level_to_order_closed(lv)
        pts = np.array([cc_abscissa(o, i + 1) for i in range(o)])
        ws = cc_weights(o)
        # 映射到 [0,1]
        pts = 0.5 * (pts + 1.0)
        ws = ws * 0.5
        return pts, ws

    def _eval_1d(pts, alpha):
        return pts ** alpha

    def _tensor_product_integral(l1, l2):
        """计算 (Q_{l1} x Q_{l2}) f"""
        p1, w1 = _cc_rule(l1)
        p2, w2 = _cc_rule(l2)
        f1 = _eval_1d(p1, exponents[0])
        f2 = _eval_1d(p2, exponents[1]) if dim > 1 else np.ones_like(p2)
        # 张量积求和
        val = 0.0
        for i in range(len(p1)):
            for j in range(len(p2)):
                val += w1[i] * w2[j] * f1[i] * f2[j]
        return val

    if dim == 1:
        # 一维直接用最高层级规则
        total = 0.0
        for l1 in range(level + 1):
            p1, w1 = _cc_rule(l1)
            f1 = _eval_1d(p1, exponents[0])
            if l1 == 0:
                total += np.sum(w1 * f1)
            else:
                total += np.sum(w1 * f1) - np.sum(_cc_rule(l1 - 1)[1] * _eval_1d(_cc_rule(l1 - 1)[0], exponents[0]))
        return total

    if dim == 2:
        # Smolyak 组合公式：
        #   A(L,2) = sum_{|l|_1 <= L+1} alpha_l * (Q_{l1} x Q_{l2})
        #   alpha_l = (-1)^{L+1-|l|_1} * C(1, L+1-|l|_1)
        #   其中 C(n,k) = 0 若 k < 0 或 k > n
        from math import comb
        total = 0.0
        max_sum = level + 1
        for l1 in range(1, max_sum + 1):
            for l2 in range(1, max_sum + 1):
                s = l1 + l2
                if s > max_sum + 1:
                    continue
                k = max_sum + 1 - s
                if k < 0 or k > 1:
                    continue
                alpha = ((-1) ** k) * comb(1, k)
                q = _tensor_product_integral(l1, l2)
                total += alpha * q
        return total

    # 高维回退到全张量积（简化）
    total = 1.0
    for d_idx in range(dim):
        p, w = _cc_rule(level)
        f = _eval_1d(p, exponents[d_idx])
        total *= np.sum(w * f)
    return total


def michaelis_menten_consumption(C: np.ndarray, rho: np.ndarray,
                                  Vmax: float, Km: float) -> np.ndarray:
    """
    Michaelis-Menten 消耗动力学。

    公式:
        f(C, rho) = Vmax * rho * C / (Km + C)

    参数:
        C: 底物浓度数组
        rho: 细胞密度数组
        Vmax: 最大反应速率
        Km: 米氏常数 (半饱和浓度)

    返回:
        consumption: 消耗速率数组
    """
    # === HOLE 1 START ===
    # 请根据 Michaelis-Menten 动力学公式实现此函数
    raise NotImplementedError("Hole_1: michaelis_menten_consumption 待实现")
    # === HOLE 1 END ===


def hypoxia_region_fraction(C: np.ndarray, threshold: float = 0.02) -> float:
    """
    计算缺氧区域比例（用于评估肿瘤坏死核形成）。

    阈值通常取氧气分压 pO2 < 2 mmHg (约 0.26% O2)，
    这里用归一化浓度的 0.02 作为阈值。
    """
    if C.size == 0:
        return 0.0
    return float(np.mean(C < threshold))
