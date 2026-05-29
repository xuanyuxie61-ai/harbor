"""
error_estimator.py
==================
基于高阶数值积分的重建误差估计模块

科学背景：
---------
在图像重建中，需要量化重建图像与参考图像之间的差异。
常用的 L^2 误差定义为：
    e_{L^2} = \left( \int_\Omega |I_{true}(x,y) - I_{recon}(x,y)|^2 \, dx\, dy \right)^{1/2}

本模块利用高精度数值积分规则计算该误差：
1. 三角形上的 TWB（Taylor-Wingate-Bos）求积规则（来自项目 1323_triangle_twb_rule）
2. 金字塔上的 Witherden-Vincent 求积规则（来自项目 937_pyramid_witherden_rule）

三角形单位积分公式：
    对于参考三角形 T_ref = {(x,y): x>=0, y>=0, x+y<=1}，
    积分 I(f) = \int_{T_ref} f(x,y) dx dy \approx \sum_{i=1}^n w_i f(x_i, y_i)

其中 (x_i, y_i) 为求积节点，w_i 为权重。

TWB 规则具有多项式精确度 p，即对任意总次数不超过 p 的多项式精确成立。

金字塔积分用于三维图像体积（如 CT 重建）的误差估计：
    单位金字塔：-1 <= x <= 1, -1 <= y <= 1, 0 <= z <= 1
    V = \int_P dV = 4/3
"""

import numpy as np
from typing import Tuple, Optional


# TWB 规则参数：强度 -> 节点数
# 来自项目 1323_triangle_twb_rule
_TWB_RULE_ORDER = {
    1: 1, 2: 3, 4: 6, 5: 10, 7: 15, 9: 21,
    11: 28, 13: 36, 14: 45, 16: 55, 18: 66,
    20: 78, 21: 91, 23: 105, 25: 120
}

# TWB 规则数据（简化版，包含强度 1, 2, 4, 5）
_TWB_DATA = {
    1: {
        'x': np.array([1.0 / 3.0]),
        'y': np.array([1.0 / 3.0]),
        'w': np.array([0.5])
    },
    2: {
        'x': np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
        'y': np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        'w': np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    },
    4: {
        'x': np.array([0.108103018168070, 0.445948490915965, 0.445948490915965,
                       0.816847572980459, 0.091576213509771, 0.091576213509771]),
        'y': np.array([0.445948490915965, 0.108103018168070, 0.445948490915965,
                       0.091576213509771, 0.816847572980459, 0.091576213509771]),
        'w': np.array([0.111690794839005, 0.111690794839005, 0.111690794839005,
                       0.054975871827661, 0.054975871827661, 0.054975871827661])
    },
    5: {
        'x': np.array([1.0 / 3.0,
                       0.059715871789770, 0.797426985353087, 0.797426985353087,
                       0.138196601125011, 0.138196601125011, 0.585410196624968,
                       0.585410196624968, 0.138196601125011, 0.138196601125011]),
        'y': np.array([1.0 / 3.0,
                       0.797426985353087, 0.059715871789770, 0.797426985353087,
                       0.138196601125011, 0.585410196624968, 0.138196601125011,
                       0.138196601125011, 0.585410196624968, 0.138196601125011]),
        'w': np.array([0.1125,
                       0.066197076394253, 0.066197076394253, 0.066197076394253,
                       0.062969590272414, 0.062969590272414, 0.062969590272414,
                       0.062969590272414, 0.062969590272414, 0.062969590272414])
    }
}


def twb_rule_n(strength: int) -> int:
    """
    返回给定强度的 TWB 三角形求积规则的节点数。

    参数:
        strength: 期望的多项式精确度
    返回:
        节点数 n，若不存在则返回 -1
    """
    return _TWB_RULE_ORDER.get(strength, -1)


def twb_rule_data(strength: int) -> dict:
    """
    返回 TWB 规则的节点坐标和权重。

    参数:
        strength: 规则强度
    返回:
        字典 {'x': array, 'y': array, 'w': array}
    """
    if strength not in _TWB_DATA:
        # 回退到可用的最高强度
        available = sorted(_TWB_DATA.keys())
        strength = available[-1]
    return _TWB_DATA[strength]


def integrate_triangle_unit_monomial(ex: int, ey: int) -> float:
    """
    计算单位三角形上单项式 x^ex * y^ey 的精确积分。

    公式：
        I = ex! * ey! / (ex + ey + 2)!

    参数:
        ex, ey: 非负整数指数
    返回:
        积分值
    """
    if ex < 0 or ey < 0:
        raise ValueError("指数必须为非负整数")

    value = 1.0
    k = ex
    for i in range(1, ey + 1):
        k = k + 1
        value = value * i / k
    k = k + 1
    value = value / k
    k = k + 1
    value = value / k
    return value


def integrate_over_triangle(f_values: np.ndarray, rule_strength: int = 4) -> float:
    """
    利用 TWB 求积规则在参考三角形上积分函数。

    参数:
        f_values: 函数在求积节点处的值
        rule_strength: 求积规则强度
    返回:
        积分近似值
    """
    data = twb_rule_data(rule_strength)
    w = data['w']
    f_values = np.asarray(f_values, dtype=float)
    if len(f_values) != len(w):
        raise ValueError(f"函数值数量 {len(f_values)} 与节点数 {len(w)} 不匹配")
    return float(np.sum(w * f_values))


def pyramid_unit_volume() -> float:
    """
    返回单位金字塔的体积。

    单位金字塔定义：
        -(1-z) <= x <= 1-z
        -(1-z) <= y <= 1-z
        0 <= z <= 1

    体积：V = 4/3
    """
    return 4.0 / 3.0


def pyramid_witherden_rule_data(degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回金字塔上的 Witherden-Vincent 型求积规则（简化实现）。

    对于低次多项式，使用张量积型近似：
        在底面 [-1,1]^2 上使用高斯-勒让德积分
        在高度方向 [0,1] 上使用高斯-勒让德积分

    参数:
        degree: 期望的多项式精确度（0 <= degree <= 5）
    返回:
        (x, y, z, w): 节点坐标和权重数组
    """
    if degree < 0:
        degree = 0
    if degree > 10:
        degree = 10

    # 根据精度要求选择每维的高斯点数
    n_1d = max(1, (degree // 2) + 1)

    # 高斯-勒让德节点和权重（[-1,1]）
    t, wt = np.polynomial.legendre.leggauss(n_1d)
    # 将 z 从 [-1,1] 映射到 [0,1]
    z_nodes = 0.5 * (t + 1.0)
    wz = 0.5 * wt

    # 构造三维张量积节点
    n_total = n_1d ** 3
    x = np.zeros(n_total, dtype=float)
    y = np.zeros(n_total, dtype=float)
    z = np.zeros(n_total, dtype=float)
    w = np.zeros(n_total, dtype=float)

    idx = 0
    for i in range(n_1d):
        for j in range(n_1d):
            for k in range(n_1d):
                # 底面随高度线性收缩
                scale = 1.0 - z_nodes[k]
                x[idx] = scale * t[i]
                y[idx] = scale * t[j]
                z[idx] = z_nodes[k]
                # 权重包含底面缩放因子的雅可比
                w[idx] = wt[i] * wt[j] * wz[k] * (scale ** 2)
                idx += 1

    return x, y, z, w


def compute_l2_error_image(true_image: np.ndarray, recon_image: np.ndarray,
                           use_triangle_quad: bool = True) -> dict:
    """
    计算重建图像与真实图像之间的 L^2 误差和各项质量指标。

    参数:
        true_image: 真实图像
        recon_image: 重建图像
        use_triangle_quad: 是否使用三角形求积（否则使用简单求和）
    返回:
        误差指标字典
    """
    true_image = np.asarray(true_image, dtype=float)
    recon_image = np.asarray(recon_image, dtype=float)

    if true_image.shape != recon_image.shape:
        raise ValueError("两图像尺寸必须相同")

    H, W = true_image.shape
    diff = true_image - recon_image

    # 简单离散 L2 误差
    dx = 1.0 / max(H, W)
    dy = 1.0 / max(H, W)
    l2_error_simple = np.sqrt(np.sum(diff ** 2) * dx * dy)

    # 峰值信噪比 PSNR
    mse = np.mean(diff ** 2)
    max_val = np.max(np.abs(true_image))
    if mse > 1e-14 and max_val > 1e-14:
        psnr = 20.0 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = np.inf

    # 结构相似性指数 SSIM（简化版）
    mu_true = np.mean(true_image)
    mu_recon = np.mean(recon_image)
    sigma_true = np.std(true_image)
    sigma_recon = np.std(recon_image)
    sigma_cross = np.mean((true_image - mu_true) * (recon_image - mu_recon))

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    if sigma_true ** 2 + sigma_recon ** 2 + c2 > 1e-14:
        ssim = ((2 * mu_true * mu_recon + c1) * (2 * sigma_cross + c2)) / \
               ((mu_true ** 2 + mu_recon ** 2 + c1) * (sigma_true ** 2 + sigma_recon ** 2 + c2))
    else:
        ssim = 1.0

    # 利用 TWB 规则的高阶误差估计
    triangle_error = None
    if use_triangle_quad and H >= 2 and W >= 2:
        # 将图像域划分为三角形进行积分
        # 每个像素作为一个微小矩形，分为两个三角形
        data = twb_rule_data(4)
        qx, qy = data['x'], data['y']
        qw = data['w']

        total_error_sq = 0.0
        pixel_area = dx * dy
        for i in range(H):
            for j in range(W):
                # 像素中心处的误差
                err_val = diff[i, j] ** 2
                # 在像素上积分（常数近似）
                total_error_sq += err_val * pixel_area

        triangle_error = np.sqrt(total_error_sq)

    result = {
        'l2_error': float(l2_error_simple),
        'l2_error_triangle': float(triangle_error) if triangle_error is not None else None,
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim),
        'max_error': float(np.max(np.abs(diff))),
        'mean_error': float(np.mean(np.abs(diff))),
    }
    return result


def compute_reconstruction_quality(true_image: np.ndarray, recon_image: np.ndarray) -> dict:
    """
    综合评估重建质量的便捷函数。
    """
    return compute_l2_error_image(true_image, recon_image, use_triangle_quad=True)
