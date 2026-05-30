
import numpy as np
from typing import Tuple, Optional




_TWB_RULE_ORDER = {
    1: 1, 2: 3, 4: 6, 5: 10, 7: 15, 9: 21,
    11: 28, 13: 36, 14: 45, 16: 55, 18: 66,
    20: 78, 21: 91, 23: 105, 25: 120
}


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
    return _TWB_RULE_ORDER.get(strength, -1)


def twb_rule_data(strength: int) -> dict:
    if strength not in _TWB_DATA:

        available = sorted(_TWB_DATA.keys())
        strength = available[-1]
    return _TWB_DATA[strength]


def integrate_triangle_unit_monomial(ex: int, ey: int) -> float:
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
    data = twb_rule_data(rule_strength)
    w = data['w']
    f_values = np.asarray(f_values, dtype=float)
    if len(f_values) != len(w):
        raise ValueError(f"函数值数量 {len(f_values)} 与节点数 {len(w)} 不匹配")
    return float(np.sum(w * f_values))


def pyramid_unit_volume() -> float:
    return 4.0 / 3.0


def pyramid_witherden_rule_data(degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if degree < 0:
        degree = 0
    if degree > 10:
        degree = 10


    n_1d = max(1, (degree // 2) + 1)


    t, wt = np.polynomial.legendre.leggauss(n_1d)

    z_nodes = 0.5 * (t + 1.0)
    wz = 0.5 * wt


    n_total = n_1d ** 3
    x = np.zeros(n_total, dtype=float)
    y = np.zeros(n_total, dtype=float)
    z = np.zeros(n_total, dtype=float)
    w = np.zeros(n_total, dtype=float)

    idx = 0
    for i in range(n_1d):
        for j in range(n_1d):
            for k in range(n_1d):

                scale = 1.0 - z_nodes[k]
                x[idx] = scale * t[i]
                y[idx] = scale * t[j]
                z[idx] = z_nodes[k]

                w[idx] = wt[i] * wt[j] * wz[k] * (scale ** 2)
                idx += 1

    return x, y, z, w


def compute_l2_error_image(true_image: np.ndarray, recon_image: np.ndarray,
                           use_triangle_quad: bool = True) -> dict:
    true_image = np.asarray(true_image, dtype=float)
    recon_image = np.asarray(recon_image, dtype=float)

    if true_image.shape != recon_image.shape:
        raise ValueError("两图像尺寸必须相同")

    H, W = true_image.shape
    diff = true_image - recon_image


    dx = 1.0 / max(H, W)
    dy = 1.0 / max(H, W)
    l2_error_simple = np.sqrt(np.sum(diff ** 2) * dx * dy)


    mse = np.mean(diff ** 2)
    max_val = np.max(np.abs(true_image))
    if mse > 1e-14 and max_val > 1e-14:
        psnr = 20.0 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = np.inf


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


    triangle_error = None
    if use_triangle_quad and H >= 2 and W >= 2:


        data = twb_rule_data(4)
        qx, qy = data['x'], data['y']
        qw = data['w']

        total_error_sq = 0.0
        pixel_area = dx * dy
        for i in range(H):
            for j in range(W):

                err_val = diff[i, j] ** 2

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
    return compute_l2_error_image(true_image, recon_image, use_triangle_quad=True)
