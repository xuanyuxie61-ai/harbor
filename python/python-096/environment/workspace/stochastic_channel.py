
import numpy as np
from typing import Tuple, Optional


def alnorm(x: float, upper: bool = False) -> float:
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
    if rho <= -1.0 + 1e-12:
        return max(0.0, alnorm(h, False) + alnorm(k, False) - 1.0)
    if rho >= 1.0 - 1e-12:
        return min(alnorm(h, False), alnorm(k, False))



    Phi_h = alnorm(h, False)
    Phi_k = alnorm(k, False)

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

    def __init__(self, step_delta: float = 0.01, seed: Optional[int] = None):
        self.step_delta = step_delta
        if seed is not None:
            np.random.seed(seed)

    def simulate(self, step_num: int, walk_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return n * self.step_delta ** 2


def set_discrete_cdf_2d(pdf_mat: np.ndarray) -> np.ndarray:
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

    r = np.sqrt(samples[0, :] ** 2 + samples[1, :] ** 2)
    gains = np.exp(-r / correlation_length)
    return gains


def sidelobe_level_cdf(level_db: float, n_elements: int,
                       array_factor_std: float = 1.0) -> float:

    mean_sll = -10.0 * np.log10(max(n_elements, 1))
    z = (level_db - mean_sll) / array_factor_std
    return alnorm(z, False)
