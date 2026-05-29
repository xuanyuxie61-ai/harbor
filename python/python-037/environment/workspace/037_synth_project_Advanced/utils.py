r"""
utils.py
通用数学工具与物理常数模块

包含：
- 物理常数（光速、普朗克常数、费米耦合常数等）
- 特殊函数：球贝塞尔函数 j_1(x)、误差函数、双阶乘
- 数值工具： bracket 搜索、向量工具、LCG 随机数生成器
- 高斯-埃尔米特求积节点与权重（Gauss-Hermite Quadrature）

参考文献：
- Helm, R. H. (1956). Phys. Rev. 104, 1466.
- Lewin, J. D., & Smith, P. F. (1996). Astroparticle Physics, 6, 87.
"""

import numpy as np
from typing import Tuple

# ============================================================================
# 物理常数（国际单位制与常用暗物质探测单位）
# ============================================================================

# 原子质量单位 [kg]
AMU_KG = 1.66053906660e-27

# 质子质量 [GeV/c^2]
M_PROTON_GEV = 0.938272

# 中子质量 [GeV/c^2]
M_NEUTRON_GEV = 0.939565

# 光速 [m/s]
C_M_S = 299792458.0

# 1 GeV/c^2 转 kg
GEV_TO_KG = 1.78266192e-27

# 1 keV 转焦耳
KEV_TO_JOULE = 1.602176634e-16

# 费米 [m]
FERMI_TO_M = 1.0e-15

# 暗物质局部密度 [GeV/cm^3]
RHO_LOCAL_GEV_CM3 = 0.3

# 太阳圆速度 [km/s]
V0_KM_S = 220.0

# 地球平均速度 [km/s]
VE_KM_S = 232.0

# 逃逸速度 [km/s]
VESC_KM_S = 544.0

# 玻尔兹曼常数 [J/K]
KB_J_K = 1.380649e-23

# 普朗克常数 [J·s]
H_PLANCK = 6.62607015e-34

# 约化普朗克常数 [J·s]
H_BAR = H_PLANCK / (2.0 * np.pi)

# 转换：km/s → m/s
KM_S_TO_M_S = 1.0e3

# 转换：GeV/c^2 → kg
def gev_to_kg(m_gev: float) -> float:
    """将质量从 GeV/c^2 转换为 kg。"""
    return m_gev * GEV_TO_KG


# ============================================================================
# 特殊函数
# ============================================================================

def spherical_bessel_j1(x: float) -> float:
    """
    一阶球贝塞尔函数 j_1(x)。

    公式：
        j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}

    当 x → 0 时，j_1(x) → x/3，采用泰勒展开避免数值抵消。
    """
    if abs(x) < 1.0e-8:
        return x / 3.0 - x**3 / 30.0 + x**5 / 840.0
    sx = np.sin(x)
    cx = np.cos(x)
    return sx / (x * x) - cx / x


def double_factorial(n: int) -> float:
    """
    双阶乘 n!! 。

    定义：
        n!! = n × (n-2) × (n-4) × ... × 1   (n 为奇数)
        n!! = n × (n-2) × (n-4) × ... × 2   (n 为偶数)
        0!! = 1!! = 1
    """
    if n < 0:
        raise ValueError("double_factorial: n 必须非负")
    result = 1.0
    while n > 1:
        result *= float(n)
        n -= 2
    return result


def erf_approx(x: float) -> float:
    """
    误差函数近似（Abramowitz & Stegun 公式 7.1.26）。

    公式：
        \operatorname{erf}(x) \approx 1 - (a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5) e^{-x^2}
        t = \frac{1}{1 + p x}, \quad p = 0.3275911
    """
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def erfc_approx(x: float) -> float:
    """互补误差函数：erfc(x) = 1 - erf(x)"""
    return 1.0 - erf_approx(x)


# ============================================================================
# 数值工具
# ============================================================================

def r8vec_bracket4(nx: int, x: np.ndarray, xval: float) -> int:
    """
    在有序数组 x 中定位 xval 所在的区间 [x[i], x[i+1]]。

    参数：
        nx: 数组长度
        x:  单调递增数组
        xval: 待查询点

    返回：
        i: 满足 x[i] <= xval <= x[i+1] 的左端点索引。
           若 xval < x[0]，返回 0；
           若 xval > x[-1]，返回 nx - 2。
    """
    if nx < 2:
        raise ValueError("r8vec_bracket4: 数组长度至少为 2")
    if xval <= x[0]:
        return 0
    if xval >= x[-1]:
        return nx - 2

    lo = 0
    hi = nx - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xval < x[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def r8_uniform_01(seed: int) -> Tuple[float, int]:
    """
    Park-Miller 最小标准线性同余生成器。

    递推公式：
        seed_{n+1} = (16807 \times seed_n) \mod (2^{31} - 1)

    返回均匀分布于 (0, 1) 的伪随机数及更新后的种子。
    """
    IA = 16807
    IM = 2147483647
    seed = (IA * seed) % IM
    if seed == 0:
        seed = 1
    r = seed / IM
    return r, seed


def r8vec_linspace(a: float, b: float, n: int) -> np.ndarray:
    """生成 [a, b] 上等距分布的 n 个点（含端点）。"""
    if n < 2:
        return np.array([a])
    return np.linspace(a, b, n)


# ============================================================================
# Gauss-Hermite 求积节点与权重（16 点规则，足够用于速度分布矩计算）
# ============================================================================

def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过特征值法计算 Gauss-Hermite 求积节点 x_i 与权重 w_i。

    理论背景：
        Hermite 多项式 H_n(x) 满足递推：
            H_{k+1}(x) = 2x H_k(x) - 2k H_{k-1}(x)
        其正交关系：
            \int_{-\infty}^{\infty} e^{-x^2} H_m(x) H_n(x) dx = \sqrt{\pi} 2^n n! \delta_{mn}

        Gauss-Hermite 求积公式：
            \int_{-\infty}^{\infty} e^{-x^2} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)

    实现：构造对称三对角矩阵（Golub-Welsch 算法），其特征值为节点，
          特征向量可导出权重。
    """
    if n < 1:
        raise ValueError("gauss_hermite_nodes_weights: n 必须 >= 1")
    if n == 1:
        return np.array([0.0]), np.array([np.sqrt(np.pi)])

    # 构造 Jacobi 矩阵（对称三对角）
    i = np.arange(1, n, dtype=float)
    beta = np.sqrt(i / 2.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)

    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    # 权重：w_i = sqrt(pi) * (v_i[0])^2
    w = np.sqrt(np.pi) * (eigvecs[0, :]) ** 2
    return x, w


def gauss_hermite_quadrature(f, n: int) -> float:
    """
    计算 \int_{-\infty}^{\infty} e^{-x^2} f(x) dx 的数值近似。

    参数：
        f: 被积函数，接受 ndarray 返回 ndarray
        n: 求积阶数

    返回：
        积分近似值
    """
    x, w = gauss_hermite_nodes_weights(n)
    return float(np.sum(w * f(x)))


# ============================================================================
# 三角函数数值积分辅助（Fekete 映射）
# ============================================================================

def triangle_area_2d(vertices: np.ndarray) -> float:
    """
    计算二维三角形有向面积。

    公式（行列式形式）：
        A = \frac{1}{2} \left| x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2) \right|
    """
    if vertices.shape != (3, 2):
        raise ValueError("triangle_area_2d: vertices 必须为 (3,2) 数组")
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


def barycentric_to_cartesian(tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """
    将重心坐标转换为笛卡尔坐标。

    公式：
        P = \lambda_1 V_1 + \lambda_2 V_2 + \lambda_3 V_3
    """
    return bary[:, 0:1] * tri[0] + bary[:, 1:2] * tri[1] + bary[:, 2:3] * tri[2]


# ============================================================================
# 自测代码
# ============================================================================

if __name__ == "__main__":
    # 测试 j_1
    assert abs(spherical_bessel_j1(0.0)) < 1e-12
    assert abs(spherical_bessel_j1(1.0) - 0.3011686789) < 1e-6

    # 测试双阶乘
    assert double_factorial(7) == 105.0
    assert double_factorial(8) == 384.0

    # 测试 Gauss-Hermite：积分 x^2 * e^{-x^2} dx = sqrt(pi)/2
    x, w = gauss_hermite_nodes_weights(16)
    integral = np.sum(w * x**2)
    assert abs(integral - np.sqrt(np.pi) / 2.0) < 1e-12

    # 测试 bracket
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert r8vec_bracket4(5, arr, 1.5) == 1
    assert r8vec_bracket4(5, arr, -1.0) == 0
    assert r8vec_bracket4(5, arr, 5.0) == 3

    print("utils.py: 所有自测通过")
