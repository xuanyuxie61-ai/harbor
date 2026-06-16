# -*- coding: utf-8 -*-
"""
splitting_kernels.py
====================

DGLAP 分裂核的高阶多项式表示与数值求值。

核心物理公式 (Leading Order DGLAP):

    P_{qq}(z) = C_F * [ (1 + z^2) / (1 - z)_+ + (3/2) delta(1 - z) ]
    P_{qg}(z) = T_R * [ z^2 + (1 - z)^2 ]
    P_{gq}(z) = C_F * [ (1 + (1 - z)^2) / z ]
    P_{gg}(z) = 2 C_A * [ z / (1 - z)_+ + (1 - z) / z + z (1 - z) ]
                + (beta_0 / 2) delta(1 - z)

其中 (+) 分布定义为:
    ∫_0^1 [f(z)]_+ g(z) dz = ∫_0^1 f(z) [g(z) - g(1)] dz

本模块将 P(z) 以正交多项式基展开, 从而与 893_polynomial 项目的
多项式代数系统直接对接:
    P(z) = sum_{k=0}^{K} a_k * phi_k(z)

采用 shifted Legendre 多项式 phi_k(z) = P_k(2z - 1) 作为基。
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable
import constants as C


# ======================================================================
# 多项式基工具 (来自 893_polynomial)
# ======================================================================
class Poly1D:
    """
    一元多项式类, 以系数列表 c[0] + c[1] x + c[2] x^2 + ... 存储。
    源自 893_polynomial 项目的 polynomial_add / polynomial_mul / polynomial_dif。
    """
    __slots__ = ('coeffs',)

    def __init__(self, coeffs: List[float]):
        if not coeffs:
            self.coeffs = [0.0]
        else:
            self.coeffs = list(coeffs)
        self._compress()

    def _compress(self) -> None:
        while len(self.coeffs) > 1 and abs(self.coeffs[-1]) < 1e-15:
            self.coeffs.pop()

    @property
    def degree(self) -> int:
        return max(0, len(self.coeffs) - 1)

    def eval(self, x: float) -> float:
        """Horner 求值"""
        s = 0.0
        for c in reversed(self.coeffs):
            s = s * x + c
        return s

    def __add__(self, other: 'Poly1D') -> 'Poly1D':
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0.0] * (n - len(self.coeffs))
        b = other.coeffs + [0.0] * (n - len(other.coeffs))
        return Poly1D([a[i] + b[i] for i in range(n)])

    def __mul__(self, other: 'Poly1D') -> 'Poly1D':
        if not self.coeffs or not other.coeffs:
            return Poly1D([0.0])
        n = len(self.coeffs) + len(other.coeffs) - 1
        r = [0.0] * n
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                r[i + j] += a * b
        return Poly1D(r)

    def __rmul__(self, scalar: float) -> 'Poly1D':
        return Poly1D([c * scalar for c in self.coeffs])

    def derivative(self) -> 'Poly1D':
        """多项式求导 (来自 893_polynomial 的 polynomial_dif)"""
        if len(self.coeffs) <= 1:
            return Poly1D([0.0])
        return Poly1D([i * c for i, c in enumerate(self.coeffs[1:], 1)])

    def integral(self, c0: float = 0.0) -> 'Poly1D':
        """不定积分 (常数项 c0)"""
        r = [c0] + [c / i for i, c in enumerate(self.coeffs, 1)]
        return Poly1D(r)

    def legendre_expand(self, order: int = 8) -> List[float]:
        """
        将多项式在 shifted Legendre 基 {P_k(2z-1)} 上展开,
        返回展开系数 a_k, 使用高斯-勒让德求积。
        """
        nodes, weights = gauss_legendre(order)
        ak = []
        for k in range(order):
            s = 0.0
            for i in range(order):
                z = nodes[i]
                p_k = shifted_legendre(k, z)
                s += weights[i] * self.eval(z) * p_k
            ak.append(s * (2.0 * k + 1.0))
        return ak

    def __repr__(self) -> str:
        terms = []
        for i, c in enumerate(self.coeffs):
            if abs(c) > 1e-15:
                if i == 0:
                    terms.append(f"{c:.6g}")
                elif i == 1:
                    terms.append(f"{c:.6g}*z")
                else:
                    terms.append(f"{c:.6g}*z^{i}")
        return " + ".join(terms) if terms else "0"


def gauss_legendre(n: int) -> Tuple[List[float], List[float]]:
    """n 点高斯-勒让德求积节点与权重, 映射到 [0,1]"""
    # 标准 GL 节点在 [-1,1], 映射到 [0,1]: z = (x+1)/2
    nodes_raw, w_raw = _gauss_legendre_raw(n)
    nodes = [(x + 1.0) / 2.0 for x in nodes_raw]
    weights = [w / 2.0 for w in w_raw]
    return nodes, weights


def _gauss_legendre_raw(n: int) -> Tuple[List[float], List[float]]:
    """标准 GL 节点在 [-1,1]"""
    nodes = []
    weights = []
    for i in range(n):
        # 初始猜测
        x = math.cos(PI * (i + 0.75) / (n + 0.5))
        for _ in range(50):
            p0, p1 = 1.0, x
            for j in range(2, n + 1):
                p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
                p0, p1 = p1, p2
            dp = n * (x * p1 - p0) / (x * x - 1.0 + 1e-300)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    return nodes, weights


PI = C.PI


def shifted_legendre(k: int, z: float) -> float:
    """shifted Legendre 多项式 P_k(2z-1), 递推计算"""
    x = 2.0 * z - 1.0
    if k == 0:
        return 1.0
    if k == 1:
        return x
    p0, p1 = 1.0, x
    for j in range(2, k + 1):
        p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
        p0, p1 = p1, p2
    return p1


# ======================================================================
# LO DGLAP 分裂核
# ======================================================================
def _plus_regularized(z: float, f: Callable[[float], float],
                      g1: Callable[[float], float] = lambda x: 1.0) -> float:
    """
    计算 [+] 正则化:
        [f(z)]_+ g(z) = f(z) * [g(z) - g(1)]
    用于处理 z -> 1 端的软发散。
    """
    return f(z) * (g1(z) - g1(1.0))


def P_qq_lo(z: float) -> float:
    """
    LO 夸克->夸克 分裂核:
        P_{qq}(z) = C_F * [ (1 + z^2) / (1 - z) - (1 - z) + (3/2) delta(1-z) ]

    正则化处理: 在 z < 1 时计算 regular 部分; z = 1 由 virtual 贡献。
    """
    if z <= C.Z_CUT or z >= 1.0 - C.Z_CUT:
        return 0.0
    reg = C.C_F * ((1.0 + z * z) / (1.0 - z) - (1.0 - z))
    return reg


def P_qg_lo(z: float) -> float:
    """
    LO 胶子->夸克 分裂核:
        P_{qg}(z) = T_R * [ z^2 + (1 - z)^2 ]
    有限, 无需正则化。
    """
    return C.T_R * (z * z + (1.0 - z) ** 2)


def P_gq_lo(z: float) -> float:
    """
    LO 夸克->胶子 分裂核:
        P_{gq}(z) = C_F * [ 1 + (1 - z)^2 ] / z
    """
    if z <= C.Z_CUT:
        return 0.0
    return C.C_F * (1.0 + (1.0 - z) ** 2) / z


def P_gg_lo(z: float) -> float:
    """
    LO 胶子->胶子 分裂核:
        P_{gg}(z) = 2 C_A [ z/(1-z) + (1-z)/z + z(1-z) ]

    正则化: z -> 0 和 z -> 1 两端均有发散。
    """
    if z <= C.Z_CUT or z >= 1.0 - C.Z_CUT:
        return 0.0
    reg = 2.0 * C.C_A * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))
    return reg


# ======================================================================
# Virtual + 分布 (+ 函数) 的 delta(1-z) 贡献
# ======================================================================
def virtual_coefficient_qq() -> float:
    """
    P_{qq} 的 virtual 部分 (来自 + 分布积分):
        gamma_{qq} = C_F * (3/2 + 2 ln(1 - z_cut))

    在小 z_cut 极限下, 该项主导自能修正。
    """
    return C.C_F * (1.5 + 2.0 * math.log(max(C.Z_CUT, 1e-10)))


def virtual_coefficient_gg(nf: int) -> float:
    """
    P_{gg} 的 virtual 部分:
        gamma_{gg} = (11 C_A - 4 T_R n_f) / 6 + 2 C_A ln(1 - z_cut)
    """
    return (11.0 * C.C_A - 4.0 * C.T_R * nf) / 6.0 + 2.0 * C.C_A * math.log(max(C.Z_CUT, 1e-10))


# ======================================================================
# NLO 修正项 (近似)
# ======================================================================
def P_qq_nlo_correction(z: float) -> float:
    """
    NLO 修正的主要项 (近似):
        P^{(1)}_{qq} ~ C_F^2 * [ -2 (1+z) ln(z) ln(1-z) / (1-z)_+
                    - (3/2) (1+z^2)/(1-z)_+ ln(1-z)
                    + ... ]

    此处只保留 leading soft-collinear 项。
    """
    if z <= C.Z_CUT or z >= 1.0 - C.Z_CUT:
        return 0.0
    cf2 = C.C_F * C.C_F
    soft_coll = -2.0 * (1.0 + z) * math.log(z) * math.log(1.0 - z) / (1.0 - z)
    return cf2 * soft_coll


# ======================================================================
# 多项式拟合与投影
# ======================================================================
def split_kernel_polynomial_projection(kernel_name: str, order: int = 8) -> Poly1D:
    """
    将指定分裂核以 Legendre 多项式投影, 返回 Poly1D 对象。
    这使分裂核可以与 893_polynomial 项目的多项式运算完全兼容。
    """
    kernels = {
        'Pqq': P_qq_lo,
        'Pqg': P_qg_lo,
        'Pgq': P_gq_lo,
        'Pgg': P_gg_lo,
    }
    if kernel_name not in kernels:
        raise ValueError(f"未知分裂核: {kernel_name}")
    f = kernels[kernel_name]

    # 高斯求积采样
    nodes, weights = gauss_legendre(order + 4)
    samples = [(z, f(z)) for z in nodes]

    # 最小二乘拟合 Legendre 基
    coeffs = [0.0] * (order + 1)
    for k in range(order + 1):
        s = 0.0
        for i, z in enumerate(nodes):
            s += weights[i] * samples[i][1] * shifted_legendre(k, z)
        coeffs[k] = s * (2.0 * k + 1.0)

    # 转换为标准幂基 (通过 Legendre -> monomial 变换)
    mono = _legendre_to_monomial(coeffs, order)
    return Poly1D(mono)


def _legendre_to_monomial(ak: List[float], order: int) -> List[float]:
    """
    将 shifted Legendre 展开系数 {a_k} 转换为标准幂基系数 {c_k}。
    P_k(2z-1) = sum_{j=0}^k T_{kj} z^j
    """
    mono = [0.0] * (order + 1)
    for k, a in enumerate(ak):
        T = _shifted_legendre_monomial_matrix_row(k)
        for j in range(len(T)):
            if j < len(mono):
                mono[j] += a * T[j]
    return mono


def _shifted_legendre_monomial_matrix_row(k: int) -> List[float]:
    """
    返回 shifted Legendre P_k(2z-1) 在幂基 {1, z, z^2, ..., z^k} 下的系数。
    利用 P_k(x) = (1/(2^k k!)) d^k/dx^k (x^2-1)^k 的显式展开。
    """
    # P_k(x) = sum_{m=0}^{floor(k/2)} (-1)^m C(2k-2m, k) C(k, m) x^{k-2m} / 2^k
    # 然后 x = 2z-1, 展开为 z 的幂次
    x_coeffs = [0.0] * (k + 1)
    for m in range(k // 2 + 1):
        binom1 = _binom(2 * k - 2 * m, k)
        binom2 = _binom(k, m)
        sign = (-1) ** m
        power = k - 2 * m
        x_coeffs[power] += sign * binom1 * binom2 / (2.0 ** k)

    # 现在 x_coeffs[j] 是 P_k(x) 中 x^j 的系数, x = 2z-1
    # (2z-1)^j = sum_{l=0}^j C(j,l) (2z)^l (-1)^{j-l}
    mono = [0.0] * (k + 1)
    for j in range(k + 1):
        if abs(x_coeffs[j]) < 1e-16:
            continue
        for l in range(j + 1):
            cl = _binom(j, l) * (2.0 ** l) * ((-1) ** (j - l))
            if l < len(mono):
                mono[l] += x_coeffs[j] * cl
    return mono


def _binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    r = 1.0
    for i in range(k):
        r = r * (n - i) / (i + 1)
    return r


# ======================================================================
# 分裂核的数值积分 (用于 shower 演化步长)
# ======================================================================
def integral_P_qq(zmin: float = 1e-3, zmax: float = 1.0 - 1e-3) -> float:
    """
    积分 int_{zmin}^{zmax} P_{qq}(z) dz
    用于计算 no-emission 概率。
    """
    n = 32
    nodes, weights = gauss_legendre(n)
    a, b = zmin, zmax
    s = 0.0
    for i in range(n):
        z = a + (b - a) * nodes[i]
        s += weights[i] * P_qq_lo(z)
    return (b - a) * s


def integral_P_gg(zmin: float = 1e-3, zmax: float = 1.0 - 1e-3) -> float:
    """int P_{gg}(z) dz"""
    n = 32
    nodes, weights = gauss_legendre(n)
    a, b = zmin, zmax
    s = 0.0
    for i in range(n):
        z = a + (b - a) * nodes[i]
        s += weights[i] * P_gg_lo(z)
    return (b - a) * s


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Splitting Kernel Polynomial Projection ===")
    for name in ['Pqq', 'Pqg', 'Pgq', 'Pgg']:
        poly = split_kernel_polynomial_projection(name, order=6)
        print(f"  {name}(z) ~ {poly}")

    print(f"\n=== Numerical Integrals ===")
    print(f"  int P_qq dz = {integral_P_qq():.6f}")
    print(f"  int P_gg dz = {integral_P_gg():.6f}")
    print(f"  virtual qq  = {virtual_coefficient_qq():.6f}")
    print(f"  virtual gg  = {virtual_coefficient_gg(5):.6f}")
