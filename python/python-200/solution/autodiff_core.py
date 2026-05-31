"""
autodiff_core.py
================
基于 Dual Number 的前向模式自动微分引擎（Forward-Mode AD Engine）。

核心数学原理
------------
引入无穷小量 ε 满足 ε² = 0，则对任意光滑函数 f，有
    f(x + ε) = f(x) + ε · f'(x)
通过重载基本算术运算，可在前向传播过程中同时求得函数值及其
一阶导数。对于高阶导数，采用嵌套 Dual Number（Hyper-Dual）
实现二阶导数的精确计算：

    x_dual = x + ε₁ x' + ε₂ x'' + ε₁ε₂ x'''

其中 ε₁² = ε₂² = (ε₁ε₂)² = 0，交叉项保留二阶混合偏导信息。

对于向量值函数 F: ℝⁿ → ℝᵐ，Jacobian 矩阵 J ∈ ℝ^{m×n} 的
第 j 列可通过将第 j 个输入设为 dual（其余为实数）一次前向
传播得到：

    J_{ij} = ∂F_i / ∂x_j

Hessian 向量积（HVP）则通过嵌套 dual 计算：

    H v = ∇²f(x) v

本模块同时提供标量/向量/矩阵级别的自动微分运算，为分子动力
学势能的高阶导数计算提供数值上无截断误差的精确梯度。
"""

import numpy as np
from typing import Union, List, Callable

Number = Union[int, float, np.ndarray]


class DualScalar:
    """
    标量 Dual Number，用于一阶导数计算。
    
    数学表示:  z = val + eps * der
    其中 eps² = 0。
    """
    __slots__ = ('val', 'der')

    def __init__(self, val: float, der: float = 0.0):
        self.val = float(val)
        self.der = float(der)

    # ------------------------------------------------------------------
    # 算术运算重载（严格遵循 dual number 代数规则）
    # ------------------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, DualScalar):
            return DualScalar(self.val + other.val, self.der + other.der)
        return DualScalar(self.val + other, self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, DualScalar):
            return DualScalar(self.val - other.val, self.der - other.der)
        return DualScalar(self.val - other, self.der)

    def __rsub__(self, other):
        return DualScalar(other - self.val, -self.der)

    def __mul__(self, other):
        if isinstance(other, DualScalar):
            # (a + εa')(b + εb') = ab + ε(ab' + a'b)
            return DualScalar(
                self.val * other.val,
                self.val * other.der + self.der * other.val
            )
        return DualScalar(self.val * other, self.der * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, DualScalar):
            # (a + εa')/(b + εb') = a/b + ε(a'b - ab')/b²
            denom = other.val * other.val
            if abs(denom) < 1e-30:
                raise ZeroDivisionError("Dual division by near-zero")
            return DualScalar(
                self.val / other.val,
                (self.der * other.val - self.val * other.der) / denom
            )
        if abs(other) < 1e-30:
            raise ZeroDivisionError("Division by near-zero scalar")
        return DualScalar(self.val / other, self.der / other)

    def __rtruediv__(self, other):
        # other / (a + εa') = other/a - ε·other·a'/a²
        if abs(self.val) < 1e-30:
            raise ZeroDivisionError("Division by near-zero dual")
        return DualScalar(
            other / self.val,
            -other * self.der / (self.val * self.val)
        )

    def __neg__(self):
        return DualScalar(-self.val, -self.der)

    def __pow__(self, power: float):
        # (a + εa')^n = a^n + ε·n·a^{n-1}·a'
        if self.val <= 0 and not float(power).is_integer():
            raise ValueError("Non-integer power of non-positive dual")
        val_new = self.val ** power
        if abs(self.val) < 1e-30 and power > 1:
            der_new = 0.0
        else:
            der_new = power * (self.val ** (power - 1)) * self.der
        return DualScalar(val_new, der_new)

    def __repr__(self):
        return f"DualScalar(val={self.val:.6g}, der={self.der:.6g})"


# ----------------------------------------------------------------------
# 数学函数重载（链式法则自动应用）
# ----------------------------------------------------------------------
def dual_sin(x: DualScalar) -> DualScalar:
    """sin(a + εa') = sin(a) + ε·cos(a)·a'"""
    return DualScalar(np.sin(x.val), np.cos(x.val) * x.der)


def dual_cos(x: DualScalar) -> DualScalar:
    """cos(a + εa') = cos(a) - ε·sin(a)·a'"""
    return DualScalar(np.cos(x.val), -np.sin(x.val) * x.der)


def dual_exp(x: DualScalar) -> DualScalar:
    """exp(a + εa') = exp(a) + ε·exp(a)·a'"""
    e = np.exp(x.val)
    return DualScalar(e, e * x.der)


def dual_log(x: DualScalar) -> DualScalar:
    """log(a + εa') = log(a) + ε·a'/a   (a > 0)"""
    if x.val <= 0:
        raise ValueError("log of non-positive dual")
    return DualScalar(np.log(x.val), x.der / x.val)


def dual_sqrt(x: DualScalar) -> DualScalar:
    """sqrt(a + εa') = sqrt(a) + ε·a'/(2·sqrt(a))"""
    if x.val < 0:
        raise ValueError("sqrt of negative dual")
    s = np.sqrt(x.val)
    if s < 1e-30:
        return DualScalar(0.0, 0.0)
    return DualScalar(s, x.der / (2.0 * s))


def dual_abs(x: DualScalar) -> DualScalar:
    """绝对值，在零点处使用次梯度 0"""
    if abs(x.val) < 1e-14:
        return DualScalar(0.0, 0.0)
    sign = 1.0 if x.val > 0 else -1.0
    return DualScalar(abs(x.val), sign * x.der)


def dual_min(x: DualScalar, y: DualScalar) -> DualScalar:
    """min 函数，在相等点取平均次梯度保证数值稳定"""
    if isinstance(y, (int, float)):
        y = DualScalar(y, 0.0)
    if x.val < y.val - 1e-12:
        return x
    elif y.val < x.val - 1e-12:
        return y
    else:
        # 接近相等时线性插值，避免导数跳跃
        alpha = 0.5
        return DualScalar(
            alpha * x.val + (1 - alpha) * y.val,
            alpha * x.der + (1 - alpha) * y.der
        )


# ======================================================================
# Hyper-Dual Number（嵌套 Dual）用于二阶导数
# ======================================================================

class HyperDualScalar:
    """
    Hyper-Dual Number 用于同时计算一阶和二阶导数。
    
    表示:  z = f0 + ε1 f1 + ε2 f2 + ε1ε2 f12
    
    其中：
        f0  = 函数值
        f1  = ∂f/∂x  (对第一个方向的导数)
        f2  = ∂f/∂y  (对第二个方向的导数)
        f12 = ∂²f/∂x∂y  (混合偏导)
    
    当 f1 = f2 时，f12 给出 Hessian 对角元信息。
    """
    __slots__ = ('f0', 'f1', 'f2', 'f12')

    def __init__(self, f0: float, f1: float = 0.0, f2: float = 0.0, f12: float = 0.0):
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.f2 = float(f2)
        self.f12 = float(f12)

    def __add__(self, other):
        if isinstance(other, HyperDualScalar):
            return HyperDualScalar(
                self.f0 + other.f0, self.f1 + other.f1,
                self.f2 + other.f2, self.f12 + other.f12
            )
        return HyperDualScalar(self.f0 + other, self.f1, self.f2, self.f12)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, HyperDualScalar):
            return HyperDualScalar(
                self.f0 - other.f0, self.f1 - other.f1,
                self.f2 - other.f2, self.f12 - other.f12
            )
        return HyperDualScalar(self.f0 - other, self.f1, self.f2, self.f12)

    def __rsub__(self, other):
        return HyperDualScalar(other - self.f0, -self.f1, -self.f2, -self.f12)

    def __mul__(self, other):
        if isinstance(other, HyperDualScalar):
            # (f0 + ε1f1 + ε2f2 + ε1ε2f12)(g0 + ε1g1 + ε2g2 + ε1ε2g12)
            # = f0g0 + ε1(f0g1 + f1g0) + ε2(f0g2 + f2g0)
            #   + ε1ε2(f0g12 + f1g2 + f2g1 + f12g0)
            return HyperDualScalar(
                self.f0 * other.f0,
                self.f0 * other.f1 + self.f1 * other.f0,
                self.f0 * other.f2 + self.f2 * other.f0,
                self.f0 * other.f12 + self.f1 * other.f2
                + self.f2 * other.f1 + self.f12 * other.f0
            )
        return HyperDualScalar(self.f0 * other, self.f1 * other,
                               self.f2 * other, self.f12 * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, HyperDualScalar):
            g0 = other.f0
            if abs(g0) < 1e-30:
                raise ZeroDivisionError("HyperDual division by near-zero")
            inv_g0 = 1.0 / g0
            inv_g0_sq = inv_g0 * inv_g0
            return HyperDualScalar(
                self.f0 * inv_g0,
                (self.f1 * g0 - self.f0 * other.f1) * inv_g0_sq,
                (self.f2 * g0 - self.f0 * other.f2) * inv_g0_sq,
                (self.f12 * g0 - self.f1 * other.f2 - self.f2 * other.f1
                 - self.f0 * other.f12 + 2 * self.f0 * other.f1 * other.f2 * inv_g0) * inv_g0_sq
            )
        if abs(other) < 1e-30:
            raise ZeroDivisionError("Division by near-zero")
        inv = 1.0 / other
        return HyperDualScalar(self.f0 * inv, self.f1 * inv,
                               self.f2 * inv, self.f12 * inv)

    def __rtruediv__(self, other):
        g0 = self.f0
        if abs(g0) < 1e-30:
            raise ZeroDivisionError("Division by near-zero hyper-dual")
        inv_g0 = 1.0 / g0
        inv_g0_sq = inv_g0 * inv_g0
        return HyperDualScalar(
            other * inv_g0,
            -other * self.f1 * inv_g0_sq,
            -other * self.f2 * inv_g0_sq,
            other * (2 * self.f1 * self.f2 * inv_g0 - self.f12) * inv_g0_sq
        )

    def __neg__(self):
        return HyperDualScalar(-self.f0, -self.f1, -self.f2, -self.f12)

    def __pow__(self, power: float):
        if self.f0 <= 0 and not float(power).is_integer():
            raise ValueError("Non-integer power of non-positive hyper-dual")
        f0p = self.f0 ** power
        if abs(self.f0) < 1e-30 and power > 1:
            return HyperDualScalar(f0p, 0.0, 0.0, 0.0)
        c1 = power * (self.f0 ** (power - 1))
        c2 = power * (power - 1) * (self.f0 ** (power - 2))
        return HyperDualScalar(
            f0p,
            c1 * self.f1,
            c1 * self.f2,
            c2 * self.f1 * self.f2 + c1 * self.f12
        )

    def __repr__(self):
        return (f"HyperDual(f0={self.f0:.6g}, f1={self.f1:.6g}, "
                f"f2={self.f2:.6g}, f12={self.f12:.6g})")


def hdual_sin(x: HyperDualScalar) -> HyperDualScalar:
    s, c = np.sin(x.f0), np.cos(x.f0)
    return HyperDualScalar(
        s, c * x.f1, c * x.f2,
        -s * x.f1 * x.f2 + c * x.f12
    )


def hdual_cos(x: HyperDualScalar) -> HyperDualScalar:
    s, c = np.sin(x.f0), np.cos(x.f0)
    return HyperDualScalar(
        c, -s * x.f1, -s * x.f2,
        -c * x.f1 * x.f2 - s * x.f12
    )


def hdual_exp(x: HyperDualScalar) -> HyperDualScalar:
    e = np.exp(x.f0)
    return HyperDualScalar(
        e, e * x.f1, e * x.f2,
        e * x.f1 * x.f2 + e * x.f12
    )


def hdual_sqrt(x: HyperDualScalar) -> HyperDualScalar:
    if x.f0 < 0:
        raise ValueError("sqrt of negative hyper-dual")
    s = np.sqrt(x.f0)
    if s < 1e-30:
        return HyperDualScalar(0.0, 0.0, 0.0, 0.0)
    inv_s = 1.0 / s
    return HyperDualScalar(
        s, 0.5 * inv_s * x.f1, 0.5 * inv_s * x.f2,
        (-0.25 * inv_s ** 3) * x.f1 * x.f2 + 0.5 * inv_s * x.f12
    )


# =======================================================================
# 向量级自动微分接口
# =======================================================================

def jacobian_vector_func(func: Callable, x: np.ndarray,
                         h: float = 1e-7) -> np.ndarray:
    """
    使用有限差分作为备用方法计算向量值函数的 Jacobian。
    在自动微分无法直接应用的复合场景中使用。
    
    J_{ij} = ∂f_i / ∂x_j
    
    采用中心差分：
        ∂f/∂x_j ≈ [f(x + h·e_j) - f(x - h·e_j)] / (2h)
    """
    n = len(x)
    f0 = func(x)
    m = len(f0) if hasattr(f0, '__len__') else 1
    J = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        J[:, j] = (np.asarray(f_plus) - np.asarray(f_minus)) / (2.0 * h)
    return J


def grad_scalar_func_ad(func: Callable, x: np.ndarray) -> np.ndarray:
    """
    使用前向模式自动微分计算标量函数的梯度。
    
    对输入向量的每个分量构造 dual number，分别前向传播，
    收集输出 dual 的 der 部分即为梯度分量。
    
    ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂x_n]ᵀ
    """
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_dual = [DualScalar(float(xv), 0.0) for xv in x]
        x_dual[i] = DualScalar(float(x[i]), 1.0)
        result = func(x_dual)
        if isinstance(result, DualScalar):
            grad[i] = result.der
        else:
            grad[i] = 0.0
    return grad


def hessian_scalar_func_fd(func: Callable, x: np.ndarray,
                           h: float = 1e-5) -> np.ndarray:
    """
    使用有限差分计算标量函数的 Hessian 矩阵。
    
    H_{ij} = ∂²f / ∂x_i ∂x_j
    
    采用二阶中心差分公式：
        ∂²f/∂x_i∂x_j ≈ [f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)] / (4h²)
    """
    n = len(x)
    H = np.zeros((n, n))
    f_base = func(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                x_pp = x.copy()
                x_mm = x.copy()
                x_pp[i] += h
                x_mm[i] -= h
                H[i, i] = (func(x_pp) - 2 * f_base + func(x_mm)) / (h * h)
            else:
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h
                H[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4.0 * h * h)
                H[j, i] = H[i, j]
    return H


def directional_derivative_ad(func: Callable, x: np.ndarray,
                               direction: np.ndarray) -> float:
    """
    使用单一路向模式自动微分计算方向导数。
    
    D_v f(x) = ∇f(x) · v = ∂f/∂x_i · v_i
    
    将 x 的每个分量设为 x_i + ε·v_i，则输出 dual 的 der 即为方向导数。
    """
    x_dual = [DualScalar(float(xv), float(vv))
              for xv, vv in zip(x, direction)]
    result = func(x_dual)
    return float(result.der) if isinstance(result, DualScalar) else 0.0


def mixed_partial_hyperdual(func: Callable, x: np.ndarray,
                            i: int, j: int) -> float:
    """
    使用 Hyper-Dual Number 计算标量函数对 x_i 和 x_j 的混合偏导数。
    
    ∂²f / ∂x_i ∂x_j
    
    构造嵌套 dual：x_i 方向上 ε1=1，x_j 方向上 ε2=1，
    输出 dual 的 f12 分量即为混合偏导。
    """
    x_hd = []
    for k, xv in enumerate(x):
        if k == i and k == j:
            x_hd.append(HyperDualScalar(float(xv), 1.0, 1.0, 0.0))
        elif k == i:
            x_hd.append(HyperDualScalar(float(xv), 1.0, 0.0, 0.0))
        elif k == j:
            x_hd.append(HyperDualScalar(float(xv), 0.0, 1.0, 0.0))
        else:
            x_hd.append(HyperDualScalar(float(xv), 0.0, 0.0, 0.0))
    result = func(x_hd)
    return float(result.f12) if isinstance(result, HyperDualScalar) else 0.0
