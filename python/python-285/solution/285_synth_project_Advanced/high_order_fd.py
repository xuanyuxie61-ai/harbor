"""
高阶有限差分算子模块
=====================
对应种子项目: 088_biharmonic_fd1d (双调和有限差分 → 高阶梯度能离散)
              434_fisher_pde_ftcs (FTCS 格式 → 时间推进框架)

物理背景:
    LGD 自由能泛函中的梯度能项涉及极化的二阶和四阶空间导数:
        f_G ~ G₁₁·(∂Pᵢ/∂xᵢ)² + G₁₂·(∂Pᵢ/∂xⱼ)²

    有效场的变分导数包含极化的拉普拉斯项:
        δF/δPₖ ~ -G·∇²Pₖ + ...

    对于高阶 LGD 理论, 还涉及双调和算子:
        δF_higher/δPₖ ~ κ·∇⁴Pₖ

    本模块实现:
        - 2阶/4阶/6阶中心差分 (一阶导数)
        - 2阶/4阶/6阶中心差分 (二阶导数)
        - 紧致 Laplacian (2D)
        - 双调和算子 (∇⁴ = ∂⁴/∂x⁴ + 2∂⁴/∂x²∂y² + ∂⁴/∂y⁴)

有限差分公式:
    二阶一阶导数 (O(h²)):
        f'(x) ≈ (-f(x+h) + f(x-h)) / (2h)

    四阶一阶导数 (O(h⁴)):
        f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)

    六阶一阶导数 (O(h⁶)):
        f'(x) ≈ (-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h)
                 - 9f(x+2h) + f(x+3h)) / (60h)

    二阶二阶导数 (O(h²)):
        f''(x) ≈ (f(x-h) - 2f(x) + f(x+h)) / h²

    四阶二阶导数 (O(h⁴)):
        f''(x) ≈ (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h)
                  - f(x+2h)) / (12h²)

    六阶二阶导数 (O(h⁶)):
        f''(x) ≈ (2f(x-3h) - 27f(x-2h) + 270f(x-h) - 495f(x)
                  + 270f(x+h) - 27f(x+2h) + 2f(x-3h)) / (180h²)

    双调和 (1D, O(h²)):
        f''''(x) ≈ (f(x-2h) - 4f(x-h) + 6f(x) - 4f(x+h)
                    + f(x+2h)) / h⁴

稳定性分析:
    FTCS 格式的 von Neumann 稳定性条件:
        dt ≤ h² / (2·L·G·(1 + cos(kh)))  ∀k
    最严格约束: dt ≤ h² / (4·L·G)

    高阶格式减小了截断误差但可能收紧稳定性约束.
    六阶格式的谱半径:
        λ_max ~ 4G/h² · (1 + 1/9 + 1/90 + ...) ≈ 4.24G/h²
"""

import numpy as np
from scipy import sparse


# ============================================================
# 一阶导数算子
# ============================================================

def first_derivative_2nd_order(f, h, axis=0):
    """
    二阶精度一阶导数 (中心差分).

    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    截断误差: O(h²)
    模板: [-1/2, 0, 1/2] / h

    参数:
        f: 输入数组 (2D)
        h: 网格间距
        axis: 微分方向 (0=x, 1=y)

    返回:
        df: 一阶导数, shape 同 f
    """
    return np.gradient(f, h, axis=axis)


def first_derivative_4th_order(f, h, axis=0):
    """
    四阶精度一阶导数 (中心差分).

    f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)

    截断误差: O(h⁴)
    模板: [1, -8, 0, 8, -1] / (12h)

    参数:
        f: 输入数组 (2D)
        h: 网格间距
        axis: 微分方向

    返回:
        df: 一阶导数
    """
    # 使用滚动实现周期性边界
    if axis == 0:
        fm2 = np.roll(f, 2, axis=0)
        fm1 = np.roll(f, 1, axis=0)
        fp1 = np.roll(f, -1, axis=0)
        fp2 = np.roll(f, -2, axis=0)
    else:
        fm2 = np.roll(f, 2, axis=1)
        fm1 = np.roll(f, 1, axis=1)
        fp1 = np.roll(f, -1, axis=1)
        fp2 = np.roll(f, -2, axis=1)

    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)


def sixth_order_first_derivative(f, h, axis=0):
    """
    六阶精度一阶导数 (中心差分).

    f'(x) ≈ (-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h)
             - 9f(x+2h) + f(x+3h)) / (60h)

    截断误差: O(h⁶)
    模板: [-1, 9, -45, 0, 45, -9, 1] / (60h)

    参数:
        f: 输入数组 (2D)
        h: 网格间距
        axis: 微分方向

    返回:
        df: 一阶导数
    """
    if axis == 0:
        fm3 = np.roll(f, 3, axis=0)
        fm2 = np.roll(f, 2, axis=0)
        fm1 = np.roll(f, 1, axis=0)
        fp1 = np.roll(f, -1, axis=0)
        fp2 = np.roll(f, -2, axis=0)
        fp3 = np.roll(f, -3, axis=0)
    else:
        fm3 = np.roll(f, 3, axis=1)
        fm2 = np.roll(f, 2, axis=1)
        fm1 = np.roll(f, 1, axis=1)
        fp1 = np.roll(f, -1, axis=1)
        fp2 = np.roll(f, -2, axis=1)
        fp3 = np.roll(f, -3, axis=1)

    coeff = 1.0 / (60.0 * h)
    return coeff * (-fm3 + 9.0 * fm2 - 45.0 * fm1 +
                    45.0 * fp1 - 9.0 * fp2 + fp3)


def sixth_order_derivative(f, h, axis=0, order=1):
    """
    通用高阶导数接口.

    参数:
        f: 输入数组
        h: 网格间距
        axis: 微分方向
        order: 导数阶数 (1 或 2)

    返回:
        导数数组
    """
    if order == 1:
        return sixth_order_first_derivative(f, h, axis)
    elif order == 2:
        return sixth_order_second_derivative(f, h, axis)
    else:
        raise ValueError(f"不支持的导数阶数: {order}")


# ============================================================
# 二阶导数算子
# ============================================================

def second_derivative_2nd_order(f, h, axis=0):
    """
    二阶精度二阶导数.

    f''(x) ≈ (f(x-h) - 2f(x) + f(x+h)) / h²

    截断误差: O(h²)
    模板: [1, -2, 1] / h²
    """
    if axis == 0:
        fm = np.roll(f, 1, axis=0)
        fp = np.roll(f, -1, axis=0)
    else:
        fm = np.roll(f, 1, axis=1)
        fp = np.roll(f, -1, axis=1)

    return (fm - 2.0 * f + fp) / (h * h)


def second_derivative_4th_order(f, h, axis=0):
    """
    四阶精度二阶导数.

    f''(x) ≈ (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h)
              - f(x+2h)) / (12h²)

    截断误差: O(h⁴)
    模板: [-1, 16, -30, 16, -1] / (12h²)
    """
    if axis == 0:
        fm2 = np.roll(f, 2, axis=0)
        fm1 = np.roll(f, 1, axis=0)
        fp1 = np.roll(f, -1, axis=0)
        fp2 = np.roll(f, -2, axis=0)
    else:
        fm2 = np.roll(f, 2, axis=1)
        fm1 = np.roll(f, 1, axis=1)
        fp1 = np.roll(f, -1, axis=1)
        fp2 = np.roll(f, -2, axis=1)

    return (-fm2 + 16.0 * fm1 - 30.0 * f + 16.0 * fp1 - fp2) / (12.0 * h * h)


def sixth_order_second_derivative(f, h, axis=0):
    """
    六阶精度二阶导数.

    标准中心差分公式 (O(h⁶)):
        f''(x) ≈ [2f(x-3h) - 27f(x-2h) + 270f(x-h) - 490f(x)
                  + 270f(x+h) - 27f(x+2h) + 2f(x+3h)] / (180h²)

    等价形式 (分子分母同乘 7, 便于整系数):
        = [14f(x-3h) - 189f(x-2h) + 1890f(x-h) - 3430f(x)
           + 1890f(x+h) - 189f(x+2h) + 14f(x+3h)] / (1260h²)

    截断误差: O(h⁶), 首项误差 (h⁶/1120)·f⁽⁸⁾(x).

    推导: 由 Taylor 展开匹配 f'', f⁽⁴⁾, f⁽⁶⁾ 项.
    设 f''(x) ≈ (1/h²) Σ c_k f(x+kh), k ∈ {-3,-2,-1,0,1,2,3},
    由对称性 c_{-k} = c_k, 待定 c_0, c_1, c_2, c_3 满足:
        2(c_1 + 4c_2 + 9c_3) = 2       (f'' 项)
        2(c_1 + 16c_2 + 81c_3) = 0     (f⁽⁴⁾ 项)
        2(c_1 + 81c_2 + 729c_3) = 0    (f⁽⁶⁾ 项)
    解得: c_3 = 2/180, c_2 = -27/180, c_1 = 270/180, c_0 = -490/180.

    谱半径:
        λ(kh) = (2/180h²)[2cos(3kh) - 27cos(2kh) + 270cos(kh) - 245]
        max|λ| ≈ 2.72/h²
    """
    if axis == 0:
        fm3 = np.roll(f, 3, axis=0)
        fm2 = np.roll(f, 2, axis=0)
        fm1 = np.roll(f, 1, axis=0)
        fp1 = np.roll(f, -1, axis=0)
        fp2 = np.roll(f, -2, axis=0)
        fp3 = np.roll(f, -3, axis=0)
    else:
        fm3 = np.roll(f, 3, axis=1)
        fm2 = np.roll(f, 2, axis=1)
        fm1 = np.roll(f, 1, axis=1)
        fp1 = np.roll(f, -1, axis=1)
        fp2 = np.roll(f, -2, axis=1)
        fp3 = np.roll(f, -3, axis=1)

    coeff = 1.0 / (180.0 * h * h)
    return coeff * (2.0 * fm3 - 27.0 * fm2 + 270.0 * fm1 - 490.0 * f +
                    270.0 * fp1 - 27.0 * fp2 + 2.0 * fp3)


# ============================================================
# 2D Laplacian 算子
# ============================================================

def compact_laplacian_2d(f, dx, dy, fd_order=6, component=None):
    """
    2D 紧致 Laplacian 算子 ∇²f = ∂²f/∂x² + ∂²f/∂y².

    参数:
        f: 2D 数组, shape (nx, ny)
        dx, dy: 网格间距
        fd_order: 精度阶数 (2, 4, 6)
        component: 若指定 'x' 仅返回 ∂²f/∂x², 'y' 仅返回 ∂²f/∂y²

    返回:
        lap: ∇²f, shape (nx, ny)
    """
    if fd_order == 2:
        d2f_dx2 = second_derivative_2nd_order(f, dx, axis=0)
        d2f_dy2 = second_derivative_2nd_order(f, dy, axis=1)
    elif fd_order == 4:
        d2f_dx2 = second_derivative_4th_order(f, dx, axis=0)
        d2f_dy2 = second_derivative_4th_order(f, dy, axis=1)
    elif fd_order == 6:
        d2f_dx2 = sixth_order_second_derivative(f, dx, axis=0)
        d2f_dy2 = sixth_order_second_derivative(f, dy, axis=1)
    else:
        raise ValueError(f"不支持的精度阶数: {fd_order}")

    if component == 'x':
        return d2f_dx2
    elif component == 'y':
        return d2f_dy2
    return d2f_dx2 + d2f_dy2


# ============================================================
# 双调和算子 (∇⁴)
# ============================================================

def biharmonic_1d_sparse(n, h):
    """
    1D 双调和算子 ∂⁴f/∂x⁴ 的稀疏矩阵表示.

    采用五点模板 (O(h²)):
        f''''(x) ≈ (f(x-2h) - 4f(x-h) + 6f(x) - 4f(x+h)
                    + f(x+2h)) / h⁴

    对应种子项目 088 (biharmonic_fd1d):
    将 1D 双调和方程求解器推广到高阶精度.

    参数:
        n: 网格点数
        h: 网格间距

    返回:
        D4: 稀疏矩阵, shape (n, n)
    """
    diags = np.zeros((5, n))
    coeff = 1.0 / (h ** 4)

    # 主对角线: 6/h⁴
    diags[0, :] = 6.0 * coeff
    # 上下第一条: -4/h⁴
    diags[1, :-1] = -4.0 * coeff
    diags[2, :-1] = -4.0 * coeff
    # 上下第二条: 1/h⁴
    diags[3, :-2] = 1.0 * coeff
    diags[4, :-2] = 1.0 * coeff

    D4 = sparse.diags(diags, [0, -1, 1, -2, 2], format='csc')

    return D4


def biharmonic_2d(f, dx, dy, fd_order=2):
    """
    2D 双调和算子 ∇⁴f.

    ∇⁴f = ∂⁴f/∂x⁴ + 2∂⁴f/∂x²∂y² + ∂⁴f/∂y⁴

    对于 LGD 泛函中的高阶梯度项, 双调和算子出现在:
        δ(∇P)⁴/δP ~ ∇⁴P

    参数:
        f: 2D 数组, shape (nx, ny)
        dx, dy: 网格间距
        fd_order: 精度阶数

    返回:
        nabla4_f: ∇⁴f
    """
    # ∂⁴f/∂x⁴
    d4f_dx4 = _fourth_derivative_x(f, dx, fd_order)
    # ∂⁴f/∂y⁴
    d4f_dy4 = _fourth_derivative_y(f, dy, fd_order)
    # 2∂⁴f/∂x²∂y² (混合导数)
    d4f_dxdy = _mixed_fourth_derivative(f, dx, dy, fd_order)

    return d4f_dx4 + 2.0 * d4f_dxdy + d4f_dy4


def _fourth_derivative_x(f, dx, fd_order=2):
    """∂⁴f/∂x⁴, 沿 x 方向的四阶导数."""
    fm2 = np.roll(f, 2, axis=0)
    fm1 = np.roll(f, 1, axis=0)
    fp1 = np.roll(f, -1, axis=0)
    fp2 = np.roll(f, -2, axis=0)

    if fd_order == 2:
        return (fm2 - 4.0 * fm1 + 6.0 * f - 4.0 * fp1 + fp2) / (dx ** 4)
    else:
        # O(h⁴) 精度
        fm3 = np.roll(f, 3, axis=0)
        fp3 = np.roll(f, -3, axis=0)
        return (-fm3 + 12.0 * fm2 - 39.0 * fm1 + 56.0 * f -
                39.0 * fp1 + 12.0 * fp2 - fp3) / (6.0 * dx ** 4)


def _fourth_derivative_y(f, dy, fd_order=2):
    """∂⁴f/∂y⁴, 沿 y 方向的四阶导数."""
    fm2 = np.roll(f, 2, axis=1)
    fm1 = np.roll(f, 1, axis=1)
    fp1 = np.roll(f, -1, axis=1)
    fp2 = np.roll(f, -2, axis=1)

    if fd_order == 2:
        return (fm2 - 4.0 * fm1 + 6.0 * f - 4.0 * fp1 + fp2) / (dy ** 4)
    else:
        fm3 = np.roll(f, 3, axis=1)
        fp3 = np.roll(f, -3, axis=1)
        return (-fm3 + 12.0 * fm2 - 39.0 * fm1 + 56.0 * f -
                39.0 * fp1 + 12.0 * fp2 - fp3) / (6.0 * dy ** 4)


def _mixed_fourth_derivative(f, dx, dy, fd_order=2):
    """
    混合四阶导数 ∂⁴f/∂x²∂y².

    采用二阶模板:
        ∂⁴f/∂x²∂y² ≈ [f(i+1,j+1) - 2f(i+1,j) + f(i+1,j-1)
                      - 2f(i,j+1) + 4f(i,j) - 2f(i,j-1)
                      + f(i-1,j+1) - 2f(i-1,j) + f(i-1,j-1)] / (dx²dy²)
    """
    fpp = np.roll(np.roll(f, -1, axis=0), -1, axis=1)
    fp0 = np.roll(f, -1, axis=0)
    fpm = np.roll(np.roll(f, -1, axis=0), 1, axis=1)
    f0p = np.roll(f, -1, axis=1)
    f0m = np.roll(f, 1, axis=1)
    fmp = np.roll(np.roll(f, 1, axis=0), -1, axis=1)
    fm0 = np.roll(f, 1, axis=0)
    fmm = np.roll(np.roll(f, 1, axis=0), 1, axis=1)

    return (fpp - 2.0 * fp0 + fpm - 2.0 * f0p + 4.0 * f -
            2.0 * f0m + fmp - 2.0 * fm0 + fmm) / (dx * dx * dy * dy)


# ============================================================
# 稳定性分析工具
# ============================================================

def von_neumann_stability_limit(G, L, dx, dy, fd_order=6):
    """
    von Neumann 稳定性分析给出的最大时间步长.

    对于扩散型方程 ∂P/∂t = L·G·∇²P:
        dt_max = 1 / (L · G · λ_max)

    其中 λ_max 是离散 Laplacian 的最大特征值.

    对于 O(h²) 二阶导数:
        λ_max = 4/dx² + 4/dy²

    对于 O(h⁴) 二阶导数:
        λ_max = (30+30)/(12dx²) + ... ≈ 2.5/dx² · 2 = 5/dx²

    对于 O(h⁶) 二阶导数:
        λ_max = 495·2/(180dx²) + ... ≈ 5.5/dx²

    参数:
        G: 梯度系数 (J·m³/C²)
        L: 动力学系数 (m/(V·s))
        dx, dy: 网格间距
        fd_order: 有限差分精度

    返回:
        dt_max: 最大稳定时间步长 (s)
    """
    if fd_order == 2:
        lambda_max = 4.0 / (dx ** 2) + 4.0 / (dy ** 2)
    elif fd_order == 4:
        # λ_max = 30/(6h²) · 2 for 2D
        lambda_max = 60.0 / (12.0 * dx ** 2) + 60.0 / (12.0 * dy ** 2)
    elif fd_order == 6:
        # λ_max = 1218/(2520h²) · 2 for 2D (corrected 6th order)
        lambda_max = 2436.0 / (2520.0 * dx ** 2) + 2436.0 / (2520.0 * dy ** 2)
    else:
        lambda_max = 4.0 / (dx ** 2) + 4.0 / (dy ** 2)

    if L * G * lambda_max < 1e-30:
        return np.inf

    return 1.0 / (L * G * lambda_max)


def spectral_radius_banded(n, G, h, fd_order=6):
    """
    带状差分算子的谱半径.

    用于判断半隐式格式中隐式部分的收敛性.

    参数:
        n: 网格大小
        G: 梯度系数
        h: 网格间距
        fd_order: 精度阶数

    返回:
        rho: 谱半径
    """
    if fd_order == 2:
        rho = 4.0 * G / (h ** 2)
    elif fd_order == 4:
        rho = 30.0 * G / (6.0 * h ** 2)
    elif fd_order == 6:
        rho = 495.0 * G / (90.0 * h ** 2)
    else:
        rho = 4.0 * G / (h ** 2)
    return rho
