#!/usr/bin/env python3
"""
high_order_fd.py
================
高阶有限差分算子模块。

实现多种高阶有限差分格式:
  - 中心差分 (2阶, 4阶, 6阶, 8阶)
  - 紧致差分 (Padé 格式, 4阶, 6阶)
  - WENO5 (加权本质无振荡, 5阶)
  - 迎风格式 (1阶, 3阶)
  - 柱坐标下的微分算子 (1/r ∂/∂r(r·) 等)

数学基础:
  中心差分 2k 阶:
    f'(x) ≈ Σ_{j=-k}^{k} c_j f(x + jh) / h
  其中系数 c_j 由 Taylor 展开匹配确定。

  WENO5 (Jiang-Shu, 1996):
    f'_{i+1/2} = ω₀ f'⁽⁰⁾ + ω₁ f'⁽¹⁾ + ω₂ f'⁽²⁾
  其中 ω_k 为非线性权重, 基于光滑性指示子 β_k。

  紧致差分 (Padé):
    α f'_{i-1} + f'_i + α f'_{i+1} = a [f_{i+1} - f_{i-1}]/(2h)
                                   + b [f_{i+2} - f_{i-2}]/(4h)
  4阶: α=1/4, a=3/2
  6阶: α=1/3, a=14/9, b=1/9
"""

import numpy as np
from scipy.linalg import solve_banded


class HighOrderFD:
    """高阶有限差分算子集合"""

    # =====================================================================
    #  中心差分系数表
    # =====================================================================
    # 格式: {阶数: (半带宽, 系数列表)}
    CENTER_COEFFS = {
        2: (1, np.array([-0.5, 0.0, 0.5])),                    # O(h²)
        4: (2, np.array([1/12, -2/3, 0.0, 2/3, -1/12])),       # O(h⁴)
        6: (3, np.array([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60])),  # O(h⁶)
        8: (4, np.array([1/280, -4/105, 1/5, -4/5, 0.0, 4/5, -1/5, 4/105, -1/280])),
    }

    @staticmethod
    def first_derivative_central(f, dx, order=4, boundary='one-sided'):
        """
        中心差分一阶导数。

        参数:
            f: 函数值数组, shape=(N,)
            dx: 网格间距
            order: 精度阶数 (2, 4, 6, 8)
            boundary: 边界处理方式 ('one-sided', 'periodic', 'extrapolate')
        返回:
            df: 一阶导数近似, shape=(N,)
        """
        f = np.asarray(f, dtype=np.float64)
        N = len(f)
        if order not in HighOrderFD.CENTER_COEFFS:
            raise ValueError(f"不支持的精度阶数: {order}, 可选: {list(HighOrderFD.CENTER_COEFFS.keys())}")

        half_bw, coeffs = HighOrderFD.CENTER_COEFFS[order]
        df = np.zeros(N)

        # 内部点
        for j, c in enumerate(coeffs):
            shift = j - half_bw
            if abs(c) > 1.0e-15:
                df[half_bw:N - half_bw] += c * f[half_bw + shift:N - half_bw + shift]
        df[half_bw:N - half_bw] /= dx

        # 边界处理
        if boundary == 'periodic':
            for i in range(half_bw):
                for j, c in enumerate(coeffs):
                    idx = (i + j - half_bw) % N
                    df[i] += c * f[idx]
                df[i] /= dx
            for i in range(N - half_bw, N):
                for j, c in enumerate(coeffs):
                    idx = (i + j - half_bw) % N
                    df[i] += c * f[idx]
                df[i] /= dx
        elif boundary == 'one-sided':
            # 使用单侧差分
            if N >= 4:
                for i in range(half_bw):
                    df[i] = HighOrderFD._one_sided_derivative(f, i, dx, forward=True, order=min(4, N - 1))
                for i in range(N - half_bw, N):
                    df[i] = HighOrderFD._one_sided_derivative(f, i, dx, forward=False, order=min(4, N - 1))
        else:  # extrapolate
            df[:half_bw] = df[half_bw]
            df[N - half_bw:] = df[N - half_bw - 1]

        return df

    @staticmethod
    def _one_sided_derivative(f, i, dx, forward=True, order=4):
        """单侧差分 (向前或向后)"""
        N = len(f)
        p = min(order, N - 1)
        if p < 1:
            return 0.0

        # 使用前向或后向差分
        if forward:
            stencil = f[i:i + p + 1]
            if len(stencil) < 2:
                return 0.0
            # 一阶近似 (高精度需要更复杂的公式)
            if p == 1:
                return (stencil[1] - stencil[0]) / dx
            elif p == 2:
                return (-3 * stencil[0] + 4 * stencil[1] - stencil[2]) / (2 * dx)
            elif p == 3:
                return (-11 * stencil[0] + 18 * stencil[1] - 9 * stencil[2] + 2 * stencil[3]) / (6 * dx)
            else:  # p=4
                return (-25 * stencil[0] + 48 * stencil[1] - 36 * stencil[2] + 16 * stencil[3] - 3 * stencil[4]) / (12 * dx)
        else:
            stencil = f[max(0, i - p):i + 1]
            if len(stencil) < 2:
                return 0.0
            if p == 1:
                return (stencil[-1] - stencil[-2]) / dx
            elif p == 2:
                return (3 * stencil[-1] - 4 * stencil[-2] + stencil[-3]) / (2 * dx)
            elif p == 3:
                return (11 * stencil[-1] - 18 * stencil[-2] + 9 * stencil[-3] - 2 * stencil[-4]) / (6 * dx)
            else:
                return (25 * stencil[-1] - 48 * stencil[-2] + 36 * stencil[-3] - 16 * stencil[-4] + 3 * stencil[-5]) / (12 * dx)

    @staticmethod
    def second_derivative_central(f, dx, order=2):
        """
        中心差分二阶导数。
        2阶: f''(x) ≈ [f(x-h) - 2f(x) + f(x+h)] / h²
        4阶: f''(x) ≈ [-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)] / (12h²)
        """
        f = np.asarray(f, dtype=np.float64)
        N = len(f)
        df = np.zeros(N)

        if order == 2:
            df[1:N - 1] = (f[0:N - 2] - 2 * f[1:N - 1] + f[2:N]) / dx ** 2
            df[0] = (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / dx ** 2
            df[N - 1] = (2 * f[N - 1] - 5 * f[N - 2] + 4 * f[N - 3] - f[N - 4]) / dx ** 2
        elif order == 4:
            df[2:N - 2] = (-f[0:N - 4] + 16 * f[1:N - 3] - 30 * f[2:N - 2] +
                           16 * f[3:N - 1] - f[4:N]) / (12 * dx ** 2)
            # 边界用低阶
            df[0] = (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / dx ** 2
            df[1] = (f[0] - 2 * f[1] + f[2]) / dx ** 2
            df[N - 2] = (f[N - 3] - 2 * f[N - 2] + f[N - 1]) / dx ** 2
            df[N - 1] = (2 * f[N - 1] - 5 * f[N - 2] + 4 * f[N - 3] - f[N - 4]) / dx ** 2
        else:
            raise ValueError(f"二阶导数仅支持 order=2,4, got {order}")

        return df

    @staticmethod
    def laplacian_2d(f, dx, dy, order=2):
        """
        二维 Laplace 算子 ∇²f = ∂²f/∂x² + ∂²f/∂y²。
        用于粘性项和热传导项的离散。
        """
        fxx = HighOrderFD.second_derivative_central(f, dx, order=order)
        # 对 y 方向求导 (转置)
        fyy = np.zeros_like(f)
        for i in range(f.shape[0]):
            fyy[i, :] = HighOrderFD.second_derivative_central(f[i, :], dy, order=order)
        return fxx + fyy


class WENO5:
    """
    WENO5 (Weighted Essentially Non-Oscillatory, 5th order) 重构。
    用于 ICF 内爆中激波捕捉和高梯度区域。

    Jiang-Shu (1996) 有限差分 WENO5:
      对 f'_{i+1/2} 的重构:
      三个候选模板:
        S₀ = {i, i+1, i+2}
        S₁ = {i-1, i, i+1}
        S₂ = {i-2, i-1, i}
      各模板上的导数近似:
        f'⁽⁰⁾ = (2f_i + 5f_{i+1} - f_{i+2}) / 6
        f'⁽¹⁾ = (-f_{i-1} + 5f_i + 2f_{i+1}) / 6  ← 注意这里符号
        f'⁽²⁾ = (2f_{i-2} - 7f_{i-1} + 11f_i) / 6
      光滑性指示子:
        β₀ = (13/12)(f_i - 2f_{i+1} + f_{i+2})² + (1/4)(3f_i - 4f_{i+1} + f_{i+2})²
        β₁ = (13/12)(f_{i-1} - 2f_i + f_{i+1})² + (1/4)(f_{i-1} - f_{i+1})²
        β₂ = (13/12)(f_{i-2} - 2f_{i-1} + f_i)² + (1/4)(f_{i-2} - 4f_{i-1} + 3f_i)²
      非线性权重:
        α_k = d_k / (ε + β_k)²
        ω_k = α_k / (α₀ + α₁ + α₂)
      理想权重: d₀=1/10, d₁=6/10, d₂=3/10
    """

    EPS = 1.0e-6  # 防止除零

    @staticmethod
    def reconstruct(f, dx, direction='right'):
        """
        WENO5 重构计算半节点通量。

        参数:
            f: 函数值, shape=(N,)
            dx: 网格间距
            direction: 'right' 计算 f'_{i+1/2}, 'left' 计算 f'_{i-1/2}
        返回:
            f_half: 半节点通量, shape=(N-1,) 或 (N,) 取决于填充
        """
        f = np.asarray(f, dtype=np.float64)
        N = len(f)
        if N < 5:
            raise ValueError(f"WENO5 至少需要 5 个点, got {N}")

        eps = WENO5.EPS
        d0, d1, d2 = 0.1, 0.6, 0.3

        f_half = np.zeros(N + 1)

        # 对内部点 i+1/2 (i = 2..N-3)
        for i in range(2, N - 2):
            if direction == 'right':
                # 三个模板的导数近似
                fp0 = (2.0 * f[i] + 5.0 * f[i + 1] - f[i + 2]) / 6.0
                fp1 = (-f[i - 1] + 5.0 * f[i] + 2.0 * f[i + 1]) / 6.0
                fp2 = (2.0 * f[i - 2] - 7.0 * f[i - 1] + 11.0 * f[i]) / 6.0

                # 光滑性指示子
                b0 = (13.0 / 12.0) * (f[i] - 2 * f[i + 1] + f[i + 2]) ** 2 + \
                     0.25 * (3 * f[i] - 4 * f[i + 1] + f[i + 2]) ** 2
                b1 = (13.0 / 12.0) * (f[i - 1] - 2 * f[i] + f[i + 1]) ** 2 + \
                     0.25 * (f[i - 1] - f[i + 1]) ** 2
                b2 = (13.0 / 12.0) * (f[i - 2] - 2 * f[i - 1] + f[i]) ** 2 + \
                     0.25 * (f[i - 2] - 4 * f[i - 1] + 3 * f[i]) ** 2
            else:
                # 左状态 (镜像)
                fp0 = (2.0 * f[i + 1] + 5.0 * f[i] - f[i - 1]) / 6.0
                fp1 = (-f[i + 2] + 5.0 * f[i + 1] + 2.0 * f[i]) / 6.0
                fp2 = (2.0 * f[i + 3] - 7.0 * f[i + 2] + 11.0 * f[i + 1]) / 6.0 if i + 3 < N else fp1

                b0 = (13.0 / 12.0) * (f[i + 1] - 2 * f[i] + f[i - 1]) ** 2 + \
                     0.25 * (3 * f[i + 1] - 4 * f[i] + f[i - 1]) ** 2
                b1 = (13.0 / 12.0) * (f[i + 2] - 2 * f[i + 1] + f[i]) ** 2 + \
                     0.25 * (f[i + 2] - f[i]) ** 2
                b2 = b1  # fallback

            # 非线性权重
            a0 = d0 / (eps + b0) ** 2
            a1 = d1 / (eps + b1) ** 2
            a2 = d2 / (eps + b2) ** 2
            a_sum = a0 + a1 + a2

            if a_sum > 0:
                w0 = a0 / a_sum
                w1 = a1 / a_sum
                w2 = a2 / a_sum
            else:
                w0, w1, w2 = d0, d1, d2

            f_half[i + 1] = w0 * fp0 + w1 * fp1 + w2 * fp2

        # 边界用低阶近似
        if N >= 3:
            f_half[0] = f[0]
            f_half[1] = f[0] + 0.5 * (f[1] - f[0])
            f_half[N - 1] = f[N - 1] - 0.5 * (f[N - 1] - f[N - 2])
            f_half[N] = f[N - 1]

        return f_half

    @staticmethod
    def flux_divergence(f, dx):
        """
        WENO5 计算通量散度 (用于守恒律 ∂u/∂t + ∂f/∂x = 0)。
        返回 (F_{i+1/2} - F_{i-1/2}) / dx
        """
        f = np.asarray(f, dtype=np.float64)
        N = len(f)

        f_right = WENO5.reconstruct(f, dx, direction='right')
        f_left = WENO5.reconstruct(f, dx, direction='left')

        # 通量散度
        div_f = np.zeros(N)
        for i in range(2, N - 2):
            div_f[i] = (f_right[i + 1] - f_left[i]) / dx

        # 边界
        if N >= 3:
            div_f[0] = (f[1] - f[0]) / dx
            div_f[1] = (f[2] - f[0]) / (2 * dx)
            div_f[N - 2] = (f[N - 1] - f[N - 3]) / (2 * dx)
            div_f[N - 1] = (f[N - 1] - f[N - 2]) / dx

        return div_f


class CompactDifference:
    """
    紧致差分 (Padé 格式)。
    求解三对角系统:
      α f'_{i-1} + f'_i + α f'_{i+1} = a [f_{i+1} - f_{i-1}]/(2h) + b [f_{i+2} - f_{i-2}]/(4h)

    4阶紧致 (α=1/4, a=3/2, b=0):
      (1/4) f'_{i-1} + f'_i + (1/4) f'_{i+1} = (3/2) [f_{i+1} - f_{i-1}]/(2h)

    6阶紧致 (α=1/3, a=14/9, b=1/9):
      (1/3) f'_{i-1} + f'_i + (1/3) f'_{i+1} = (14/9) [f_{i+1} - f_{i-1}]/(2h) + (1/9) [f_{i+2} - f_{i-2}]/(4h)
    """

    @staticmethod
    def derivative(f, dx, order=4):
        """
        紧致差分一阶导数。

        参数:
            f: 函数值, shape=(N,)
            dx: 网格间距
            order: 精度阶数 (4 或 6)
        返回:
            df: 一阶导数, shape=(N,)
        """
        f = np.asarray(f, dtype=np.float64)
        N = len(f)
        df = np.zeros(N)

        if order == 4:
            alpha = 0.25
            a = 1.5
            # RHS: a [f_{i+1} - f_{i-1}]/(2h)
            rhs = np.zeros(N)
            rhs[1:N - 1] = a * (f[2:N] - f[0:N - 2]) / (2.0 * dx)
            rhs[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * dx)
            rhs[N - 1] = (3 * f[N - 1] - 4 * f[N - 2] + f[N - 3]) / (2 * dx)
        elif order == 6:
            alpha = 1.0 / 3.0
            a = 14.0 / 9.0
            b = 1.0 / 9.0
            rhs = np.zeros(N)
            rhs[2:N - 2] = a * (f[3:N - 1] - f[1:N - 3]) / (2 * dx) + \
                           b * (f[4:N] - f[0:N - 4]) / (4 * dx)
            # 边界用低阶
            rhs[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * dx)
            rhs[1] = (f[2] - f[0]) / (2 * dx)
            rhs[N - 2] = (f[N - 1] - f[N - 3]) / (2 * dx)
            rhs[N - 1] = (3 * f[N - 1] - 4 * f[N - 2] + f[N - 3]) / (2 * dx)
        else:
            raise ValueError(f"紧致差分仅支持 order=4,6, got {order}")

        # 求解三对角系统
        # 下对角: α, 主对角: 1, 上对角: α
        ab = np.zeros((3, N))
        ab[0, 1:] = alpha  # 上对角
        ab[1, :] = 1.0      # 主对角
        ab[2, :-1] = alpha  # 下对角

        df = solve_banded((1, 1), ab, rhs)
        return df


class CylindricalOperators:
    """
    柱坐标 (r, z) 下的微分算子。
    ICF 内爆模拟在轴对称柱坐标下进行。

    关键算子:
      散度 (轴对称):
        ∇·F = (1/r) ∂(r F_r)/∂r + ∂F_z/∂z

      Laplace 算子 (轴对称):
        ∇²φ = (1/r) ∂/∂r(r ∂φ/∂r) + ∂²φ/∂z²

      梯度:
        ∇φ = (∂φ/∂r, ∂φ/∂z)
    """

    @staticmethod
    def divergence_axisym(F_r, F_z, r, dr, dz):
        """
        轴对称散度: ∇·F = (1/r) ∂(r F_r)/∂r + ∂F_z/∂z
        """
        N_r = F_r.shape[0]
        N_z = F_r.shape[1]
        div = np.zeros_like(F_r)

        # (1/r) ∂(r F_r)/∂r
        rFr = r[:, np.newaxis] * F_r
        drFr = HighOrderFD.first_derivative_central(rFr[:, 0], dr, order=4)
        for j in range(N_z):
            drFr_j = HighOrderFD.first_derivative_central(rFr[:, j], dr, order=4)
            div[:, j] += drFr_j / np.maximum(r, 1.0e-30)

        # ∂F_z/∂z
        for i in range(N_r):
            div[i, :] += HighOrderFD.first_derivative_central(F_z[i, :], dz, order=4)

        return div

    @staticmethod
    def laplacian_axisym(phi, r, dr, dz):
        """
        轴对称 Laplace: ∇²φ = (1/r) ∂/∂r(r ∂φ/∂r) + ∂²φ/∂z²
        """
        N_r, N_z = phi.shape
        lap = np.zeros_like(phi)

        # (1/r) ∂/∂r(r ∂φ/∂r)
        dphi_dr = np.zeros_like(phi)
        for j in range(N_z):
            dphi_dr[:, j] = HighOrderFD.first_derivative_central(phi[:, j], dr, order=4)

        r_dphi_dr = r[:, np.newaxis] * dphi_dr
        dr_r_dphi_dr = np.zeros_like(phi)
        for j in range(N_z):
            dr_r_dphi_dr[:, j] = HighOrderFD.first_derivative_central(r_dphi_dr[:, j], dr, order=4)

        lap += dr_r_dphi_dr / np.maximum(r[:, np.newaxis], 1.0e-30)

        # ∂²φ/∂z²
        for i in range(N_r):
            lap[i, :] += HighOrderFD.second_derivative_central(phi[i, :], dz, order=2)

        return lap


def compute_fd_error_convergence(func_exact, func_deriv_exact, x_range, N_list, fd_order=4):
    """
    计算有限差分算子的网格收敛性。
    通过逐步加密网格, 验证误差是否按 O(h^p) 收敛。

    参数:
        func_exact: 精确函数
        func_deriv_exact: 精确导数
        x_range: (x_min, x_max)
        N_list: 网格点数列表
        fd_order: 有限差分阶数
    返回:
        dict: {N: error, convergence_rate: [...]}
    """
    results = {'N': [], 'h': [], 'error': [], 'rate': []}
    prev_error = None

    for N in N_list:
        x = np.linspace(x_range[0], x_range[1], N)
        dx = x[1] - x[0]
        f = func_exact(x)
        df_num = HighOrderFD.first_derivative_central(f, dx, order=fd_order)
        df_exact = func_deriv_exact(x)

        error = np.max(np.abs(df_num - df_exact))
        h = dx
        results['N'].append(N)
        results['h'].append(h)
        results['error'].append(error)

        if prev_error is not None and prev_error > 0 and error > 0:
            rate = np.log(prev_error / error) / np.log(h / results['h'][-2]) if len(results['h']) >= 2 else 0
            results['rate'].append(rate)
        prev_error = error

    return results
