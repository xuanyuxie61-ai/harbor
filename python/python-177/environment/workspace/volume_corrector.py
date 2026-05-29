# -*- coding: utf-8 -*-
"""
volume_corrector.py
===================
体积守恒修正与外部力场模块。

融合原始项目:
  - 1025_ripple_ode: 非线性 ODE 驱动思想（用于外部振荡力场）
  - 1218_test_min: Brent 一维优化（用于最优修正系数搜索）

核心数学公式
------------
1. 体积守恒约束:
   d/dt ∫_{Ω⁻(t)} dx = 0
   即内部区域体积（面积）不随时间变化。

2. 体积守恒修正（Sussman & Fatemi, 1999）:
   对演化后的水平集 φ*，施加修正:
   φ(x) = φ*(x) + λ
   其中 λ 由体积守恒条件确定:
   ∫_{φ*+λ<0} dx = V_0
   这等价于寻找 λ 使得 F(λ) = V(λ) - V_0 = 0。

3. Brent 法求解 F(λ)=0（融入 test_min 思想）:
   结合黄金分割搜索与抛物线插值，
   收敛阶数约为 1.324（超线性）。
   对单峰函数，误差界:
   |λ* - λ_n| ≤ 3·ε·|λ*| + tol

4. 外部振荡力场（融入 ripple_ode 思想）:
   将 ripple ODE dy/dt = sin(t·y) 推广为界面外力:
   f_ext(x,y,t) = A · sin(ω t) · sin(k_x x) · sin(k_y y)
   或更一般地:
   f_ext(x,y,t) = A · sin( t · g(x,y) )
   其中 g(x,y) 为空间分布函数。

5. 修正后的水平集方程:
   ∂φ/∂t + (V_n + f_ext) |∇φ| = ε κ |∇φ| + λ |∇φ|
   其中 λ 为体积守恒拉格朗日乘子。
"""

import numpy as np


class VolumeCorrector:
    """
    体积守恒修正器，使用 Brent 法搜索最优修正量。
    """

    def __init__(self, levelset, target_volume=None):
        self.ls = levelset
        if target_volume is None:
            self.target_volume = levelset.compute_volume()
        else:
            self.target_volume = target_volume

    def _volume_after_shift(self, lam):
        """
        计算水平集平移 λ 后的体积:
        V(λ) = ∫_{φ+λ<0} dx
        """
        phi_shifted = self.ls.phi + lam
        vol = np.sum(phi_shifted < 0) * self.ls.dx * self.ls.dy
        return vol

    def _volume_residual(self, lam):
        """
        体积守恒残差:
        F(λ) = V(λ) - V_0
        """
        return self._volume_after_shift(lam) - self.target_volume

    def correct_volume_brent(self, a=-1.0, b=1.0, tol=1e-10, max_iter=100):
        """
        Brent 法搜索 λ 使得 V(λ) = V_0。
        算法源自 test_min/p00_fmin 的黄金分割+抛物线插值思想。

        参数:
            a, b : 初始搜索区间 [a,b]
            tol  : 收敛容差
            max_iter : 最大迭代次数
        返回:
            lam_opt : 最优修正量
            it      : 实际迭代次数
        """
        c = 0.5 * (3.0 - np.sqrt(5.0))  # 黄金分割比的平方的逆
        eps = np.sqrt(np.finfo(float).eps)

        v = a + c * (b - a)
        w = v
        x = v
        e = 0.0
        fx = self._volume_residual(x)
        fv = fx
        fw = fx

        for it in range(max_iter):
            midpoint = 0.5 * (a + b)
            tol1 = eps * abs(x) + tol / 3.0
            tol2 = 2.0 * tol1

            if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
                break

            if abs(e) <= tol1:
                if midpoint <= x:
                    e = a - x
                else:
                    e = b - x
                d = c * e
            else:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p_val = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0:
                    p_val = -p_val
                q = abs(q)
                r = e
                e = d

                if abs(0.5 * q * r) <= abs(p_val) or p_val <= q * (a - x) or q * (b - x) <= p_val:
                    if midpoint <= x:
                        e = a - x
                    else:
                        e = b - x
                    d = c * e
                else:
                    d = p_val / q
                    u = x + d
                    if (u - a) < tol2:
                        d = abs(tol1) * np.sign(midpoint - x)
                    if (b - u) < tol2:
                        d = abs(tol1) * np.sign(midpoint - x)

            if tol1 <= abs(d):
                u = x + d
            elif abs(d) < tol1:
                u = x + abs(tol1) * np.sign(d)
            else:
                u = x + abs(tol1) * np.sign(d)

            fu = self._volume_residual(u)

            if fu <= fx:
                if x <= u:
                    a = x
                else:
                    b = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu

        lam_opt = x
        self.ls.phi += lam_opt
        return lam_opt, it + 1

    def correct_volume_simple(self):
        """
        简单的体积守恒修正：用二分法搜索 λ。
        作为 Brent 法的备选，保证鲁棒性。

        注意: V(λ) 是 λ 的减函数（λ 越大，φ+λ<0 的区域越小）。
        因此 F(λ)=V(λ)-V0 是减函数。
        """
        lo, hi = -2.0, 2.0
        # 扩大搜索范围以确保 bracket
        for _ in range(5):
            if self._volume_residual(lo) < 0:
                lo *= 2.0
            if self._volume_residual(hi) > 0:
                hi *= 2.0

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = self._volume_residual(mid)
            if fmid > 0:
                # V(mid) > V0，需要增大 λ 来减小体积
                lo = mid
            else:
                # V(mid) < V0，需要减小 λ 来增大体积
                hi = mid
        lam_opt = 0.5 * (lo + hi)
        self.ls.phi += lam_opt
        return lam_opt


class ExternalForcing:
    """
    外部力场生成器，融入 ripple_ode 的非线性振荡思想。
    """

    @staticmethod
    def ripple_like_forcing(X, Y, t, A=0.1):
        """
        类 ripple ODE 的非线性振荡力场:
        f_ext(x,y,t) = A · sin( t · (x² + y²) )
        源自 ripple_deriv: dy/dt = sin(t·y) 的思想，
        将耦合变量推广为空间位置。
        """
        r2 = X ** 2 + Y ** 2
        r2 = np.clip(r2, 1e-8, None)
        return A * np.sin(t * r2)

    @staticmethod
    def oscillatory_normal_forcing(X, Y, t, A=0.05, omega=2.0, kx=5.0, ky=5.0):
        """
        时空振荡法向力场:
        f_ext = A · sin(ω t) · sin(kx x) · sin(ky y)
        模拟周期性外部驱动（如声波、振动等）。
        """
        return A * np.sin(omega * t) * np.sin(kx * X) * np.sin(ky * Y)

    @staticmethod
    def gravitational_forcing(X, Y, g=1.0, angle=0.0):
        """
        倾斜重力场（法向投影）:
        f_ext = g · (x cosθ + y sinθ)
        模拟重力驱动的界面运动。
        """
        return g * (X * np.cos(angle) + Y * np.sin(angle))

    @staticmethod
    def combined_forcing(X, Y, t, A1=0.05, A2=0.03, omega=2.0):
        """
        组合力场：振荡 + ripple-like。
        """
        f1 = ExternalForcing.oscillatory_normal_forcing(X, Y, t, A=A1, omega=omega)
        f2 = ExternalForcing.ripple_like_forcing(X, Y, t, A=A2)
        return f1 + f2
