"""
stability_analysis.py
基于 105_boundary_locus2 的 ODE 数值方法稳定性区域分析思想，
构建血凝级联刚性 ODE 求解器的稳定性诊断工具。

科学背景：
    血凝级联反应ODE系统具有多时间尺度特征（stiffness）：
    - 快速反应：TF-VIIa 复合物形成（ms级）
    - 慢速反应：纤维蛋白聚合（min级）
    时间尺度比可达 10⁵ 以上，属于严重刚性系统。

    隐式梯形法的稳定性函数为：
        R(z) = (1 + z/2) / (1 - z/2)
    绝对稳定区域满足 |R(z)| ≤ 1。

    对血凝系统，局部 Jacobian 的特征值 λ 分布在复平面左半部，
    最大模特征值决定最大允许步长 h_max = c / |λ_max|，
    其中 c 为稳定性边界到虚轴的距离。

数学公式：
    1. 稳定性函数：
       R(z) = ρ(e^{iθ}) / σ(e^{iθ}),  z = hλ

    2. 刚性比：
       S = |Re(λ_max)| / |Re(λ_min)|

    3. A-稳定性判定：
       方法对 Re(z) < 0 的区域绝对稳定。
"""

import numpy as np


class StabilityAnalyzer:
    """
    数值方法稳定性分析器。
    """

    def __init__(self):
        pass

    @staticmethod
    def trapezoidal_stability_function(z):
        """
        隐式梯形法的稳定性函数：
            R(z) = (1 + z/2) / (1 - z/2)
        """
        z = np.asarray(z, dtype=complex)
        denom = 1.0 - 0.5 * z
        # 边界处理：避免除零
        denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
        return (1.0 + 0.5 * z) / denom

    @staticmethod
    def boundary_locus(method="trapezoidal", n_points=400):
        """
        计算稳定性区域的边界曲线。
        基于 105_boundary_locus2 的思想，追踪 |R(z)| = 1 的曲线。

        参数:
            method   : str, "trapezoidal" 或 "implicit_euler"
            n_points : int, 边界采样点数

        返回:
            z_boundary : ndarray, 边界上的复数点
        """
        theta = np.linspace(0, 2 * np.pi, n_points)
        z_unit = np.exp(1j * theta)

        if method == "trapezoidal":
            # 对梯形法，|R(z)| = 1 的边界是虚轴
            # 但我们可以追踪稳定性函数在单位圆上的像的逆
            # R(z) = (1+z/2)/(1-z/2) => z = 2(R-1)/(R+1)
            z_boundary = 2.0 * (z_unit - 1.0) / (z_unit + 1.0 + 1e-30)
        elif method == "implicit_euler":
            # R(z) = 1 / (1 - z) => z = 1 - 1/R
            z_boundary = 1.0 - 1.0 / (z_unit + 1e-30)
        elif method == "explicit_euler":
            # R(z) = 1 + z => z = R - 1
            z_boundary = z_unit - 1.0
        else:
            raise ValueError(f"未知方法: {method}")
        return z_boundary

    @staticmethod
    def is_stable(method, z):
        """
        判断给定 z = hλ 是否在方法的绝对稳定区域内。
        """
        z = np.asarray(z, dtype=complex)
        if method == "trapezoidal":
            # A-稳定：整个左半平面稳定
            return np.real(z) <= 1e-12
        elif method == "implicit_euler":
            return np.real(z) <= 1e-12
        elif method == "explicit_euler":
            return np.abs(1.0 + z) <= 1.0 + 1e-12
        else:
            raise ValueError(f"未知方法: {method}")

    def analyze_jacobian_stiffness(self, J):
        """
        分析 Jacobian 矩阵的刚性特征。

        参数:
            J : ndarray, Jacobian 矩阵

        返回:
            eigenvalues : ndarray, 特征值
            stiffness_ratio : float, 刚性比
            h_max_euler   : float, 显式Euler最大步长
            h_max_trap    : float, 梯形法建议步长
        """
        # TODO: 修复 Hole 2 —— Jacobian刚性分析核心科学计算
        # 需要实现：特征值分解、左半平面特征值筛选、刚性比计算、
        # 显式Euler和梯形法最大允许步长估计
        pass

    def recommend_solver(self, J):
        """
        根据 Jacobian 特征推荐求解器。
        """
        eigs, stiff_ratio, h_euler, h_trap = self.analyze_jacobian_stiffness(J)
        print("=" * 60)
        print("ODE 系统刚性分析")
        print("=" * 60)
        print(f"特征值实部范围: [{np.min(np.real(eigs)):.4e}, {np.max(np.real(eigs)):.4e}]")
        print(f"刚性比 S = {stiff_ratio:.2e}")
        if stiff_ratio > 1e3:
            print("  -> 系统严重刚性，强烈建议使用隐式方法（梯形法/BDF）")
        elif stiff_ratio > 1e1:
            print("  -> 系统中度刚性，建议使用隐式方法")
        else:
            print("  -> 系统非刚性，显式方法可接受")
        print(f"显式Euler最大步长: {h_euler:.4e} s")
        print(f"梯形法建议步长: {h_trap:.4e} s")
        return stiff_ratio, h_trap


def demo_stability():
    """
    演示：分析模拟的血凝 Jacobian 的刚性。
    """
    analyzer = StabilityAnalyzer()

    # 模拟一个12×12的血凝Jacobian（具有多时间尺度）
    np.random.seed(42)
    J = np.random.randn(12, 12) * 0.1
    # 引入刚性：一些快速反应
    J[0, 0] = -500.0   # TF-VIIa 快速
    J[1, 1] = -50.0    # IXa 中等
    J[4, 4] = -5.0     # IIa 较慢
    J[5, 5] = -0.5     # Fibrin 很慢
    J[6, 6] = -0.05    # APC 极慢
    # 添加耦合
    J[4, 2] = 20.0
    J[5, 4] = 2.0

    stiff_ratio, h_trap = analyzer.recommend_solver(J)

    # 验证梯形法稳定性
    z = -1.0 + 2.0j
    R = analyzer.trapezoidal_stability_function(z)
    print(f"\n梯形法稳定性函数在 z={z}: |R(z)| = {abs(R):.4f}")
    print(f"是否稳定: {analyzer.is_stable('trapezoidal', z)}")

    return analyzer


if __name__ == "__main__":
    demo_stability()
