"""
nonlinear_root.py
基于种子项目 806_nonlin_bisect 的二分法求根

在核统计模型中，需要求解中子化学势 μ_n 使得中子数密度约束满足：
    n_n = (1/π²) ∫_0^∞ p² dp / [exp((E(p)-μ_n)/kT) + 1]
其中 E(p) = sqrt(p²c² + m_n²c⁴) 为相对论性能量。

该方程可简化为单变量非线性方程 f(μ_n) = 0，可用二分法在已知变号区间内求解。
"""

import numpy as np


def bisect_root(f, a, b, tol=None, max_iter=100):
    """
    二分法求根：在区间 [a,b] 上求解 f(x)=0，要求 f(a)·f(b) < 0。

    收敛性：每次迭代区间减半，
        |x_{k+1} - x*| ≤ |b_k - a_k| / 2
        达到容差 tol 所需迭代次数 N ≥ log2((b-a)/tol)。

    参数:
        f : callable, 目标函数
        a, b : float, 初始区间端点
        tol : float, 容差，默认 10*eps*|b-a|
        max_iter : int, 最大迭代次数

    返回:
        root : float, 近似根
        info : dict, 包含迭代次数、残差等信息
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.3e} 与 f(b)={fb:.3e} 同号，无法应用二分法")
    if np.isinf(fa) or np.isinf(fb) or np.isnan(fa) or np.isnan(fb):
        raise ValueError("区间端点处函数值为 Inf 或 NaN")

    if tol is None:
        tol = 10.0 * np.finfo(float).eps * abs(b - a)

    # 确保 tol 为正
    tol = max(tol, np.finfo(float).eps * abs(b - a))

    for k in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or abs(b - a) < tol:
            return c, {"iter": k, "residual": fc, "interval_width": abs(b - a)}
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    c = 0.5 * (a + b)
    fc = f(c)
    return c, {"iter": max_iter, "residual": fc, "interval_width": abs(b - a),
               "warning": "Maximum iterations reached"}


def solve_neutron_chemical_potential(n_n, T, m_n=1.67492749804e-24):
    """
    求解简并中子气的化学势 μ_n。

    采用非相对论近似下的 Fermi-Dirac 积分：
        n_n = (2m_n)^{3/2} / (2π²ħ³) ∫_0^∞ sqrt(E) dE / [exp((E-μ)/kT)+1]
    令 F_{1/2}(η) 为 Fermi-Dirac 积分，η = μ/kT，则：
        n_n = (m_n kT / (2π ħ²))^{3/2} · (4/√π) F_{1/2}(η)

    参数:
        n_n : float, 中子数密度 (cm^{-3})
        T : float, 温度 (K)
        m_n : float, 中子质量 (g)

    返回:
        mu : float, 化学势 (erg)
    """
    kB = 1.380649e-16  # erg/K
    hbar = 1.054571817e-27  # erg·s
    thermal_wavelength = np.sqrt(2.0 * np.pi * hbar ** 2 / (m_n * kB * T))
    n_quantum = 2.0 / thermal_wavelength ** 3  # 量子浓度量级

    # 定义残差函数 f(eta) = (4/sqrt(pi))*F_{1/2}(eta) - n_n/n_th
    def fermi_dirac_half(eta):
        """F_{1/2}(eta) 的数值近似，采用 Sommerfeld 展开 + 低温修正"""
        if eta > 10.0:
            # 强简并：Sommerfeld 展开
            return (2.0 / 3.0) * eta ** 1.5 * (1.0 + np.pi ** 2 / (8.0 * eta ** 2))
        elif eta < -10.0:
            # 经典极限：Boltzmann 近似
            return np.sqrt(np.pi) / 2.0 * np.exp(eta)
        else:
            # 数值积分
            xs = np.linspace(0, 50, 2000)
            dx = xs[1] - xs[0]
            integrand = np.sqrt(xs) / (np.exp(xs - eta) + 1.0)
            return np.trapezoid(integrand, xs)

    target = n_n / n_quantum

    def residual(eta):
        return (4.0 / np.sqrt(np.pi)) * fermi_dirac_half(eta) - target

    # 寻找变号区间
    eta_low = -50.0
    eta_high = 200.0
    # 确保存在根
    f_low = residual(eta_low)
    f_high = residual(eta_high)
    if f_low * f_high > 0:
        # 边界处理：如果都是正，说明中子密度极低，采用经典极限
        if f_low > 0 and f_high > 0:
            return kB * T * (-np.log(target * np.sqrt(np.pi) / 2.0)), {"mode": "classical_limit"}
        else:
            raise RuntimeError("无法找到化学势的变号区间")

    eta_root, info = bisect_root(residual, eta_low, eta_high, tol=1e-8)
    mu = eta_root * kB * T
    info["eta"] = eta_root
    info["target"] = target
    return mu, info


def test_nonlinear_root():
    """自包含测试"""
    # 中子星 crust 典型参数
    n_n = 1e30  # cm^{-3}
    T = 1e9  # K
    mu, info = solve_neutron_chemical_potential(n_n, T)
    print(f"[nonlinear_root] Neutron chemical potential μ_n = {mu:.3e} erg")
    print(f"[nonlinear_root] eta = {info.get('eta', 'N/A')}, iter = {info.get('iter', 'N/A')}")

    # 测试二分法在简单函数上
    f = lambda x: np.sin(x) - 0.5
    root, info = bisect_root(f, 0.0, np.pi / 2.0)
    print(f"[nonlinear_root] sin(x)=0.5 root = {root:.6f}, exact = {np.arcsin(0.5):.6f}")


if __name__ == "__main__":
    test_nonlinear_root()
