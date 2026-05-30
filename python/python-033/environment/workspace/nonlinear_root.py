
import numpy as np


def bisect_root(f, a, b, tol=None, max_iter=100):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.3e} 与 f(b)={fb:.3e} 同号，无法应用二分法")
    if np.isinf(fa) or np.isinf(fb) or np.isnan(fa) or np.isnan(fb):
        raise ValueError("区间端点处函数值为 Inf 或 NaN")

    if tol is None:
        tol = 10.0 * np.finfo(float).eps * abs(b - a)


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
    kB = 1.380649e-16
    hbar = 1.054571817e-27
    thermal_wavelength = np.sqrt(2.0 * np.pi * hbar ** 2 / (m_n * kB * T))
    n_quantum = 2.0 / thermal_wavelength ** 3


    def fermi_dirac_half(eta):
        if eta > 10.0:

            return (2.0 / 3.0) * eta ** 1.5 * (1.0 + np.pi ** 2 / (8.0 * eta ** 2))
        elif eta < -10.0:

            return np.sqrt(np.pi) / 2.0 * np.exp(eta)
        else:

            xs = np.linspace(0, 50, 2000)
            dx = xs[1] - xs[0]
            integrand = np.sqrt(xs) / (np.exp(xs - eta) + 1.0)
            return np.trapezoid(integrand, xs)

    target = n_n / n_quantum

    def residual(eta):
        return (4.0 / np.sqrt(np.pi)) * fermi_dirac_half(eta) - target


    eta_low = -50.0
    eta_high = 200.0

    f_low = residual(eta_low)
    f_high = residual(eta_high)
    if f_low * f_high > 0:

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

    n_n = 1e30
    T = 1e9
    mu, info = solve_neutron_chemical_potential(n_n, T)
    print(f"[nonlinear_root] Neutron chemical potential μ_n = {mu:.3e} erg")
    print(f"[nonlinear_root] eta = {info.get('eta', 'N/A')}, iter = {info.get('iter', 'N/A')}")


    f = lambda x: np.sin(x) - 0.5
    root, info = bisect_root(f, 0.0, np.pi / 2.0)
    print(f"[nonlinear_root] sin(x)=0.5 root = {root:.6f}, exact = {np.arcsin(0.5):.6f}")


if __name__ == "__main__":
    test_nonlinear_root()
