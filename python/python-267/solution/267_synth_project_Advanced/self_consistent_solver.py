"""
self_consistent_solver.py — 自洽 Fermi 能级与载流子密度求解
==============================================================

本模块实现拓扑绝缘体中 Fermi 能级的自洽确定, 通过固定点迭代
求解载流子密度守恒方程。

物理问题
--------
给定边界态的态密度 D(E) 和总载流子密度 n_total, 求化学势 μ 使得:

    n_total = ∫ D(E) f(E-μ, T) dE

其中 f(E,T) = 1/(1+exp(E/(kBT))) 是 Fermi-Dirac 分布。

这是一个非线性方程, 因为 μ 同时出现在 f 中。

求解方法 (借鉴 807_nonlin_fixed_point):
1. **不动点迭代**: μ_{n+1} = g(μ_n)
2. **Newton-Raphson**: μ_{n+1} = μ_n - F(μ_n)/F'(μ_n)
3. **二分法**: 在已知区间内搜索

其中 F(μ) = ∫ D(E) f(E-μ) dE - n_total = 0

收敛判据
--------
|μ_{n+1} - μ_n| < tol  或  |F(μ_n)| < tol

来源映射
--------
- 807_nonlin_fixed_point: 不动点迭代与 Newton 法
- 1435_zoomin: Brent 法等高级求根方法
"""

import numpy as np
from typing import Callable, Tuple, Dict, Optional


def fermi_dirac(E: np.ndarray, mu: float, temperature: float) -> np.ndarray:
    """
    Fermi-Dirac 分布函数。

    f(E, μ, T) = 1 / (1 + exp((E-μ)/(kB T)))

    数值安全版本: 对大参数使用渐近展开避免溢出。

    Parameters
    ----------
    E : ndarray
        能量 (eV)
    mu : float
        化学势 (eV)
    temperature : float
        温度 (eV, kB=1)

    Returns
    -------
    f : ndarray
    """
    kBT = max(temperature, 1e-15)
    x = (E - mu) / kBT

    # 数值安全
    f = np.zeros_like(x)
    mask_pos = x > 500
    mask_neg = x < -500
    mask_mid = ~(mask_pos | mask_neg)

    f[mask_pos] = 0.0
    f[mask_neg] = 1.0
    f[mask_mid] = 1.0 / (1.0 + np.exp(x[mask_mid]))

    return f


def fermi_dirac_derivative(E: np.ndarray, mu: float,
                           temperature: float) -> np.ndarray:
    """
    Fermi-Dirac 分布对 μ 的导数 (即对 E 的导数的负值)。

    ∂f/∂μ = -∂f/∂E = f(1-f)/(kB T)

    这是一个以 μ 为中心的峰状函数, 宽度 ~ kBT。

    Parameters
    ----------
    E, mu, temperature

    Returns
    -------
    dfdmu : ndarray
    """
    f = fermi_dirac(E, mu, temperature)
    kBT = max(temperature, 1e-15)
    return f * (1.0 - f) / kBT


def density_of_states_bhz(E: np.ndarray, params_bhz,
                          Nk: int = 50) -> np.ndarray:
    """
    计算 BHZ 模型的态密度 (通过 BZ 积分)。

    D(E) = (1/A_BZ) Σ_k δ(E - Eₙ(k))

    数值上用小展宽 η 的 Lorentzian 代替 δ 函数:
    δ_η(E) = (η/π) / (E² + η²)

    Parameters
    ----------
    E : ndarray
        能量网格
    params_bhz : BHZParameters
    Nk : int
        k 网格点数

    Returns
    -------
    dos : ndarray, shape (len(E),)
    """
    from bhz_hamiltonian import bhz_h_k_analytic

    eta = 0.005  # 展宽 (eV)
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    dkx = kx[1] - kx[0] if Nk > 1 else 1.0
    dky = ky[1] - ky[0] if Nk > 1 else 1.0

    # 收集所有本征值
    all_evals = []
    for ikx in kx:
        for iky in ky:
            evals = bhz_h_k_analytic(ikx, iky, params_bhz)
            all_evals.extend(evals)
    all_evals = np.array(all_evals)

    # Lorentzian 展宽的 DOS
    dos = np.zeros_like(E)
    for ie in range(len(E)):
        lorentz = (eta / np.pi) / ((E[ie] - all_evals) ** 2 + eta ** 2)
        dos[ie] = np.sum(lorentz) * dkx * dky / (4 * np.pi ** 2)

    return dos


def carrier_density(mu: float, energies: np.ndarray,
                    dos: np.ndarray, temperature: float) -> float:
    """
    计算载流子密度: n(μ) = ∫ D(E) f(E-μ, T) dE

    使用梯形积分。

    Parameters
    ----------
    mu : float
    energies : ndarray
    dos : ndarray
    temperature : float

    Returns
    -------
    n : float
    """
    f = fermi_dirac(energies, mu, temperature)
    integrand = dos * f
    n = np.trapz(integrand, energies)
    return n


def fixed_point_iteration(n_target: float, energies: np.ndarray,
                          dos: np.ndarray, temperature: float,
                          mu_init: float = 0.0,
                          max_iter: int = 200, tol: float = 1e-10,
                          mixing: float = 0.3) -> Dict:
    """
    不动点迭代法求解自洽化学势。

    迭代格式:
        n(μ_n) = ∫ D(E) f(E-μ_n) dE
        μ_{n+1} = μ_n + α × (n_target - n(μ_n)) / D(μ_n)

    其中 α 是混合参数 (0 < α ≤ 1), 控制收敛速度和稳定性。

    更简单的固定点格式:
        g(μ) = μ + α(n_target - n(μ))
        μ_{n+1} = g(μ_n)

    Parameters
    ----------
    n_target : float
        目标载流子密度
    energies, dos : ndarray
    temperature : float
    mu_init : float
        初始猜测
    max_iter : int
    tol : float
    mixing : float
        混合参数

    Returns
    -------
    result : dict
    """
    mu = mu_init
    history = {'mu': [mu], 'residual': [], 'iterations': 0}

    for it in range(max_iter):
        n_current = carrier_density(mu, energies, dos, temperature)
        residual = n_current - n_target

        history['residual'].append(abs(residual))

        if abs(residual) < tol:
            history['iterations'] = it
            history['converged'] = True
            break

        # 不动点更新: 使用 DOS 在 μ 处的值作为步长估计
        # dn/dμ = ∫ D(E) (-∂f/∂μ) dE ≈ D(μ) (低温近似)
        dos_at_mu_idx = np.argmin(np.abs(energies - mu))
        dos_mu = max(dos[dos_at_mu_idx], 1e-10)

        delta_mu = mixing * residual / dos_mu
        mu = mu - delta_mu  # 减号: 如果 n > n_target, μ 需要降低

        # 安全限制
        if abs(delta_mu) > 1.0:
            delta_mu = np.sign(delta_mu) * 1.0
            mu = mu - delta_mu + mixing * residual / dos_mu
            mu = mu_init + (mu - mu_init) * 0.5  # 回退

        history['mu'].append(mu)
    else:
        history['converged'] = False
        history['iterations'] = max_iter

    history['mu_final'] = mu
    history['n_final'] = carrier_density(mu, energies, dos, temperature)

    return history


def newton_solver(n_target: float, energies: np.ndarray,
                  dos: np.ndarray, temperature: float,
                  mu_init: float = 0.0,
                  max_iter: int = 100, tol: float = 1e-12) -> Dict:
    """
    Newton-Raphson 法求解自洽化学势。

    F(μ) = n(μ) - n_target = 0
    F'(μ) = dn/dμ = ∫ D(E) (∂f/∂μ) dE

    Newton 步: μ_{n+1} = μ_n - F(μ_n)/F'(μ_n)

    二次收敛 (在根附近)。

    Parameters
    ----------
    n_target, energies, dos, temperature, mu_init, max_iter, tol

    Returns
    -------
    result : dict
    """
    mu = mu_init
    history = {'mu': [mu], 'residual': []}

    for it in range(max_iter):
        n_current = carrier_density(mu, energies, dos, temperature)
        F = n_current - n_target

        # F'(μ) = ∫ D(E) ∂f/∂μ dE
        dfdmu = fermi_dirac_derivative(energies, mu, temperature)
        F_prime = np.trapz(dos * dfdmu, energies)

        history['residual'].append(abs(F))

        if abs(F) < tol:
            history['converged'] = True
            history['iterations'] = it
            break

        if abs(F_prime) < 1e-30:
            # 导数太小, 切换到二分法
            break

        delta = F / F_prime
        mu = mu - delta

        # 安全限制: 步长不超过 0.5 eV
        if abs(delta) > 0.5:
            mu = mu + delta - 0.5 * np.sign(delta)

        history['mu'].append(mu)
    else:
        history['converged'] = False
        history['iterations'] = max_iter

    history['mu_final'] = mu
    history['n_final'] = carrier_density(mu, energies, dos, temperature)

    return history


def bisection_solver(n_target: float, energies: np.ndarray,
                     dos: np.ndarray, temperature: float,
                     mu_low: float = -1.0, mu_high: float = 1.0,
                     max_iter: int = 100, tol: float = 1e-12) -> Dict:
    """
    二分法求解自洽化学势 (最稳健的方法)。

    在 [μ_low, μ_high] 区间内搜索, 保证线性收敛。

    Parameters
    ----------
    n_target, energies, dos, temperature
    mu_low, mu_high : float
        搜索区间
    max_iter : int
    tol : float

    Returns
    -------
    result : dict
    """
    def F(mu):
        return carrier_density(mu, energies, dos, temperature) - n_target

    F_low = F(mu_low)
    F_high = F(mu_high)

    if F_low * F_high > 0:
        # 区间不包含根, 尝试扩展
        while F_low * F_high > 0 and mu_low > -10:
            mu_low -= 0.5
            mu_high += 0.5
            F_low = F(mu_low)
            F_high = F(mu_high)

        if F_low * F_high > 0:
            return {
                'converged': False,
                'mu_final': 0.5 * (mu_low + mu_high),
                'n_final': carrier_density(0.5*(mu_low+mu_high), energies, dos, temperature),
                'iterations': 0,
                'error': '搜索区间内无根'
            }

    history = {'mu': [], 'residual': []}

    for it in range(max_iter):
        mu_mid = 0.5 * (mu_low + mu_high)
        F_mid = F(mu_mid)

        history['mu'].append(mu_mid)
        history['residual'].append(abs(F_mid))

        if abs(F_mid) < tol or (mu_high - mu_low) < tol:
            return {
                'converged': True,
                'mu_final': mu_mid,
                'n_final': carrier_density(mu_mid, energies, dos, temperature),
                'iterations': it,
                'residual_history': history['residual']
            }

        if F_low * F_mid < 0:
            mu_high = mu_mid
            F_high = F_mid
        else:
            mu_low = mu_mid
            F_low = F_mid

    return {
        'converged': False,
        'mu_final': 0.5 * (mu_low + mu_high),
        'n_final': carrier_density(0.5*(mu_low+mu_high), energies, dos, temperature),
        'iterations': max_iter
    }


def self_consistent_solve(n_target: float, params_bhz,
                          temperature: float = 0.001,
                          method: str = 'newton',
                          Nk_dos: int = 30) -> Dict:
    """
    自洽求解的统一接口。

    Parameters
    ----------
    n_target : float
        目标载流子密度
    params_bhz : BHZParameters
    temperature : float
    method : str
        'fixed_point', 'newton', 'bisection'
    Nk_dos : int
        DOS 计算的 k 网格

    Returns
    -------
    result : dict
    """
    # 计算 DOS
    E_min = params_bhz.C - 2.0 * abs(params_bhz.M0) - 0.5
    E_max = params_bhz.C + 2.0 * abs(params_bhz.M0) + 0.5
    energies = np.linspace(E_min, E_max, 500)
    dos = density_of_states_bhz(energies, params_bhz, Nk_dos)

    # 求解
    if method == 'fixed_point':
        result = fixed_point_iteration(n_target, energies, dos, temperature)
    elif method == 'newton':
        result = newton_solver(n_target, energies, dos, temperature)
    elif method == 'bisection':
        result = bisection_solver(n_target, energies, dos, temperature)
    else:
        raise ValueError(f"未知方法: {method}")

    result['method'] = method
    result['n_target'] = n_target
    result['temperature'] = temperature
    result['energies'] = energies
    result['dos'] = dos

    return result


def self_consistency_report(result: Dict) -> str:
    """生成自洽求解的报告。"""
    lines = []
    lines.append("=" * 60)
    lines.append("自洽 Fermi 能级求解报告")
    lines.append("=" * 60)
    lines.append(f"求解方法: {result['method']}")
    lines.append(f"目标载流子密度: {result['n_target']:.6e}")
    lines.append(f"温度: {result['temperature']:.6f} eV")
    lines.append(f"\n结果:")
    lines.append(f"  收敛: {result.get('converged', 'N/A')}")
    lines.append(f"  迭代次数: {result.get('iterations', 'N/A')}")
    lines.append(f"  化学势 μ = {result['mu_final']:.10f} eV")
    lines.append(f"  实际密度 n = {result['n_final']:.6e}")
    lines.append(f"  密度误差 = {abs(result['n_final'] - result['n_target']):.6e}")

    if 'residual_history' in result:
        res = result['residual_history']
        if len(res) > 1:
            lines.append(f"\n  收敛历史 (前5步):")
            for i, r in enumerate(res[:5]):
                lines.append(f"    步 {i}: |F| = {r:.6e}")
            if len(res) > 5:
                lines.append(f"    ...")
                lines.append(f"    步 {len(res)-1}: |F| = {res[-1]:.6e}")

    return "\n".join(lines)
