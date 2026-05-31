"""
romberg_integrator.py
=====================
多维数值积分器：Romberg 积分与 Monte Carlo 积分。

融合种子项目：
  - 805_nintlib : Romberg_nd, monte_carlo_nd, box_nd, p5_nd, sum2_nd, tuple_next

科学应用：
  在非线性声学中，高维数值积分用于计算：
  - 声束总能量 :math:`E = \int \int \int \frac{p^2}{2 \rho_0 c_0^2} dV dt`
  - 频谱能量密度
  - 熵产生率的空间积分
  - 传感器阵列的协方差矩阵期望
"""

import numpy as np


def tuple_next(m, base, tuple_vec):
    """
    生成元组序列的下一个元素。

    原始算法来自 805_nintlib/tuple_next.m。
    用于多维积分的多重循环索引生成。

    Parameters
    ----------
    m : int
        元组维度。
    base : int
        每维基数（1-based）。
    tuple_vec : np.ndarray, shape (m,)
        当前元组。

    Returns
    -------
    tuple_vec : np.ndarray
        下一个元组。
    rank : int
        若为 -1 表示序列结束。
    """
    tuple_vec = np.asarray(tuple_vec, dtype=int)
    rank = -1
    for i in range(m - 1, -1, -1):
        if tuple_vec[i] < base:
            tuple_vec[i] += 1
            rank = 0
            return tuple_vec, rank
        tuple_vec[i] = 1
    return tuple_vec, rank


def monte_carlo_nd(func, a, b, dim_num, n_eval):
    """
    多维 Monte Carlo 积分。

    原始算法来自 805_nintlib/monte_carlo_nd.m。

    .. math::
        I \approx \frac{V}{N} \sum_{i=1}^{N} f(x_i)

    其中 :math:`V = \prod_{j=1}^{d} (b_j - a_j)`。

    Parameters
    ----------
    func : callable
        被积函数 func(x) -> float，x 为 shape (dim_num,) 的数组。
    a : np.ndarray, shape (dim_num,)
        积分下限。
    b : np.ndarray, shape (dim_num,)
        积分上限。
    dim_num : int
        维度。
    n_eval : int
        采样点数。

    Returns
    -------
    float
        积分估计值。
    float
        标准误差估计。
    int
        函数求值次数。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    volume = np.prod(b - a)
    total = 0.0
    total_sq = 0.0

    for _ in range(n_eval):
        x = a + np.random.rand(dim_num) * (b - a)
        fx = func(x)
        total += fx
        total_sq += fx ** 2

    mean = total / n_eval
    variance = (total_sq / n_eval) - mean ** 2
    std_err = volume * np.sqrt(variance / n_eval)
    result = volume * mean
    return result, std_err, n_eval


def romberg_nd(func, a, b, dim_num, sub_num, it_max, tol):
    """
    多维 Romberg 积分。

    原始算法来自 805_nintlib/romberg_nd.m。
    基于中点法则的 Richardson 外推。

    .. math::
        I_{m}^{(k)} = I_{m}^{(k-1)} +
        \frac{I_{m}^{(k-1)} - I_{m-1}^{(k-1)}}{(n_m / n_{m-1})^2 - 1}

    Parameters
    ----------
    func : callable
        被积函数 func(x) -> float。
    a : np.ndarray, shape (dim_num,)
        下限。
    b : np.ndarray, shape (dim_num,)
        上限。
    dim_num : int
        维度。
    sub_num : np.ndarray, shape (dim_num,)
        初始每维子区间数。
    it_max : int
        最大迭代次数。
    tol : float
        相对误差容差。

    Returns
    -------
    float
        积分结果。
    int
        收敛标志 (1=成功, -1=未收敛)。
    int
        函数求值次数。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sub_num = np.asarray(sub_num, dtype=int)

    if dim_num < 1:
        raise ValueError("dim_num must be >= 1.")
    if it_max < 1:
        raise ValueError("it_max must be >= 1.")
    if np.any(sub_num <= 0):
        raise ValueError("sub_num must be positive.")

    eval_num = 0
    ind = 0
    rnderr = np.finfo(float).eps
    iwork2 = np.zeros(it_max, dtype=int)
    iwork2[0] = 1
    if it_max > 1:
        iwork2[1] = 2

    sub_num2 = sub_num.copy()
    table = np.zeros(it_max, dtype=float)

    result = 0.0
    result_old = 0.0

    for it in range(it_max):
        weight = np.prod((b - a) / sub_num2)
        sum_val = 0.0

        # 生成所有中点并求值
        iwork = np.ones(dim_num, dtype=int)
        while True:
            x = ((2.0 * sub_num2 - 2.0 * iwork + 1.0) * a +
                 (2.0 * iwork - 1.0) * b) / (2.0 * sub_num2)
            sum_val += func(x)
            eval_num += 1

            kdim = dim_num - 1
            while kdim >= 0:
                if iwork[kdim] < sub_num2[kdim]:
                    iwork[kdim] += 1
                    break
                iwork[kdim] = 1
                kdim -= 1
            if kdim < 0:
                break

        table[it] = weight * sum_val

        if it == 0:
            result = table[0]
            result_old = result
            if it_max <= 1:
                ind = 1
                break
            if it_max > 1:
                sub_num2 = iwork2[it + 1] * sub_num2 if (it + 1) < it_max else sub_num2
            continue

        # Richardson 外推
        for ll in range(2, it + 2):
            i = it + 1 - ll
            factor = (iwork2[i] ** 2) / (iwork2[it] ** 2 - iwork2[i] ** 2)
            table[i] = table[i + 1] + (table[i + 1] - table[i]) * factor

        result = table[0]

        if abs(result - result_old) <= abs(result * (tol + rnderr)):
            ind = 1
            break

        if it >= it_max - 1:
            ind = -1
            break

        result_old = result
        if it + 1 < it_max:
            iwork2[it + 1] = round(1.5 * iwork2[it])
            sub_num2 = iwork2[it + 1] * sub_num2

    return result, ind, eval_num


class AcousticEnergyIntegrator:
    """
    封装声学能量相关的多维积分计算。
    """

    def __init__(self, physics):
        self.physics = physics

    def beam_energy_3d(self, p_func, r_max, z_max, tau_max,
                       n_samples=10000, method='monte_carlo'):
        r"""
        计算声束总能量：

        .. math::
            E = \int_{0}^{r_{max}} \int_{0}^{z_{max}} \int_{-\tau_{max}}^{\tau_{max}}
            \frac{p^2(r, z, \tau)}{2 \rho_0 c_0^2} \, 2\pi r \, d\tau \, dz \, dr

        Parameters
        ----------
        p_func : callable
            压力场函数 p(r, z, tau) -> float。
        r_max, z_max, tau_max : float
            积分上限。
        n_samples : int
            Monte Carlo 采样数。
        method : str
            'monte_carlo' 或 'romberg'。

        Returns
        -------
        float
            总能量 (J)。
        """
        rho0 = self.physics.rho0
        c0 = self.physics.c0
        prefactor = 1.0 / (2.0 * rho0 * c0 ** 2)

        if method == 'monte_carlo':
            def integrand(x):
                r, z, tau = x
                if r < 0.0 or z < 0.0 or abs(tau) > tau_max:
                    return 0.0
                p_val = p_func(r, z, tau)
                if not np.isfinite(p_val):
                    return 0.0
                return prefactor * p_val ** 2 * 2.0 * np.pi * r

            a = np.array([0.0, 0.0, -tau_max])
            b = np.array([r_max, z_max, tau_max])
            result, _, _ = monte_carlo_nd(integrand, a, b, 3, n_samples)
            return result

        elif method == 'romberg':
            def integrand(x):
                r, z, tau = x
                if r < 0.0:
                    r = 0.0
                p_val = p_func(r, z, tau)
                if not np.isfinite(p_val):
                    return 0.0
                return prefactor * p_val ** 2 * 2.0 * np.pi * r

            a = np.array([0.0, 0.0, -tau_max])
            b = np.array([r_max, z_max, tau_max])
            sub_num = np.array([4, 4, 4])
            result, ind, evals = romberg_nd(integrand, a, b, 3, sub_num, it_max=4, tol=1e-3)
            if ind != 1:
                # 若 Romberg 未收敛，回退到 Monte Carlo
                result, _, _ = monte_carlo_nd(integrand, a, b, 3, n_samples)
            return result
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def spatial_average_pressure(self, p_field, r_grid, z_grid):
        r"""
        计算空间平均压力（轴对称加权平均）：

        .. math::
            \bar{p}(z) = \frac{\int_0^{r_{max}} p(r,z) \, 2\pi r \, dr}{\pi r_{max}^2}

        Parameters
        ----------
        p_field : np.ndarray, shape (Nr, Nz)
            压力场。
        r_grid : np.ndarray, shape (Nr,)
            径向坐标。
        z_grid : np.ndarray, shape (Nz,)
            轴向坐标。

        Returns
        -------
        np.ndarray, shape (Nz,)
            轴向平均压力。
        """
        p_field = np.asarray(p_field, dtype=float)
        r_grid = np.asarray(r_grid, dtype=float)
        if p_field.ndim != 2:
            raise ValueError("p_field must be 2D.")
        Nr, Nz = p_field.shape
        if r_grid.size != Nr:
            raise ValueError("r_grid size must match p_field first dimension.")

        p_avg = np.zeros(Nz, dtype=float)
        for j in range(Nz):
            integrand = p_field[:, j] * 2.0 * np.pi * r_grid
            # 梯形法则
            p_avg[j] = np.trapezoid(integrand, r_grid) / (np.pi * r_grid[-1] ** 2)
        return p_avg
