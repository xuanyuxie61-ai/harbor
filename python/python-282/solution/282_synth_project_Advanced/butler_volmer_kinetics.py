"""
butler_volmer_kinetics.py
=========================

Butler-Volmer 电化学动力学与 Newton 非线性求解模块。

融合种子项目：
    125_burgers_steady_viscous：Newton 迭代求解非线性离散系统

核心公式（Butler-Volmer 方程）：
    j = j0 * [ exp(alpha_a * F * eta / (R*T))
              - exp(-alpha_c * F * eta / (R*T)) ]

其中：
    j    — 局部电流密度 [A/m^2]
    j0   — 交换电流密度 [A/m^2]
    eta  — 过电位 eta = phi_s - phi_e - U_eq [V]
    alpha_a, alpha_c — 阳极/阴极传递系数

Newton 求解的离散非线性方程组（稳态 SEI 生长）：
    F_i(u) = 0.5*(u_{i+1}^2 - u_{i-1}^2)/(2 dx)
             - nu * (u_{i+1} - 2u_i + u_{i-1})/dx^2 = 0
对应 SEI 中稳态 Nernst-Planck 方程的非线性对流-扩散平衡。

作者: DA-Synthesis
"""

import math
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


# ============================================================
#  Butler-Volmer 动力学核心函数
# ============================================================

def butler_volmer_current(eta, j0, alpha_a, alpha_c, temperature=None):
    """
    计算 Butler-Volmer 电流密度。

    数学形式：
        j = j0 * [exp(α_a F η / (RT)) - exp(-α_c F η / (RT))]

    Parameters
    ----------
    eta : float
        过电位 [V]。
    j0 : float
        交换电流密度 [A/m^2]。
    alpha_a : float
        阳极传递系数。
    alpha_c : float
        阴极传递系数。
    temperature : float or None
        温度 [K]，None 时使用 T_OPER。

    Returns
    -------
    float
        电流密度 [A/m^2]。
    """
    if temperature is None:
        temperature = P.T_OPER
    vt = P.R_GAS * temperature / P.FARADAY
    if abs(vt) < 1.0e-30:
        return 0.0

    # 数值安全：限制指数参数范围
    arg_a = alpha_a * eta / vt
    arg_c = alpha_c * eta / vt
    arg_a = max(-80.0, min(80.0, arg_a))
    arg_c = max(-80.0, min(80.0, arg_c))

    return j0 * (math.exp(arg_a) - math.exp(-arg_c))


def butler_volmer_derivative(eta, j0, alpha_a, alpha_c, temperature=None):
    """
    Butler-Volmer 电流对过电位的导数 dj/dη（用于 Newton 迭代）。

    数学形式：
        dj/dη = j0 * [α_a F/(RT) exp(α_a F η/(RT))
                      + α_c F/(RT) exp(-α_c F η/(RT))]

    Parameters
    ----------
    eta : float
        过电位。
    j0 : float
        交换电流密度。
    alpha_a, alpha_c : float
        传递系数。
    temperature : float or None
        温度。

    Returns
    -------
    float
        dj/dη [A/(m^2 V)]。
    """
    if temperature is None:
        temperature = P.T_OPER
    vt = P.R_GAS * temperature / P.FARADAY
    if abs(vt) < 1.0e-30:
        return 0.0

    arg_a = max(-80.0, min(80.0, alpha_a * eta / vt))
    arg_c = max(-80.0, min(80.0, alpha_c * eta / vt))
    fac_a = alpha_a / vt * math.exp(arg_a)
    fac_c = alpha_c / vt * math.exp(-arg_c)

    return j0 * (fac_a + fac_c)


def sei_growth_rate(phi_s, phi_e, c_li_surface, temperature=None):
    """
    计算 SEI 生长速率（局部电流密度 / (n F)）。

    数学形式（Faraday 定律）：
        v_sei = j_sei / (n * F) * M_mol / rho_sei
    其中 j_sei 为 SEI 形成反应的 Butler-Volmer 电流。

    Parameters
    ----------
    phi_s : float
        固相电位 [V]。
    phi_e : float
        液相电位 [V]。
    c_li_surface : float
        表面 Li+ 浓度 [mol/m^3]。
    temperature : float or None
        温度 [K]。

    Returns
    -------
    float
        SEI 生长速率 [m/s]。
    """
    eta = phi_s - phi_e - P.U_EQ_SEI
    # 浓度依赖的交换电流密度
    c_ref = P.C_LI_INIT
    j0_sei = P.FARADAY * P.K0_SEI * (c_li_surface / c_ref) ** P.ALPHA_A_SEI
    j_sei = butler_volmer_current(
        eta, j0_sei, P.ALPHA_A_SEI, P.ALPHA_C_SEI, temperature
    )
    n_electron = 2  # SEI 形成涉及 2 电子转移
    v_sei = j_sei * P.M_MOL_SEI / (n_electron * P.FARADAY * P.RHO_SEI)
    return v_sei


# ============================================================
#  稳态非线性问题 Newton 求解器（参考 burgers_steady_viscous）
# ============================================================

def solve_steady_state_np(alpha_left, beta_right, nu_eff, n_nodes, dx,
                          max_newton_steps=50, resid_tol=1.0e-8,
                          step_tol=1.0e-10):
    """
    使用 Newton 方法求解稳态 Nernst-Planck 方程的离散非线性系统。

    数学模型（稳态 NP 方程）：
        d/dx [D dc/dx - mu c dφ/dx] = 0
    在简化 1D 情形下，等价于：
        D c'' - v c' = 0  (对流-扩散)

    离散形式（参考 burgers_steady_viscous.m）：
        F_i = 0.5*(u_{i+1}^2 - u_{i-1}^2)/(2 dx)
              - nu * (u_{i+1} - 2 u_i + u_{i-1})/dx^2 = 0

    Jacobian 矩阵（三对角稀疏）：
        J_{i,i-1} = -u_{i-1}/(2 dx) - nu/dx^2
        J_{i,i}   = 2 nu/dx^2
        J_{i,i+1} = u_{i+1}/(2 dx) - nu/dx^2

    Parameters
    ----------
    alpha_left : float
        左边界 Dirichlet 值。
    beta_right : float
        右边界 Dirichlet 值。
    nu_eff : float
        有效粘度/扩散系数。
    n_nodes : int
        网格节点数。
    dx : float
        空间步长。
    max_newton_steps : int
        Newton 最大迭代次数。
    resid_tol : float
        残差收敛容限。
    step_tol : float
        步长收敛容限。

    Returns
    -------
    u : list[float]
        收敛后的解。
    n_steps : int
        实际 Newton 步数。
    converged : bool
        是否收敛。
    residual_history : list[float]
        每步残差历史。
    """
    if n_nodes < 3:
        raise ValueError("Newton 求解至少需要 3 个节点")
    if abs(nu_eff) < 1.0e-30:
        raise ValueError("nu_eff 不能为零")

    # 初始猜测：线性插值
    u = [alpha_left + (beta_right - alpha_left) * i / (n_nodes - 1)
         for i in range(n_nodes)]

    residual_history = []
    newton_step = 0
    converged = False

    while newton_step <= max_newton_steps:
        # 计算残差 F(u)
        f = [0.0] * n_nodes
        f[0] = u[0] - alpha_left
        for i in range(1, n_nodes - 1):
            f[i] = (0.5 * (u[i + 1] ** 2 - u[i - 1] ** 2) / (2.0 * dx)
                    - nu_eff * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx))
        f[n_nodes - 1] = u[n_nodes - 1] - beta_right

        f_norm = max(abs(fi) for fi in f)
        residual_history.append(f_norm)

        if f_norm < resid_tol:
            converged = True
            break

        # 构造三对角 Jacobian 并求解 J du = -f
        # Thomas 算法（三对角矩阵求解）
        a_diag = [0.0] * n_nodes   # 下次对角
        b_diag = [0.0] * n_nodes   # 主对角
        c_diag = [0.0] * n_nodes   # 上次对角
        rhs = [0.0] * n_nodes      # 右端

        b_diag[0] = 1.0
        rhs[0] = -f[0]
        for i in range(1, n_nodes - 1):
            a_diag[i] = -2.0 * u[i - 1] / (4.0 * dx) - nu_eff / (dx * dx)
            b_diag[i] = 2.0 * nu_eff / (dx * dx)
            c_diag[i] = 2.0 * u[i + 1] / (4.0 * dx) - nu_eff / (dx * dx)
            rhs[i] = -f[i]
        b_diag[n_nodes - 1] = 1.0
        rhs[n_nodes - 1] = -f[n_nodes - 1]

        # Thomas 算法
        du = _thomas_solve(a_diag, b_diag, c_diag, rhs)

        du_norm = max(abs(d) for d in du)
        u_norm = max(abs(ui) for ui in u)
        if du_norm < step_tol * (u_norm + 1.0):
            converged = True
            break

        # 更新解: delta = J^{-1}(-F), u <- u + delta
        u = [u[i] + du[i] for i in range(n_nodes)]
        newton_step += 1

    return u, newton_step, converged, residual_history


def _thomas_solve(a, b, c, d):
    """
    Thomas 算法求解三对角线性方程组。

    数学形式：
        a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i

    Parameters
    ----------
    a, b, c, d : list[float]
        三对角矩阵的三个对角线和右端向量。

    Returns
    -------
    x : list[float]
        解向量。
    """
    n = len(b)
    c_star = [0.0] * n
    d_star = [0.0] * n

    if abs(b[0]) < 1.0e-30:
        raise ValueError("Thomas 算法主元为零")

    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n):
        m = a[i] / (b[i] - a[i] * c_star[i - 1]) if i < n else 0.0
        denom = b[i] - a[i] * c_star[i - 1]
        if abs(denom) < 1.0e-30:
            c_star[i] = 0.0
            d_star[i] = 0.0
        else:
            c_star[i] = c[i] / denom if i < n - 1 else 0.0
            d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom

    x = [0.0] * n
    x[n - 1] = d_star[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
    return x


if __name__ == "__main__":
    # 测试 Butler-Volmer 电流
    eta_test = 0.05  # V
    j_test = butler_volmer_current(
        eta_test, 1.0, P.ALPHA_A_GRAPHITE, P.ALPHA_C_GRAPHITE
    )
    dj_test = butler_volmer_derivative(
        eta_test, 1.0, P.ALPHA_A_GRAPHITE, P.ALPHA_C_GRAPHITE
    )
    print(f"[butler_volmer] eta = {eta_test*1000:.1f} mV")
    print(f"  j  = {j_test:.6e} A/m^2")
    print(f"  dj = {dj_test:.6e} A/(m^2 V)")

    # 测试 Newton 求解器
    u_sol, n_step, conv, res_hist = solve_steady_state_np(
        alpha_left=0.01, beta_right=0.0,
        nu_eff=1.0, n_nodes=21, dx=0.05
    )
    print(f"\n[butler_volmer] Newton 稳态求解:")
    print(f"  收敛: {conv}, 步数: {n_step}")
    if res_hist:
        print(f"  初始残差: {res_hist[0]:.6e}")
        print(f"  最终残差: {res_hist[-1]:.6e}")
