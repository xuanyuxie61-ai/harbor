r"""
stability_coupling.py
=====================
流固耦合系统的稳定性分析与矩阵对数范数计算模块。

科学背景
--------
涡激振动流固耦合系统可线性化为耦合常微分方程组：

    \dot{U} = J U

其中 U = [\omega, \psi, X, \dot{X}]^T 为状态向量，J 为耦合 Jacobian 矩阵。
矩阵对数范数（logarithmic norm）\mu_p(J) 定义为：

    \mu_p(J) = \lim_{h \to 0^+} \frac{\|I + h J\|_p - 1}{h}

对 p=1, 2, \infty 有显式表达式：
- p=1:  \mu_1(J) = \max_j \left( \Re(J_{jj}) + \sum_{i \ne j} |J_{ij}| \right)
- p=2:  \mu_2(J) = \max_k \lambda_k \left( \frac{J + J^H}{2} \right)
- p=\infty: \mu_\infty(J) = \max_i \left( \Re(J_{ii}) + \sum_{j \ne i} |J_{ij}| \right)

稳定性判据
----------
若 \mu_p(J) < 0，则 \|\exp(J t)\|_p \le \exp(\mu_p(J) t) \to 0 (t \to \infty)，
系统渐近稳定。

在流固耦合中，Jacobian 包含：
- J_{ff}: 流体子系统（对流-扩散算子离散矩阵）
- J_{ss}: 结构子系统（质量-弹簧-阻尼）
- J_{fs}: 流体对结构的耦合（升力/阻力）
- J_{sf}: 结构对流体的耦合（网格运动、边界条件）

本模块同时提供 Gershgorin 圆盘估计与伪谱分析接口。

对应原种子项目：
- 697_log_norm（矩阵对数范数计算：l1, l2, l∞ 三种范数）
r"""

import numpy as np


def log_norm(A, p):
    r"""
    计算矩阵 A 的对数范数 \mu_p(A)。

    参数
    ----
    A : ndarray, shape (N, N)
        方阵，可为复数。
    p : {1, 2, np.inf}
        范数类型。

    返回
    ----
    mu : float
        对数范数值。

    边界处理
    --------
    若 p 不是 1, 2, inf，抛出 ValueError。
    若 A 不是方阵，抛出 ValueError。
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("log_norm: A 必须是方阵。")

    if p == 1:
        # \mu_1(A) = max_j ( Re(diag(A)) + sum_{i!=j} |A_{ij}| )
        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=0)
        mu = np.max(c + d)
    elif p == 2:
        # \mu_2(A) = max eigenvalue of 0.5*(A + A^H)
        B = 0.5 * (A + A.conj().T)
        eigvals = np.linalg.eigvalsh(B)
        mu = np.max(eigvals)
    elif p == np.inf:
        # \mu_\infty(A) = max_i ( Re(diag(A)) + sum_{j!=i} |A_{ij}| )
        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=1)
        mu = np.max(c + d)
    else:
        raise ValueError("log_norm: p 必须是 1, 2 或 np.inf。")

    return float(mu)


def compute_cfl_matrix_1d(v, k, dx):
    r"""
    构造一维对流扩散算子的离散 Jacobian（中心差分）。

    对应原种子项目 352_fd1d_advection_diffusion_steady 中的离散矩阵：
    A(i,i-1) = -v/(2dx) - k/dx^2
    A(i,i)   =  2k/dx^2
    A(i,i+1) =  v/(2dx) - k/dx^2
    """
    nx = int(np.ceil(1.0 / dx)) + 1
    A = np.zeros((nx, nx))
    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    for i in range(1, nx - 1):
        A[i, i - 1] = -v / (2.0 * dx) - k / (dx ** 2)
        A[i, i] = 2.0 * k / (dx ** 2)
        A[i, i + 1] = v / (2.0 * dx) - k / (dx ** 2)

    return A


def build_fsi_jacobian(nx, ny, nu, dx, dy, dt,
                       mass, damping, stiffness, rho_f, D_cyl,
                       coupling_gain=1.0):
    r"""
    构造简化流固耦合系统的 Jacobian 矩阵。

    模型假设：
    - 流体用一维尾流模型近似（von Karman 涡街简化）。
    - 结构为单自由度横向振动。
    - 耦合通过升力系数反馈实现。

    状态变量：U = [\omega_1, ..., \omega_{nx}, y, v_y]^T

    流体部分：\dot{\omega} = A_f \omega + B_f v_y  （结构速度影响尾流）
    结构部分：\dot{y} = v_y
               \dot{v_y} = -k/m * y - c/m * v_y + coupling_gain * C_L(\omega)

    其中 C_L 近似为升力系数对涡量的线性响应：C_L \approx \sum_i c_i \omega_i。
    """
    n_fluid = nx
    n_total = n_fluid + 2

    J = np.zeros((n_total, n_total))

    # 流体子矩阵（简化的一维对流扩散，周期边界）
    v_conv = 1.0  # 特征对流速度
    for i in range(n_fluid):
        im = (i - 1) % n_fluid
        ip = (i + 1) % n_fluid
        J[i, im] = -v_conv / (2.0 * dx) - nu / (dx ** 2)
        J[i, i] = 2.0 * nu / (dx ** 2)
        J[i, ip] = v_conv / (2.0 * dx) - nu / (dx ** 2)

    # 结构对流体的耦合（结构速度改变尾流）
    # 简化：结构速度 v_y 在中心位置注入涡量
    center_i = n_fluid // 2
    J[center_i, n_total - 1] = coupling_gain * 0.1

    # 结构子矩阵
    J[n_total - 2, n_total - 1] = 1.0  # \dot{y} = v_y
    J[n_total - 1, n_total - 2] = -stiffness / mass
    J[n_total - 1, n_total - 1] = -damping / mass

    # 流体对结构的耦合（升力）
    lift_sensitivity = coupling_gain * (0.5 * rho_f * v_conv ** 2 * D_cyl) / mass
    for i in range(n_fluid):
        J[n_total - 1, i] = lift_sensitivity * np.sin(2.0 * np.pi * i / n_fluid) / n_fluid

    return J


def gershgorin_bounds(A):
    r"""
    Gershgorin 圆盘估计特征值范围。
    对每个对角元 a_{ii}，圆盘中心为 a_{ii}，半径为 R_i = \sum_{j \ne i} |a_{ij}|。

    返回
    ----
    lambda_min_est, lambda_max_est : float
        特征值实部的下界与上界估计。
    """
    n = A.shape[0]
    centers = np.real(np.diag(A))
    radii = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))

    lambda_min_est = np.min(centers - radii)
    lambda_max_est = np.max(centers + radii)
    return lambda_min_est, lambda_max_est


def pseudospectrum_abscissa(A, epsilon=1e-3, num_points=100):
    r"""
    计算矩阵 A 的 \epsilon-伪谱横坐标：

        \alpha_\epsilon(A) = \max \{ \Re(z) : z \in \Lambda_\epsilon(A) \}

    其中 \Lambda_\epsilon(A) = \{ z \in \mathbb{C} : \sigma_{min}(zI - A) \le \epsilon \}。

    采用随机采样近似（简化版，用于快速评估）。
    """
    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    lambda_max_real = np.max(np.real(eigvals))

    # 在最大实部特征值附近采样
    center = lambda_max_real + 1j * np.max(np.imag(eigvals))
    best = lambda_max_real

    for _ in range(num_points):
        z = center + epsilon * (np.random.randn() + 1j * np.random.randn())
        M = z * np.eye(n) - A
        sigma_min = np.min(np.linalg.svd(M, compute_uv=False))
        if sigma_min <= epsilon:
            best = max(best, np.real(z))

    return best


def analyze_stability(nx=20, nu=0.01, mass=1.0, damping=0.1,
                      stiffness=10.0, rho_f=1.0, D_cyl=0.1):
    r"""
    对流固耦合系统进行稳定性分析并打印报告。
    """
    dx = 1.0 / (nx - 1)
    dy = dx
    dt = 0.01

    J = build_fsi_jacobian(nx, 1, nu, dx, dy, dt,
                           mass, damping, stiffness, rho_f, D_cyl)

    mu_1 = log_norm(J, 1)
    mu_2 = log_norm(J, 2)
    mu_inf = log_norm(J, np.inf)

    lam_min, lam_max = gershgorin_bounds(J)
    eigvals = np.linalg.eigvals(J)
    spectral_abscissa = np.max(np.real(eigvals))

    report = {
        'mu_1': mu_1,
        'mu_2': mu_2,
        'mu_inf': mu_inf,
        'gershgorin_min': lam_min,
        'gershgorin_max': lam_max,
        'spectral_abscissa': spectral_abscissa,
        'stable_mu1': mu_1 < 0,
        'stable_spectral': spectral_abscissa < 0,
    }
    return report
