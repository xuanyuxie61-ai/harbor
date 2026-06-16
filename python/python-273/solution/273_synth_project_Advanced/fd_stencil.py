"""
fd_stencil.py — 高阶有限差分模板与紧致差分格式
===============================================

融合种子项目:
  - 368_fd2d_poisson: 2D Poisson 方程 5 点差分模板
  - 004_alpert_rule: Alpert 混合 Gauss-梯形求积规则 (高阶精度)
  - 170_chinese_remainder_theorem: CRT 索引分解 -> 多维模板索引映射

物理背景:
  声子动力学方程中的空间离散化需要高阶精度以正确再现
  色散关系的解析结构。标准 2 阶中心差分引入 O(h^2) 色散误差,
  而 6 阶/8 阶模板可将误差降至 O(h^6)/O(h^8)。

核心公式:
  2阶中心差分: f''(x) ≈ (-f_{i+1} + 2f_i - f_{i-1}) / h^2
  4阶中心差分: f''(x) ≈ (-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2}) / (12h^2)
  6阶中心差分: 7 点模板, 系数 [-1/90, 3/20, -3/2, 49/18, -3/2, 3/20, -1/90] / h^2
  8阶中心差分: 9 点模板, 系数 [1/560, -8/315, 1/5, -8/5, 205/72, -8/5, 1/5, -8/315, 1/560] / h^2

  紧致 (Pade) 差分:
    alpha * f''_{i-1} + f''_i + alpha * f''_{i+1}
      = a * (f_{i+1} - 2f_i + f_{i-1}) / h^2
    其中 alpha = 1/10, a = 6/5 -> 6阶紧致格式
"""

import numpy as np
from typing import Tuple, List, Dict


# 预计算的高阶中心差分系数 (二阶导数)
FD_STENCIL_2ND = np.array([1.0, -2.0, 1.0])  # O(h^2)
FD_STENCIL_4TH = np.array([
    -1.0 / 12, 16.0 / 12, -30.0 / 12, 16.0 / 12, -1.0 / 12
])  # O(h^4)
FD_STENCIL_6TH = np.array([
    1.0 / 90, -6.0 / 90, 15.0 / 90,
    -20.0 / 90,  # 注意: 中心系数为 (2*1 + 2*6 + 2*15 + (-40))/90 的补
    # 修正为标准6阶系数
])
# 精确6阶系数
FD_STENCIL_6TH = np.array([
    1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0,
    -49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0
])
# 精确8阶系数
FD_STENCIL_8TH = np.array([
    -1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
    -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0
])


def get_fd_stencil(order: int) -> np.ndarray:
    """获取指定阶数的中心差分模板 (二阶导数)"""
    stencil_map = {2: FD_STENCIL_2ND, 4: FD_STENCIL_4TH,
                   6: FD_STENCIL_6TH, 8: FD_STENCIL_8TH}
    if order not in stencil_map:
        raise ValueError(f"不支持的精度阶数: {order}, 可选: {list(stencil_map.keys())}")
    return stencil_map[order].copy()


def build_1d_laplacian_matrix(
    n_points: int, h: float, order: int = 4,
    bc_type: str = 'dirichlet',
) -> np.ndarray:
    """
    构建 1D Laplacian 矩阵 (融合 fd2d_poisson 的矩阵组装)。

    -Delta u = f, 使用 order 阶差分模板。

    边界条件:
      'dirichlet': u(0) = u(L) = 0
      'periodic': u(0) = u(L), u'(0) = u'(L)
      'neumann': u'(0) = u'(L) = 0

    参数:
        n_points: 内部格点数
        h: 网格间距
        order: 差分精度阶数 (2/4/6/8)
        bc_type: 边界条件类型

    返回:
        L: (n_points, n_points) 离散 Laplacian 矩阵
    """
    stencil = get_fd_stencil(order)
    half_width = len(stencil) // 2
    # 模板乘以 1/h^2
    scaled_stencil = stencil / h ** 2

    L = np.zeros((n_points, n_points))

    for i in range(n_points):
        for k, s in enumerate(stencil):
            j = i + k - half_width
            if bc_type == 'periodic':
                j_mod = j % n_points
                L[i, j_mod] += s
            elif bc_type == 'dirichlet':
                if 0 <= j < n_points:
                    L[i, j] += s
                # 边界外的项归入右端项 (此处矩阵中不体现)
            elif bc_type == 'neumann':
                if 0 <= j < n_points:
                    L[i, j] += s
                elif j < 0:
                    # 镜像: u_{-1} = u_1 -> 贡献到 u_1
                    L[i, -j] += s
                elif j >= n_points:
                    # 镜像: u_{N} = u_{N-2}
                    mirror = 2 * (n_points - 1) - j
                    if 0 <= mirror < n_points:
                        L[i, mirror] += s

    return L


def build_2d_laplacian_matrix(
    nx: int, ny: int, hx: float, hy: float,
    order: int = 2, bc_type: str = 'dirichlet',
) -> np.ndarray:
    """
    构建 2D Laplacian 矩阵 (直接融合 fd2d_poisson)。
    5 点模板 (order=2) 或 9 点模板 (order=4)。

    融合 fd2d_poisson 的 eqn = j*nx + i 映射。

    返回:
        L: (nx*ny, nx*ny) 离散 Laplacian
    """
    N = nx * ny
    L = np.zeros((N, N))

    if order == 2:
        for j in range(ny):
            for i in range(nx):
                eqn = j * nx + i
                is_boundary = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1)
                if is_boundary and bc_type == 'dirichlet':
                    L[eqn, eqn] = 1.0
                else:
                    L[eqn, eqn] = 2.0 / hx ** 2 + 2.0 / hy ** 2
                    if i > 0:
                        if bc_type == 'dirichlet' and i - 1 == 0 and j in (0, ny - 1):
                            pass  # 角点
                        L[eqn, eqn - 1] = -1.0 / hx ** 2
                    if i < nx - 1:
                        L[eqn, eqn + 1] = -1.0 / hx ** 2
                    if j > 0:
                        L[eqn, eqn - nx] = -1.0 / hy ** 2
                    if j < ny - 1:
                        L[eqn, eqn + nx] = -1.0 / hy ** 2
    else:
        # 4阶: 使用 1D 4阶模板的 Kronecker 积
        Lx = build_1d_laplacian_matrix(nx, hx, order=4, bc_type=bc_type)
        Ly = build_1d_laplacian_matrix(ny, hy, order=4, bc_type=bc_type)
        Ix = np.eye(nx)
        Iy = np.eye(ny)
        L = np.kron(Iy, Lx) + np.kron(Ly, Ix)

    return L


def build_3d_laplacian_matrix(
    nx: int, ny: int, nz: int,
    hx: float, hy: float, hz: float,
    order: int = 2, bc_type: str = 'periodic',
) -> np.ndarray:
    """
    构建 3D Laplacian 矩阵。用于声子动力学的大规模稀疏求解。

    L_3D = L_x (x) I_y (x) I_z + I_x (x) L_y (x) I_z + I_x (x) I_y (x) L_z
    """
    Lx = build_1d_laplacian_matrix(nx, hx, order=order, bc_type=bc_type)
    Ly = build_1d_laplacian_matrix(ny, hy, order=order, bc_type=bc_type)
    Lz = build_1d_laplacian_matrix(nz, hz, order=order, bc_type=bc_type)
    Ix = np.eye(nx)
    Iy = np.eye(ny)
    Iz = np.eye(nz)
    L = np.kron(Iz, np.kron(Iy, Lx)) + np.kron(Iz, np.kron(Ly, Ix)) + np.kron(Lz, np.kron(Iy, Ix))
    return L


def compute_fd_error_order(
    func, d2func, x0: float, h_values: np.ndarray,
    order: int,
) -> Dict[str, float]:
    """
    数值验证差分模板的精度阶数。
    计算 ||D_h^2 f - f''|| 随 h 的收敛率。

    融合 alpert_rule 的验证思想: 用已知精确解检验数值格式。

    返回:
        字典: {'errors': array, 'observed_order': float}
    """
    stencil = get_fd_stencil(order)
    half_w = len(stencil) // 2
    errors = []
    for h in h_values:
        pts = x0 + (np.arange(len(stencil)) - half_w) * h
        f_vals = func(pts)
        d2_approx = np.dot(stencil, f_vals) / h ** 2
        d2_exact = d2func(x0)
        errors.append(abs(d2_approx - d2_exact))
    errors = np.array(errors)
    # 收敛阶: p ≈ log(e1/e2) / log(h1/h2)
    if len(errors) >= 2 and errors[0] > 1e-16 and errors[-1] > 1e-16:
        obs_order = np.log(errors[0] / errors[-1]) / np.log(h_values[-1] / h_values[0])
    else:
        obs_order = 0.0
    return {'errors': errors, 'observed_order': obs_order}


def alpert_quadrature_nodes_weights(
    rule_type: str = 'regular', rule_index: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alpert 混合 Gauss-梯形求积节点与权重 (融合 alpert_rule)。

    用于声子 DOS 中奇异积分的计算:
      DOS(omega) = integral_delta(omega - omega(k)) dk
    在 van Hove 奇点附近，被积函数具有可积奇异性,
    需要特殊求积规则。

    参数:
        rule_type: 'regular', 'power', 'log'
        rule_index: 规则编号

    返回:
        nodes, weights: 求积节点和权重
    """
    # 简化的 Alpert 规则实现
    # 使用 Gauss-Legendre 节点近似
    if rule_type == 'regular':
        n_pts = max(4, rule_index * 2)
        nodes, weights = np.polynomial.legendre.leggauss(n_pts)
        # 映射到 [0, 1]
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights
    elif rule_type == 'power':
        # 半奇性: x^(-1/2) 型, Gauss-Jacobi 近似
        n_pts = max(4, rule_index * 2)
        nodes, weights = np.polynomial.legendre.leggauss(n_pts)
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights * np.sqrt(nodes + 1e-15)
    elif rule_type == 'log':
        # 对数奇性: log(x) 型
        n_pts = max(4, rule_index * 2)
        nodes, weights = np.polynomial.legendre.leggauss(n_pts)
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights
        # 对数修正
        correction = np.where(nodes > 1e-10, -nodes * np.log(nodes + 1e-15), 0.0)
        weights = weights + 0.01 * correction
    else:
        raise ValueError(f"未知 Alpert 规则类型: {rule_type}")

    return nodes, weights


def spectral_analysis_fd(
    stencil: np.ndarray, h: float, n_k: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    有限差分模板的谱分析 (修正波数 vs 真实波数)。

    对模板 S = {s_j}, 离散符号色散关系为:
      k_modified^2 * h^2 = -sum_j s_j * exp(i * k * j * h) / h^2

    修正波数 kh_eff 满足:
      (kh_eff)^2 = -sum_j s_j * exp(i * j * kh)

    融合 声子色散的 k.h 参数化。
    """
    kh = np.linspace(0, np.pi, n_k)
    half_w = len(stencil) // 2
    kh_eff_sq = np.zeros(n_k, dtype=complex)
    for j, s in enumerate(stencil):
        m = j - half_w
        kh_eff_sq += s * np.exp(1j * m * kh)
    kh_eff_sq = -kh_eff_sq
    kh_eff = np.sqrt(np.maximum(np.real(kh_eff_sq), 0.0))
    return kh, kh_eff
