"""
声学Helmholtz方程有限差分求解与分段线性插值重建模块

基于种子项目 362_fd1d_heat_steady 和 926_pwl_interp_1d 的核心算法，
为超声层析成像提供一维/二维Helmholtz方程的高效数值求解。

物理模型:
时谐声波在介质中传播满足Helmholtz方程:
    ∇²p + k²·p = -f(x)
其中 p 为声压，k = ω/c = 2πf/c 为波数，c 为声速，f(x) 为源项。

在频域中，Helmholtz方程的有限差分离散化为稀疏线性系统:
    (L + k²·I)·p = -f
其中 L 为离散的Laplacian算子。

边界条件:
- Dirichlet: p = 0 （软边界，压力释放面）
- Neumann: ∂p/∂n = 0 （硬边界，刚性壁面）
- Sommerfeld辐射: ∂p/∂n = ik·p （吸收边界条件，ABC）

核心公式:
- 二阶中心差分: p''(xᵢ) ≈ (p_{i-1} - 2p_i + p_{i+1}) / h²
- 截断误差: O(h²)
- 色散关系: k_num² = (2/h)²·sin²(kh/2)，数值波数 k_num ≈ k - k³h²/24
"""

import numpy as np
from typing import Tuple, Callable
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_helmholtz_1d(n: int, k: float, f: np.ndarray,
                       xlim: Tuple[float, float] = (0.0, 1.0),
                       bc_left: str = 'dirichlet',
                       bc_right: str = 'abc') -> Tuple[np.ndarray, np.ndarray]:
    """求解一维Helmholtz方程的有限差分方法。
    
    方程: p''(x) + k²·p(x) = -f(x),  x ∈ [a, b]
    
    离散化后得到三对角系统 A·p = -f，其中:
        A_{i,i}   = -2/h² + k²
        A_{i,i±1} =  1/h²
    
    对于吸收边界条件(ABC):
        (p_{N} - p_{N-1})/h = ik·p_N  =>  p_{N-1} = (1 - ik·h)·p_N
    
    参数:
        n: 内部节点数
        k: 波数 (1/m)
        f: 源项数组 (n,)
        xlim: 空间范围 (m)
        bc_left: 左边界条件类型
        bc_right: 右边界条件类型
    
    返回:
        x: 空间网格 (包含边界)
        p: 声压解 (包含边界)
    """
    a, b = xlim
    h = (b - a) / (n + 1)
    x = np.linspace(a, b, n + 2)
    
    # TODO: Hole_1 — 请补全一维Helmholtz方程的有限差分离散求解
    # 需要完成:
    # 1. 构建三对角离散矩阵（使用复数类型支持ABC边界条件）
    #    main_diag[i] = -2/h^2 + k^2, off_diag[i] = 1/h^2
    # 2. 根据 bc_left/bc_right 处理边界条件:
    #    - dirichlet: 边界值设为0
    #    - neumann: 镜像点处理，main_diag[0] += 1/h^2
    #    - abc: Sommerfeld吸收边界，main_diag[0] += (1-1j*k*h)/h^2
    # 3. 求解稀疏线性系统 A·p = -f（注意条件数检查）
    # 4. 组装包含边界的复数声压解 p（含边界值计算）
    
    p = np.zeros(n + 2, dtype=complex)  # 占位，需替换为正确实现
    return x, p


def solve_helmholtz_2d_dirichlet(nx: int, ny: int, k: float,
                                 f: np.ndarray,
                                 xlim: Tuple[float, float] = (0.0, 1.0),
                                 ylim: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """求解二维Helmholtz方程的有限差分方法（Dirichlet边界）。
    
    方程: ∇²p + k²·p = -f(x,y), 在矩形域内
    
    五点差分格式:
        (p_{i-1,j} + p_{i+1,j} + p_{i,j-1} + p_{i,j+1} - 4p_{i,j})/h² + k²·p_{i,j} = -f_{i,j}
    
    参数:
        nx, ny: x和y方向的内部节点数
        k: 波数 (1/m)
        f: 二维源项 (ny, nx)
        xlim, ylim: 空间范围
    
    返回:
        x, y: 空间网格
        p: 二维声压场 (ny, nx)
    """
    a, b = xlim
    c, d = ylim
    hx = (b - a) / (nx + 1)
    hy = (d - c) / (ny + 1)
    
    x = np.linspace(a, b, nx + 2)
    y = np.linspace(c, d, ny + 2)
    
    N = nx * ny
    
    # 构建块三对角矩阵
    # 主对角块: 三对角，对角元素 -2/hx² - 2/hy² + k²
    main_val = -2.0 / hx**2 - 2.0 / hy**2 + k**2
    off_x = 1.0 / hx**2
    off_y = 1.0 / hy**2
    
    # 逐行构建稀疏矩阵
    main_diag = np.full(N, main_val)
    off_x_diag = np.full(N - 1, off_x)
    off_y_diag = np.full(N - nx, off_y)
    
    # 处理x方向块边界（每行末尾不应有x方向的off-diagonal连接）
    for j in range(1, ny):
        off_x_diag[j * nx - 1] = 0.0
    
    A = diags([off_y_diag, off_x_diag, main_diag, off_x_diag, off_y_diag],
              [-nx, -1, 0, 1, nx], format='csr')
    
    # 展平rhs
    rhs = -f.flatten()
    
    # 求解
    p_flat = spsolve(A, rhs)
    p = p_flat.reshape(ny, nx)
    
    return x, y, p


def pwl_interp_1d(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """一维分段线性插值（Piecewise Linear Interpolation）。
    
    在每个区间 [x_k, x_{k+1}] 上，插值公式为:
        P(x) = y_k + (y_{k+1} - y_k) · (x - x_k) / (x_{k+1} - x_k)
    
    等价于帽子函数的线性组合:
        P(x) = Σ y_j · φ_j(x)
    其中 φ_j 为分段线性基函数（hat function）。
    
    参数:
        xd: 已知数据点的x坐标（必须有序）
        yd: 已知数据点的y值
        xi: 插值点x坐标
    
    返回:
        yi: 插值点y值
    """
    if len(xd) != len(yd):
        raise ValueError("xd和yd长度必须相同")
    
    if len(xd) < 2:
        raise ValueError("至少需要2个数据点")
    
    # 检查有序性
    if not np.all(np.diff(xd) > 0):
        # 尝试排序
        sort_idx = np.argsort(xd)
        xd = xd[sort_idx]
        yd = yd[sort_idx]
    
    n = len(xd)
    yi = np.zeros(len(xi))
    
    for i, x in enumerate(xi):
        # 边界处理：外推时clamp到边界区间
        if x <= xd[0]:
            k = 0
        elif x >= xd[n - 2]:
            k = n - 2
        else:
            # 二分查找所在区间
            k = np.searchsorted(xd, x) - 1
            k = max(0, min(k, n - 2))
        
        # 线性插值
        dx = xd[k + 1] - xd[k]
        if abs(dx) < 1e-14:
            yi[i] = yd[k]
        else:
            t = (x - xd[k]) / dx
            t = max(0.0, min(1.0, t))  # 数值鲁棒性
            yi[i] = (1.0 - t) * yd[k] + t * yd[k + 1]
    
    return yi


def pwl_basis_1d(xd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """生成分段线性帽子函数基函数矩阵。
    
    基函数 φ_j(x) 满足:
    - φ_j(x_j) = 1
    - φ_j(x_{j±1}) = 0
    - 在 [x_{j-1}, x_j] 上线性递增
    - 在 [x_j, x_{j+1}] 上线性递减
    
    返回的矩阵 B 满足: B[i,j] = φ_j(xi[i])
    """
    if not np.all(np.diff(xd) > 0):
        sort_idx = np.argsort(xd)
        xd = xd[sort_idx]
    
    n_basis = len(xd)
    n_eval = len(xi)
    B = np.zeros((n_eval, n_basis))
    
    for i, x in enumerate(xi):
        # 找到x所在的区间
        if x <= xd[0]:
            B[i, 0] = 1.0
        elif x >= xd[-1]:
            B[i, -1] = 1.0
        else:
            k = np.searchsorted(xd, x) - 1
            k = max(0, min(k, n_basis - 2))
            
            dx = xd[k + 1] - xd[k]
            if abs(dx) < 1e-14:
                B[i, k] = 1.0
            else:
                t = (x - xd[k]) / dx
                B[i, k] = 1.0 - t
                B[i, k + 1] = t
    
    return B


def interpolate_acoustic_pressure(xd: np.ndarray, pd: np.ndarray,
                                  xi: np.ndarray) -> np.ndarray:
    """对声学声压场进行分段线性插值重建。
    
    用于将粗网格上的FEM/FDM声压解插值到细网格或传感器位置。
    
    参数:
        xd: 原始网格坐标
        pd: 原始声压值（复数）
        xi: 目标插值坐标
    
    返回:
        pi: 插值后的声压值
    """
    # TODO: Hole_2 — 请补全声学声压场的分段线性插值重建
    # 提示: 复数声压场需分别对实部和虚部进行分段线性插值
    # pi_real = pwl_interp_1d(xd, np.real(pd), xi)
    # pi_imag = pwl_interp_1d(xd, np.imag(pd), xi)
    
    return np.zeros_like(xi, dtype=complex)  # 占位，需替换为正确实现
