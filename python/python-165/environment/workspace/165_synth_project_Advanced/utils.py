"""
utils.py
通用数值工具模块
融合种子项目：circle_arc_grid, change_polynomial, diff2_center
"""

import numpy as np
from typing import List, Tuple


def circle_arc_grid(cx: float, cy: float, r: float, theta_start: float,
                    theta_end: float, n: int) -> np.ndarray:
    """
    生成圆弧上的均匀网格点。

    参数化方程：
        x(θ) = c_x + r·cos(θ)
        y(θ) = c_y + r·sin(θ)
    其中 θ ∈ [θ_start, θ_end]（单位：度），被均匀剖分为 n 段。

    在智能电网中，用于生成环形配电网（ring distribution network）的
    母线节点坐标，使馈线沿圆周均匀分布。
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    theta_start_rad = np.deg2rad(theta_start)
    theta_end_rad = np.deg2rad(theta_end)
    thetas = np.linspace(theta_start_rad, theta_end_rad, n)
    x = cx + r * np.cos(thetas)
    y = cy + r * np.sin(thetas)
    return np.column_stack((x, y))


def polynomial_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    多项式卷积乘法（对应 change_polynomial 核心思想）。

    若 P(z) = Σ p_k z^k，Q(z) = Σ q_k z^k，则
        R(z) = P(z)·Q(z) = Σ r_n z^n
    其中
        r_n = Σ_{k=0}^{n} p_k · q_{n-k}

    在电力经济调度中，用于生成函数枚举机组启停组合的总成本分布。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.convolve(p, q)


def diff2_center(f: callable, x: float, h: float = 1e-5) -> float:
    """
    中心差分近似二阶导数：
        f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2

    截断误差为 O(h^2)。

    在潮流计算中用于数值验证雅可比矩阵 Hessian 项的正定性，
    以保证牛顿-拉夫逊迭代的局部二次收敛条件。
    """
    if h <= 0:
        raise ValueError("step size h must be positive")
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def triangle_area_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    计算二维三角形的有向面积（叉积公式）：
        S = 0.5 * | (b-a) × (c-a) |
          = 0.5 * | (x_b-x_a)(y_c-y_a) - (x_c-x_a)(y_b-y_a) |

    在电网三相不平衡分析中，用于计算电压相量形成的相量三角形面积，
    作为三相不平衡度的几何判据。
    """
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1])
                     - (c[0] - a[0]) * (b[1] - a[1]))


def triangle_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    利用余弦定理计算三角形内角：
        α = arccos( (b^2+c^2-a^2) / (2bc) )
    返回三个角 [α, β, γ]（单位：弧度）。
    """
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    lab = np.linalg.norm(b - a)
    lbc = np.linalg.norm(c - b)
    lca = np.linalg.norm(a - c)
    alpha = np.arccos(np.clip((lab**2 + lca**2 - lbc**2)
                      / (2 * lab * lca + 1e-15), -1.0, 1.0))
    beta = np.arccos(np.clip((lab**2 + lbc**2 - lca**2)
                     / (2 * lab * lbc + 1e-15), -1.0, 1.0))
    gamma = np.pi - alpha - beta
    return np.array([alpha, beta, gamma])


def i4mat_rref(a: np.ndarray) -> np.ndarray:
    """
    整数矩阵的 Reduced Row Echelon Form (RREF)。

    采用高斯-约当消元，避免浮点舍入：
        1) 选主元（非零元）。
        2) 用 gcd 将主元行归一化为互质整数。
        3) 消去该列所有其他行。

    在电网状态估计中，用于检测量测冗余度和可观测性分析：
    量测矩阵 H 的秩亏情况决定系统是否可观测。
    """
    a = np.array(a, dtype=np.int64)
    m, n = a.shape
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        i = r
        while i < m and a[i, lead] == 0:
            i += 1
        if i == m:
            lead += 1
            continue
        a[[r, i]] = a[[i, r]]
        # 归一化主元行为互质整数
        piv = a[r, lead]
        g = int(np.gcd.reduce(np.abs(a[r])))
        if g > 1:
            a[r] //= g
            piv //= g
        for j in range(m):
            if j != r and a[j, lead] != 0:
                factor = a[j, lead]
                a[j] = piv * a[j] - factor * a[r]
                # 再次约分防止溢出
                g2 = int(np.gcd.reduce(np.abs(a[j])))
                if g2 > 1:
                    a[j] //= g2
        lead += 1
    return a


def rk4_step(f: callable, t: float, y: np.ndarray, h: float,
             args: Tuple = ()) -> np.ndarray:
    """
    经典四阶龙格-库塔单步积分：
        k1 = h·f(t, y)
        k2 = h·f(t+h/2, y+k1/2)
        k3 = h·f(t+h/2, y+k2/2)
        k4 = h·f(t+h, y+k3)
        y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6

    局部截断误差 O(h^5)，全局误差 O(h^4)。
    用于电力系统摇摆方程的暂态稳定数值仿真。
    """
    y = np.asarray(y, dtype=np.float64)
    k1 = h * np.asarray(f(t, y, *args), dtype=np.float64)
    k2 = h * np.asarray(f(t + 0.5 * h, y + 0.5 * k1, *args), dtype=np.float64)
    k3 = h * np.asarray(f(t + 0.5 * h, y + 0.5 * k2, *args), dtype=np.float64)
    k4 = h * np.asarray(f(t + h, y + k3, *args), dtype=np.float64)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def sort_heap_external(n: int, indx: int, i: int, j: int, isgn: int):
    """
    外部堆排序辅助（源自 i4lib 的 sort_heap_external）。
    用于电网节点编号重排与拓扑优化中的排序子程序。
    """
    if indx < 0:
        i = (n - 1) // 2
        j = n - 1
        indx = 1
        return indx, i, j, isgn
    if isgn <= 0:
        j -= 1
        if j == 0:
            indx = 0
        else:
            i = 0
            indx = 2
        return indx, i, j, isgn
    i += 1
    if i == j:
        indx = 3
    else:
        indx = 2
    return indx, i, j, isgn
