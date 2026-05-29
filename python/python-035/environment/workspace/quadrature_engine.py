"""
quadrature_engine.py
多维数值积分引擎：用于计算希格斯衰变微分截面的相空间积分

基于三个种子项目重构:
  - 943_quad_rule: 高斯求积规则生成 (IQPACK / Golub-Welsch)
  - 945_quad_trapezoid: 复合梯形公式
  - 1144_square_felippa_rule: 2D 张量积 Gauss-Legendre 求积

物理应用:
  相空间积分 I = int dPhi_4 |M|^2
  其中 |M|^2 为螺旋度振幅的模方
  数值上通常降维到 1D/2D 积分:
    - m_Z1 积分
    - m_Z2 积分  
    - 角分布积分
"""
import numpy as np
import math
from constants import TINY

# ============================================================
# 1. Golub-Welsch 算法: Jacobi 矩阵特征值求高斯节点与权重
# ============================================================
def imtqlx(n, d, e, z):
    """
    Golub-Welsch 技术: 对称三对角 Jacobi 矩阵对角化
    
    使用 numpy.linalg.eigh 稳定求解特征值与特征向量，
    替代传统隐式 QL 迭代。
    
    T 矩阵:
      T_{i,i} = d_i
      T_{i,i+1} = T_{i+1,i} = e_i
    
    返回特征值 (节点) 和第一分量平方根权重因子
    """
    d = np.asarray(d, dtype=float).copy()
    e = np.asarray(e, dtype=float)
    z = np.asarray(z, dtype=float)
    
    if n == 1:
        return d, z
    
    # 构造对称三对角矩阵
    T = np.diag(d) + np.diag(e[:n-1], 1) + np.diag(e[:n-1], -1)
    eigvals, eigvecs = np.linalg.eigh(T)
    
    # 第一分量
    z_out = eigvecs[0, :]
    return eigvals, z_out


def legendre_gauss_rule(n):
    """
    生成 n 点 Gauss-Legendre 求积规则，区间 [-1, 1]
    
    正交多项式: P_n(x), 权函数 w(x) = 1
    求积公式: int_{-1}^{1} f(x) dx ~ sum_{i=1}^n w_i * f(x_i)
    
    精确度: 2n-1 次多项式
    
    使用 numpy 稳定实现。
    """
    if n < 1:
        return np.array([]), np.array([])
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def jacobi_gauss_rule(n, alpha, beta):
    """
    生成 n 点 Gauss-Jacobi 求积规则，区间 [-1, 1]
    
    权函数: w(x) = (1-x)^alpha * (1+x)^beta
    
    Jacobi 矩阵递推系数:
      a_i = (beta^2 - alpha^2) / ((2i+alpha+beta)*(2i+alpha+beta+2))  (对角)
      b_i = 2/(2i+alpha+beta) * sqrt(i*(i+alpha)*(i+beta)*(i+alpha+beta) /
                     ((2i+alpha+beta-1)*(2i+alpha+beta+1)))  (次对角)
    """
    if n < 1:
        return np.array([]), np.array([])
    ab = alpha + beta
    d = np.zeros(n)
    e = np.zeros(n)
    
    d[0] = (beta - alpha) / (ab + 2.0) if ab + 2.0 != 0.0 else 0.0
    if n > 1:
        e[0] = np.sqrt(4.0 * (1.0 + alpha) * (1.0 + beta) / ((ab + 3.0) * (ab + 2.0) ** 2))
    
    for i in range(2, n + 1):
        denom1 = 2.0 * i + ab
        d[i - 1] = (beta ** 2 - alpha ** 2) / (denom1 * (denom1 + 2.0)) if denom1 != 0.0 else 0.0
        if i < n:
            num = i * (i + alpha) * (i + beta) * (i + ab)
            den = (denom1 - 1.0) * (denom1 + 1.0) * denom1 ** 2
            e[i - 1] = np.sqrt(max(num / den, 0.0)) if den > 0.0 else 0.0
    
    # 归一化权重因子
    norm = 2.0 ** (ab + 1.0) * math.gamma(alpha + 1.0) * math.gamma(beta + 1.0) / math.gamma(ab + 2.0)
    z = np.zeros(n)
    z[0] = 1.0
    nodes, weights_vec = imtqlx(n, d, e, z)
    weights = norm * weights_vec ** 2
    return nodes, weights


# ============================================================
# 2. 复合梯形公式 (映射 945_quad_trapezoid)
# ============================================================
def composite_trapezoidal(f, a, b, n):
    """
    复合梯形公式:
      Q_n = h/2 * [f(a) + 2*sum_{i=1}^{n-1} f(x_i) + f(b)]
      h = (b - a) / n
      x_i = a + i*h
    
    误差: E = -(b-a)^3 / (12*n^2) * f''(xi)
    
    参数:
        f: 被积函数
        a, b: 区间
        n: 子区间数
    返回:
        积分近似值
    """
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x = a + i * h
        total += f(x)
    return h * total


def adaptive_trapezoidal(f, a, b, tol=1.0e-8, max_level=20):
    """
    自适应梯形积分 (Richardson 外推)
    
    步骤:
      T_1 = (b-a)/2 * (f(a)+f(b))
      T_2 = T_1/2 + (b-a)/2 * f((a+b)/2)
      误差估计 |T_2 - T_1| / 3
      若误差 > tol，递归细分
    """
    def recurse(lo, hi, fl, fh, whole, level):
        if level >= max_level:
            return whole
        m = (lo + hi) / 2.0
        fm = f(m)
        left = (m - lo) / 2.0 * (fl + fm)
        right = (hi - m) / 2.0 * (fm + fh)
        delta = left + right - whole
        if abs(delta) < 3.0 * tol * (hi - lo) / (b - a):
            return left + right + delta / 3.0
        return recurse(lo, m, fl, fm, left, level + 1) + recurse(m, hi, fm, fh, right, level + 1)
    
    fa = f(a)
    fb = f(b)
    whole = (b - a) / 2.0 * (fa + fb)
    return recurse(a, b, fa, fb, whole, 0)


# ============================================================
# 3. 2D 张量积求积 (映射 1144_square_felippa_rule)
# ============================================================
def tensor_product_2d(rule_1d_a, rule_1d_b, func, rect_a, rect_b):
    """
    在矩形区域 [a1,b1] x [a2,b2] 上使用两个 1D 求积规则的张量积
    
    公式:
      I = sum_{i=1}^{n1} sum_{j=1}^{n2} w_i * w_j * f(x_i, y_j)
    
    参数:
        rule_1d_a: (nodes_a, weights_a) for x 方向
        rule_1d_b: (nodes_b, weights_b) for y 方向
        func: f(x, y) 被积函数
        rect_a: (a1, b1) x 方向区间
        rect_b: (a2, b2) y 方向区间
    返回:
        积分近似值
    """
    nodes_a, weights_a = rule_1d_a
    nodes_b, weights_b = rule_1d_b
    a1, b1 = rect_a
    a2, b2 = rect_b
    
    # 将 [-1,1] 映射到实际区间
    scale_a = (b1 - a1) / 2.0
    shift_a = (a1 + b1) / 2.0
    scale_b = (b2 - a2) / 2.0
    shift_b = (a2 + b2) / 2.0
    
    total = 0.0
    for i in range(len(nodes_a)):
        x = scale_a * nodes_a[i] + shift_a
        for j in range(len(nodes_b)):
            y = scale_b * nodes_b[j] + shift_b
            total += weights_a[i] * weights_b[j] * func(x, y)
    
    return total * scale_a * scale_b


def gauss_legendre_2d(n1, n2, func, rect_a, rect_b):
    """
    2D Gauss-Legendre 张量积求积的便捷接口
    """
    nodes1, weights1 = legendre_gauss_rule(n1)
    nodes2, weights2 = legendre_gauss_rule(n2)
    return tensor_product_2d((nodes1, weights1), (nodes2, weights2), func, rect_a, rect_b)


# ============================================================
# 4. 高维求积: 递归降维积分
# ============================================================
def integrate_nested(rules_1d, func, ranges):
    """
    多维张量积求积的递归实现
    
    参数:
        rules_1d: [(nodes_i, weights_i), ...] 每个维度的 1D 规则
        func: f(x1, x2, ..., xd)
        ranges: [(a1,b1), (a2,b2), ...]
    返回:
        积分值
    """
    dim = len(rules_1d)
    scales = [(ranges[i][1] - ranges[i][0]) / 2.0 for i in range(dim)]
    shifts = [(ranges[i][0] + ranges[i][1]) / 2.0 for i in range(dim)]
    
    nodes_list = [rules_1d[i][0] for i in range(dim)]
    weights_list = [rules_1d[i][1] for i in range(dim)]
    n_list = [len(nodes_list[i]) for i in range(dim)]
    
    total = 0.0
    
    def recurse(d, point, w_prod):
        nonlocal total
        if d == dim:
            total += w_prod * func(*point)
            return
        for i in range(n_list[d]):
            x = scales[d] * nodes_list[d][i] + shifts[d]
            point[d] = x
            recurse(d + 1, point, w_prod * weights_list[d][i])
    
    point = [0.0] * dim
    recurse(0, point, 1.0)
    
    jacobian = 1.0
    for s in scales:
        jacobian *= s
    return total * jacobian


# ============================================================
# 5. 物理积分: Breit-Wigner 加权积分
# ============================================================
def breit_wigner(m, m0, gamma):
    """
    相对论 Breit-Wigner 分布:
      BW(m) = (1/pi) * (m0*gamma) / ((m^2 - m0^2)^2 + m0^2 * gamma^2)
    
    归一化: int_0^inf BW(m) dm = 1 (在窄宽度近似下)
    """
    denom = (m ** 2 - m0 ** 2) ** 2 + (m0 * gamma) ** 2
    if denom < TINY:
        return 0.0
    return (1.0 / np.pi) * (m0 * gamma) / denom


def integrate_dsigma_dm1dm2(matrix_element_sq, m_higgs, m_z, gamma_z, n_points=16):
    """
    计算双微分截面对 m_z1, m_z2 的积分
    
    I = int_{m_ll_min}^{m_h-m_ll_min} dm1 int_{m_ll_min}^{m_h-m1} dm2
        |M(m1,m2)|^2 * BW(m1) * BW(m2)
    
    使用 2D Gauss-Legendre 求积 (映射 1144_square_felippa_rule)
    """
    m_ll_min = 0.001
    
    # === HOLE 2 BEGIN ===
    # TODO: 实现双微分截面对 m_z1, m_z2 的 2D 数值积分
    # 物理要求:
    #   1. 定义被积函数: integrand(m1, m2) = |M(m1,m2)|^2 * BW(m1,m_z,gamma_z) * BW(m2,m_z,gamma_z)
    #   2. 运动学约束检查: m1 + m2 <= m_higgs, m1 >= m_ll_min, m2 >= m_ll_min
    #   3. 积分区域: [m_ll_min, m_higgs-m_ll_min] x [m_ll_min, m_higgs-m_ll_min]
    #   4. 使用 gauss_legendre_2d(n_points, n_points, integrand, (a1,b1), (a2,b2)) 进行张量积求积
    #   5. 返回 max(result, 0.0)
    # 注意: matrix_element_sq 参数为 Hole 1 所在文件的函数，需协同修复
    raise NotImplementedError("HOLE 2: 请实现 integrate_dsigma_dm1dm2")
    # === HOLE 2 END ===


# ============================================================
# 6. 复合 Simpson 规则 (作为梯形公式的补充)
# ============================================================
def composite_simpson(f, a, b, n):
    """
    复合 Simpson 公式 (n 必须为偶数)
      Q = h/3 * [f0 + f_n + 4*sum_{奇数} f_i + 2*sum_{偶数} f_i]
      误差 O(h^4)
    """
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 1:
            total += 4.0 * f(x)
        else:
            total += 2.0 * f(x)
    return h * total / 3.0
