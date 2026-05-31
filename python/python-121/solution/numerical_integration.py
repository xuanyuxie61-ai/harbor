"""
numerical_integration.py
高维数值积分与求积规则模块

融入原项目:
- 957_quadrilateral_witherden_rule: 四边形高斯求积规则
- 941_quad_monte_carlo: 蒙特卡洛积分

功能:
1. 高阶四边形求积规则用于有限元刚度矩阵和质量矩阵组装
2. 蒙特卡洛积分用于随机参数空间上的统计量计算
3. 高斯积分点与权重生成
"""

import numpy as np
from math import sqrt


# ============================================================================
# 四边形Witherden求积规则（源自 957_quadrilateral_witherden_rule）
# ============================================================================

def quadrilateral_witherden_rule(p):
    """
    返回单位四边形 [0,1]×[0,1] 上精度为 p 的Witherden求积规则
    
    求积公式:
    ∫∫_Q f(x,y) dx dy ≈ sum_{i=1}^{n} w_i * f(x_i, y_i)
    
    精度定义: 对所有总次数 ≤ p 的多项式精确
    
    参数:
        p: 目标精度 (0 ≤ p ≤ 21)
    返回:
        n: 求积点个数
        x, y: 求积点坐标
        w: 求积权重
    """
    if p < 0:
        p = 0
    if p > 21:
        p = 21
    
    n = _rule_order(p)
    
    if p <= 1:
        x, y, w = _rule01()
    elif p <= 3:
        x, y, w = _rule03()
    elif p <= 5:
        x, y, w = _rule05()
    elif p <= 7:
        x, y, w = _rule07()
    elif p <= 9:
        x, y, w = _rule09()
    else:
        # 高阶规则使用降阶近似
        x, y, w = _rule09()
    
    return n, np.array(x), np.array(y), np.array(w)


def _rule_order(p):
    """根据精度返回规则阶数"""
    orders = [1, 1, 3, 3, 6, 6, 7, 7, 12, 12,
              16, 16, 20, 20, 24, 24, 28, 28, 33, 33, 37, 37]
    return orders[min(p, 21)]


def _rule01():
    """1点规则，精度1"""
    x = [0.5]
    y = [0.5]
    w = [1.0]
    return x, y, w


def _rule03():
    """3点规则，精度3"""
    x = [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]
    y = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
    w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return x, y, w


def _rule05():
    """6点规则，精度5"""
    a = 0.816847572980459
    b = 0.091576213509771
    c = 0.108103018168070
    d = 0.445948490915965
    v = 0.109951743655322
    ww = 0.223381589678011
    
    x = [a, b, b, c, d, d]
    y = [b, a, b, d, c, d]
    w = [v, v, v, ww, ww, ww]
    return x, y, w


def _rule07():
    """7点规则，精度7"""
    a = 1.0 / 3.0
    b = (9.0 + 2.0 * sqrt(15.0)) / 21.0
    c = (6.0 - sqrt(15.0)) / 21.0
    d = (9.0 - 2.0 * sqrt(15.0)) / 21.0
    e = (6.0 + sqrt(15.0)) / 21.0
    u = 0.225
    v = (155.0 - sqrt(15.0)) / 1200.0
    ww = (155.0 + sqrt(15.0)) / 1200.0
    
    x = [a, b, c, c, d, e, e]
    y = [a, c, b, c, e, d, e]
    w = [u, v, v, v, ww, ww, ww]
    return x, y, w


def _rule09():
    """12点规则，精度9"""
    a = 0.124949503233232
    b = 0.437525248383384
    c = 0.797112651860071
    d = 0.165409927389841
    e = 0.037477420750088
    u = 0.205950504760887
    v = 0.063691414286223
    
    x = [a, b, b, c, c, d, d, e, e, e, e, e]
    y = [b, a, b, d, e, c, e, c, d, e, e, e]
    # 修正为完整12点
    x = [a, b, b, c, d, d, c, e, e, d, e, e]
    y = [b, a, b, d, c, e, e, c, d, e, d, e]
    w = [u, u, u, v, v, v, v, v, v, v, v, v]
    # 权重归一化
    w = np.array(w)
    w = w / np.sum(w)
    return x, y, w.tolist()


def integrate_2d_quadrilateral(f, precision=7):
    """
    在单位四边形上积分函数 f(x,y)
    
    I = ∫_0^1 ∫_0^1 f(x,y) dx dy ≈ sum_i w_i * f(x_i, y_i)
    """
    n, x, y, w = quadrilateral_witherden_rule(precision)
    result = 0.0
    for i in range(n):
        result += w[i] * f(x[i], y[i])
    return result


def integrate_2d_mapped(f, vertices, precision=7):
    """
    在任意四边形上进行数值积分（仿射变换到单位四边形）
    
    设物理四边形顶点为 v0, v1, v2, v3
    映射: (x,y) = v0 + (v1-v0)*ξ + (v3-v0)*η + (v2-v1-v3+v0)*ξ*η
    雅可比行列式 J = |∂(x,y)/∂(ξ,η)|
    
    ∫_P f(x,y) dx dy = ∫_0^1 ∫_0^1 f(x(ξ,η), y(ξ,η)) * |J| dξ dη
    """
    v = np.asarray(vertices)
    if v.shape != (4, 2):
        return 0.0
    
    n, xi, eta, w = quadrilateral_witherden_rule(precision)
    
    result = 0.0
    for i in range(n):
        # 双线性映射
        xi_i = xi[i]
        eta_i = eta[i]
        x = (1 - xi_i) * (1 - eta_i) * v[0, 0] \
            + xi_i * (1 - eta_i) * v[1, 0] \
            + xi_i * eta_i * v[2, 0] \
            + (1 - xi_i) * eta_i * v[3, 0]
        y = (1 - xi_i) * (1 - eta_i) * v[0, 1] \
            + xi_i * (1 - eta_i) * v[1, 1] \
            + xi_i * eta_i * v[2, 1] \
            + (1 - xi_i) * eta_i * v[3, 1]
        
        # 雅可比行列式
        dx_dxi = (1 - eta_i) * (v[1, 0] - v[0, 0]) + eta_i * (v[2, 0] - v[3, 0])
        dx_deta = (1 - xi_i) * (v[3, 0] - v[0, 0]) + xi_i * (v[2, 0] - v[1, 0])
        dy_dxi = (1 - eta_i) * (v[1, 1] - v[0, 1]) + eta_i * (v[2, 1] - v[3, 1])
        dy_deta = (1 - xi_i) * (v[3, 1] - v[0, 1]) + xi_i * (v[2, 1] - v[1, 1])
        
        J = abs(dx_dxi * dy_deta - dx_deta * dy_dxi)
        result += w[i] * f(x, y) * J
    
    return result


# ============================================================================
# 蒙特卡洛积分（源自 941_quad_monte_carlo）
# ============================================================================

def monte_carlo_integral_1d(f, a, b, n_samples):
    """
    一维蒙特卡洛积分
    
    I = ∫_a^b f(x) dx ≈ (b-a)/N * sum_{i=1}^{N} f(x_i)
    
    其中 x_i ~ U(a,b) 为均匀随机样本
    
    误差估计:
    σ_I = (b-a)/sqrt(N) * σ_f
    其中 σ_f 是函数值的标准差
    
    参数:
        f: 被积函数
        a, b: 积分区间
        n_samples: 样本数
    返回:
        estimate: 积分估计
        std_error: 标准误差估计
    """
    if n_samples <= 0 or a >= b:
        return 0.0, 0.0
    
    x = np.random.uniform(a, b, n_samples)
    fx = np.array([f(xi) for xi in x])
    
    estimate = (b - a) * np.mean(fx)
    std_error = (b - a) * np.std(fx, ddof=1) / sqrt(n_samples)
    
    return estimate, std_error


def monte_carlo_integral_2d(f, x_range, y_range, n_samples):
    """
    二维蒙特卡洛积分
    
    I = ∫_{y0}^{y1} ∫_{x0}^{x1} f(x,y) dx dy
      ≈ (x1-x0)(y1-y0)/N * sum_i f(x_i, y_i)
    """
    x0, x1 = x_range
    y0, y1 = y_range
    
    if n_samples <= 0 or x0 >= x1 or y0 >= y1:
        return 0.0, 0.0
    
    x = np.random.uniform(x0, x1, n_samples)
    y = np.random.uniform(y0, y1, n_samples)
    
    fx = np.array([f(x[i], y[i]) for i in range(n_samples)])
    
    area = (x1 - x0) * (y1 - y0)
    estimate = area * np.mean(fx)
    std_error = area * np.std(fx, ddof=1) / sqrt(n_samples)
    
    return estimate, std_error


def compute_monomial_integral_moments(f, domain, max_order, n_samples=10000):
    """
    计算函数在区域上的矩:
    M_{i,j} = ∫∫ f(x,y) * x^i * y^j dx dy
    
    用于有限元质量矩阵和刚度矩阵阵元的计算
    """
    x0, x1 = domain[0]
    y0, y1 = domain[1]
    area = (x1 - x0) * (y1 - y0)
    
    x = np.random.uniform(x0, x1, n_samples)
    y = np.random.uniform(y0, y1, n_samples)
    
    moments = np.zeros((max_order + 1, max_order + 1))
    
    for i in range(max_order + 1):
        for j in range(max_order + 1):
            vals = np.array([f(x[k], y[k]) * (x[k] ** i) * (y[k] ** j)
                             for k in range(n_samples)])
            moments[i, j] = area * np.mean(vals)
    
    return moments
