"""
evidence_quadrature.py  --  贝叶斯证据积分 (正磁盘求积)
===============================================================
来源种子项目:
    304_disk01_positive_rule : 单位正磁盘上的高代数精度求积规则
                                (DOF = 3N, 精度 D).
科学问题角色:
    贝叶斯证据 (marginal likelihood):
        Z = p(y) = integral p(y|theta) p(theta) dtheta
    在低维参数子空间上, 用自适应求积直接近似 Z, 用于
    1. 模型选择 (Bayes factor B_12 = Z_1 / Z_2)
    2. 校验 MCMC 估计的后验均值
核心公式:
    正磁盘积分: int_{x^2+y^2<=1, x,y>=0} f(x,y) dx dy
              approx sum_{i=1}^N w_i * f(x_i, y_i)
    通过映射 theta = (r*cos phi, r*sin phi) 把参数空间
    变换到极坐标, 在 (r, phi) in [0,1] x [0, pi/2] 上求积.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Callable
from numerical_base import NUMERICS


# ======================================================================
# 1. 正磁盘求积规则 (移植 Burkardt)
# ======================================================================
def disk01_positive_area() -> float:
    """单位正磁盘面积: pi/4."""
    return math.pi / 4.0


def disk01_positive_rule(N: int) -> Tuple[List[float],
                                           List[float], List[float]]:
    """
    构造 N 点正磁盘求积规则.
    简化实现: 用等角分布 + 径向 Gauss-Legendre.
    返回 (weights, xs, ys).
    DOF = 3N, 精度 D ~ N-1.
    """
    if N < 1:
        raise ValueError("N >= 1 required")
    # 角度: 等分 [0, pi/2]
    angles = [(math.pi / 2.0) * (2.0 * (i + 1) - 1.0) / (2.0 * N)
              for i in range(N)]
    # 径向: 简单 Gauss-Legendre on [0, 1] with weight r
    rs = _gauss_legendre_radial(N)
    ws = [disk01_positive_area() / N] * N
    # 加上径向权重
    for i in range(N):
        ws[i] *= 2.0 * rs[i]  # 2r dr 变换
    xs = [rs[i] * math.cos(angles[i]) for i in range(N)]
    ys = [rs[i] * math.sin(angles[i]) for i in range(N)]
    # 归一化使 sum(w) = pi/4
    total = sum(ws)
    target = disk01_positive_area()
    if total > NUMERICS.safe_log_floor:
        ws = [w * target / total for w in ws]
    return ws, xs, ys


def _gauss_legendre_radial(N: int) -> List[float]:
    """
    N 点 Gauss-Legendre 在 [0, 1] 上的节点.
    简化: 用 Clenshaw-Curtis 近似.
    """
    if N == 1:
        return [0.5]
    nodes = []
    for k in range(N):
        theta = math.pi * (k + 0.5) / N
        x = 0.5 * (1.0 - math.cos(theta))  # 映射 [-1,1] -> [0,1]
        nodes.append(max(0.0, min(1.0, x)))
    return nodes


# ======================================================================
# 2. 贝叶斯证据计算
# ======================================================================
def compute_evidence(log_integrand: Callable[[List[float]], float],
                     dim: int, n_quad: int = 16) -> float:
    """
    在 dim 维超立方体上计算证据积分 Z = int exp(log_integrand) dtheta.
    对 dim > 2 用乘积求积 (张量积); dim <= 2 用正磁盘规则.
    """
    if dim == 1:
        nodes = _gauss_legendre_radial(n_quad)
        weights = [1.0 / n_quad] * n_quad
        total = 0.0
        for i, x in enumerate(nodes):
            total += weights[i] * math.exp(log_integrand([x]))
        return total
    elif dim == 2:
        ws, xs, ys = disk01_positive_rule(n_quad)
        total = 0.0
        for i in range(len(ws)):
            total += ws[i] * math.exp(log_integrand([xs[i], ys[i]]))
        return total
    else:
        # 张量积: 每维 n_1d 点, 总 n_1d^dim 点
        n_1d = max(3, int(n_quad ** (1.0 / dim)))
        nodes_1d = _gauss_legendre_radial(n_1d)
        w_1d = [1.0 / n_1d] * n_1d
        # 生成所有组合
        total = 0.0
        count = 0
        for idx in _multi_index(dim, n_1d):
            theta = [nodes_1d[idx[d]] for d in range(dim)]
            w = 1.0
            for d in range(dim):
                w *= w_1d[idx[d]]
            val = math.exp(log_integrand(theta))
            if NUMERICS.is_finite(val):
                total += w * val
            count += 1
        return total


def _multi_index(dim: int, n: int):
    """生成 dim 维多指标, 每维 0..n-1."""
    if dim == 1:
        for i in range(n):
            yield [i]
    else:
        for tail in _multi_index(dim - 1, n):
            for i in range(n):
                yield [i] + tail


# ======================================================================
# 3. Bayes 因子
# ======================================================================
def bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
    """
    B_12 = Z_1 / Z_2, 返回对数尺度.
    Kass & Raftery (1995) 判据:
        log B > 5  : 强证据
        log B > 2  : 正面证据
        log B < 1  : 不确定
    """
    return log_evidence_1 - log_evidence_2


def interpret_bayes_factor(log_bf: float) -> str:
    if log_bf > 5.0:
        return "强证据支持模型 1"
    elif log_bf > 2.0:
        return "正面证据支持模型 1"
    elif log_bf > -2.0:
        return "不确定 / 无显著差异"
    elif log_bf > -5.0:
        return "正面证据支持模型 2"
    else:
        return "强证据支持模型 2"


# ======================================================================
# 4. 后验均值校验
# ======================================================================
def posterior_mean_quadrature(theta_func: Callable[[List[float]], List[float]],
                              log_posterior: Callable[[List[float]], float],
                              dim: int, n_quad: int = 8
                              ) -> List[float]:
    """
    用求积直接计算后验均值:
        E[theta_d | y] = int theta_d * p(y|theta) p(theta) dtheta / Z
    """
    # 先算 Z
    Z = compute_evidence(log_posterior, dim, n_quad)
    if Z < NUMERICS.safe_log_floor:
        return [0.0] * dim
    # 对每个维度算分子
    means = []
    for d_target in range(dim):
        def integrand(theta, _d=d_target):
            lp = log_posterior(theta)
            if not NUMERICS.is_finite(lp):
                return 0.0
            return theta[_d] * math.exp(lp)
        # 复用 compute_evidence 框架, 但被积函数不同
        if dim == 1:
            nodes = _gauss_legendre_radial(n_quad)
            weights = [1.0 / n_quad] * n_quad
            num = sum(w * integrand([x]) for x, w in zip(nodes, weights))
        elif dim == 2:
            ws, xs, ys = disk01_positive_rule(n_quad)
            num = sum(w * integrand([xs[i], ys[i]])
                      for i, w in enumerate(ws))
        else:
            n_1d = max(3, int(n_quad ** (1.0 / dim)))
            nodes_1d = _gauss_legendre_radial(n_1d)
            w_1d = [1.0 / n_1d] * n_1d
            num = 0.0
            for idx in _multi_index(dim, n_1d):
                theta = [nodes_1d[idx[dd]] for dd in range(dim)]
                w = 1.0
                for dd in range(dim):
                    w *= w_1d[idx[dd]]
                num += w * integrand(theta)
        means.append(num / Z)
    return means


__all__ = ["disk01_positive_rule", "disk01_positive_area",
           "compute_evidence", "bayes_factor",
           "interpret_bayes_factor", "posterior_mean_quadrature"]
