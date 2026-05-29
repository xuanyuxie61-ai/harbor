"""
quadrature_rules.py
====================
博士级数值积分规则库，为膜蛋白静电势、结合自由能计算提供
高阶、高精度的求积节点与权重。

核心数学内容：
  - Clenshaw-Curtis 求积：基于 Chebyshev 节点
    $x_i = \cos\left(\frac{i\pi}{n}\right)$, $w_i$ 由离散余弦变换得到
  - Jacobi-Gauss 求积：用于 $(1-x)^\alpha(1+x)^\beta$ 权积分
  - 广义 Hermite-Gauss 求积：用于 $|x|^\alpha e^{-x^2}$ 权积分
  - Laguerre-Gauss 求积：用于 $e^{-x}$ 权积分（径向 Coulomb 势）
  - Wandzura 三角形求积：用于膜片单元的面积分

种子项目映射：
  - 1054_sandia_rules        →  Clenshaw-Curtis, Jacobi, Hermite, Laguerre
  - 641_laguerre_polynomial  →  Laguerre 求积与 Jacobi 矩阵对角化
  - 1324_triangle_wandzura_rule →  三角形高阶对称求积
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Clenshaw-Curtis 求积（种子项目 1054_sandia_rules）
# ---------------------------------------------------------------------------
def clenshaw_curtis_compute(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Clenshaw-Curtis 求积规则的节点与权重。

    节点：
        $x_i = \cos\left(\frac{(i-1)\pi}{n-1}\right)$, $i=1,\dots,n$
    权重由离散余弦级数系数给出：
        $w_i = \frac{c_i}{n-1}\left(1 - \sum_{j=1}^{\lfloor(n-1)/2\rfloor}
            \frac{b_j}{4j^2-1} \cos(2j\theta_i) \right)$
    其中 $c_1 = c_n = 1/2$, $c_i = 1$ (其他)，$b_j = 2$ (一般) 或 $1$ ($2j=n-1$)。

    参数边界：
        order >= 1
    """
    if order < 1:
        raise ValueError("clenshaw_curtis_compute: order must be >= 1.")

    x = np.zeros(order, dtype=float)
    w = np.zeros(order, dtype=float)

    if order == 1:
        x[0] = 0.0
        w[0] = 2.0
        return x, w

    for i in range(order):
        x[i] = np.cos((order - 1 - i) * np.pi / (order - 1))

    x[0] = -1.0
    if order % 2 == 1:
        x[(order - 1) // 2] = 0.0
    x[-1] = 1.0

    # TODO: Hole 1 — 实现 Clenshaw-Curtis 权重计算
    # 根据离散余弦级数计算权重 w[i]，并进行归一化
    # 提示：权重之和应等于 2.0（对应区间 [-1,1]）
    raise NotImplementedError("Hole 1: Clenshaw-Curtis weight calculation not implemented.")


# ---------------------------------------------------------------------------
# Jacobi-Gauss 求积（种子项目 1054_sandia_rules）
# ---------------------------------------------------------------------------
def jacobi_compute(n: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Elhay-Kautsky 算法计算 Jacobi-Gauss 求积。

    积分：
        $\int_{-1}^{1} (1-x)^\alpha (1+x)^\beta f(x) \, dx$
        $\approx \sum_{i=1}^{n} w_i f(x_i)$

    零阶矩：
        $\mu_0 = 2^{\alpha+\beta+1} \frac{\Gamma(\alpha+1)\Gamma(\beta+1)}{\Gamma(2+\alpha+\beta)}$

    Jacobi 矩阵递推：
        $a_1 = \frac{\beta-\alpha}{2+\alpha+\beta}$
        $b_1^2 = \frac{4(1+\alpha)(1+\beta)}{(3+\alpha+\beta)(2+\alpha+\beta)^2}$
        $a_i = \frac{(\beta+\alpha)(\beta-\alpha)}{(abi-2)abi}$
        $b_i^2 = \frac{4i(i+\alpha)(i+\beta)(i+\alpha+\beta)}{(abi-1)(abi+1)abi^2}$
    其中 $abi = 2i + \alpha + \beta$。

    参数边界：
        n >= 1, alpha > -1, beta > -1
    """
    if n < 1:
        raise ValueError("jacobi_compute: n must be >= 1.")
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_compute: alpha and beta must be > -1.")

    from scipy.special import gamma as gamma_func

    zemu = (2.0 ** (alpha + beta + 1.0)) * gamma_func(alpha + 1.0) * gamma_func(beta + 1.0) / gamma_func(2.0 + alpha + beta)

    if n == 1:
        x = np.array([(beta - alpha) / (2.0 + alpha + beta)], dtype=float)
        w = np.array([zemu], dtype=float)
        return x, w

    diag = np.zeros(n, dtype=float)
    offdiag = np.zeros(n, dtype=float)

    diag[0] = (beta - alpha) / (2.0 + alpha + beta)
    offdiag[0] = 4.0 * (1.0 + alpha) * (1.0 + beta) / ((3.0 + alpha + beta) * (2.0 + alpha + beta) ** 2)

    for i in range(1, n):
        abi = 2.0 * (i + 1) + alpha + beta
        diag[i] = (beta + alpha) * (beta - alpha) / ((abi - 2.0) * abi)
        offdiag[i] = (4.0 * (i + 1) * (i + 1 + alpha) * (i + 1 + beta) * (i + 1 + alpha + beta)
                      / ((abi - 1.0) * (abi + 1.0) * abi * abi))

    offdiag = np.sqrt(offdiag)
    offdiag[-1] = 0.0

    # 对称三对角矩阵特征值分解
    x, eigvecs = _sym_tridiag_eig(diag, offdiag[:-1])
    w = eigvecs[0, :] ** 2 * zemu

    return x, w


# ---------------------------------------------------------------------------
# 广义 Hermite-Gauss 求积（种子项目 1054_sandia_rules）
# ---------------------------------------------------------------------------
def gen_hermite_compute(n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算广义 Hermite-Gauss 求积规则。

    积分：
        $\int_{-\infty}^{+\infty} |x|^\alpha e^{-x^2} f(x) \, dx$
        $\approx \sum_{i=1}^{n} w_i f(x_i)$

    零阶矩：
        $\mu_0 = \Gamma\left(\frac{\alpha+1}{2}\right)$

    Jacobi 矩阵非对角元：
        $b_i^2 = \frac{i+\alpha}{2}$  (i 奇)
        $b_i^2 = \frac{i}{2}$       (i 偶)

    参数边界：
        n >= 1, alpha > -1
    """
    if n < 1:
        raise ValueError("gen_hermite_compute: n must be >= 1.")
    if alpha <= -1.0:
        raise ValueError("gen_hermite_compute: alpha must be > -1.")

    from scipy.special import gamma as gamma_func

    zemu = gamma_func((alpha + 1.0) / 2.0)

    offdiag = np.zeros(n, dtype=float)
    for i in range(n):
        idx = i + 1
        if idx % 2 == 1:
            offdiag[i] = (idx + alpha) / 2.0
        else:
            offdiag[i] = idx / 2.0

    offdiag = np.sqrt(offdiag)
    diag = np.zeros(n, dtype=float)

    x, eigvecs = _sym_tridiag_eig(diag, offdiag[:-1])
    w = eigvecs[0, :] ** 2 * zemu
    return x, w


# ---------------------------------------------------------------------------
# Laguerre-Gauss 求积（种子项目 641_laguerre_polynomial）
# ---------------------------------------------------------------------------
def laguerre_quadrature_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 Laguerre 多项式 $L_n(x)$ 的 Gauss-Laguerre 求积。

    积分：
        $\int_{0}^{+\infty} e^{-x} f(x) \, dx \approx \sum_{i=1}^{n} w_i f(x_i)$

    Jacobi 矩阵：
        对角元 $a_i = 2i - 1$
        次对角元 $b_i = \sqrt{i}$

    参数边界：
        n >= 1
    """
    if n < 1:
        raise ValueError("laguerre_quadrature_rule: n must be >= 1.")

    zemu = 1.0
    offdiag = np.sqrt(np.arange(1, n + 1, dtype=float))
    diag = np.array([2.0 * i - 1.0 for i in range(1, n + 1)], dtype=float)

    x, eigvecs = _sym_tridiag_eig(diag, offdiag[:-1])
    w = eigvecs[0, :] ** 2 * zemu
    return x, w


# ---------------------------------------------------------------------------
# Wandzura 三角形对称求积（种子项目 1324_triangle_wandzura_rule）
# ---------------------------------------------------------------------------
def wandzura_rule(rule_index: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 Wandzura 三角形对称求积规则的节点（重心坐标）与权重。

    参考：
        Stephen Wandzura, Hong Xiao,
        "Symmetric Quadrature Rules on a Triangle",
        Computers and Mathematics with Applications, 45(12), 2003, 1829-1840.

    此处实现 Wandzura 6 阶规则（6 points, 精确到 5 次多项式）
    和 12 阶规则（12 points）。

    参数边界：
        rule_index 取 1 (6阶) 或 2 (12阶)

    返回：
        xy : shape (n, 3) 的重心坐标 (L1, L2, L3)
        w  : shape (n,) 的权重（面积归一化权重之和 = 1）
    """
    if rule_index == 1:
        # 6-point 5th-degree rule (Stroud 1971 / Wandzura 2003 表1)
        # 重心坐标与权重
        a1 = 0.816847572980458
        b1 = 0.091576213509771
        w1 = 0.109951743655322

        a2 = 0.108103018168070
        b2 = 0.445948490915965
        w2 = 0.223381589678011

        xy = np.array([
            [a1, b1, b1],
            [b1, a1, b1],
            [b1, b1, a1],
            [a2, b2, b2],
            [b2, a2, b2],
            [b2, b2, a2],
        ], dtype=float)
        w = np.array([w1, w1, w1, w2, w2, w2], dtype=float)
        # 归一化使权重和为 1
        w = w / np.sum(w)
        return xy, w

    elif rule_index == 2:
        # 12-point 7th-degree rule (Wandzura 2003)
        a1 = 0.873821971016996
        b1 = 0.063089014491502
        w1 = 0.050844906370207

        a2 = 0.501426509658179
        b2 = 0.249286745170910
        w2 = 0.116786275726379

        a3 = 0.636502499121399
        b3 = 0.310352451033785
        c3 = 0.053145049844816
        w3 = 0.082851075618374

        xy = np.array([
            [a1, b1, b1],
            [b1, a1, b1],
            [b1, b1, a1],
            [a2, b2, b2],
            [b2, a2, b2],
            [b2, b2, a2],
            [a3, b3, c3],
            [a3, c3, b3],
            [b3, a3, c3],
            [b3, c3, a3],
            [c3, a3, b3],
            [c3, b3, a3],
        ], dtype=float)
        w = np.array([w1]*3 + [w2]*3 + [w3]*6, dtype=float)
        w = w / np.sum(w)
        return xy, w

    else:
        raise ValueError("wandzura_rule: rule_index must be 1 or 2.")


def integrate_triangle(f, vertices: np.ndarray, rule_index: int = 1) -> float:
    """
    在三角形上对函数 f 进行数值积分。

    参数：
        f        : 函数句柄，接受形状 (n, 2) 的物理坐标数组，返回 (n,) 的值
        vertices : shape (3, 2) 的三角形顶点
        rule_index: Wandzura 规则编号

    数学关系：
        $\int_{T} f(x,y) \, dA = |J| \sum_{k} w_k f(x_k, y_k)$
    其中 $|J| = 2 \times \text{Area}(T)$ 为 Jacobian 行列式绝对值。
    """
    if vertices.shape != (3, 2):
        raise ValueError("integrate_triangle: vertices must have shape (3, 2).")

    xy_bary, w = wandzura_rule(rule_index)

    # 重心坐标 -> 物理坐标
    # [x, y] = L1*v1 + L2*v2 + L3*v3
    phys = xy_bary @ vertices  # shape (n, 2)

    # Jacobian = |det([v2-v1, v3-v1])|
    J = np.linalg.det(np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0])))
    area = 0.5 * abs(J)

    vals = f(phys)
    return float(area * np.dot(w, vals))


# ---------------------------------------------------------------------------
# 对称三对角矩阵特征值求解（Golub-Welsch 算法核心）
# ---------------------------------------------------------------------------
def _sym_tridiag_eig(diag: np.ndarray, offdiag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算实对称三对角矩阵的特征值与特征向量。

    使用 numpy.linalg.eigh 的包装，确保数值稳定性。
    对应种子项目 641_laguerre_polynomial 中的 imtqlx 算法思想。
    """
    n = diag.shape[0]
    T = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
    w, v = np.linalg.eigh(T)
    # 按特征值升序排列（与标准求积节点一致）
    idx = np.argsort(w)
    return w[idx], v[:, idx]
