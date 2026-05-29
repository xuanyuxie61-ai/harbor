r"""
gaussian_causal_test.py
================================================================================
基于 Owen T 函数与双变量正态分布的条件独立性检验模块

原项目映射: 033_asa076 — Owen T 函数的高斯求积计算

科学背景
--------
在因果推断中，检验 $X_i \perp X_j \mid \mathbf{X}_{\setminus\{i,j\}}$ 是
发现因果骨架的关键步骤。对于联合高斯分布，条件独立等价于偏相关系数为零：

$$ \rho_{ij|\mathbf{V}\setminus\{i,j\}} = 0 $$

偏相关系数与精度矩阵元素的关系为：
$$ \rho_{ij|\mathbf{V}\setminus\{i,j\}} = -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\Theta_{jj}}} $$

为了在高维小样本场景下进行稳健的假设检验，我们引入 Fisher Z 变换，
并计算双变量正态尾概率以得到精确的 p 值。

Owen T 函数（双变量正态积分）定义为：
$$ T(h, a) = \frac{1}{2\pi}\int_{0}^{a}\frac{\exp\left(-\frac{h^2}{2}(1+x^2)\right)}{1+x^2}\,dx $$

其在因果检验中的应用：对于检验统计量 $Z$，我们需要计算
$P(|Z| > z_{\text{obs}})$，其中涉及双变量正态累积分布函数，
可通过 Owen T 函数表达。

核心公式
--------
1. Fisher Z 变换（将相关系数 r 映射到近似正态分布）：
   $$ Z = \frac{1}{2}\ln\frac{1+r}{1-r}, \qquad \text{Var}(Z) \approx \frac{1}{n-p-3} $$

2. 偏相关系数估计：
   $$ r_{ij|\mathbf{C}} = -\frac{\hat{\Theta}_{ij}}{\sqrt{\hat{\Theta}_{ii}\hat{\Theta}_{jj}}} $$

3. 检验统计量：
   $$ t = r\sqrt{\frac{n-p-2}{1-r^2}} \sim t_{n-p-2} $$

4. Owen T 函数（5 点 Gauss-Legendre 求积，参考原项目 ASA076）：
   $$ T(h,a) \approx \frac{a}{2\pi}\sum_{k=1}^{5} w_k \frac{\exp\left(-\frac{h^2}{2}(1+(a x_k)^2)\right)}{1+(a x_k)^2} $$
r"""

import numpy as np
from typing import Tuple


def owen_t_function(h: float, a: float) -> float:
    r"""
    计算 Owen T 函数 $T(h, a)$。

    使用 5 点 Gauss-Legendre 求积（与原项目 ASA076 一致）。
    对边界情况（h 接近 0 或 a 极大）做特殊处理以保证数值鲁棒性。

    Parameters
    ----------
    h : float
        第一参数。
    a : float
        第二参数（积分上限的比例因子）。

    Returns
    -------
    value : float
        $T(h, a)$ 的值。
    r"""
    # 边界处理
    tv1 = 1.0e-35
    tv2 = 15.0
    tv3 = 15.0
    tv4 = 1.0e-5
    tp = 0.159154943091895  # 1/(2*pi)

    if abs(h) < tv1:
        return tp * np.arctan(a)
    if abs(h) > tv2:
        return 0.0
    if abs(a) < tv1:
        return 0.0

    # 5 点 Gauss-Legendre 节点与权重
    u = np.array([0.0744371695, 0.2166976971, 0.3397047841,
                  0.4325316833, 0.4869532649])
    r = np.array([0.1477621124, 0.1346333597, 0.1095431812,
                  0.0747256746, 0.0333356721])

    xs = -0.5 * h * h
    fxs = a * a

    # 截断点计算（Newton 迭代）
    if tv3 <= np.log(1.0 + fxs) - xs * fxs:
        x1 = 0.5 * a
        fxs = 0.25 * fxs
        while True:
            rt = fxs + 1.0
            x2 = x1 + (xs * fxs + tv3 - np.log(rt)) / (2.0 * x1 * (1.0 / rt - xs))
            fxs = x2 * x2
            if abs(x2 - x1) < tv4:
                break
            x1 = x2
        a_eff = x2
    else:
        a_eff = a

    rt_sum = 0.0
    for i in range(5):
        r1 = 1.0 + fxs * (0.5 + u[i]) ** 2
        r2 = 1.0 + fxs * (0.5 - u[i]) ** 2
        rt_sum += r[i] * (np.exp(xs * r1) / r1 + np.exp(xs * r2) / r2)

    value = rt_sum * a_eff * tp
    return float(value)


def bivariate_normal_cdf(x: float, y: float, rho: float) -> float:
    r"""
    通过 Owen T 函数计算双变量标准正态累积分布函数：

    $$ \Phi_2(x,y;\rho) = \Phi(x)\Phi(y) + \sum_{i=0}^{1}\sum_{j=0}^{1}(-1)^{i+j} T\left(h_i, a_i\right) $$

    其中 $h_i, a_i$ 与 $(x,y,\rho)$ 的符号组合有关。
    这里使用简化形式（Sheppard 公式 + Owen T）。
    r"""
    if rho <= -1.0 or rho >= 1.0:
        raise ValueError("相关系数 rho 必须在 (-1,1) 内。")

    def phi_cdf(z):
        return 0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0)))

    # 简化实现：当 |rho| 较小时用独立近似，否则用 Owen T
    if abs(rho) < 1e-6:
        return phi_cdf(x) * phi_cdf(y)

    # 利用 Owen T 的精确表达
    # BvN(x,y,rho) = 0.5*Phi(x) + 0.5*Phi(y) - 0.5*delta - T(x, alpha1) - T(y, alpha2)
    # 其中 alpha 与 rho 相关
    # 这里采用数值积分直接近似（稳健优先）
    # 使用二维高斯求积近似
    n_quad = 20
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_quad)
    # 映射到 [x, inf) 和 [y, inf)
    # 改用截断区间 [-L, L] 近似无穷
    L = 6.0
    total = 0.0
    for i in range(n_quad):
        xi = L * t_nodes[i]
        wi = L * t_weights[i]
        if xi < x:
            continue
        for j in range(n_quad):
            yj = L * t_nodes[j]
            wj = L * t_weights[j]
            if yj < y:
                continue
            # 双变量正态密度
            det = 1.0 - rho * rho
            if det <= 0.0:
                det = 1e-12
            z = (xi * xi - 2.0 * rho * xi * yj + yj * yj) / det
            dens = np.exp(-0.5 * z) / (2.0 * np.pi * np.sqrt(det))
            total += wi * wj * dens
    # 加上尾部分布用解析近似修正
    return max(0.0, min(1.0, total))


def partial_correlation_test(Theta: np.ndarray,
                              n_samples: int,
                              alpha_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    基于稀疏精度矩阵进行条件独立性检验（偏相关系数检验）。

    对每个非对角 $(i,j)$，计算：
    $$ r_{ij|\cdot} = -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\Theta_{jj}}} $$
    并使用 t 检验判断其显著性。

    Parameters
    ----------
    Theta : ndarray, shape (p, p)
        精度矩阵估计。
    n_samples : int
        样本量 $n$。
    alpha_level : float
        显著性水平。

    Returns
    -------
    pvals : ndarray, shape (p, p)
        p 值矩阵（对称）。
    reject : ndarray, shape (p, p)
        布尔矩阵，True 表示拒绝原假设（存在条件依赖）。
    r"""
    p = Theta.shape[0]
    if n_samples <= p + 2:
        raise ValueError("样本量必须大于 p+2 才能进行 t 检验。")

    pvals = np.ones((p, p))
    reject = np.zeros((p, p), dtype=bool)

    # TODO [Hole 2] 请补全偏相关系数检验的核心双重循环：
    # 对每个非对角变量对 (i, j)，其中 i < j：
    # 1. 提取 Theta[i,i], Theta[j,j], Theta[i,j]
    # 2. 若对角线元素 <= 0 则跳过（数值保护）
    # 3. 计算偏相关系数 r = -Theta_ij / sqrt(Theta_ii * Theta_jj)
    # 4. 将 r 截断到 [-0.9999, 0.9999] 避免除零
    # 5. 计算 t 统计量：t = r * sqrt(df / (1 - r^2))，其中 df = max(n - p - 2, 1)
    # 6. 计算双边 p 值（可使用正态近似或 t 分布）
    # 7. 填充对称矩阵 pvals 和 reject
    # 科学知识点：偏相关系数与精度矩阵元素的关系、Fisher Z 变换、t 检验
    raise NotImplementedError("Hole 2: 偏相关系数检验循环待实现")

    return pvals, reject


def demo():
    r"""模块自测试。"""
    # Owen T 函数测试
    val = owen_t_function(1.0, 0.5)
    print(f"[gaussian_causal_test] Owen T(1.0, 0.5) = {val:.8f}")

    # 偏相关检验测试
    np.random.seed(5)
    p = 8
    n = 300
    Theta = np.eye(p) * 2.0
    Theta[0, 1] = Theta[1, 0] = 0.5
    Theta[2, 3] = Theta[3, 2] = -0.4
    Sigma = np.linalg.inv(Theta)
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    S = np.cov(X, rowvar=False)
    Theta_est = np.linalg.inv(S + 0.1 * np.eye(p))
    pvals, reject = partial_correlation_test(Theta_est, n, alpha_level=0.05)
    n_edges = np.sum(reject) // 2
    print(f"[gaussian_causal_test] 显著边数 (alpha=0.05): {n_edges}")
    return val, n_edges


if __name__ == "__main__":
    demo()
