"""
recruitment_models.py
鱼类种群补充量（Stock-Recruitment）模型模块
整合 Sigmoid 高阶导数、Beverton-Holt 模型、Ricker 模型及
Allee 效应修正，用于渔业资源动态预测

核心科学公式：
1. Sigmoid 函数 s(x) = 1 / (1 + exp(-x))
   n 阶导数展开：s^{(n)}(x) = \sum_{j=1}^{n+1} c_j s(x)^j
2. Beverton-Holt 补充模型：
   R(S) = \alpha S / (1 + \beta S)
3. Ricker 补充模型：
   R(S) = \alpha S \exp(-\beta S)
4. Sigmoid-Allee 修正模型（本模块创新）：
   R_{SA}(S) = R(S) \cdot \sigma(S - S_{crit})
   其中 \sigma 为 sigmoid 阈值函数，S_{crit} 为 Allee 效应临界亲体量
"""

import numpy as np
from utils import NumericalConfig, safe_divide


def sigmoid(x):
    """
    标准 Sigmoid 函数
    s(x) = 1 / (1 + exp(-x))
    当 x 很大或很小时进行数值截断以避免溢出
    """
    x = np.asarray(x, dtype=float)
    # 截断防止溢出
    x_clip = np.clip(x, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def sigmoid_derivative_coef(n):
    """
    计算 Sigmoid 函数 n 阶导数的幂级数展开系数

    理论基础：
    s^{(n)}(x) = \sum_{j=1}^{n+1} coef[j] \cdot s(x)^j

    系数由组合公式给出：
    coef_k = \sum_{j=0}^{k} (-1)^{k-j} C(k,j) (j+1)^n

    Parameters
    ----------
    n : int
        导数阶数，n >= 0

    Returns
    -------
    coef : ndarray, shape (n+2,)
        展开系数，coef[0] 恒为 0，coef[1] 对应 s(x)^1 的系数
    """
    if n < 0:
        raise ValueError("Derivative order n must be non-negative")

    coef = np.zeros(n + 2, dtype=float)

    for k in range(n + 1):
        cnk = 0.0
        mop = -1.0
        for j in range(k + 1):
            mop = -mop
            cnk += mop * ((j + 1) ** n) * comb(k, j)
        coef[k + 1] = cnk

    return coef


def comb(n, k):
    """
    组合数 C(n,k) = n! / (k!(n-k)!)
    """
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - k + i) / i
    return result


def sigmoid_derivative(n, x):
    """
    计算 Sigmoid 函数在 x 处的 n 阶导数值

    利用展开式 s^{(n)}(x) = \sum_{j=1}^{n+1} c_j s(x)^j
    该公式避免了直接高阶微分带来的数值不稳定性

    Parameters
    ----------
    n : int
        导数阶数
    x : float or array_like
        自变量

    Returns
    -------
    d : float or ndarray
        n 阶导数值
    """
    coef = sigmoid_derivative_coef(n)
    s = sigmoid(x)

    d = np.zeros_like(np.asarray(x), dtype=float)
    for j in range(1, n + 2):
        d += coef[j] * (s ** j)
    return d


def beverton_holt(S, alpha, beta):
    """
    经典 Beverton-Holt 亲体-补充量模型

    公式：
        R(S) = \frac{\alpha S}{1 + \beta S}

    生物学解释：
    - S: 亲体量（Spawning Stock Biomass）
    - \alpha: 单位亲体补充率（密度无关阶段）
    - \beta: 密度依赖系数（栖息地承载力限制）
    - 渐近补充量 R_{\infty} = \alpha / \beta

    Parameters
    ----------
    S : float or ndarray
        亲体量，要求 S >= 0
    alpha : float
        密度无关补充系数，alpha > 0
    beta : float
        密度依赖系数，beta > 0

    Returns
    -------
    R : float or ndarray
        补充量
    """
    S = np.asarray(S, dtype=float)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")

    # 确保非负
    S = np.maximum(S, 0.0)
    denom = 1.0 + beta * S
    return safe_divide(alpha * S, denom, 0.0)


def ricker_recruitment(S, alpha, beta):
    """
    经典 Ricker 亲体-补充量模型

    公式：
        R(S) = \alpha S \exp(-\beta S)

    生物学特征：
    - 存在最优亲体量 S_{opt} = 1/\beta，此时 R_{max} = \alpha / (e \beta)
    - 过高亲体量导致补充量下降（过度竞争效应）

    Parameters
    ----------
    S : float or ndarray
        亲体量
    alpha : float
        初始斜率参数
    beta : float
        密度依赖衰减系数

    Returns
    -------
    R : float or ndarray
        补充量
    """
    S = np.asarray(S, dtype=float)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")

    S = np.maximum(S, 0.0)
    return alpha * S * np.exp(-beta * S)


def sigmoid_allee_recruitment(S, alpha, beta, S_crit, steepness=10.0):
    """
    Sigmoid-Allee 修正补充模型（博士级创新模型）

    在经典 Beverton-Holt 基础上引入 Allee 效应阈值：
        R_{SA}(S) = \frac{\alpha S}{1 + \beta S} \cdot \sigma(S - S_{crit})

    其中 \sigma(z) = 1 / (1 + \exp(-steepness \cdot z)) 为平滑阈值函数

    生物学意义：
    - S_{crit}: Allee 效应临界亲体量
    - 当 S < S_{crit} 时，种群面临灭绝风险（补充量急剧下降）
    - steepness 控制阈值过渡的陡峭程度

    参数约束：
    - S_crit >= 0
    - steepness > 0（建议 5~20）

    Parameters
    ----------
    S : float or ndarray
        亲体量
    alpha, beta : float
        Beverton-Holt 参数
    S_crit : float
        Allee 效应临界值
    steepness : float, optional
        Sigmoid 陡峭系数，默认 10.0

    Returns
    -------
    R : float or ndarray
        Allee 修正后的补充量
    """
    S = np.asarray(S, dtype=float)
    if S_crit < 0:
        raise ValueError("S_crit must be non-negative")
    if steepness <= 0:
        raise ValueError("steepness must be positive")

    R_bh = beverton_holt(S, alpha, beta)
    allee_factor = sigmoid(steepness * (S - S_crit))
    return R_bh * allee_factor


def recruitment_derivative(S, alpha, beta, S_crit, steepness, model_type='allee'):
    """
    计算补充量函数对亲体量的导数 dR/dS
    用于种群稳定性分析和最优控制

    对于 Sigmoid-Allee 模型：
        dR/dS = R_{BH}'(S) \cdot \sigma(z) + R_{BH}(S) \cdot \sigma'(z) \cdot steepness
    其中 z = steepness * (S - S_crit)

    Parameters
    ----------
    S : float
        亲体量
    alpha, beta, S_crit, steepness : float
        模型参数
    model_type : str
        'bh', 'ricker', 或 'allee'

    Returns
    -------
    dR_dS : float
        导数值
    """
    S = float(S)
    if S < 0:
        S = 0.0

    if model_type == 'bh':
        # d/dS [alpha*S/(1+beta*S)] = alpha / (1+beta*S)^2
        return alpha / ((1.0 + beta * S) ** 2)

    elif model_type == 'ricker':
        # d/dS [alpha*S*exp(-beta*S)] = alpha*exp(-beta*S)*(1 - beta*S)
        return alpha * np.exp(-beta * S) * (1.0 - beta * S)

    elif model_type == 'allee':
        R_bh = beverton_holt(S, alpha, beta)
        z = steepness * (S - S_crit)
        sigma_z = sigmoid(z)
        # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        dsigma_dz = sigma_z * (1.0 - sigma_z)
        dRbh_dS = alpha / ((1.0 + beta * S) ** 2)
        return dRbh_dS * sigma_z + R_bh * dsigma_dz * steepness

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
