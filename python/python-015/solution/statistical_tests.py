"""
statistical_tests.py
拓扑半金属中的统计检验与概率分析

凝聚态物理应用：
在实验数据分析中，需要统计检验来确认Weyl节点的存在性：
1. Weyl节点配对检验（超几何分布）
2. 能带间隙的统计显著性检验（正态分布）
3. Berry曲率极值的统计分布

核心统计模型：
超几何分布：
    P(X=k) = C(K,k) * C(N-K, n-k) / C(N,n)

在Weyl节点配对检验中：
- N：总k点数
- K：候选Weyl节点数
- n：抽样检验数
- k：确认配对的节点数

若k显著大于期望值，则支持Weyl节点存在。

正态近似（大样本极限）：
    mean = n*K/N
    var = n*K/N * (1-K/N) * (N-n)/(N-1)

基于种子项目044_asa152（chyper函数和alnorm函数）。
"""

import numpy as np
from typing import Tuple
from scipy.special import gammaln


def alnorm(x: float, upper: bool = False) -> float:
    """
    标准正态分布的累积概率
    
    基于种子项目044_asa152中的alnorm函数。
    
    使用Hill算法（Algorithm AS 66）：
    对于小z（|z| <= 1.28）：有理函数近似
    对于大z（|z| > 1.28）：渐近展开
    
    Parameters
    ----------
    x : float
    upper : bool
        True: P(Z > x)
        False: P(Z < x)
    
    Returns
    -------
    p : float
    """
    # Hill算法常数
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p_const = 0.39894228044
    q = 0.39990348504
    r = 0.398942280385
    utzero = 18.66
    
    up = upper
    z = x
    
    if z < 0.0:
        up = not up
        z = -z
    
    if ltone < z and (not up or utzero < z):
        return 0.0 if up else 1.0
    
    y = 0.5 * z * z
    
    if z <= con:
        value = 0.5 - z * (p_const - q * y
                           / (y + a1 + b1
                              / (y + a2 + b2
                                 / (y + a3))))
    else:
        value = r * np.exp(-y) \
                / (z + c1 + d1
                   / (z + c2 + d2
                      / (z + c3 + d3
                         / (z + c4 + d4
                            / (z + c5 + d5
                               / (z + c6))))))
    
    if not up:
        value = 1.0 - value
    
    return value


def chyper(point: bool, kk: int, ll: int, mm: int, nn: int) -> Tuple[float, int]:
    """
    超几何分布的点概率或累积概率
    
    基于种子项目044_asa152中的chyper函数。
    
    超几何分布：
        P(X = ll) = C(nn, ll) * C(mm-nn, kk-ll) / C(mm, kk)
    
    Parameters
    ----------
    point : bool
        True: 计算点概率P(X=ll)
        False: 计算累积概率P(X<=ll)
    kk : int
        样本大小
    ll : int
        样本中成功数
    mm : int
        总体大小
    nn : int
        总体中成功数
    
    Returns
    -------
    value : float
    ifault : int
        错误码（0表示无错误）
    """
    elimit = -88.0
    mbig = 600
    mvbig = 1000
    rootpi = 2.506628274631001
    scale = 1.0e35
    
    ifault = 0
    value = 0.0
    
    # 参数检查
    if nn < 0 or mm < nn or kk < 0 or mm < kk:
        ifault = 1
        return value, ifault
    
    if ll < 0 or mm - nn < kk - ll:
        ifault = 2
        return value, ifault
    
    if not point:
        value = 1.0
    
    if nn < ll or kk < ll:
        ifault = 2
        return value, ifault
    
    ifault = 0
    value = 1.0
    
    if kk == 0 or kk == mm or nn == 0 or nn == mm:
        return value, ifault
    
    if not point and ll == min(kk, nn):
        return value, ifault
    
    p_ratio = nn / max(mm - nn, 1)
    
    # 正态近似条件
    if (16.0 * max(p_ratio, 1.0 / max(p_ratio, 1e-15)) < min(kk, mm - kk)
            and mvbig < mm and elimit > -100.0):
        mean = kk * nn / mm
        sig = np.sqrt(mean * ((mm - nn) / mm) * ((mm - kk) / (mm - 1)))
        
        if point:
            arg = -0.5 * (((ll - mean) / max(sig, 1e-15)) ** 2)
            if elimit <= arg:
                value = np.exp(arg) / (sig * rootpi)
            else:
                value = 0.0
        else:
            value = alnorm((ll + 0.5 - mean) / max(sig, 1e-15), False)
        
        return value, ifault
    
    # 精确计算
    if min(nn - 1, mm - nn) < min(kk - 1, mm - kk):
        kk, nn = nn, kk
    
    dir_flag = True
    if mm - kk < kk - 1:
        dir_flag = False
        ll = nn - ll
        kk = mm - kk
    
    if mbig < mm:
        # 使用对数阶乘
        p = (gammaln(nn + 1) - gammaln(mm + 1) + gammaln(mm - kk + 1)
             + gammaln(kk + 1) + gammaln(mm - nn + 1) - gammaln(ll + 1)
             - gammaln(nn - ll + 1) - gammaln(kk - ll + 1)
             - gammaln(mm - nn - kk + ll + 1))
        
        if elimit <= p:
            value = np.exp(p)
        else:
            value = 0.0
    else:
        # 直接乘积
        for i in range(1, ll + 1):
            value = value * (kk - i + 1) * (nn - i + 1) / (ll - i + 1) / (mm - i + 1)
        
        if ll != kk:
            j = mm - nn + ll
            for i in range(ll + 1, kk + 1):
                value = value * (j - i + 1) / (mm - i + 1)
    
    if point:
        return value, ifault
    
    # 累积概率
    if value == 0.0:
        if mm <= mbig:
            p = (gammaln(nn + 1) - gammaln(mm + 1) + gammaln(kk + 1)
                 + gammaln(mm - nn + 1) - gammaln(ll + 1)
                 - gammaln(nn - ll + 1) - gammaln(kk - ll + 1)
                 - gammaln(mm - nn - kk + ll + 1) + gammaln(mm - kk + 1))
        
        p = p + np.log(scale)
        if p < elimit:
            ifault = 3
            if (nn * kk + nn + kk + 1) / (mm + 2) < ll:
                value = 1.0
            return value, ifault
        else:
            p = np.exp(p)
    else:
        p = value * scale
    
    pt = 0.0
    nl = nn - ll
    kl = kk - ll
    mnkl = mm - nn - kl + 1
    
    if ll <= kl:
        for i in range(1, ll + 1):
            p = p * (ll - i + 1) * (mnkl - i + 1) / (nl + i) / (kl + i)
            pt = pt + p
    else:
        dir_flag = False
        for j in range(kl):
            p = p * (nl - j) * (kl - j) / (ll + j + 1) / (mnkl + j)
            pt = pt + p
    
    if p == 0.0:
        ifault = 3
    
    if dir_flag:
        value = value + (pt / scale)
    else:
        value = 1.0 - (pt / scale)
    
    return value, ifault


def weyl_node_pairing_test(n_total_kpoints: int, n_candidate_nodes: int,
                            n_tested: int, n_confirmed_pairs: int) -> Tuple[float, float]:
    """
    Weyl节点配对显著性检验
    
    使用超几何分布检验Weyl节点是否显著配对。
    
    H0（零假设）：Weyl节点随机分布，不配对的概率为p0
    H1（备择假设）：Weyl节点显著配对
    
    Parameters
    ----------
    n_total_kpoints : int
        总k点数
    n_candidate_nodes : int
        候选Weyl节点数
    n_tested : int
        检验的节点数
    n_confirmed_pairs : int
        确认配对的节点数
    
    Returns
    -------
    p_value : float
        超几何检验的p值
    expected : float
        期望配对数
    """
    # 总体：N = n_total_kpoints
    # 成功项：K = n_candidate_nodes（假设其中一半可配对）
    K = n_candidate_nodes // 2
    
    # 超几何分布参数
    N = n_total_kpoints
    n = n_tested
    k = n_confirmed_pairs
    
    # 计算P(X >= k)
    p_value = 0.0
    for i in range(k, min(n, K) + 1):
        p, _ = chyper(True, n, i, N, K)
        p_value += p
    
    expected = n * K / N
    
    return p_value, expected


def berry_curvature_significance(omega_values: np.ndarray,
                                  threshold: float = 0.5) -> Tuple[float, float]:
    """
    Berry曲率极值的统计显著性检验
    
    使用正态近似检验Berry曲率是否显著偏离零。
    
    Parameters
    ----------
    omega_values : np.ndarray
        Berry曲率采样值
    threshold : float
        显著性阈值
    
    Returns
    -------
    z_score : float
    p_value : float
    """
    mean = np.mean(omega_values)
    std = np.std(omega_values)
    
    if std < 1e-15:
        return 0.0, 1.0
    
    n = len(omega_values)
    z_score = mean / (std / np.sqrt(n))
    
    # 双尾检验
    p_value = 2.0 * alnorm(abs(z_score), True)
    
    return z_score, p_value
