"""
stochastic_sampler.py
准随机序列与随机采样模块

融入原项目:
- 803_niederreiter2: Niederreiter基2低差异序列
- 713_maple_area: Hammersley准蒙特卡洛面积估计

功能:
1. Niederreiter低差异序列生成用于参数空间均匀采样
2. Hammersley序列用于心脏组织区域上的准随机采样
3. 低差异序列的统计性质验证
"""

import numpy as np
from math import sqrt, log


# ============================================================================
# Niederreiter基2序列（源自 803_niederreiter2）
# ============================================================================

# 预计算的不可约多项式（用于Niederreiter序列生成）
# C(J,R) 系数表（打包存储）
_NIEDERREITER_CJ = None
_NIEDERREITER_DIM = None
_NIEDERREITER_NEXTQ = None
_NIEDERREITER_KEY = None


def _calcc2(dim):
    """
    计算Niederreiter序列的C(J,R)系数
    
    基于GF(2)上的不可约多项式:
    p_1(x) = x + 1
    p_2(x) = x^2 + x + 1
    p_3(x) = x^3 + x + 1
    ...
    
    对于每个维度 j，C(J,R) 由多项式 p_j(x) 的系数决定
    """
    nbits = 31
    maxdim = 20
    
    if dim <= 0 or dim > maxdim:
        raise ValueError(f"Dimension must be between 1 and {maxdim}")
    
    # 不可约多项式度数
    degrees = [1, 2, 3, 3, 4, 4, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 7, 7, 7, 7]
    
    # 不可约多项式系数（二进制表示）
    poly = [3, 7, 11, 13, 25, 37, 59, 47, 61, 55,
            87, 91, 103, 115, 117, 173, 199, 185, 227, 217]
    
    cj = np.zeros((dim, nbits), dtype=np.int64)
    
    for i in range(dim):
        deg = degrees[i]
        p = poly[i]
        
        # 初始化
        nextq = np.zeros(nbits + 1, dtype=np.int64)
        for j in range(deg):
            nextq[j] = 0
        nextq[deg] = 1
        
        for j in range(deg + 1, nbits + 1):
            # 递推: q_j = sum_{k=1}^{deg} a_k * q_{j-k}
            # 其中 a_k 是多项式系数
            nextq[j] = 0
            for k in range(1, deg + 1):
                if (p >> k) & 1:
                    nextq[j] ^= nextq[j - k]
        
        # 计算 C(J,R)
        for r in range(nbits):
            cj[i, r] = nextq[r + 1]
    
    return cj


def niederreiter2_init(dim):
    """
    初始化Niederreiter基2序列生成器
    
    数学原理:
    Niederreiter序列在s维单位立方体 [0,1]^s 上生成低差异点列
    差异度 D_N = O((log N)^s / N)
    优于纯随机序列的 O(1/sqrt(N))
    """
    global _NIEDERREITER_CJ, _NIEDERREITER_DIM, _NIEDERREITER_NEXTQ, _NIEDERREITER_KEY
    
    _NIEDERREITER_DIM = dim
    _NIEDERREITER_KEY = -1
    _NIEDERREITER_CJ = _calcc2(dim)
    _NIEDERREITER_NEXTQ = np.zeros(dim, dtype=np.int64)


def niederreiter2_generate(key):
    """
    生成Niederreiter基2序列的第key个元素
    
    使用Gray码技术加速生成:
    g(k) = k XOR (k >> 1)
    
    参数:
        key: 序列索引 (≥ 0)
    返回:
        quasi: dim维准随机向量
        key_new: 下一个索引
    """
    global _NIEDERREITER_CJ, _NIEDERREITER_DIM, _NIEDERREITER_NEXTQ, _NIEDERREITER_KEY
    
    if _NIEDERREITER_DIM is None:
        raise RuntimeError("Niederreiter generator not initialized")
    
    dim = _NIEDERREITER_DIM
    nbits = 31
    recip = 2.0 ** (-nbits)
    
    # 如果key不连续，需要重新初始化NEXTQ
    if key != _NIEDERREITER_KEY + 1:
        gray = key ^ (key >> 1)
        _NIEDERREITER_NEXTQ = np.zeros(dim, dtype=np.int64)
        r = 0
        while gray > 0:
            if gray & 1:
                for i in range(dim):
                    _NIEDERREITER_NEXTQ[i] ^= _NIEDERREITER_CJ[i, r]
            gray >>= 1
            r += 1
    
    # 生成准随机数
    quasi = _NIEDERREITER_NEXTQ.astype(float) * recip
    
    # 找到key中最低位的0位置
    r = 0
    i = key
    while i & 1:
        r += 1
        i >>= 1
    
    if r >= nbits:
        raise RuntimeError("Too many calls to Niederreiter generator")
    
    # 更新NEXTQ
    for i in range(dim):
        _NIEDERREITER_NEXTQ[i] ^= _NIEDERREITER_CJ[i, r]
    
    _NIEDERREITER_KEY = key
    return quasi, key + 1


def generate_niederreiter_sequence(dim, n):
    """
    生成n个Niederreiter点
    """
    niederreiter2_init(dim)
    points = np.zeros((n, dim))
    key = 0
    for i in range(n):
        points[i], key = niederreiter2_generate(key)
    return points


# ============================================================================
# Hammersley序列（源自 713_maple_area）
# ============================================================================

def _radical_inverse(n, base):
    """
    根逆函数（van der Corput序列）
    
    φ_b(n) = sum_{k=0}^{∞} a_k(n) * b^{-k-1}
    其中 n = sum_{k=0}^{∞} a_k(n) * b^k
    """
    result = 0.0
    f = 1.0 / base
    while n > 0:
        result += f * (n % base)
        n //= base
        f /= base
    return result


def hammersley_sequence(i1, i2, m, n_base):
    """
    生成Hammersley序列的第i1到i2个元素
    
    Hammersley点:
    x_i = (i/N, φ_{p1}(i), φ_{p2}(i), ..., φ_{pm-1}(i))
    
    其中 p_j 是第j个素数，φ是根逆函数
    
    差异度: D_N = O((log N)^{s-1} / N)
    
    参数:
        i1, i2: 起始和结束索引
        m: 空间维度
        n_base: 第一个分量的基数
    返回:
        points: (m, |i2-i1|+1) 的Hammersley点
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    if i1 <= i2:
        i3 = 1
    else:
        i3 = -1
    
    if n_base <= 0:
        n_base = 1
    
    l = abs(i2 - i1) + 1
    points = np.zeros((m, l))
    k = 0
    
    for i in range(i1, i2 + i3, i3):
        points[0, k] = (i % (n_base + 1)) / n_base if n_base > 0 else 0.0
        for j in range(1, m):
            if j - 1 < len(primes):
                points[j, k] = _radical_inverse(i, primes[j - 1])
        k += 1
    
    return points


def estimate_area_qmc(polygon_vertices, bbox, n_samples=5000):
    """
    使用准蒙特卡洛方法估计多边形面积
    
    方法:
    1. 在包围盒 [xmin,xmax]×[ymin,ymax] 内生成N个Hammersley点
    2. 统计落在多边形内的点数 N_in
    3. 面积估计 A ≈ (N_in/N) * A_bbox
    
    误差: O((log N)/N)，优于普通MC的 O(1/sqrt(N))
    
    参数:
        polygon_vertices: (N,2) 多边形顶点
        bbox: (xmin, xmax, ymin, ymax) 包围盒
        n_samples: 采样点数
    返回:
        area_estimate: 面积估计
        area_bbox: 包围盒面积
    """
    vertices = np.asarray(polygon_vertices)
    xmin, xmax, ymin, ymax = bbox
    
    if xmax <= xmin or ymax <= ymin or n_samples <= 0:
        return 0.0, 0.0
    
    # 生成Hammersley点
    hammersley = hammersley_sequence(0, n_samples - 1, 2, n_samples - 1)
    
    # 映射到包围盒
    x = xmin + (xmax - xmin) * hammersley[0, :]
    y = ymin + (ymax - ymin) * hammersley[1, :]
    
    # 点在多边形内测试（ray casting算法）
    inside_count = 0
    n_vert = len(vertices)
    
    for k in range(n_samples):
        qx, qy = x[k], y[k]
        inside = False
        j = n_vert - 1
        for i in range(n_vert):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi < qy <= yj) or (qy <= yi and yj < qy)):
                if qx <= (xj - xi) * (qy - yi) / (yj - yi) + xi:
                    inside = not inside
            j = i
        if inside:
            inside_count += 1
    
    area_bbox = (xmax - xmin) * (ymax - ymin)
    area_estimate = (inside_count / n_samples) * area_bbox
    
    return area_estimate, area_bbox


# ============================================================================
# 心脏电生理参数采样
# ============================================================================

def sample_conductivity_parameters(n_samples, method='niederreiter'):
    """
    采样心肌电导率参数
    
    心肌电导率张量:
    σ = σ_f * e_f ⊗ e_f + σ_t * e_t ⊗ e_t + σ_n * e_n ⊗ e_n
    
    其中:
    - σ_f: 纤维方向电导率 (0.2 - 0.6 S/m)
    - σ_t: 横纤维方向电导率 (0.02 - 0.08 S/m)
    - σ_n: 法向电导率 (0.01 - 0.04 S/m)
    
    参数:
        n_samples: 采样数
        method: 'niederreiter' 或 'hammersley'
    返回:
        samples: (n_samples, 3) 电导率样本
    """
    if method == 'niederreiter':
        points = generate_niederreiter_sequence(3, n_samples)
    else:
        points = hammersley_sequence(0, n_samples - 1, 3, n_samples - 1)
        points = points.T
    
    # 映射到物理范围
    sigma_f_min, sigma_f_max = 0.2, 0.6
    sigma_t_min, sigma_t_max = 0.02, 0.08
    sigma_n_min, sigma_n_max = 0.01, 0.04
    
    samples = np.zeros((n_samples, 3))
    samples[:, 0] = sigma_f_min + (sigma_f_max - sigma_f_min) * points[:, 0]
    samples[:, 1] = sigma_t_min + (sigma_t_max - sigma_t_min) * points[:, 1]
    samples[:, 2] = sigma_n_min + (sigma_n_max - sigma_n_min) * points[:, 2]
    
    return samples


def compute_discrepancy(points):
    """
    计算点集的星差异度（star discrepancy）
    
    D_N* = sup_{J} |A(J;P)/N - vol(J)|
    
    用于评估准随机序列的质量
    """
    n, dim = points.shape
    if n == 0:
        return 1.0
    
    # 简化的差异度估计: L2差异度
    d2 = 0.0
    for i in range(n):
        for j in range(n):
            prod = 1.0
            for k in range(dim):
                prod *= (1.0 - max(points[i, k], points[j, k]))
            d2 += prod
    
    d2 = d2 / (n ** 2)
    
    # 理论项
    term2 = 0.0
    for i in range(n):
        prod = 1.0
        for k in range(dim):
            prod *= (1.0 - points[i, k] ** 2) / 2.0
        term2 += prod
    term2 = term2 * (2.0 ** (1 - dim)) / n
    
    term3 = 3.0 ** (-dim)
    
    l2_disc = sqrt(d2 - term2 + term3)
    return l2_disc
