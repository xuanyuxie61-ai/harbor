"""
special_math.py
===============
博士级科学计算特殊函数库

融合原始项目：
  - 1096_sncndn：Jacobi 椭圆函数 SN, CN, DN (Bulirsch 算术-几何平均法)
  - 219_cordic：CORDIC 迭代算法用于高效三角函数计算
  - 058_atkinson：三对角线性系统 LU 分解与求解

数学物理模型：
  1. Jacobi 椭圆函数定义：
       u = ∫_0^φ dt / √(1 - m sin² t)
       sn(u|m) = sin φ,  cn(u|m) = cos φ,  dn(u|m) = √(1 - m sin² φ)
  2. CORDIC 旋转矩阵：
       R_j = [ 1,      -σ_j 2^{-j} ;
               σ_j 2^{-j},   1      ]
       其中 σ_j = sign(θ_j)
  3. 三对角系统 M x = f 的 LU 分解：
       α_j = a_j / β_{j-1},   β_j = b_j - α_j c_{j-1}
"""

import numpy as np


# ---------------------------------------------------------------------------
# Jacobi Elliptic Functions (from 1096_sncndn)
# ---------------------------------------------------------------------------
def jacobi_elliptic(u: float, m: float, maxit: int = 25):
    """
    计算 Jacobi 椭圆函数 sn(u|m), cn(u|m), dn(u|m)。

    采用 Bulirsch 算术-几何平均 (AGM) 迭代法：
        令 m_comp = 1 - m，对 m_comp 重复开平方直至收敛：
            a_{k+1} = (a_k + √(a_k b_k)) / 2
        收敛后利用反三角函数关系恢复 sn, cn, dn。

    参数
    ----
    u : float
        自变量 u (实数)
    m : float
        参数 m (0 ≤ m ≤ 1 为常规情形，m > 1 时自动进行模变换)
    maxit : int
        最大迭代次数

    返回
    ----
    sn, cn, dn : float
        Jacobi 椭圆函数值
    """
    if not np.isfinite(u) or not np.isfinite(m):
        raise ValueError("jacobi_elliptic: u 和 m 必须为有限实数")

    m_comp = 1.0 - m
    u_copy = float(u)

    # 边界情况 m = 1
    if abs(m_comp) < 1e-15:
        if abs(u_copy) < 1e-15:
            return 0.0, 1.0, 1.0
        return np.tanh(u_copy), 1.0 / np.cosh(u_copy), 1.0 / np.cosh(u_copy)

    # 模变换处理 m > 1
    if m > 1.0:
        d = 1.0 - m_comp
        if abs(d) < 1e-15:
            raise ValueError("jacobi_elliptic: 模变换分母为零")
        m_comp = -m_comp / d
        d = np.sqrt(d)
        u_copy = d * u_copy
    else:
        d = 1.0

    ca = np.sqrt(np.finfo(float).eps)
    a = 1.0
    dn = 1.0
    l = maxit

    m_array = np.zeros(maxit)
    n_array = np.zeros(maxit)

    for i in range(maxit):
        m_array[i] = a
        m_comp = np.sqrt(m_comp)
        n_array[i] = m_comp
        c = 0.5 * (a + m_comp)
        if abs(a - m_comp) <= ca * a:
            l = i
            break
        m_comp = a * m_comp
        a = c

    u_copy = c * u_copy
    sn = np.sin(u_copy)
    cn = np.cos(u_copy)

    if abs(sn) > 1e-15:
        a_val = cn / sn
        c_val = a_val * c
        for i in range(l, -1, -1):
            b = m_array[i]
            a_val = c_val * a_val
            c_val = dn * c_val
            dn = (n_array[i] + a_val) / (b + a_val)
            a_val = c_val / b
        a_val = 1.0 / np.sqrt(c_val * c_val + 1.0)
        if sn < 0:
            sn = -a_val
        else:
            sn = a_val
        cn = c_val * sn

    if m > 1.0:
        a_val = dn
        dn = cn
        cn = a_val
        sn = sn / d

    return sn, cn, dn


# ---------------------------------------------------------------------------
# CORDIC Sine/Cosine (from 219_cordic)
# ---------------------------------------------------------------------------
CORDIC_ANGLES = np.array([
    7.8539816339744830962E-01, 4.6364760900080611621E-01, 2.4497866312686415417E-01,
    1.2435499454676143503E-01, 6.2418809995957348474E-02, 3.1239833430268276254E-02,
    1.5623728620476830803E-02, 7.8123410601011112965E-03, 3.9062301319669718276E-03,
    1.9531225164788186851E-03, 9.7656218955931943040E-04, 4.8828121119489827547E-04,
    2.4414062014936176402E-04, 1.2207031189367020424E-04, 6.1035156174208775022E-05,
    3.0517578115526096862E-05, 1.5258789061315762107E-05, 7.6293945311019702634E-06,
    3.8146972656064962829E-06, 1.9073486328101870354E-06, 9.5367431640596087942E-07,
    4.7683715820308885993E-07, 2.3841857910155798249E-07, 1.1920928955078068531E-07,
    5.9604644775390554414E-08, 2.9802322387695303677E-08, 1.4901161193847655147E-08,
    7.4505805969238279871E-09, 3.7252902984619140453E-09, 1.8626451492309570291E-09,
    9.3132257461547851536E-10, 4.6566128730773925778E-10, 2.3283064365386962890E-10,
    1.1641532182693481445E-10, 5.8207660913467407226E-11, 2.9103830456733703613E-11,
    1.4551915228366851807E-11, 7.2759576141834259033E-12, 3.6379788070917129517E-12,
    1.8189894035458564758E-12, 9.0949470177292823792E-13, 4.5474735088646411896E-13,
    2.2737367544323205948E-13, 1.1368683772161602974E-13, 5.6843418860808014870E-14,
    2.8421709430404007435E-14, 1.4210854715202003717E-14, 7.1054273576010018587E-15,
    3.5527136788005009294E-15, 1.7763568394002504647E-15, 8.8817841970012523234E-16,
    4.4408920985006261617E-16, 2.2204460492503130808E-16, 1.1102230246251565404E-16,
    5.5511151231257827021E-17, 2.7755575615628913511E-17, 1.3877787807814456755E-17,
    6.9388939039072283776E-18, 3.4694469519536141888E-18, 1.7347234759768070944E-18
], dtype=float)

CORDIC_KPROD = np.array([
    0.70710678118654752440, 0.63245553203367586640, 0.61357199107789634961,
    0.60883391251775242102, 0.60764825625616820093, 0.60735177014129595905,
    0.60727764409352599905, 0.60725911229889273006, 0.60725447933256232972,
    0.60725332108987516334, 0.60725303152913433540, 0.60725295913894481363,
    0.60725294104139716351, 0.60725293651701023413, 0.60725293538591350073,
    0.60725293510313931731, 0.60725293503244577146, 0.60725293501477238499,
    0.60725293501035403837, 0.60725293500924945172, 0.60725293500897330506,
    0.60725293500890426839, 0.60725293500888700922, 0.60725293500888269443,
    0.60725293500888161574, 0.60725293500888134606, 0.60725293500888127864,
    0.60725293500888126179, 0.60725293500888125757, 0.60725293500888125652,
    0.60725293500888125626, 0.60725293500888125619, 0.60725293500888125617
], dtype=float)


def cordic_sin_cos(beta: float, n: int = 30):
    """
    CORDIC 算法计算 cos(β), sin(β)。

    算法核心：将角度 β 分解为一系列预定小角度 θ_j 的加权和：
        β = Σ σ_j θ_j,   σ_j ∈ {+1, -1}
    通过迭代旋转向量 [1, 0]^T，每次旋转角度 σ_j θ_j：
        x_{j+1} = x_j - σ_j y_j 2^{-j}
        y_{j+1} = y_j + σ_j x_j 2^{-j}
    最终乘以收敛常数 K_n = Π_j cos(θ_j) 进行模长校正。

    参数
    ----
    beta : float
        输入角度（弧度）
    n : int
        CORDIC 迭代次数（≥10 可获得较好精度）

    返回
    ----
    cos_val, sin_val : float
    """
    if not np.isfinite(beta):
        raise ValueError("cordic_sin_cos: beta 必须为有限实数")
    if n < 1:
        raise ValueError("cordic_sin_cos: 迭代次数 n 必须 ≥ 1")

    # 将角度归约到 [-π, π]
    theta = beta % (2.0 * np.pi)
    if theta > np.pi:
        theta -= 2.0 * np.pi
    elif theta < -np.pi:
        theta += 2.0 * np.pi

    # 进一步归约到 [-π/2, π/2]
    sign_factor = 1.0
    if theta < -0.5 * np.pi:
        theta += np.pi
        sign_factor = -1.0
    elif theta > 0.5 * np.pi:
        theta -= np.pi
        sign_factor = -1.0

    x = 1.0
    y = 0.0
    poweroftwo = 1.0
    angle = CORDIC_ANGLES[0] if len(CORDIC_ANGLES) > 0 else np.pi / 4.0

    for j in range(n):
        sigma = -1.0 if theta < 0.0 else 1.0
        factor = sigma * poweroftwo
        x_new = x - factor * y
        y_new = y + factor * x
        x, y = x_new, y_new
        theta -= sigma * angle
        poweroftwo *= 0.5
        if j + 1 < len(CORDIC_ANGLES):
            angle = CORDIC_ANGLES[j + 1]
        else:
            angle *= 0.5

    if n > 0:
        k = CORDIC_KPROD[min(n - 1, len(CORDIC_KPROD) - 1)]
        x *= k
        y *= k

    return sign_factor * x, sign_factor * y


# ---------------------------------------------------------------------------
# Tridiagonal Solver (from 058_atkinson)
# ---------------------------------------------------------------------------
def tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, f: np.ndarray):
    """
    求解三对角线性系统 M x = f，其中：
        M(i, i-1) = a[i]   (i = 1..n-1, 0-indexed)
        M(i, i)   = b[i]   (i = 0..n-1)
        M(i, i+1) = c[i]   (i = 0..n-2)

    采用 LU 分解：
        α_i = a_i / β_{i-1}
        β_i = b_i - α_i c_{i-1}
    前代：  f_i ← f_i - α_i f_{i-1}
    回代：  x_n = f_n / β_n,  x_i = (f_i - c_i x_{i+1}) / β_i

    参数
    ----
    a, b, c, f : np.ndarray
        一维数组，长度均为 n

    返回
    ----
    x : np.ndarray
        解向量

    TODO (Hole 2): 实现三对角矩阵的 LU 分解与前代/回代求解。
    注意输入数组 a, b, c, f 的内存覆盖语义与边界条件处理。
    """
    # === HOLE 2 BEGIN ===
    raise NotImplementedError("Hole 2: tridiag_solve 尚未实现")
    # === HOLE 2 END ===


def tridiag_solve_multi(a, b, c, F):
    """
    对多右端项求解三对角系统（用于时间步进中批量求解空间切片）。
    """
    n = F.shape[0]
    nrhs = F.shape[1] if F.ndim > 1 else 1
    X = np.zeros_like(F)
    for k in range(nrhs):
        X[:, k] = tridiag_solve(a.copy(), b.copy(), c.copy(), F[:, k].copy())
    return X
